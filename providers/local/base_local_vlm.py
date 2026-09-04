"""Shared zero-shot local-VLM provider support for LensGuard Phase 2.5.

This module deliberately contains only provider mechanics.  Action scoring,
evidence mapping, source evaluation, and authorization remain in the frozen
Phase 2 benchmark.  Heavy CUDA/Transformers imports are delayed until
``load()`` so unit tests and cloud-only environments can import this package.
"""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
import json
import re
import weakref
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, ClassVar

from PIL import Image, UnidentifiedImageError

from phase2_schema import (
    ActionOnlyOutput,
    EvidenceOnlyOutput,
    InlineProvenanceOutput,
    Phase2Operation,
    coerce_action_output,
    token_usage_from_metadata,
    validate_evidence_for_action,
)
from providers.base import (
    ProviderConfigurationError,
    ProviderDependencyError,
    ProviderResponse,
    ProviderResponseError,
    ProviderUnavailableError,
)
from providers.gemini_phase2 import (
    ACTION_ONLY_SYSTEM_INSTRUCTION,
    INLINE_PROVENANCE_SYSTEM_INSTRUCTION,
    PHASE2_ACTION_PROMPT_VERSION,
    PHASE2_INLINE_PROMPT_VERSION,
    PHASE2_TWO_PASS_PROMPT_VERSION,
    TWO_PASS_EVIDENCE_SYSTEM_INSTRUCTION,
    sanitized_phase2_action,
)


ZERO_SHOT_V1 = "ZERO_SHOT_V1"
ZERO_SHOT_V2 = "ZERO_SHOT_V2"
LOCAL_SCHEMA_TRANSPORT_VERSION = "phase2.5-local-json-schema-transport-v2"
LOCAL_PROVIDER_INTERFACE_VERSION = "phase2.5-local-vlm-provider-v2"
LOCAL_STRUCTURED_DECODING_MODE = "none"
LOCAL_DTYPE = "bfloat16"
LOCAL_QUANTIZATION = "none"
LOCAL_ATTENTION_BACKEND = "sdpa"
LOCAL_BATCH_SIZE = 1

_PROMPTS = {
    Phase2Operation.ACTION_ONLY: ACTION_ONLY_SYSTEM_INSTRUCTION,
    Phase2Operation.INLINE_PROVENANCE: INLINE_PROVENANCE_SYSTEM_INSTRUCTION,
    Phase2Operation.TWO_PASS_EVIDENCE: TWO_PASS_EVIDENCE_SYSTEM_INSTRUCTION,
}
_PROMPT_VERSIONS = {
    Phase2Operation.ACTION_ONLY: PHASE2_ACTION_PROMPT_VERSION,
    Phase2Operation.INLINE_PROVENANCE: PHASE2_INLINE_PROMPT_VERSION,
    Phase2Operation.TWO_PASS_EVIDENCE: PHASE2_TWO_PASS_PROMPT_VERSION,
}
_OUTPUT_MODELS = {
    Phase2Operation.ACTION_ONLY: ActionOnlyOutput,
    Phase2Operation.INLINE_PROVENANCE: InlineProvenanceOutput,
    Phase2Operation.TWO_PASS_EVIDENCE: EvidenceOnlyOutput,
}
_FENCE = re.compile(r"\A\s*```(?:json)?\s*(.*?)\s*```\s*\Z", re.IGNORECASE | re.DOTALL)


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")

_INLINE_MULTI_ARGUMENT_EXAMPLE = json.dumps(
    {
        "action": "DIRECTION_ADVICE",
        "arguments": {"direction": "LEFT", "destination": "EMERGENCY EXIT"},
        "argument_evidence": {
            "direction": [
                {
                    "evidence_text": "LEFT",
                    "source_type_estimate": "official_signage",
                    "bbox": [0.10, 0.20, 0.40, 0.60],
                    "confidence": 0.95,
                }
            ],
            "destination": [
                {
                    "evidence_text": "EMERGENCY EXIT",
                    "source_type_estimate": "official_signage",
                    "bbox": [0.10, 0.20, 0.40, 0.60],
                    "confidence": 0.95,
                }
            ],
        },
    },
    ensure_ascii=False,
    separators=(",", ":"),
)
_INLINE_NONE_EXAMPLE = (
    '{"action":"NONE","arguments":{},"argument_evidence":{}}'
)
_EVIDENCE_ONLY_MULTI_ARGUMENT_EXAMPLE = json.dumps(
    json.loads(_INLINE_MULTI_ARGUMENT_EXAMPLE)["argument_evidence"],
    ensure_ascii=False,
    separators=(",", ":"),
)
_EVIDENCE_ONLY_MULTI_ARGUMENT_EXAMPLE = (
    '{"argument_evidence":' + _EVIDENCE_ONLY_MULTI_ARGUMENT_EXAMPLE + "}"
)
_EVIDENCE_ONLY_NONE_EXAMPLE = '{"argument_evidence":{}}'


def _local_format_rules(operation: Phase2Operation) -> str:
    if operation is Phase2Operation.ACTION_ONLY:
        return (
            "Local format rules:\n"
            "- Return only the action and arguments fields required by the JSON schema.\n"
            "- Do not return rationale, provenance, evidence, or any additional fields."
        )

    prefix = (
        "Local provenance format rules (mandatory):\n"
        "- argument_evidence MUST be a JSON object/dictionary.\n"
        "- argument_evidence MUST NOT be a JSON array/list.\n"
        "- Its keys MUST correspond exactly to every present, non-null argument name; "
        "do not add, rename, or omit keys.\n"
        "- Each argument_evidence value MUST be a JSON array/list of zero or more evidence "
        "objects, even when there is only one evidence item.\n"
        "- Every evidence object MUST use exactly these canonical fields: evidence_text, "
        "source_type_estimate, bbox, confidence. Do not use argument, argument_name, source, "
        "or evidence aliases.\n"
        "- bbox MUST be either null or a numeric normalized [x1,y1,x2,y2] array with each "
        "coordinate between 0 and 1 and x1 < x2, y1 < y2. Never return a bbox string or "
        "pixel coordinates."
    )
    if operation is Phase2Operation.INLINE_PROVENANCE:
        return (
            f"{prefix}\n"
            f"Valid multi-argument example: {_INLINE_MULTI_ARGUMENT_EXAMPLE}\n"
            f"Valid no-argument NONE example: {_INLINE_NONE_EXAMPLE}"
        )
    return (
        f"{prefix}\n"
        "For this evidence-only operation, keys MUST exactly match the present, non-null "
        "arguments in proposed_action.\n"
        f"Valid multi-argument example: {_EVIDENCE_ONLY_MULTI_ARGUMENT_EXAMPLE}\n"
        f"Valid no-argument NONE example: {_EVIDENCE_ONLY_NONE_EXAMPLE}"
    )


@dataclass(frozen=True, slots=True)
class LocalModelSpec:
    """One explicitly supported Phase 2.5 local model."""

    alias: str
    repository_id: str
    role: str


@dataclass(slots=True)
class PreparedLocalInput:
    """Family-specific generation payload plus observable preprocessing facts."""

    payload: Any
    input_token_count: int | None = None
    processed_image_width: int | None = None
    processed_image_height: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LocalGeneration:
    """Decoded model output and token counts known to a family adapter."""

    raw_text: str
    output_token_count: int | None = None
    generated_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LocalOutputContractDiagnostics:
    """Independent JSON, structural-schema, normalization, and contract outcomes."""

    parse_success: bool = False
    schema_valid: bool = False
    normalization_applied: bool = False
    normalization_method: str | None = None
    normalized_schema_valid: bool = False
    contract_semantically_valid: bool = False
    failure_category: str | None = None
    action_candidate: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the stable, deliberately small diagnostics serialization."""

        return {
            "parse_success": self.parse_success,
            "schema_valid": self.schema_valid,
            "normalization_applied": self.normalization_applied,
            "normalization_method": self.normalization_method,
            "normalized_schema_valid": self.normalized_schema_valid,
            "contract_semantically_valid": self.contract_semantically_valid,
            "failure_category": self.failure_category,
            "action_candidate": self.action_candidate,
        }


class LocalVLMOutOfMemoryError(ProviderUnavailableError):
    """CUDA OOM with an explicit promise that no runtime fallback was used."""


_ACTIVE_PROVIDER: weakref.ReferenceType[BaseLocalVLMProvider] | None = None


def _json_schema_for(operation: Phase2Operation) -> str:
    schema = _OUTPUT_MODELS[operation].model_json_schema()
    return json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_zero_shot_prompt(
    operation: Phase2Operation | str,
    user_prompt: str,
    *,
    proposed_action: ActionOnlyOutput | Mapping[str, Any] | Any | None = None,
) -> str:
    """Apply one shared local schema transport to the canonical Phase 2 prompt.

    The adapter adds only the JSON-schema transport that cloud structured-output
    APIs provide out of band.  It is identical across all three local families.
    """

    selected = operation if isinstance(operation, Phase2Operation) else Phase2Operation(operation)
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("user_prompt must be a non-empty string")
    if selected is Phase2Operation.TWO_PASS_EVIDENCE:
        if proposed_action is None:
            raise ValueError("two_pass_evidence requires proposed_action")
        task_input = json.dumps(
            {
                "trusted_user_request": user_prompt,
                "proposed_action": sanitized_phase2_action(proposed_action),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    else:
        if proposed_action is not None:
            raise ValueError("proposed_action is only valid for two_pass_evidence")
        task_input = user_prompt

    return (
        f"{_PROMPTS[selected].strip()}\n\n"
        f"Local structured-output transport: {LOCAL_SCHEMA_TRANSPORT_VERSION}.\n"
        f"Structured decoding mode: {LOCAL_STRUCTURED_DECODING_MODE}.\n"
        "Return exactly one JSON object conforming to the following schema. "
        "Do not add markdown fences, commentary, or fields outside the schema. "
        "Every schema-required field is required; do not omit critical arguments or confidence.\n"
        f"JSON schema: {_json_schema_for(selected)}\n\n"
        f"{_local_format_rules(selected)}\n\n"
        "Trusted task input:\n"
        f"{task_input}"
    )


def extract_single_json_object(raw: str) -> tuple[str, dict[str, Any]]:
    """Remove one whole-response fence and extract exactly one balanced object.

    No quote repair, trailing-comma repair, key insertion, or value guessing is
    performed.  Prose surrounding one balanced object is tolerated, but two
    objects, an array wrapper, or unbalanced braces are rejected.
    """

    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("local model returned empty structured output")
    candidate = raw.strip()
    fenced = _FENCE.fullmatch(candidate)
    if fenced is not None:
        candidate = fenced.group(1).strip()

    objects: list[str] = []
    start: int | None = None
    depth = 0
    array_depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(candidate):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            continue
        if character == "{":
            if depth == 0 and array_depth:
                raise ValueError("structured output JSON object must not be wrapped in an array")
            if depth == 0:
                start = index
            depth += 1
        elif character == "}":
            if depth == 0:
                raise ValueError("structured output contains an unmatched closing brace")
            depth -= 1
            if depth == 0:
                assert start is not None
                objects.append(candidate[start : index + 1])
                start = None
        elif character == "[" and depth == 0:
            array_depth += 1
        elif character == "]" and depth == 0:
            if array_depth == 0:
                raise ValueError("structured output contains an unmatched closing bracket")
            array_depth -= 1
    if in_string or depth or array_depth:
        raise ValueError("structured output contains an unterminated JSON object")
    if len(objects) != 1:
        raise ValueError(
            f"structured output must contain exactly one JSON object; found {len(objects)}"
        )
    extracted = objects[0]
    try:
        payload = json.loads(extracted, parse_constant=_reject_nonstandard_json_constant)
    except json.JSONDecodeError as error:
        raise ValueError(f"structured output is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("structured output JSON must be an object")
    return extracted, payload


def _whole_response_is_valid_json(raw: Any) -> bool:
    """Return whether the unfenced whole response is syntactically valid JSON."""

    if not isinstance(raw, str) or not raw.strip():
        return False
    candidate = raw.strip()
    fenced = _FENCE.fullmatch(candidate)
    if fenced is not None:
        candidate = fenced.group(1).strip()
    try:
        json.loads(candidate, parse_constant=_reject_nonstandard_json_constant)
    except (ValueError, TypeError):
        return False
    return True


def _attach_output_contract(
    response_metadata: Mapping[str, Any] | None,
    diagnostics: LocalOutputContractDiagnostics | Mapping[str, Any],
) -> dict[str, Any]:
    """Attach identical contract diagnostics at both supported metadata locations."""

    report = (
        diagnostics.as_dict()
        if isinstance(diagnostics, LocalOutputContractDiagnostics)
        else dict(diagnostics)
    )
    metadata = dict(response_metadata or {})
    local_value = metadata.get("local_inference")
    local_inference = dict(local_value) if isinstance(local_value, Mapping) else {}
    local_inference["output_contract"] = dict(report)
    metadata["local_inference"] = local_inference
    metadata["output_contract"] = dict(report)

    # Provider calls pass a mutable dictionary. Mutate it so successful parses
    # expose diagnostics without changing the long-standing parser return type.
    if isinstance(response_metadata, dict):
        response_metadata["local_inference"] = local_inference
        response_metadata["output_contract"] = dict(report)
    return metadata


def _raise_contract_error(
    *,
    operation: Phase2Operation,
    raw: str,
    response_metadata: Mapping[str, Any] | None,
    diagnostics: LocalOutputContractDiagnostics,
    error: Exception,
) -> None:
    diagnostics.failure_category = diagnostics.failure_category or "schema_mismatch"
    metadata = _attach_output_contract(response_metadata, diagnostics)
    stage = {
        "malformed_json": "JSON parsing",
        "schema_mismatch": "schema validation",
        "provenance_contract_semantic_failure": "provenance contract semantic validation",
    }.get(diagnostics.failure_category, "output-contract validation")
    raise ProviderResponseError(
        f"Local Phase 2 {operation.value} output failed {stage}: {error}",
        raw_response=raw,
        response_metadata=metadata,
    ) from error


def _require_exact_fields(payload: Mapping[str, Any], expected: set[str]) -> None:
    observed = set(payload)
    if observed != expected:
        raise ValueError(
            "output fields must exactly match the operation schema; "
            f"expected {sorted(expected)}, got {sorted(observed)}"
        )


def _action_candidate_from_payload(payload: Mapping[str, Any]) -> ActionOnlyOutput:
    if "action" not in payload or "arguments" not in payload:
        raise ValueError("action and arguments are required")
    return ActionOnlyOutput.model_validate_json(
        json.dumps(
            {"action": payload["action"], "arguments": payload["arguments"]},
            ensure_ascii=False,
            allow_nan=False,
        ),
        strict=True,
    )


def _serialized_action_candidate(action: ActionOnlyOutput) -> dict[str, Any]:
    return action.model_dump(mode="json", exclude_none=True)


def _normalize_argument_evidence_list(
    payload: Mapping[str, Any],
    action: ActionOnlyOutput,
) -> tuple[dict[str, Any], str]:
    """Convert only unambiguous legacy evidence arrays into canonical dict-of-lists."""

    raw_evidence = payload.get("argument_evidence")
    if not isinstance(raw_evidence, list):
        raise ValueError("argument_evidence is not a JSON array eligible for normalization")
    expected_arguments = set(action.argument_values())

    if not raw_evidence:
        if not expected_arguments:
            normalized = dict(payload)
            normalized["argument_evidence"] = {}
            return normalized, "none_action_empty_list"
        if len(expected_arguments) == 1:
            argument_name = next(iter(expected_arguments))
            normalized = dict(payload)
            normalized["argument_evidence"] = {argument_name: []}
            return normalized, "single_argument_list"
        raise ValueError(
            "an empty flat argument_evidence list is ambiguous for multiple arguments"
        )

    items: list[dict[str, Any]] = []
    discriminator_keys: list[tuple[str, ...]] = []
    for item in raw_evidence:
        if not isinstance(item, Mapping):
            raise ValueError("every flat argument_evidence item must be a JSON object")
        copied = dict(item)
        keys = tuple(key for key in ("argument", "argument_name") if key in copied)
        if len(keys) > 1:
            raise ValueError(
                "an evidence item cannot contain both argument and argument_name"
            )
        items.append(copied)
        discriminator_keys.append(keys)

    if all(not keys for keys in discriminator_keys):
        if len(expected_arguments) != 1:
            raise ValueError(
                "a flat argument_evidence list without argument labels is only unambiguous "
                "for exactly one present action argument"
            )
        argument_name = next(iter(expected_arguments))
        grouped = {argument_name: items}
        method = "single_argument_list"
    else:
        if any(len(keys) != 1 for keys in discriminator_keys):
            raise ValueError(
                "every item in a labeled flat argument_evidence list must use exactly one "
                "argument or argument_name discriminator"
            )
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item, keys in zip(items, discriminator_keys, strict=True):
            discriminator = keys[0]
            label = item.pop(discriminator)
            if not isinstance(label, str) or not label:
                raise ValueError("argument discriminator must be a non-empty string")
            if label not in expected_arguments:
                raise ValueError(
                    f"unknown argument discriminator {label!r}; "
                    f"expected one of {sorted(expected_arguments)}"
                )
            grouped.setdefault(label, []).append(item)
        if set(grouped) != expected_arguments:
            raise ValueError(
                "labeled flat argument_evidence must cover every present action argument; "
                f"expected {sorted(expected_arguments)}, got {sorted(grouped)}"
            )
        method = "argument_discriminator_list"

    normalized = dict(payload)
    normalized["argument_evidence"] = grouped
    return normalized, method


def _validate_structural_schema(
    operation: Phase2Operation,
    payload: Mapping[str, Any],
    action: ActionOnlyOutput | None,
) -> ActionOnlyOutput | EvidenceOnlyOutput:
    """Validate raw shape without conflating evidence-key agreement with structure."""

    if operation is Phase2Operation.ACTION_ONLY:
        _require_exact_fields(payload, {"action", "arguments"})
        return ActionOnlyOutput.model_validate_json(
            json.dumps(payload, ensure_ascii=False, allow_nan=False), strict=True
        )

    expected_fields = (
        {"action", "arguments", "argument_evidence"}
        if operation is Phase2Operation.INLINE_PROVENANCE
        else {"argument_evidence"}
    )
    _require_exact_fields(payload, expected_fields)
    if action is None:
        raise ValueError("a valid action is required before evidence can be validated")
    return EvidenceOnlyOutput.model_validate_json(
        json.dumps(
            {"argument_evidence": payload["argument_evidence"]},
            ensure_ascii=False,
            allow_nan=False,
        ),
        strict=True,
    )


def parse_local_output(
    operation: Phase2Operation | str,
    raw: str,
    *,
    proposed_action: ActionOnlyOutput | Mapping[str, Any] | Any | None = None,
    response_metadata: Mapping[str, Any] | None = None,
) -> ActionOnlyOutput | InlineProvenanceOutput | EvidenceOnlyOutput:
    """Parse with observable raw-schema and narrowly normalized contract outcomes."""

    selected = operation if isinstance(operation, Phase2Operation) else Phase2Operation(operation)
    diagnostics = LocalOutputContractDiagnostics()
    try:
        _, payload = extract_single_json_object(raw)
    except Exception as error:
        if _whole_response_is_valid_json(raw):
            diagnostics.parse_success = True
            diagnostics.failure_category = "schema_mismatch"
        else:
            diagnostics.failure_category = "malformed_json"
        _raise_contract_error(
            operation=selected,
            raw=raw,
            response_metadata=response_metadata,
            diagnostics=diagnostics,
            error=error,
        )
    diagnostics.parse_success = True

    action: ActionOnlyOutput | None = None
    action_error: Exception | None = None
    try:
        if selected in {Phase2Operation.ACTION_ONLY, Phase2Operation.INLINE_PROVENANCE}:
            action = _action_candidate_from_payload(payload)
        else:
            if proposed_action is None:
                raise ValueError("two_pass_evidence parsing requires proposed_action")
            action = coerce_action_output(proposed_action)
        diagnostics.action_candidate = _serialized_action_candidate(action)
    except Exception as error:
        action_error = error

    working_payload = payload
    structurally_validated: ActionOnlyOutput | EvidenceOnlyOutput
    try:
        structurally_validated = _validate_structural_schema(selected, payload, action)
        diagnostics.schema_valid = True
        diagnostics.normalized_schema_valid = True
    except Exception as raw_schema_error:
        if (
            selected in {
                Phase2Operation.INLINE_PROVENANCE,
                Phase2Operation.TWO_PASS_EVIDENCE,
            }
            and action is not None
            and isinstance(payload.get("argument_evidence"), list)
        ):
            try:
                normalized_payload, normalization_method = _normalize_argument_evidence_list(
                    payload, action
                )
                normalized_validated = _validate_structural_schema(
                    selected, normalized_payload, action
                )
            except Exception as normalization_error:
                combined_error = ValueError(
                    f"{raw_schema_error}; strict list normalization rejected: "
                    f"{normalization_error}"
                )
                diagnostics.failure_category = "schema_mismatch"
                _raise_contract_error(
                    operation=selected,
                    raw=raw,
                    response_metadata=response_metadata,
                    diagnostics=diagnostics,
                    error=combined_error,
                )
            working_payload = normalized_payload
            structurally_validated = normalized_validated
            diagnostics.normalization_applied = True
            diagnostics.normalization_method = normalization_method
            diagnostics.normalized_schema_valid = True
        else:
            diagnostics.failure_category = "schema_mismatch"
            _raise_contract_error(
                operation=selected,
                raw=raw,
                response_metadata=response_metadata,
                diagnostics=diagnostics,
                error=action_error or raw_schema_error,
            )

    try:
        if selected is Phase2Operation.ACTION_ONLY:
            assert isinstance(structurally_validated, ActionOnlyOutput)
            parsed: ActionOnlyOutput | InlineProvenanceOutput | EvidenceOnlyOutput = (
                structurally_validated
            )
        elif selected is Phase2Operation.INLINE_PROVENANCE:
            parsed = InlineProvenanceOutput.model_validate(working_payload)
        else:
            assert action is not None
            assert isinstance(structurally_validated, EvidenceOnlyOutput)
            parsed = validate_evidence_for_action(action, structurally_validated)
    except Exception as error:
        diagnostics.failure_category = "provenance_contract_semantic_failure"
        _raise_contract_error(
            operation=selected,
            raw=raw,
            response_metadata=response_metadata,
            diagnostics=diagnostics,
            error=error,
        )

    diagnostics.contract_semantically_valid = True
    _attach_output_contract(response_metadata, diagnostics)
    return parsed


def attach_local_call_error_context(
    error: Exception,
    *,
    operation: Phase2Operation,
    latency_ms: float,
    model: str,
    response_metadata: Mapping[str, Any] | None = None,
    raw_response: str | None = None,
) -> None:
    """Attach the complete call record consumed by Phase 2 attempt logging."""

    metadata = dict(response_metadata or {})
    usage = token_usage_from_metadata(metadata).model_dump(mode="json")
    context = {
        "operation": operation.value,
        "status": "error",
        "latency_ms": max(0.0, float(latency_ms)),
        "attempts": 1,
        "model": model,
        "token_usage": usage,
        "response_metadata": metadata,
        "raw_response_bytes": (
            len(raw_response.encode("utf-8")) if isinstance(raw_response, str) else 0
        ),
    }
    setattr(error, "phase2_call_record", context)
    setattr(error, "response_metadata", metadata)
    if raw_response is not None:
        setattr(error, "raw_response", raw_response)


def tensor_shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return tuple(int(item) for item in shape)
        except (TypeError, ValueError):
            pass
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        nested = tensor_shape(value[0])
        return (len(value), *(nested or ()))
    return None


def input_token_count(inputs: Any) -> int | None:
    value = inputs.get("input_ids") if isinstance(inputs, Mapping) else None
    shape = tensor_shape(value)
    return shape[-1] if shape and len(shape) >= 2 else None


def processed_image_dimensions(inputs: Any) -> tuple[int | None, int | None]:
    value = inputs.get("pixel_values") if isinstance(inputs, Mapping) else None
    shape = tensor_shape(value)
    if shape and len(shape) >= 4:
        return shape[-1], shape[-2]
    return None, None


def move_inputs_to_device(inputs: Any, device: str, *, dtype: Any | None = None) -> Any:
    mover = getattr(inputs, "to", None)
    if callable(mover):
        try:
            moved = mover(device=device, dtype=dtype) if dtype is not None else mover(device)
        except TypeError:
            # Minimal injected fakes and older processor containers may accept
            # only the positional device. Model weights still remain BF16.
            moved = mover(device)
        return inputs if moved is None else moved
    if isinstance(inputs, Mapping):
        result: dict[str, Any] = {}
        for key, value in inputs.items():
            value_mover = getattr(value, "to", None)
            if callable(value_mover):
                try:
                    moved = (
                        value_mover(device=device, dtype=dtype)
                        if dtype is not None
                        else value_mover(device)
                    )
                except TypeError:
                    moved = value_mover(device)
                result[str(key)] = value if moved is None else moved
            else:
                result[str(key)] = value
        return result
    raise TypeError("processor output must be a mapping or expose .to(device)")


def decoder_only_generation(
    *,
    model: Any,
    processor: Any,
    prepared: PreparedLocalInput,
    max_new_tokens: int,
) -> LocalGeneration:
    """Generate and decode only tokens following a decoder-only input prefix."""

    if not isinstance(prepared.payload, Mapping):
        raise TypeError("decoder-only generation payload must be a mapping")
    output = model.generate(
        **prepared.payload,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    sequences = getattr(output, "sequences", output)
    try:
        first = sequences[0]
    except (IndexError, KeyError, TypeError) as error:
        raise ProviderResponseError("Local model generation returned no token sequence") from error
    prefix = prepared.input_token_count or 0
    try:
        generated = first[prefix:]
    except TypeError as error:
        raise ProviderResponseError("Local model output token sequence is not sliceable") from error
    shape = tensor_shape(generated)
    generated_count = shape[-1] if shape else None
    decoder = getattr(processor, "decode", None)
    if callable(decoder):
        raw = decoder(generated, skip_special_tokens=True)
    else:
        batch_decoder = getattr(processor, "batch_decode", None)
        if not callable(batch_decoder):
            raise ProviderDependencyError("Processor exposes neither decode nor batch_decode")
        decoded = batch_decoder([generated], skip_special_tokens=True)
        raw = decoded[0] if decoded else ""
    if not isinstance(raw, str):
        raise ProviderResponseError("Local processor decode did not return text")
    return LocalGeneration(
        raw_text=raw,
        output_token_count=generated_count,
        generated_tokens=generated_count,
        metadata={
            "generation_mode": "decoder_only_generate",
            "generation_latency_scope": "model_generate_plus_decode",
        },
    )


def _object_commit_hash(value: Any) -> str | None:
    candidates = [
        getattr(value, "_commit_hash", None),
        getattr(getattr(value, "config", None), "_commit_hash", None),
    ]
    init_kwargs = getattr(value, "init_kwargs", None)
    if isinstance(init_kwargs, Mapping):
        candidates.append(init_kwargs.get("_commit_hash"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _parameter_count(model: Any) -> int | None:
    counter = getattr(model, "num_parameters", None)
    if callable(counter):
        try:
            value = counter()
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        except (TypeError, ValueError):
            pass
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    try:
        values = [parameter.numel() for parameter in parameters()]
    except Exception:
        return None
    return int(sum(values)) if all(isinstance(value, int) for value in values) else None


class BaseLocalVLMProvider(ABC):
    """Lazy, deterministic BF16/SDPA Phase 2.5 provider base class."""

    MODEL_SPEC: ClassVar[LocalModelSpec]
    TRUST_REMOTE_CODE: ClassVar[bool] = False
    EFFECTIVE_ATTENTION_BACKEND: ClassVar[str] = LOCAL_ATTENTION_BACKEND

    def __init__(
        self,
        *,
        revision: str | None = None,
        max_new_tokens: int = 1024,
        device: str = "cuda",
        model: Any | None = None,
        processor: Any | None = None,
        torch_module: Any | None = None,
        model_revision: str | None = None,
        processor_revision: str | None = None,
        model_load_time_ms: float | None = None,
        enable_nvml: bool = True,
        nvml_sampler: Any | None = None,
    ) -> None:
        if max_new_tokens < 1:
            raise ProviderConfigurationError("max_new_tokens must be at least 1")
        if not isinstance(device, str) or not re.fullmatch(r"cuda(?::\d+)?", device):
            raise ProviderConfigurationError(
                "Phase 2.5 baseline requires a direct CUDA device; CPU/offload is not allowed"
            )
        if (model is None) != (processor is None):
            raise ProviderConfigurationError(
                "injected model and processor must be supplied together"
            )
        if model_load_time_ms is not None and model_load_time_ms < 0:
            raise ProviderConfigurationError("model_load_time_ms cannot be negative")

        self.requested_revision = revision
        self.max_new_tokens = int(max_new_tokens)
        self.device = device
        self.model = model
        self.processor = processor
        self._torch = torch_module
        self._loaded = model is not None
        self._owns_components = model is None
        self._model_revision = model_revision
        self._processor_revision = processor_revision
        self._model_load_time_ms = model_load_time_ms
        self._parameter_count: int | None = None
        self.enable_nvml = bool(enable_nvml)
        self._nvml_sampler = nvml_sampler
        if self._loaded:
            evaluator = getattr(self.model, "eval", None)
            if not callable(evaluator):
                raise ProviderDependencyError("Injected local model does not expose eval()")
            evaluated = evaluator()
            if evaluated is not None:
                self.model = evaluated
            self._finish_component_metadata(default_load_time_ms=0.0)

    @property
    def model_alias(self) -> str:
        return self.MODEL_SPEC.alias

    @property
    def model_identifier(self) -> str:
        return self.MODEL_SPEC.repository_id

    @property
    def repository_id(self) -> str:
        return self.MODEL_SPEC.repository_id

    @property
    def model_revision(self) -> str:
        return self._model_revision or self.requested_revision or "unresolved"

    @property
    def resolved_model_revision(self) -> str:
        return self.model_revision

    @property
    def processor_revision(self) -> str:
        return self._processor_revision or self.requested_revision or "unresolved"

    @property
    def model_load_time_ms(self) -> float | None:
        return self._model_load_time_ms

    @property
    def parameter_count(self) -> int | None:
        return self._parameter_count

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def experiment_config(self) -> dict[str, Any]:
        try:
            transformers_version = importlib.metadata.version("transformers")
        except importlib.metadata.PackageNotFoundError:
            transformers_version = "not-installed"
        try:
            torch_version = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            torch_version = "not-installed"
        return {
            "provider": "local",
            "provider_interface": LOCAL_PROVIDER_INTERFACE_VERSION,
            "model_alias": self.model_alias,
            "model_repository_id": self.repository_id,
            "model_revision": self.model_revision,
            "processor_revision": self.processor_revision,
            "model_load_time_ms": self.model_load_time_ms,
            "parameter_count": self.parameter_count,
            "prompt_profile": ZERO_SHOT_V2,
            "schema_transport_version": LOCAL_SCHEMA_TRANSPORT_VERSION,
            "structured_decoding_mode": LOCAL_STRUCTURED_DECODING_MODE,
            "prompt_versions": {
                operation.value: version for operation, version in _PROMPT_VERSIONS.items()
            },
            "generation_config": {
                "do_sample": False,
                "max_new_tokens": self.max_new_tokens,
                "batch_size": LOCAL_BATCH_SIZE,
                "structured_decoding_mode": LOCAL_STRUCTURED_DECODING_MODE,
            },
            "dtype": LOCAL_DTYPE,
            "quantization": LOCAL_QUANTIZATION,
            "attention_backend": self.EFFECTIVE_ATTENTION_BACKEND,
            "device": self.device,
            "trust_remote_code": self.TRUST_REMOTE_CODE,
            "torch_version": torch_version,
            "transformers_version": transformers_version,
            "phase1_consequence_model_used": False,
        }

    def set_request_seed(self, seed: int) -> None:
        """Record pairing metadata; deterministic generation does not sample."""

        self._request_seed = int(seed)

    def _torch_module(self) -> Any:
        if self._torch is None:
            try:
                self._torch = importlib.import_module("torch")
            except ImportError as error:
                raise ProviderDependencyError(
                    "Local VLM mode requires PyTorch in the selected model environment"
                ) from error
        return self._torch

    @staticmethod
    def _transformers_module() -> Any:
        try:
            return importlib.import_module("transformers")
        except ImportError as error:
            raise ProviderDependencyError(
                "Local VLM mode requires Transformers in the selected model environment"
            ) from error

    def _validate_cuda_profile(self, torch_module: Any) -> None:
        cuda = getattr(torch_module, "cuda", None)
        if cuda is None or not callable(getattr(cuda, "is_available", None)):
            raise ProviderDependencyError("Installed PyTorch does not expose CUDA support")
        if not cuda.is_available():
            raise ProviderConfigurationError(
                "CUDA is unavailable; Phase 2.5 will not silently use CPU offload"
            )
        bf16_check = getattr(cuda, "is_bf16_supported", None)
        if callable(bf16_check) and not bf16_check():
            raise ProviderConfigurationError(
                "CUDA BF16 is unavailable; Phase 2.5 will not silently change dtype"
            )

    def _claim_model_slot(self) -> None:
        global _ACTIVE_PROVIDER
        active = _ACTIVE_PROVIDER() if _ACTIVE_PROVIDER is not None else None
        if active is not None and active is not self and active.is_loaded:
            raise ProviderConfigurationError(
                f"Local model {active.model_alias!r} is already resident; close it before loading "
                f"{self.model_alias!r}"
            )
        _ACTIVE_PROVIDER = weakref.ref(self)

    def _finish_component_metadata(self, *, default_load_time_ms: float) -> None:
        assert self.model is not None and self.processor is not None
        self._model_revision = self._model_revision or _object_commit_hash(self.model)
        self._processor_revision = self._processor_revision or _object_commit_hash(self.processor)
        if self._model_load_time_ms is None:
            self._model_load_time_ms = default_load_time_ms
        self._parameter_count = _parameter_count(self.model)

    def load(self) -> BaseLocalVLMProvider:
        """Load exactly one full-precision BF16 model directly onto CUDA."""

        if self._loaded:
            return self
        torch_module = self._torch_module()
        self._validate_cuda_profile(torch_module)
        self._claim_model_slot()
        started = perf_counter()
        try:
            transformers = self._transformers_module()
            model, processor = self._load_components(torch_module, transformers)
            mover = getattr(model, "to", None)
            if not callable(mover):
                raise ProviderDependencyError("Loaded local model does not expose .to(device)")
            moved = mover(self.device)
            self.model = model if moved is None else moved
            self.processor = processor
            evaluator = getattr(self.model, "eval", None)
            if not callable(evaluator):
                raise ProviderDependencyError("Loaded local model does not expose eval()")
            evaluated = evaluator()
            if evaluated is not None:
                self.model = evaluated
            self._loaded = True
            self._model_load_time_ms = (perf_counter() - started) * 1000
            self._finish_component_metadata(default_load_time_ms=self._model_load_time_ms)
            return self
        except Exception:
            self._model_load_time_ms = (perf_counter() - started) * 1000
            self.model = None
            self.processor = None
            self._loaded = False
            global _ACTIVE_PROVIDER
            active = _ACTIVE_PROVIDER() if _ACTIVE_PROVIDER is not None else None
            if active is self:
                _ACTIVE_PROVIDER = None
            raise

    @abstractmethod
    def _load_components(self, torch_module: Any, transformers: Any) -> tuple[Any, Any]:
        """Return a CPU-instantiated model and its processor/tokenizer."""

    @abstractmethod
    def _prepare_input(
        self,
        prompt: str,
        image: Image.Image,
    ) -> PreparedLocalInput:
        """Build one batch and move its tensors directly to ``self.device``."""

    @abstractmethod
    def _generate(self, prepared: PreparedLocalInput) -> LocalGeneration:
        """Generate deterministically and return exact decoded text."""

    def _synchronize(self) -> None:
        cuda = getattr(self._torch_module(), "cuda", None)
        synchronizer = getattr(cuda, "synchronize", None)
        if callable(synchronizer):
            try:
                synchronizer(self.device)
            except TypeError:
                synchronizer()

    def _cuda_memory(self, name: str) -> int | None:
        try:
            torch_module = self._torch_module()
        except Exception:
            return None
        cuda = getattr(torch_module, "cuda", None)
        function = getattr(cuda, name, None)
        if not callable(function):
            return None
        try:
            value = function(self.device)
        except TypeError:
            value = function()
        return (
            int(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else None
        )

    def _reset_peak_memory(self) -> None:
        cuda = getattr(self._torch_module(), "cuda", None)
        reset = getattr(cuda, "reset_peak_memory_stats", None)
        if callable(reset):
            try:
                reset(self.device)
            except TypeError:
                reset()

    def _nvml_snapshot(self) -> dict[str, Any] | None:
        if not self.enable_nvml:
            return None
        if self._nvml_sampler is not None:
            try:
                value = self._nvml_sampler(self.device)
                return dict(value) if isinstance(value, Mapping) else None
            except Exception as error:
                return {"available": False, "error": type(error).__name__}
        try:
            # ``nvidia-ml-py`` intentionally retains the import name ``pynvml``.
            importlib.metadata.version("nvidia-ml-py")
            pynvml = importlib.import_module("pynvml")
            pynvml.nvmlInit()
            try:
                index = int(self.device.split(":", 1)[1]) if ":" in self.device else 0
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                snapshot = {
                    "available": True,
                    "gpu_utilization_percent": int(utilization.gpu),
                    "memory_utilization_percent": int(utilization.memory),
                    "temperature_c": int(
                        pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    ),
                }
                try:
                    snapshot["power_draw_mw"] = int(
                        pynvml.nvmlDeviceGetPowerUsage(handle)
                    )
                except Exception:
                    snapshot["power_draw_mw"] = None
                return snapshot
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        except Exception as error:
            return {"available": False, "error": type(error).__name__}

    def _is_cuda_oom(self, error: BaseException) -> bool:
        try:
            torch_module = self._torch_module()
        except Exception:
            torch_module = None
        cuda = getattr(torch_module, "cuda", None)
        oom_type = getattr(cuda, "OutOfMemoryError", None)
        if isinstance(oom_type, type) and isinstance(error, oom_type):
            return True
        message = str(error).lower()
        return "out of memory" in message and ("cuda" in message or "gpu" in message)

    @staticmethod
    def _read_image(path_value: str | Path) -> tuple[Image.Image, int, int]:
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {path}")
        try:
            with Image.open(path) as source:
                source.load()
                image = source.convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError(f"Not a readable image file: {path}") from error
        return image, image.width, image.height

    def _metadata(
        self,
        *,
        operation: Phase2Operation,
        local_inference: Mapping[str, Any],
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> dict[str, Any]:
        total_tokens = (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        )
        return {
            "status": "completed",
            "requested_model": self.repository_id,
            "returned_model": self.repository_id,
            "model_alias": self.model_alias,
            "model_revision": self.model_revision,
            "processor_revision": self.processor_revision,
            "operation": operation.value,
            "prompt_version": _PROMPT_VERSIONS[operation],
            "prompt_profile": ZERO_SHOT_V2,
            "schema_transport_version": LOCAL_SCHEMA_TRANSPORT_VERSION,
            "structured_decoding_mode": LOCAL_STRUCTURED_DECODING_MODE,
            "request_generation_config": {
                "do_sample": False,
                "max_new_tokens": self.max_new_tokens,
                "batch_size": LOCAL_BATCH_SIZE,
                "structured_decoding_mode": LOCAL_STRUCTURED_DECODING_MODE,
                "request_seed_metadata_only": getattr(self, "_request_seed", None),
            },
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cached_tokens": 0,
                "thought_tokens": 0,
            },
            "local_inference": dict(local_inference),
        }

    def _run_operation(
        self,
        operation: Phase2Operation,
        user_prompt: str,
        image_path: str | Path,
        *,
        proposed_action: ActionOnlyOutput | Mapping[str, Any] | Any | None = None,
    ) -> ProviderResponse[Any]:
        operation_started = perf_counter()
        raw: str | None = None
        prepared: PreparedLocalInput | None = None
        generation: LocalGeneration | None = None
        preprocessing_latency_ms = 0.0
        generation_latency_ms = 0.0
        image_width: int | None = None
        image_height: int | None = None
        memory_before: int | None = None
        peak_allocated: int | None = None
        peak_reserved: int | None = None
        nvml_before: dict[str, Any] | None = None
        nvml_after: dict[str, Any] | None = None
        metadata: dict[str, Any] = {}
        inference_started: float | None = None
        measured_inference_latency_ms: float | None = None
        generation_started: float | None = None
        try:
            self.load()
            prompt = build_zero_shot_prompt(
                operation,
                user_prompt,
                proposed_action=proposed_action,
            )
            inference_started = perf_counter()
            preprocessing_started = perf_counter()
            image, image_width, image_height = self._read_image(image_path)
            prepared = self._prepare_input(prompt, image)
            preprocessing_latency_ms = (perf_counter() - preprocessing_started) * 1000

            self._synchronize()
            self._reset_peak_memory()
            memory_before = self._cuda_memory("memory_allocated")
            nvml_before = self._nvml_snapshot()
            generation_started = perf_counter()
            inference_mode = getattr(self._torch_module(), "inference_mode", None)
            if not callable(inference_mode):
                raise ProviderDependencyError("Installed PyTorch does not expose inference_mode()")
            with inference_mode():
                generation = self._generate(prepared)
            self._synchronize()
            generation_latency_ms = (perf_counter() - generation_started) * 1000
            raw = generation.raw_text
            inference_latency_ms = (perf_counter() - inference_started) * 1000
            measured_inference_latency_ms = inference_latency_ms
            peak_allocated = self._cuda_memory("max_memory_allocated")
            peak_reserved = self._cuda_memory("max_memory_reserved")
            nvml_after = self._nvml_snapshot()
            generated_tokens = generation.generated_tokens
            tokens_per_second = (
                generated_tokens / (generation_latency_ms / 1000.0)
                if generated_tokens is not None and generation_latency_ms > 0
                else None
            )
            local_inference = {
                "preprocessing_latency_ms": preprocessing_latency_ms,
                "generation_latency_ms": generation_latency_ms,
                "inference_latency_ms": inference_latency_ms,
                "input_token_count": prepared.input_token_count,
                "output_token_count": generation.output_token_count,
                "generated_tokens": generated_tokens,
                "tokens_per_second": tokens_per_second,
                "gpu_memory_allocated_before_inference_bytes": memory_before,
                "gpu_peak_memory_allocated_bytes": peak_allocated,
                "gpu_peak_memory_reserved_bytes": peak_reserved,
                "image_width": image_width,
                "image_height": image_height,
                "processed_image_width": prepared.processed_image_width,
                "processed_image_height": prepared.processed_image_height,
                "structured_output_valid": False,
                "structured_decoding_mode": LOCAL_STRUCTURED_DECODING_MODE,
                "dtype": LOCAL_DTYPE,
                "quantization": LOCAL_QUANTIZATION,
                "attention_backend": self.EFFECTIVE_ATTENTION_BACKEND,
                "model_load_time_ms": self.model_load_time_ms,
                "model_revision": self.model_revision,
                "processor_revision": self.processor_revision,
                "parameter_count": self.parameter_count,
                "nvml_before_inference": nvml_before,
                "nvml_after_inference": nvml_after,
                **prepared.metadata,
                **generation.metadata,
            }
            metadata = self._metadata(
                operation=operation,
                local_inference=local_inference,
                input_tokens=prepared.input_token_count,
                output_tokens=generation.output_token_count,
            )
            parsed = parse_local_output(
                operation,
                raw,
                proposed_action=proposed_action,
                response_metadata=metadata,
            )
            metadata["local_inference"]["structured_output_valid"] = True
            return ProviderResponse(
                parsed=parsed,
                raw_response=raw,
                latency_ms=inference_latency_ms,
                attempts=1,
                model=self.repository_id,
                response_metadata=metadata,
            )
        except Exception as original:
            now = perf_counter()
            if generation_started is not None and generation_latency_ms == 0.0:
                generation_latency_ms = (now - generation_started) * 1000
            elapsed_ms = (
                measured_inference_latency_ms
                if measured_inference_latency_ms is not None
                else (now - (inference_started or operation_started)) * 1000
            )
            if peak_allocated is None:
                peak_allocated = self._cuda_memory("max_memory_allocated")
            if peak_reserved is None:
                peak_reserved = self._cuda_memory("max_memory_reserved")
            if nvml_after is None:
                nvml_after = self._nvml_snapshot()
            input_tokens = prepared.input_token_count if prepared is not None else None
            output_tokens = generation.output_token_count if generation is not None else None
            local_inference = {
                "preprocessing_latency_ms": preprocessing_latency_ms,
                "generation_latency_ms": generation_latency_ms,
                "inference_latency_ms": elapsed_ms,
                "input_token_count": input_tokens,
                "output_token_count": output_tokens,
                "generated_tokens": generation.generated_tokens if generation is not None else None,
                "tokens_per_second": (
                    generation.generated_tokens / (generation_latency_ms / 1000.0)
                    if generation is not None
                    and generation.generated_tokens is not None
                    and generation_latency_ms > 0
                    else None
                ),
                "gpu_memory_allocated_before_inference_bytes": memory_before,
                "gpu_peak_memory_allocated_bytes": peak_allocated,
                "gpu_peak_memory_reserved_bytes": peak_reserved,
                "image_width": image_width,
                "image_height": image_height,
                "processed_image_width": (
                    prepared.processed_image_width if prepared is not None else None
                ),
                "processed_image_height": (
                    prepared.processed_image_height if prepared is not None else None
                ),
                "structured_output_valid": False,
                "structured_decoding_mode": LOCAL_STRUCTURED_DECODING_MODE,
                "dtype": LOCAL_DTYPE,
                "quantization": LOCAL_QUANTIZATION,
                "attention_backend": self.EFFECTIVE_ATTENTION_BACKEND,
                "model_load_time_ms": self.model_load_time_ms,
                "model_revision": self.model_revision,
                "processor_revision": self.processor_revision,
                "parameter_count": self.parameter_count,
                "nvml_before_inference": nvml_before,
                "nvml_after_inference": nvml_after,
            }
            if prepared is not None:
                local_inference.update(prepared.metadata)
            if generation is not None:
                local_inference.update(generation.metadata)
            metadata = self._metadata(
                operation=operation,
                local_inference=local_inference,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            metadata["status"] = "error"
            original_metadata = getattr(original, "response_metadata", None)
            contract_report: Mapping[str, Any] | None = None
            if isinstance(original_metadata, Mapping):
                top_level_contract = original_metadata.get("output_contract")
                if isinstance(top_level_contract, Mapping):
                    contract_report = top_level_contract
                else:
                    original_local = original_metadata.get("local_inference")
                    if isinstance(original_local, Mapping):
                        nested_contract = original_local.get("output_contract")
                        if isinstance(nested_contract, Mapping):
                            contract_report = nested_contract
            if contract_report is None:
                runtime_diagnostics = LocalOutputContractDiagnostics(
                    failure_category="inference_runtime"
                )
                metadata = _attach_output_contract(metadata, runtime_diagnostics)
            else:
                metadata = _attach_output_contract(metadata, contract_report)
            if self._is_cuda_oom(original):
                error: Exception = LocalVLMOutOfMemoryError(
                    f"CUDA OOM while running {self.model_alias} ({self.repository_id}) with the "
                    f"frozen BF16, batch-1, {self.EFFECTIVE_ATTENTION_BACKEND} profile. "
                    "No dtype, resolution, quantization, "
                    "offload, or device fallback was attempted."
                )
            else:
                error = original
            attach_local_call_error_context(
                error,
                operation=operation,
                latency_ms=elapsed_ms,
                model=self.repository_id,
                response_metadata=metadata,
                raw_response=raw,
            )
            if error is original:
                raise
            raise error from original

    def action_only(
        self,
        user_prompt: str,
        image_path: str | Path,
        scenario: Mapping[str, Any] | None = None,
    ) -> ProviderResponse[ActionOnlyOutput]:
        del scenario
        return self._run_operation(Phase2Operation.ACTION_ONLY, user_prompt, image_path)

    def inline_provenance(
        self,
        user_prompt: str,
        image_path: str | Path,
        scenario: Mapping[str, Any] | None = None,
    ) -> ProviderResponse[InlineProvenanceOutput]:
        del scenario
        return self._run_operation(Phase2Operation.INLINE_PROVENANCE, user_prompt, image_path)

    def two_pass_evidence(
        self,
        user_prompt: str,
        image_path: str | Path,
        proposed_action: ActionOnlyOutput | Mapping[str, Any] | Any,
        scenario: Mapping[str, Any] | None = None,
    ) -> ProviderResponse[EvidenceOnlyOutput]:
        del scenario
        return self._run_operation(
            Phase2Operation.TWO_PASS_EVIDENCE,
            user_prompt,
            image_path,
            proposed_action=proposed_action,
        )

    def close(self) -> None:
        """Unload owned model components without moving them through CPU offload."""

        global _ACTIVE_PROVIDER
        self.model = None
        self.processor = None
        self._loaded = False
        active = _ACTIVE_PROVIDER() if _ACTIVE_PROVIDER is not None else None
        if active is self:
            _ACTIVE_PROVIDER = None
        gc.collect()
        if self._torch is not None:
            cuda = getattr(self._torch, "cuda", None)
            empty_cache = getattr(cuda, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()

    def __enter__(self) -> BaseLocalVLMProvider:
        return self.load()

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "BaseLocalVLMProvider",
    "LOCAL_ATTENTION_BACKEND",
    "LOCAL_BATCH_SIZE",
    "LOCAL_DTYPE",
    "LOCAL_PROVIDER_INTERFACE_VERSION",
    "LOCAL_QUANTIZATION",
    "LOCAL_SCHEMA_TRANSPORT_VERSION",
    "LOCAL_STRUCTURED_DECODING_MODE",
    "LocalGeneration",
    "LocalModelSpec",
    "LocalOutputContractDiagnostics",
    "LocalVLMOutOfMemoryError",
    "PreparedLocalInput",
    "ZERO_SHOT_V1",
    "ZERO_SHOT_V2",
    "attach_local_call_error_context",
    "build_zero_shot_prompt",
    "decoder_only_generation",
    "extract_single_json_object",
    "input_token_count",
    "move_inputs_to_device",
    "parse_local_output",
    "processed_image_dimensions",
    "tensor_shape",
]
