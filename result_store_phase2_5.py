"""Retry-safe Phase 2.5 persistence layered on the frozen Phase 2 validator."""

from __future__ import annotations

import csv
import json
import math
import re
from copy import deepcopy
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from result_store import (
    append_jsonl,
    attempt_accounting,
    final_trials_from_attempts,
    read_jsonl,
    write_csv,
)
from result_store_phase2 import validate_phase2_attempts


PHASE2_5_IDENTITY_FIELDS = (
    "scene_id",
    "condition",
    "architecture_arm",
    "provider",
    "model_id",
    "model_revision",
    "run",
    "prompt_version",
    "dataset_version",
    "policy_version",
)

PHASE2_5_TELEMETRY_FIELDS = (
    "model_load_time_ms",
    "preprocessing_latency_ms",
    "inference_latency_ms",
    "generation_latency_ms",
    "thin_gate_latency_ms",
    "evidence_mapper_latency_ms",
    "input_token_count",
    "output_token_count",
    "generated_tokens",
    "tokens_per_second",
    "gpu_memory_allocated_before_inference_bytes",
    "gpu_peak_memory_allocated_bytes",
    "gpu_peak_memory_reserved_bytes",
    "model_dtype",
    "quantization",
    "attention_backend",
    "image_width",
    "image_height",
)

_LATENCY_FIELDS = frozenset(
    {
        "model_load_time_ms",
        "preprocessing_latency_ms",
        "inference_latency_ms",
        "generation_latency_ms",
        "thin_gate_latency_ms",
        "evidence_mapper_latency_ms",
    }
)
_COUNT_FIELDS = frozenset(
    {"input_token_count", "output_token_count", "generated_tokens"}
)
_MEMORY_FIELDS = frozenset(
    {
        "gpu_memory_allocated_before_inference_bytes",
        "gpu_peak_memory_allocated_bytes",
        "gpu_peak_memory_reserved_bytes",
    }
)
_DIMENSION_FIELDS = frozenset({"image_width", "image_height"})
_RUNTIME_LABEL_FIELDS = frozenset(
    {"model_dtype", "quantization", "attention_backend"}
)
_TELEMETRY_ALIASES = {
    "input_tokens": "input_token_count",
    "output_tokens": "output_token_count",
    "gpu_memory_allocated_before_inference": (
        "gpu_memory_allocated_before_inference_bytes"
    ),
    "gpu_peak_memory_allocated": "gpu_peak_memory_allocated_bytes",
    "gpu_peak_memory_reserved": "gpu_peak_memory_reserved_bytes",
}
_BYTE_UNIT_MULTIPLIERS = {
    "b": 1,
    "byte": 1,
    "bytes": 1,
    "kb": 1_000,
    "mb": 1_000_000,
    "gb": 1_000_000_000,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
}

CoreValidator = Callable[[Iterable[dict[str, Any]]], None]
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_V2_PROFILE = "ZERO_SHOT_V2"
_TRI_STATE_FIELDS = frozenset(
    {
        "action_correct",
        "critical_argument_correct",
        "unsafe_execution",
        "provenance_semantically_valid",
    }
)
_CONTRACT_BOOLEAN_FIELDS = frozenset(
    {
        "parse_success",
        "schema_valid",
        "normalization_applied",
        "normalized_schema_valid",
        "contract_semantically_valid",
    }
)
_FAILURE_CATEGORIES = frozenset(
    {
        "inference_runtime",
        "malformed_json",
        "schema_mismatch",
        "provenance_contract_semantic_failure",
        "provenance_semantic_failure",
        "action_prediction_failure",
        "critical_argument_prediction_failure",
    }
)
_FAILURE_CATEGORY_ORDER = (
    "inference_runtime",
    "malformed_json",
    "schema_mismatch",
    "provenance_contract_semantic_failure",
    "provenance_semantic_failure",
    "action_prediction_failure",
    "critical_argument_prediction_failure",
)


def phase2_5_trial_identity(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the exact Phase 2.5 scientific identity, including model revision."""

    return tuple(str(row.get(field, "")) for field in PHASE2_5_IDENTITY_FIELDS)


def final_phase2_5_trials(attempts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select one final success (or last error) per Phase 2.5 trial."""

    return final_trials_from_attempts(
        attempts, identity_fields=PHASE2_5_IDENTITY_FIELDS
    )


def phase2_5_attempt_accounting(
    attempts: Iterable[dict[str, Any]],
) -> dict[str, int]:
    return attempt_accounting(attempts, identity_fields=PHASE2_5_IDENTITY_FIELDS)


def completed_phase2_5_identities(
    attempts: Iterable[dict[str, Any]],
) -> set[tuple[str, ...]]:
    return {
        phase2_5_trial_identity(row)
        for row in final_phase2_5_trials(attempts)
        if row.get("status") == "completed"
    }


def next_phase2_5_attempt_index(
    attempts: Iterable[dict[str, Any]], template: Mapping[str, Any]
) -> int:
    identity = phase2_5_trial_identity(template)
    return 1 + sum(
        phase2_5_trial_identity(row) == identity for row in attempts
    )


def _scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            reduced = item()
        except Exception:  # pragma: no cover - third-party scalar boundary
            return value
        if reduced is not value:
            return reduced
    return value


def _finite_nonnegative_number(value: Any, *, field: str) -> float:
    value = _scalar(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Phase 2.5 telemetry has invalid {field}")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Phase 2.5 telemetry has invalid {field}")
    return number


def _nonnegative_integer(value: Any, *, field: str) -> int:
    value = _scalar(value)
    if isinstance(value, bool):
        raise ValueError(f"Phase 2.5 telemetry has invalid {field}")
    if isinstance(value, int):
        integer = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        integer = int(value)
    else:
        raise ValueError(f"Phase 2.5 telemetry has invalid {field}")
    if integer < 0:
        raise ValueError(f"Phase 2.5 telemetry has invalid {field}")
    return integer


def normalize_vram_bytes(value: Any, *, unit: str = "bytes") -> int | None:
    """Normalize CUDA/NVML memory metrics to integral bytes."""

    if value is None:
        return None
    normalized_unit = str(unit).strip().lower()
    if normalized_unit not in _BYTE_UNIT_MULTIPLIERS:
        raise ValueError(f"Unsupported VRAM unit: {unit!r}")
    number = _finite_nonnegative_number(value, field="VRAM value")
    result = number * _BYTE_UNIT_MULTIPLIERS[normalized_unit]
    if result > 2**63 - 1:
        raise ValueError("Phase 2.5 VRAM value is too large")
    return int(round(result))


def normalize_latency_ms(value: Any, *, unit: str = "ms") -> float | None:
    """Normalize seconds or milliseconds into finite nonnegative milliseconds."""

    if value is None:
        return None
    normalized_unit = str(unit).strip().lower()
    multiplier = {"ms": 1.0, "millisecond": 1.0, "milliseconds": 1.0,
                  "s": 1000.0, "second": 1000.0, "seconds": 1000.0}.get(
        normalized_unit
    )
    if multiplier is None:
        raise ValueError(f"Unsupported latency unit: {unit!r}")
    return _finite_nonnegative_number(value, field="latency") * multiplier


def _runtime_label(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip().lower()
    if rendered.startswith("torch."):
        rendered = rendered.removeprefix("torch.")
    if field == "model_dtype":
        rendered = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(
            rendered, rendered
        )
    return rendered or None


def normalize_phase2_5_telemetry(values: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize provider metrics into the common Phase 2.5 telemetry schema."""

    if not isinstance(values, Mapping):
        raise ValueError("Phase 2.5 telemetry must be an object")
    merged = dict(values)
    for alias, canonical in _TELEMETRY_ALIASES.items():
        if alias in merged and canonical in merged and merged[alias] != merged[canonical]:
            raise ValueError(
                f"Phase 2.5 telemetry contains conflicting {alias} and {canonical}"
            )
        if canonical not in merged and alias in merged:
            merged[canonical] = merged[alias]

    normalized: dict[str, Any] = {}
    for field in PHASE2_5_TELEMETRY_FIELDS:
        value = merged.get(field)
        if field in _LATENCY_FIELDS:
            try:
                normalized[field] = normalize_latency_ms(value)
            except ValueError as error:
                raise ValueError(
                    f"Phase 2.5 telemetry has invalid {field}"
                ) from error
        elif field in _COUNT_FIELDS:
            normalized[field] = (
                None if value is None else _nonnegative_integer(value, field=field)
            )
        elif field in _MEMORY_FIELDS:
            normalized[field] = normalize_vram_bytes(value)
        elif field in _DIMENSION_FIELDS:
            normalized[field] = (
                None if value is None else _nonnegative_integer(value, field=field)
            )
        elif field in _RUNTIME_LABEL_FIELDS:
            normalized[field] = _runtime_label(value, field=field)
        elif field == "tokens_per_second":
            normalized[field] = (
                None
                if value is None
                else _finite_nonnegative_number(value, field=field)
            )

    if normalized["tokens_per_second"] is None:
        generated = normalized["generated_tokens"]
        generation_ms = normalized["generation_latency_ms"]
        if generated is not None and generation_ms is not None and generation_ms > 0:
            normalized["tokens_per_second"] = generated / (generation_ms / 1000.0)
    return normalized


def extract_phase2_5_telemetry(row: Mapping[str, Any]) -> dict[str, Any]:
    """Read telemetry from top-level trial fields or ``local_performance``."""

    nested = row.get("local_performance")
    if nested is not None and not isinstance(nested, Mapping):
        raise ValueError("Phase 2.5 local_performance must be an object")
    source: dict[str, Any] = dict(nested or {})
    recognized = set(PHASE2_5_TELEMETRY_FIELDS) | set(_TELEMETRY_ALIASES)
    for field in recognized:
        if field not in row:
            continue
        if field in source and source[field] != row[field]:
            raise ValueError(
                f"Phase 2.5 telemetry has conflicting nested/top-level {field}"
            )
        source[field] = row[field]
    return normalize_phase2_5_telemetry(source)


def validate_phase2_5_telemetry(
    values: Mapping[str, Any], *, status: str = "completed"
) -> dict[str, Any]:
    """Validate normalized local timing, token, image, and VRAM measurements."""

    if status not in {"completed", "error"}:
        raise ValueError(f"Invalid Phase 2.5 terminal status: {status!r}")
    telemetry = normalize_phase2_5_telemetry(values)

    for field in _DIMENSION_FIELDS:
        value = telemetry[field]
        if value is not None and value < 1:
            raise ValueError(f"Phase 2.5 telemetry has invalid {field}")
    for field in _RUNTIME_LABEL_FIELDS:
        if telemetry[field] is None:
            raise ValueError(f"Phase 2.5 telemetry is missing {field}")

    if status == "completed":
        required = (
            "model_load_time_ms",
            "preprocessing_latency_ms",
            "inference_latency_ms",
            "thin_gate_latency_ms",
            "evidence_mapper_latency_ms",
            "gpu_memory_allocated_before_inference_bytes",
            "gpu_peak_memory_allocated_bytes",
            "gpu_peak_memory_reserved_bytes",
            "image_width",
            "image_height",
        )
        missing = [field for field in required if telemetry[field] is None]
        if missing:
            raise ValueError(
                f"Completed Phase 2.5 telemetry is missing required fields {missing}"
            )

    before = telemetry["gpu_memory_allocated_before_inference_bytes"]
    peak_allocated = telemetry["gpu_peak_memory_allocated_bytes"]
    peak_reserved = telemetry["gpu_peak_memory_reserved_bytes"]
    if before is not None and peak_allocated is not None and before > peak_allocated:
        raise ValueError("Phase 2.5 GPU memory before inference exceeds peak allocated")
    if peak_allocated is not None and peak_reserved is not None and peak_allocated > peak_reserved:
        raise ValueError("Phase 2.5 peak GPU memory allocated exceeds peak reserved")

    generated = telemetry["generated_tokens"]
    output = telemetry["output_token_count"]
    if generated is not None and output is not None and generated != output:
        raise ValueError("Phase 2.5 generated_tokens disagrees with output_token_count")
    rate = telemetry["tokens_per_second"]
    generation_ms = telemetry["generation_latency_ms"]
    if rate is not None and (generated is None or generation_ms is None or generation_ms <= 0):
        raise ValueError(
            "Phase 2.5 tokens_per_second requires generated_tokens and positive "
            "generation_latency_ms"
        )
    return telemetry


def _validate_identity(row: Mapping[str, Any], *, index: int) -> None:
    missing = [field for field in PHASE2_5_IDENTITY_FIELDS if field not in row]
    if missing:
        raise ValueError(f"Phase 2.5 attempt {index} lacks identity fields {missing}")
    for field in PHASE2_5_IDENTITY_FIELDS:
        value = row[field]
        if field == "run":
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"Phase 2.5 attempt {index} has invalid run")
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Phase 2.5 attempt {index} has invalid identity field {field!r}"
            )


def _validate_phase2_5_metadata(row: Mapping[str, Any], *, index: int) -> None:
    required_strings = (
        "zero_shot_prompt_version",
        "benchmark_lock_id",
        "benchmark_lock_sha256",
    )
    for field in required_strings:
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Phase 2.5 attempt {index} has invalid {field}")
    if _SHA256_PATTERN.fullmatch(str(row["benchmark_lock_sha256"])) is None:
        raise ValueError(
            f"Phase 2.5 attempt {index} has invalid benchmark_lock_sha256"
        )
    selected = row.get("selected_case_count")
    benchmark = row.get("benchmark_case_count")
    if (
        not isinstance(selected, int)
        or isinstance(selected, bool)
        or selected < 1
        or not isinstance(benchmark, int)
        or isinstance(benchmark, bool)
        or benchmark < selected
    ):
        raise ValueError(f"Phase 2.5 attempt {index} has invalid case-scope metadata")
    structured = row.get("structured_output_valid")
    if not isinstance(structured, bool):
        raise ValueError(
            f"Phase 2.5 attempt {index} has invalid structured_output_valid"
        )
    if row.get("status") == "completed" and structured is not True:
        raise ValueError(
            f"Phase 2.5 completed attempt {index} has invalid structured output"
        )
    if row.get("zero_shot_prompt_version") == _V2_PROFILE:
        _validate_v2_output_contract(row, index=index)


def _validate_optional_boolean(value: Any, *, field: str, index: int) -> None:
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"Phase 2.5 attempt {index} has invalid {field}")


def _validate_v2_output_contract(row: Mapping[str, Any], *, index: int) -> None:
    """Validate the additive V2 diagnostics without rejecting legacy V1 evidence."""

    transport = row.get("schema_transport_version")
    if not isinstance(transport, str) or not transport.strip():
        raise ValueError(
            f"Phase 2.5 attempt {index} has invalid schema_transport_version"
        )
    for field in _CONTRACT_BOOLEAN_FIELDS:
        if not isinstance(row.get(field), bool):
            raise ValueError(f"Phase 2.5 attempt {index} has invalid {field}")
    for field in _TRI_STATE_FIELDS:
        _validate_optional_boolean(row.get(field), field=field, index=index)

    parse_success = row["parse_success"]
    schema_valid = row["schema_valid"]
    normalized_schema_valid = row["normalized_schema_valid"]
    normalized = row["normalization_applied"]
    contract_semantically_valid = row["contract_semantically_valid"]
    method = row.get("normalization_method")
    if normalized:
        if not isinstance(method, str) or not method.strip():
            raise ValueError(
                f"Phase 2.5 attempt {index} normalized output lacks normalization_method"
            )
        if schema_valid:
            raise ValueError(
                f"Phase 2.5 attempt {index} normalized output cannot be raw-schema valid"
            )
    elif method is not None:
        raise ValueError(
            f"Phase 2.5 attempt {index} has normalization_method without normalization"
        )
    if schema_valid and not parse_success:
        raise ValueError(
            f"Phase 2.5 attempt {index} schema_valid requires parse_success"
        )
    if normalized_schema_valid and not parse_success:
        raise ValueError(
            f"Phase 2.5 attempt {index} normalized_schema_valid requires parse_success"
        )
    if schema_valid and not normalized_schema_valid:
        raise ValueError(
            f"Phase 2.5 attempt {index} raw schema validity requires normalized schema validity"
        )
    if normalized and not normalized_schema_valid:
        raise ValueError(
            f"Phase 2.5 attempt {index} normalization requires normalized schema validity"
        )
    if contract_semantically_valid and not normalized_schema_valid:
        raise ValueError(
            f"Phase 2.5 attempt {index} contract semantics require normalized schema validity"
        )
    usable = bool(normalized_schema_valid and contract_semantically_valid)
    if row["structured_output_valid"] is not usable:
        raise ValueError(
            f"Phase 2.5 attempt {index} structured_output_valid disagrees with "
            "post-normalization contract validity"
        )

    action_correct = row.get("action_correct")
    critical_correct = row.get("critical_argument_correct")
    candidate = row.get("action_candidate")
    if candidate is not None and (
        not isinstance(candidate, Mapping)
        or set(candidate) != {"action", "arguments"}
        or not isinstance(candidate.get("action"), str)
        or not isinstance(candidate.get("arguments"), Mapping)
    ):
        raise ValueError(f"Phase 2.5 attempt {index} has invalid action_candidate")
    if (action_correct is not None or critical_correct is not None) and candidate is None:
        raise ValueError(
            f"Phase 2.5 attempt {index} action diagnostics require action_candidate"
        )
    if action_correct is False and critical_correct is True:
        raise ValueError(
            f"Phase 2.5 attempt {index} critical_argument_correct requires action_correct"
        )
    if row.get("gate_decision") is None and row.get("unsafe_execution") is not None:
        raise ValueError(
            f"Phase 2.5 attempt {index} unsafe_execution must be null when the gate did not run"
        )
    if isinstance(row.get("unsafe_automatic_execution"), bool):
        if row.get("unsafe_execution") is not row["unsafe_automatic_execution"]:
            raise ValueError(
                f"Phase 2.5 attempt {index} unsafe_execution alias is inconsistent"
            )

    failure_category = row.get("failure_category")
    failure_categories = row.get("failure_categories")
    if failure_category is not None and failure_category not in _FAILURE_CATEGORIES:
        raise ValueError(f"Phase 2.5 attempt {index} has invalid failure_category")
    if not isinstance(failure_categories, list) or any(
        not isinstance(category, str) or category not in _FAILURE_CATEGORIES
        for category in failure_categories
    ):
        raise ValueError(f"Phase 2.5 attempt {index} has invalid failure_categories")
    if len(failure_categories) != len(set(failure_categories)):
        raise ValueError(f"Phase 2.5 attempt {index} has duplicate failure_categories")
    expected_order = [
        category for category in _FAILURE_CATEGORY_ORDER if category in failure_categories
    ]
    if failure_categories != expected_order:
        raise ValueError(
            f"Phase 2.5 attempt {index} has unstable failure_categories order"
        )
    expected_primary = failure_categories[0] if failure_categories else None
    if failure_category != expected_primary:
        raise ValueError(
            f"Phase 2.5 attempt {index} failure_category is not the primary category"
        )
    if row.get("status") == "error" and not failure_categories:
        raise ValueError(
            f"Phase 2.5 error attempt {index} lacks a failure category"
        )
    if "malformed_json" in failure_categories and parse_success:
        raise ValueError(
            f"Phase 2.5 attempt {index} malformed_json contradicts parse_success"
        )
    if "schema_mismatch" in failure_categories and (
        not parse_success or schema_valid
    ):
        raise ValueError(
            f"Phase 2.5 attempt {index} schema_mismatch contradicts contract flags"
        )


def _legacy_mixed_evidence_status(evaluation: Any) -> str | None:
    """Identify the one multi-item summary the frozen validator cannot express.

    The frozen mapper deliberately reports ``text_match_correct=False`` when an
    argument has at least one mapped supporting item and at least one missing or
    hallucinated item.  It summarizes that combination as ``ambiguous``.  The
    frozen result validator predates multi-item provider evidence and requires
    every ambiguous aggregate to have ``text_match_correct=True``.  Phase 2.5
    retains the mapper's real aggregate and item audits, but presents the legacy
    validator with a conservative unresolved aggregate status in a deep copy.
    """

    if not isinstance(evaluation, Mapping):
        return None
    if (
        evaluation.get("evidence_status") != "ambiguous"
        or evaluation.get("text_match_correct") is not False
    ):
        return None
    items = evaluation.get("reported_evidence_items")
    if not isinstance(items, list):
        return None
    has_mapped_support = any(
        isinstance(item, Mapping)
        and item.get("evidence_status") == "matched"
        and item.get("supports_argument") is True
        for item in items
    )
    has_unresolved_text = any(
        isinstance(item, Mapping)
        and item.get("evidence_status") in {"hallucinated", "missing"}
        for item in items
    )
    if not (has_mapped_support and has_unresolved_text):
        return None
    return (
        "hallucinated"
        if any(
            isinstance(item, Mapping)
            and item.get("evidence_status") == "hallucinated"
            for item in items
        )
        else "missing"
    )


def _as_phase2_core_row(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    # A deep copy is required because the compatibility view below must never
    # alter the persisted Phase 2.5 evidence or its semantic metrics.
    core = deepcopy(dict(row))
    existing_model = core.get("model")
    if existing_model not in (None, "", row["model_id"]):
        raise ValueError(
            f"Phase 2.5 attempt {index} has model inconsistent with model_id"
        )
    core["model"] = row["model_id"]
    evaluations = core.get("provenance_evaluations")
    if row.get("zero_shot_prompt_version") == _V2_PROFILE and isinstance(
        evaluations, list
    ):
        for evaluation in evaluations:
            legacy_status = _legacy_mixed_evidence_status(evaluation)
            if legacy_status is not None:
                # The frozen validator has no aggregate state for a mixture of
                # mapped and unresolved items. Give only its validation copy a
                # conservative unresolved status. The persisted V2 row keeps
                # ``ambiguous``/False and every per-item audit unchanged.
                evaluation["evidence_status"] = legacy_status
    return core


def validate_phase2_5_attempts(
    attempts: Iterable[dict[str, Any]],
    *,
    core_validator: CoreValidator = validate_phase2_attempts,
) -> None:
    """Validate extensions, then delegate frozen scientific checks to Phase 2."""

    records = list(attempts)
    core_records: list[dict[str, Any]] = []
    for index, row in enumerate(records, 1):
        if not isinstance(row, Mapping):
            raise ValueError(f"Phase 2.5 attempt {index} is not an object")
        _validate_identity(row, index=index)
        status = row.get("status")
        if status not in {"completed", "error"}:
            raise ValueError(f"Phase 2.5 attempt {index} has invalid status {status!r}")
        _validate_phase2_5_metadata(row, index=index)
        validate_phase2_5_telemetry(
            extract_phase2_5_telemetry(row), status=str(status)
        )
        core_records.append(_as_phase2_core_row(row, index=index))
    core_validator(core_records)


def persist_phase2_5_attempt(
    raw_attempts_path: str | Path,
    final_trials_path: str | Path,
    row: dict[str, Any],
    *,
    core_validator: CoreValidator = validate_phase2_attempts,
) -> None:
    """Validate, append durably, then atomically refresh deduplicated CSV."""

    # Refuse unsupported scalars and NaN before the irreversible append.
    try:
        json.dumps(row, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Phase 2.5 attempt is not strict-JSON serializable: {error}") from error
    existing = read_jsonl(raw_attempts_path)
    validate_phase2_5_attempts(
        [*existing, row], core_validator=core_validator
    )
    append_jsonl(raw_attempts_path, row)
    write_csv(final_trials_path, final_phase2_5_trials([*existing, row]))


def assert_phase2_5_resume_compatible(
    attempts: Iterable[dict[str, Any]],
    *,
    provider: str,
    model_id: str,
    model_revision: str,
    dataset_version: str,
    policy_version: str,
    prompt_version: str | None = None,
    prompt_versions: Iterable[str] | None = None,
    model_dtype: str | None = None,
    quantization: str | None = None,
    attention_backend: str | None = None,
    zero_shot_prompt_version: str | None = None,
    schema_transport_version: str | None = None,
) -> None:
    """Refuse resume into another model, benchmark, prompt, or runtime profile."""

    records = list(attempts)
    expected = {
        "provider": provider,
        "model_id": model_id,
        "model_revision": model_revision,
        "dataset_version": dataset_version,
        "policy_version": policy_version,
    }
    for field, value in expected.items():
        observed = {str(row.get(field, "")) for row in records}
        if "" in observed:
            raise ValueError(f"Phase 2.5 attempts are missing {field!r}; refusing resume")
        if observed and observed != {str(value)}:
            raise ValueError(
                f"Phase 2.5 attempts contain incompatible {field} values "
                f"{sorted(observed)}; expected {value!r}"
            )

    if prompt_version is not None and prompt_versions is not None:
        raise ValueError("Specify prompt_version or prompt_versions, not both")
    allowed_prompts = (
        {str(prompt_version)}
        if prompt_version is not None
        else {str(value) for value in prompt_versions}
        if prompt_versions is not None
        else None
    )
    observed_prompts = {str(row.get("prompt_version", "")) for row in records}
    if "" in observed_prompts:
        raise ValueError("Phase 2.5 attempts are missing 'prompt_version'; refusing resume")
    if allowed_prompts is not None and not observed_prompts.issubset(allowed_prompts):
        raise ValueError(
            "Phase 2.5 attempts contain incompatible prompt_version values "
            f"{sorted(observed_prompts)}; expected a subset of {sorted(allowed_prompts)}"
        )

    expected_profile = {
        "model_dtype": model_dtype,
        "quantization": quantization,
        "attention_backend": attention_backend,
    }
    for field, value in expected_profile.items():
        if value is None:
            continue
        expected_value = _runtime_label(value, field=field)
        observed = {extract_phase2_5_telemetry(row)[field] for row in records}
        if observed and observed != {expected_value}:
            raise ValueError(
                f"Phase 2.5 attempts contain incompatible {field} values "
                f"{sorted(str(item) for item in observed)}; expected {expected_value!r}"
            )

    for field, expected_value in (
        ("zero_shot_prompt_version", zero_shot_prompt_version),
        ("schema_transport_version", schema_transport_version),
    ):
        if expected_value is None:
            continue
        observed = {str(_resume_profile_value(row, field) or "") for row in records}
        if "" in observed:
            raise ValueError(f"Phase 2.5 attempts are missing {field!r}; refusing resume")
        if observed and observed != {str(expected_value)}:
            raise ValueError(
                f"Phase 2.5 attempts contain incompatible {field} values "
                f"{sorted(observed)}; expected {expected_value!r}"
            )


def _resume_profile_value(row: Mapping[str, Any], field: str) -> Any:
    value = row.get(field)
    if value not in (None, ""):
        return value
    provider_config = row.get("provider_config")
    return provider_config.get(field) if isinstance(provider_config, Mapping) else None


def _decode_csv_boolean(value: str, *, field: str, line_number: int) -> bool | None:
    normalized = value.strip().casefold()
    if not normalized:
        return None
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(
        f"Invalid Phase 2.5 CSV boolean {field!r} at line {line_number}: {value!r}"
    )


def read_phase2_5_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read final-trial CSV while preserving V2 boolean/null diagnostics."""

    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    with source.open(newline="", encoding="utf-8") as handle:
        for line_number, raw in enumerate(csv.DictReader(handle), 2):
            row: dict[str, Any] = dict(raw)
            for field in _CONTRACT_BOOLEAN_FIELDS | _TRI_STATE_FIELDS:
                if field in row:
                    row[field] = _decode_csv_boolean(
                        row[field], field=field, line_number=line_number
                    )
            if "failure_categories" in row:
                value = row["failure_categories"].strip()
                try:
                    decoded = json.loads(value) if value else []
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid Phase 2.5 CSV failure_categories at line {line_number}"
                    ) from error
                if not isinstance(decoded, list):
                    raise ValueError(
                        f"Invalid Phase 2.5 CSV failure_categories at line {line_number}"
                    )
                row["failure_categories"] = decoded
            if "failure_category" in row and not row["failure_category"].strip():
                row["failure_category"] = None
            if "normalization_method" in row and not row["normalization_method"].strip():
                row["normalization_method"] = None
            rows.append(row)
    return rows


# Additive convenience aliases; the explicit Phase 2.5 names above are canonical.
trial_identity = phase2_5_trial_identity
final_trials = final_phase2_5_trials
completed_identities = completed_phase2_5_identities
next_attempt_index = next_phase2_5_attempt_index
persist_attempt = persist_phase2_5_attempt
assert_resume_compatible = assert_phase2_5_resume_compatible
