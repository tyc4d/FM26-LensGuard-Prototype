"""Deterministic, quota-free Phase 2 provider.

Unlike the Gemini implementation, this test provider intentionally consumes
scenario metadata so every arm and policy branch can be exercised without API
calls.  Mock rows must remain separately labelled and are never scientific
Gemini evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from firewall.action_normalizer import (
    normalize_destination,
    normalize_direction,
    normalize_phone_number,
    normalize_url,
)
from phase2_schema import (
    ActionOnlyOutput,
    ArgumentEvidence,
    EvidenceOnlyOutput,
    InlineProvenanceOutput,
    Phase2Arm,
    Phase2ArmResult,
    Phase2Operation,
    SourceTypeEstimate,
    call_metadata_from_response,
    canonical_phase2_arm,
    coerce_action_output,
    validate_evidence_for_action,
)

from .base import ProviderResponse

PHASE2_MOCK_MODEL = "mock-phase2-deterministic-v1"
_MOCK_PROMPT_VERSIONS = {
    Phase2Operation.ACTION_ONLY: "phase2-action-v1",
    Phase2Operation.INLINE_PROVENANCE: "phase2-inline-provenance-v2",
    Phase2Operation.TWO_PASS_EVIDENCE: "phase2-two-pass-evidence-v2",
}

ATTACK_CONDITIONS = frozenset(
    {
        "BENIGN_UNTRUSTED_SUBSTITUTION",
        "AUTHORITY_IMPERSONATION",
        "OBVIOUS_INJECTION_CONTROL",
    }
)
GROUND_TRUTH_CONDITIONS = frozenset({"CLEAN_TRUSTED", "EXPLICIT_USER_OVERRIDE"})
SPECIAL_CONDITIONS = frozenset({"NO_VERIFIED_GROUND_TRUTH", "TRUSTED_BUT_CONFLICTING_UPDATE"})

_PRIMARY_ARGUMENT = {
    "CALL": "target_number",
    "OPEN_URL": "url",
    "DIRECTION_ADVICE": "direction",
}

_SOURCE_TYPE_ALIASES = {item.value: item for item in SourceTypeEstimate}
_SOURCE_TYPE_ALIASES.update(
    {
        "user_prompt": SourceTypeEstimate.EXPLICIT_USER,
        "qr_code": SourceTypeEstimate.QR_CODE_UNVERIFIED,
        "neutral_distractor": SourceTypeEstimate.UNKNOWN_VISUAL_SOURCE,
    }
)


def _action_name(value: Any) -> str:
    return str(getattr(value, "value", value)).strip().upper()


def _primary_argument(scenario: Mapping[str, Any], action: str) -> str:
    raw = scenario.get("critical_argument_name") or _PRIMARY_ARGUMENT.get(action)
    if not raw:
        raise ValueError(f"No primary critical argument for action {action!r}")
    return str(raw)


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return dict(value)


def _selected_value(
    scenario: Mapping[str, Any],
    condition: str,
    behavior: str,
    action: str,
    argument: str,
    seed: int,
) -> Any:
    if behavior == "attacker_target":
        value = scenario.get("attacker_target")
        if value in (None, ""):
            raise ValueError("attacker_target behavior requires attacker_target")
        return value
    if behavior == "trusted_update":
        value = scenario.get("trusted_update_value")
        if isinstance(value, Mapping):
            value = value.get(argument)
        if value in (None, ""):
            raise ValueError("trusted_update behavior requires trusted_update_value")
        return value
    if behavior == "arbitrary_wrong":
        configured = scenario.get("arbitrary_wrong_target") or scenario.get("random_wrong_target")
        if configured not in (None, ""):
            return configured
        material = (
            f"{seed}:{scenario.get('scenario_id', '')}:{condition}:{action}:{argument}"
        ).encode()
        suffix = int(hashlib.sha256(material).hexdigest()[:8], 16)
        if action == "CALL":
            return f"0999{suffix % 1_000_000:06d}"
        if action == "OPEN_URL":
            return f"https://wrong-{suffix % 100_000}.invalid"
        if action == "DIRECTION_ADVICE":
            current = str(
                _mapping(
                    scenario.get("ground_truth_arguments", {}),
                    field="ground_truth_arguments",
                ).get("direction", "")
            ).upper()
            return next(
                candidate
                for candidate in ("LEFT", "RIGHT", "STRAIGHT", "BACK")
                if candidate != current
            )
        return f"wrong-{suffix:08x}"
    raise ValueError(f"Unsupported selected-value behavior: {behavior!r}")


def _normalize_value(action: str, argument: str, value: Any) -> str:
    raw = str(value)
    if action == "CALL" and argument == "target_number":
        return normalize_phone_number(raw)
    if action == "OPEN_URL" and argument == "url":
        return normalize_url(raw)
    if action == "DIRECTION_ADVICE" and argument == "direction":
        return normalize_direction(raw)
    if action == "DIRECTION_ADVICE" and argument == "destination":
        return normalize_destination(raw)
    return raw.strip()


def _values_match(action: str, argument: str, left: Any, right: Any) -> bool:
    try:
        return _normalize_value(action, argument, left) == _normalize_value(action, argument, right)
    except (TypeError, ValueError):
        return str(left).strip() == str(right).strip()


def _source_category(value: Any) -> SourceTypeEstimate:
    normalized = re.sub(r"[\s-]+", "_", str(value or "unknown").strip().lower())
    return _SOURCE_TYPE_ALIASES.get(normalized, SourceTypeEstimate.UNKNOWN)


def _prompt_mentions(user_prompt: str, value: str) -> bool:
    def comparable(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.casefold())

    needle = comparable(value)
    return bool(needle) and needle in comparable(user_prompt)


class MockPhase2Provider:
    """Deterministic Phase 2 arm provider driven by trusted fixture metadata."""

    def __init__(self, *, seed: int = 0, latency_ms_per_call: float = 1.0) -> None:
        if latency_ms_per_call < 0:
            raise ValueError("latency_ms_per_call cannot be negative")
        self.seed = int(seed)
        self.latency_ms_per_call = float(latency_ms_per_call)
        self._request_seed = self.seed
        self._call_count = 0

    @property
    def model_identifier(self) -> str:
        return PHASE2_MOCK_MODEL

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def experiment_config(self) -> dict[str, Any]:
        return {
            "provider": "mock",
            "provider_interface": "phase2",
            "seed": self.seed,
            "latency_ms_per_call": self.latency_ms_per_call,
            "prompt_versions": {
                operation.value: version for operation, version in _MOCK_PROMPT_VERSIONS.items()
            },
            "sdk_internal_retries": 0,
            "phase1_consequence_model_used": False,
        }

    def set_request_seed(self, seed: int) -> None:
        self._request_seed = int(seed)

    def close(self) -> None:
        return None

    def _scenario(self, scenario: Mapping[str, Any] | None) -> Mapping[str, Any]:
        if scenario is None:
            raise ValueError("MockPhase2Provider requires scenario metadata")
        return scenario

    def _action_output(
        self,
        scenario: Mapping[str, Any],
        arm: Phase2Arm,
    ) -> ActionOnlyOutput:
        per_arm = scenario.get("mock_phase2_outputs")
        if isinstance(per_arm, Mapping):
            explicit = per_arm.get(arm.value) or per_arm.get(arm.value.lower())
            if isinstance(explicit, Mapping):
                if "action_output" in explicit:
                    explicit = explicit["action_output"]
                return ActionOnlyOutput.model_validate(explicit)

        explicit = scenario.get("mock_action_output") or scenario.get("mock_proposed_action")
        if explicit is not None:
            return ActionOnlyOutput.model_validate(explicit)

        action = _action_name(
            scenario.get("ground_truth_action") or scenario.get("action_family") or "NONE"
        )
        arguments = _mapping(
            scenario.get("ground_truth_arguments", {}),
            field="ground_truth_arguments",
        )
        condition = str(scenario.get("condition", "")).upper()
        behavior = str(scenario.get("mock_behavior", "")).strip().lower()
        if not behavior:
            if scenario.get("trusted_user_override") or condition in GROUND_TRUTH_CONDITIONS:
                behavior = "ground_truth"
            elif condition == "TRUSTED_BUT_CONFLICTING_UPDATE":
                behavior = "trusted_update"
            elif condition in ATTACK_CONDITIONS or condition == "NO_VERIFIED_GROUND_TRUTH":
                behavior = "attacker_target"
            else:
                behavior = "ground_truth"

        if behavior == "none":
            action = "NONE"
            arguments = {}
        elif behavior != "ground_truth":
            argument = _primary_argument(scenario, action)
            arguments[argument] = _selected_value(
                scenario,
                condition,
                behavior,
                action,
                argument,
                self._request_seed,
            )

        return ActionOnlyOutput.model_validate({"action": action, "arguments": arguments})

    def _claim_region(
        self,
        scenario: Mapping[str, Any],
        action: str,
        argument: str,
        value: str,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
        regions = scenario.get("regions", [])
        if not isinstance(regions, Sequence) or isinstance(regions, (str, bytes)):
            raise TypeError("regions must be a list")
        for region in regions:
            if not isinstance(region, Mapping):
                continue
            claims = region.get("claims", [])
            if not isinstance(claims, Sequence) or isinstance(claims, (str, bytes)):
                continue
            for claim in claims:
                if not isinstance(claim, Mapping):
                    continue
                claim_action = _action_name(claim.get("action", action))
                claim_argument = str(claim.get("argument") or claim.get("argument_name") or "")
                if claim_action != action or claim_argument != argument:
                    continue
                if _values_match(action, argument, claim.get("value"), value):
                    return region, claim
        return None

    def _evidence(
        self,
        user_prompt: str,
        scenario: Mapping[str, Any],
        action_output: ActionOnlyOutput,
        arm: Phase2Arm,
    ) -> dict[str, list[ArgumentEvidence]]:
        per_arm = scenario.get("mock_phase2_outputs")
        if isinstance(per_arm, Mapping):
            explicit = per_arm.get(arm.value) or per_arm.get(arm.value.lower())
            if isinstance(explicit, Mapping) and "argument_evidence" in explicit:
                raw_evidence = explicit["argument_evidence"]
                if not isinstance(raw_evidence, Mapping):
                    raise TypeError("mock argument_evidence must be an object")
                return self._validated_evidence_mapping(raw_evidence)
        explicit_evidence = scenario.get("mock_argument_evidence")
        if explicit_evidence is not None:
            if not isinstance(explicit_evidence, Mapping):
                raise TypeError("mock_argument_evidence must be an object")
            return self._validated_evidence_mapping(explicit_evidence)

        action = action_output.action.value
        condition = str(scenario.get("condition", "")).upper()
        primary = _PRIMARY_ARGUMENT.get(action)
        evidence: dict[str, list[ArgumentEvidence]] = {}
        for argument, value in action_output.argument_values().items():
            user_supplied = (
                (scenario.get("trusted_user_override") or condition == "EXPLICIT_USER_OVERRIDE")
                and argument == primary
            ) or (argument == "destination" and _prompt_mentions(user_prompt, value))
            if user_supplied:
                evidence[argument] = [
                    ArgumentEvidence(
                        evidence_text=value,
                        source_type_estimate=SourceTypeEstimate.EXPLICIT_USER,
                        bbox=None,
                        confidence=1.0,
                    )
                ]
                continue

            match = self._claim_region(scenario, action, argument, value)
            if match is not None:
                region, claim = match
                evidence_text = claim.get("evidence_text") or claim.get("value") or value
                evidence[argument] = [
                    ArgumentEvidence(
                        evidence_text=str(evidence_text),
                        source_type_estimate=_source_category(region.get("source_type")),
                        bbox=region.get("bbox"),
                        confidence=1.0,
                    )
                ]
                continue

            evidence[argument] = []
        return evidence

    @staticmethod
    def _validated_evidence_mapping(
        raw_evidence: Mapping[str, Any],
    ) -> dict[str, list[ArgumentEvidence]]:
        result: dict[str, list[ArgumentEvidence]] = {}
        for key, raw_items in raw_evidence.items():
            if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes, Mapping)):
                raise TypeError("mock argument evidence values must be lists")
            result[str(key)] = [ArgumentEvidence.model_validate(item) for item in raw_items]
        return result

    @staticmethod
    def _approximate_tokens(text: str) -> int:
        return max(1, (len(text.encode("utf-8")) + 3) // 4)

    def _response(
        self,
        operation: Phase2Operation,
        parsed: ActionOnlyOutput | InlineProvenanceOutput | EvidenceOnlyOutput,
        *,
        user_prompt: str,
    ) -> ProviderResponse[Any]:
        raw = json.dumps(
            parsed.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        input_tokens = 1092 + self._approximate_tokens(user_prompt)
        output_tokens = self._approximate_tokens(raw)
        self._call_count += 1
        # Keep the declared mock latency physically represented so the runner's
        # wall-clock end-to-end metric cannot be shorter than its model-time
        # component. This is still fixture timing, never performance evidence.
        if self.latency_ms_per_call:
            time.sleep(self.latency_ms_per_call / 1000.0)
        metadata = {
            "status": "completed",
            "requested_model": self.model_identifier,
            "returned_model": self.model_identifier,
            "errors": [],
            "mock": True,
            "operation": operation.value,
            "prompt_version": _MOCK_PROMPT_VERSIONS[operation],
            "mock_call_index": self._call_count,
            "request_generation_config": {"seed": self._request_seed},
            "usage": {
                "total_input_tokens": input_tokens,
                "total_output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "total_cached_tokens": 0,
                "total_thought_tokens": 0,
            },
        }
        return ProviderResponse(
            parsed=parsed,
            raw_response=raw,
            latency_ms=self.latency_ms_per_call,
            attempts=1,
            model=self.model_identifier,
            response_metadata=metadata,
        )

    def action_only(
        self,
        user_prompt: str,
        image_path: str | Path,
        scenario: Mapping[str, Any] | None = None,
    ) -> ProviderResponse[ActionOnlyOutput]:
        del image_path
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("user_prompt must be a non-empty string")
        metadata = self._scenario(scenario)
        parsed = self._action_output(metadata, Phase2Arm.ACTION_ONLY)
        return self._response(
            Phase2Operation.ACTION_ONLY,
            parsed,
            user_prompt=user_prompt,
        )

    def inline_provenance(
        self,
        user_prompt: str,
        image_path: str | Path,
        scenario: Mapping[str, Any] | None = None,
    ) -> ProviderResponse[InlineProvenanceOutput]:
        del image_path
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("user_prompt must be a non-empty string")
        metadata = self._scenario(scenario)
        action = self._action_output(metadata, Phase2Arm.INLINE_PROVENANCE)
        parsed = InlineProvenanceOutput(
            action=action.action,
            arguments=action.arguments,
            argument_evidence=self._evidence(
                user_prompt,
                metadata,
                action,
                Phase2Arm.INLINE_PROVENANCE,
            ),
        )
        return self._response(
            Phase2Operation.INLINE_PROVENANCE,
            parsed,
            user_prompt=user_prompt,
        )

    def two_pass_evidence(
        self,
        user_prompt: str,
        image_path: str | Path,
        proposed_action: ActionOnlyOutput | Mapping[str, Any] | Any,
        scenario: Mapping[str, Any] | None = None,
    ) -> ProviderResponse[EvidenceOnlyOutput]:
        del image_path
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("user_prompt must be a non-empty string")
        metadata = self._scenario(scenario)
        action = coerce_action_output(proposed_action)
        parsed = EvidenceOnlyOutput(
            argument_evidence=self._evidence(
                user_prompt,
                metadata,
                action,
                Phase2Arm.TWO_PASS_PROVENANCE,
            )
        )
        parsed = validate_evidence_for_action(action, parsed)
        second_pass_text = json.dumps(
            {
                "trusted_user_request": user_prompt,
                "proposed_action": {
                    "action": action.action.value,
                    "arguments": action.arguments.model_dump(mode="json", exclude_none=True),
                },
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return self._response(
            Phase2Operation.TWO_PASS_EVIDENCE,
            parsed,
            user_prompt=second_pass_text,
        )

    def run_arm(
        self,
        arm: Phase2Arm | str,
        user_prompt: str,
        image_path: str | Path,
        scenario: Mapping[str, Any] | None = None,
        *,
        reused_action_response: ProviderResponse[ActionOnlyOutput] | None = None,
    ) -> Phase2ArmResult:
        selected_arm = canonical_phase2_arm(arm)

        if selected_arm is Phase2Arm.INLINE_PROVENANCE:
            inline = self.inline_provenance(user_prompt, image_path, scenario)
            return Phase2ArmResult(
                arm=selected_arm,
                action_output=inline.parsed.action_output(),
                argument_evidence=inline.parsed.argument_evidence,
                calls=[call_metadata_from_response(Phase2Operation.INLINE_PROVENANCE, inline)],
            )

        if selected_arm is Phase2Arm.ORACLE_PROVENANCE and reused_action_response is not None:
            action_response = reused_action_response
            reused = True
        else:
            action_response = self.action_only(user_prompt, image_path, scenario)
            reused = False

        action_call = call_metadata_from_response(
            Phase2Operation.ACTION_ONLY,
            action_response,
        )
        if selected_arm in {
            Phase2Arm.ACTION_ONLY,
            Phase2Arm.ORACLE_PROVENANCE,
        }:
            return Phase2ArmResult(
                arm=selected_arm,
                action_output=action_response.parsed,
                calls=[action_call],
                reused_action_only=reused,
            )

        evidence = self.two_pass_evidence(
            user_prompt,
            image_path,
            action_response.parsed,
            scenario,
        )
        return Phase2ArmResult(
            arm=selected_arm,
            action_output=action_response.parsed,
            argument_evidence=evidence.parsed.argument_evidence,
            calls=[
                action_call,
                call_metadata_from_response(Phase2Operation.TWO_PASS_EVIDENCE, evidence),
            ],
        )


__all__ = ["PHASE2_MOCK_MODEL", "MockPhase2Provider"]
