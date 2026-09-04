"""Model-free Thin Trusted Gate for LensGuard Phase 3.5.

The gate composes strict reference validation, deterministic grounding, and the
task-specific evidence policy.  It has no action execution capability and never
changes a proposed argument.  Consequently an attacker-selected value remains
visible in the returned audit record even when the decision is ``BLOCK``.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from phase3_5_constants import EVIDENCE_SCHEMA_VERSION, MODEL_CONTRACT_VERSION
from provenance.grounding_validator_phase3_5 import (
    GroundingStatus,
    GroundingValidationResult,
    validate_argument_grounding,
)
from provenance.reference_validator_phase3_5 import (
    EvidenceReferenceValidation,
    ReferenceIssue,
    registry_frame_id,
    validate_evidence_references,
)

from .task_policy_phase3_5 import (
    PolicyDisposition,
    TaskPolicyResult,
    evaluate_task_evidence_policy,
    load_phase3_5_action_registry,
)


class GateDecision(StrEnum):
    ALLOW = "ALLOW"
    ESCALATE = "ESCALATE"
    BLOCK = "BLOCK"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Phase35ArgumentGateAssessment(_StrictModel):
    argument_name: str
    argument_value: Any = None
    evidence_references: tuple[str, ...] = ()
    grounding_status: GroundingStatus
    grounding_confidence: float | None = None
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()
    unreferenced_conflicting_evidence_ids: tuple[str, ...] = ()
    evidence_relationship_satisfied: bool
    policy_disposition: PolicyDisposition
    policy_rule_ids: tuple[str, ...] = ()


class Phase35GateDecision(_StrictModel):
    decision: GateDecision
    action: str
    proposed_arguments: dict[str, Any] = Field(default_factory=dict)
    argument_evidence_refs: dict[str, Any] = Field(default_factory=dict)
    reference_contract_valid: bool
    reference_issues: tuple[ReferenceIssue, ...] = ()
    grounding_statuses: dict[str, GroundingStatus] = Field(default_factory=dict)
    argument_assessments: dict[str, Phase35ArgumentGateAssessment] = Field(
        default_factory=dict
    )
    static_effects: tuple[str, ...] = ()
    reversibility: str
    default_risk: str
    policy_rules_triggered: tuple[str, ...] = ()
    user_message: str
    evidence_schema_version: str = EVIDENCE_SCHEMA_VERSION
    model_contract_version: str = MODEL_CONTRACT_VERSION
    policy_version: str
    action_registry_version: str
    frame_id: str | None = None
    thin_gate_latency_ms: float = Field(ge=0.0)
    dry_run: bool = True
    auto_corrected: bool = False
    model_free: bool = True


def _payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python", exclude_none=True)
        if isinstance(dumped, Mapping):
            return dict(dumped)
    return {}


def _action_string(payload: Mapping[str, Any]) -> str:
    raw = payload.get("action", "")
    return str(getattr(raw, "value", raw)).strip().upper()


def _assert_consistent_precomputed(
    action: str,
    registry: Any,
    references: EvidenceReferenceValidation,
    grounding: GroundingValidationResult,
    task_policy: TaskPolicyResult,
) -> None:
    frame_id = registry_frame_id(registry)
    if references.action != action or grounding.action != action or task_policy.action != action:
        raise ValueError("precomputed Phase 3.5 results refer to a different action")
    if references.frame_id != frame_id or grounding.frame_id != frame_id:
        raise ValueError("precomputed Phase 3.5 results refer to a different registry frame")
    if grounding.reference_validation != references:
        raise ValueError("grounding result was not derived from the supplied reference validation")


def evaluate_thin_gate_phase3_5(
    action_output: Any,
    registry: Any,
    *,
    user_intent: str = "",
    frame_id: str | None = None,
    reference_validation: EvidenceReferenceValidation | None = None,
    grounding: GroundingValidationResult | None = None,
    task_policy_result: TaskPolicyResult | None = None,
    policy: Mapping[str, Any] | str | Path | None = None,
    action_registry: Mapping[str, Any] | str | Path | None = None,
) -> Phase35GateDecision:
    """Return ``ALLOW``, ``ESCALATE``, or ``BLOCK`` without executing an action."""

    # user_intent is an explicit architectural input, but authorization is
    # represented by pre-built USER evidence.  The gate must not parse free text
    # into a hidden authority channel.
    if not isinstance(user_intent, str):
        raise TypeError("user_intent must be a string")

    payload = _payload(action_output)
    action = _action_string(payload)
    if reference_validation is None:
        reference_validation = validate_evidence_references(
            action_output,
            registry,
            frame_id=frame_id,
        )
    if grounding is None:
        grounding = validate_argument_grounding(
            action_output,
            registry,
            reference_validation=reference_validation,
            frame_id=frame_id,
        )
    if task_policy_result is None:
        task_policy_result = evaluate_task_evidence_policy(
            action_output,
            registry,
            grounding,
            policy=policy,
            action_registry=action_registry,
        )

    _assert_consistent_precomputed(
        action,
        registry,
        reference_validation,
        grounding,
        task_policy_result,
    )
    registry_config = load_phase3_5_action_registry(action_registry)
    definition = registry_config["actions"].get(action)
    if not isinstance(definition, Mapping):
        raise ValueError(f"action {action!r} is absent from Phase 3.5 action registry")

    decision_started = perf_counter()
    if not reference_validation.contract_valid:
        decision = GateDecision.BLOCK
    elif task_policy_result.disposition is PolicyDisposition.BLOCK:
        decision = GateDecision.BLOCK
    elif task_policy_result.disposition is PolicyDisposition.ESCALATE:
        decision = GateDecision.ESCALATE
    else:
        decision = GateDecision.ALLOW

    argument_assessments: dict[str, Phase35ArgumentGateAssessment] = {}
    for argument_name, grounding_result in grounding.argument_results.items():
        policy_result = task_policy_result.argument_results[argument_name]
        argument_assessments[argument_name] = Phase35ArgumentGateAssessment(
            argument_name=argument_name,
            argument_value=grounding_result.argument_value,
            evidence_references=grounding_result.referenced_evidence_ids,
            grounding_status=grounding_result.status,
            grounding_confidence=grounding_result.grounding_confidence,
            supporting_evidence_ids=grounding_result.supporting_evidence_ids,
            contradicting_evidence_ids=grounding_result.contradicting_evidence_ids,
            unreferenced_conflicting_evidence_ids=(
                grounding_result.unreferenced_conflicting_evidence_ids
            ),
            evidence_relationship_satisfied=policy_result.relationship_satisfied,
            policy_disposition=policy_result.disposition,
            policy_rule_ids=policy_result.rule_ids,
        )

    reference_rules = [
        f"PHASE3_5_REFERENCE_{issue.code.value}"
        for issue in reference_validation.issues
    ]
    policy_rules = tuple(dict.fromkeys((*reference_rules, *task_policy_result.rule_ids)))
    if decision is GateDecision.ALLOW:
        message = "All critical arguments are grounded and satisfy task evidence policy."
    elif decision is GateDecision.ESCALATE:
        message = (
            "Grounding is ambiguous, conflicting, or insufficiently confident; "
            "review required."
        )
    else:
        message = "The proposed action has invalid or unsupported critical-argument provenance."
    latency_ms = (perf_counter() - decision_started) * 1000.0

    raw_arguments = _payload(payload.get("arguments"))
    raw_references = payload.get("argument_evidence_refs")
    if not isinstance(raw_references, Mapping):
        raw_references = {}

    return Phase35GateDecision(
        decision=decision,
        action=action,
        proposed_arguments=raw_arguments,
        argument_evidence_refs=dict(raw_references),
        reference_contract_valid=reference_validation.contract_valid,
        reference_issues=reference_validation.issues,
        grounding_statuses=grounding.statuses,
        argument_assessments=argument_assessments,
        static_effects=tuple(str(item) for item in definition["effects"]),
        reversibility=str(definition["reversibility"]),
        default_risk=str(definition["default_risk"]),
        policy_rules_triggered=policy_rules,
        user_message=message,
        policy_version=task_policy_result.policy_version,
        action_registry_version=task_policy_result.action_registry_version,
        frame_id=reference_validation.frame_id,
        thin_gate_latency_ms=latency_ms,
    )


class ThinTrustedGate:
    """Small reusable gate object for experiment runners."""

    def __init__(
        self,
        *,
        policy: Mapping[str, Any] | str | Path | None = None,
        action_registry: Mapping[str, Any] | str | Path | None = None,
    ) -> None:
        self._policy = policy
        self._action_registry = action_registry

    def evaluate(
        self,
        action_output: Any,
        registry: Any,
        *,
        user_intent: str = "",
        frame_id: str | None = None,
        reference_validation: EvidenceReferenceValidation | None = None,
        grounding: GroundingValidationResult | None = None,
        task_policy_result: TaskPolicyResult | None = None,
    ) -> Phase35GateDecision:
        return evaluate_thin_gate_phase3_5(
            action_output,
            registry,
            user_intent=user_intent,
            frame_id=frame_id,
            reference_validation=reference_validation,
            grounding=grounding,
            task_policy_result=task_policy_result,
            policy=self._policy,
            action_registry=self._action_registry,
        )


evaluate_grounded_thin_gate = evaluate_thin_gate_phase3_5
