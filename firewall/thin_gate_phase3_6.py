"""Uncertainty-aware, model-free Thin Trusted Gate for LensGuard Phase 3.6.

The gate always recomputes argument analysis against the supplied immutable
registry. It never executes an action, ranks visual trust, repairs a proposal,
or derives a decision from an action-model confidence report.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from phase3_5_constants import ACTION_REGISTRY_VERSION, Phase35ActionType
from phase3_6_constants import (
    ACTION_MODEL_CONTRACT_VERSION,
    EVIDENCE_REGISTRY_SCHEMA_VERSION,
    GATE_POLICY_VERSION,
    GROUNDING_SCHEMA_VERSION,
    UNCERTAINTY_SCHEMA_VERSION,
)
from phase3_6_schema import (
    AuthenticityStatus,
    EscalationReasonCode,
    StructuredEscalation,
    UncertaintyStatus,
)
from provenance.evidence_analysis_phase3_6 import (
    ArgumentEvidenceAnalysis,
    EvidenceAnalysisContext,
    EvidenceAnalysisResult,
    RelationshipFinding,
    analyze_evidence_uncertainty,
)
from provenance.evidence_registry_phase3_5 import EvidenceRegistry
from provenance.reference_validator_phase3_5 import (
    EvidenceReferenceValidation,
    ReferenceIssue,
    validate_evidence_references,
)

from .task_policy_phase3_5 import load_phase3_5_action_registry


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE3_6_POLICY_PATH = _PROJECT_ROOT / "config/policy_phase3_6.yaml"


class GateDecision(StrEnum):
    ALLOW = "ALLOW"
    ESCALATE = "ESCALATE"
    BLOCK = "BLOCK"


class GateReasonCode(StrEnum):
    INVALID_REFERENCE = "INVALID_REFERENCE"
    UNSUPPORTED_ARGUMENT = "UNSUPPORTED_ARGUMENT"
    SEMANTIC_RELATIONSHIP_MISMATCH = "SEMANTIC_RELATIONSHIP_MISMATCH"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    AUTHENTICITY_UNKNOWN = "AUTHENTICITY_UNKNOWN"
    SAFETY_INVARIANT = "SAFETY_INVARIANT"
    LOW_PERCEPTION_CONFIDENCE = "LOW_PERCEPTION_CONFIDENCE"
    ALLOW_SUPPORTED = "ALLOW_SUPPORTED"


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _ImmutableDict(dict[Any, Any]):
    """JSON-serializable dict whose mutation methods fail closed."""

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("gate audit mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __ior__(self, _other: Any) -> _ImmutableDict:
        self._immutable()
        return self


class Phase36ArgumentGateAssessment(_FrozenStrictModel):
    argument_name: str
    decision: GateDecision
    reason_code: GateReasonCode
    reason_codes_triggered: tuple[GateReasonCode, ...]
    analysis: ArgumentEvidenceAnalysis
    cross_argument_relationship_mismatch: bool = False
    safety_invariant_triggered: bool = False
    policy_rule_ids: tuple[str, ...] = ()

    @field_validator("analysis", mode="after")
    @classmethod
    def freeze_analysis(cls, value: ArgumentEvidenceAnalysis) -> ArgumentEvidenceAnalysis:
        return _freeze_argument_analysis(value)

    @model_validator(mode="after")
    def validate_binding(self) -> Phase36ArgumentGateAssessment:
        if self.argument_name != self.analysis.argument:
            raise ValueError("argument gate assessment is bound to another argument")
        expected_reasons = _argument_reason_codes(
            self.analysis,
            cross_argument_relationship_mismatch=(
                self.cross_argument_relationship_mismatch
            ),
            safety_invariant_triggered=self.safety_invariant_triggered,
        )
        if self.reason_codes_triggered != expected_reasons:
            raise ValueError("argument reasons contradict its evidence and policy findings")
        if self.reason_code is not expected_reasons[0]:
            raise ValueError("argument primary reason violates gate priority")
        expected_decision = _decision_for_reason(self.reason_code)
        if self.decision is not expected_decision:
            raise ValueError("argument decision contradicts its reason code")
        expected_rules = (f"PHASE3_6_{self.reason_code.value}",)
        if self.policy_rule_ids != expected_rules:
            raise ValueError("argument policy rule audit contradicts its primary reason")
        return self


class Phase36GateResult(_FrozenStrictModel):
    decision: GateDecision
    reason_code: GateReasonCode
    triggering_argument: str | None = None
    reason_codes_triggered: tuple[GateReasonCode, ...]
    action: str
    proposed_arguments: dict[str, Any] = Field(default_factory=dict)
    argument_evidence_refs: dict[str, Any] = Field(default_factory=dict)
    argument_target_object_ids: dict[str, str] = Field(default_factory=dict)
    reference_contract_valid: bool
    reference_issues: tuple[ReferenceIssue, ...] = ()
    reference_validation: EvidenceReferenceValidation
    uncertainty_statuses: dict[str, UncertaintyStatus] = Field(default_factory=dict)
    argument_assessments: dict[str, Phase36ArgumentGateAssessment] = Field(
        default_factory=dict
    )
    evidence_analysis: EvidenceAnalysisResult | None = None
    escalation: StructuredEscalation | None = None
    grounded_hazard_evidence_ids: tuple[str, ...] = ()
    static_effects: tuple[str, ...] = ()
    reversibility: str
    default_risk: str
    policy_rules_triggered: tuple[str, ...] = ()
    user_message: str = Field(min_length=1)
    evidence_schema_version: Literal[EVIDENCE_REGISTRY_SCHEMA_VERSION] = (
        EVIDENCE_REGISTRY_SCHEMA_VERSION
    )
    model_contract_version: Literal[ACTION_MODEL_CONTRACT_VERSION] = (
        ACTION_MODEL_CONTRACT_VERSION
    )
    grounding_schema_version: Literal[GROUNDING_SCHEMA_VERSION] = GROUNDING_SCHEMA_VERSION
    uncertainty_schema_version: Literal[UNCERTAINTY_SCHEMA_VERSION] = (
        UNCERTAINTY_SCHEMA_VERSION
    )
    policy_version: Literal[GATE_POLICY_VERSION] = GATE_POLICY_VERSION
    action_registry_version: Literal[ACTION_REGISTRY_VERSION] = ACTION_REGISTRY_VERSION
    frame_id: str | None = None
    thin_gate_latency_ms: float = Field(ge=0.0)
    dry_run: Literal[True] = True
    auto_corrected: Literal[False] = False
    model_free: Literal[True] = True

    @field_validator(
        "proposed_arguments",
        "argument_evidence_refs",
        "argument_target_object_ids",
        "uncertainty_statuses",
        "argument_assessments",
        mode="after",
    )
    @classmethod
    def freeze_audit_mapping(cls, value: dict[Any, Any]) -> dict[Any, Any]:
        return _deep_freeze(value)

    @field_validator("reference_validation", mode="after")
    @classmethod
    def freeze_reference_validation(
        cls, value: EvidenceReferenceValidation
    ) -> EvidenceReferenceValidation:
        return _freeze_reference_validation(value)

    @field_validator("evidence_analysis", mode="after")
    @classmethod
    def freeze_evidence_analysis(
        cls, value: EvidenceAnalysisResult | None
    ) -> EvidenceAnalysisResult | None:
        return None if value is None else _freeze_evidence_analysis(value)

    @model_validator(mode="after")
    def validate_decision_contract(self) -> Phase36GateResult:
        if any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in self.argument_target_object_ids.values()
        ):
            raise ValueError("target-object identifiers must be nonblank canonical strings")
        if self.reference_contract_valid != self.reference_validation.contract_valid:
            raise ValueError("reference validity fields disagree")
        if self.reference_issues != self.reference_validation.issues:
            raise ValueError("reference issue fields disagree")
        if self.frame_id != self.reference_validation.frame_id:
            raise ValueError("gate frame differs from reference-validation frame")
        if self.evidence_analysis is None:
            if self.reference_contract_valid:
                raise ValueError("a valid reference contract requires fresh evidence analysis")
            if self.argument_assessments or self.uncertainty_statuses:
                raise ValueError("invalid contracts cannot carry uncomputed analysis results")
            expected_reasons = (GateReasonCode.INVALID_REFERENCE,)
            expected_triggering_argument = None
        else:
            if not self.reference_contract_valid:
                raise ValueError("invalid contracts must not carry evidence analysis")
            if self.evidence_analysis.reference_validation != self.reference_validation:
                raise ValueError("evidence analysis used another reference validation")
            if self.evidence_analysis.action != self.action:
                raise ValueError("evidence analysis used another action")
            if set(self.argument_assessments) != set(
                self.evidence_analysis.argument_results
            ):
                raise ValueError("argument assessments must preserve every analysis result")
            expected_statuses = self.evidence_analysis.statuses
            if self.uncertainty_statuses != expected_statuses:
                raise ValueError("uncertainty statuses do not match evidence analysis")
            for argument, assessment in self.argument_assessments.items():
                if assessment.analysis != self.evidence_analysis.argument_results[argument]:
                    raise ValueError("argument assessment does not preserve fresh analysis")
            expected_arguments = self.evidence_analysis.argument_results
            if set(self.proposed_arguments) != set(expected_arguments):
                raise ValueError("proposed arguments do not match fresh evidence analysis")
            for argument, result in expected_arguments.items():
                proposed = self.proposed_arguments[argument]
                if type(proposed) is not type(result.argument_value) or (
                    proposed != result.argument_value
                ):
                    raise ValueError("proposed argument value differs from fresh analysis")
            if set(self.argument_evidence_refs) != set(expected_arguments):
                raise ValueError("evidence-reference map does not match fresh analysis")
            for argument, result in expected_arguments.items():
                raw_references = self.argument_evidence_refs[argument]
                if not isinstance(raw_references, (list, tuple)) or tuple(
                    raw_references
                ) != result.referenced_evidence_ids:
                    raise ValueError("evidence references differ from fresh analysis")
            unknown_targets = set(self.argument_target_object_ids) - set(expected_arguments)
            if unknown_targets:
                raise ValueError("target-object map names unknown action arguments")

            expected_hazards = _grounded_hazard_ids(
                self.action,
                self.evidence_analysis,
            )
            if self.grounded_hazard_evidence_ids != expected_hazards:
                raise ValueError("grounded hazard evidence differs from fresh analysis")
            cross_mismatch_argument = _cross_argument_mismatch(
                self.action,
                self.argument_target_object_ids,
            )
            for argument, assessment in self.argument_assessments.items():
                if assessment.cross_argument_relationship_mismatch is not (
                    argument == cross_mismatch_argument
                ):
                    raise ValueError("cross-argument relationship finding is inconsistent")
                if assessment.safety_invariant_triggered is not (
                    argument == "safe_to_proceed" and bool(expected_hazards)
                ):
                    raise ValueError("argument safety finding is inconsistent")

            triggering_by_reason: dict[GateReasonCode, str] = {}
            for argument in self.reference_validation.expected_arguments:
                assessment = self.argument_assessments[argument]
                for reason in assessment.reason_codes_triggered:
                    if reason is not GateReasonCode.ALLOW_SUPPORTED:
                        triggering_by_reason.setdefault(reason, argument)
            expected_reasons = tuple(
                reason for reason in _REASON_PRIORITY if reason in triggering_by_reason
            )
            if expected_reasons:
                expected_triggering_argument = triggering_by_reason[expected_reasons[0]]
            else:
                expected_reasons = (GateReasonCode.ALLOW_SUPPORTED,)
                expected_triggering_argument = None

        if self.reason_codes_triggered != expected_reasons:
            raise ValueError("aggregate gate reasons contradict fresh argument findings")
        expected_primary = expected_reasons[0]
        if self.reason_code is not expected_primary:
            raise ValueError("primary gate reason violates deterministic priority")
        if self.triggering_argument != expected_triggering_argument:
            raise ValueError("triggering argument violates deterministic priority")
        if self.decision is not _decision_for_reason(expected_primary):
            raise ValueError("gate decision contradicts fresh analysis")

        expected_rules = tuple(
            f"PHASE3_6_{reason.value}" for reason in expected_reasons
        )
        if GateReasonCode.SAFETY_INVARIANT in expected_reasons:
            expected_rules = (
                *expected_rules,
                "PHASE3_6_SAFETY_GROUNDED_HAZARD_VETO",
            )
        if self.policy_rules_triggered != expected_rules:
            raise ValueError("policy rule audit does not match triggered reasons")
        if self.decision is GateDecision.ESCALATE:
            if self.escalation is None:
                raise ValueError("ESCALATE requires a structured escalation")
            if self.triggering_argument != self.escalation.argument:
                raise ValueError("escalation argument differs from triggering argument")
            if self.action != self.escalation.action.value:
                raise ValueError("escalation action differs from gate action")
            if self.reason_code.value != self.escalation.reason_code.value:
                raise ValueError("escalation reason differs from gate reason")
        elif self.escalation is not None:
            raise ValueError("only ESCALATE may carry a structured escalation")
        if self.decision is GateDecision.ALLOW:
            if any(
                assessment.decision is not GateDecision.ALLOW
                for assessment in self.argument_assessments.values()
            ):
                raise ValueError("ALLOW cannot contain a non-allow argument assessment")
        if self.escalation is not None and self.triggering_argument is not None:
            expected_candidates = self.argument_assessments[
                self.triggering_argument
            ].analysis.uncertainty.candidate_values
            if self.escalation.candidate_values != expected_candidates:
                raise ValueError("escalation candidates must preserve analysis candidates")
        return self


_REASON_PRIORITY = (
    GateReasonCode.INVALID_REFERENCE,
    GateReasonCode.UNSUPPORTED_ARGUMENT,
    GateReasonCode.SEMANTIC_RELATIONSHIP_MISMATCH,
    GateReasonCode.CONFLICTING_EVIDENCE,
    GateReasonCode.LOW_PERCEPTION_CONFIDENCE,
    GateReasonCode.INSUFFICIENT_EVIDENCE,
    GateReasonCode.AUTHENTICITY_UNKNOWN,
    GateReasonCode.SAFETY_INVARIANT,
    GateReasonCode.ALLOW_SUPPORTED,
)

_EXPECTED_STATUS_DISPOSITIONS = {
    UncertaintyStatus.SUPPORTED.value: "PASS",
    UncertaintyStatus.UNSUPPORTED.value: "BLOCK",
    UncertaintyStatus.AMBIGUOUS.value: "ESCALATE",
    UncertaintyStatus.CONFLICTING.value: "ESCALATE",
    UncertaintyStatus.INSUFFICIENT_EVIDENCE.value: "ESCALATE",
    UncertaintyStatus.AUTHENTICITY_UNKNOWN.value: "ESCALATE",
    UncertaintyStatus.MISSING.value: "BLOCK",
    UncertaintyStatus.INVALID_REFERENCE.value: "BLOCK",
}


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _ImmutableDict(
            (key, _deep_freeze(item)) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def _freeze_reference_validation(
    value: EvidenceReferenceValidation,
) -> EvidenceReferenceValidation:
    return value.model_copy(
        update={"argument_results": _deep_freeze(value.argument_results)}
    )


def _freeze_argument_analysis(
    value: ArgumentEvidenceAnalysis,
) -> ArgumentEvidenceAnalysis:
    return value.model_copy(
        update={
            "relationship_assessments": _deep_freeze(
                value.relationship_assessments
            )
        }
    )


def _freeze_evidence_analysis(value: EvidenceAnalysisResult) -> EvidenceAnalysisResult:
    return value.model_copy(
        update={
            "argument_results": _deep_freeze(
                {
                    argument: _freeze_argument_analysis(result)
                    for argument, result in value.argument_results.items()
                }
            ),
            "reference_validation": _freeze_reference_validation(
                value.reference_validation
            ),
        }
    )


def _decision_for_reason(reason: GateReasonCode) -> GateDecision:
    if reason in {
        GateReasonCode.INVALID_REFERENCE,
        GateReasonCode.UNSUPPORTED_ARGUMENT,
        GateReasonCode.SEMANTIC_RELATIONSHIP_MISMATCH,
        GateReasonCode.SAFETY_INVARIANT,
    }:
        return GateDecision.BLOCK
    if reason is GateReasonCode.ALLOW_SUPPORTED:
        return GateDecision.ALLOW
    return GateDecision.ESCALATE


def _load_yaml_mapping(
    value: Mapping[str, Any] | str | Path | None,
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload: Any = dict(value)
    else:
        path = DEFAULT_PHASE3_6_POLICY_PATH if value is None else Path(value)
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ValueError(f"could not read Phase 3.6 gate policy: {path}") from error
        except yaml.YAMLError as error:
            raise ValueError(f"invalid Phase 3.6 gate policy YAML: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("Phase 3.6 gate policy must be a mapping")
    return deepcopy(payload)


def load_phase3_6_gate_policy(
    value: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Load the exact versioned policy; overrides cannot weaken its semantics."""

    policy = _load_yaml_mapping(value)
    required_keys = {
        "policy_version",
        "grounding_schema_version",
        "uncertainty_schema_version",
        "model_contract_version",
        "description",
        "reason_priority",
        "uncertainty_status_dispositions",
        "perception_confidence",
        "safety_hazard_veto",
    }
    if set(policy) != required_keys:
        raise ValueError("Phase 3.6 gate policy has unexpected or missing fields")
    expected_versions = {
        "policy_version": GATE_POLICY_VERSION,
        "grounding_schema_version": GROUNDING_SCHEMA_VERSION,
        "uncertainty_schema_version": UNCERTAINTY_SCHEMA_VERSION,
        "model_contract_version": ACTION_MODEL_CONTRACT_VERSION,
    }
    for key, expected in expected_versions.items():
        if policy.get(key) != expected:
            raise ValueError(f"Phase 3.6 gate policy {key} must be exactly {expected!r}")
    if not isinstance(policy.get("description"), str) or not policy["description"].strip():
        raise ValueError("Phase 3.6 gate policy requires a description")
    expected_priority = [reason.value for reason in _REASON_PRIORITY]
    if policy.get("reason_priority") != expected_priority:
        raise ValueError("Phase 3.6 gate reason priority cannot be changed")
    if policy.get("uncertainty_status_dispositions") != _EXPECTED_STATUS_DISPOSITIONS:
        raise ValueError("Phase 3.6 uncertainty dispositions cannot be changed")
    confidence = policy.get("perception_confidence")
    if confidence != {
        "numeric_thresholds": None,
        "calibrated_categorical_input_available": False,
        "uncalibrated_numeric_values_are_diagnostic_only": True,
    }:
        raise ValueError("Phase 3.6 must not invent a perception-confidence threshold")
    expected_veto = {
        "action": "SAFETY_ADVICE",
        "positive_argument": "safe_to_proceed",
        "hazard_argument": "hazard",
        "normalized_positive_value": "true",
        "normalized_no_hazard_value": "NONE",
        "disposition": "BLOCK",
        "rule_id": "PHASE3_6_SAFETY_GROUNDED_HAZARD_VETO",
    }
    veto = policy.get("safety_hazard_veto")
    if not isinstance(veto, Mapping):
        raise ValueError("Phase 3.6 gate policy requires safety_hazard_veto")
    if {key: veto.get(key) for key in expected_veto} != expected_veto:
        raise ValueError("Phase 3.6 safety hazard veto cannot be weakened")
    if set(veto) != {*expected_veto, "note"}:
        raise ValueError("Phase 3.6 safety hazard veto has unexpected fields")
    if not isinstance(veto.get("note"), str) or not str(veto["note"]).strip():
        raise ValueError("Phase 3.6 safety hazard veto requires a note")
    return policy


def _payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        rendered = dump(mode="python", exclude_none=True)
        if isinstance(rendered, Mapping):
            return dict(rendered)
    return {}


def _action_string(payload: Mapping[str, Any]) -> str:
    value = payload.get("action", "")
    return str(getattr(value, "value", value)).strip().upper()


def _validated_registry(registry: Any) -> EvidenceRegistry:
    if isinstance(registry, EvidenceRegistry):
        return registry
    payload = _payload(registry)
    frame_id = payload.get("frame_id")
    items = payload.get("items")
    schema_version = payload.get("schema_version")
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("serialized registry requires a nonblank frame_id")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise ValueError("serialized registry requires an items array")
    if not isinstance(schema_version, str):
        raise ValueError("serialized registry requires its schema_version")
    return EvidenceRegistry(frame_id, items, schema_version=schema_version)


def _selected_relationships(
    result: ArgumentEvidenceAnalysis,
) -> tuple[Any, ...]:
    return tuple(
        result.relationship_assessments[evidence_id]
        for evidence_id in result.referenced_evidence_ids
        if evidence_id in result.relationship_assessments
    )


def _has_semantic_mismatch(result: ArgumentEvidenceAnalysis) -> bool:
    for relationship in _selected_relationships(result):
        if RelationshipFinding.MISMATCH in {
            relationship.content_type_relationship,
            relationship.target_object_relationship,
            relationship.semantic_role_relationship,
            relationship.argument_relationship,
        }:
            return True
    return False


def _has_clear_value_failure(result: ArgumentEvidenceAnalysis) -> bool:
    if result.status is not UncertaintyStatus.UNSUPPORTED:
        return False
    if result.normalized_argument_value is None:
        return True
    selected = _selected_relationships(result)
    value_findings = {relationship.value_relationship for relationship in selected}
    if RelationshipFinding.MATCH in value_findings:
        return False
    if RelationshipFinding.MISMATCH in value_findings:
        return True
    return not _has_semantic_mismatch(result)


def _has_unresolved_alternate_candidate(result: ArgumentEvidenceAnalysis) -> bool:
    if any(
        evidence_id.startswith("USER:")
        for evidence_id in result.supporting_evidence_ids
    ):
        return False
    selected_ids = set(result.referenced_evidence_ids)
    established_values = set(result.conflict_set.distinct_values)
    for evidence_id, relationship in result.relationship_assessments.items():
        if evidence_id in selected_ids or not relationship.candidate_values:
            continue
        non_value_findings = {
            relationship.content_type_relationship,
            relationship.target_object_relationship,
            relationship.semantic_role_relationship,
            relationship.argument_relationship,
        }
        if RelationshipFinding.MISMATCH in non_value_findings:
            continue
        if RelationshipFinding.NOT_ASSESSED not in non_value_findings:
            continue
        if any(
            candidate not in established_values
            for candidate in relationship.candidate_values
        ):
            return True
    return False


def _argument_reason_codes(
    result: ArgumentEvidenceAnalysis,
    *,
    cross_argument_relationship_mismatch: bool = False,
    safety_invariant_triggered: bool = False,
) -> tuple[GateReasonCode, ...]:
    findings: set[GateReasonCode] = set()
    if result.status in {UncertaintyStatus.MISSING, UncertaintyStatus.INVALID_REFERENCE}:
        findings.add(GateReasonCode.INVALID_REFERENCE)
    if _has_clear_value_failure(result):
        findings.add(GateReasonCode.UNSUPPORTED_ARGUMENT)
    if _has_semantic_mismatch(result) or cross_argument_relationship_mismatch:
        findings.add(GateReasonCode.SEMANTIC_RELATIONSHIP_MISMATCH)
    if result.status is UncertaintyStatus.UNSUPPORTED and not findings:
        findings.add(GateReasonCode.UNSUPPORTED_ARGUMENT)
    if result.conflict_set.has_conflict:
        findings.add(GateReasonCode.CONFLICTING_EVIDENCE)
    if result.status in {
        UncertaintyStatus.INSUFFICIENT_EVIDENCE,
        UncertaintyStatus.AMBIGUOUS,
    } or _has_unresolved_alternate_candidate(result):
        findings.add(GateReasonCode.INSUFFICIENT_EVIDENCE)
    if result.uncertainty.authenticity_status is AuthenticityStatus.UNKNOWN:
        findings.add(GateReasonCode.AUTHENTICITY_UNKNOWN)
    if safety_invariant_triggered:
        findings.add(GateReasonCode.SAFETY_INVARIANT)
    reasons = tuple(reason for reason in _REASON_PRIORITY if reason in findings)
    return reasons or (GateReasonCode.ALLOW_SUPPORTED,)


def _argument_assessment(
    result: ArgumentEvidenceAnalysis,
    *,
    cross_argument_relationship_mismatch: bool = False,
    safety_invariant_triggered: bool = False,
) -> Phase36ArgumentGateAssessment:
    reasons = _argument_reason_codes(
        result,
        cross_argument_relationship_mismatch=cross_argument_relationship_mismatch,
        safety_invariant_triggered=safety_invariant_triggered,
    )
    reason = reasons[0]
    return Phase36ArgumentGateAssessment(
        argument_name=result.argument,
        decision=_decision_for_reason(reason),
        reason_code=reason,
        reason_codes_triggered=reasons,
        analysis=result,
        cross_argument_relationship_mismatch=cross_argument_relationship_mismatch,
        safety_invariant_triggered=safety_invariant_triggered,
        policy_rule_ids=(f"PHASE3_6_{reason.value}",),
    )


def _cross_argument_mismatch(
    action: str,
    target_object_ids: Mapping[str, str],
) -> str | None:
    if action != "RESTAURANT_RESERVATION":
        return None
    restaurant = target_object_ids.get("restaurant")
    number = target_object_ids.get("target_number")
    if restaurant is not None and number is not None and restaurant != number:
        return "target_number"
    return None


def _grounded_hazard_ids(
    action: str,
    analysis: EvidenceAnalysisResult,
) -> tuple[str, ...]:
    if action != "SAFETY_ADVICE":
        return ()
    positive = analysis.argument_results.get("safe_to_proceed")
    hazard = analysis.argument_results.get("hazard")
    if positive is None or hazard is None:
        return ()
    if positive.normalized_argument_value != "true":
        return ()
    found = []
    for evidence_id, relationship in hazard.relationship_assessments.items():
        if evidence_id.startswith("USER:") or not relationship.plausible_candidate:
            continue
        if any(candidate != "NONE" for candidate in relationship.candidate_values):
            found.append(evidence_id)
    return tuple(sorted(found))


def _escalation_message(reason: GateReasonCode, argument: str) -> str:
    if reason is GateReasonCode.CONFLICTING_EVIDENCE:
        return (
            f"Multiple plausible values were found for {argument}. "
            "Please confirm which one to use."
        )
    if reason is GateReasonCode.AUTHENTICITY_UNKNOWN:
        return (
            f"I found a value for {argument}, but I cannot verify from the current view "
            "whether the physical information is original. Do you want to use it?"
        )
    if reason is GateReasonCode.LOW_PERCEPTION_CONFIDENCE:
        return f"Perception quality for {argument} is unresolved. Please inspect or confirm it."
    return (
        f"There is not enough evidence to safely establish {argument}. "
        "Please inspect or confirm it."
    )


def _structured_escalation(
    *,
    action: str,
    argument: str,
    reason: GateReasonCode,
    result: ArgumentEvidenceAnalysis,
    grounded_hazard_evidence_ids: tuple[str, ...] = (),
) -> StructuredEscalation:
    message = _escalation_message(reason, argument)
    if grounded_hazard_evidence_ids:
        message += " Grounded hazard evidence is also present; do not proceed automatically."
    return StructuredEscalation(
        reason_code=EscalationReasonCode(reason.value),
        action=Phase35ActionType(action),
        argument=argument,
        candidate_values=result.uncertainty.candidate_values,
        message=message,
    )


def _invalid_contract_result(
    *,
    action: str,
    payload: Mapping[str, Any],
    references: EvidenceReferenceValidation,
    definition: Mapping[str, Any] | None,
    latency_ms: float,
) -> Phase36GateResult:
    raw_references = payload.get("argument_evidence_refs")
    return Phase36GateResult(
        decision=GateDecision.BLOCK,
        reason_code=GateReasonCode.INVALID_REFERENCE,
        reason_codes_triggered=(GateReasonCode.INVALID_REFERENCE,),
        action=action,
        proposed_arguments=_payload(payload.get("arguments")),
        argument_evidence_refs=(
            dict(raw_references) if isinstance(raw_references, Mapping) else {}
        ),
        reference_contract_valid=False,
        reference_issues=references.issues,
        reference_validation=references,
        static_effects=tuple(str(item) for item in (definition or {}).get("effects", ())),
        reversibility=str((definition or {}).get("reversibility", "unknown")),
        default_risk=str((definition or {}).get("default_risk", "unknown")),
        policy_rules_triggered=("PHASE3_6_INVALID_REFERENCE",),
        user_message="The action has an invalid or incomplete evidence-reference contract.",
        frame_id=references.frame_id,
        thin_gate_latency_ms=latency_ms,
    )


def evaluate_thin_gate_phase3_6(
    action_output: Any,
    registry: Any,
    *,
    user_intent: str = "",
    frame_id: str | None = None,
    argument_target_object_ids: Mapping[str, str] | None = None,
    evidence_contexts: Mapping[
        str, EvidenceAnalysisContext | Mapping[str, Any]
    ]
    | None = None,
    evidence_analysis: EvidenceAnalysisResult | None = None,
    policy: Mapping[str, Any] | str | Path | None = None,
    action_registry: Mapping[str, Any] | str | Path | None = None,
) -> Phase36GateResult:
    """Return a dry-run ALLOW, ESCALATE, or BLOCK decision in fixed priority."""

    started = perf_counter()
    if not isinstance(user_intent, str):
        raise TypeError("user_intent must be a string")
    policy_config = load_phase3_6_gate_policy(policy)
    action_registry_config = load_phase3_5_action_registry(action_registry)
    canonical_registry = _validated_registry(registry)
    payload = _payload(action_output)
    action = _action_string(payload)
    definition = action_registry_config["actions"].get(action)
    references = validate_evidence_references(
        action_output,
        canonical_registry,
        frame_id=frame_id,
    )
    if not references.contract_valid:
        if evidence_analysis is not None:
            raise ValueError("precomputed analysis is invalid when the reference contract fails")
        latency_ms = (perf_counter() - started) * 1000.0
        return _invalid_contract_result(
            action=action,
            payload=payload,
            references=references,
            definition=definition if isinstance(definition, Mapping) else None,
            latency_ms=latency_ms,
        )

    fresh_analysis = analyze_evidence_uncertainty(
        action_output,
        canonical_registry,
        argument_target_object_ids=argument_target_object_ids,
        evidence_contexts=evidence_contexts,
        reference_validation=references,
        frame_id=frame_id,
    )
    if evidence_analysis is not None and evidence_analysis != fresh_analysis:
        raise ValueError("precomputed analysis does not match fresh Phase 3.6 analysis")
    if not isinstance(definition, Mapping):
        raise ValueError(f"action {action!r} is absent from the unchanged action registry")

    target_object_ids = {
        key: value.strip()
        for key, value in (argument_target_object_ids or {}).items()
    }
    cross_mismatch_argument = _cross_argument_mismatch(action, target_object_ids)
    grounded_hazards = _grounded_hazard_ids(action, fresh_analysis)
    assessments = {
        argument: _argument_assessment(
            result,
            cross_argument_relationship_mismatch=(
                argument == cross_mismatch_argument
            ),
            safety_invariant_triggered=(
                argument == "safe_to_proceed" and bool(grounded_hazards)
            ),
        )
        for argument, result in fresh_analysis.argument_results.items()
    }
    triggering_by_reason: dict[GateReasonCode, str] = {}
    for argument in fresh_analysis.reference_validation.expected_arguments:
        for reason in assessments[argument].reason_codes_triggered:
            if reason is not GateReasonCode.ALLOW_SUPPORTED:
                triggering_by_reason.setdefault(reason, argument)

    triggered = tuple(
        reason for reason in _REASON_PRIORITY if reason in triggering_by_reason
    )
    if triggered:
        primary = triggered[0]
        triggering_argument = triggering_by_reason[primary]
        decision = _decision_for_reason(primary)
    else:
        primary = GateReasonCode.ALLOW_SUPPORTED
        triggering_argument = None
        decision = GateDecision.ALLOW
        triggered = (primary,)

    escalation = None
    if decision is GateDecision.ESCALATE:
        assert triggering_argument is not None
        escalation = _structured_escalation(
            action=action,
            argument=triggering_argument,
            reason=primary,
            result=fresh_analysis.argument_results[triggering_argument],
            grounded_hazard_evidence_ids=grounded_hazards,
        )

    raw_references = payload.get("argument_evidence_refs")
    rules = [f"PHASE3_6_{reason.value}" for reason in triggered]
    if grounded_hazards:
        rules.append(str(policy_config["safety_hazard_veto"]["rule_id"]))
    if decision is GateDecision.ALLOW:
        message = "All critical arguments satisfy Phase 3.6 grounding and uncertainty policy."
    elif decision is GateDecision.ESCALATE:
        assert escalation is not None
        message = escalation.message
    elif primary is GateReasonCode.SAFETY_INVARIANT:
        message = "Positive safety advice is blocked because grounded hazard evidence exists."
    else:
        message = "The proposed action has invalid or unsupported argument evidence."
    latency_ms = (perf_counter() - started) * 1000.0
    return Phase36GateResult(
        decision=decision,
        reason_code=primary,
        triggering_argument=triggering_argument,
        reason_codes_triggered=triggered,
        action=action,
        proposed_arguments=_payload(payload.get("arguments")),
        argument_evidence_refs=(
            dict(raw_references) if isinstance(raw_references, Mapping) else {}
        ),
        argument_target_object_ids=target_object_ids,
        reference_contract_valid=True,
        reference_issues=references.issues,
        reference_validation=references,
        uncertainty_statuses=fresh_analysis.statuses,
        argument_assessments=assessments,
        evidence_analysis=fresh_analysis,
        escalation=escalation,
        grounded_hazard_evidence_ids=grounded_hazards,
        static_effects=tuple(str(item) for item in definition["effects"]),
        reversibility=str(definition["reversibility"]),
        default_risk=str(definition["default_risk"]),
        policy_rules_triggered=tuple(dict.fromkeys(rules)),
        user_message=message,
        frame_id=references.frame_id,
        thin_gate_latency_ms=latency_ms,
    )


class ThinTrustedGatePhase36:
    """Reusable Phase 3.6 gate with immutable configuration references."""

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
        **kwargs: Any,
    ) -> Phase36GateResult:
        return evaluate_thin_gate_phase3_6(
            action_output,
            registry,
            policy=self._policy,
            action_registry=self._action_registry,
            **kwargs,
        )


evaluate_uncertainty_aware_gate = evaluate_thin_gate_phase3_6
ThinTrustedGate = ThinTrustedGatePhase36
