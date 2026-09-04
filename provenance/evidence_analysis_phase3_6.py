"""Deterministic conflict and semantic-relationship analysis for Phase 3.6.

This layer does not make a gate decision.  It normalizes candidates, keeps the
four relationship findings separately inspectable, constructs conflict sets,
and represents caller-supplied authenticity uncertainty without inferring
authenticity from visual alignment, confidence, source labels, or attack labels.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from phase3_5_constants import CRITICAL_ARGUMENTS
from phase3_6_constants import GROUNDING_SCHEMA_VERSION
from phase3_6_schema import (
    ArgumentUncertaintyAssessment,
    AuthenticityStatus,
    EvidenceConfidenceDimensions,
    UncertaintyStatus,
)
from provenance.evidence_registry_phase3_5 import EvidenceRegistry
from provenance.grounding_validator_phase3_5 import (
    candidate_values_for_evidence,
    normalize_grounding_value,
)
from provenance.reference_validator_phase3_5 import (
    ArgumentReferenceStatus,
    EvidenceReferenceValidation,
    evidence_field,
    registry_get,
    registry_items,
    validate_evidence_references,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELATIONSHIP_CONFIG = _PROJECT_ROOT / "config/evidence_relationships_phase3_6.yaml"
_EXPECTED_ARGUMENTS = {
    action.value: tuple(arguments) for action, arguments in CRITICAL_ARGUMENTS.items()
}


class RelationshipFinding(StrEnum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    NOT_ASSESSED = "NOT_ASSESSED"
    NOT_REQUIRED = "NOT_REQUIRED"


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceAnalysisContext(_FrozenStrictModel):
    """Non-model sidecar facts available to deterministic analysis.

    Target association is compared by exact identifier.  Authenticity is an
    explicit upstream assessment and is never derived here from physical attack
    mode, control class, source label, bounding box, or confidence.
    """

    evidence_id: str = Field(min_length=1)
    associated_target_object_id: str | None = None
    authenticity_status: AuthenticityStatus = AuthenticityStatus.NOT_ASSESSED
    authenticity_basis: str | None = None

    @field_validator("evidence_id", "associated_target_object_id", "authenticity_basis")
    @classmethod
    def strip_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("analysis context strings must not be blank")
        return cleaned

    @model_validator(mode="after")
    def validate_authenticity(self) -> EvidenceAnalysisContext:
        is_user = self.evidence_id.startswith("USER:")
        if is_user:
            if self.associated_target_object_id is not None:
                raise ValueError("USER evidence cannot declare a camera target object")
            if self.authenticity_status is not AuthenticityStatus.NOT_REQUIRED:
                raise ValueError("USER evidence context must use authenticity NOT_REQUIRED")
        elif self.authenticity_status is AuthenticityStatus.NOT_REQUIRED:
            raise ValueError("NOT_REQUIRED authenticity is reserved for USER evidence")
        if (
            self.authenticity_status is AuthenticityStatus.ESTABLISHED
            and self.authenticity_basis is None
        ):
            raise ValueError("ESTABLISHED authenticity requires an auditable basis")
        if (
            self.authenticity_status is not AuthenticityStatus.ESTABLISHED
            and self.authenticity_basis is not None
        ):
            raise ValueError("authenticity_basis is valid only for ESTABLISHED status")
        return self


class EvidenceRelationshipAssessment(_FrozenStrictModel):
    evidence_id: str
    candidate_values: tuple[str, ...] = ()
    content_type_relationship: RelationshipFinding
    value_relationship: RelationshipFinding
    target_object_relationship: RelationshipFinding
    semantic_role_relationship: RelationshipFinding
    argument_relationship: RelationshipFinding
    task_context_satisfied: bool
    supports_proposed_argument: bool
    plausible_candidate: bool
    reasons: tuple[str, ...] = ()

    @field_validator("candidate_values")
    @classmethod
    def validate_candidates(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value for value in values) or len(values) != len(set(values)):
            raise ValueError("candidate_values must be unique nonblank normalized values")
        return values

    @model_validator(mode="after")
    def validate_derived_findings(self) -> EvidenceRelationshipAssessment:
        expected_context = (
            self.content_type_relationship is RelationshipFinding.MATCH
            and self.semantic_role_relationship is RelationshipFinding.MATCH
            and self.argument_relationship is RelationshipFinding.MATCH
            and self.target_object_relationship
            in {RelationshipFinding.MATCH, RelationshipFinding.NOT_REQUIRED}
        )
        if self.task_context_satisfied is not expected_context:
            raise ValueError("task_context_satisfied contradicts relationship findings")
        expected_support = (
            expected_context and self.value_relationship is RelationshipFinding.MATCH
        )
        if self.supports_proposed_argument is not expected_support:
            raise ValueError("supports_proposed_argument contradicts relationship findings")
        if self.plausible_candidate is not (bool(self.candidate_values) and expected_context):
            raise ValueError("plausible_candidate contradicts relationship findings")
        return self


class ConflictCandidate(_FrozenStrictModel):
    normalized_value: str
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_candidate(self) -> ConflictCandidate:
        if not self.normalized_value:
            raise ValueError("normalized conflict value must not be blank")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("conflict evidence IDs must be unique")
        if self.evidence_ids != tuple(sorted(self.evidence_ids)):
            raise ValueError("conflict evidence IDs must use canonical order")
        return self


class ArgumentConflictSet(_FrozenStrictModel):
    argument: str
    candidates: tuple[ConflictCandidate, ...] = ()

    @model_validator(mode="after")
    def validate_conflict_set(self) -> ArgumentConflictSet:
        if not self.argument:
            raise ValueError("conflict argument must not be blank")
        values = tuple(candidate.normalized_value for candidate in self.candidates)
        if len(values) != len(set(values)):
            raise ValueError("conflict values must be unique")
        if values != tuple(sorted(values)):
            raise ValueError("conflict values must use canonical order")
        return self

    @property
    def distinct_values(self) -> tuple[str, ...]:
        return tuple(candidate.normalized_value for candidate in self.candidates)

    @property
    def has_conflict(self) -> bool:
        return len(self.candidates) > 1


class ArgumentEvidenceAnalysis(_FrozenStrictModel):
    argument: str
    argument_value: Any = None
    normalized_argument_value: str | None = None
    referenced_evidence_ids: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    relationship_assessments: dict[str, EvidenceRelationshipAssessment] = Field(
        default_factory=dict
    )
    conflict_set: ArgumentConflictSet
    uncertainty: ArgumentUncertaintyAssessment

    @property
    def status(self) -> UncertaintyStatus:
        return self.uncertainty.status

    @model_validator(mode="after")
    def validate_internal_bindings(self) -> ArgumentEvidenceAnalysis:
        if self.conflict_set.argument != self.argument:
            raise ValueError("conflict set belongs to a different argument")
        if self.uncertainty.argument != self.argument:
            raise ValueError("uncertainty belongs to a different argument")
        if self.uncertainty.candidate_values != self.conflict_set.distinct_values:
            raise ValueError("uncertainty candidates must equal the canonical conflict set")
        if (
            self.status is UncertaintyStatus.CONFLICTING
            and not self.conflict_set.has_conflict
        ):
            raise ValueError("CONFLICTING status requires a multi-value conflict set")
        for key, assessment in self.relationship_assessments.items():
            if key != assessment.evidence_id:
                raise ValueError("relationship map key must equal embedded evidence_id")
        if not set(self.supporting_evidence_ids) <= set(self.referenced_evidence_ids):
            raise ValueError("supporting evidence must be selected by the proposal")
        for evidence_id in self.supporting_evidence_ids:
            assessment = self.relationship_assessments.get(evidence_id)
            if assessment is None or not assessment.supports_proposed_argument:
                raise ValueError("supporting evidence must have a supporting relationship")
        return self


class EvidenceAnalysisResult(_FrozenStrictModel):
    grounding_schema_version: str = GROUNDING_SCHEMA_VERSION
    action: str
    frame_id: str | None
    argument_results: dict[str, ArgumentEvidenceAnalysis] = Field(default_factory=dict)
    reference_validation: EvidenceReferenceValidation

    @model_validator(mode="after")
    def validate_version_and_arguments(self) -> EvidenceAnalysisResult:
        if self.grounding_schema_version != GROUNDING_SCHEMA_VERSION:
            raise ValueError(
                f"grounding_schema_version must be exactly {GROUNDING_SCHEMA_VERSION!r}"
            )
        if set(self.argument_results) != set(self.reference_validation.expected_arguments):
            raise ValueError("analysis arguments must match reference-validation arguments")
        if self.action != self.reference_validation.action:
            raise ValueError("analysis action must match reference validation")
        if self.frame_id != self.reference_validation.frame_id:
            raise ValueError("analysis frame must match reference validation")
        for key, result in self.argument_results.items():
            if key != result.argument:
                raise ValueError("argument result key must match embedded argument")
        return self

    @property
    def statuses(self) -> dict[str, UncertaintyStatus]:
        return {name: result.status for name, result in self.argument_results.items()}


def load_relationship_config(
    value: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload: Any = dict(value)
    else:
        path = DEFAULT_RELATIONSHIP_CONFIG if value is None else Path(value)
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ValueError(f"could not read Phase 3.6 relationship config: {path}") from error
        except yaml.YAMLError as error:
            raise ValueError(f"invalid Phase 3.6 relationship YAML: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("Phase 3.6 relationship config must be a mapping")
    if payload.get("grounding_schema_version") != GROUNDING_SCHEMA_VERSION:
        raise ValueError("Phase 3.6 relationship config has the wrong grounding version")
    actions = payload.get("actions")
    if not isinstance(actions, Mapping) or set(actions) != set(_EXPECTED_ARGUMENTS):
        raise ValueError("relationship config must define every Phase 3.6 action exactly")
    for action, expected_arguments in _EXPECTED_ARGUMENTS.items():
        requirements = actions[action]
        if not isinstance(requirements, Mapping) or set(requirements) != set(
            expected_arguments
        ):
            raise ValueError(f"relationship config arguments are invalid for {action}")
        for argument, requirement in requirements.items():
            if not isinstance(requirement, Mapping):
                raise ValueError(f"relationship requirement {action}.{argument} must be a mapping")
            required_keys = {
                "allowed_content_types",
                "semantic_roles",
                "target_object_required_for_camera",
                "explicit_user_required",
            }
            if set(requirement) != required_keys:
                raise ValueError(
                    f"{action}.{argument} relationship keys must be exactly "
                    f"{sorted(required_keys)}"
                )
            allowed = requirement.get("allowed_content_types")
            roles = requirement.get("semantic_roles")
            if not isinstance(allowed, list) or not allowed or not all(
                isinstance(item, str) and item for item in allowed
            ):
                raise ValueError(f"{action}.{argument} requires allowed_content_types")
            if len(allowed) != len(set(allowed)):
                raise ValueError(f"{action}.{argument} allowed_content_types must be unique")
            if not isinstance(roles, list) or not all(
                isinstance(item, str) and item for item in roles
            ):
                raise ValueError(f"{action}.{argument} requires semantic_roles")
            if len(roles) != len(set(roles)):
                raise ValueError(f"{action}.{argument} semantic_roles must be unique")
            for flag in (
                "target_object_required_for_camera",
                "explicit_user_required",
            ):
                if type(requirement.get(flag)) is not bool:
                    raise ValueError(f"{action}.{argument}.{flag} must be boolean")
    return payload


def normalize_candidate_value(action: str, argument: str, value: Any) -> str:
    """Phase 3.6's narrow canonicalization; it performs no locale guessing."""

    return normalize_grounding_value(action.strip().upper(), argument, value)


def _payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        rendered = dump(mode="python", exclude_none=True)
        if isinstance(rendered, Mapping):
            return dict(rendered)
    return {}


def _enum_text(value: Any) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", value)).strip()


def _is_explicit_user(item: Any, evidence_id: str, argument: str) -> bool:
    return (
        evidence_id == f"USER:{argument}"
        and evidence_field(item, "frame_id") is None
        and evidence_field(item, "bbox") is None
        and _enum_text(evidence_field(item, "content_type", "")).lower() == "user_input"
        and _enum_text(evidence_field(item, "semantic_role", "")) == argument
        and (
            _enum_text(evidence_field(item, "registry_origin", "")).lower()
            == "user_prompt"
            or _enum_text(evidence_field(item, "physical_source", "")).lower()
            == "explicit_user"
        )
    )


def _matching_claims(item: Any, action: str, argument: str) -> tuple[Any, ...]:
    matches = []
    for claim in evidence_field(item, "claims", ()) or ():
        claim_action = _enum_text(evidence_field(claim, "action", "")).upper()
        claim_argument = _enum_text(evidence_field(claim, "argument", ""))
        if claim_action == action and claim_argument == argument:
            matches.append(claim)
    return tuple(matches)


def _has_other_claims(item: Any, action: str, argument: str) -> bool:
    claims = tuple(evidence_field(item, "claims", ()) or ())
    return bool(claims) and not _matching_claims(item, action, argument)


_DIRECTION_EDGE_PATTERN = re.compile(
    r"(?:←|→|↑|↓|⬅(?:️)?|➡(?:️)?|⬆(?:️)?|⬇(?:️)?|"
    r"\b(?:LEFT|RIGHT|STRAIGHT(?:\s+AHEAD)?|AHEAD|FORWARD|BACK|BACKWARD|"
    r"NORTH|SOUTH|EAST|WEST|NORTHEAST|NORTHWEST|SOUTHEAST|SOUTHWEST|"
    r"NE|NW|SE|SW)\b)",
    flags=re.IGNORECASE,
)


def _phase3_6_candidate_values(
    item: Any,
    action: str,
    argument: str,
    proposed: str,
) -> tuple[str, ...]:
    """Extract candidates without hiding alternative destinations/identities."""

    base: list[str] = []
    item_without_claims = _payload(item)
    item_without_claims["claims"] = []
    # Phase 3.5 deliberately preferred a structured claim over content. For
    # conflict analysis both channels must remain visible: a claim must never
    # suppress a second concrete value present in the observed region.
    for candidate in (
        *candidate_values_for_evidence(item, action, argument, proposed),
        *candidate_values_for_evidence(
            item_without_claims, action, argument, proposed
        ),
    ):
        if candidate not in base:
            base.append(candidate)
    content = unicodedata.normalize("NFKC", str(evidence_field(item, "content", "")))
    raw_candidates: list[str] = []
    if not base and action == "DIRECTION_ADVICE" and argument == "destination":
        for line in content.splitlines():
            candidate = _DIRECTION_EDGE_PATTERN.sub(" ", line)
            candidate = re.sub(r"\s+", " ", candidate).strip(" :-–—|,.;")
            if candidate:
                raw_candidates.append(candidate)
    elif not base and action == "RESTAURANT_RESERVATION" and argument == "restaurant":
        candidate = re.sub(r"\s+", " ", content).strip()
        if candidate:
            raw_candidates.append(candidate)
    for raw in raw_candidates:
        try:
            normalized = normalize_candidate_value(action, argument, raw)
        except (TypeError, ValueError, OverflowError):
            continue
        if normalized not in base:
            base.append(normalized)
    return tuple(base)


def _context_map(
    contexts: Mapping[str, EvidenceAnalysisContext | Mapping[str, Any]] | None,
) -> dict[str, EvidenceAnalysisContext]:
    parsed: dict[str, EvidenceAnalysisContext] = {}
    for key, raw in (contexts or {}).items():
        context = (
            raw
            if isinstance(raw, EvidenceAnalysisContext)
            else EvidenceAnalysisContext.model_validate(raw)
        )
        if key != context.evidence_id:
            raise ValueError("analysis context key must equal its evidence_id")
        parsed[key] = context
    return parsed


def _validated_registry(registry: Any) -> EvidenceRegistry:
    """Require either the immutable registry or its canonical serialized form."""

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


def _relationship_for_evidence(
    *,
    item: Any,
    evidence_id: str,
    action: str,
    argument: str,
    proposed: str,
    requirement: Mapping[str, Any],
    expected_target_object_id: str | None,
    context: EvidenceAnalysisContext | None,
) -> EvidenceRelationshipAssessment:
    candidates = _phase3_6_candidate_values(item, action, argument, proposed)
    content_type = _enum_text(evidence_field(item, "content_type", "")).lower()
    allowed_content_types = {
        str(value).lower() for value in requirement["allowed_content_types"]
    }
    content_finding = (
        RelationshipFinding.MATCH
        if content_type in allowed_content_types
        else RelationshipFinding.MISMATCH
    )
    if proposed in candidates:
        value_finding = RelationshipFinding.MATCH
    elif candidates:
        value_finding = RelationshipFinding.MISMATCH
    else:
        value_finding = RelationshipFinding.NOT_ASSESSED

    is_user = evidence_id.startswith("USER:")
    exact_user = _is_explicit_user(item, evidence_id, argument)
    claims = _matching_claims(item, action, argument)
    observed_role = _enum_text(evidence_field(item, "semantic_role", "")).lower()
    allowed_roles = {str(value).lower() for value in requirement["semantic_roles"]}

    if is_user:
        semantic_finding = (
            RelationshipFinding.MATCH if exact_user else RelationshipFinding.MISMATCH
        )
        argument_finding = semantic_finding
    elif observed_role:
        semantic_finding = (
            RelationshipFinding.MATCH
            if observed_role in allowed_roles
            else RelationshipFinding.MISMATCH
        )
        argument_finding = semantic_finding
    elif claims:
        # A structured claim binds a value to an argument, but contains neither
        # a semantic role nor a target-object identifier. Its claim_role is
        # deliberately ignored: no claim is privileged.
        semantic_finding = RelationshipFinding.NOT_ASSESSED
        argument_finding = RelationshipFinding.MATCH
    elif _has_other_claims(item, action, argument):
        semantic_finding = RelationshipFinding.NOT_ASSESSED
        argument_finding = RelationshipFinding.MISMATCH
    else:
        semantic_finding = RelationshipFinding.NOT_ASSESSED
        argument_finding = RelationshipFinding.NOT_ASSESSED

    if is_user or not requirement["target_object_required_for_camera"]:
        target_finding = RelationshipFinding.NOT_REQUIRED
    elif expected_target_object_id is not None and context is not None:
        if context.associated_target_object_id is None:
            target_finding = RelationshipFinding.NOT_ASSESSED
        else:
            target_finding = (
                RelationshipFinding.MATCH
                if context.associated_target_object_id == expected_target_object_id
                else RelationshipFinding.MISMATCH
            )
    else:
        target_finding = RelationshipFinding.NOT_ASSESSED

    if requirement["explicit_user_required"] and not exact_user:
        semantic_finding = RelationshipFinding.MISMATCH
        argument_finding = RelationshipFinding.MISMATCH

    acceptable_target = target_finding in {
        RelationshipFinding.MATCH,
        RelationshipFinding.NOT_REQUIRED,
    }
    task_context_satisfied = all(
        finding is RelationshipFinding.MATCH
        for finding in (content_finding, semantic_finding, argument_finding)
    ) and acceptable_target
    supports = task_context_satisfied and value_finding is RelationshipFinding.MATCH
    # Conflict candidates must pass every non-value task relationship. Unknown
    # target/role context remains inspectable but cannot manufacture a conflict.
    plausible = bool(candidates) and task_context_satisfied

    reasons = (
        f"value_relationship={value_finding.value}",
        f"target_object_relationship={target_finding.value}",
        f"semantic_role_relationship={semantic_finding.value}",
        f"argument_relationship={argument_finding.value}",
    )
    return EvidenceRelationshipAssessment(
        evidence_id=evidence_id,
        candidate_values=tuple(candidates),
        content_type_relationship=content_finding,
        value_relationship=value_finding,
        target_object_relationship=target_finding,
        semantic_role_relationship=semantic_finding,
        argument_relationship=argument_finding,
        task_context_satisfied=task_context_satisfied,
        supports_proposed_argument=supports,
        plausible_candidate=plausible,
        reasons=reasons,
    )


def _build_conflict_set(
    argument: str,
    relationships: Mapping[str, EvidenceRelationshipAssessment],
    *,
    evidence_domain: set[str] | None = None,
) -> ArgumentConflictSet:
    evidence_by_value: dict[str, set[str]] = {}
    for evidence_id, assessment in relationships.items():
        if evidence_domain is not None and evidence_id not in evidence_domain:
            continue
        if not assessment.plausible_candidate:
            continue
        for value in assessment.candidate_values:
            evidence_by_value.setdefault(value, set()).add(evidence_id)
    candidates = tuple(
        ConflictCandidate(normalized_value=value, evidence_ids=tuple(sorted(evidence_ids)))
        for value, evidence_ids in sorted(evidence_by_value.items())
    )
    return ArgumentConflictSet(argument=argument, candidates=candidates)


def _authenticity_for_evidence(
    evidence_ids: tuple[str, ...],
    contexts: Mapping[str, EvidenceAnalysisContext],
    registry: Any,
) -> tuple[AuthenticityStatus, str | None]:
    if not evidence_ids:
        return AuthenticityStatus.NOT_ASSESSED, None
    statuses = []
    bases = []
    for evidence_id in evidence_ids:
        if evidence_id.startswith("USER:"):
            statuses.append(AuthenticityStatus.NOT_REQUIRED)
            continue
        context = contexts.get(evidence_id)
        status = (
            context.authenticity_status
            if context is not None
            else AuthenticityStatus.NOT_ASSESSED
        )
        origin = _enum_text(
            evidence_field(registry_get(registry, evidence_id), "registry_origin", "")
        ).lower()
        if status is AuthenticityStatus.NOT_ASSESSED and origin in {
            "physical_annotation",
            "automatic_perception",
            "automatic_registry",
        }:
            status = AuthenticityStatus.UNKNOWN
        statuses.append(status)
        if context is not None and context.authenticity_basis is not None:
            bases.append(f"{evidence_id}: {context.authenticity_basis}")
    if AuthenticityStatus.UNKNOWN in statuses:
        return AuthenticityStatus.UNKNOWN, None
    if statuses and all(status is AuthenticityStatus.NOT_REQUIRED for status in statuses):
        return AuthenticityStatus.NOT_REQUIRED, None
    if statuses and all(status is AuthenticityStatus.ESTABLISHED for status in statuses):
        return AuthenticityStatus.ESTABLISHED, "; ".join(bases)
    return AuthenticityStatus.NOT_ASSESSED, None


def _empty_argument_result(
    *,
    argument: str,
    value: Any,
    reference_ids: tuple[str, ...],
    status: UncertaintyStatus,
    reason: str,
) -> ArgumentEvidenceAnalysis:
    uncertainty = ArgumentUncertaintyAssessment(
        argument=argument,
        argument_value=value,
        status=status,
        authenticity_status=AuthenticityStatus.NOT_ASSESSED,
        evidence_ids=reference_ids,
        grounding_confidence=None,
        reasons=(reason,),
    )
    return ArgumentEvidenceAnalysis(
        argument=argument,
        argument_value=value,
        referenced_evidence_ids=reference_ids,
        conflict_set=ArgumentConflictSet(argument=argument),
        uncertainty=uncertainty,
    )


def _analyze_argument(
    *,
    action: str,
    argument: str,
    value: Any,
    registry: Any,
    reference_validation: EvidenceReferenceValidation,
    requirement: Mapping[str, Any],
    expected_target_object_id: str | None,
    contexts: Mapping[str, EvidenceAnalysisContext],
) -> ArgumentEvidenceAnalysis:
    reference_result = reference_validation.argument_results[argument]
    if reference_result.status is ArgumentReferenceStatus.MISSING:
        return _empty_argument_result(
            argument=argument,
            value=value,
            reference_ids=reference_result.reference_ids,
            status=UncertaintyStatus.MISSING,
            reason="critical argument has no complete evidence-reference coverage",
        )
    if reference_result.status is ArgumentReferenceStatus.INVALID_REFERENCE:
        return _empty_argument_result(
            argument=argument,
            value=value,
            reference_ids=reference_result.reference_ids,
            status=UncertaintyStatus.INVALID_REFERENCE,
            reason="one or more evidence references failed strict validation",
        )
    try:
        proposed = normalize_candidate_value(action, argument, value)
    except (TypeError, ValueError, OverflowError):
        return _empty_argument_result(
            argument=argument,
            value=value,
            reference_ids=reference_result.reference_ids,
            status=UncertaintyStatus.UNSUPPORTED,
            reason="argument value cannot be deterministically normalized",
        )

    relationships: dict[str, EvidenceRelationshipAssessment] = {}
    for item in registry_items(registry):
        evidence_id = evidence_field(item, "evidence_id")
        if not isinstance(evidence_id, str):
            continue
        assessment = _relationship_for_evidence(
            item=item,
            evidence_id=evidence_id,
            action=action,
            argument=argument,
            proposed=proposed,
            requirement=requirement,
            expected_target_object_id=expected_target_object_id,
            context=contexts.get(evidence_id),
        )
        if assessment.candidate_values or evidence_id in reference_result.resolved_evidence_ids:
            relationships[evidence_id] = assessment

    selected = tuple(reference_result.resolved_evidence_ids)
    selected_assessments = [relationships[evidence_id] for evidence_id in selected]
    supporting = tuple(
        assessment.evidence_id
        for assessment in selected_assessments
        if assessment.supports_proposed_argument
    )
    selected_value_matches = [
        assessment
        for assessment in selected_assessments
        if assessment.value_relationship is RelationshipFinding.MATCH
    ]
    selected_value_mismatches = [
        assessment
        for assessment in selected_assessments
        if assessment.value_relationship is RelationshipFinding.MISMATCH
    ]
    selected_mismatches = [
        assessment
        for assessment in selected_assessments
        if RelationshipFinding.MISMATCH
        in {
            assessment.content_type_relationship,
            assessment.target_object_relationship,
            assessment.semantic_role_relationship,
            assessment.argument_relationship,
        }
    ]
    selected_unassessed = [
        assessment
        for assessment in selected_value_matches
        if RelationshipFinding.NOT_ASSESSED
        in {
            assessment.target_object_relationship,
            assessment.semantic_role_relationship,
            assessment.argument_relationship,
        }
    ]
    selected_extras = [
        assessment
        for assessment in selected_assessments
        if not assessment.supports_proposed_argument
    ]
    explicit_user_support = {
        evidence_id
        for evidence_id in supporting
        if _is_explicit_user(registry_get(registry, evidence_id), evidence_id, argument)
    }
    conflict_set = _build_conflict_set(
        argument,
        relationships,
        evidence_domain=explicit_user_support or None,
    )

    authenticity_evidence_ids = tuple(
        sorted(
            {
                *supporting,
                *(
                    evidence_id
                    for candidate in conflict_set.candidates
                    for evidence_id in candidate.evidence_ids
                ),
            }
        )
    )
    authenticity_status, authenticity_basis = _authenticity_for_evidence(
        authenticity_evidence_ids, contexts, registry
    )
    if not selected_value_matches and selected_value_mismatches:
        status = UncertaintyStatus.UNSUPPORTED
        reason = "referenced evidence does not support the proposed normalized value"
    elif selected_mismatches:
        status = UncertaintyStatus.UNSUPPORTED
        reason = "a required task evidence relationship is a clear mismatch"
    elif conflict_set.has_conflict:
        status = UncertaintyStatus.CONFLICTING
        reason = "multiple distinct plausible normalized values are present"
    elif not selected_value_matches or selected_unassessed or not supporting:
        status = UncertaintyStatus.INSUFFICIENT_EVIDENCE
        reason = "a required task evidence relationship could not be established"
    elif selected_extras:
        status = UncertaintyStatus.AMBIGUOUS
        reason = "the selected set mixes support with unrelated or contradictory evidence"
    elif authenticity_status is AuthenticityStatus.UNKNOWN:
        status = UncertaintyStatus.AUTHENTICITY_UNKNOWN
        reason = "the value is grounded but physical authenticity remains unresolved"
    else:
        status = UncertaintyStatus.SUPPORTED
        reason = "the proposed value and all required task relationships are supported"

    relevant_ids = tuple(sorted(relationships))
    confidence_dimensions = tuple(
        EvidenceConfidenceDimensions(
            evidence_id=evidence_id,
            detection_confidence=evidence_field(
                registry_get(registry, evidence_id), "detection_confidence"
            ),
            ocr_confidence=evidence_field(registry_get(registry, evidence_id), "ocr_confidence"),
            # Phase 3.6 has no calibrated grounding-confidence estimator. A
            # pre-existing registry number is therefore not authoritative.
            grounding_confidence=None,
        )
        for evidence_id in relevant_ids
    )
    uncertainty = ArgumentUncertaintyAssessment(
        argument=argument,
        argument_value=value,
        status=status,
        authenticity_status=authenticity_status,
        authenticity_basis=authenticity_basis,
        candidate_values=conflict_set.distinct_values,
        evidence_ids=relevant_ids,
        evidence_confidences=confidence_dimensions,
        grounding_confidence=None,
        reasons=(reason,),
    )
    return ArgumentEvidenceAnalysis(
        argument=argument,
        argument_value=value,
        normalized_argument_value=proposed,
        referenced_evidence_ids=reference_result.reference_ids,
        supporting_evidence_ids=supporting,
        relationship_assessments=relationships,
        conflict_set=conflict_set,
        uncertainty=uncertainty,
    )


def analyze_evidence_uncertainty(
    action_output: Any,
    registry: Any,
    *,
    relationship_config: Mapping[str, Any] | str | Path | None = None,
    argument_target_object_ids: Mapping[str, str] | None = None,
    evidence_contexts: Mapping[
        str, EvidenceAnalysisContext | Mapping[str, Any]
    ]
    | None = None,
    reference_validation: EvidenceReferenceValidation | None = None,
    frame_id: str | None = None,
) -> EvidenceAnalysisResult:
    """Analyze every argument independently without changing the proposal."""

    config = load_relationship_config(relationship_config)
    registry = _validated_registry(registry)
    current_reference_validation = validate_evidence_references(
        action_output, registry, frame_id=frame_id
    )
    if reference_validation is None:
        reference_validation = current_reference_validation
    elif reference_validation != current_reference_validation:
        raise ValueError(
            "precomputed reference validation does not match the current output and registry"
        )
    payload = _payload(action_output)
    action = _enum_text(payload.get("action", reference_validation.action)).upper()
    arguments = _payload(payload.get("arguments"))
    requirements = config["actions"].get(action)
    if not isinstance(requirements, Mapping):
        raise ValueError(f"action {action!r} is absent from Phase 3.6 relationships")
    targets = dict(argument_target_object_ids or {})
    unknown_targets = set(targets) - set(reference_validation.expected_arguments)
    if unknown_targets:
        raise ValueError(f"target-object context names unknown arguments {sorted(unknown_targets)}")
    if any(not isinstance(value, str) or not value.strip() for value in targets.values()):
        raise ValueError("target-object identifiers must be nonblank strings")
    targets = {key: value.strip() for key, value in targets.items()}
    if action == "DIRECTION_ADVICE" and {"direction", "destination"} <= set(targets):
        if targets["direction"] != targets["destination"]:
            raise ValueError(
                "navigation direction and destination must share one target-object identifier"
            )
    contexts = _context_map(evidence_contexts)
    registry_ids = {
        evidence_field(item, "evidence_id") for item in registry_items(registry)
    }
    unknown_contexts = set(contexts) - registry_ids
    if unknown_contexts:
        raise ValueError(f"analysis context refers to unknown evidence {sorted(unknown_contexts)}")

    results = {}
    for argument in reference_validation.expected_arguments:
        requirement = requirements.get(argument)
        if not isinstance(requirement, Mapping):
            results[argument] = _empty_argument_result(
                argument=argument,
                value=arguments.get(argument),
                reference_ids=reference_validation.argument_results[argument].reference_ids,
                status=UncertaintyStatus.UNSUPPORTED,
                reason="argument has no Phase 3.6 relationship requirement",
            )
            continue
        results[argument] = _analyze_argument(
            action=action,
            argument=argument,
            value=arguments.get(argument),
            registry=registry,
            reference_validation=reference_validation,
            requirement=requirement,
            expected_target_object_id=targets.get(argument),
            contexts=contexts,
        )
    return EvidenceAnalysisResult(
        action=action,
        frame_id=reference_validation.frame_id,
        argument_results=results,
        reference_validation=reference_validation,
    )


analyze_argument_evidence = analyze_evidence_uncertainty
