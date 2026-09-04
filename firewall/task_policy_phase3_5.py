"""Task-specific evidence relationships for LensGuard Phase 3.5.

This policy consumes deterministic grounding and immutable registry metadata.
It never asks the action VLM whether evidence is trusted, malicious, official,
or authorized.  In particular, camera evidence is not rejected merely because
it came from a camera; each argument is evaluated against its task relationship.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from phase3_5_constants import (
    ACTION_REGISTRY_VERSION,
    EVIDENCE_SCHEMA_VERSION,
    MODEL_CONTRACT_VERSION,
    POLICY_VERSION,
)
from provenance.grounding_validator_phase3_5 import (
    GroundingStatus,
    GroundingValidationResult,
    candidate_values_for_evidence,
)
from provenance.reference_validator_phase3_5 import evidence_field, registry_get, registry_items


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE3_5_POLICY_PATH = _PROJECT_ROOT / "config/policy_phase3_5.yaml"
DEFAULT_PHASE3_5_ACTION_REGISTRY_PATH = _PROJECT_ROOT / "config/action_registry_phase3_5.yaml"


class PolicyDisposition(StrEnum):
    PASS = "PASS"
    ESCALATE = "ESCALATE"
    BLOCK = "BLOCK"


_DISPOSITION_RANK = {
    PolicyDisposition.PASS: 0,
    PolicyDisposition.ESCALATE: 1,
    PolicyDisposition.BLOCK: 2,
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArgumentPolicyAssessment(_StrictModel):
    argument_name: str
    disposition: PolicyDisposition
    relationship_satisfied: bool
    supporting_evidence_ids: tuple[str, ...] = ()
    rule_ids: tuple[str, ...] = ()
    messages: tuple[str, ...] = ()


class TaskPolicyResult(_StrictModel):
    action: str
    disposition: PolicyDisposition
    argument_results: dict[str, ArgumentPolicyAssessment] = Field(default_factory=dict)
    rule_ids: tuple[str, ...] = ()
    messages: tuple[str, ...] = ()
    policy_version: str
    action_registry_version: str


def _load_yaml_mapping(
    value: Mapping[str, Any] | str | Path | None,
    *,
    default_path: Path,
    label: str,
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload: Any = dict(value)
    else:
        path = default_path if value is None else Path(value)
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"could not read {label}: {path}") from exc
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid {label} YAML: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a mapping")
    return deepcopy(payload)


def load_phase3_5_action_registry(
    value: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    registry = _load_yaml_mapping(
        value,
        default_path=DEFAULT_PHASE3_5_ACTION_REGISTRY_PATH,
        label="Phase 3.5 action registry",
    )
    version = registry.get("registry_version")
    if version != ACTION_REGISTRY_VERSION:
        raise ValueError(
            f"Phase 3.5 registry_version must be exactly {ACTION_REGISTRY_VERSION!r}"
        )
    if registry.get("evidence_schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError("Phase 3.5 action registry evidence-schema version mismatch")
    if registry.get("model_contract_version") != MODEL_CONTRACT_VERSION:
        raise ValueError("Phase 3.5 action registry model-contract version mismatch")
    actions = registry.get("actions")
    if not isinstance(actions, Mapping) or not actions:
        raise ValueError("Phase 3.5 action registry requires an actions mapping")
    for action, definition in actions.items():
        if not isinstance(action, str) or not isinstance(definition, Mapping):
            raise ValueError("action registry entries must be named mappings")
        critical = definition.get("critical_arguments")
        if not isinstance(critical, list) or any(
            not isinstance(argument, str) or not argument for argument in critical
        ):
            raise ValueError(f"{action}.critical_arguments must be a list of names")
        if len(critical) != len(set(critical)):
            raise ValueError(f"{action}.critical_arguments contains duplicates")
        effects = definition.get("effects")
        if not isinstance(effects, list) or any(not isinstance(item, str) for item in effects):
            raise ValueError(f"{action}.effects must be a list of strings")
        for required in ("action_kind", "reversibility", "default_risk"):
            if not isinstance(definition.get(required), str):
                raise ValueError(f"{action}.{required} must be a string")
    return registry


def load_phase3_5_task_policy(
    value: Mapping[str, Any] | str | Path | None = None,
    *,
    action_registry: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    policy = _load_yaml_mapping(
        value,
        default_path=DEFAULT_PHASE3_5_POLICY_PATH,
        label="Phase 3.5 policy",
    )
    registry = load_phase3_5_action_registry(action_registry)
    if policy.get("policy_version") != POLICY_VERSION:
        raise ValueError(f"Phase 3.5 policy_version must be exactly {POLICY_VERSION!r}")
    if policy.get("action_registry_version") != registry["registry_version"]:
        raise ValueError("Phase 3.5 policy/action-registry version mismatch")

    status_decisions = policy.get("grounding_status_decisions")
    if not isinstance(status_decisions, Mapping):
        raise ValueError("policy requires grounding_status_decisions")
    expected_statuses = {status.value for status in GroundingStatus}
    if set(status_decisions) != expected_statuses:
        raise ValueError(
            "grounding_status_decisions must define exactly "
            f"{sorted(expected_statuses)}"
        )
    for status, disposition in status_decisions.items():
        try:
            PolicyDisposition(disposition)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid disposition for grounding status {status}") from exc

    confidence = policy.get("confidence")
    if not isinstance(confidence, Mapping):
        raise ValueError("policy requires confidence settings")
    for field in ("minimum_detection_confidence", "minimum_ocr_confidence"):
        threshold = confidence.get(field)
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise ValueError(f"confidence.{field} must be numeric")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(f"confidence.{field} must be between zero and one")
    for field in (
        "require_detection_for_registry_origins",
        "require_ocr_for_text_registry_origins",
    ):
        origins = confidence.get(field)
        if not isinstance(origins, list) or any(not isinstance(item, str) for item in origins):
            raise ValueError(f"confidence.{field} must be a list of strings")

    actions = policy.get("actions")
    if not isinstance(actions, Mapping):
        raise ValueError("policy requires actions")
    if set(actions) != set(registry["actions"]):
        raise ValueError("Phase 3.5 policy and action registry must define the same actions")
    for action, definition in actions.items():
        if not isinstance(definition, Mapping):
            raise ValueError(f"policy action {action} must be a mapping")
        requirements = definition.get("argument_requirements")
        if not isinstance(requirements, Mapping):
            raise ValueError(f"{action}.argument_requirements must be a mapping")
        expected_arguments = set(registry["actions"][action]["critical_arguments"])
        if set(requirements) != expected_arguments:
            raise ValueError(
                f"{action} policy arguments differ from action registry: "
                f"expected {sorted(expected_arguments)}, got {sorted(requirements)}"
            )
        for argument, requirement in requirements.items():
            if not isinstance(requirement, Mapping):
                raise ValueError(f"{action}.{argument} requirement must be a mapping")
            if not isinstance(requirement.get("relation"), str):
                raise ValueError(f"{action}.{argument} requires a relation")
            content_types = requirement.get("allowed_content_types")
            if not isinstance(content_types, list) or not content_types:
                raise ValueError(f"{action}.{argument} requires allowed_content_types")
    return policy


def _payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python", exclude_none=True)
        if isinstance(dumped, Mapping):
            return dict(dumped)
    return {}


def _enum_lower(value: Any) -> str:
    return str(getattr(value, "value", value)).strip().lower()


def _more_restrictive(
    left: PolicyDisposition, right: PolicyDisposition
) -> PolicyDisposition:
    return left if _DISPOSITION_RANK[left] >= _DISPOSITION_RANK[right] else right


def _is_explicit_user_record(evidence_id: str, item: Any) -> bool:
    return (
        evidence_id.startswith("USER:")
        and evidence_field(item, "frame_id") is None
        and evidence_field(item, "bbox") is None
        and _enum_lower(evidence_field(item, "content_type", "")) == "user_input"
        and (
            _enum_lower(evidence_field(item, "registry_origin", "")) == "user_prompt"
            or _enum_lower(evidence_field(item, "physical_source", "")) == "explicit_user"
        )
    )


def _confidence_rules(
    item: Any,
    evidence_id: str,
    confidence_policy: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    rules: list[str] = []
    messages: list[str] = []
    if evidence_id.startswith("USER:"):
        return rules, messages

    origin = _enum_lower(evidence_field(item, "registry_origin", ""))
    content_type = _enum_lower(evidence_field(item, "content_type", ""))
    detection = evidence_field(item, "detection_confidence")
    ocr = evidence_field(item, "ocr_confidence")
    minimum_detection = float(confidence_policy["minimum_detection_confidence"])
    minimum_ocr = float(confidence_policy["minimum_ocr_confidence"])
    detection_required = origin in {
        item.lower() for item in confidence_policy["require_detection_for_registry_origins"]
    }
    ocr_required = content_type == "text" and origin in {
        item.lower() for item in confidence_policy["require_ocr_for_text_registry_origins"]
    }

    if detection is None and detection_required:
        rules.append("PHASE3_5_DETECTION_CONFIDENCE_MISSING")
        messages.append(f"{evidence_id} lacks required detector confidence")
    elif detection is not None and float(detection) < minimum_detection:
        rules.append("PHASE3_5_DETECTION_CONFIDENCE_LOW")
        messages.append(f"{evidence_id} has insufficient detector confidence")

    if ocr is None and ocr_required:
        rules.append("PHASE3_5_OCR_CONFIDENCE_MISSING")
        messages.append(f"{evidence_id} lacks required OCR confidence")
    elif ocr is not None and float(ocr) < minimum_ocr:
        rules.append("PHASE3_5_OCR_CONFIDENCE_LOW")
        messages.append(f"{evidence_id} has insufficient OCR confidence")
    return rules, messages


def _argument_policy(
    *,
    action: str,
    argument_name: str,
    grounding: Any,
    requirement: Mapping[str, Any],
    registry: Any,
    policy: Mapping[str, Any],
) -> ArgumentPolicyAssessment:
    disposition = PolicyDisposition(
        policy["grounding_status_decisions"][grounding.status.value]
    )
    rules: list[str] = [
        f"PHASE3_5_{action}_{argument_name}_{grounding.status.value}"
    ]
    messages: list[str] = [
        f"{argument_name} grounding status is {grounding.status.value}"
    ]
    relationship_satisfied = grounding.status in {
        GroundingStatus.SUPPORTED,
        GroundingStatus.AMBIGUOUS,
        GroundingStatus.CONFLICTING,
    } and bool(grounding.supporting_evidence_ids)

    supporting_items = [
        (evidence_id, registry_get(registry, evidence_id))
        for evidence_id in grounding.supporting_evidence_ids
    ]
    allowed_content_types = {
        str(item).lower() for item in requirement.get("allowed_content_types", [])
    }
    if supporting_items and any(
        _enum_lower(evidence_field(item, "content_type", "")) not in allowed_content_types
        for _, item in supporting_items
    ):
        relationship_satisfied = False
        disposition = PolicyDisposition.BLOCK
        rules.append(f"PHASE3_5_{action}_{argument_name}_CONTENT_TYPE_VIOLATION")
        messages.append(f"{argument_name} is supported by an ineligible content type")

    allowed_roles = {
        str(item).lower() for item in requirement.get("semantic_roles_if_present", [])
    }
    observed_roles = []
    for evidence_id, item in supporting_items:
        # USER evidence is deliberately named for its argument (for example,
        # ``semantic_role=time``).  Camera-only role vocabularies must not turn
        # that trusted non-camera channel into a false relationship violation.
        if _is_explicit_user_record(evidence_id, item):
            continue
        role = evidence_field(item, "semantic_role")
        if role is not None:
            observed_roles.append(_enum_lower(role))
    # The current Phase 2 adapter correctly leaves absent semantic roles null.
    # If roles are available, however, all selected support must satisfy the
    # declared relationship; an unrelated number cannot inherit phone authority.
    if observed_roles and allowed_roles and any(
        role not in allowed_roles for role in observed_roles
    ):
        relationship_satisfied = False
        disposition = PolicyDisposition.BLOCK
        rules.append(f"PHASE3_5_{action}_{argument_name}_SEMANTIC_ROLE_VIOLATION")
        messages.append(f"{argument_name} is bound to the wrong semantic role")

    if requirement.get("require_explicit_user_evidence"):
        if not supporting_items or not all(
            _is_explicit_user_record(evidence_id, item)
            for evidence_id, item in supporting_items
        ):
            relationship_satisfied = False
            disposition = PolicyDisposition.BLOCK
            rules.append(f"PHASE3_5_{action}_{argument_name}_USER_EVIDENCE_REQUIRED")
            messages.append(f"{argument_name} must bind to explicit USER evidence")

    for evidence_id, item in supporting_items:
        confidence_rules, confidence_messages = _confidence_rules(
            item,
            evidence_id,
            policy["confidence"],
        )
        if confidence_rules:
            disposition = _more_restrictive(disposition, PolicyDisposition.ESCALATE)
            rules.extend(confidence_rules)
            messages.extend(confidence_messages)

    return ArgumentPolicyAssessment(
        argument_name=argument_name,
        disposition=disposition,
        relationship_satisfied=relationship_satisfied,
        supporting_evidence_ids=grounding.supporting_evidence_ids,
        rule_ids=tuple(dict.fromkeys(rules)),
        messages=tuple(messages),
    )


def _hazard_evidence_ids(registry: Any) -> tuple[str, ...]:
    found: list[str] = []
    for item in registry_items(registry):
        evidence_id = evidence_field(item, "evidence_id")
        if not isinstance(evidence_id, str) or evidence_id.startswith("USER:"):
            continue
        candidates = candidate_values_for_evidence(
            item,
            "SAFETY_ADVICE",
            "hazard",
            "NONE",
        )
        if any(candidate != "NONE" for candidate in candidates):
            found.append(evidence_id)
    return tuple(found)


def evaluate_task_evidence_policy(
    action_output: Any,
    registry: Any,
    grounding: GroundingValidationResult,
    *,
    policy: Mapping[str, Any] | str | Path | None = None,
    action_registry: Mapping[str, Any] | str | Path | None = None,
) -> TaskPolicyResult:
    """Apply configured argument relationships and cross-evidence safety rules."""

    registry_config = load_phase3_5_action_registry(action_registry)
    policy_config = load_phase3_5_task_policy(policy, action_registry=registry_config)
    payload = _payload(action_output)
    action = str(getattr(payload.get("action"), "value", payload.get("action", ""))).upper()
    action_policy = policy_config["actions"].get(action)
    if not isinstance(action_policy, Mapping):
        raise ValueError(f"action {action!r} is not registered in Phase 3.5 policy")

    argument_results: dict[str, ArgumentPolicyAssessment] = {}
    all_rules: list[str] = []
    all_messages: list[str] = []
    disposition = PolicyDisposition.PASS
    requirements = action_policy["argument_requirements"]
    for argument_name, grounding_result in grounding.argument_results.items():
        requirement = requirements.get(argument_name)
        if not isinstance(requirement, Mapping):
            # A reference to an extension argument that has no policy is a clear
            # policy violation, not something the gate may silently ignore.
            assessment = ArgumentPolicyAssessment(
                argument_name=argument_name,
                disposition=PolicyDisposition.BLOCK,
                relationship_satisfied=False,
                rule_ids=(f"PHASE3_5_{action}_{argument_name}_NO_POLICY",),
                messages=("argument has no registered evidence relationship",),
            )
        else:
            assessment = _argument_policy(
                action=action,
                argument_name=argument_name,
                grounding=grounding_result,
                requirement=requirement,
                registry=registry,
                policy=policy_config,
            )
        argument_results[argument_name] = assessment
        disposition = _more_restrictive(disposition, assessment.disposition)
        all_rules.extend(assessment.rule_ids)
        all_messages.extend(assessment.messages)

    if action == "DIRECTION_ADVICE":
        direction = grounding.argument_results.get("direction")
        if direction is not None and direction.status is GroundingStatus.CONFLICTING:
            conflict_rule = action_policy["conflict_rule"]
            conflict_disposition = PolicyDisposition(conflict_rule["disposition"])
            disposition = _more_restrictive(disposition, conflict_disposition)
            all_rules.append(str(conflict_rule["id"]))
            all_messages.append("conflicting direction-bearing regions require escalation")

    if action == "SAFETY_ADVICE":
        arguments = _payload(payload.get("arguments"))
        safe_value = arguments.get("safe_to_proceed")
        positive_claim = safe_value is True or (
            isinstance(safe_value, str) and safe_value.strip().lower() == "true"
        )
        hazards = _hazard_evidence_ids(registry)
        if positive_claim and hazards:
            veto_rule = action_policy["hazard_veto_rule"]
            veto_disposition = PolicyDisposition(veto_rule["disposition"])
            disposition = _more_restrictive(disposition, veto_disposition)
            all_rules.append(str(veto_rule["id"]))
            all_messages.append(
                "positive safety advice is vetoed by grounded hazard evidence: "
                + ", ".join(hazards)
            )

    return TaskPolicyResult(
        action=action,
        disposition=disposition,
        argument_results=argument_results,
        rule_ids=tuple(dict.fromkeys(all_rules)),
        messages=tuple(all_messages),
        policy_version=str(policy_config["policy_version"]),
        action_registry_version=str(registry_config["registry_version"]),
    )


evaluate_task_policy = evaluate_task_evidence_policy
