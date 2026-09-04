"""Deterministic Phase 2 gate over mapped evidence and estimated provenance.

The thin gate performs no model calls and never executes an action. Static
effects, reversibility, risk, and action-specific source trust come from the
Phase 1 action registry. Region ground truth is ignored in automatic mode and
may be consulted only by the explicitly selected experiment-only oracle mode;
the annotation itself never enters the operational gate output.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from provenance.evidence_mapper import (
    ActionEvidenceMap,
    ArgumentEvidenceMapping,
    MappingStatus,
    normalize_argument_value,
)

from .action_normalizer import CRITICAL_ARGUMENTS, critical_arguments_for, normalize_action
from .action_schema import ActionType, Decision, ProposedAction

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_POLICY = _PROJECT_ROOT / "config/policy_phase2.yaml"
_DEFAULT_REGISTRY = _PROJECT_ROOT / "config/action_registry.yaml"
_DECISION_RANK = {
    Decision.ALLOW: 0,
    Decision.WARN: 1,
    Decision.CONFIRM: 2,
    Decision.BLOCK: 3,
}


class GateProvenanceMode(StrEnum):
    """Provenance authority available to the deterministic gate.

    Model-estimated source categories are descriptive evidence only. Oracle
    mode is an explicit experiment-only opt-in that may consult region source
    annotations supplied by the benchmark.
    """

    MODEL_ESTIMATED = "MODEL_ESTIMATED_PROVENANCE"
    ORACLE = "ORACLE_REGION_PROVENANCE"


class AuthorizationBasis(StrEnum):
    NONE = "none"
    EXPLICIT_USER = "explicit_user"
    TRUSTED_REFERENCE = "trusted_reference"
    AUTHENTICATED_UPDATE = "authenticated_update"
    ORACLE_SOURCE = "oracle_source"


class AuthenticatedUpdate(BaseModel):
    """Value and source proven by a separate authenticated input channel."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    value: str = Field(min_length=1)
    source: str = Field(min_length=1)


class ArgumentGateAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    argument_name: str
    argument_value: str | None
    evidence_status: MappingStatus
    matched_region_id: str | None
    model_source_estimate: str | None
    model_source_confidence: float | None
    model_source_in_trusted_registry: bool
    conflict_with_reference: bool | None
    user_authorization_corroborated: bool
    trusted_reference_match: bool
    authenticated_update_match: bool
    authenticated_update_source: str | None
    oracle_source_used: bool
    authorization_basis: AuthorizationBasis
    trusted_source: bool
    decision: Decision
    policy_rule_id: str


class ThinGateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Decision
    action: ActionType
    critical_arguments: dict[str, str] = Field(default_factory=dict)
    argument_assessments: dict[str, ArgumentGateAssessment] = Field(default_factory=dict)
    static_effects: list[str] = Field(default_factory=list)
    reversibility: str
    default_risk: str
    policy_rules_triggered: list[str] = Field(default_factory=list)
    user_message: str
    policy_version: str
    registry_version: str
    provenance_mode: str = GateProvenanceMode.MODEL_ESTIMATED.value
    dry_run: bool = True


def _normalize_action_input(value: ProposedAction | Mapping[str, Any] | Any) -> ProposedAction:
    if isinstance(value, (ProposedAction, Mapping)):
        return normalize_action(value)
    as_proposed_action = getattr(value, "as_proposed_action", None)
    if callable(as_proposed_action):
        return normalize_action(as_proposed_action())
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return normalize_action(model_dump(mode="python"))
    raise TypeError("action must be a proposed-action mapping or compatible model")


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
            raise ValueError(f"Could not read {label}: {path}") from exc
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid {label} YAML: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a mapping")
    return deepcopy(payload)


def load_thin_gate_policy(
    path: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    policy = _load_yaml_mapping(path, default_path=_DEFAULT_POLICY, label="Phase 2 policy")
    if not isinstance(policy.get("policy_version"), str):
        raise ValueError("Phase 2 policy requires policy_version")
    threshold = policy.get("minimum_source_confidence")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError("minimum_source_confidence must be numeric")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("minimum_source_confidence must be between 0 and 1")
    actions = policy.get("actions")
    if not isinstance(actions, Mapping):
        raise ValueError("Phase 2 policy requires an actions mapping")
    for action in ActionType:
        if action.value not in actions or not isinstance(actions[action.value], Mapping):
            raise ValueError(f"Phase 2 policy is missing {action.value}")
        if action is ActionType.NONE:
            continue
        action_policy = actions[action.value]
        uncorroborated_rule = _rule(
            action_policy,
            "uncorroborated_trusted_source_rule",
        )
        if Decision(uncorroborated_rule["decision"]) is Decision.ALLOW:
            raise ValueError(
                f"{action.value} uncorroborated trusted-looking source rule must escalate"
            )
    return policy


def load_action_registry(
    path: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    registry = _load_yaml_mapping(path, default_path=_DEFAULT_REGISTRY, label="action registry")
    if not isinstance(registry.get("registry_version"), str):
        raise ValueError("action registry requires registry_version")
    if not isinstance(registry.get("actions"), Mapping):
        raise ValueError("action registry requires an actions mapping")
    return registry


def _rule(action_policy: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = action_policy.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Phase 2 action policy is missing {name}")
    if not isinstance(value.get("id"), str) or not value["id"]:
        raise ValueError(f"Phase 2 {name} requires a rule id")
    try:
        Decision(value.get("decision"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Phase 2 {name} has an invalid decision") from exc
    if not isinstance(value.get("message"), str) or not value["message"].strip():
        raise ValueError(f"Phase 2 {name} requires a message")
    return value


def _status_rule(action_policy: Mapping[str, Any], status: MappingStatus) -> Mapping[str, Any]:
    status_rules = action_policy.get("status_rules")
    if not isinstance(status_rules, Mapping):
        raise ValueError("Phase 2 action policy requires status_rules")
    return _rule(status_rules, status.value)


def _reference_conflict(
    action: ActionType,
    argument: str,
    proposed_value: str | None,
    reference_arguments: Mapping[str, Any],
) -> bool | None:
    if proposed_value is None or argument not in reference_arguments:
        return None
    reference = reference_arguments[argument]
    if not isinstance(reference, str):
        raise TypeError(f"reference argument {argument!r} must be a string")
    return normalize_argument_value(action, argument, proposed_value) != normalize_argument_value(
        action, argument, reference
    )


def _argument_value_matches(
    action: ActionType,
    argument: str,
    proposed_value: str | None,
    candidate_value: Any,
    *,
    label: str,
) -> bool:
    if proposed_value is None:
        return False
    if not isinstance(candidate_value, str):
        raise TypeError(f"{label} argument {argument!r} must be a string")
    return normalize_argument_value(action, argument, proposed_value) == normalize_argument_value(
        action, argument, candidate_value
    )


def _parse_provenance_mode(value: GateProvenanceMode | str) -> GateProvenanceMode:
    raw = getattr(value, "value", value)
    try:
        return GateProvenanceMode(raw)
    except (TypeError, ValueError) as exc:
        expected = ", ".join(item.value for item in GateProvenanceMode)
        raise ValueError(f"provenance_mode must be one of: {expected}") from exc


def _validate_argument_keys(
    values: Mapping[str, Any],
    registered_arguments: tuple[str, ...],
    *,
    label: str,
) -> None:
    unknown = set(values) - set(registered_arguments)
    if unknown:
        raise ValueError(f"{label} contains unknown critical arguments: {sorted(unknown)}")


def _parse_authenticated_updates(
    values: Mapping[str, AuthenticatedUpdate | Mapping[str, Any]],
    *,
    registered_arguments: tuple[str, ...],
    trusted_sources: set[str],
    trusted_update_sources: set[str],
) -> dict[str, AuthenticatedUpdate]:
    _validate_argument_keys(values, registered_arguments, label="authenticated_updates")
    parsed: dict[str, AuthenticatedUpdate] = {}
    for argument, raw_update in values.items():
        update = (
            raw_update
            if isinstance(raw_update, AuthenticatedUpdate)
            else AuthenticatedUpdate.model_validate(raw_update)
        )
        source = update.source.strip().lower()
        if source not in trusted_sources:
            raise ValueError(
                f"authenticated update source {source!r} is not trusted for this action"
            )
        if source not in trusted_update_sources:
            raise ValueError(
                f"source {source!r} is not an authenticated update source for this action"
            )
        parsed[argument] = update.model_copy(update={"source": source})
    return parsed


def _missing_attribution(argument: str, value: str | None) -> ArgumentEvidenceMapping:
    return ArgumentEvidenceMapping(
        argument=argument,
        value=value,
        normalized_value=value,
        status=MappingStatus.MISSING,
        reason="No mapping record was supplied for this critical argument.",
    )


def _user_authorization_matches(
    action: ActionType,
    argument: str,
    proposed_value: str | None,
    user_authorized_arguments: Mapping[str, Any],
) -> bool:
    if proposed_value is None:
        return False
    authorized_value = user_authorized_arguments.get(argument)
    if not isinstance(authorized_value, str):
        return False
    try:
        return normalize_argument_value(
            action, argument, proposed_value
        ) == normalize_argument_value(action, argument, authorized_value)
    except (TypeError, ValueError):
        return False


def _assess_argument(
    *,
    action: ActionType,
    argument: str,
    value: str | None,
    attribution: ArgumentEvidenceMapping,
    action_policy: Mapping[str, Any],
    trusted_sources: set[str],
    minimum_confidence: float,
    reference_arguments: Mapping[str, Any],
    trusted_reference_arguments: Mapping[str, Any],
    user_authorized_arguments: Mapping[str, Any],
    authenticated_updates: Mapping[str, AuthenticatedUpdate],
    provenance_mode: GateProvenanceMode,
) -> tuple[ArgumentGateAssessment, str]:
    comparison_reference = (
        reference_arguments if argument in reference_arguments else trusted_reference_arguments
    )
    conflict = _reference_conflict(action, argument, value, comparison_reference)
    model_source = (
        attribution.model_source_estimate.strip().lower()
        if attribution.model_source_estimate is not None
        else None
    )
    model_source_in_trusted_registry = (
        model_source in trusted_sources if model_source is not None else False
    )
    user_corroborated = _user_authorization_matches(
        action,
        argument,
        value,
        user_authorized_arguments,
    )
    trusted_reference_match = argument in trusted_reference_arguments and _argument_value_matches(
        action,
        argument,
        value,
        trusted_reference_arguments[argument],
        label="trusted reference",
    )
    authenticated_update = authenticated_updates.get(argument)
    authenticated_update_match = authenticated_update is not None and _argument_value_matches(
        action,
        argument,
        value,
        authenticated_update.value,
        label="authenticated update",
    )
    oracle_source = (
        attribution.region_ground_truth_source.strip().lower()
        if (
            provenance_mode is GateProvenanceMode.ORACLE
            and attribution.region_ground_truth_source is not None
        )
        else None
    )
    authorization_basis = AuthorizationBasis.NONE
    trusted = False
    oracle_source_used = False

    if user_corroborated:
        # This authority comes from a separate trusted input parse, never from
        # the model's source label or benchmark ground truth.
        selected_rule = _rule(action_policy, "explicit_user_rule")
        authorization_basis = AuthorizationBasis.EXPLICIT_USER
        trusted = True
    elif attribution.status is not MappingStatus.MATCHED:
        # Independently known values do not turn absent or fabricated visual
        # lineage into valid provenance. Explicit user authority is the only
        # non-visual path allowed to bypass this check.
        selected_rule = _status_rule(action_policy, attribution.status)
    elif provenance_mode is GateProvenanceMode.MODEL_ESTIMATED and model_source is None:
        selected_rule = _rule(action_policy, "missing_source_rule")
    elif provenance_mode is GateProvenanceMode.MODEL_ESTIMATED and (
        attribution.model_source_confidence is None
        or attribution.model_source_confidence < minimum_confidence
    ):
        # A separately known value does not fabricate confidence in the
        # self-reported sensor-to-argument lineage under test.
        selected_rule = _rule(action_policy, "low_confidence_rule")
    elif trusted_reference_match:
        # Equality is checked against an explicitly separate, trusted data
        # channel. ``reference_arguments`` alone is diagnostic and cannot
        # authorize an action because the benchmark also supplies it.
        selected_rule = _rule(action_policy, "trusted_reference_rule")
        authorization_basis = AuthorizationBasis.TRUSTED_REFERENCE
        trusted = True
    elif authenticated_update_match:
        selected_rule = _rule(action_policy, "trusted_update_rule")
        authorization_basis = AuthorizationBasis.AUTHENTICATED_UPDATE
        trusted = True
    else:
        if provenance_mode is GateProvenanceMode.ORACLE:
            # This is the only branch allowed to consult benchmark source
            # annotations. Automatic decisions never reach this value.
            oracle_source_used = True
            source = oracle_source
            source_confidence_sufficient = True
        else:
            source = model_source
            source_confidence_sufficient = (
                attribution.model_source_confidence is not None
                and attribution.model_source_confidence >= minimum_confidence
            )

        if source is None:
            selected_rule = _rule(action_policy, "missing_source_rule")
        elif not source_confidence_sufficient:
            selected_rule = _rule(action_policy, "low_confidence_rule")
        elif source == "explicit_user":
            selected_rule = _rule(action_policy, "uncorroborated_explicit_user_rule")
        elif provenance_mode is GateProvenanceMode.MODEL_ESTIMATED:
            # A model can describe a source as trusted-looking; it cannot
            # authenticate that source. Even a high-confidence trusted label is
            # therefore escalated unless a separate authority channel matched.
            selected_rule = _rule(
                action_policy,
                (
                    "uncorroborated_trusted_source_rule"
                    if model_source_in_trusted_registry
                    else "untrusted_source_rule"
                ),
            )
        elif source in trusted_sources:
            trusted = True
            authorization_basis = AuthorizationBasis.ORACLE_SOURCE
            trusted_update_sources = {
                str(item).strip().lower()
                for item in action_policy.get("trusted_update_sources", [])
            }
            if conflict is True:
                selected_rule = _rule(
                    action_policy,
                    (
                        "oracle_trusted_update_rule"
                        if source in trusted_update_sources
                        else "oracle_trusted_conflict_rule"
                    ),
                )
            else:
                selected_rule = _rule(action_policy, "oracle_trusted_source_rule")
        else:
            selected_rule = _rule(action_policy, "untrusted_source_rule")

    decision = Decision(selected_rule["decision"])
    assessment = ArgumentGateAssessment(
        argument_name=argument,
        argument_value=value,
        evidence_status=attribution.status,
        matched_region_id=attribution.selected_region_id,
        model_source_estimate=model_source,
        model_source_confidence=attribution.model_source_confidence,
        model_source_in_trusted_registry=model_source_in_trusted_registry,
        conflict_with_reference=conflict,
        user_authorization_corroborated=user_corroborated,
        trusted_reference_match=trusted_reference_match,
        authenticated_update_match=authenticated_update_match,
        authenticated_update_source=(
            authenticated_update.source if authenticated_update is not None else None
        ),
        oracle_source_used=oracle_source_used,
        authorization_basis=authorization_basis,
        trusted_source=trusted,
        decision=decision,
        policy_rule_id=str(selected_rule["id"]),
    )
    return assessment, str(selected_rule["message"])


def evaluate_thin_gate(
    action: ProposedAction | Mapping[str, Any],
    evidence_map: ActionEvidenceMap | Mapping[str, Any],
    *,
    registry: Mapping[str, Any] | str | Path | None = None,
    policy: Mapping[str, Any] | str | Path | None = None,
    reference_arguments: Mapping[str, Any] | None = None,
    trusted_reference_arguments: Mapping[str, Any] | None = None,
    user_authorized_arguments: Mapping[str, Any] | None = None,
    authenticated_updates: Mapping[str, AuthenticatedUpdate | Mapping[str, Any]] | None = None,
    provenance_mode: GateProvenanceMode | str = GateProvenanceMode.MODEL_ESTIMATED,
) -> ThinGateDecision:
    """Return an auditable, dry-run decision without invoking an LLM.

    ``user_authorized_arguments`` must be derived deterministically from the
    trusted user-input channel. Model-estimated ``explicit_user`` is never enough
    to authorize an action by itself. Likewise, ``trusted_reference_arguments``
    and ``authenticated_updates`` must come from separately authenticated data
    channels. ``reference_arguments`` is comparison-only and cannot authorize.

    Region ground-truth source annotations are ignored unless the caller opts
    into ``ORACLE_REGION_PROVENANCE`` for the experiment-only oracle arm.
    """

    proposed = _normalize_action_input(action)
    mapped = (
        evidence_map
        if isinstance(evidence_map, ActionEvidenceMap)
        else ActionEvidenceMap.model_validate(evidence_map)
    )
    if mapped.action is not proposed.action:
        raise ValueError("evidence map action does not match proposed action")
    registry_config = load_action_registry(registry)
    policy_config = load_thin_gate_policy(policy)
    action_policy = policy_config["actions"][proposed.action.value]
    parsed_provenance_mode = _parse_provenance_mode(provenance_mode)

    if proposed.action is ActionType.NONE:
        selected_rule = _rule(action_policy, "no_action_rule")
        return ThinGateDecision(
            decision=Decision(selected_rule["decision"]),
            action=proposed.action,
            static_effects=[],
            reversibility="high",
            default_risk="low",
            policy_rules_triggered=[str(selected_rule["id"])],
            user_message=str(selected_rule["message"]),
            policy_version=str(policy_config["policy_version"]),
            registry_version=str(registry_config["registry_version"]),
            provenance_mode=parsed_provenance_mode,
        )

    registry_action = registry_config["actions"].get(proposed.action.value)
    if not isinstance(registry_action, Mapping):
        raise ValueError(f"Action registry is missing {proposed.action.value}")
    registered_arguments = tuple(
        str(item) for item in registry_action.get("critical_arguments", [])
    )
    if registered_arguments != CRITICAL_ARGUMENTS[proposed.action]:
        raise ValueError("Action registry critical arguments do not match the action schema")
    trusted_sources = {
        str(item).strip().lower() for item in registry_action.get("trusted_sources", [])
    }
    if not trusted_sources:
        raise ValueError(f"{proposed.action.value} has no trusted sources in the registry")
    trusted_update_sources = {
        str(item).strip().lower() for item in action_policy.get("trusted_update_sources", [])
    }
    if not trusted_update_sources.issubset(trusted_sources):
        raise ValueError(
            f"{proposed.action.value} trusted update sources must also be registry-trusted"
        )

    diagnostic_references = {} if reference_arguments is None else reference_arguments
    trusted_references = {} if trusted_reference_arguments is None else trusted_reference_arguments
    trusted_user_arguments = {} if user_authorized_arguments is None else user_authorized_arguments
    raw_authenticated_updates = {} if authenticated_updates is None else authenticated_updates
    for label, values in (
        ("reference_arguments", diagnostic_references),
        ("trusted_reference_arguments", trusted_references),
        ("user_authorized_arguments", trusted_user_arguments),
    ):
        if not isinstance(values, Mapping):
            raise TypeError(f"{label} must be a mapping")
        _validate_argument_keys(values, registered_arguments, label=label)
    if not isinstance(raw_authenticated_updates, Mapping):
        raise TypeError("authenticated_updates must be a mapping")
    parsed_authenticated_updates = _parse_authenticated_updates(
        raw_authenticated_updates,
        registered_arguments=registered_arguments,
        trusted_sources=trusted_sources,
        trusted_update_sources=trusted_update_sources,
    )

    critical_arguments = critical_arguments_for(proposed)
    threshold = float(
        action_policy.get(
            "minimum_source_confidence",
            policy_config["minimum_source_confidence"],
        )
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("action minimum_source_confidence must be between 0 and 1")

    assessments: dict[str, ArgumentGateAssessment] = {}
    messages: dict[str, str] = {}
    for argument in registered_arguments:
        value = critical_arguments.get(argument)
        attribution = mapped.arguments.get(argument) or _missing_attribution(argument, value)
        if attribution.argument != argument:
            raise ValueError(f"evidence mapping key {argument!r} disagrees with its argument field")
        if attribution.normalized_value is not None and value is not None:
            expected_value = normalize_argument_value(proposed.action, argument, value)
            if attribution.normalized_value != expected_value:
                raise ValueError(f"evidence mapping for {argument!r} belongs to a different value")
        assessment, message = _assess_argument(
            action=proposed.action,
            argument=argument,
            value=value,
            attribution=attribution,
            action_policy=action_policy,
            trusted_sources=trusted_sources,
            minimum_confidence=threshold,
            reference_arguments=diagnostic_references,
            trusted_reference_arguments=trusted_references,
            user_authorized_arguments=trusted_user_arguments,
            authenticated_updates=parsed_authenticated_updates,
            provenance_mode=parsed_provenance_mode,
        )
        assessments[argument] = assessment
        messages[argument] = message

    strictest_argument, strictest = max(
        assessments.items(), key=lambda item: _DECISION_RANK[item[1].decision]
    )
    return ThinGateDecision(
        decision=strictest.decision,
        action=proposed.action,
        critical_arguments=critical_arguments,
        argument_assessments=assessments,
        static_effects=[str(item) for item in registry_action.get("effects", [])],
        reversibility=str(registry_action.get("reversibility", "unknown")),
        default_risk=str(registry_action.get("default_risk", "unknown")),
        policy_rules_triggered=[assessment.policy_rule_id for assessment in assessments.values()],
        user_message=messages[strictest_argument],
        policy_version=str(policy_config["policy_version"]),
        registry_version=str(registry_config["registry_version"]),
        provenance_mode=parsed_provenance_mode,
    )


thin_gate_decision = evaluate_thin_gate


__all__ = [
    "AuthenticatedUpdate",
    "AuthorizationBasis",
    "ArgumentGateAssessment",
    "GateProvenanceMode",
    "ThinGateDecision",
    "evaluate_thin_gate",
    "load_action_registry",
    "load_thin_gate_policy",
    "thin_gate_decision",
]
