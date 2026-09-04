"""Deterministic, YAML-configured policy evaluation.

Gemini may propose an action and predict consequences, but it cannot choose the
decision returned here. Matching policy guards compose monotonically: the most
restrictive decision wins unless an inspectable terminal rule applies.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from .action_normalizer import (
    critical_argument_matches,
    critical_arguments_for,
    normalize_action,
    normalize_url,
)
from .action_schema import (
    ActionType,
    ConsequencePrediction,
    Decision,
    FirewallDecision,
    ProposedAction,
    ProvenanceSource,
)

DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[1] / "config" / "policy.yaml"
_CONFIGURED_CONFLICT = "CONFIGURED_CONFLICT_DECISION"
_SUPPORTED_CONDITIONS = {"always", "source_in", "source_not_in", "conflict"}
_DECISION_RANK = {
    Decision.ALLOW: 0,
    Decision.WARN: 1,
    Decision.CONFIRM: 2,
    Decision.BLOCK: 3,
}


def load_policy(path: str | Path | None = None) -> dict[str, Any]:
    """Load and structurally validate the inspectable YAML policy."""

    policy_path = Path(path) if path is not None else DEFAULT_POLICY_PATH
    try:
        payload = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read policy file: {policy_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid policy YAML: {policy_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("policy must be a YAML mapping")
    if not isinstance(payload.get("policy_version"), str):
        raise ValueError("policy_version must be a string")
    consequence_only = payload.get("consequence_only")
    if not isinstance(consequence_only, dict):
        raise ValueError("policy.consequence_only must be a mapping")
    for field in ("high_severity_decision", "otherwise_decision"):
        try:
            Decision(consequence_only.get(field))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"consequence_only.{field} must be a valid decision") from exc

    consequence_escalation = payload.get("consequence_escalation")
    if not isinstance(consequence_escalation, dict):
        raise ValueError("policy.consequence_escalation must be a mapping")
    severity_decisions = consequence_escalation.get("severity_decisions")
    if not isinstance(severity_decisions, dict):
        raise ValueError("consequence_escalation.severity_decisions must be a mapping")
    for severity, decision in severity_decisions.items():
        if str(severity) not in {"low", "medium", "high"}:
            raise ValueError(f"invalid consequence escalation severity: {severity!r}")
        try:
            Decision(decision)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid consequence escalation decision for {severity!r}"
            ) from exc
    if not isinstance(consequence_escalation.get("rule_id"), str):
        raise ValueError("consequence_escalation.rule_id must be a string")
    if not isinstance(consequence_escalation.get("message"), str):
        raise ValueError("consequence_escalation.message must be a string")
    actions = payload.get("actions")
    if not isinstance(actions, dict):
        raise ValueError("policy.actions must be a mapping")

    for action in ActionType:
        action_policy = actions.get(action.value)
        if not isinstance(action_policy, dict):
            raise ValueError(f"policy is missing action {action.value}")
        rules = action_policy.get("rules")
        if not isinstance(rules, list) or not rules:
            raise ValueError(f"{action.value} policy must contain at least one rule")
        seen_ids: set[str] = set()
        for rule in rules:
            if not isinstance(rule, dict):
                raise ValueError(f"{action.value} policy contains a non-mapping rule")
            rule_id = rule.get("id")
            if not isinstance(rule_id, str) or not rule_id:
                raise ValueError(f"{action.value} rule is missing an id")
            if rule_id in seen_ids:
                raise ValueError(f"duplicate policy rule id: {rule_id}")
            seen_ids.add(rule_id)
            if not isinstance(rule.get("priority"), int):
                raise ValueError(f"{rule_id}.priority must be an integer")
            if "terminal" in rule and not isinstance(rule["terminal"], bool):
                raise ValueError(f"{rule_id}.terminal must be a boolean")
            conditions = rule.get("when")
            if not isinstance(conditions, dict) or not conditions:
                raise ValueError(f"{rule_id}.when must be a non-empty mapping")
            unsupported = set(conditions) - _SUPPORTED_CONDITIONS
            if unsupported:
                raise ValueError(
                    f"{rule_id} has unsupported conditions: {sorted(unsupported)}"
                )
            raw_decision = rule.get("decision")
            if raw_decision != _CONFIGURED_CONFLICT:
                try:
                    Decision(raw_decision)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{rule_id} has an invalid decision") from exc
            elif "conflict_decision" not in action_policy:
                raise ValueError(
                    f"{rule_id} uses configured conflict decision, but none is set"
                )
            if not isinstance(rule.get("message"), str) or not rule["message"].strip():
                raise ValueError(f"{rule_id}.message must be a non-empty string")

        try:
            Decision(action_policy.get("conflict_decision"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{action.value}.conflict_decision must be a valid decision"
            ) from exc

        argument_guards = action_policy.get("argument_guards", {})
        if not isinstance(argument_guards, dict):
            raise ValueError(f"{action.value}.argument_guards must be a mapping")
        for argument, guard in argument_guards.items():
            if not isinstance(argument, str) or not argument:
                raise ValueError(f"{action.value} has an invalid guarded argument")
            if not isinstance(guard, dict):
                raise ValueError(f"{action.value}.{argument} guard must be a mapping")
            if not isinstance(guard.get("id"), str) or not guard["id"]:
                raise ValueError(f"{action.value}.{argument} guard is missing an id")
            trusted_sources = guard.get("trusted_sources")
            if not isinstance(trusted_sources, list) or not trusted_sources:
                raise ValueError(
                    f"{action.value}.{argument}.trusted_sources must be a non-empty list"
                )
            for source in trusted_sources:
                try:
                    ProvenanceSource(str(source))
                except ValueError as exc:
                    raise ValueError(
                        f"{action.value}.{argument} guard has invalid source {source!r}"
                    ) from exc
            try:
                Decision(guard.get("decision"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{action.value}.{argument} guard has an invalid decision"
                ) from exc
            if not isinstance(guard.get("message"), str) or not guard["message"].strip():
                raise ValueError(f"{action.value}.{argument} guard needs a message")
    return deepcopy(payload)


def validate_registry_policy(
    registry: Mapping[str, Any], policy: Mapping[str, Any]
) -> None:
    """Reject registry/policy drift that could corrupt event labels or decisions."""

    if not isinstance(registry.get("registry_version"), str):
        raise ValueError("action registry must declare a string registry_version")
    registry_actions = registry.get("actions")
    policy_actions = policy.get("actions")
    if not isinstance(registry_actions, Mapping) or not isinstance(policy_actions, Mapping):
        raise ValueError("registry and policy must each contain an actions mapping")

    for action_name, registry_action in registry_actions.items():
        if not isinstance(registry_action, Mapping):
            raise ValueError(f"registry action {action_name} must be a mapping")
        action_policy = policy_actions.get(action_name)
        if not isinstance(action_policy, Mapping):
            raise ValueError(f"policy is missing registry action {action_name}")

        primary = action_policy.get("critical_argument")
        guards = action_policy.get("argument_guards", {})
        policy_arguments = ({str(primary)} if primary else set()) | set(guards)
        registry_arguments = {
            str(item) for item in registry_action.get("critical_arguments", [])
        }
        if policy_arguments != registry_arguments:
            raise ValueError(
                f"{action_name} critical arguments differ between registry "
                f"{sorted(registry_arguments)} and policy {sorted(policy_arguments)}"
            )

        trusted = {str(item) for item in registry_action.get("trusted_sources", [])}
        untrusted = {str(item) for item in registry_action.get("untrusted_sources", [])}
        if not trusted or not untrusted or trusted & untrusted:
            raise ValueError(f"{action_name} registry source sets are empty or overlap")

        source_decisions: dict[str, set[Decision]] = {}
        for rule in action_policy.get("rules", []):
            if not isinstance(rule, Mapping):
                continue
            conditions = rule.get("when")
            if not isinstance(conditions, Mapping) or "source_in" not in conditions:
                continue
            raw_decision = rule.get("decision")
            if raw_decision == _CONFIGURED_CONFLICT:
                raw_decision = action_policy.get("conflict_decision")
            decision = Decision(raw_decision)
            for source in conditions["source_in"]:
                source_decisions.setdefault(str(source), set()).add(decision)

        unknown_policy_sources = set(source_decisions) - trusted - untrusted
        if unknown_policy_sources:
            raise ValueError(
                f"{action_name} policy sources are absent from the registry: "
                f"{sorted(unknown_policy_sources)}"
            )
        for source in trusted:
            if Decision.ALLOW not in source_decisions.get(source, set()):
                raise ValueError(
                    f"{action_name} trusted source {source!r} has no explicit ALLOW rule"
                )
        default_untrusted = Decision(registry_action.get("default_untrusted_decision"))
        for source in untrusted:
            if default_untrusted not in source_decisions.get(source, set()):
                raise ValueError(
                    f"{action_name} untrusted source {source!r} has no explicit "
                    f"{default_untrusted.value} rule"
                )
        if Decision(registry_action.get("conflict_decision")) != Decision(
            action_policy.get("conflict_decision")
        ):
            raise ValueError(f"{action_name} conflict decision differs from the registry")


def _as_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json", exclude_none=True)
        if isinstance(dumped, dict) and "sources" in dumped:
            dumped = dumped["sources"]
        return dict(dumped)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _source_value(value: Any) -> str:
    if isinstance(value, ProvenanceSource):
        return value.value
    if value is None:
        return ProvenanceSource.UNKNOWN_VISUAL_SOURCE.value
    return str(value).strip().lower()


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return None


def _flag_for_action(value: Any, action: ActionType, critical_name: str) -> bool | None:
    direct = _bool_value(value)
    if direct is not None:
        return direct
    if isinstance(value, Mapping):
        for key in (action.value, action.value.lower(), critical_name):
            if key in value:
                result = _bool_value(value[key])
                if result is not None:
                    return result
    return None


def _extract_verified_value(
    scenario: Mapping[str, Any], action: ActionType, critical_name: str
) -> Any:
    nested_fields = (
        "verified_arguments",
        "official_arguments",
    )
    for field in nested_fields:
        nested = scenario.get(field)
        if isinstance(nested, Mapping) and critical_name in nested:
            return nested[critical_name]

    ground_truth_action = scenario.get("ground_truth_action")
    if isinstance(ground_truth_action, Mapping):
        arguments = ground_truth_action.get("arguments")
        if isinstance(arguments, Mapping) and critical_name in arguments:
            return arguments[critical_name]

    verified_reference = scenario.get("verified_reference")
    if isinstance(verified_reference, Mapping):
        arguments = verified_reference.get("arguments")
        if isinstance(arguments, Mapping) and critical_name in arguments:
            return arguments[critical_name]

    ground_truth_arguments = scenario.get("ground_truth_arguments")
    if isinstance(ground_truth_arguments, Mapping) and critical_name in ground_truth_arguments:
        return ground_truth_arguments[critical_name]

    scalar_fields = (
        "verified_argument",
        "verified_value",
        "official_argument",
        "official_value",
        "ground_truth_argument",
    )
    for field in scalar_fields:
        if scenario.get(field) is not None:
            return scenario[field]

    if action is ActionType.OPEN_URL and scenario.get("verified_domain") is not None:
        return scenario["verified_domain"]
    if action is ActionType.CALL and scenario.get("verified_number") is not None:
        return scenario["verified_number"]
    if action is ActionType.DIRECTION_ADVICE and scenario.get("official_direction") is not None:
        return scenario["official_direction"]
    return None


def _domains_match(proposed: str, verified: Any) -> bool:
    if isinstance(verified, Mapping):
        verified = verified.get("url") or verified.get("domain")
    if not isinstance(verified, str):
        return False
    try:
        proposed_host = urlsplit(normalize_url(proposed)).hostname
        verified_host = urlsplit(normalize_url(verified)).hostname
    except (TypeError, ValueError):
        return False
    return proposed_host == verified_host


def _has_verified_conflict(
    action: ActionType,
    critical_arguments: Mapping[str, str],
    scenario: Mapping[str, Any],
    critical_name: str,
) -> bool:
    # Prefer a value-level comparison whenever the scenario supplies a verified
    # reference. Scene flags describe whether conflicting content is present,
    # not whether the model's selected argument conflicts. Using a scene flag
    # first would falsely warn a model that resisted the alternate and selected
    # the verified value.
    proposed_value = critical_arguments.get(critical_name)
    verified_value = _extract_verified_value(scenario, action, critical_name)
    if proposed_value is not None and verified_value is not None:
        if action is ActionType.OPEN_URL:
            return not _domains_match(proposed_value, verified_value)
        return not critical_argument_matches(
            action, critical_arguments, {critical_name: verified_value}
        )

    # Explicit flags remain a compatibility fallback for compact scenarios that
    # genuinely do not provide a comparable verified argument.
    flag_fields = (
        "verified_conflict",
        "verified_source_conflict",
        "has_verified_conflict",
        "target_conflicts_verified_source",
        "domain_conflicts_verified_source",
        "conflicting_official_information",
        "conflict",
    )
    for field in flag_fields:
        if field in scenario:
            result = _flag_for_action(scenario[field], action, critical_name)
            if result is not None:
                return result

    return False


def _condition_matches(
    conditions: Mapping[str, Any], *, source: str, conflict: bool
) -> bool:
    if "always" in conditions and _bool_value(conditions["always"]) is not True:
        return False
    if "source_in" in conditions:
        allowed = {_source_value(item) for item in conditions["source_in"]}
        if source not in allowed:
            return False
    if "source_not_in" in conditions:
        denied = {_source_value(item) for item in conditions["source_not_in"]}
        if source in denied:
            return False
    if "conflict" in conditions:
        expected = _bool_value(conditions["conflict"])
        if expected is None or conflict is not expected:
            return False
    return True


class _FormatContext(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def evaluate_policy(
    action: ProposedAction | Mapping[str, Any],
    provenance: Mapping[str, Any] | Any,
    scenario: Mapping[str, Any] | Any | None = None,
    consequence: ConsequencePrediction | Mapping[str, Any] | None = None,
    policy_config: Mapping[str, Any] | str | Path | None = None,
) -> FirewallDecision:
    """Compose matching deterministic guards and return an auditable decision."""

    normalized = normalize_action(action)
    critical_arguments = critical_arguments_for(normalized)
    provenance_map = {
        str(key): _source_value(value)
        for key, value in _as_mapping(provenance, name="provenance").items()
    }
    scenario_map = _as_mapping(scenario, name="scenario")
    if policy_config is None or isinstance(policy_config, (str, Path)):
        config = load_policy(policy_config)
    else:
        config = deepcopy(dict(policy_config))

    prediction: ConsequencePrediction | None
    if consequence is None:
        prediction = None
    elif isinstance(consequence, ConsequencePrediction):
        prediction = consequence
    else:
        prediction = ConsequencePrediction.model_validate(consequence)

    actions = config.get("actions")
    if not isinstance(actions, Mapping):
        raise ValueError("policy_config.actions must be a mapping")
    action_policy = actions.get(normalized.action.value)
    if not isinstance(action_policy, Mapping):
        raise ValueError(f"policy has no rules for {normalized.action.value}")

    critical_name = action_policy.get("critical_argument")
    if normalized.action is ActionType.NONE:
        critical_name = None
        source = "none"
        conflict = False
    else:
        if not isinstance(critical_name, str) or critical_name not in critical_arguments:
            raise ValueError(
                f"policy critical argument is missing from {normalized.action.value}"
            )
        source = _source_value(provenance_map.get(critical_name))
        provenance_map.setdefault(critical_name, source)
        conflict = _has_verified_conflict(
            normalized.action,
            critical_arguments,
            scenario_map,
            critical_name,
        )

    rules = action_policy.get("rules")
    if not isinstance(rules, list):
        raise ValueError(f"{normalized.action.value} policy rules must be a list")
    indexed_rules = list(enumerate(rules))
    indexed_rules.sort(key=lambda item: (-int(item[1].get("priority", 0)), item[0]))

    matching: list[Mapping[str, Any]] = []
    for _, rule in indexed_rules:
        conditions = rule.get("when", {})
        if isinstance(conditions, Mapping) and _condition_matches(
            conditions, source=source, conflict=conflict
        ):
            matching.append(rule)
    if not matching:
        raise ValueError(f"no deterministic rule matched {normalized.action.value}")

    # An `always` rule is a fallback, not an independent signal. Terminal rules
    # make explicit-user precedence visible in YAML instead of depending on
    # incidental list order. Otherwise every matching independent signal is
    # retained and the strictest decision wins; a conflict can never downgrade
    # an untrusted-source CONFIRM to WARN.
    specific = [
        rule
        for rule in matching
        if not (
            isinstance(rule.get("when"), Mapping)
            and set(rule["when"]) == {"always"}
            and _bool_value(rule["when"]["always"]) is True
        )
    ]
    applicable = specific or matching
    terminal = [rule for rule in applicable if rule.get("terminal") is True]
    terminal_matched = bool(terminal)
    if terminal_matched:
        applicable = terminal

    resolved: list[tuple[Mapping[str, Any], Decision]] = []
    for rule in applicable:
        raw_rule_decision = rule.get("decision")
        if raw_rule_decision == _CONFIGURED_CONFLICT:
            raw_rule_decision = action_policy.get("conflict_decision")
        try:
            resolved.append((rule, Decision(raw_rule_decision)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"rule {rule.get('id')} has an invalid decision") from exc

    selected, decision = max(
        resolved,
        key=lambda item: (
            _DECISION_RANK[item[1]],
            int(item[0].get("priority", 0)),
        ),
    )

    primary_value = critical_arguments.get(str(critical_name), "")
    message_context = _FormatContext(
        action=normalized.action.value,
        critical_argument=str(critical_name or ""),
        value=primary_value,
        source=source,
    )
    message = str(selected["message"]).format_map(message_context)
    triggered_rules = [str(rule["id"]) for rule, _ in resolved]

    argument_guards = action_policy.get("argument_guards", {})
    if not isinstance(argument_guards, Mapping):
        raise ValueError(f"{normalized.action.value}.argument_guards must be a mapping")
    for argument, guard_value in argument_guards.items():
        if argument not in critical_arguments or not isinstance(guard_value, Mapping):
            continue
        argument_source = _source_value(provenance_map.get(argument))
        provenance_map.setdefault(argument, argument_source)
        trusted_sources = {
            _source_value(item) for item in guard_value.get("trusted_sources", [])
        }
        if argument_source in trusted_sources:
            continue
        guard_decision = Decision(guard_value.get("decision"))
        triggered_rules.append(str(guard_value.get("id")))
        if _DECISION_RANK[guard_decision] > _DECISION_RANK[decision]:
            decision = guard_decision
            guard_context = _FormatContext(
                action=normalized.action.value,
                critical_argument=str(argument),
                value=critical_arguments[argument],
                source=argument_source,
            )
            message = str(guard_value.get("message", "")).format_map(guard_context)

    consequence_escalation = config.get("consequence_escalation", {})
    severity_decisions = consequence_escalation.get("severity_decisions", {})
    if (
        prediction is not None
        and normalized.action is not ActionType.NONE
        and not terminal_matched
        and prediction.severity.value in severity_decisions
    ):
        consequence_decision = Decision(severity_decisions[prediction.severity.value])
        triggered_rules.append(str(consequence_escalation.get("rule_id")))
        if _DECISION_RANK[consequence_decision] > _DECISION_RANK[decision]:
            decision = consequence_decision
            message = str(consequence_escalation.get("message", "")).format_map(
                _FormatContext(
                    action=normalized.action.value,
                    severity=prediction.severity.value,
                    critical_argument=str(critical_name or ""),
                    value=primary_value,
                    source=source,
                )
            )
    relevant_provenance = {
        name: provenance_map[name]
        for name in critical_arguments
        if name in provenance_map
    }

    return FirewallDecision(
        decision=decision,
        action=normalized.action,
        critical_arguments=critical_arguments,
        provenance=relevant_provenance,
        predicted_consequences=prediction.effects if prediction else [],
        policy_rules_triggered=triggered_rules,
        user_message=message,
    )
