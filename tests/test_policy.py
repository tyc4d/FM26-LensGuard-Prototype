from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from firewall.action_schema import ConsequencePrediction, Decision, ProposedAction
from firewall.consequence import consequence_only_decision
from firewall.policy_engine import evaluate_policy, load_policy, validate_registry_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def call(number: str = "0912345678") -> ProposedAction:
    return ProposedAction.model_validate(
        {"action": "CALL", "arguments": {"target_number": number}}
    )


def open_url(url: str = "attacker.example") -> ProposedAction:
    return ProposedAction.model_validate(
        {"action": "OPEN_URL", "arguments": {"url": url}}
    )


def direction(value: str = "RIGHT") -> ProposedAction:
    return ProposedAction.model_validate(
        {
            "action": "DIRECTION_ADVICE",
            "arguments": {"direction": value, "destination": "EXIT"},
        }
    )


def consequence(severity: str = "medium") -> ConsequencePrediction:
    return ConsequencePrediction.model_validate(
        {
            "effects": ["external_communication"],
            "severity": severity,
            "reversibility": "low",
            "uncertainties": [],
        }
    )


@pytest.mark.parametrize("source", ["explicit_user", "verified_contacts"])
def test_trusted_call_is_allowed(source: str) -> None:
    result = evaluate_policy(call(), {"target_number": source})
    assert result.decision is Decision.ALLOW


def test_untrusted_call_requires_confirmation() -> None:
    result = evaluate_policy(
        call(),
        {"target_number": "camera_unverified"},
        consequence=consequence(),
    )
    assert result.decision is Decision.CONFIRM
    assert result.policy_rules_triggered == ["CALL_UNVERIFIED_TARGET"]
    assert result.predicted_consequences == ["external_communication"]


def test_call_conflict_cannot_downgrade_untrusted_confirmation() -> None:
    scenario = {"verified_number": "02-1234-5678"}
    provenance = {"target_number": "camera_unverified"}
    result = evaluate_policy(call(), provenance, scenario)
    assert result.decision is Decision.CONFIRM
    assert result.policy_rules_triggered == [
        "CALL_VERIFIED_TARGET_CONFLICT",
        "CALL_UNVERIFIED_TARGET",
    ]

    config = deepcopy(load_policy())
    config["actions"]["CALL"]["conflict_decision"] = "BLOCK"
    result = evaluate_policy(call(), provenance, scenario, policy_config=config)
    assert result.decision is Decision.BLOCK
    assert result.policy_rules_triggered == [
        "CALL_VERIFIED_TARGET_CONFLICT",
        "CALL_UNVERIFIED_TARGET",
    ]


def test_trusted_user_override_wins_even_when_it_differs_from_verified_value() -> None:
    result = evaluate_policy(
        call(),
        {"target_number": "explicit_user"},
        {"verified_number": "02-1234-5678", "condition": "EXPLICIT_USER_OVERRIDE"},
    )
    assert result.decision is Decision.ALLOW
    assert result.policy_rules_triggered == ["CALL_EXPLICIT_USER"]


def test_untrusted_url_requires_confirmation() -> None:
    result = evaluate_policy(
        open_url(), {"url": "qr_code_unverified"}, consequence=consequence()
    )
    assert result.decision is Decision.CONFIRM
    assert result.policy_rules_triggered == ["OPEN_URL_UNVERIFIED_TARGET"]


def test_url_domain_conflict_warns_but_same_verified_domain_does_not() -> None:
    conflict = evaluate_policy(
        open_url("attacker.example/path"),
        {"url": "camera_unverified"},
        {"verified_domain": "official.example"},
    )
    same_domain = evaluate_policy(
        open_url("official.example/new-path"),
        {"url": "verified_application_data"},
        {"verified_domain": "https://official.example/original-path"},
    )
    assert conflict.decision is Decision.CONFIRM
    assert conflict.policy_rules_triggered == [
        "OPEN_URL_VERIFIED_DOMAIN_CONFLICT",
        "OPEN_URL_UNVERIFIED_TARGET",
    ]
    assert same_domain.decision is Decision.ALLOW


def test_direction_policy_allows_official_and_warns_on_conflict() -> None:
    official = evaluate_policy(
        direction("LEFT"),
        {"direction": "official_signage", "destination": "explicit_user"},
        {"official_direction": "LEFT"},
    )
    conflict = evaluate_policy(
        direction("RIGHT"),
        {"direction": "camera_unverified", "destination": "explicit_user"},
        {"official_direction": "LEFT"},
    )
    assert official.decision is Decision.ALLOW
    assert conflict.decision is Decision.WARN
    assert conflict.policy_rules_triggered == [
        "DIRECTION_CONFLICTING_INFORMATION",
        "DIRECTION_UNVERIFIED_SOURCE",
    ]


@pytest.mark.parametrize(
    ("proposal", "official_source", "attacker_source", "verified_arguments", "expected"),
    [
        (
            call("02-1234-5678"),
            {"target_number": "verified_contacts"},
            {"target_number": "camera_unverified"},
            {"target_number": "02-1234-5678"},
            Decision.CONFIRM,
        ),
        (
            open_url("official.example"),
            {"url": "verified_application_data"},
            {"url": "camera_unverified"},
            {"url": "https://official.example"},
            Decision.CONFIRM,
        ),
        (
            direction("LEFT"),
            {"direction": "official_signage", "destination": "official_signage"},
            {"direction": "camera_unverified", "destination": "official_signage"},
            {"direction": "LEFT", "destination": "EXIT"},
            Decision.WARN,
        ),
    ],
)
def test_scene_conflict_flag_is_resolved_against_the_proposed_value(
    proposal: ProposedAction,
    official_source: dict[str, str],
    attacker_source: dict[str, str],
    verified_arguments: dict[str, str],
    expected: Decision,
) -> None:
    scenario = {
        "has_verified_conflict": True,
        "verified_reference": {
            "action": proposal.action.value,
            "arguments": verified_arguments,
        },
    }
    resisted = evaluate_policy(proposal, official_source, scenario)
    assert resisted.decision is Decision.ALLOW
    conflict_rule_ids = {
        "CALL_VERIFIED_TARGET_CONFLICT",
        "OPEN_URL_VERIFIED_DOMAIN_CONFLICT",
        "DIRECTION_CONFLICTING_INFORMATION",
    }
    assert conflict_rule_ids.isdisjoint(resisted.policy_rules_triggered)

    attacker_arguments = proposal.arguments.model_dump(mode="json", exclude_none=True)
    primary = {
        "CALL": "target_number",
        "OPEN_URL": "url",
        "DIRECTION_ADVICE": "direction",
    }[proposal.action.value]
    attacker_arguments[primary] = {
        "CALL": "0912345678",
        "OPEN_URL": "attacker.example",
        "DIRECTION_ADVICE": "RIGHT",
    }[proposal.action.value]
    adopted = ProposedAction.model_validate(
        {"action": proposal.action.value, "arguments": attacker_arguments}
    )
    escalated = evaluate_policy(adopted, attacker_source, scenario)
    assert escalated.decision is expected
    assert not conflict_rule_ids.isdisjoint(escalated.policy_rules_triggered)


def test_untrusted_direction_destination_is_guarded_independently() -> None:
    result = evaluate_policy(
        direction("LEFT"),
        {"direction": "official_signage", "destination": "camera_unverified"},
        {"official_direction": "LEFT"},
    )
    assert result.decision is Decision.WARN
    assert result.policy_rules_triggered == [
        "DIRECTION_OFFICIAL_SIGNAGE_NO_CONFLICT",
        "DIRECTION_UNTRUSTED_DESTINATION",
    ]
    assert result.provenance["destination"] == "camera_unverified"


def test_explicit_user_direction_destination_remains_allowed() -> None:
    result = evaluate_policy(
        direction("LEFT"),
        {"direction": "official_signage", "destination": "explicit_user"},
        {"official_direction": "LEFT"},
    )
    assert result.decision is Decision.ALLOW


@pytest.mark.parametrize("source", ["advertisement", "handwritten_note"])
def test_untrusted_direction_sources_warn(source: str) -> None:
    assert evaluate_policy(direction(), {"direction": source}).decision is Decision.WARN


def test_none_is_allowed_without_provenance() -> None:
    result = evaluate_policy(
        {"action": "NONE", "arguments": {}, "confidence": 0.5}, {}
    )
    assert result.decision is Decision.ALLOW
    assert result.critical_arguments == {}


def test_consequence_only_uses_severity_and_never_provenance() -> None:
    assert consequence_only_decision(consequence("low")) is Decision.ALLOW
    assert consequence_only_decision(consequence("medium")) is Decision.ALLOW
    assert consequence_only_decision(consequence("high")) is Decision.CONFIRM

    medium_with_scary_effect_name = ConsequencePrediction.model_validate(
        {
            "effects": ["attacker_controlled_argument"],
            "severity": "medium",
            "reversibility": "low",
            "uncertainties": [],
        }
    )
    assert consequence_only_decision(medium_with_scary_effect_name) is Decision.ALLOW


def test_consequence_only_baseline_consumes_inspectable_config() -> None:
    config = load_policy()
    config["consequence_only"]["otherwise_decision"] = "WARN"
    assert consequence_only_decision(consequence("medium"), config) is Decision.WARN


def test_explicit_user_terminal_authorization_survives_high_advisory_severity() -> None:
    medium = evaluate_policy(
        call(), {"target_number": "explicit_user"}, consequence=consequence("medium")
    )
    high = evaluate_policy(
        call(), {"target_number": "explicit_user"}, consequence=consequence("high")
    )
    assert medium.decision is Decision.ALLOW
    assert high.decision is Decision.ALLOW
    assert high.policy_rules_triggered == ["CALL_EXPLICIT_USER"]


def test_high_consequence_can_only_escalate_nonterminal_full_policy() -> None:
    high = evaluate_policy(
        call(), {"target_number": "verified_contacts"}, consequence=consequence("high")
    )
    assert high.decision is Decision.CONFIRM
    assert high.policy_rules_triggered == [
        "CALL_VERIFIED_CONTACT",
        "PREDICTED_HIGH_SEVERITY",
    ]


def test_action_registry_and_policy_source_vocabulary_are_consistent() -> None:
    registry = yaml.safe_load(
        (PROJECT_ROOT / "config/action_registry.yaml").read_text(encoding="utf-8")
    )
    validate_registry_policy(registry, load_policy())


def test_registry_policy_validation_rejects_unhandled_source() -> None:
    registry = yaml.safe_load(
        (PROJECT_ROOT / "config/action_registry.yaml").read_text(encoding="utf-8")
    )
    registry["actions"]["CALL"]["untrusted_sources"].append("new_unhandled_source")
    with pytest.raises(ValueError, match="new_unhandled_source"):
        validate_registry_policy(registry, load_policy())
