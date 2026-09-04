from copy import deepcopy
from pathlib import Path

import pytest

from firewall.action_schema import Decision
from firewall.thin_gate import (
    AuthorizationBasis,
    GateProvenanceMode,
    evaluate_thin_gate,
    load_thin_gate_policy,
)
from phase2_schema import ActionOnlyOutput
from provenance import map_action_evidence, map_provider_argument_evidence

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def call(number: str = "0912345678") -> dict:
    return {"action": "CALL", "arguments": {"target_number": number}}


def open_url(url: str = "https://updated.example") -> dict:
    return {"action": "OPEN_URL", "arguments": {"url": url}}


def direction() -> dict:
    return {
        "action": "DIRECTION_ADVICE",
        "arguments": {"direction": "RIGHT", "destination": "EXIT"},
    }


def region(
    text: str,
    *,
    source: str | None,
    confidence: float | None = 0.95,
    ground_truth: str | None = None,
) -> list[dict]:
    return [
        {
            "region_id": "support",
            "text": text,
            "model_source_estimate": source,
            "model_source_confidence": confidence,
            "region_ground_truth_source": ground_truth,
        }
    ]


def test_untrusted_call_without_ground_truth_requires_confirmation() -> None:
    action = call()
    evidence = map_action_evidence(
        action, region("0912345678", source="camera_unverified", ground_truth=None)
    )
    result = evaluate_thin_gate(action, evidence)

    assert result.decision is Decision.CONFIRM
    assert result.policy_rules_triggered == ["PHASE2_CALL_UNTRUSTED_SOURCE"]
    assert result.static_effects == [
        "external_communication",
        "caller_identity_disclosure",
        "possible_social_engineering",
        "possible_call_charge",
    ]
    assert result.reversibility == "low"
    assert result.default_risk == "medium"
    assert result.dry_run is True


def test_trusted_looking_model_label_cannot_authorize_automatic_action() -> None:
    action = call()
    trusted_ground_truth = map_action_evidence(
        action,
        region(
            "0912345678",
            source="verified_contacts",
            ground_truth="verified_contacts",
        ),
    )
    untrusted_ground_truth = map_action_evidence(
        action,
        region(
            "0912345678",
            source="verified_contacts",
            ground_truth="camera_unverified",
        ),
    )

    trusted_gt_result = evaluate_thin_gate(action, trusted_ground_truth)
    untrusted_gt_result = evaluate_thin_gate(action, untrusted_ground_truth)

    assert trusted_gt_result == untrusted_gt_result
    assert trusted_gt_result.decision is Decision.CONFIRM
    assert trusted_gt_result.policy_rules_triggered == ["PHASE2_CALL_TRUSTED_LABEL_UNCORROBORATED"]
    assessment = trusted_gt_result.argument_assessments["target_number"]
    assert assessment.model_source_in_trusted_registry is True
    assert assessment.trusted_source is False
    assert assessment.authorization_basis is AuthorizationBasis.NONE
    assert assessment.oracle_source_used is False
    assert "region_ground_truth_source" not in trusted_gt_result.model_dump_json()


def test_oracle_source_labels_require_explicit_mode() -> None:
    action = call()
    evidence = map_action_evidence(
        action,
        region(
            "0912345678",
            source="camera_unverified",
            ground_truth="verified_contacts",
        ),
    )

    automatic = evaluate_thin_gate(action, evidence)
    oracle = evaluate_thin_gate(
        action,
        evidence,
        provenance_mode=GateProvenanceMode.ORACLE,
    )

    assert automatic.decision is Decision.CONFIRM
    assert automatic.policy_rules_triggered == ["PHASE2_CALL_UNTRUSTED_SOURCE"]
    assert automatic.argument_assessments["target_number"].oracle_source_used is False
    assert oracle.decision is Decision.ALLOW
    assert oracle.policy_rules_triggered == ["PHASE2_CALL_ORACLE_TRUSTED_SOURCE"]
    oracle_assessment = oracle.argument_assessments["target_number"]
    assert oracle_assessment.oracle_source_used is True
    assert oracle_assessment.authorization_basis is AuthorizationBasis.ORACLE_SOURCE
    assert oracle_assessment.trusted_source is True
    assert oracle.provenance_mode == GateProvenanceMode.ORACLE.value


def test_oracle_uses_untrusted_ground_truth_instead_of_trusted_model_label() -> None:
    action = call()
    evidence = map_action_evidence(
        action,
        region(
            "0912345678",
            source="verified_contacts",
            ground_truth="camera_unverified",
        ),
    )
    result = evaluate_thin_gate(
        action,
        evidence,
        provenance_mode="ORACLE_REGION_PROVENANCE",
    )
    assert result.decision is Decision.CONFIRM
    assert result.policy_rules_triggered == ["PHASE2_CALL_UNTRUSTED_SOURCE"]
    assert result.argument_assessments["target_number"].trusted_source is False


def test_explicit_user_authority_is_preserved_despite_old_reference() -> None:
    action = call("0912345678")
    evidence = map_action_evidence(
        action, region("0912345678", source="explicit_user", confidence=1.0)
    )
    result = evaluate_thin_gate(
        action,
        evidence,
        reference_arguments={"target_number": "02-1234-5678"},
        user_authorized_arguments={"target_number": "0912-345-678"},
    )
    assert result.decision is Decision.ALLOW
    assert result.policy_rules_triggered == ["PHASE2_CALL_EXPLICIT_USER"]
    assert result.argument_assessments["target_number"].conflict_with_reference is True
    assert result.argument_assessments["target_number"].user_authorization_corroborated is True


def test_model_explicit_user_claim_without_trusted_corroboration_escalates() -> None:
    action = call("0912345678")
    evidence = map_action_evidence(
        action,
        region("0912345678", source="explicit_user", confidence=1.0),
    )
    result = evaluate_thin_gate(action, evidence)
    assert result.decision is Decision.CONFIRM
    assert result.policy_rules_triggered == ["PHASE2_CALL_EXPLICIT_USER_UNCORROBORATED"]
    assert result.argument_assessments["target_number"].user_authorization_corroborated is False
    wrong_value = evaluate_thin_gate(
        action,
        evidence,
        user_authorized_arguments={"target_number": "0912345679"},
    )
    assert wrong_value.decision is Decision.CONFIRM
    assert wrong_value.policy_rules_triggered == ["PHASE2_CALL_EXPLICIT_USER_UNCORROBORATED"]


def test_trusted_user_authorization_does_not_depend_on_model_source_guess() -> None:
    action = call("0912345678")
    evidence = map_action_evidence(
        action,
        region("0912345678", source="camera_unverified", confidence=0.2),
    )
    result = evaluate_thin_gate(
        action,
        evidence,
        user_authorized_arguments={"target_number": "0912345678"},
    )
    assert result.decision is Decision.ALLOW
    assert result.policy_rules_triggered == ["PHASE2_CALL_EXPLICIT_USER"]


def test_conflicting_trusted_looking_update_requires_separate_authentication() -> None:
    action = open_url()
    evidence = map_action_evidence(
        action,
        region(
            "https://updated.example",
            source="verified_application_data",
            confidence=0.99,
        ),
    )
    uncorroborated = evaluate_thin_gate(
        action,
        evidence,
        reference_arguments={"url": "https://old.example"},
    )
    authenticated = evaluate_thin_gate(
        action,
        evidence,
        reference_arguments={"url": "https://old.example"},
        authenticated_updates={
            "url": {
                "value": "https://updated.example",
                "source": "verified_application_data",
            }
        },
    )

    assert uncorroborated.decision is Decision.CONFIRM
    assert uncorroborated.policy_rules_triggered == ["PHASE2_URL_TRUSTED_LABEL_UNCORROBORATED"]
    assert authenticated.decision is Decision.ALLOW
    assert authenticated.policy_rules_triggered == ["PHASE2_URL_TRUSTED_UPDATE"]
    assessment = authenticated.argument_assessments["url"]
    assert assessment.authenticated_update_match is True
    assert assessment.authenticated_update_source == "verified_application_data"
    assert assessment.authorization_basis is AuthorizationBasis.AUTHENTICATED_UPDATE
    assert assessment.trusted_source is True


def test_only_exact_authenticated_update_value_is_authoritative() -> None:
    action = open_url()
    evidence = map_action_evidence(
        action,
        region(
            "https://updated.example",
            source="verified_application_data",
            confidence=0.99,
        ),
    )
    result = evaluate_thin_gate(
        action,
        evidence,
        authenticated_updates={
            "url": {
                "value": "https://different.example",
                "source": "verified_application_data",
            }
        },
    )
    assert result.decision is Decision.CONFIRM
    assessment = result.argument_assessments["url"]
    assert assessment.authenticated_update_match is False
    assert assessment.authorization_basis is AuthorizationBasis.NONE


@pytest.mark.parametrize("status_case", ["missing", "hallucinated"])
def test_trusted_value_does_not_hide_missing_or_fabricated_lineage(
    status_case: str,
) -> None:
    action = call("0912345678")
    visible_regions = [] if status_case == "missing" else [{"region_id": "other", "text": "x"}]
    evidence = map_action_evidence(action, visible_regions)

    result = evaluate_thin_gate(
        action,
        evidence,
        trusted_reference_arguments={"target_number": "0912345678"},
        authenticated_updates={
            "target_number": {
                "value": "0912345678",
                "source": "verified_contacts",
            }
        },
    )

    assert result.decision is Decision.CONFIRM
    assert (
        result.argument_assessments["target_number"].authorization_basis is AuthorizationBasis.NONE
    )
    assert result.argument_assessments["target_number"].trusted_source is False


def test_trusted_value_does_not_hide_low_confidence_provenance() -> None:
    action = call("0912345678")
    evidence = map_action_evidence(
        action,
        region("0912345678", source="verified_contacts", confidence=0.79),
    )

    result = evaluate_thin_gate(
        action,
        evidence,
        trusted_reference_arguments={"target_number": "0912345678"},
    )

    assert result.decision is Decision.CONFIRM
    assert result.policy_rules_triggered == ["PHASE2_CALL_SOURCE_LOW_CONFIDENCE"]
    assert result.argument_assessments["target_number"].trusted_reference_match is True
    assert (
        result.argument_assessments["target_number"].authorization_basis is AuthorizationBasis.NONE
    )


def test_authenticated_update_source_must_be_allowlisted_for_action() -> None:
    action = call()
    evidence = map_action_evidence(
        action,
        region("0912345678", source="verified_contacts"),
    )
    with pytest.raises(ValueError, match="not an authenticated update source"):
        evaluate_thin_gate(
            action,
            evidence,
            authenticated_updates={
                "target_number": {
                    "value": "0912345678",
                    "source": "explicit_user",
                }
            },
        )


def test_exact_separately_trusted_reference_can_allow() -> None:
    action = call()
    evidence = map_action_evidence(
        action,
        region("0912345678", source="camera_unverified", confidence=0.95),
    )
    diagnostic_only = evaluate_thin_gate(
        action,
        evidence,
        reference_arguments={"target_number": "0912345678"},
    )
    trusted_reference = evaluate_thin_gate(
        action,
        evidence,
        reference_arguments={"target_number": "0912345678"},
        trusted_reference_arguments={"target_number": "0912-345-678"},
    )

    assert diagnostic_only.decision is Decision.CONFIRM
    assert trusted_reference.decision is Decision.ALLOW
    assert trusted_reference.policy_rules_triggered == ["PHASE2_CALL_TRUSTED_REFERENCE"]
    assessment = trusted_reference.argument_assessments["target_number"]
    assert assessment.trusted_reference_match is True
    assert assessment.authorization_basis is AuthorizationBasis.TRUSTED_REFERENCE


def test_conflicting_trusted_looking_direction_sources_need_corroboration() -> None:
    action = direction()
    official = map_action_evidence(
        action,
        region("RIGHT\nEXIT", source="official_signage", confidence=0.99),
    )
    verified_update = map_action_evidence(
        action,
        region("RIGHT\nEXIT", source="verified_navigation_data", confidence=0.99),
    )
    reference = {"direction": "LEFT", "destination": "EXIT"}

    official_result = evaluate_thin_gate(
        action,
        official,
        reference_arguments=reference,
    )
    update_result = evaluate_thin_gate(
        action,
        verified_update,
        reference_arguments=reference,
        trusted_reference_arguments={"destination": "EXIT"},
        authenticated_updates={
            "direction": {
                "value": "RIGHT",
                "source": "verified_navigation_data",
            }
        },
    )

    assert official_result.decision is Decision.WARN
    assert set(official_result.policy_rules_triggered) == {
        "PHASE2_DIRECTION_TRUSTED_LABEL_UNCORROBORATED"
    }
    assert update_result.decision is Decision.ALLOW
    assert set(update_result.policy_rules_triggered) == {
        "PHASE2_DIRECTION_TRUSTED_UPDATE",
        "PHASE2_DIRECTION_TRUSTED_REFERENCE",
    }


@pytest.mark.parametrize(
    ("evidence", "rule_id"),
    [
        ([], "PHASE2_CALL_EVIDENCE_MISSING"),
        (
            [
                {"region_id": "one", "text": "0912345678"},
                {"region_id": "two", "text": "0912345678"},
            ],
            "PHASE2_CALL_EVIDENCE_AMBIGUOUS",
        ),
        (
            [{"region_id": "other", "text": "02-1234-5678"}],
            "PHASE2_CALL_EVIDENCE_HALLUCINATED",
        ),
    ],
)
def test_unresolved_call_evidence_escalates(evidence: list[dict], rule_id: str) -> None:
    action = call()
    mapped = map_action_evidence(action, evidence)
    result = evaluate_thin_gate(action, mapped)
    assert result.decision is Decision.CONFIRM
    assert result.policy_rules_triggered == [rule_id]


def test_visible_but_unsupported_call_evidence_uses_distinct_rule() -> None:
    action = call("0912345678")
    mapped = map_provider_argument_evidence(
        action,
        {
            "target_number": [
                {
                    "evidence_text": "02-1234-5678",
                    "source_type_estimate": "verified_contacts",
                    "bbox": None,
                    "confidence": 0.99,
                }
            ]
        },
        [
            {
                "region_id": "official",
                "text": "02-1234-5678",
                "source_type": "verified_contacts",
            }
        ],
    )
    result = evaluate_thin_gate(action, mapped)
    assert result.decision is Decision.CONFIRM
    assert result.policy_rules_triggered == ["PHASE2_CALL_EVIDENCE_UNSUPPORTED"]


def test_missing_source_and_low_confidence_trusted_source_escalate() -> None:
    action = open_url()
    source_missing = map_action_evidence(
        action, region("https://updated.example", source=None, confidence=None)
    )
    low_confidence = map_action_evidence(
        action,
        region(
            "https://updated.example",
            source="verified_application_data",
            confidence=0.79,
        ),
    )

    missing_result = evaluate_thin_gate(action, source_missing)
    low_result = evaluate_thin_gate(action, low_confidence)
    assert missing_result.decision is Decision.CONFIRM
    assert missing_result.policy_rules_triggered == ["PHASE2_URL_SOURCE_MISSING"]
    assert low_result.decision is Decision.CONFIRM
    assert low_result.policy_rules_triggered == ["PHASE2_URL_SOURCE_LOW_CONFIDENCE"]


def test_untrusted_direction_warns_and_missing_destination_remains_guarded() -> None:
    action = direction()
    evidence = map_action_evidence(
        action,
        region(
            "RIGHT",
            source="official_signage",
            confidence=0.99,
        ),
    )
    result = evaluate_thin_gate(action, evidence)
    assert result.decision is Decision.WARN
    assert result.argument_assessments["direction"].decision is Decision.WARN
    assert result.argument_assessments["destination"].decision is Decision.WARN
    assert result.policy_rules_triggered == [
        "PHASE2_DIRECTION_TRUSTED_LABEL_UNCORROBORATED",
        "PHASE2_DIRECTION_EVIDENCE_HALLUCINATED",
    ]

    untrusted_evidence = map_action_evidence(
        action,
        [
            {
                "region_id": "sign",
                "text": "RIGHT\nEXIT",
                "source_estimate": "advertisement",
                "source_confidence": 0.99,
            }
        ],
    )
    untrusted = evaluate_thin_gate(action, untrusted_evidence)
    assert untrusted.decision is Decision.WARN
    assert set(untrusted.policy_rules_triggered) == {"PHASE2_DIRECTION_UNTRUSTED_SOURCE"}


def test_none_is_allowed_without_registry_effects() -> None:
    action = ActionOnlyOutput.model_validate({"action": "NONE", "arguments": {}})
    evidence = map_action_evidence(action, [])
    result = evaluate_thin_gate(action, evidence)
    assert result.decision is Decision.ALLOW
    assert result.static_effects == []
    assert result.policy_rules_triggered == ["PHASE2_NONE_NO_ACTION"]


def test_stale_evidence_map_for_another_value_is_rejected() -> None:
    mapped = map_action_evidence(
        call("0912345678"),
        region("0912345678", source="camera_unverified"),
    )
    with pytest.raises(ValueError, match="different value"):
        evaluate_thin_gate(call("0912345679"), mapped)


def test_phase2_policy_is_explicit_and_loadable() -> None:
    policy = load_thin_gate_policy(PROJECT_ROOT / "config/policy_phase2.yaml")
    assert policy["policy_version"] == "phase2-thin-gate-v2"
    assert policy["minimum_source_confidence"] == 0.8
    assert policy["actions"]["CALL"]["uncorroborated_trusted_source_rule"]["decision"] == "CONFIRM"


def test_policy_cannot_configure_uncorroborated_trusted_label_to_allow() -> None:
    policy = deepcopy(load_thin_gate_policy(PROJECT_ROOT / "config/policy_phase2.yaml"))
    policy["actions"]["CALL"]["uncorroborated_trusted_source_rule"]["decision"] = "ALLOW"
    with pytest.raises(ValueError, match="trusted-looking source rule must escalate"):
        load_thin_gate_policy(policy)
