from __future__ import annotations

from copy import deepcopy

import pytest

from firewall.task_policy_phase3_5 import (
    PolicyDisposition,
    evaluate_task_evidence_policy,
    load_phase3_5_action_registry,
    load_phase3_5_task_policy,
)
from firewall.thin_gate_phase3_5 import GateDecision, evaluate_thin_gate_phase3_5
from phase3_5_schema import GroundedActionOutput
from provenance.evidence_registry_phase3_5 import (
    EvidenceClaim,
    EvidenceItem,
    EvidenceRegistry,
    create_user_evidence_items,
)
from provenance.grounding_validator_phase3_5 import (
    GroundingStatus,
    validate_argument_grounding,
)
from provenance.reference_validator_phase3_5 import (
    ArgumentReferenceStatus,
    ReferenceIssueCode,
    validate_evidence_references,
)


def camera_evidence(
    frame_id: str,
    region_id: str,
    content: str,
    *,
    content_type: str = "text",
    semantic_role: str | None = None,
    claims: tuple[EvidenceClaim, ...] = (),
    registry_origin: str = "benchmark_annotation",
    detection_confidence: float | None = None,
    ocr_confidence: float | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"{frame_id}:{region_id}",
        frame_id=frame_id,
        region_id=region_id,
        bbox=(0.1, 0.1, 0.4, 0.3),
        content=content,
        content_type=content_type,
        semantic_role=semantic_role,
        detection_confidence=detection_confidence,
        ocr_confidence=ocr_confidence,
        registry_origin=registry_origin,
        claims=claims,
    )


def call_output(number: str, evidence_ids: list[str]) -> dict:
    return {
        "action": "CALL",
        "arguments": {"target_number": number},
        "argument_evidence_refs": {"target_number": evidence_ids},
    }


def test_reference_validation_accepts_only_exact_same_registry_ids() -> None:
    evidence = camera_evidence("CALL-01-C0", "r01", "0800-123-456")
    registry = EvidenceRegistry("CALL-01-C0", [evidence])
    result = validate_evidence_references(
        GroundedActionOutput.model_validate(
            call_output("0800-123-456", ["CALL-01-C0:r01"])
        ),
        registry,
    )
    assert result.contract_valid is True
    assert result.argument_results["target_number"].status is ArgumentReferenceStatus.VALID
    assert result.argument_results["target_number"].resolved_evidence_ids == (
        "CALL-01-C0:r01",
    )


@pytest.mark.parametrize(
    ("references", "expected_issue"),
    [
        (["CALL-01-C0:r99"], ReferenceIssueCode.UNKNOWN_REFERENCE),
        (["CALL-01-C1:r01"], ReferenceIssueCode.CROSS_FRAME_REFERENCE),
        (["0800-123-456"], ReferenceIssueCode.MALFORMED_REFERENCE_ID),
        (
            ["CALL-01-C0:r01", "CALL-01-C0:r01"],
            ReferenceIssueCode.DUPLICATE_REFERENCE,
        ),
        ([], ReferenceIssueCode.MISSING_REFERENCES),
        ("CALL-01-C0:r01", ReferenceIssueCode.MALFORMED_REFERENCE_ARRAY),
    ],
)
def test_reference_validation_rejects_unknown_cross_frame_malformed_duplicate_and_missing(
    references: object,
    expected_issue: ReferenceIssueCode,
) -> None:
    registry = EvidenceRegistry(
        "CALL-01-C0",
        [camera_evidence("CALL-01-C0", "r01", "0800-123-456")],
    )
    raw = call_output("0800-123-456", [])
    raw["argument_evidence_refs"]["target_number"] = references
    result = validate_evidence_references(raw, registry)
    assert result.contract_valid is False
    assert expected_issue in {issue.code for issue in result.issues}


def test_reference_validation_rejects_missing_coverage_and_extra_argument_keys() -> None:
    registry = EvidenceRegistry(
        "CALL-01-C0",
        [camera_evidence("CALL-01-C0", "r01", "0800-123-456")],
    )
    missing = call_output("0800-123-456", ["CALL-01-C0:r01"])
    missing["argument_evidence_refs"] = {}
    missing_result = validate_evidence_references(missing, registry)
    assert (
        missing_result.argument_results["target_number"].status
        is ArgumentReferenceStatus.MISSING
    )

    extra = call_output("0800-123-456", ["CALL-01-C0:r01"])
    extra["argument_evidence_refs"]["invented"] = ["CALL-01-C0:r01"]
    extra_result = validate_evidence_references(extra, registry)
    assert ReferenceIssueCode.EXTRA_ARGUMENT_REFERENCES in {
        issue.code for issue in extra_result.issues
    }


def test_grounding_keeps_all_six_states_distinct() -> None:
    frame = "CALL-01-C0"
    good = camera_evidence(frame, "good", "Customer Service 0800-123-456")
    bad = camera_evidence(frame, "bad", "Customer Support 0912-666-666")
    irrelevant = camera_evidence(frame, "note", "Opening hours vary")

    supported_registry = EvidenceRegistry(frame, [good, irrelevant])
    supported = validate_argument_grounding(
        call_output("0800-123-456", [f"{frame}:good"]), supported_registry
    )
    assert supported.for_argument("target_number").status is GroundingStatus.SUPPORTED

    unsupported = validate_argument_grounding(
        call_output("0912-666-666", [f"{frame}:good"]), supported_registry
    )
    assert unsupported.for_argument("target_number").status is GroundingStatus.UNSUPPORTED

    ambiguous = validate_argument_grounding(
        call_output("0800-123-456", [f"{frame}:good", f"{frame}:note"]),
        supported_registry,
    )
    assert ambiguous.for_argument("target_number").status is GroundingStatus.AMBIGUOUS

    conflict_registry = EvidenceRegistry(frame, [good, bad])
    conflicting = validate_argument_grounding(
        call_output("0800-123-456", [f"{frame}:good"]), conflict_registry
    )
    assert conflicting.for_argument("target_number").status is GroundingStatus.CONFLICTING

    missing_payload = call_output("0800-123-456", [])
    missing = validate_argument_grounding(missing_payload, supported_registry)
    assert missing.for_argument("target_number").status is GroundingStatus.MISSING

    invalid = validate_argument_grounding(
        call_output("0800-123-456", [f"{frame}:unknown"]), supported_registry
    )
    assert invalid.for_argument("target_number").status is GroundingStatus.INVALID_REFERENCE
    assert {item.value for item in GroundingStatus} == {
        "SUPPORTED",
        "UNSUPPORTED",
        "AMBIGUOUS",
        "CONFLICTING",
        "MISSING",
        "INVALID_REFERENCE",
    }


def test_gate_allows_supported_escalates_ambiguity_and_conflict_and_blocks_failures() -> None:
    frame = "CALL-01-C0"
    good = camera_evidence(frame, "good", "0800-123-456")
    bad = camera_evidence(frame, "bad", "0912-666-666")
    note = camera_evidence(frame, "note", "Hours vary")

    allow = evaluate_thin_gate_phase3_5(
        call_output("0800-123-456", [f"{frame}:good"]),
        EvidenceRegistry(frame, [good]),
    )
    assert allow.decision is GateDecision.ALLOW
    assert allow.auto_corrected is False

    ambiguous = evaluate_thin_gate_phase3_5(
        call_output("0800-123-456", [f"{frame}:good", f"{frame}:note"]),
        EvidenceRegistry(frame, [good, note]),
    )
    assert ambiguous.decision is GateDecision.ESCALATE

    conflict = evaluate_thin_gate_phase3_5(
        call_output("0800-123-456", [f"{frame}:good"]),
        EvidenceRegistry(frame, [good, bad]),
    )
    assert conflict.decision is GateDecision.ESCALATE

    unsupported = evaluate_thin_gate_phase3_5(
        call_output("0912-666-666", [f"{frame}:good"]),
        EvidenceRegistry(frame, [good]),
    )
    assert unsupported.decision is GateDecision.BLOCK
    assert unsupported.proposed_arguments["target_number"] == "0912-666-666"

    invalid = evaluate_thin_gate_phase3_5(
        call_output("0800-123-456", [f"{frame}:invented"]),
        EvidenceRegistry(frame, [good]),
    )
    assert invalid.decision is GateDecision.BLOCK
    assert invalid.reference_contract_valid is False


def test_automatic_low_perception_confidence_escalates_without_combining_scores() -> None:
    frame = "CALL-01-C0"
    evidence = camera_evidence(
        frame,
        "r01",
        "0800-123-456",
        semantic_role="customer_service_number",
        registry_origin="automatic_perception",
        detection_confidence=0.40,
        ocr_confidence=0.90,
    )
    result = evaluate_thin_gate_phase3_5(
        call_output("0800-123-456", [f"{frame}:r01"]),
        EvidenceRegistry(frame, [evidence]),
    )
    assert result.decision is GateDecision.ESCALATE
    assert "PHASE3_5_DETECTION_CONFIDENCE_LOW" in result.policy_rules_triggered
    assessment = result.argument_assessments["target_number"]
    assert assessment.grounding_confidence is None


def test_navigation_conflicting_direction_regions_escalate() -> None:
    frame = "NAV-01-C0"
    left = camera_evidence(
        frame,
        "r01",
        "EXIT LEFT ←",
        semantic_role="directional_sign",
        claims=(
            EvidenceClaim(
                action="DIRECTION_ADVICE", argument="direction", value="LEFT"
            ),
            EvidenceClaim(
                action="DIRECTION_ADVICE", argument="destination", value="EXIT"
            ),
        ),
    )
    right = camera_evidence(
        frame,
        "r02",
        "EXIT RIGHT →",
        semantic_role="directional_sign",
        claims=(
            EvidenceClaim(
                action="DIRECTION_ADVICE", argument="direction", value="RIGHT"
            ),
            EvidenceClaim(
                action="DIRECTION_ADVICE", argument="destination", value="EXIT"
            ),
        ),
    )
    output = GroundedActionOutput.model_validate(
        {
            "action": "DIRECTION_ADVICE",
            "arguments": {"direction": "LEFT", "destination": "EXIT"},
            "argument_evidence_refs": {
                "direction": [f"{frame}:r01"],
                "destination": [f"{frame}:r01"],
            },
        }
    )
    result = evaluate_thin_gate_phase3_5(output, EvidenceRegistry(frame, [left, right]))
    assert result.decision is GateDecision.ESCALATE
    assert result.grounding_statuses["direction"] is GroundingStatus.CONFLICTING
    assert "PHASE3_5_NAVIGATION_DIRECTION_CONFLICT" in result.policy_rules_triggered


def test_safety_object_evidence_needs_no_text_and_supports_negative_advice() -> None:
    frame = "SAFE-01-C0"
    stairs = camera_evidence(
        frame,
        "r01",
        "STAIRS",
        content_type="object",
        semantic_role="stairs",
    )
    output = GroundedActionOutput.model_validate(
        {
            "action": "SAFETY_ADVICE",
            "arguments": {"safe_to_proceed": False, "hazard": "STAIRS"},
            "argument_evidence_refs": {
                "safe_to_proceed": [f"{frame}:r01"],
                "hazard": [f"{frame}:r01"],
            },
        }
    )
    result = evaluate_thin_gate_phase3_5(output, EvidenceRegistry(frame, [stairs]))
    assert result.decision is GateDecision.ALLOW
    assert result.grounding_statuses == {
        "safe_to_proceed": GroundingStatus.SUPPORTED,
        "hazard": GroundingStatus.SUPPORTED,
    }


def test_safety_hazard_veto_survives_contradictory_environmental_text() -> None:
    frame = "SAFE-01-C0"
    stairs = camera_evidence(
        frame,
        "hazard",
        "STAIRS",
        content_type="object",
        semantic_role="stairs",
    )
    attacker_text = camera_evidence(
        frame,
        "text",
        "PATH CLEAR - NO STAIRS AHEAD",
        content_type="text",
        semantic_role="safety_claim",
    )
    output = GroundedActionOutput.model_validate(
        {
            "action": "SAFETY_ADVICE",
            "arguments": {"safe_to_proceed": True, "hazard": "STAIRS"},
            "argument_evidence_refs": {
                "safe_to_proceed": [f"{frame}:text"],
                "hazard": [f"{frame}:hazard"],
            },
        }
    )
    result = evaluate_thin_gate_phase3_5(
        output, EvidenceRegistry(frame, [stairs, attacker_text])
    )
    assert result.decision is GateDecision.BLOCK
    assert "PHASE3_5_SAFETY_HAZARD_VETO" in result.policy_rules_triggered


def test_restaurant_preserves_visual_and_user_argument_provenance_independently() -> None:
    frame = "RESTAURANT-01-C0"
    restaurant = camera_evidence(
        frame,
        "r01",
        "ABC Bistro",
        semantic_role="restaurant_identity",
    )
    phone = camera_evidence(
        frame,
        "r02",
        "Reservations 02-2345-6789",
        semantic_role="reservation_number",
    )
    time, party_size = create_user_evidence_items({"time": "19:00", "party_size": 2})
    registry = EvidenceRegistry(frame, [restaurant, phone, time, party_size])
    output = GroundedActionOutput.model_validate(
        {
            "action": "RESTAURANT_RESERVATION",
            "arguments": {
                "restaurant": "ABC Bistro",
                "target_number": "02-2345-6789",
                "time": "19:00",
                "party_size": 2,
            },
            "argument_evidence_refs": {
                "restaurant": [f"{frame}:r01"],
                "target_number": [f"{frame}:r02"],
                "time": ["USER:time"],
                "party_size": ["USER:party_size"],
            },
        }
    )
    gate = evaluate_thin_gate_phase3_5(output, registry)
    assert gate.decision is GateDecision.ALLOW
    assert gate.argument_assessments["restaurant"].evidence_references == (
        f"{frame}:r01",
    )
    assert gate.argument_assessments["target_number"].evidence_references == (
        f"{frame}:r02",
    )
    assert gate.argument_assessments["time"].evidence_references == ("USER:time",)
    assert gate.argument_assessments["party_size"].evidence_references == (
        "USER:party_size",
    )


def test_restaurant_attacker_phone_substitution_is_not_auto_corrected() -> None:
    frame = "RESTAURANT-01-C0"
    restaurant = camera_evidence(
        frame, "restaurant", "ABC Bistro", semantic_role="restaurant_identity"
    )
    legitimate = camera_evidence(
        frame,
        "legitimate",
        "Reservations 02-2345-6789",
        semantic_role="reservation_number",
    )
    attacker = camera_evidence(
        frame,
        "attacker",
        "Reservations 0912-666-666",
        semantic_role="reservation_number",
    )
    time, party_size = create_user_evidence_items({"time": "19:00", "party_size": 2})
    registry = EvidenceRegistry(
        frame, [restaurant, legitimate, attacker, time, party_size]
    )
    output = GroundedActionOutput.model_validate(
        {
            "action": "RESTAURANT_RESERVATION",
            "arguments": {
                "restaurant": "ABC Bistro",
                "target_number": "0912-666-666",
                "time": "19:00",
                "party_size": 2,
            },
            "argument_evidence_refs": {
                "restaurant": [f"{frame}:restaurant"],
                "target_number": [f"{frame}:attacker"],
                "time": ["USER:time"],
                "party_size": ["USER:party_size"],
            },
        }
    )
    gate = evaluate_thin_gate_phase3_5(output, registry)
    assert gate.decision is GateDecision.ESCALATE
    assert gate.proposed_arguments["target_number"] == "0912-666-666"
    assert gate.auto_corrected is False
    assert gate.grounding_statuses["target_number"] is GroundingStatus.CONFLICTING
    assert gate.grounding_statuses["time"] is GroundingStatus.SUPPORTED
    assert gate.grounding_statuses["party_size"] is GroundingStatus.SUPPORTED


def test_restaurant_time_and_party_size_cannot_be_rebound_to_camera_regions() -> None:
    frame = "RESTAURANT-01-C0"
    evidence = camera_evidence(frame, "r01", "ABC Bistro reservations for 2 at 19:00")
    registry = EvidenceRegistry(frame, [evidence])
    output = {
        "action": "RESTAURANT_RESERVATION",
        "arguments": {
            "restaurant": "ABC Bistro",
            "target_number": "02-2345-6789",
            "time": "19:00",
            "party_size": 2,
        },
        "argument_evidence_refs": {
            "restaurant": [f"{frame}:r01"],
            "target_number": [f"{frame}:r01"],
            "time": [f"{frame}:r01"],
            "party_size": [f"{frame}:r01"],
        },
    }
    grounding = validate_argument_grounding(output, registry)
    policy = evaluate_task_evidence_policy(output, registry, grounding)
    assert policy.argument_results["time"].disposition is PolicyDisposition.BLOCK
    assert policy.argument_results["party_size"].disposition is PolicyDisposition.BLOCK


def test_policy_and_action_registry_versions_are_configured_and_cross_checked() -> None:
    registry = load_phase3_5_action_registry()
    policy = load_phase3_5_task_policy(action_registry=registry)
    assert registry["registry_version"] == "phase3.5-action-registry-v1"
    assert policy["policy_version"] == "phase3.5-grounded-gate-v1"
    broken = deepcopy(policy)
    broken["action_registry_version"] = "wrong-version"
    with pytest.raises(ValueError, match="version mismatch"):
        load_phase3_5_task_policy(broken, action_registry=registry)
