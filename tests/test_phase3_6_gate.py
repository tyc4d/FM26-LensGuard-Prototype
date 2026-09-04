from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from firewall.thin_gate_phase3_6 import (
    GateDecision,
    GateReasonCode,
    Phase36GateResult,
    evaluate_thin_gate_phase3_6,
    load_phase3_6_gate_policy,
)
from phase3_5_schema import GroundedActionOutput
from phase3_6_schema import AuthenticityStatus, UncertaintyStatus
from provenance.evidence_analysis_phase3_6 import (
    EvidenceAnalysisContext,
    analyze_evidence_uncertainty,
)
from provenance.evidence_registry_phase3_5 import (
    EvidenceClaim,
    EvidenceItem,
    EvidenceRegistry,
    canonical_evidence_id,
    create_user_evidence_items,
)


FRAME = "phase36-gate-fixture"


def _camera(
    region: str,
    content: str,
    *,
    semantic_role: str | None,
    content_type: str = "text",
    **overrides: Any,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=canonical_evidence_id(FRAME, region),
        frame_id=FRAME,
        region_id=region,
        bbox=(0.1, 0.1, 0.4, 0.3),
        content=content,
        content_type=content_type,
        semantic_role=semantic_role,
        registry_origin="physical_annotation",
        **overrides,
    )


def _context(
    item: EvidenceItem,
    target: str | None,
    *,
    authenticity: AuthenticityStatus | str = AuthenticityStatus.ESTABLISHED,
) -> EvidenceAnalysisContext:
    status = AuthenticityStatus(authenticity)
    return EvidenceAnalysisContext(
        evidence_id=item.evidence_id,
        associated_target_object_id=target,
        authenticity_status=status,
        authenticity_basis=(
            "deterministic gate fixture authentication"
            if status is AuthenticityStatus.ESTABLISHED
            else None
        ),
    )


def _call_output(number: str, evidence_id: str) -> GroundedActionOutput:
    return GroundedActionOutput.model_validate(
        {
            "action": "CALL",
            "arguments": {"target_number": number},
            "argument_evidence_refs": {"target_number": [evidence_id]},
        }
    )


def _call_gate(
    output: GroundedActionOutput | dict[str, Any],
    items: list[EvidenceItem],
    *,
    contexts: dict[str, EvidenceAnalysisContext] | None = None,
):
    return evaluate_thin_gate_phase3_6(
        output,
        EvidenceRegistry(FRAME, items),
        argument_target_object_ids={"target_number": "product-1"},
        evidence_contexts=(
            contexts
            if contexts is not None
            else {item.evidence_id: _context(item, "product-1") for item in items}
        ),
    )


def _service_number(
    region: str,
    number: str,
    *,
    semantic_role: str | None = "customer_service_number",
    **overrides: Any,
) -> EvidenceItem:
    return _camera(
        region,
        f"Customer Service: {number}",
        semantic_role=semantic_role,
        **overrides,
    )


def test_supported_call_is_allowed_without_argument_repair() -> None:
    service = _service_number("service", "0800-123-456")
    output = _call_output("0800-123-456", service.evidence_id)

    result = _call_gate(output, [service])

    assert result.decision is GateDecision.ALLOW
    assert result.reason_code is GateReasonCode.ALLOW_SUPPORTED
    assert result.reason_codes_triggered == (GateReasonCode.ALLOW_SUPPORTED,)
    assert result.triggering_argument is None
    assert result.escalation is None
    assert result.proposed_arguments == {"target_number": "0800-123-456"}
    assert result.argument_evidence_refs == {
        "target_number": (service.evidence_id,)
    }
    assert result.auto_corrected is False
    assessment = result.argument_assessments["target_number"]
    assert assessment.decision is GateDecision.ALLOW
    assert assessment.analysis.status is UncertaintyStatus.SUPPORTED


def test_concrete_value_mismatch_is_blocked_as_unsupported() -> None:
    service = _service_number("service", "0800-123-456")
    output = _call_output("0912-666-666", service.evidence_id)

    result = _call_gate(output, [service])

    assert result.decision is GateDecision.BLOCK
    assert result.reason_code is GateReasonCode.UNSUPPORTED_ARGUMENT
    assert result.triggering_argument == "target_number"
    assert result.argument_assessments["target_number"].analysis.status is (
        UncertaintyStatus.UNSUPPORTED
    )
    assert result.escalation is None


def test_exact_value_with_wrong_semantic_role_is_blocked_as_relationship_mismatch() -> None:
    serial = _service_number(
        "serial",
        "0800-123-456",
        semantic_role="serial_number",
    )
    output = _call_output("0800-123-456", serial.evidence_id)

    result = _call_gate(output, [serial])

    assert result.decision is GateDecision.BLOCK
    assert result.reason_code is GateReasonCode.SEMANTIC_RELATIONSHIP_MISMATCH
    assert result.triggering_argument == "target_number"
    assert result.argument_assessments["target_number"].analysis.status is (
        UncertaintyStatus.UNSUPPORTED
    )


@pytest.mark.parametrize(
    "references",
    (
        {"target_number": [f"{FRAME}:unknown"]},
        {"target_number": ["different-frame:service"]},
        {"target_number": ["not-an-evidence-id"]},
        {"target_number": [f"{FRAME}:service", f"{FRAME}:service"]},
        {"target_number": []},
        {},
    ),
)
def test_invalid_or_missing_reference_contract_is_blocked(
    references: dict[str, list[str]],
) -> None:
    service = _service_number("service", "0800-123-456")
    output = {
        "action": "CALL",
        "arguments": {"target_number": "0800-123-456"},
        "argument_evidence_refs": references,
    }

    result = _call_gate(output, [service])

    assert result.decision is GateDecision.BLOCK
    assert result.reason_code is GateReasonCode.INVALID_REFERENCE
    assert result.reference_contract_valid is False
    assert result.evidence_analysis is None
    assert result.argument_assessments == {}


def test_two_distinct_task_valid_numbers_escalate_with_structured_candidates() -> None:
    first = _service_number("first", "0800-123-456")
    second = _service_number("second", "0912-666-666")
    output = _call_output("0800-123-456", first.evidence_id)

    result = _call_gate(output, [first, second])

    assert result.decision is GateDecision.ESCALATE
    assert result.reason_code is GateReasonCode.CONFLICTING_EVIDENCE
    assert result.triggering_argument == "target_number"
    assert result.uncertainty_statuses == {
        "target_number": UncertaintyStatus.CONFLICTING
    }
    assert result.escalation is not None
    assert result.escalation.reason_code.value == "CONFLICTING_EVIDENCE"
    assert result.escalation.argument == "target_number"
    assert result.escalation.candidate_values == ("0800123456", "0912666666")
    assert result.proposed_arguments["target_number"] == "0800-123-456"
    assert result.auto_corrected is False

    forged_allow = result.model_dump(mode="python")
    forged_allow.update(
        {
            "decision": "ALLOW",
            "reason_code": "ALLOW_SUPPORTED",
            "reason_codes_triggered": ["ALLOW_SUPPORTED"],
            "triggering_argument": None,
            "escalation": None,
        }
    )
    with pytest.raises(ValidationError, match="aggregate gate reasons"):
        Phase36GateResult.model_validate(forged_allow)


def test_unresolved_unselected_phone_candidate_escalates_instead_of_allowing() -> None:
    selected = _service_number("selected", "0800-123-456")
    unresolved = _service_number("unresolved", "0912-666-666")
    result = _call_gate(
        _call_output("0800-123-456", selected.evidence_id),
        [selected, unresolved],
        contexts={selected.evidence_id: _context(selected, "product-1")},
    )

    assert result.evidence_analysis is not None
    assert result.evidence_analysis.argument_results["target_number"].status is (
        UncertaintyStatus.SUPPORTED
    )
    assert result.decision is GateDecision.ESCALATE
    assert result.reason_code is GateReasonCode.INSUFFICIENT_EVIDENCE
    assessment = result.argument_assessments["target_number"]
    assert assessment.reason_code is GateReasonCode.INSUFFICIENT_EVIDENCE


def test_grounded_camera_value_with_unknown_authenticity_escalates() -> None:
    service = _service_number("service", "0800-123-456")
    contexts = {
        service.evidence_id: _context(
            service,
            "product-1",
            authenticity=AuthenticityStatus.UNKNOWN,
        )
    }

    result = _call_gate(
        _call_output("0800-123-456", service.evidence_id),
        [service],
        contexts=contexts,
    )

    assert result.decision is GateDecision.ESCALATE
    assert result.reason_code is GateReasonCode.AUTHENTICITY_UNKNOWN
    assert result.uncertainty_statuses["target_number"] is (
        UncertaintyStatus.AUTHENTICITY_UNKNOWN
    )
    assert result.escalation is not None
    assert result.escalation.candidate_values == ("0800123456",)


def test_unestablished_semantic_relationship_escalates_as_insufficient() -> None:
    service = _service_number(
        "service",
        "0800-123-456",
        semantic_role=None,
    )

    result = _call_gate(
        _call_output("0800-123-456", service.evidence_id),
        [service],
    )

    assert result.decision is GateDecision.ESCALATE
    assert result.reason_code is GateReasonCode.INSUFFICIENT_EVIDENCE
    assert result.uncertainty_statuses["target_number"] is (
        UncertaintyStatus.INSUFFICIENT_EVIDENCE
    )
    assert result.escalation is not None


def test_uncalibrated_low_numeric_confidences_do_not_create_a_threshold() -> None:
    service = _service_number(
        "service",
        "0800-123-456",
        detection_confidence=0.01,
        ocr_confidence=0.01,
        grounding_confidence=0.01,
    )

    result = _call_gate(
        _call_output("0800-123-456", service.evidence_id),
        [service],
    )

    assert result.decision is GateDecision.ALLOW
    assert GateReasonCode.LOW_PERCEPTION_CONFIDENCE not in (
        result.reason_codes_triggered
    )
    confidence = result.argument_assessments[
        "target_number"
    ].analysis.uncertainty.evidence_confidences[0]
    assert confidence.detection_confidence == pytest.approx(0.01)
    assert confidence.ocr_confidence == pytest.approx(0.01)
    assert confidence.grounding_confidence is None
    assert result.argument_assessments[
        "target_number"
    ].analysis.uncertainty.grounding_confidence is None

    weakened = deepcopy(load_phase3_6_gate_policy())
    weakened["perception_confidence"]["numeric_thresholds"] = 0.7
    with pytest.raises(ValueError, match="must not invent"):
        load_phase3_6_gate_policy(weakened)


def test_none_action_with_no_critical_arguments_is_allowed() -> None:
    output = GroundedActionOutput.model_validate(
        {
            "action": "NONE",
            "arguments": {},
            "argument_evidence_refs": {},
        }
    )

    result = evaluate_thin_gate_phase3_6(
        output,
        EvidenceRegistry(FRAME, []),
    )

    assert result.decision is GateDecision.ALLOW
    assert result.reason_code is GateReasonCode.ALLOW_SUPPORTED
    assert result.argument_assessments == {}
    assert result.uncertainty_statuses == {}


def test_stale_precomputed_analysis_is_rejected_after_fresh_recomputation() -> None:
    first = _service_number("first", "0800-123-456")
    second = _service_number("second", "0912-666-666")
    registry = EvidenceRegistry(FRAME, [first, second])
    contexts = {
        item.evidence_id: _context(item, "product-1") for item in (first, second)
    }
    targets = {"target_number": "product-1"}
    stale = analyze_evidence_uncertainty(
        _call_output("0800-123-456", first.evidence_id),
        registry,
        argument_target_object_ids=targets,
        evidence_contexts=contexts,
    )

    with pytest.raises(ValueError, match="does not match fresh"):
        evaluate_thin_gate_phase3_6(
            _call_output("0912-666-666", second.evidence_id),
            registry,
            argument_target_object_ids=targets,
            evidence_contexts=contexts,
            evidence_analysis=stale,
        )


def test_equal_precomputed_analysis_is_accepted_but_recomputed() -> None:
    service = _service_number("service", "0800-123-456")
    output = _call_output("0800-123-456", service.evidence_id)
    registry = EvidenceRegistry(FRAME, [service])
    contexts = {service.evidence_id: _context(service, "product-1")}
    targets = {"target_number": "product-1"}
    precomputed = analyze_evidence_uncertainty(
        output,
        registry,
        argument_target_object_ids=targets,
        evidence_contexts=contexts,
    )

    result = evaluate_thin_gate_phase3_6(
        output,
        registry,
        argument_target_object_ids=targets,
        evidence_contexts=contexts,
        evidence_analysis=precomputed,
    )

    assert result.decision is GateDecision.ALLOW
    assert result.evidence_analysis == precomputed


def test_gate_audit_rejects_tampering_and_runtime_mapping_mutation() -> None:
    service = _service_number("service", "0800-123-456")
    result = _call_gate(
        _call_output("0800-123-456", service.evidence_id),
        [service],
    )

    changed_argument = result.model_dump(mode="python")
    changed_argument["proposed_arguments"]["target_number"] = "0912-666-666"
    with pytest.raises(ValidationError, match="differs from fresh analysis"):
        Phase36GateResult.model_validate(changed_argument)

    changed_reference = result.model_dump(mode="python")
    changed_reference["argument_evidence_refs"]["target_number"] = [
        f"{FRAME}:unknown"
    ]
    with pytest.raises(ValidationError, match="references differ"):
        Phase36GateResult.model_validate(changed_reference)

    false_allow = result.model_dump(mode="python")
    false_allow["uncertainty_statuses"]["target_number"] = "CONFLICTING"
    with pytest.raises(ValidationError, match="statuses do not match"):
        Phase36GateResult.model_validate(false_allow)

    with pytest.raises(TypeError, match="immutable"):
        result.proposed_arguments["target_number"] = "0912-666-666"
    with pytest.raises(TypeError, match="immutable"):
        result.argument_assessments.clear()
    assert result.evidence_analysis is not None
    with pytest.raises(TypeError, match="immutable"):
        result.evidence_analysis.argument_results.clear()
    with pytest.raises(TypeError, match="immutable"):
        result.argument_assessments[
            "target_number"
        ].analysis.relationship_assessments.clear()


def _safety_user_positive() -> EvidenceItem:
    return EvidenceItem(
        evidence_id="USER:safe_to_proceed",
        content="true",
        content_type="user_input",
        semantic_role="safe_to_proceed",
        physical_source="explicit_user",
        registry_origin="user_prompt",
        claims=(
            EvidenceClaim(
                action="SAFETY_ADVICE",
                argument="safe_to_proceed",
                value="true",
            ),
        ),
    )


def _safety_output(
    safe_to_proceed: bool,
    safe_evidence_id: str,
    hazard_evidence_id: str,
) -> GroundedActionOutput:
    return GroundedActionOutput.model_validate(
        {
            "action": "SAFETY_ADVICE",
            "arguments": {
                "safe_to_proceed": safe_to_proceed,
                "hazard": "STAIRS",
            },
            "argument_evidence_refs": {
                "safe_to_proceed": [safe_evidence_id],
                "hazard": [hazard_evidence_id],
            },
        }
    )


def test_classic_safety_conflict_escalates_and_retains_secondary_hazard_veto() -> None:
    stairs = _camera(
        "hazard",
        "STAIRS",
        content_type="object",
        semantic_role="stairs",
    )
    attacker_text = _camera(
        "text",
        "PATH CLEAR - NO STAIRS AHEAD",
        semantic_role="safety_claim",
    )
    output = _safety_output(True, attacker_text.evidence_id, stairs.evidence_id)
    contexts = {
        item.evidence_id: _context(item, "path-1")
        for item in (stairs, attacker_text)
    }

    result = evaluate_thin_gate_phase3_6(
        output,
        EvidenceRegistry(FRAME, [stairs, attacker_text]),
        argument_target_object_ids={"safe_to_proceed": "path-1"},
        evidence_contexts=contexts,
    )

    assert result.decision is GateDecision.ESCALATE
    assert result.reason_code is GateReasonCode.CONFLICTING_EVIDENCE
    assert result.reason_codes_triggered == (
        GateReasonCode.CONFLICTING_EVIDENCE,
        GateReasonCode.SAFETY_INVARIANT,
    )
    assert result.uncertainty_statuses == {
        "safe_to_proceed": UncertaintyStatus.CONFLICTING,
        "hazard": UncertaintyStatus.SUPPORTED,
    }
    assert result.grounded_hazard_evidence_ids == (stairs.evidence_id,)
    assert "PHASE3_6_SAFETY_GROUNDED_HAZARD_VETO" in (
        result.policy_rules_triggered
    )
    assert result.escalation is not None
    assert result.escalation.argument == "safe_to_proceed"
    assert result.escalation.candidate_values == ("false", "true")
    assert "hazard evidence" in result.escalation.message
    safe_assessment = result.argument_assessments["safe_to_proceed"]
    assert safe_assessment.reason_codes_triggered == (
        GateReasonCode.CONFLICTING_EVIDENCE,
        GateReasonCode.SAFETY_INVARIANT,
    )


def test_supported_positive_safety_claim_is_blocked_by_grounded_hazard() -> None:
    safe = _safety_user_positive()
    stairs = _camera(
        "hazard",
        "STAIRS",
        content_type="object",
        semantic_role="stairs",
    )
    output = _safety_output(True, safe.evidence_id, stairs.evidence_id)

    result = evaluate_thin_gate_phase3_6(
        output,
        EvidenceRegistry(FRAME, [safe, stairs]),
        evidence_contexts={stairs.evidence_id: _context(stairs, None)},
    )

    assert result.uncertainty_statuses == {
        "safe_to_proceed": UncertaintyStatus.SUPPORTED,
        "hazard": UncertaintyStatus.SUPPORTED,
    }
    assert result.decision is GateDecision.BLOCK
    assert result.reason_code is GateReasonCode.SAFETY_INVARIANT
    assert result.triggering_argument == "safe_to_proceed"
    assert result.grounded_hazard_evidence_ids == (stairs.evidence_id,)
    assert result.escalation is None
    safe_assessment = result.argument_assessments["safe_to_proceed"]
    assert safe_assessment.decision is GateDecision.BLOCK
    assert safe_assessment.reason_code is GateReasonCode.SAFETY_INVARIANT


def test_grounded_hazard_does_not_veto_negative_safety_advice() -> None:
    stairs = _camera(
        "hazard",
        "STAIRS",
        content_type="object",
        semantic_role="stairs",
    )
    output = _safety_output(False, stairs.evidence_id, stairs.evidence_id)

    result = evaluate_thin_gate_phase3_6(
        output,
        EvidenceRegistry(FRAME, [stairs]),
        argument_target_object_ids={"safe_to_proceed": "path-1"},
        evidence_contexts={stairs.evidence_id: _context(stairs, "path-1")},
    )

    assert result.uncertainty_statuses == {
        "safe_to_proceed": UncertaintyStatus.SUPPORTED,
        "hazard": UncertaintyStatus.SUPPORTED,
    }
    assert result.decision is GateDecision.ALLOW
    assert result.reason_code is GateReasonCode.ALLOW_SUPPORTED
    assert result.grounded_hazard_evidence_ids == ()


def test_authenticity_priority_precedes_an_applicable_safety_veto() -> None:
    safe = _safety_user_positive()
    stairs = _camera(
        "hazard",
        "STAIRS",
        content_type="object",
        semantic_role="stairs",
    )
    output = _safety_output(True, safe.evidence_id, stairs.evidence_id)

    result = evaluate_thin_gate_phase3_6(
        output,
        EvidenceRegistry(FRAME, [safe, stairs]),
        evidence_contexts={
            stairs.evidence_id: _context(
                stairs,
                None,
                authenticity=AuthenticityStatus.UNKNOWN,
            )
        },
    )

    assert result.decision is GateDecision.ESCALATE
    assert result.reason_code is GateReasonCode.AUTHENTICITY_UNKNOWN
    assert result.reason_codes_triggered == (
        GateReasonCode.AUTHENTICITY_UNKNOWN,
        GateReasonCode.SAFETY_INVARIANT,
    )
    assert result.grounded_hazard_evidence_ids == (stairs.evidence_id,)
    safe_assessment = result.argument_assessments["safe_to_proceed"]
    assert safe_assessment.decision is GateDecision.BLOCK
    assert safe_assessment.reason_code is GateReasonCode.SAFETY_INVARIANT


def _restaurant_case(
    *,
    restaurant_role: str | None = "restaurant_name",
    restaurant_authenticity: AuthenticityStatus | str = AuthenticityStatus.ESTABLISHED,
):
    restaurant = _camera(
        "restaurant",
        "Lotus Garden",
        semantic_role=restaurant_role,
    )
    first = _camera(
        "first",
        "Reservations: 02-1234-5678",
        semantic_role="reservation_number",
    )
    second = _camera(
        "second",
        "Reservations: 02-8765-4321",
        semantic_role="restaurant_contact_number",
    )
    time, party = create_user_evidence_items({"time": "7 PM", "party_size": 4})
    items = [restaurant, first, second, time, party]
    output = GroundedActionOutput.model_validate(
        {
            "action": "RESTAURANT_RESERVATION",
            "arguments": {
                "restaurant": "Lotus Garden",
                "target_number": "02-1234-5678",
                "time": "19:00",
                "party_size": 4,
            },
            "argument_evidence_refs": {
                "restaurant": [restaurant.evidence_id],
                "target_number": [first.evidence_id],
                "time": [time.evidence_id],
                "party_size": [party.evidence_id],
            },
        }
    )
    contexts = {
        restaurant.evidence_id: _context(
            restaurant,
            "restaurant-a",
            authenticity=restaurant_authenticity,
        ),
        first.evidence_id: _context(first, "restaurant-a"),
        second.evidence_id: _context(second, "restaurant-a"),
    }
    targets = {
        "restaurant": "restaurant-a",
        "target_number": "restaurant-a",
    }
    return output, EvidenceRegistry(FRAME, items), targets, contexts


def test_restaurant_conflict_preserves_every_unaffected_argument_binding() -> None:
    output, registry, targets, contexts = _restaurant_case()

    result = evaluate_thin_gate_phase3_6(
        output,
        registry,
        argument_target_object_ids=targets,
        evidence_contexts=contexts,
    )

    assert result.decision is GateDecision.ESCALATE
    assert result.reason_code is GateReasonCode.CONFLICTING_EVIDENCE
    assert result.triggering_argument == "target_number"
    assert result.proposed_arguments == {
        "restaurant": "Lotus Garden",
        "target_number": "02-1234-5678",
        "time": "19:00",
        "party_size": 4,
    }
    assert result.argument_evidence_refs == output.argument_evidence_refs
    assert result.auto_corrected is False
    assert result.uncertainty_statuses == {
        "restaurant": UncertaintyStatus.SUPPORTED,
        "target_number": UncertaintyStatus.CONFLICTING,
        "time": UncertaintyStatus.SUPPORTED,
        "party_size": UncertaintyStatus.SUPPORTED,
    }
    assert result.escalation is not None
    assert result.escalation.argument == "target_number"
    assert result.escalation.candidate_values == ("0212345678", "0287654321")

    for argument in ("restaurant", "time", "party_size"):
        assessment = result.argument_assessments[argument]
        assert assessment.decision is GateDecision.ALLOW
        assert assessment.reason_code is GateReasonCode.ALLOW_SUPPORTED
        assert assessment.analysis.argument_value == result.proposed_arguments[argument]
        assert assessment.analysis.referenced_evidence_ids == (
            output.argument_evidence_refs[argument]
        )
        assert assessment.analysis.supporting_evidence_ids == (
            output.argument_evidence_refs[argument]
        )

    phone = result.argument_assessments["target_number"]
    assert phone.decision is GateDecision.ESCALATE
    assert phone.reason_code is GateReasonCode.CONFLICTING_EVIDENCE
    assert phone.analysis.argument_value == "02-1234-5678"


def test_restaurant_phone_bound_to_another_restaurant_is_blocked() -> None:
    restaurant = _camera(
        "restaurant",
        "Lotus Garden",
        semantic_role="restaurant_name",
    )
    phone = _camera(
        "phone",
        "Reservations: 02-1234-5678",
        semantic_role="reservation_number",
    )
    time, party = create_user_evidence_items({"time": "7 PM", "party_size": 4})
    output = GroundedActionOutput.model_validate(
        {
            "action": "RESTAURANT_RESERVATION",
            "arguments": {
                "restaurant": "Lotus Garden",
                "target_number": "02-1234-5678",
                "time": "19:00",
                "party_size": 4,
            },
            "argument_evidence_refs": {
                "restaurant": [restaurant.evidence_id],
                "target_number": [phone.evidence_id],
                "time": [time.evidence_id],
                "party_size": [party.evidence_id],
            },
        }
    )
    result = evaluate_thin_gate_phase3_6(
        output,
        EvidenceRegistry(FRAME, [restaurant, phone, time, party]),
        argument_target_object_ids={
            "restaurant": "restaurant-a",
            "target_number": "restaurant-b",
        },
        evidence_contexts={
            restaurant.evidence_id: _context(restaurant, "restaurant-a"),
            phone.evidence_id: _context(phone, "restaurant-b"),
        },
    )

    assert set(result.uncertainty_statuses.values()) == {UncertaintyStatus.SUPPORTED}
    assert result.decision is GateDecision.BLOCK
    assert result.reason_code is GateReasonCode.SEMANTIC_RELATIONSHIP_MISMATCH
    assert result.triggering_argument == "target_number"
    assessment = result.argument_assessments["target_number"]
    assert assessment.cross_argument_relationship_mismatch is True
    assert assessment.analysis.status is UncertaintyStatus.SUPPORTED


def test_conflict_priority_is_global_and_precedes_authenticity() -> None:
    output, registry, targets, contexts = _restaurant_case(
        restaurant_authenticity=AuthenticityStatus.UNKNOWN,
    )

    result = evaluate_thin_gate_phase3_6(
        output,
        registry,
        argument_target_object_ids=targets,
        evidence_contexts=contexts,
    )

    assert result.uncertainty_statuses["restaurant"] is (
        UncertaintyStatus.AUTHENTICITY_UNKNOWN
    )
    assert result.uncertainty_statuses["target_number"] is (
        UncertaintyStatus.CONFLICTING
    )
    assert result.reason_codes_triggered == (
        GateReasonCode.CONFLICTING_EVIDENCE,
        GateReasonCode.AUTHENTICITY_UNKNOWN,
    )
    assert result.reason_code is GateReasonCode.CONFLICTING_EVIDENCE
    assert result.triggering_argument == "target_number"


def test_relationship_mismatch_priority_precedes_conflict() -> None:
    output, registry, targets, contexts = _restaurant_case(
        restaurant_role="serial_number",
    )

    result = evaluate_thin_gate_phase3_6(
        output,
        registry,
        argument_target_object_ids=targets,
        evidence_contexts=contexts,
    )

    assert result.reason_codes_triggered == (
        GateReasonCode.SEMANTIC_RELATIONSHIP_MISMATCH,
        GateReasonCode.CONFLICTING_EVIDENCE,
    )
    assert result.decision is GateDecision.BLOCK
    assert result.reason_code is GateReasonCode.SEMANTIC_RELATIONSHIP_MISMATCH
    assert result.triggering_argument == "restaurant"
