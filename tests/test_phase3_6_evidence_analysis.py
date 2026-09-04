from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from phase3_5_schema import GroundedActionOutput
from phase3_6_schema import AuthenticityStatus, UncertaintyStatus
from provenance.evidence_analysis_phase3_6 import (
    ArgumentEvidenceAnalysis,
    EvidenceAnalysisContext,
    EvidenceRelationshipAssessment,
    RelationshipFinding,
    analyze_evidence_uncertainty,
    normalize_candidate_value,
)
from provenance.evidence_registry_phase3_5 import (
    EvidenceClaim,
    EvidenceItem,
    EvidenceRegistry,
    canonical_evidence_id,
    create_user_evidence_items,
)


FRAME = "phase36-fixture"


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
    target: str,
    *,
    authenticity: str = "ESTABLISHED",
) -> EvidenceAnalysisContext:
    return EvidenceAnalysisContext(
        evidence_id=item.evidence_id,
        associated_target_object_id=target,
        authenticity_status=authenticity,
        authenticity_basis=(
            "deterministic fixture authentication"
            if authenticity == "ESTABLISHED"
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


def _analyze_call(
    items: list[EvidenceItem],
    *,
    selected: EvidenceItem | None = None,
    contexts: dict[str, EvidenceAnalysisContext] | None = None,
):
    selected = items[0] if selected is None else selected
    return analyze_evidence_uncertainty(
        _call_output("0800-123-456", selected.evidence_id),
        EvidenceRegistry(FRAME, items),
        argument_target_object_ids={"target_number": "product-1"},
        evidence_contexts=(
            contexts
            if contexts is not None
            else {item.evidence_id: _context(item, "product-1") for item in items}
        ),
    ).argument_results["target_number"]


def test_candidate_normalization_is_narrow_and_deterministic() -> None:
    assert normalize_candidate_value("CALL", "target_number", "0800-123-456") == (
        "0800123456"
    )
    assert normalize_candidate_value("CALL", "target_number", "(0800) 123 456") == (
        "0800123456"
    )
    assert normalize_candidate_value("CALL", "target_number", "０８００１２３４５６") == (
        "0800123456"
    )
    assert normalize_candidate_value("CALL", "target_number", "+1 800 123 456") != (
        normalize_candidate_value("CALL", "target_number", "1 800 123 456")
    )
    with pytest.raises(ValueError):
        normalize_candidate_value("CALL", "target_number", "0800-123-456 ext 9")


def test_call_one_candidate_is_supported_with_four_relationship_findings() -> None:
    service = _camera(
        "service", "Customer Service: 0800-123-456", semantic_role="customer_service_number"
    )
    result = _analyze_call([service])
    relationship = result.relationship_assessments[service.evidence_id]
    assert result.status is UncertaintyStatus.SUPPORTED
    assert result.normalized_argument_value == "0800123456"
    assert result.conflict_set.distinct_values == ("0800123456",)
    assert relationship.value_relationship is RelationshipFinding.MATCH
    assert relationship.target_object_relationship is RelationshipFinding.MATCH
    assert relationship.semantic_role_relationship is RelationshipFinding.MATCH
    assert relationship.argument_relationship is RelationshipFinding.MATCH
    assert relationship.task_context_satisfied is True


def test_call_two_distinct_plausible_numbers_form_a_conflict() -> None:
    original = _camera(
        "original", "Customer Service: 0800-123-456", semantic_role="customer_service_number"
    )
    alternate = _camera(
        "alternate", "Customer Service: 0912-666-666", semantic_role="contact_number"
    )
    result = _analyze_call([alternate, original], selected=original)
    assert result.status is UncertaintyStatus.CONFLICTING
    assert result.conflict_set.has_conflict is True
    assert result.conflict_set.distinct_values == ("0800123456", "0912666666")
    assert [candidate.evidence_ids for candidate in result.conflict_set.candidates] == [
        (original.evidence_id,),
        (alternate.evidence_id,),
    ]


def test_two_selected_plausible_numbers_remain_a_conflict() -> None:
    original = _camera(
        "original", "Customer Service: 0800-123-456", semantic_role="customer_service_number"
    )
    alternate = _camera(
        "alternate", "Customer Service: 0912-666-666", semantic_role="contact_number"
    )
    analysis = analyze_evidence_uncertainty(
        GroundedActionOutput.model_validate(
            {
                "action": "CALL",
                "arguments": {"target_number": "0800-123-456"},
                "argument_evidence_refs": {
                    "target_number": [original.evidence_id, alternate.evidence_id]
                },
            }
        ),
        EvidenceRegistry(FRAME, [original, alternate]),
        argument_target_object_ids={"target_number": "product-1"},
        evidence_contexts={
            item.evidence_id: _context(item, "product-1")
            for item in (original, alternate)
        },
    )
    result = analysis.argument_results["target_number"]
    assert result.status is UncertaintyStatus.CONFLICTING
    assert result.conflict_set.distinct_values == ("0800123456", "0912666666")


def test_conflict_keeps_unselected_authenticity_unknown_inspectable() -> None:
    first = _camera(
        "first", "Customer Service: 0800-123-456", semantic_role="customer_service_number"
    )
    second = _camera(
        "second", "Customer Service: 0912-666-666", semantic_role="contact_number"
    )
    contexts = {
        first.evidence_id: _context(first, "product-1"),
        second.evidence_id: _context(second, "product-1", authenticity="UNKNOWN"),
    }
    result = _analyze_call([first, second], contexts=contexts)
    assert result.status is UncertaintyStatus.CONFLICTING
    assert result.uncertainty.authenticity_status is AuthenticityStatus.UNKNOWN


def test_equivalent_duplicate_numbers_do_not_form_a_conflict() -> None:
    first = _camera(
        "first", "Customer Service: 0800-123-456", semantic_role="customer_service_number"
    )
    second = _camera(
        "second", "Contact: (0800) 123 456", semantic_role="contact_number"
    )
    result = _analyze_call([second, first], selected=first)
    assert result.status is UncertaintyStatus.SUPPORTED
    assert result.conflict_set.has_conflict is False
    assert result.conflict_set.candidates[0].evidence_ids == (
        first.evidence_id,
        second.evidence_id,
    )


def test_unrelated_phone_evidence_is_not_a_plausible_conflict_candidate() -> None:
    service = _camera(
        "service", "Customer Service: 0800-123-456", semantic_role="customer_service_number"
    )
    serial = _camera(
        "serial", "Device serial: 0912-666-666", semantic_role="serial_number"
    )
    result = _analyze_call([service, serial])
    unrelated = result.relationship_assessments[serial.evidence_id]
    assert result.status is UncertaintyStatus.SUPPORTED
    assert result.conflict_set.distinct_values == ("0800123456",)
    assert unrelated.value_relationship is RelationshipFinding.MISMATCH
    assert unrelated.semantic_role_relationship is RelationshipFinding.MISMATCH
    assert unrelated.plausible_candidate is False


def test_phone_bound_to_a_different_target_is_excluded_from_conflict() -> None:
    service = _camera(
        "service", "Customer Service: 0800-123-456", semantic_role="customer_service_number"
    )
    other_product = _camera(
        "other", "Customer Service: 0912-666-666", semantic_role="customer_service_number"
    )
    contexts = {
        service.evidence_id: _context(service, "product-1"),
        other_product.evidence_id: _context(other_product, "product-2"),
    }
    result = _analyze_call([service, other_product], contexts=contexts)
    relationship = result.relationship_assessments[other_product.evidence_id]
    assert result.status is UncertaintyStatus.SUPPORTED
    assert result.conflict_set.distinct_values == ("0800123456",)
    assert relationship.target_object_relationship is RelationshipFinding.MISMATCH
    assert relationship.plausible_candidate is False


def test_selected_number_with_wrong_semantic_role_is_unsupported() -> None:
    serial = _camera(
        "serial", "Device serial: 0800-123-456", semantic_role="serial_number"
    )
    result = _analyze_call([serial])
    relationship = result.relationship_assessments[serial.evidence_id]
    assert result.status is UncertaintyStatus.UNSUPPORTED
    assert relationship.value_relationship is RelationshipFinding.MATCH
    assert relationship.semantic_role_relationship is RelationshipFinding.MISMATCH


def test_missing_and_invalid_references_remain_distinct() -> None:
    service = _camera(
        "service", "Customer Service: 0800-123-456", semantic_role="customer_service_number"
    )
    registry = EvidenceRegistry(FRAME, [service])
    missing = analyze_evidence_uncertainty(
        {
            "action": "CALL",
            "arguments": {"target_number": "0800-123-456"},
            "argument_evidence_refs": {},
        },
        registry,
    )
    invalid = analyze_evidence_uncertainty(
        {
            "action": "CALL",
            "arguments": {"target_number": "0800-123-456"},
            "argument_evidence_refs": {"target_number": [f"{FRAME}:unknown"]},
        },
        registry,
    )
    assert missing.argument_results["target_number"].status is UncertaintyStatus.MISSING
    assert (
        invalid.argument_results["target_number"].status
        is UncertaintyStatus.INVALID_REFERENCE
    )
    assert not missing.argument_results["target_number"].relationship_assessments
    assert not invalid.argument_results["target_number"].relationship_assessments


def test_wrong_requested_frame_returns_invalid_reference_analysis() -> None:
    service = _camera(
        "service", "Customer Service: 0800-123-456", semantic_role="customer_service_number"
    )
    analysis = analyze_evidence_uncertainty(
        _call_output("0800-123-456", service.evidence_id),
        EvidenceRegistry(FRAME, [service]),
        frame_id="different-frame",
    )
    assert analysis.frame_id == "different-frame"
    assert not analysis.reference_validation.contract_valid
    assert (
        analysis.argument_results["target_number"].status
        is UncertaintyStatus.INVALID_REFERENCE
    )


def test_generic_phone_role_is_not_customer_service_relationship() -> None:
    generic = _camera(
        "generic", "Phone: 0800-123-456", semantic_role="phone_number"
    )
    result = _analyze_call([generic])
    relationship = result.relationship_assessments[generic.evidence_id]
    assert result.status is UncertaintyStatus.UNSUPPORTED
    assert relationship.value_relationship is RelationshipFinding.MATCH
    assert relationship.semantic_role_relationship is RelationshipFinding.MISMATCH


def test_missing_target_association_is_insufficient_not_a_mismatch() -> None:
    service = _camera(
        "service", "Customer Service: 0800-123-456", semantic_role="customer_service_number"
    )
    result = _analyze_call([service], contexts={})
    relationship = result.relationship_assessments[service.evidence_id]
    assert result.status is UncertaintyStatus.INSUFFICIENT_EVIDENCE
    assert relationship.target_object_relationship is RelationshipFinding.NOT_ASSESSED


def test_unreadable_value_is_insufficient_not_unsupported() -> None:
    unreadable = _camera(
        "unreadable", "Customer Service: [unreadable]", semantic_role="customer_service_number"
    )
    result = _analyze_call([unreadable])
    relationship = result.relationship_assessments[unreadable.evidence_id]
    assert result.status is UncertaintyStatus.INSUFFICIENT_EVIDENCE
    assert relationship.value_relationship is RelationshipFinding.NOT_ASSESSED
    assert relationship.target_object_relationship is RelationshipFinding.MATCH
    assert relationship.semantic_role_relationship is RelationshipFinding.MATCH


def test_authenticity_unknown_is_explicit_and_not_inferred_from_source_or_confidence() -> None:
    visible = _camera(
        "visible",
        "Customer Service: 0800-123-456",
        semantic_role="customer_service_number",
        physical_source="attacker_sticker",
        control_class="attacker_controlled",
        supports_ground_truth=False,
        detection_confidence=0.01,
        ocr_confidence=0.01,
        grounding_confidence=0.99,
    )
    unassessed_context = {
        visible.evidence_id: _context(
            visible, "product-1", authenticity="NOT_ASSESSED"
        )
    }
    unresolved = _analyze_call([visible], contexts=unassessed_context)
    assert unresolved.status is UncertaintyStatus.AUTHENTICITY_UNKNOWN
    assert unresolved.uncertainty.authenticity_status is AuthenticityStatus.UNKNOWN

    unknown_context = {
        visible.evidence_id: _context(visible, "product-1", authenticity="UNKNOWN")
    }
    unknown = _analyze_call([visible], contexts=unknown_context)
    assert unknown.status is UncertaintyStatus.AUTHENTICITY_UNKNOWN
    assert unknown.uncertainty.authenticity_status is AuthenticityStatus.UNKNOWN
    assert unknown.uncertainty.grounding_confidence is None
    [dimensions] = unknown.uncertainty.evidence_confidences
    assert dimensions.detection_confidence == 0.01
    assert dimensions.ocr_confidence == 0.01
    assert dimensions.grounding_confidence is None


def test_claim_only_evidence_cannot_establish_role_or_target_relationship() -> None:
    first = _camera(
        "first",
        "Customer Service: 0800-123-456",
        semantic_role=None,
        claims=(
            EvidenceClaim(
                action="CALL",
                argument="target_number",
                value="0800-123-456",
                claim_role="verified_reference",
            ),
        ),
    )
    analysis = analyze_evidence_uncertainty(
        _call_output("0800-123-456", first.evidence_id),
        EvidenceRegistry(FRAME, [first]),
        argument_target_object_ids={"target_number": "legacy-task-target"},
    )
    result = analysis.argument_results["target_number"]
    relationship = result.relationship_assessments[first.evidence_id]
    assert result.status is UncertaintyStatus.INSUFFICIENT_EVIDENCE
    assert result.conflict_set.distinct_values == ()
    assert relationship.semantic_role_relationship is RelationshipFinding.NOT_ASSESSED
    assert relationship.target_object_relationship is RelationshipFinding.NOT_ASSESSED
    assert relationship.argument_relationship is RelationshipFinding.MATCH
    assert result.uncertainty.authenticity_status is AuthenticityStatus.NOT_ASSESSED


def test_claim_role_labels_do_not_rank_task_valid_conflicting_candidates() -> None:
    first = _camera(
        "first",
        "Customer Service: 0800-123-456",
        semantic_role="customer_service_number",
        claims=(
            EvidenceClaim(
                action="CALL",
                argument="target_number",
                value="0800-123-456",
                claim_role="verified_reference",
            ),
        ),
    )
    second = _camera(
        "second",
        "Customer Service: 0912-666-666",
        semantic_role="customer_service_number",
        claims=(
            EvidenceClaim(
                action="CALL",
                argument="target_number",
                value="0912-666-666",
                claim_role="attacker_selected_alternate",
            ),
        ),
    )
    contexts = {
        item.evidence_id: _context(item, "product-1") for item in (first, second)
    }
    result = _analyze_call([first, second], contexts=contexts)
    assert result.status is UncertaintyStatus.CONFLICTING
    assert result.conflict_set.distinct_values == ("0800123456", "0912666666")


def test_claim_does_not_suppress_a_second_visible_candidate() -> None:
    mixed = _camera(
        "mixed",
        "Customer Service: 0800-123-456 or 0912-666-666",
        semantic_role="customer_service_number",
        claims=(
            EvidenceClaim(
                action="CALL",
                argument="target_number",
                value="0800-123-456",
                claim_role="verified_reference",
            ),
        ),
    )
    result = _analyze_call([mixed])
    assert result.status is UncertaintyStatus.CONFLICTING
    assert result.conflict_set.distinct_values == ("0800123456", "0912666666")


def test_explicit_user_value_scopes_out_camera_conflict_candidates() -> None:
    [user_number] = create_user_evidence_items({"target_number": "0800-123-456"})
    camera = _camera(
        "camera", "Customer Service: 0912-666-666", semantic_role="customer_service_number"
    )
    output = _call_output("0800-123-456", user_number.evidence_id)
    analysis = analyze_evidence_uncertainty(
        output,
        EvidenceRegistry(FRAME, [camera, user_number]),
        argument_target_object_ids={"target_number": "product-1"},
        evidence_contexts={camera.evidence_id: _context(camera, "product-1")},
    )
    result = analysis.argument_results["target_number"]
    assert result.status is UncertaintyStatus.SUPPORTED
    assert result.conflict_set.distinct_values == ("0800123456",)
    assert result.supporting_evidence_ids == (user_number.evidence_id,)


def test_camera_context_cannot_mark_authenticity_not_required() -> None:
    camera = _camera(
        "camera", "Customer Service: 0800-123-456", semantic_role="customer_service_number"
    )
    with pytest.raises(ValueError, match="reserved for USER evidence"):
        EvidenceAnalysisContext(
            evidence_id=camera.evidence_id,
            associated_target_object_id="product-1",
            authenticity_status="NOT_REQUIRED",
        )


def test_serialized_registry_is_revalidated_before_user_override_scoping() -> None:
    [user_number] = create_user_evidence_items({"target_number": "0800-123-456"})
    camera = _camera(
        "camera", "Customer Service: 0912-666-666", semantic_role="customer_service_number"
    )
    snapshot = EvidenceRegistry(FRAME, [user_number, camera]).model_dump()
    snapshot["items"][0]["registry_origin"] = "automatic_perception"
    with pytest.raises(ValidationError, match="USER evidence must have registry_origin"):
        analyze_evidence_uncertainty(
            _call_output("0800-123-456", user_number.evidence_id),
            snapshot,
            argument_target_object_ids={"target_number": "product-1"},
            evidence_contexts={camera.evidence_id: _context(camera, "product-1")},
        )


def test_serialized_registry_rejects_duplicate_evidence_ids() -> None:
    service = _camera(
        "service", "Customer Service: 0800-123-456", semantic_role="customer_service_number"
    )
    snapshot = EvidenceRegistry(FRAME, [service]).model_dump()
    snapshot["items"].append(dict(snapshot["items"][0]))
    with pytest.raises(ValueError, match="duplicate evidence_id"):
        analyze_evidence_uncertainty(
            _call_output("0800-123-456", service.evidence_id), snapshot
        )


def test_relationship_result_models_reject_contradictory_derived_fields() -> None:
    with pytest.raises(ValidationError, match="task_context_satisfied"):
        EvidenceRelationshipAssessment(
            evidence_id="frame:r01",
            candidate_values=("0800123456",),
            content_type_relationship="MATCH",
            value_relationship="MATCH",
            target_object_relationship="MISMATCH",
            semantic_role_relationship="MATCH",
            argument_relationship="MATCH",
            task_context_satisfied=True,
            supports_proposed_argument=True,
            plausible_candidate=True,
        )


def test_argument_analysis_rejects_cross_argument_conflict_set() -> None:
    service = _camera(
        "service", "Customer Service: 0800-123-456", semantic_role="customer_service_number"
    )
    valid = _analyze_call([service])
    payload = valid.model_dump(mode="python")
    payload["conflict_set"]["argument"] = "direction"
    with pytest.raises(ValidationError, match="different argument"):
        ArgumentEvidenceAnalysis.model_validate(payload)


def test_argument_analysis_rejects_conflict_status_without_conflict_set() -> None:
    service = _camera(
        "service", "Customer Service: 0800-123-456", semantic_role="customer_service_number"
    )
    valid = _analyze_call([service])
    payload = valid.model_dump(mode="python")
    payload["uncertainty"]["status"] = "CONFLICTING"
    with pytest.raises(ValidationError, match="at least two distinct candidates"):
        ArgumentEvidenceAnalysis.model_validate(payload)


def test_navigation_distinct_directions_for_same_target_conflict() -> None:
    left = _camera("left", "EXIT LEFT", semantic_role="exit_direction")
    right = _camera("right", "EXIT RIGHT", semantic_role="directional_sign")
    [destination] = create_user_evidence_items({"destination": "EXIT"})
    registry = EvidenceRegistry(FRAME, [right, destination, left])
    output = GroundedActionOutput.model_validate(
        {
            "action": "DIRECTION_ADVICE",
            "arguments": {"direction": "LEFT", "destination": "EXIT"},
            "argument_evidence_refs": {
                "direction": [left.evidence_id],
                "destination": [destination.evidence_id],
            },
        }
    )
    contexts = {
        item.evidence_id: _context(item, "exit-a") for item in (left, right)
    }
    analysis = analyze_evidence_uncertainty(
        output,
        registry,
        argument_target_object_ids={"direction": "exit-a", "destination": "exit-a"},
        evidence_contexts=contexts,
    )
    assert analysis.argument_results["direction"].status is UncertaintyStatus.CONFLICTING
    assert analysis.argument_results["direction"].conflict_set.distinct_values == (
        "LEFT",
        "RIGHT",
    )
    assert analysis.argument_results["destination"].status is UncertaintyStatus.SUPPORTED


def test_navigation_direction_for_another_destination_is_not_a_conflict() -> None:
    exit_left = _camera("exit", "EXIT LEFT", semantic_role="exit_direction")
    platform_right = _camera(
        "platform", "PLATFORM 4 RIGHT", semantic_role="directional_sign"
    )
    [destination] = create_user_evidence_items({"destination": "EXIT"})
    output = GroundedActionOutput.model_validate(
        {
            "action": "DIRECTION_ADVICE",
            "arguments": {"direction": "LEFT", "destination": "EXIT"},
            "argument_evidence_refs": {
                "direction": [exit_left.evidence_id],
                "destination": [destination.evidence_id],
            },
        }
    )
    analysis = analyze_evidence_uncertainty(
        output,
        EvidenceRegistry(FRAME, [platform_right, exit_left, destination]),
        argument_target_object_ids={"direction": "exit-a", "destination": "exit-a"},
        evidence_contexts={
            exit_left.evidence_id: _context(exit_left, "exit-a"),
            platform_right.evidence_id: _context(platform_right, "platform-4"),
        },
    )
    direction = analysis.argument_results["direction"]
    assert direction.status is UncertaintyStatus.SUPPORTED
    assert direction.conflict_set.distinct_values == ("LEFT",)
    other = direction.relationship_assessments[platform_right.evidence_id]
    assert other.target_object_relationship is RelationshipFinding.MISMATCH


def test_destination_label_context_does_not_self_conflict() -> None:
    direction = _camera("direction", "Emergency Exit Left", semantic_role="exit_direction")
    destination = _camera(
        "destination", "Emergency Exit Left", semantic_role="exit_destination"
    )
    output = GroundedActionOutput.model_validate(
        {
            "action": "DIRECTION_ADVICE",
            "arguments": {"direction": "LEFT", "destination": "EXIT"},
            "argument_evidence_refs": {
                "direction": [direction.evidence_id],
                "destination": [destination.evidence_id],
            },
        }
    )
    analysis = analyze_evidence_uncertainty(
        output,
        EvidenceRegistry(FRAME, [direction, destination]),
        argument_target_object_ids={"direction": "exit-a", "destination": "exit-a"},
        evidence_contexts={
            direction.evidence_id: _context(direction, "exit-a"),
            destination.evidence_id: _context(destination, "exit-a"),
        },
    )
    destination_result = analysis.argument_results["destination"]
    assert destination_result.status is UncertaintyStatus.SUPPORTED
    assert destination_result.conflict_set.distinct_values == ("EXIT",)


def test_precomputed_reference_validation_must_match_current_input() -> None:
    first = _camera(
        "first", "Customer Service: 0800-123-456", semantic_role="customer_service_number"
    )
    second = _camera(
        "second", "Customer Service: 0912-666-666", semantic_role="customer_service_number"
    )
    registry = EvidenceRegistry(FRAME, [first, second])
    stale = analyze_evidence_uncertainty(
        _call_output("0800-123-456", first.evidence_id),
        registry,
        argument_target_object_ids={"target_number": "product-1"},
        evidence_contexts={
            item.evidence_id: _context(item, "product-1") for item in (first, second)
        },
    ).reference_validation
    with pytest.raises(ValueError, match="does not match the current output"):
        analyze_evidence_uncertainty(
            _call_output("0912-666-666", second.evidence_id),
            registry,
            reference_validation=stale,
        )


def test_restaurant_conflict_preserves_unaffected_argument_results() -> None:
    restaurant = _camera(
        "restaurant", "Lotus Garden", semantic_role="restaurant_name"
    )
    first = _camera(
        "first", "Reservations: 02-1234-5678", semantic_role="reservation_number"
    )
    second = _camera(
        "second", "Reservations: 02-8765-4321", semantic_role="restaurant_contact_number"
    )
    time, party = create_user_evidence_items({"time": "7 PM", "party_size": 4})
    registry = EvidenceRegistry(FRAME, [restaurant, first, second, time, party])
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
        item.evidence_id: _context(item, "restaurant-a")
        for item in (restaurant, first, second)
    }
    analysis = analyze_evidence_uncertainty(
        output,
        registry,
        argument_target_object_ids={
            "restaurant": "restaurant-a",
            "target_number": "restaurant-a",
        },
        evidence_contexts=contexts,
    )
    assert analysis.statuses == {
        "restaurant": UncertaintyStatus.SUPPORTED,
        "target_number": UncertaintyStatus.CONFLICTING,
        "time": UncertaintyStatus.SUPPORTED,
        "party_size": UncertaintyStatus.SUPPORTED,
    }
    for argument, expected_id in {
        "restaurant": restaurant.evidence_id,
        "time": time.evidence_id,
        "party_size": party.evidence_id,
    }.items():
        result = analysis.argument_results[argument]
        assert result.referenced_evidence_ids == (expected_id,)
        assert result.supporting_evidence_ids == (expected_id,)


def test_restaurant_label_context_does_not_self_conflict() -> None:
    restaurant = _camera(
        "restaurant", "Lotus Garden Reservations", semantic_role="restaurant_name"
    )
    phone = _camera(
        "phone", "Reservations: 02-1234-5678", semantic_role="reservation_number"
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
    analysis = analyze_evidence_uncertainty(
        output,
        EvidenceRegistry(FRAME, [restaurant, phone, time, party]),
        argument_target_object_ids={
            "restaurant": "restaurant-a",
            "target_number": "restaurant-a",
        },
        evidence_contexts={
            restaurant.evidence_id: _context(restaurant, "restaurant-a"),
            phone.evidence_id: _context(phone, "restaurant-a"),
        },
    )
    identity = analysis.argument_results["restaurant"]
    assert identity.status is UncertaintyStatus.SUPPORTED
    assert identity.conflict_set.distinct_values == ("LOTUS GARDEN",)


def test_restaurant_time_requires_exact_user_argument_binding() -> None:
    restaurant = _camera(
        "restaurant", "Lotus Garden", semantic_role="restaurant_name"
    )
    phone = _camera(
        "phone", "Reservations: 02-1234-5678", semantic_role="reservation_number"
    )
    wrong_time, party = create_user_evidence_items(
        {"alternate_time": "19:00", "party_size": 4}
    )
    registry = EvidenceRegistry(FRAME, [restaurant, phone, wrong_time, party])
    output = {
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
            "time": [wrong_time.evidence_id],
            "party_size": [party.evidence_id],
        },
    }
    analysis = analyze_evidence_uncertainty(
        output,
        registry,
        argument_target_object_ids={
            "restaurant": "restaurant-a",
            "target_number": "restaurant-a",
        },
        evidence_contexts={
            restaurant.evidence_id: _context(restaurant, "restaurant-a"),
            phone.evidence_id: _context(phone, "restaurant-a"),
        },
    )
    time_result = analysis.argument_results["time"]
    relationship = time_result.relationship_assessments[wrong_time.evidence_id]
    assert time_result.status is UncertaintyStatus.UNSUPPORTED
    assert relationship.value_relationship is RelationshipFinding.MATCH
    assert relationship.argument_relationship is RelationshipFinding.MISMATCH
