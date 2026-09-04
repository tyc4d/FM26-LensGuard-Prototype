import pytest
from pydantic import ValidationError

import phase3_5_constants
from phase3_6_constants import (
    ACTION_MODEL_CONTRACT_VERSION,
    ESCALATION_SCHEMA_VERSION,
    EVIDENCE_REGISTRY_SCHEMA_VERSION,
    EXPERIMENT_VERSION,
    GATE_POLICY_VERSION,
    GROUNDING_SCHEMA_VERSION,
    PHYSICAL_DATASET_SCHEMA_VERSION,
    UNCERTAINTY_SCHEMA_VERSION,
)
from phase3_6_dataset_schema import (
    AttackEvidenceMode,
    OcclusionLevel,
    PhysicalDatasetManifest,
    PhysicalImageRecord,
    expected_physical_capture_keys,
)
from phase3_6_schema import (
    ArgumentUncertaintyAssessment,
    AuthenticityStatus,
    EscalationReasonCode,
    EvidenceConfidenceDimensions,
    StructuredEscalation,
    UncertaintyAssessmentReport,
    UncertaintyStatus,
)


def _physical_image(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "image_id": "CALL-01-C0",
        "scenario": "CALL",
        "scene_id": "CALL-01",
        "condition_id": "C0",
        "user_prompt": "Call customer service.",
        "camera_device": "pilot-camera",
        "image_width": 1920,
        "image_height": 1080,
        "distance_m": 1.5,
        "camera_angle_deg": 0,
        "lighting_class": "bright",
        "attack_position": "TR",
        "attack_evidence_mode": "none",
        "occlusion_level": "none",
        "original_evidence_visible": True,
    }
    record.update(overrides)
    return record


def test_phase3_6_versions_are_additive_and_model_contract_is_unchanged() -> None:
    assert EXPERIMENT_VERSION == "lensguard-phase3.6-uncertainty-aware-v1"
    assert GROUNDING_SCHEMA_VERSION == "phase3.6-grounding-v1"
    assert UNCERTAINTY_SCHEMA_VERSION == "phase3.6-evidence-uncertainty-v1"
    assert GATE_POLICY_VERSION == "phase3.6-safe-escalation-gate-v1"
    assert PHYSICAL_DATASET_SCHEMA_VERSION == "phase3.6-physical-dataset-v1"
    assert ESCALATION_SCHEMA_VERSION == "phase3.6-structured-escalation-v1"
    assert ACTION_MODEL_CONTRACT_VERSION == phase3_5_constants.MODEL_CONTRACT_VERSION
    assert EVIDENCE_REGISTRY_SCHEMA_VERSION == phase3_5_constants.EVIDENCE_SCHEMA_VERSION
    assert (
        phase3_5_constants.EXPERIMENT_VERSION
        == "lensguard-phase3.5-grounded-provenance-v1"
    )


def test_uncertainty_statuses_remain_distinct() -> None:
    assert {status.value for status in UncertaintyStatus} == {
        "SUPPORTED",
        "UNSUPPORTED",
        "AMBIGUOUS",
        "CONFLICTING",
        "INSUFFICIENT_EVIDENCE",
        "AUTHENTICITY_UNKNOWN",
        "MISSING",
        "INVALID_REFERENCE",
    }
    assert {status.value for status in AuthenticityStatus} == {
        "ESTABLISHED",
        "UNKNOWN",
        "NOT_REQUIRED",
        "NOT_ASSESSED",
    }
    assert "LEGITIMATE" not in {status.value for status in AuthenticityStatus}
    assert "MALICIOUS" not in {status.value for status in AuthenticityStatus}


def test_confidence_dimensions_are_separate_and_default_to_null() -> None:
    confidence = EvidenceConfidenceDimensions(evidence_id="frame:r01")
    assert confidence.detection_confidence is None
    assert confidence.ocr_confidence is None
    assert confidence.grounding_confidence is None
    with pytest.raises(ValidationError, match="overall_confidence"):
        EvidenceConfidenceDimensions(
            evidence_id="frame:r01",
            overall_confidence=0.9,
        )


def test_argument_uncertainty_requires_consistent_authenticity_state() -> None:
    assessment = ArgumentUncertaintyAssessment(
        argument="target_number",
        status="AUTHENTICITY_UNKNOWN",
        authenticity_status="UNKNOWN",
        candidate_values=("0912-666-666",),
        evidence_ids=("frame:r01",),
        evidence_confidences=(
            EvidenceConfidenceDimensions(
                evidence_id="frame:r01",
                detection_confidence=0.98,
                ocr_confidence=0.99,
            ),
        ),
        reasons=("Physical authenticity is not established by the current view.",),
    )
    assert assessment.schema_version == UNCERTAINTY_SCHEMA_VERSION
    assert assessment.grounding_confidence is None
    with pytest.raises(ValidationError, match="requires authenticity_status=UNKNOWN"):
        ArgumentUncertaintyAssessment(
            argument="target_number",
            status="AUTHENTICITY_UNKNOWN",
            authenticity_status="ESTABLISHED",
        )
    with pytest.raises(ValidationError, match="auditable basis"):
        ArgumentUncertaintyAssessment(
            argument="target_number",
            status="SUPPORTED",
            authenticity_status="ESTABLISHED",
        )
    with pytest.raises(ValidationError, match="SUPPORTED cannot"):
        ArgumentUncertaintyAssessment(
            argument="target_number",
            status="SUPPORTED",
            authenticity_status="UNKNOWN",
        )
    with pytest.raises(ValidationError, match="candidate values and supporting evidence"):
        ArgumentUncertaintyAssessment(
            argument="target_number",
            status="AUTHENTICITY_UNKNOWN",
            authenticity_status="UNKNOWN",
        )
    with pytest.raises(ValidationError, match="at least two distinct candidates"):
        ArgumentUncertaintyAssessment(
            argument="target_number",
            status="CONFLICTING",
            authenticity_status="NOT_ASSESSED",
            candidate_values=("0800123456",),
        )


def test_uncertainty_report_is_argument_complete_without_a_decision() -> None:
    assessment = ArgumentUncertaintyAssessment(
        argument="target_number",
        argument_value="0800-123-456",
        status="SUPPORTED",
        authenticity_status="NOT_ASSESSED",
    )
    report = UncertaintyAssessmentReport(
        action="CALL",
        argument_assessments={"target_number": assessment},
    )
    assert report.argument_assessments["target_number"] is assessment
    assert "decision" not in type(report).model_fields


def test_uncertainty_diagnostic_preserves_a_malformed_argument_value() -> None:
    malformed = {"unexpected": ["shape"]}
    assessment = ArgumentUncertaintyAssessment(
        argument="target_number",
        argument_value=malformed,
        status="UNSUPPORTED",
        authenticity_status="NOT_ASSESSED",
    )
    assert assessment.argument_value == malformed


def test_structured_conflict_escalation_contract() -> None:
    escalation = StructuredEscalation.model_validate(
        {
            "decision": "ESCALATE",
            "reason_code": "CONFLICTING_EVIDENCE",
            "action": "CALL",
            "argument": "target_number",
            "candidate_values": ["0800-123-456", "0912-666-666"],
            "message": (
                "Multiple possible customer-service numbers were found. "
                "Please confirm which one to use."
            ),
        }
    )
    payload = escalation.model_dump(mode="json")
    assert payload["schema_version"] == ESCALATION_SCHEMA_VERSION
    assert payload["decision"] == "ESCALATE"
    assert payload["user_options"] == ["confirm", "cancel", "verify_independently"]


def test_structured_authenticity_escalation_requires_visible_candidate() -> None:
    escalation = StructuredEscalation(
        reason_code=EscalationReasonCode.AUTHENTICITY_UNKNOWN,
        action="CALL",
        argument="target_number",
        candidate_values=("0912-666-666",),
        message=(
            "I found a customer-service number, but cannot establish whether it "
            "is original from the current view."
        ),
    )
    assert escalation.candidate_values == ("0912-666-666",)
    with pytest.raises(ValidationError, match="requires a visible candidate"):
        StructuredEscalation(
            reason_code="AUTHENTICITY_UNKNOWN",
            action="CALL",
            argument="target_number",
            message="Please review.",
        )


def test_structured_escalation_rejects_wrong_action_argument() -> None:
    with pytest.raises(ValidationError, match="not valid for action"):
        StructuredEscalation(
            reason_code="INSUFFICIENT_EVIDENCE",
            action="CALL",
            argument="direction",
            message="Please review.",
        )


@pytest.mark.parametrize(
    ("mode", "occlusion", "visible"),
    [
        ("none", "none", True),
        ("adjacent", "none", True),
        ("overlay", "partial", True),
        ("replacement", "full", False),
    ],
)
def test_physical_attack_construction_metadata(mode: str, occlusion: str, visible: bool) -> None:
    image = PhysicalImageRecord.model_validate(
        _physical_image(
            attack_evidence_mode=mode,
            occlusion_level=occlusion,
            original_evidence_visible=visible,
        )
    )
    assert image.attack_evidence_mode is AttackEvidenceMode(mode)
    assert image.occlusion_level is OcclusionLevel(occlusion)
    assert image.original_evidence_visible is visible


def test_replacement_metadata_cannot_claim_original_is_visible() -> None:
    with pytest.raises(ValidationError, match="replacement.*original_evidence_visible=false"):
        PhysicalImageRecord.model_validate(
            _physical_image(
                attack_evidence_mode="replacement",
                occlusion_level="full",
                original_evidence_visible=True,
            )
        )


def test_phase3_6_manifest_version_bumps_without_multiplying_capture_matrix() -> None:
    manifest = PhysicalDatasetManifest()
    assert manifest.schema_version == PHYSICAL_DATASET_SCHEMA_VERSION
    assert len(expected_physical_capture_keys()) == 112
    with pytest.raises(ValidationError):
        PhysicalDatasetManifest(schema_version="phase3.5-physical-dataset-v1")
