import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from phase3_5_constants import (
    EXPECTED_PHYSICAL_IMAGE_COUNT,
    PHYSICAL_BASE_SCENES,
    PHYSICAL_CONDITION_SPECS,
    PhysicalConditionId,
    ScenarioFamily,
)
from phase3_5_schema import GroundedActionOutput
from phase3_6_constants import (
    PHYSICAL_DATASET_SCHEMA_VERSION,
    PHYSICAL_PILOT_PROTOCOL_VERSION,
)
from phase3_6_dataset_schema import (
    SOFTWARE_VALIDATION_FIXTURE_CLASS,
    _CAMERA_RELATIONSHIPS,
    AttackEvidenceMode,
    PhysicalImageAnnotations,
    PhysicalPilotProtocol,
    PhysicalPilotValidationFixture,
    expected_physical_capture_keys,
    load_physical_pilot_protocol,
    validate_complete_physical_manifest,
    validate_complete_physical_topology,
    validate_physical_annotation_record,
    validate_physical_pilot_manifest,
)
from phase3_6_schema import AuthenticityStatus, UncertaintyStatus
from provenance.evidence_analysis_phase3_6 import (
    analyze_evidence_uncertainty,
    load_relationship_config,
)
from provenance.evidence_registry_phase3_5 import create_user_evidence_items
from provenance.physical_pilot_phase3_6 import (
    Phase36PhysicalAnnotationRegistryAdapter,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "phase3_6_physical_pilot"


def _load_fixture(name: str) -> tuple[dict[str, object], PhysicalPilotValidationFixture]:
    payload = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return payload, PhysicalPilotValidationFixture.model_validate(payload)


def _record_payload(
    scene_id: str,
    condition_id: PhysicalConditionId,
    mode: AttackEvidenceMode,
) -> dict[str, object]:
    condition = PHYSICAL_CONDITION_SPECS[condition_id]
    prefix = scene_id.rsplit("-", 1)[0]
    scenario = {
        "CALL": "CALL",
        "NAV": "NAVIGATION",
        "SAFE": "SAFETY",
        "RESTAURANT": "RESTAURANT_RESERVATION",
    }[prefix]
    target_object_id = f"{scene_id.lower()}-target"
    evidence_role = {
        "CALL": "customer_service_number",
        "NAV": "directional_sign",
        "SAFE": "hazard",
        "RESTAURANT": "restaurant_contact_number",
    }[prefix]
    occlusion, visible = {
        AttackEvidenceMode.NONE: ("none", True),
        AttackEvidenceMode.ADJACENT: ("none", True),
        AttackEvidenceMode.OVERLAY: ("partial", True),
        AttackEvidenceMode.REPLACEMENT: ("full", False),
    }[mode]
    regions: list[dict[str, object]] = []
    original_ids: list[str] = []
    if visible:
        regions.append(
            {
                "region_id": "r01",
                "associated_target_object_id": target_object_id,
                "bbox": [0.1, 0.1, 0.45, 0.25],
                "region_type": "printed_text",
                "semantic_role": evidence_role,
                "ground_truth_text": "Original evidence",
                "physical_source": "official_sign",
                "control_class": "legitimate",
                "supports_ground_truth": True,
            }
        )
        original_ids.append("r01")
    attack_ids: list[str] = []
    if mode is not AttackEvidenceMode.NONE:
        regions.append(
            {
                "region_id": "r02",
                "associated_target_object_id": target_object_id,
                "bbox": [0.5, 0.1, 0.85, 0.25],
                "region_type": "printed_text",
                "semantic_role": (
                    "safety_claim" if prefix == "SAFE" else evidence_role
                ),
                "ground_truth_text": "Constructed evidence",
                "physical_source": "attacker_sticker",
                "control_class": "attacker_controlled",
                "supports_ground_truth": False,
            }
        )
        attack_ids.append("r02")
    if prefix == "SAFE":
        regions.append(
            {
                "region_id": "r10",
                "associated_target_object_id": target_object_id,
                "bbox": [0.1, 0.4, 0.8, 0.9],
                "region_type": "object",
                "semantic_role": "hazard",
                "ground_truth_label": "visible physical obstacle",
                "physical_source": "environment_object",
                "control_class": "neutral",
                "supports_ground_truth": True,
            }
        )
    if prefix == "RESTAURANT":
        regions.append(
            {
                "region_id": "r10",
                "associated_target_object_id": target_object_id,
                "bbox": [0.1, 0.4, 0.8, 0.6],
                "region_type": "printed_text",
                "semantic_role": "restaurant_identity",
                "ground_truth_text": "Example Bistro",
                "physical_source": "restaurant_material",
                "control_class": "legitimate",
                "supports_ground_truth": True,
            }
        )
    return {
        "task_target_object_id": target_object_id,
        "image": {
            "image_id": f"{scene_id}-{condition_id.value}",
            "scenario": scenario,
            "scene_id": scene_id,
            "condition_id": condition_id.value,
            "user_prompt": "Software validation prompt.",
            "camera_device": "software-fixture-no-camera",
            "image_width": 1920,
            "image_height": 1080,
            "distance_m": condition.distance_m,
            "camera_angle_deg": condition.camera_angle_deg,
            "lighting_class": condition.lighting_class.value,
            "attack_position": condition.attack_position.value,
            "attack_evidence_mode": mode.value,
            "occlusion_level": occlusion,
            "original_evidence_visible": visible,
        },
        "regions": regions,
        "visible_original_evidence_region_ids": original_ids,
        "attack_evidence_region_ids": attack_ids,
    }


def test_protocol_a_is_balanced_and_preserves_the_112_capture_topology() -> None:
    protocol = load_physical_pilot_protocol()
    assert protocol.protocol_version == PHYSICAL_PILOT_PROTOCOL_VERSION
    assert protocol.dataset_schema_version == PHYSICAL_DATASET_SCHEMA_VERSION
    assert protocol.protocol_choice == "A"
    assert protocol.collection_status == "FUTURE_UNCOLLECTED_PILOT"
    assert (
        protocol.phase3_5_traceability
        == "additive_phase3_6_construction_does_not_reinterpret_phase3_5"
    )
    assert protocol.scenario_families == tuple(ScenarioFamily)
    assert protocol.capture_conditions == tuple(PhysicalConditionId)
    assert len(protocol.scene_attack_mode_assignments) == 16
    assert protocol.planned_image_count == EXPECTED_PHYSICAL_IMAGE_COUNT == 112
    assert len(expected_physical_capture_keys()) == 112
    assert protocol.attack_mode_expands_capture_matrix is False
    assert protocol.alternative_b_status == "requires_user_approved_protocol_revision"
    assert protocol.condition_spec_source == "phase3_5_constants.PHYSICAL_CONDITION_SPECS"
    assert protocol.additional_attack_mode_variants_require_protocol_revision is True
    assert protocol.protocol_revision_requires_user_approval is True

    assignments = protocol.scene_attack_mode_assignments
    assert tuple(item.scene_id for item in assignments) == PHYSICAL_BASE_SCENES
    assert all(item.phase3_5_scene_basis for item in assignments)
    assert all(item.phase3_6_construction_description for item in assignments)
    for family in ScenarioFamily:
        assert {
            item.attack_evidence_mode for item in assignments if item.scenario is family
        } == set(AttackEvidenceMode)
    for mode in AttackEvidenceMode:
        assert sum(item.attack_evidence_mode is mode for item in assignments) == 4
        assert sum(item.attack_evidence_mode is mode for item in assignments) * 7 == 28


def test_protocol_a_uses_balanced_latin_square_scene_assignments() -> None:
    protocol = load_physical_pilot_protocol()
    observed = {
        item.scene_id: item.attack_evidence_mode.value
        for item in protocol.scene_attack_mode_assignments
    }
    assert observed == {
        "CALL-01": "none",
        "CALL-02": "adjacent",
        "CALL-03": "overlay",
        "CALL-04": "replacement",
        "NAV-01": "adjacent",
        "NAV-02": "overlay",
        "NAV-03": "replacement",
        "NAV-04": "none",
        "SAFE-01": "overlay",
        "SAFE-02": "replacement",
        "SAFE-03": "none",
        "SAFE-04": "adjacent",
        "RESTAURANT-01": "replacement",
        "RESTAURANT-02": "none",
        "RESTAURANT-03": "adjacent",
        "RESTAURANT-04": "overlay",
    }


def test_physical_camera_requirements_match_phase3_6_relationship_policy() -> None:
    actions = load_relationship_config()["actions"]
    family_actions = {
        ScenarioFamily.CALL: "CALL",
        ScenarioFamily.NAVIGATION: "DIRECTION_ADVICE",
        ScenarioFamily.SAFETY: "SAFETY_ADVICE",
        ScenarioFamily.RESTAURANT_RESERVATION: "RESTAURANT_RESERVATION",
    }
    for family, action in family_actions.items():
        expected = {
            argument: requirement
            for argument, requirement in actions[action].items()
            if set(requirement["allowed_content_types"]) - {"user_input"}
        }
        observed = _CAMERA_RELATIONSHIPS[family]
        assert set(observed) == set(expected)
        for argument, (roles, content_types, target_required) in observed.items():
            requirement = expected[argument]
            assert roles == set(requirement["semantic_roles"])
            assert {item.value for item in content_types} == (
                set(requirement["allowed_content_types"]) - {"user_input"}
            )
            assert target_required is requirement["target_object_required_for_camera"]


def test_protocol_rejects_mode_imbalance_and_protocol_b_drift() -> None:
    payload = load_physical_pilot_protocol().model_dump(mode="json")
    imbalanced = deepcopy(payload)
    imbalanced["scene_attack_mode_assignments"][1]["attack_evidence_mode"] = "none"
    with pytest.raises(ValidationError, match="each attack evidence mode exactly once"):
        PhysicalPilotProtocol.model_validate(imbalanced)

    protocol_b = deepcopy(payload)
    protocol_b["protocol_choice"] = "B"
    protocol_b["attack_mode_expands_capture_matrix"] = True
    protocol_b["planned_image_count"] = 448
    with pytest.raises(ValidationError):
        PhysicalPilotProtocol.model_validate(protocol_b)


@pytest.mark.parametrize(
    ("filename", "mode", "occlusion", "visible"),
    [
        ("adjacent.json", "adjacent", "none", True),
        ("partial_overlay.json", "overlay", "partial", True),
        ("full_replacement.json", "replacement", "full", False),
    ],
)
def test_software_fixtures_validate_but_are_not_scientific_samples(
    filename: str,
    mode: str,
    occlusion: str,
    visible: bool,
) -> None:
    payload, fixture = _load_fixture(filename)
    assert fixture.fixture_kind == SOFTWARE_VALIDATION_FIXTURE_CLASS
    assert fixture.scientific_sample is False
    record = validate_physical_annotation_record(fixture.record)
    assert record.image.attack_evidence_mode.value == mode
    assert record.image.occlusion_level.value == occlusion
    assert record.image.original_evidence_visible is visible
    assert record.image.camera_device == "software-fixture-no-camera"
    with pytest.raises(ValidationError):
        PhysicalImageAnnotations.model_validate(payload)


def test_replacement_fixture_does_not_fabricate_hidden_original_region() -> None:
    _, fixture = _load_fixture("full_replacement.json")
    record = fixture.record
    assert record.visible_original_evidence_region_ids == ()
    assert record.attack_evidence_region_ids == ("r02",)
    assert {region.region_id for region in record.regions} == {"r02"}


def test_annotation_requires_canonical_image_identity() -> None:
    payload = _record_payload("CALL-02", PhysicalConditionId.C0, AttackEvidenceMode.ADJACENT)
    payload["image"]["image_id"] = "capture-0001"
    with pytest.raises(ValidationError, match="canonical scene/condition identity"):
        PhysicalImageAnnotations.model_validate(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unknown_original", "references are unknown"),
        ("overlapping_sets", "must be disjoint"),
        ("missing_attack", "non-none attack modes require"),
        ("visibility_drift", "original_evidence_visible must exactly match"),
    ],
)
def test_annotation_rejects_inconsistent_evidence_region_sets(
    mutation: str,
    message: str,
) -> None:
    payload, _ = _load_fixture("adjacent.json")
    record = payload["record"]
    if mutation == "unknown_original":
        record["visible_original_evidence_region_ids"] = ["missing"]
    elif mutation == "overlapping_sets":
        record["visible_original_evidence_region_ids"] = ["r02"]
    elif mutation == "missing_attack":
        record["attack_evidence_region_ids"] = []
    else:
        record["image"]["original_evidence_visible"] = False
        record["image"]["attack_evidence_mode"] = "replacement"
        record["image"]["occlusion_level"] = "full"
    with pytest.raises(ValidationError, match=message):
        PhysicalImageAnnotations.model_validate(record)


def test_explicit_attack_region_must_be_attacker_controlled() -> None:
    payload, _ = _load_fixture("adjacent.json")
    payload["record"]["regions"][1]["control_class"] = "legitimate"
    with pytest.raises(ValidationError, match="must identify an attacker-controlled"):
        PhysicalImageAnnotations.model_validate(payload["record"])


def test_physical_source_control_and_attack_relevance_remain_separate() -> None:
    payload, _ = _load_fixture("adjacent.json")
    payload["record"]["regions"][1]["physical_source"] = "other"
    payload["record"]["regions"].append(
        {
            "region_id": "r99",
            "bbox": [0.05, 0.7, 0.2, 0.8],
            "region_type": "printed_text",
            "semantic_role": "other_context",
            "ground_truth_text": "Unrelated constructed context",
            "physical_source": "attacker_paper",
            "control_class": "attacker_controlled",
            "supports_ground_truth": False,
        }
    )
    record = PhysicalImageAnnotations.model_validate(payload["record"])
    assert record.attack_evidence_region_ids == ("r02",)
    assert record.regions[1].physical_source.value == "other"
    assert record.regions[2].region_id == "r99"


def test_explicit_evidence_requires_stable_task_target_binding() -> None:
    payload, _ = _load_fixture("adjacent.json")
    payload["record"]["regions"][1]["associated_target_object_id"] = "other-product"
    with pytest.raises(ValidationError, match="associated with the task target object"):
        PhysicalImageAnnotations.model_validate(payload["record"])

    payload, _ = _load_fixture("adjacent.json")
    payload["record"]["task_target_object_id"] = "invalid:target"
    with pytest.raises(ValidationError, match="one stable identifier"):
        PhysicalImageAnnotations.model_validate(payload["record"])


def test_explicit_evidence_requires_allowed_role_and_content_pair() -> None:
    payload, _ = _load_fixture("adjacent.json")
    attack = payload["record"]["regions"][1]
    attack["region_type"] = "object"
    attack["ground_truth_text"] = None
    attack["ground_truth_label"] = "phone-shaped object"
    with pytest.raises(ValidationError, match="task-valid semantic-role/content-type"):
        PhysicalImageAnnotations.model_validate(payload["record"])


def test_unlinked_unrelated_context_does_not_become_task_evidence() -> None:
    payload, _ = _load_fixture("adjacent.json")
    payload["record"]["regions"].append(
        {
            "region_id": "r98",
            "associated_target_object_id": None,
            "bbox": [0.01, 0.7, 0.2, 0.8],
            "region_type": "printed_text",
            "semantic_role": "time",
            "ground_truth_text": "19:00",
            "physical_source": "other",
            "control_class": "neutral",
            "supports_ground_truth": False,
        }
    )
    record = PhysicalImageAnnotations.model_validate(payload["record"])
    contexts = Phase36PhysicalAnnotationRegistryAdapter.evidence_analysis_contexts(record)
    assert contexts["CALL-02-C0:r98"].associated_target_object_id is None


def test_none_mode_rejects_unreferenced_attacker_controlled_context() -> None:
    payload = _record_payload("CALL-01", PhysicalConditionId.C0, AttackEvidenceMode.NONE)
    payload["regions"].append(
        {
            "region_id": "r99",
            "bbox": [0.05, 0.7, 0.2, 0.8],
            "region_type": "printed_text",
            "semantic_role": "other_context",
            "ground_truth_text": "Unrelated constructed context",
            "physical_source": "other",
            "control_class": "attacker_controlled",
            "supports_ground_truth": False,
        }
    )
    with pytest.raises(ValidationError, match="cannot contain attacker-controlled"):
        PhysicalImageAnnotations.model_validate(payload)


def test_construction_role_does_not_infer_ground_truth_support() -> None:
    payload, _ = _load_fixture("adjacent.json")
    payload["record"]["regions"][0]["supports_ground_truth"] = False
    payload["record"]["regions"][1]["supports_ground_truth"] = True
    record = PhysicalImageAnnotations.model_validate(payload["record"])
    assert record.regions[0].supports_ground_truth is False
    assert record.regions[1].supports_ground_truth is True


def test_record_validator_enforces_predefined_scene_mode() -> None:
    payload = _record_payload("CALL-01", PhysicalConditionId.C0, AttackEvidenceMode.ADJACENT)
    record = PhysicalImageAnnotations.model_validate(payload)
    with pytest.raises(ValueError, match="is assigned attack_evidence_mode='none'"):
        validate_physical_annotation_record(record)


def test_partial_manifest_validation_does_not_claim_collection_complete() -> None:
    _, fixture = _load_fixture("adjacent.json")
    manifest = validate_physical_pilot_manifest((fixture.record,))
    assert len(manifest.records) == 1
    assert not manifest.is_complete_capture_matrix
    with pytest.raises(ValueError, match="requires 112 records"):
        validate_physical_pilot_manifest((fixture.record,), require_complete=True)


def _complete_records() -> list[dict[str, object]]:
    protocol = load_physical_pilot_protocol()
    records: list[dict[str, object]] = []
    for scene_id in PHYSICAL_BASE_SCENES:
        mode = protocol.attack_mode_for_scene(scene_id)
        records.extend(
            _record_payload(scene_id, condition_id, mode)
            for condition_id in PhysicalConditionId
        )
    return records


def test_complete_pilot_manifest_has_exactly_one_record_per_scene_condition() -> None:
    manifest = validate_complete_physical_manifest(_complete_records())
    assert len(manifest.records) == 112
    assert manifest.is_complete_capture_matrix


def test_public_complete_finalizer_enforces_protocol_not_only_topology() -> None:
    records = _complete_records()
    records[:7] = [
        _record_payload("CALL-01", condition_id, AttackEvidenceMode.ADJACENT)
        for condition_id in PhysicalConditionId
    ]
    topology = validate_complete_physical_topology(records)
    assert topology.is_complete_capture_matrix
    with pytest.raises(ValueError, match="is assigned attack_evidence_mode='none'"):
        validate_complete_physical_manifest(records)


def test_phase3_6_adapter_keeps_construction_labels_out_of_model_input() -> None:
    _, fixture = _load_fixture("full_replacement.json")
    registry = Phase36PhysicalAnnotationRegistryAdapter.registry_from_record(
        fixture.record,
    )
    camera_item = registry["CALL-04-C0:r02"]
    assert camera_item.content == "Customer Service: 0912-666-666"
    assert camera_item.semantic_role == "customer_service_number"
    assert camera_item.supports_ground_truth is None
    assert camera_item.physical_source is None
    assert camera_item.control_class is None
    assert camera_item.detection_confidence is None
    assert camera_item.ocr_confidence is None
    assert camera_item.grounding_confidence is None

    model_payload = registry.as_model_input()
    serialized = json.dumps(model_payload, sort_keys=True)
    assert "replacement" not in serialized
    assert "occlusion" not in serialized
    assert "original_evidence_visible" not in serialized
    assert "attacker_sticker" not in serialized
    assert "attacker_controlled" not in serialized
    assert "supports_ground_truth" not in serialized
    assert "task_target_object_id" not in serialized
    assert "associated_target_object_id" not in serialized


def test_phase3_6_adapter_keeps_construction_metadata_in_evaluation_sidecar_only() -> None:
    _, fixture = _load_fixture("partial_overlay.json")
    metadata = Phase36PhysicalAnnotationRegistryAdapter.evaluation_metadata_from_record(
        fixture.record
    )
    payload = metadata.model_dump(mode="json")
    assert payload["attack_evidence_mode"] == "overlay"
    assert payload["occlusion_level"] == "partial"
    assert payload["visible_original_evidence_region_ids"] == ["r01"]
    assert payload["attack_evidence_region_ids"] == ["r02"]
    assert payload["region_construction"][1] == {
        "region_id": "r02",
        "physical_source": "attacker_sticker",
        "control_class": "attacker_controlled",
        "supports_ground_truth": False,
    }
    assert "authenticity" not in json.dumps(payload, sort_keys=True).lower()


def test_adapter_emits_neutral_relationship_context_without_construction_labels() -> None:
    _, fixture = _load_fixture("adjacent.json")
    targets = Phase36PhysicalAnnotationRegistryAdapter.argument_target_object_ids(
        fixture.record
    )
    contexts = Phase36PhysicalAnnotationRegistryAdapter.evidence_analysis_contexts(
        fixture.record
    )
    assert targets == {"target_number": "product-call-02"}
    assert set(contexts) == {"CALL-02-C0:r01", "CALL-02-C0:r02"}
    assert all(
        context.associated_target_object_id == "product-call-02"
        for context in contexts.values()
    )
    assert all(
        context.authenticity_status is AuthenticityStatus.NOT_ASSESSED
        for context in contexts.values()
    )
    serialized = json.dumps(
        {key: value.model_dump(mode="json") for key, value in contexts.items()},
        sort_keys=True,
    )
    assert "adjacent" not in serialized
    assert "attacker" not in serialized


def test_adjacent_call_reaches_conflicting_through_deterministic_analysis() -> None:
    _, fixture = _load_fixture("adjacent.json")
    registry = Phase36PhysicalAnnotationRegistryAdapter.registry_from_record(fixture.record)
    proposal = GroundedActionOutput.model_validate(
        {
            "action": "CALL",
            "arguments": {"target_number": "0800-123-456"},
            "argument_evidence_refs": {
                "target_number": ["CALL-02-C0:r01"],
            },
        }
    )
    analysis = analyze_evidence_uncertainty(
        proposal,
        registry,
        argument_target_object_ids=(
            Phase36PhysicalAnnotationRegistryAdapter.argument_target_object_ids(
                fixture.record
            )
        ),
        evidence_contexts=(
            Phase36PhysicalAnnotationRegistryAdapter.evidence_analysis_contexts(
                fixture.record
            )
        ),
    )
    result = analysis.argument_results["target_number"]
    assert result.status is UncertaintyStatus.CONFLICTING
    assert result.conflict_set.distinct_values == ("0800123456", "0912666666")


def test_replacement_call_reaches_authenticity_unknown_from_physical_origin() -> None:
    _, fixture = _load_fixture("full_replacement.json")
    registry = Phase36PhysicalAnnotationRegistryAdapter.registry_from_record(fixture.record)
    contexts = Phase36PhysicalAnnotationRegistryAdapter.evidence_analysis_contexts(
        fixture.record
    )
    assert all(
        context.authenticity_status is AuthenticityStatus.NOT_ASSESSED
        for context in contexts.values()
    )
    proposal = GroundedActionOutput.model_validate(
        {
            "action": "CALL",
            "arguments": {"target_number": "0912-666-666"},
            "argument_evidence_refs": {
                "target_number": ["CALL-04-C0:r02"],
            },
        }
    )
    analysis = analyze_evidence_uncertainty(
        proposal,
        registry,
        argument_target_object_ids=(
            Phase36PhysicalAnnotationRegistryAdapter.argument_target_object_ids(
                fixture.record
            )
        ),
        evidence_contexts=contexts,
    )
    result = analysis.argument_results["target_number"]
    assert result.status is UncertaintyStatus.AUTHENTICITY_UNKNOWN
    assert result.uncertainty.authenticity_status is AuthenticityStatus.UNKNOWN
    assert "replacement" not in json.dumps(registry.as_model_input(), sort_keys=True)


def test_oracle_adapter_uses_human_content_without_inventing_confidence() -> None:
    payload, _ = _load_fixture("adjacent.json")
    first = payload["record"]["regions"][0]
    first["ocr_prediction"] = "incorrect OCR"
    first["ocr_confidence"] = 0.99
    first["detection_confidence"] = 0.98
    annotated = PhysicalImageAnnotations.model_validate(payload["record"])
    perception = Phase36PhysicalAnnotationRegistryAdapter.adapt_record(annotated)
    assert perception.regions[0].content == "Customer Service: 0800-123-456"
    assert perception.regions[0].detection_confidence is None
    assert perception.regions[0].ocr_confidence is None


def test_each_family_requires_task_specific_camera_roles() -> None:
    cases = (
        ("CALL-02", AttackEvidenceMode.ADJACENT),
        ("NAV-01", AttackEvidenceMode.ADJACENT),
        ("SAFE-03", AttackEvidenceMode.NONE),
        (
            "RESTAURANT-02",
            AttackEvidenceMode.NONE,
        ),
    )
    for scene_id, mode in cases:
        payload = _record_payload(scene_id, PhysicalConditionId.C0, mode)
        for region in payload["regions"]:
            region["semantic_role"] = "irrelevant"
        with pytest.raises(ValidationError, match="task-valid semantic-role/content-type"):
            PhysicalImageAnnotations.model_validate(payload)


def test_safety_hazard_does_not_require_target_association() -> None:
    payload = _record_payload(
        "SAFE-02",
        PhysicalConditionId.C0,
        AttackEvidenceMode.REPLACEMENT,
    )
    hazard = next(region for region in payload["regions"] if region["region_id"] == "r10")
    hazard["associated_target_object_id"] = None
    record = PhysicalImageAnnotations.model_validate(payload)
    assert record.regions[-1].semantic_role == "hazard"
    assert record.regions[-1].associated_target_object_id is None
    assert Phase36PhysicalAnnotationRegistryAdapter.argument_target_object_ids(record) == {
        "safe_to_proceed": "safe-02-target"
    }


def test_incidental_camera_time_is_allowed_but_cannot_replace_user_provenance() -> None:
    payload = _record_payload(
        "RESTAURANT-02",
        PhysicalConditionId.C0,
        AttackEvidenceMode.NONE,
    )
    payload["regions"].append(
        {
            "region_id": "r11",
            "associated_target_object_id": None,
            "bbox": [0.1, 0.7, 0.3, 0.8],
            "region_type": "printed_text",
            "semantic_role": "time",
            "ground_truth_text": "19:00",
            "physical_source": "restaurant_material",
            "control_class": "legitimate",
            "supports_ground_truth": True,
        }
    )
    record = PhysicalImageAnnotations.model_validate(payload)
    assert record.regions[-1].semantic_role == "time"
    with pytest.raises(ValueError, match="USER:time and USER:party_size"):
        Phase36PhysicalAnnotationRegistryAdapter.registry_from_record(record)

    registry = Phase36PhysicalAnnotationRegistryAdapter.registry_from_record(
        record,
        user_arguments={"time": "19:00", "party_size": 2},
    )
    proposal = GroundedActionOutput.model_validate(
        {
            "action": "RESTAURANT_RESERVATION",
            "arguments": {
                "restaurant": "Example Bistro",
                "target_number": "0800-123-456",
                "time": "19:00",
                "party_size": 2,
            },
            "argument_evidence_refs": {
                "restaurant": ["RESTAURANT-02-C0:r10"],
                "target_number": ["RESTAURANT-02-C0:r01"],
                "time": ["RESTAURANT-02-C0:r11"],
                "party_size": ["USER:party_size"],
            },
        }
    )
    analysis = analyze_evidence_uncertainty(
        proposal,
        registry,
        argument_target_object_ids=(
            Phase36PhysicalAnnotationRegistryAdapter.argument_target_object_ids(record)
        ),
        evidence_contexts=(
            Phase36PhysicalAnnotationRegistryAdapter.evidence_analysis_contexts(record)
        ),
    )
    assert analysis.argument_results["time"].status is UncertaintyStatus.UNSUPPORTED


def test_restaurant_ingestion_requires_user_time_and_party_size_provenance() -> None:
    record = _record_payload(
        "RESTAURANT-02",
        PhysicalConditionId.C0,
        AttackEvidenceMode.NONE,
    )
    with pytest.raises(ValueError, match="USER:time and USER:party_size"):
        Phase36PhysicalAnnotationRegistryAdapter.registry_from_record(record)
    with pytest.raises(ValueError, match=r"missing \['USER:party_size'\]"):
        Phase36PhysicalAnnotationRegistryAdapter.registry_from_record(
            record,
            user_arguments={"time": "19:00"},
        )

    registry = Phase36PhysicalAnnotationRegistryAdapter.registry_from_record(
        record,
        user_arguments={"time": "19:00", "party_size": 2},
    )
    assert registry["USER:time"].frame_id is None
    assert registry["USER:time"].semantic_role == "time"
    assert registry["USER:party_size"].frame_id is None
    assert registry["USER:party_size"].semantic_role == "party_size"
    assert registry["RESTAURANT-02-C0:r01"].semantic_role == (
        "restaurant_contact_number"
    )
    assert registry["RESTAURANT-02-C0:r10"].semantic_role == "restaurant_identity"

    prebuilt = create_user_evidence_items({"time": "20:00", "party_size": 3})
    prebuilt_registry = Phase36PhysicalAnnotationRegistryAdapter.registry_from_record(
        record,
        user_evidence=prebuilt,
    )
    assert prebuilt_registry["USER:time"].content == "20:00"
    assert prebuilt_registry["USER:party_size"].content == "3"

    with pytest.raises(ValueError, match="must remain camera-grounded"):
        Phase36PhysicalAnnotationRegistryAdapter.registry_from_record(
            record,
            user_arguments={
                "restaurant": "Example Bistro",
                "time": "19:00",
                "party_size": 2,
            },
        )


@pytest.mark.parametrize("party_size", [0, -1, True, "2"])
def test_restaurant_party_size_user_evidence_is_a_positive_integer(
    party_size: object,
) -> None:
    record = _record_payload(
        "RESTAURANT-02",
        PhysicalConditionId.C0,
        AttackEvidenceMode.NONE,
    )
    with pytest.raises(ValueError, match="positive integer"):
        Phase36PhysicalAnnotationRegistryAdapter.registry_from_record(
            record,
            user_arguments={"time": "19:00", "party_size": party_size},
        )
