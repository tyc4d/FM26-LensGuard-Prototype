"""Phase 3.6 physical-pilot schema with explicit attack construction metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

from phase3_5_constants import (
    ControlClass,
    EvidenceContentType,
    EXPECTED_PHYSICAL_IMAGE_COUNT,
    PHYSICAL_BASE_SCENES,
    PHYSICAL_CONDITION_SPECS,
    PhysicalConditionId,
    ScenarioFamily,
)
from phase3_5_dataset_schema import (
    PhysicalImageRecord as Phase35PhysicalImageRecord,
    PhysicalRegionRecord as Phase35PhysicalRegionRecord,
    scenario_for_scene,
)
from phase3_6_constants import (
    PHYSICAL_DATASET_SCHEMA_VERSION,
    PHYSICAL_PILOT_PROTOCOL_VERSION,
)


DEFAULT_PHYSICAL_PILOT_PROTOCOL = (
    Path(__file__).resolve().parent / "config" / "physical_pilot_phase3_6.yaml"
)
SOFTWARE_VALIDATION_FIXTURE_CLASS = "SOFTWARE_VALIDATION_ONLY"
_REGION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_CAMERA_RELATIONSHIPS = {
    ScenarioFamily.CALL: {
        "target_number": (
            {"customer_service_number", "contact_number"},
            {EvidenceContentType.TEXT},
            True,
        ),
    },
    ScenarioFamily.NAVIGATION: {
        "direction": (
            {"direction", "exit_direction", "directional_sign"},
            {
                EvidenceContentType.TEXT,
                EvidenceContentType.SYMBOL,
                EvidenceContentType.SPATIAL,
            },
            True,
        ),
        "destination": (
            {"destination", "exit_destination", "directional_sign"},
            {
                EvidenceContentType.TEXT,
                EvidenceContentType.SYMBOL,
                EvidenceContentType.SPATIAL,
            },
            True,
        ),
    },
    ScenarioFamily.SAFETY: {
        "safe_to_proceed": (
            {
                "safety_claim",
                "hazard",
                "path_condition",
                "stairs",
                "obstacle",
                "barrier",
                "hole",
                "blocked_path",
                "height_difference",
            },
            {
                EvidenceContentType.TEXT,
                EvidenceContentType.OBJECT,
                EvidenceContentType.SPATIAL,
                EvidenceContentType.SYMBOL,
                EvidenceContentType.OTHER,
            },
            True,
        ),
        "hazard": (
            {
                "hazard",
                "stairs",
                "obstacle",
                "barrier",
                "hole",
                "blocked_path",
                "height_difference",
            },
            {
                EvidenceContentType.TEXT,
                EvidenceContentType.OBJECT,
                EvidenceContentType.SPATIAL,
                EvidenceContentType.SYMBOL,
                EvidenceContentType.OTHER,
            },
            False,
        ),
    },
    ScenarioFamily.RESTAURANT_RESERVATION: {
        "restaurant": (
            {"restaurant_identity", "restaurant_name"},
            {EvidenceContentType.TEXT, EvidenceContentType.SYMBOL},
            True,
        ),
        "target_number": (
            {"reservation_number", "restaurant_contact_number", "contact_number"},
            {EvidenceContentType.TEXT},
            True,
        ),
    },
}


class AttackEvidenceMode(StrEnum):
    NONE = "none"
    ADJACENT = "adjacent"
    OVERLAY = "overlay"
    REPLACEMENT = "replacement"


class OcclusionLevel(StrEnum):
    NONE = "none"
    PARTIAL = "partial"
    FULL = "full"


class PhysicalCollectionStrategy(StrEnum):
    """Collection topology choices; only Protocol A is authorized in Phase 3.6."""

    ASSIGN_ONE_ATTACK_MODE_PER_PREDEFINED_SCENE = (
        "assign_one_attack_mode_per_predefined_scene"
    )


def target_required_camera_arguments(
    scenario: ScenarioFamily,
) -> tuple[str, ...]:
    """Return camera arguments whose relationship requires the task target."""

    return tuple(
        argument
        for argument, (_, _, target_required) in _CAMERA_RELATIONSHIPS[scenario].items()
        if target_required
    )


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PhysicalImageRecord(Phase35PhysicalImageRecord):
    """A capture record describing construction, not inferred maliciousness."""

    attack_evidence_mode: AttackEvidenceMode
    occlusion_level: OcclusionLevel
    original_evidence_visible: StrictBool

    @model_validator(mode="after")
    def validate_attack_construction(self) -> PhysicalImageRecord:
        expected_image_id = f"{self.scene_id}-{self.condition_id.value}"
        if self.image_id != expected_image_id:
            raise ValueError(
                "image_id must be the canonical scene/condition identity "
                f"{expected_image_id!r}"
            )
        expected = {
            AttackEvidenceMode.NONE: (OcclusionLevel.NONE, True),
            AttackEvidenceMode.ADJACENT: (OcclusionLevel.NONE, True),
            AttackEvidenceMode.OVERLAY: (OcclusionLevel.PARTIAL, True),
            AttackEvidenceMode.REPLACEMENT: (OcclusionLevel.FULL, False),
        }[self.attack_evidence_mode]
        observed = (self.occlusion_level, self.original_evidence_visible)
        if observed != expected:
            raise ValueError(
                f"attack_evidence_mode={self.attack_evidence_mode.value!r} requires "
                f"occlusion_level={expected[0].value!r} and "
                f"original_evidence_visible={str(expected[1]).lower()}"
            )
        return self


class PhysicalRegionRecord(Phase35PhysicalRegionRecord):
    """Phase 3.6 region type; ground truth and runtime channels stay separate.

    Attacker-controlled describes who constructed a visible item. It does not
    classify that item's value as malicious or establish any item's authenticity.
    """

    # Deterministic relationship annotation. This is separate from construction
    # source/control and never constitutes an authenticity assessment.
    associated_target_object_id: str | None = None

    @field_validator("associated_target_object_id")
    @classmethod
    def validate_target_object_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if _REGION_ID_RE.fullmatch(cleaned) is None:
            raise ValueError("associated_target_object_id must be one stable identifier")
        return cleaned


class PhysicalImageAnnotations(_FrozenStrictModel):
    image: PhysicalImageRecord
    regions: tuple[PhysicalRegionRecord, ...] = Field(min_length=1)
    task_target_object_id: str = Field(min_length=1)
    # These are construction-ground-truth references, not detector predictions.
    # Context regions can remain outside both sets.
    visible_original_evidence_region_ids: tuple[str, ...] = ()
    attack_evidence_region_ids: tuple[str, ...] = ()

    @field_validator(
        "task_target_object_id",
        "visible_original_evidence_region_ids",
        "attack_evidence_region_ids",
    )
    @classmethod
    def validate_stable_identifiers(
        cls,
        value: str | tuple[str, ...],
    ) -> str | tuple[str, ...]:
        if isinstance(value, str):
            cleaned_value = value.strip()
            if _REGION_ID_RE.fullmatch(cleaned_value) is None:
                raise ValueError("task_target_object_id must be one stable identifier")
            return cleaned_value
        values = value
        cleaned = tuple(value.strip() for value in values)
        if any(_REGION_ID_RE.fullmatch(value) is None for value in cleaned):
            raise ValueError("evidence region references must be stable region identifiers")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("evidence region references must be unique")
        return cleaned

    @model_validator(mode="after")
    def validate_attack_construction_regions(self) -> PhysicalImageAnnotations:
        region_ids = [region.region_id for region in self.regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("region IDs must be unique within an image")

        original_ids = set(self.visible_original_evidence_region_ids)
        attack_ids = set(self.attack_evidence_region_ids)
        if original_ids & attack_ids:
            raise ValueError("original and attack evidence region references must be disjoint")
        unknown = (original_ids | attack_ids) - set(region_ids)
        if unknown:
            raise ValueError(f"evidence region references are unknown: {sorted(unknown)}")

        regions_by_id = {region.region_id: region for region in self.regions}
        annotated_attacker_ids = {
            region.region_id
            for region in self.regions
            if region.control_class is ControlClass.ATTACKER_CONTROLLED
        }
        if not attack_ids.issubset(annotated_attacker_ids):
            raise ValueError(
                "every attack_evidence_region_id must identify an "
                "attacker-controlled region"
            )
        if any(
            regions_by_id[region_id].control_class is ControlClass.ATTACKER_CONTROLLED
            for region_id in original_ids
        ):
            raise ValueError("original evidence cannot be attacker-controlled")
        linked_ids = original_ids | attack_ids
        incorrectly_associated = {
            region_id
            for region_id in linked_ids
            if regions_by_id[region_id].associated_target_object_id
            != self.task_target_object_id
        }
        if incorrectly_associated:
            raise ValueError(
                "explicit original/attack evidence must be associated with the "
                f"task target object; invalid {sorted(incorrectly_associated)}"
            )

        requirements = _CAMERA_RELATIONSHIPS[self.image.scenario]

        def matching_arguments(region: PhysicalRegionRecord) -> set[str]:
            return {
                argument
                for argument, (roles, content_types, _) in requirements.items()
                if region.semantic_role in roles and region.content_type in content_types
            }

        irrelevant_linked = {
            region_id
            for region_id in linked_ids
            if not matching_arguments(regions_by_id[region_id])
        }
        if irrelevant_linked:
            raise ValueError(
                "explicit original/attack evidence must have a task-valid "
                f"semantic-role/content-type pair; invalid {sorted(irrelevant_linked)}"
            )
        has_original = bool(original_ids)
        if has_original is not self.image.original_evidence_visible:
            raise ValueError(
                "original_evidence_visible must exactly match whether visible original "
                "evidence region IDs are annotated"
            )
        has_attack = bool(attack_ids)
        if self.image.attack_evidence_mode is AttackEvidenceMode.NONE and (
            has_attack or annotated_attacker_ids
        ):
            raise ValueError(
                "attack_evidence_mode='none' cannot contain attacker-controlled evidence"
            )
        if self.image.attack_evidence_mode is not AttackEvidenceMode.NONE and not has_attack:
            raise ValueError("non-none attack modes require visible attack evidence")

        target_required_arguments = set(
            target_required_camera_arguments(self.image.scenario)
        )
        satisfied_arguments = {
            argument
            for region in self.regions
            for argument in matching_arguments(region)
            if (
                argument not in target_required_arguments
                or region.associated_target_object_id == self.task_target_object_id
            )
        }
        missing_arguments = set(requirements) - satisfied_arguments
        if missing_arguments:
            raise ValueError(
                f"{self.image.scenario.value} annotations lack target-associated, "
                "task-valid camera evidence for arguments "
                f"{sorted(missing_arguments)}"
            )
        if self.image.scenario is ScenarioFamily.SAFETY:
            hazard_roles = {
                "hazard",
                "stairs",
                "obstacle",
                "barrier",
                "hole",
                "blocked_path",
                "height_difference",
            }
            has_physical_hazard = any(
                region.semantic_role in hazard_roles
                and region.content_type
                in {
                    EvidenceContentType.OBJECT,
                    EvidenceContentType.SPATIAL,
                    EvidenceContentType.OTHER,
                }
                for region in self.regions
            )
            if not has_physical_hazard:
                raise ValueError("SAFETY annotations require visible physical hazard evidence")
        return self


class SceneAttackModeAssignment(_FrozenStrictModel):
    """The single construction mode assigned to one predefined base scene."""

    scene_id: str = Field(min_length=1)
    scenario: ScenarioFamily
    attack_evidence_mode: AttackEvidenceMode
    phase3_5_scene_basis: str = Field(min_length=1)
    phase3_6_construction_description: str = Field(min_length=1)

    @field_validator(
        "scene_id",
        "phase3_5_scene_basis",
        "phase3_6_construction_description",
    )
    @classmethod
    def strip_assignment_strings(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("assignment string fields must not be blank")
        return cleaned

    @model_validator(mode="after")
    def validate_scene_family(self) -> SceneAttackModeAssignment:
        expected = scenario_for_scene(self.scene_id)
        if self.scenario is not expected:
            raise ValueError(
                f"scene {self.scene_id!r} belongs to {expected.value}, "
                f"not {self.scenario.value}"
            )
        return self


class PhysicalPilotProtocol(_FrozenStrictModel):
    """Versioned Protocol A plan that preserves the approved 112-image count.

    Attack mode is assigned once per base scene and repeated across that
    scene's seven capture conditions. Capturing extra attack-mode variants is
    Protocol B and is intentionally outside this contract until the user
    approves a protocol revision and a new planned image count.
    """

    protocol_version: Literal[PHYSICAL_PILOT_PROTOCOL_VERSION] = (
        PHYSICAL_PILOT_PROTOCOL_VERSION
    )
    dataset_schema_version: Literal[PHYSICAL_DATASET_SCHEMA_VERSION] = (
        PHYSICAL_DATASET_SCHEMA_VERSION
    )
    protocol_choice: Literal["A"] = "A"
    collection_status: Literal["FUTURE_UNCOLLECTED_PILOT"] = (
        "FUTURE_UNCOLLECTED_PILOT"
    )
    phase3_5_traceability: Literal[
        "additive_phase3_6_construction_does_not_reinterpret_phase3_5"
    ] = "additive_phase3_6_construction_does_not_reinterpret_phase3_5"
    collection_strategy: Literal[
        PhysicalCollectionStrategy.ASSIGN_ONE_ATTACK_MODE_PER_PREDEFINED_SCENE
    ] = PhysicalCollectionStrategy.ASSIGN_ONE_ATTACK_MODE_PER_PREDEFINED_SCENE
    condition_spec_source: Literal[
        "phase3_5_constants.PHYSICAL_CONDITION_SPECS"
    ] = "phase3_5_constants.PHYSICAL_CONDITION_SPECS"
    scenario_families: tuple[ScenarioFamily, ...]
    base_scene_count: Literal[16] = 16
    capture_conditions: tuple[PhysicalConditionId, ...]
    capture_condition_count: Literal[7] = 7
    planned_image_count: Literal[112] = 112
    attack_mode_expands_capture_matrix: Literal[False] = False
    alternative_b_status: Literal["requires_user_approved_protocol_revision"] = (
        "requires_user_approved_protocol_revision"
    )
    additional_attack_mode_variants_require_protocol_revision: Literal[True] = True
    protocol_revision_requires_user_approval: Literal[True] = True
    scene_attack_mode_assignments: tuple[SceneAttackModeAssignment, ...] = Field(
        min_length=16,
        max_length=16,
    )

    @model_validator(mode="after")
    def validate_protocol_a_matrix(self) -> PhysicalPilotProtocol:
        if set(PHYSICAL_CONDITION_SPECS) != set(PhysicalConditionId):
            raise ValueError("frozen physical condition specifications are incomplete")
        if self.scenario_families != tuple(ScenarioFamily):
            raise ValueError("scenario_families must list the four declared families exactly")
        if self.capture_conditions != tuple(PhysicalConditionId):
            raise ValueError("capture_conditions must list C0 through C6 exactly once in order")

        assignments = self.scene_attack_mode_assignments
        assigned_scene_ids = tuple(item.scene_id for item in assignments)
        if assigned_scene_ids != PHYSICAL_BASE_SCENES:
            raise ValueError(
                "scene_attack_mode_assignments must list all 16 predefined scenes "
                "exactly once in canonical order"
            )
        for family in ScenarioFamily:
            family_modes = {
                item.attack_evidence_mode
                for item in assignments
                if item.scenario is family
            }
            if family_modes != set(AttackEvidenceMode):
                raise ValueError(
                    f"{family.value} must assign each attack evidence mode exactly once"
                )
        mode_counts = {
            mode: sum(item.attack_evidence_mode is mode for item in assignments)
            for mode in AttackEvidenceMode
        }
        if set(mode_counts.values()) != {4}:
            raise ValueError("Protocol A must assign exactly four scenes to every attack mode")
        calculated_count = len(assignments) * len(self.capture_conditions)
        if calculated_count != EXPECTED_PHYSICAL_IMAGE_COUNT:
            raise ValueError("Protocol A must retain the 16 x 7 capture matrix")
        return self

    def attack_mode_for_scene(self, scene_id: str) -> AttackEvidenceMode:
        for assignment in self.scene_attack_mode_assignments:
            if assignment.scene_id == scene_id:
                return assignment.attack_evidence_mode
        raise ValueError(f"scene_id is not assigned by the physical pilot protocol: {scene_id!r}")


class PhysicalPilotValidationFixture(_FrozenStrictModel):
    """Audit label that distinguishes software fixtures from scientific samples."""

    fixture_kind: Literal[SOFTWARE_VALIDATION_FIXTURE_CLASS]
    scientific_sample: Literal[False]
    description: str = Field(min_length=1)
    record: PhysicalImageAnnotations

    @field_validator("description")
    @classmethod
    def strip_description(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("fixture description must not be blank")
        return cleaned


class PhysicalDatasetManifest(_FrozenStrictModel):
    schema_version: Literal[PHYSICAL_DATASET_SCHEMA_VERSION] = (
        PHYSICAL_DATASET_SCHEMA_VERSION
    )
    records: tuple[PhysicalImageAnnotations, ...] = ()

    @model_validator(mode="after")
    def validate_manifest(self) -> PhysicalDatasetManifest:
        image_ids = [record.image.image_id for record in self.records]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("image IDs must be unique")
        capture_keys = self.capture_keys()
        if len(capture_keys) != len(set(capture_keys)):
            raise ValueError("each scene/condition capture key must be unique")
        return self

    @property
    def is_complete_capture_matrix(self) -> bool:
        return set(self.capture_keys()) == expected_physical_capture_keys()

    def capture_keys(self) -> tuple[tuple[str, PhysicalConditionId], ...]:
        return tuple(
            (record.image.scene_id, record.image.condition_id) for record in self.records
        )


def expected_physical_capture_keys() -> set[tuple[str, PhysicalConditionId]]:
    """Keep the existing 16 x 7 protocol; attack mode is record metadata."""

    return {
        (scene_id, condition_id)
        for scene_id in PHYSICAL_BASE_SCENES
        for condition_id in PhysicalConditionId
    }


def validate_complete_physical_topology(
    records: Iterable[PhysicalImageAnnotations | dict[str, Any]],
) -> PhysicalDatasetManifest:
    """Validate only the generic 16 x 7 topology, not the prescribed mode plan."""

    manifest = PhysicalDatasetManifest(records=tuple(records))
    if len(manifest.records) != EXPECTED_PHYSICAL_IMAGE_COUNT:
        raise ValueError(
            f"complete physical corpus requires {EXPECTED_PHYSICAL_IMAGE_COUNT} records"
        )
    missing = expected_physical_capture_keys() - set(manifest.capture_keys())
    if missing:
        rendered = sorted((scene, condition.value) for scene, condition in missing)
        raise ValueError(f"physical capture matrix is incomplete; missing {rendered}")
    return manifest


def load_physical_pilot_protocol(
    value: PhysicalPilotProtocol | Mapping[str, Any] | str | Path | None = None,
) -> PhysicalPilotProtocol:
    """Load and strictly validate the versioned Phase 3.6 collection plan."""

    if isinstance(value, PhysicalPilotProtocol):
        return value
    if isinstance(value, Mapping):
        payload: Any = dict(value)
    else:
        path = DEFAULT_PHYSICAL_PILOT_PROTOCOL if value is None else Path(value)
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ValueError(f"could not read Phase 3.6 physical pilot protocol: {path}") from error
        except yaml.YAMLError as error:
            raise ValueError(f"invalid Phase 3.6 physical pilot YAML: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Phase 3.6 physical pilot protocol must be a mapping")
    return PhysicalPilotProtocol.model_validate(payload)


def validate_physical_annotation_record(
    value: PhysicalImageAnnotations | Mapping[str, Any],
    *,
    protocol: PhysicalPilotProtocol | Mapping[str, Any] | str | Path | None = None,
) -> PhysicalImageAnnotations:
    """Validate one annotation and its predefined Protocol A mode assignment."""

    record = (
        value
        if isinstance(value, PhysicalImageAnnotations)
        else PhysicalImageAnnotations.model_validate(value)
    )
    resolved_protocol = load_physical_pilot_protocol(protocol)
    expected_mode = resolved_protocol.attack_mode_for_scene(record.image.scene_id)
    if record.image.attack_evidence_mode is not expected_mode:
        raise ValueError(
            f"scene {record.image.scene_id!r} is assigned "
            f"attack_evidence_mode={expected_mode.value!r}, got "
            f"{record.image.attack_evidence_mode.value!r}"
        )
    return record


def validate_physical_pilot_manifest(
    records: Iterable[PhysicalImageAnnotations | dict[str, Any]],
    *,
    protocol: PhysicalPilotProtocol | Mapping[str, Any] | str | Path | None = None,
    require_complete: bool = False,
) -> PhysicalDatasetManifest:
    """Validate partial or complete records against the Protocol A mode plan."""

    resolved_protocol = load_physical_pilot_protocol(protocol)
    source_records = tuple(records)
    manifest = PhysicalDatasetManifest(records=source_records)
    for record in manifest.records:
        validate_physical_annotation_record(record, protocol=resolved_protocol)
    if require_complete:
        validate_complete_physical_topology(manifest.records)
    return manifest


def validate_complete_physical_manifest(
    records: Iterable[PhysicalImageAnnotations | dict[str, Any]],
    *,
    protocol: PhysicalPilotProtocol | Mapping[str, Any] | str | Path | None = None,
) -> PhysicalDatasetManifest:
    """Validate a complete 112-record corpus against the prescribed pilot plan."""

    return validate_physical_pilot_manifest(
        records,
        protocol=protocol,
        require_complete=True,
    )


Phase36ImageRecord = PhysicalImageRecord
Phase36RegionRecord = PhysicalRegionRecord
Phase36PhysicalRecord = PhysicalImageAnnotations
