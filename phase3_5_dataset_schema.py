"""Schemas for the future 16-scene by 7-condition physical corpus.

Human annotations and runtime perception outputs have distinct field names so
OCR or detector predictions can never overwrite physical ground truth.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from phase3_5_constants import (
    EXPECTED_PHYSICAL_IMAGE_COUNT,
    PHYSICAL_BASE_SCENES,
    PHYSICAL_CONDITION_SPECS,
    PHYSICAL_DATASET_SCHEMA_VERSION,
    AttackPosition,
    ControlClass,
    EvidenceContentType,
    LightingClass,
    PhysicalConditionId,
    ScenarioFamily,
    SCENE_FAMILY_PREFIX,
)
from provenance.evidence_registry_phase3_5 import NormalizedBBox


class PhysicalSource(StrEnum):
    """Closed ground-truth vocabulary for future physical-region records."""

    ORIGINAL_PACKAGING = "original_packaging"
    OFFICIAL_SIGN = "official_sign"
    ENVIRONMENT_OBJECT = "environment_object"
    ATTACKER_STICKER = "attacker_sticker"
    ATTACKER_PAPER = "attacker_paper"
    RESTAURANT_MATERIAL = "restaurant_material"
    USER_INPUT = "user_input"
    OTHER = "other"


class PhysicalRegionType(StrEnum):
    PRINTED_TEXT = "printed_text"
    OBJECT = "object"
    SPATIAL = "spatial"
    SYMBOL = "symbol"
    OTHER = "other"


_CONTENT_TYPE_BY_REGION_TYPE = {
    PhysicalRegionType.PRINTED_TEXT: EvidenceContentType.TEXT,
    PhysicalRegionType.OBJECT: EvidenceContentType.OBJECT,
    PhysicalRegionType.SPATIAL: EvidenceContentType.SPATIAL,
    PhysicalRegionType.SYMBOL: EvidenceContentType.SYMBOL,
    PhysicalRegionType.OTHER: EvidenceContentType.OTHER,
}


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def scenario_for_scene(scene_id: str) -> ScenarioFamily:
    if scene_id not in PHYSICAL_BASE_SCENES:
        raise ValueError(f"scene_id must be one of the 16 declared physical scenes: {scene_id!r}")
    prefix = scene_id.rsplit("-", 1)[0]
    return SCENE_FAMILY_PREFIX[prefix]


class PhysicalImageRecord(_FrozenStrictModel):
    image_id: str = Field(min_length=1)
    scenario: ScenarioFamily
    scene_id: str = Field(min_length=1)
    condition_id: PhysicalConditionId
    user_prompt: str = Field(min_length=1)
    camera_device: str = Field(min_length=1)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    distance_m: float = Field(gt=0.0)
    camera_angle_deg: float
    lighting_class: LightingClass
    measured_lux: float | None = Field(default=None, ge=0.0)
    attack_position: AttackPosition

    @field_validator("image_id", "scene_id", "user_prompt", "camera_device")
    @classmethod
    def strip_nonblank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("physical image string fields must not be blank")
        return cleaned

    @model_validator(mode="after")
    def validate_declared_scene_and_condition(self) -> PhysicalImageRecord:
        expected_scenario = scenario_for_scene(self.scene_id)
        if self.scenario is not expected_scenario:
            raise ValueError(
                f"scene {self.scene_id!r} belongs to {expected_scenario.value}, "
                f"not {self.scenario.value}"
            )
        condition = PHYSICAL_CONDITION_SPECS[self.condition_id]
        if not math.isclose(self.distance_m, condition.distance_m, abs_tol=1e-9):
            raise ValueError(
                f"{self.condition_id.value} requires distance_m={condition.distance_m}"
            )
        if not math.isclose(
            self.camera_angle_deg, float(condition.camera_angle_deg), abs_tol=1e-9
        ):
            raise ValueError(
                f"{self.condition_id.value} requires "
                f"camera_angle_deg={condition.camera_angle_deg}"
            )
        if self.lighting_class is not condition.lighting_class:
            raise ValueError(
                f"{self.condition_id.value} requires "
                f"lighting_class={condition.lighting_class.value!r}"
            )
        if self.attack_position is not condition.attack_position:
            raise ValueError(
                f"{self.condition_id.value} requires "
                f"attack_position={condition.attack_position.value!r}"
            )
        return self


class PhysicalRegionRecord(_FrozenStrictModel):
    region_id: str = Field(min_length=1)
    bbox: NormalizedBBox
    region_type: PhysicalRegionType
    semantic_role: str | None = None
    ground_truth_text: str | None = None
    # Non-text evidence (for example stairs or a barrier) is described here;
    # it must not be forced into an OCR text field.
    ground_truth_label: str | None = None
    physical_source: PhysicalSource
    control_class: ControlClass
    supports_ground_truth: bool

    # Runtime predictions are additive and cannot overwrite the fields above.
    ocr_prediction: str | None = None
    ocr_confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    detection_confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    grounding_confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None

    @field_validator(
        "region_id",
        "semantic_role",
        "ground_truth_text",
        "ground_truth_label",
        "ocr_prediction",
    )
    @classmethod
    def strip_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("physical region string fields must not be blank")
        return cleaned

    @model_validator(mode="after")
    def validate_ground_truth_channel(self) -> PhysicalRegionRecord:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.region_id) is None:
            raise ValueError("region_id must be one stable identifier component")
        if self.physical_source is PhysicalSource.USER_INPUT:
            raise ValueError(
                "camera regions cannot use physical_source='user_input'; "
                "user values belong in non-camera USER evidence"
            )
        if self.region_type is PhysicalRegionType.PRINTED_TEXT:
            if self.ground_truth_text is None:
                raise ValueError("printed_text regions require ground_truth_text")
        elif self.ground_truth_label is None:
            raise ValueError("non-text regions require ground_truth_label")
        if self.ocr_confidence is not None and self.ocr_prediction is None:
            raise ValueError("ocr_confidence requires an ocr_prediction")
        return self

    @property
    def content_type(self) -> EvidenceContentType:
        return _CONTENT_TYPE_BY_REGION_TYPE[self.region_type]

    @property
    def ground_truth_content(self) -> str:
        if self.region_type is PhysicalRegionType.PRINTED_TEXT:
            assert self.ground_truth_text is not None
            return self.ground_truth_text
        assert self.ground_truth_label is not None
        return self.ground_truth_label


class PhysicalImageAnnotations(_FrozenStrictModel):
    image: PhysicalImageRecord
    regions: tuple[PhysicalRegionRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_regions(self) -> PhysicalImageAnnotations:
        region_ids = [region.region_id for region in self.regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("region IDs must be unique within an image")
        return self


class PhysicalDatasetManifest(_FrozenStrictModel):
    schema_version: str = PHYSICAL_DATASET_SCHEMA_VERSION
    records: tuple[PhysicalImageAnnotations, ...] = ()

    @model_validator(mode="after")
    def validate_manifest(self) -> PhysicalDatasetManifest:
        if self.schema_version != PHYSICAL_DATASET_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be exactly {PHYSICAL_DATASET_SCHEMA_VERSION!r}"
            )
        image_ids = [record.image.image_id for record in self.records]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("image IDs must be unique")
        capture_keys = [
            (record.image.scene_id, record.image.condition_id) for record in self.records
        ]
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
    return {
        (scene_id, condition_id)
        for scene_id in PHYSICAL_BASE_SCENES
        for condition_id in PhysicalConditionId
    }


def validate_complete_physical_manifest(
    records: Iterable[PhysicalImageAnnotations | dict[str, Any]],
) -> PhysicalDatasetManifest:
    """Validate a finished 112-image capture manifest.

    Partial manifests remain valid during collection; this explicit finalizer is
    the check used when claiming that all 16 x 7 captures are present.
    """

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


# Short aliases used in collection tooling and documentation.
Phase35ImageRecord = PhysicalImageRecord
Phase35RegionRecord = PhysicalRegionRecord
Phase35PhysicalRecord = PhysicalImageAnnotations
