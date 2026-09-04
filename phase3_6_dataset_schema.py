"""Phase 3.6 physical-pilot schema with explicit attack construction metadata."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from phase3_5_constants import (
    EXPECTED_PHYSICAL_IMAGE_COUNT,
    PHYSICAL_BASE_SCENES,
    PhysicalConditionId,
)
from phase3_5_dataset_schema import (
    PhysicalImageRecord as Phase35PhysicalImageRecord,
    PhysicalRegionRecord as Phase35PhysicalRegionRecord,
)
from phase3_6_constants import PHYSICAL_DATASET_SCHEMA_VERSION


class AttackEvidenceMode(StrEnum):
    NONE = "none"
    ADJACENT = "adjacent"
    OVERLAY = "overlay"
    REPLACEMENT = "replacement"


class OcclusionLevel(StrEnum):
    NONE = "none"
    PARTIAL = "partial"
    FULL = "full"


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PhysicalImageRecord(Phase35PhysicalImageRecord):
    """A capture record describing construction, not inferred maliciousness."""

    attack_evidence_mode: AttackEvidenceMode
    occlusion_level: OcclusionLevel
    original_evidence_visible: StrictBool

    @model_validator(mode="after")
    def validate_attack_construction(self) -> PhysicalImageRecord:
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
    """Phase 3.6 region type; ground truth and runtime channels stay separate."""


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


def validate_complete_physical_manifest(
    records: Iterable[PhysicalImageAnnotations | dict[str, Any]],
) -> PhysicalDatasetManifest:
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


Phase36ImageRecord = PhysicalImageRecord
Phase36RegionRecord = PhysicalRegionRecord
Phase36PhysicalRecord = PhysicalImageAnnotations
