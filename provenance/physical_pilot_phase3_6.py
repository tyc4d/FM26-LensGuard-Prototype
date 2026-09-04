"""Safe ingestion boundary for Phase 3.6 physical-pilot annotations.

Human ground-truth content is useful for deterministic software validation,
but construction metadata is not action-model evidence. This adapter keeps
attack mode, occlusion, original visibility, physical source, and control class
in an evaluation-only sidecar. It never produces an authenticity assessment.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from phase3_5_constants import (
    ControlClass,
    EvidenceContentType,
    PhysicalConditionId,
    RegistryOrigin,
    ScenarioFamily,
)
from phase3_5_dataset_schema import PhysicalSource
from phase3_6_constants import (
    PHYSICAL_DATASET_SCHEMA_VERSION,
    PHYSICAL_PILOT_PROTOCOL_VERSION,
)
from phase3_6_dataset_schema import (
    AttackEvidenceMode,
    OcclusionLevel,
    PhysicalImageAnnotations,
    PhysicalPilotProtocol,
    target_required_camera_arguments,
    validate_physical_annotation_record,
)
from phase3_6_schema import AuthenticityStatus
from provenance.evidence_analysis_phase3_6 import EvidenceAnalysisContext
from provenance.evidence_registry_phase3_5 import (
    EvidenceItem,
    EvidenceRegistry,
    canonical_evidence_id,
    create_user_evidence_items,
)
from provenance.perception_phase3_5 import (
    EvidenceRegion,
    PerceptionMode,
    PerceptionResult,
)


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PhysicalRegionConstructionMetadata(_FrozenStrictModel):
    """Dataset construction labels retained outside action-model evidence."""

    region_id: str
    physical_source: PhysicalSource
    control_class: ControlClass
    supports_ground_truth: bool


class PhysicalAttackEvaluationMetadata(_FrozenStrictModel):
    """Evaluation-only scene construction facts, never authenticity claims."""

    dataset_schema_version: Literal[PHYSICAL_DATASET_SCHEMA_VERSION] = (
        PHYSICAL_DATASET_SCHEMA_VERSION
    )
    protocol_version: Literal[PHYSICAL_PILOT_PROTOCOL_VERSION] = (
        PHYSICAL_PILOT_PROTOCOL_VERSION
    )
    image_id: str
    scenario: ScenarioFamily
    scene_id: str
    condition_id: PhysicalConditionId
    attack_evidence_mode: AttackEvidenceMode
    occlusion_level: OcclusionLevel
    original_evidence_visible: bool
    visible_original_evidence_region_ids: tuple[str, ...]
    attack_evidence_region_ids: tuple[str, ...]
    region_construction: tuple[PhysicalRegionConstructionMetadata, ...]


class Phase36PhysicalAnnotationRegistryAdapter:
    """Convert validated pilot annotations into a non-leaking Oracle registry.

    Human content, boxes, and semantic roles are copied for Oracle software
    validation. Supports-ground-truth, physical source, and control class are
    deliberately set to null in registry items because those dataset labels
    could reveal the construction assignment. Their original values remain in
    the separate evaluation sidecar.
    """

    @staticmethod
    def _validate(
        record: PhysicalImageAnnotations | Mapping[str, Any],
        protocol: PhysicalPilotProtocol | Mapping[str, Any] | str | Path | None,
    ) -> PhysicalImageAnnotations:
        return validate_physical_annotation_record(record, protocol=protocol)

    @classmethod
    def adapt_record(
        cls,
        record: PhysicalImageAnnotations | Mapping[str, Any],
        *,
        protocol: PhysicalPilotProtocol | Mapping[str, Any] | str | Path | None = None,
    ) -> PerceptionResult:
        annotated = cls._validate(record, protocol)
        regions = tuple(
            EvidenceRegion(
                region_id=region.region_id,
                bbox=region.bbox,
                content=region.ground_truth_content,
                content_type=region.content_type,
                detection_confidence=None,
                ocr_confidence=None,
                semantic_role=region.semantic_role,
                physical_source=None,
                control_class=None,
                supports_ground_truth=None,
            )
            for region in annotated.regions
        )
        return PerceptionResult(
            frame_id=annotated.image.image_id,
            regions=regions,
            mode=PerceptionMode.ORACLE_REGISTRY,
            registry_origin=RegistryOrigin.PHYSICAL_ANNOTATION,
            backend_name="phase3.6_physical_human_annotations",
        )

    @classmethod
    def registry_from_record(
        cls,
        record: PhysicalImageAnnotations | Mapping[str, Any],
        *,
        protocol: PhysicalPilotProtocol | Mapping[str, Any] | str | Path | None = None,
        user_evidence: Iterable[EvidenceItem] = (),
        user_arguments: Mapping[str, object] | None = None,
    ) -> EvidenceRegistry:
        annotated = cls._validate(record, protocol)
        cls._validate_restaurant_user_arguments(annotated, user_arguments)
        evidence = tuple(user_evidence)
        if user_arguments:
            evidence += create_user_evidence_items(user_arguments)
        cls._validate_restaurant_user_provenance(annotated, evidence)
        return cls.adapt_record(annotated, protocol=protocol).to_registry(
            user_evidence=evidence
        )

    @staticmethod
    def _validate_restaurant_user_arguments(
        annotated: PhysicalImageAnnotations,
        user_arguments: Mapping[str, object] | None,
    ) -> None:
        if annotated.image.scenario is not ScenarioFamily.RESTAURANT_RESERVATION:
            return
        # Callers may provide already-validated EvidenceItem instances through
        # ``user_evidence`` instead. The combined evidence channel is checked
        # by ``_validate_restaurant_user_provenance`` below.
        if user_arguments is None:
            return
        supplied = dict(user_arguments)
        missing = {"time", "party_size"} - set(supplied)
        if missing:
            rendered = [f"USER:{name}" for name in sorted(missing)]
            raise ValueError(
                "RESTAURANT_RESERVATION ingestion requires explicit USER:time and "
                f"USER:party_size evidence; missing {rendered}"
            )
        forbidden = {"restaurant", "target_number"} & set(supplied)
        if forbidden:
            raise ValueError(
                "restaurant and target_number must remain camera-grounded in the "
                f"physical pilot; found {sorted(forbidden)}"
            )
        time_value = supplied["time"]
        if not isinstance(time_value, str) or not time_value.strip():
            raise ValueError("restaurant USER:time must be a nonblank string")
        party_size = supplied["party_size"]
        if isinstance(party_size, bool) or not isinstance(party_size, int) or party_size < 1:
            raise ValueError("USER:party_size must be a positive integer")

    @staticmethod
    def _validate_restaurant_user_provenance(
        annotated: PhysicalImageAnnotations,
        evidence: tuple[EvidenceItem, ...],
    ) -> None:
        if annotated.image.scenario is not ScenarioFamily.RESTAURANT_RESERVATION:
            return
        ids = [item.evidence_id for item in evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("restaurant USER evidence IDs must be unique")
        by_id = {item.evidence_id: item for item in evidence}
        required = {"USER:time", "USER:party_size"}
        missing = required - set(by_id)
        if missing:
            raise ValueError(
                "RESTAURANT_RESERVATION ingestion requires explicit USER:time and "
                f"USER:party_size evidence; missing {sorted(missing)}"
            )
        forbidden = {"USER:restaurant", "USER:target_number"} & set(by_id)
        if forbidden:
            raise ValueError(
                "restaurant and target_number must remain camera-grounded in the "
                f"physical pilot; found {sorted(forbidden)}"
            )
        for argument in ("time", "party_size"):
            item = by_id[f"USER:{argument}"]
            if (
                item.content_type is not EvidenceContentType.USER_INPUT
                or item.semantic_role != argument
                or item.registry_origin is not RegistryOrigin.USER_PROMPT
                or item.frame_id is not None
                or item.region_id is not None
                or item.bbox is not None
            ):
                raise ValueError(
                    f"USER:{argument} must be explicit non-camera USER evidence"
                )
        try:
            party_size = json.loads(by_id["USER:party_size"].content)
        except json.JSONDecodeError as error:
            raise ValueError("USER:party_size must encode a positive integer") from error
        if isinstance(party_size, bool) or not isinstance(party_size, int) or party_size < 1:
            raise ValueError("USER:party_size must encode a positive integer")

    @classmethod
    def evaluation_metadata_from_record(
        cls,
        record: PhysicalImageAnnotations | Mapping[str, Any],
        *,
        protocol: PhysicalPilotProtocol | Mapping[str, Any] | str | Path | None = None,
    ) -> PhysicalAttackEvaluationMetadata:
        annotated = cls._validate(record, protocol)
        return PhysicalAttackEvaluationMetadata(
            image_id=annotated.image.image_id,
            scenario=annotated.image.scenario,
            scene_id=annotated.image.scene_id,
            condition_id=annotated.image.condition_id,
            attack_evidence_mode=annotated.image.attack_evidence_mode,
            occlusion_level=annotated.image.occlusion_level,
            original_evidence_visible=annotated.image.original_evidence_visible,
            visible_original_evidence_region_ids=(
                annotated.visible_original_evidence_region_ids
            ),
            attack_evidence_region_ids=annotated.attack_evidence_region_ids,
            region_construction=tuple(
                PhysicalRegionConstructionMetadata(
                    region_id=region.region_id,
                    physical_source=region.physical_source,
                    control_class=region.control_class,
                    supports_ground_truth=region.supports_ground_truth,
                )
                for region in annotated.regions
            ),
        )

    @classmethod
    def argument_target_object_ids(
        cls,
        record: PhysicalImageAnnotations | Mapping[str, Any],
        *,
        protocol: PhysicalPilotProtocol | Mapping[str, Any] | str | Path | None = None,
    ) -> dict[str, str]:
        """Return deterministic argument-to-task-target relationship context."""

        annotated = cls._validate(record, protocol)
        return {
            argument: annotated.task_target_object_id
            for argument in target_required_camera_arguments(
                annotated.image.scenario
            )
        }

    @classmethod
    def evidence_analysis_contexts(
        cls,
        record: PhysicalImageAnnotations | Mapping[str, Any],
        *,
        protocol: PhysicalPilotProtocol | Mapping[str, Any] | str | Path | None = None,
    ) -> dict[str, EvidenceAnalysisContext]:
        """Return neutral relationship context without authenticity inference."""

        annotated = cls._validate(record, protocol)
        frame_id = annotated.image.image_id
        return {
            canonical_evidence_id(frame_id, region.region_id): EvidenceAnalysisContext(
                evidence_id=canonical_evidence_id(frame_id, region.region_id),
                associated_target_object_id=region.associated_target_object_id,
                authenticity_status=AuthenticityStatus.NOT_ASSESSED,
            )
            for region in annotated.regions
        }


PhysicalPilotRegistryAdapter = Phase36PhysicalAnnotationRegistryAdapter
