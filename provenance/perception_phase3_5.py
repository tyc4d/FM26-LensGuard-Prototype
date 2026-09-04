"""Perception boundary and Oracle adapter for LensGuard Phase 3.5.

``ORACLE_REGISTRY`` consumes existing benchmark annotations and is explicitly
not a real-world perception measurement.  ``AUTOMATIC_REGISTRY`` accepts only
an independent region-extraction backend; the action VLM is downstream and is
never used as the authoritative OCR or bounding-box source.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from phase3_5_constants import (
    ControlClass,
    EvidenceContentType,
    EvidencePhysicalSource,
    RegistryOrigin,
)
from phase3_5_dataset_schema import PhysicalImageAnnotations
from provenance.evidence_registry_phase3_5 import (
    EvidenceClaim,
    EvidenceItem,
    EvidenceRegistry,
    NormalizedBBox,
    canonical_evidence_id,
    create_user_evidence_items,
)


class PerceptionMode(StrEnum):
    ORACLE_REGISTRY = "ORACLE_REGISTRY"
    AUTOMATIC_REGISTRY = "AUTOMATIC_REGISTRY"


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceRegion(_FrozenStrictModel):
    """Region output of perception before a frame-scoped ID is assigned."""

    region_id: str = Field(min_length=1)
    bbox: NormalizedBBox | None = None
    content: str = Field(min_length=1)
    content_type: EvidenceContentType
    detection_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    ocr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    semantic_role: str | None = None
    physical_source: EvidencePhysicalSource | None = None
    control_class: ControlClass | None = None
    supports_ground_truth: bool | None = None
    benchmark_source_label: str | None = None
    content_claimed_authority: str | None = None
    claims: tuple[EvidenceClaim, ...] = ()

    @field_validator(
        "region_id",
        "content",
        "semantic_role",
        "benchmark_source_label",
        "content_claimed_authority",
    )
    @classmethod
    def strip_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("perception string fields must not be blank")
        return cleaned

    @model_validator(mode="after")
    def validate_region_id(self) -> EvidenceRegion:
        # Reuse the same lexical rules that later create stable registry IDs.
        canonical_evidence_id("FRAME", self.region_id)
        if self.content_type is EvidenceContentType.USER_INPUT:
            raise ValueError("camera perception cannot emit USER evidence")
        return self


class PerceptionResult(_FrozenStrictModel):
    frame_id: str = Field(min_length=1)
    regions: tuple[EvidenceRegion, ...]
    mode: PerceptionMode
    registry_origin: RegistryOrigin
    backend_name: str | None = None

    @field_validator("frame_id", "backend_name")
    @classmethod
    def strip_identifiers(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("perception identifiers must not be blank")
        return cleaned

    @model_validator(mode="after")
    def validate_result_shape(self) -> PerceptionResult:
        canonical_evidence_id(self.frame_id, "region")
        region_ids = [region.region_id for region in self.regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("perception region IDs must be unique within a frame")
        expected_origins = {
            PerceptionMode.ORACLE_REGISTRY: {
                RegistryOrigin.BENCHMARK_ANNOTATION,
                RegistryOrigin.PHYSICAL_ANNOTATION,
            },
            PerceptionMode.AUTOMATIC_REGISTRY: {
                RegistryOrigin.AUTOMATIC_PERCEPTION,
            },
        }[self.mode]
        if self.registry_origin not in expected_origins:
            raise ValueError(
                f"{self.mode.value} received incompatible registry_origin="
                f"{self.registry_origin.value!r}"
            )
        if self.mode is PerceptionMode.ORACLE_REGISTRY:
            for region in self.regions:
                if region.detection_confidence is not None or region.ocr_confidence is not None:
                    raise ValueError("Oracle perception must not invent confidence values")
        return self

    def to_registry(
        self,
        *,
        user_evidence: Iterable[EvidenceItem] = (),
    ) -> EvidenceRegistry:
        camera_items = tuple(
            EvidenceItem(
                evidence_id=canonical_evidence_id(self.frame_id, region.region_id),
                frame_id=self.frame_id,
                region_id=region.region_id,
                bbox=region.bbox,
                content=region.content,
                content_type=region.content_type,
                semantic_role=region.semantic_role,
                physical_source=region.physical_source,
                control_class=region.control_class,
                supports_ground_truth=region.supports_ground_truth,
                detection_confidence=region.detection_confidence,
                ocr_confidence=region.ocr_confidence,
                grounding_confidence=None,
                registry_origin=self.registry_origin,
                benchmark_source_label=region.benchmark_source_label,
                content_claimed_authority=region.content_claimed_authority,
                claims=region.claims,
            )
            for region in self.regions
        )
        return EvidenceRegistry(self.frame_id, (*camera_items, *tuple(user_evidence)))


class PerceptionInterface(ABC):
    """Only component authorized to provide authoritative camera regions."""

    mode: PerceptionMode

    @abstractmethod
    def perceive(
        self,
        frame_id: str,
        image: str | Path | Any | None = None,
    ) -> PerceptionResult:
        raise NotImplementedError

    def build_registry(
        self,
        frame_id: str,
        image: str | Path | Any | None = None,
        *,
        user_arguments: Mapping[str, object] | None = None,
        user_evidence: Iterable[EvidenceItem] = (),
    ) -> EvidenceRegistry:
        supplied_user_evidence = tuple(user_evidence)
        if user_arguments:
            supplied_user_evidence += create_user_evidence_items(user_arguments)
        return self.perceive(frame_id, image).to_registry(
            user_evidence=supplied_user_evidence
        )


class OracleRegistryAdapter(PerceptionInterface):
    """Adapter from frozen Phase 2 annotations to Oracle perception records.

    The adapter is lossless for existing region IDs, text, boxes, source labels,
    authority labels, and claims.  It deliberately leaves every new Phase 3.5
    semantic/physical/control/confidence field null.
    """

    mode = PerceptionMode.ORACLE_REGISTRY

    def __init__(
        self,
        records: Iterable[Mapping[str, Any]] | Mapping[str, Any] = (),
    ) -> None:
        if isinstance(records, Mapping) and "scenario_id" in records:
            source_records = (records,)
        elif isinstance(records, Mapping):
            source_records = tuple(records.values())
        else:
            source_records = tuple(records)
        indexed: dict[str, Mapping[str, Any]] = {}
        for record in source_records:
            frame_id = self._frame_id(record)
            if frame_id in indexed:
                raise ValueError(f"duplicate Oracle frame_id {frame_id!r}")
            indexed[frame_id] = record
        self._records = dict(indexed)

    @staticmethod
    def _frame_id(record: Mapping[str, Any]) -> str:
        raw = record.get("scenario_id", record.get("frame_id"))
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Phase 2 record requires scenario_id")
        frame_id = raw.strip()
        canonical_evidence_id(frame_id, "region")
        return frame_id

    @classmethod
    def adapt_phase2_record(cls, record: Mapping[str, Any]) -> PerceptionResult:
        frame_id = cls._frame_id(record)
        raw_regions = record.get("regions")
        if not isinstance(raw_regions, Sequence) or isinstance(raw_regions, (str, bytes)):
            raise ValueError("Phase 2 record requires a regions array")
        regions: list[EvidenceRegion] = []
        for raw_region in raw_regions:
            if not isinstance(raw_region, Mapping):
                raise ValueError("Phase 2 regions must be objects")
            raw_claims = raw_region.get("claims", ())
            if not isinstance(raw_claims, Sequence) or isinstance(raw_claims, (str, bytes)):
                raise ValueError("Phase 2 region claims must be an array")
            claims = tuple(EvidenceClaim.model_validate(claim) for claim in raw_claims)
            regions.append(
                EvidenceRegion(
                    region_id=raw_region.get("region_id"),
                    bbox=raw_region.get("bbox"),
                    content=raw_region.get("text"),
                    content_type=EvidenceContentType.TEXT,
                    detection_confidence=None,
                    ocr_confidence=None,
                    semantic_role=None,
                    physical_source=None,
                    control_class=None,
                    supports_ground_truth=None,
                    benchmark_source_label=raw_region.get("source_type"),
                    content_claimed_authority=raw_region.get(
                        "content_claimed_authority"
                    ),
                    claims=claims,
                )
            )
        return PerceptionResult(
            frame_id=frame_id,
            regions=tuple(regions),
            mode=PerceptionMode.ORACLE_REGISTRY,
            registry_origin=RegistryOrigin.BENCHMARK_ANNOTATION,
            backend_name="phase2_benchmark_annotations",
        )

    def perceive(
        self,
        frame_id: str,
        image: str | Path | Any | None = None,
    ) -> PerceptionResult:
        del image
        try:
            record = self._records[frame_id]
        except KeyError as error:
            raise KeyError(f"no Oracle annotation exists for frame {frame_id!r}") from error
        return self.adapt_phase2_record(record)

    @classmethod
    def registry_from_phase2_record(
        cls,
        record: Mapping[str, Any],
        *,
        user_evidence: Iterable[EvidenceItem] = (),
        user_arguments: Mapping[str, object] | None = None,
    ) -> EvidenceRegistry:
        evidence = tuple(user_evidence)
        if user_arguments:
            evidence += create_user_evidence_items(user_arguments)
        return cls.adapt_phase2_record(record).to_registry(user_evidence=evidence)


class PhysicalAnnotationRegistryAdapter:
    """Convert a future physical human annotation into Oracle perception.

    Ground-truth content and labels are copied losslessly. Runtime OCR/detector
    predictions are deliberately not substituted for those human fields, and
    Oracle confidence remains null.
    """

    @staticmethod
    def adapt_record(
        record: PhysicalImageAnnotations | Mapping[str, Any],
    ) -> PerceptionResult:
        annotated = (
            record
            if isinstance(record, PhysicalImageAnnotations)
            else PhysicalImageAnnotations.model_validate(record)
        )
        regions = tuple(
            EvidenceRegion(
                region_id=region.region_id,
                bbox=region.bbox,
                content=region.ground_truth_content,
                content_type=region.content_type,
                detection_confidence=None,
                ocr_confidence=None,
                semantic_role=region.semantic_role,
                physical_source=region.physical_source.value,
                control_class=region.control_class,
                supports_ground_truth=region.supports_ground_truth,
            )
            for region in annotated.regions
        )
        return PerceptionResult(
            frame_id=annotated.image.image_id,
            regions=regions,
            mode=PerceptionMode.ORACLE_REGISTRY,
            registry_origin=RegistryOrigin.PHYSICAL_ANNOTATION,
            backend_name="phase3.5_physical_human_annotations",
        )

    @classmethod
    def registry_from_record(
        cls,
        record: PhysicalImageAnnotations | Mapping[str, Any],
        *,
        user_evidence: Iterable[EvidenceItem] = (),
        user_arguments: Mapping[str, object] | None = None,
    ) -> EvidenceRegistry:
        evidence = tuple(user_evidence)
        if user_arguments:
            evidence += create_user_evidence_items(user_arguments)
        return cls.adapt_record(record).to_registry(user_evidence=evidence)


class AutomaticPerceptionBackend(ABC):
    """Independent OCR/detection backend contract, intentionally action-free."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def extract_regions(
        self,
        frame_id: str,
        image: str | Path | Any,
    ) -> Sequence[EvidenceRegion | Mapping[str, Any]]:
        """Return regions only: never an action, policy result, or free-form bbox."""

        raise NotImplementedError


class AutomaticRegistryPerception(PerceptionInterface):
    """Registry builder around an explicitly independent perception backend."""

    mode = PerceptionMode.AUTOMATIC_REGISTRY

    def __init__(self, backend: AutomaticPerceptionBackend) -> None:
        if not isinstance(backend, AutomaticPerceptionBackend):
            raise TypeError(
                "automatic perception requires an AutomaticPerceptionBackend, "
                "not an action-model provider"
            )
        self._backend = backend

    def perceive(
        self,
        frame_id: str,
        image: str | Path | Any | None = None,
    ) -> PerceptionResult:
        if image is None:
            raise ValueError("automatic perception requires an image/frame payload")
        raw_regions = self._backend.extract_regions(frame_id, image)
        if isinstance(raw_regions, (str, bytes)) or not isinstance(raw_regions, Sequence):
            raise TypeError("automatic backend must return a sequence of regions")
        regions = tuple(
            region
            if isinstance(region, EvidenceRegion)
            else EvidenceRegion.model_validate(region)
            for region in raw_regions
        )
        return PerceptionResult(
            frame_id=frame_id,
            regions=regions,
            mode=PerceptionMode.AUTOMATIC_REGISTRY,
            registry_origin=RegistryOrigin.AUTOMATIC_PERCEPTION,
            backend_name=self._backend.backend_name,
        )


def oracle_registry_from_phase2_record(
    record: Mapping[str, Any],
    *,
    user_evidence: Iterable[EvidenceItem] = (),
    user_arguments: Mapping[str, object] | None = None,
) -> EvidenceRegistry:
    return OracleRegistryAdapter.registry_from_phase2_record(
        record,
        user_evidence=user_evidence,
        user_arguments=user_arguments,
    )


# Explicit naming aliases for experiment code and documentation.
OraclePerceptionInterface = OracleRegistryAdapter
Phase2OracleRegistryAdapter = OracleRegistryAdapter
PhysicalOracleRegistryAdapter = PhysicalAnnotationRegistryAdapter
AutomaticPerceptionInterface = AutomaticRegistryPerception
