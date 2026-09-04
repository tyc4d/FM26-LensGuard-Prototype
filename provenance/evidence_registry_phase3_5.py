"""Immutable, pre-inference Evidence Registry for LensGuard Phase 3.5.

The registry is constructed by an independent perception path before action
model inference.  It is intentionally not an OCR repair layer: references are
exact identifiers, and an unknown or cross-frame identifier is never mapped to
the nearest annotated region.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator, model_validator

from phase3_5_constants import (
    ControlClass,
    EVIDENCE_SCHEMA_VERSION,
    EvidenceContentType,
    EvidencePhysicalSource,
    RegistryOrigin,
)


_ID_COMPONENT = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_FRAME_ID_RE = re.compile(rf"^{_ID_COMPONENT}$")
_REGION_ID_RE = re.compile(rf"^{_ID_COMPONENT}(?::{_ID_COMPONENT})*$")
_EVIDENCE_ID_RE = re.compile(rf"^{_ID_COMPONENT}(?::{_ID_COMPONENT})+$")
_USER_ARGUMENT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


class EvidenceRegistryError(ValueError):
    """Base exception for deterministic registry contract failures."""


class MalformedEvidenceReferenceError(EvidenceRegistryError):
    """The reference is not a syntactically valid evidence identifier."""


class UnknownEvidenceReferenceError(EvidenceRegistryError):
    """The reference is well-formed but absent from this registry."""


class CrossFrameEvidenceReferenceError(EvidenceRegistryError):
    """The reference identifies camera evidence belonging to another frame."""


class DuplicateEvidenceReferenceError(EvidenceRegistryError):
    """A reference sequence repeats the same evidence identifier."""


class NormalizedBBox(RootModel[tuple[float, float, float, float]]):
    """Normalized ``(x1, y1, x2, y2)`` coordinates with positive extent."""

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_coordinates(self) -> NormalizedBBox:
        x1, y1, x2, y2 = self.root
        if not all(0.0 <= coordinate <= 1.0 for coordinate in self.root):
            raise ValueError("bbox coordinates must be normalized to [0, 1]")
        if x1 >= x2 or y1 >= y2:
            raise ValueError("bbox must satisfy x1 < x2 and y1 < y2")
        return self


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceClaim(_FrozenStrictModel):
    """Lossless benchmark claim metadata; never created by the action VLM."""

    action: str = Field(min_length=1)
    argument: str = Field(min_length=1)
    value: str = Field(min_length=1)
    claim_role: str | None = None

    @field_validator("action", "argument", "value", "claim_role")
    @classmethod
    def strip_nonblank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("claim fields must not be blank")
        return cleaned


def is_well_formed_evidence_id(value: object) -> bool:
    """Return whether ``value`` has the lexical shape of an evidence ID."""

    return isinstance(value, str) and _EVIDENCE_ID_RE.fullmatch(value) is not None


def canonical_evidence_id(frame_id: str, region_id: str) -> str:
    """Build the sole canonical identifier for a camera region in a frame."""

    if not isinstance(frame_id, str) or _FRAME_ID_RE.fullmatch(frame_id) is None:
        raise MalformedEvidenceReferenceError(
            "frame_id must be one nonblank identifier component without ':'"
        )
    if not isinstance(region_id, str) or _REGION_ID_RE.fullmatch(region_id) is None:
        raise MalformedEvidenceReferenceError(
            "region_id must contain nonblank ':'-separated identifier components"
        )
    return f"{frame_id}:{region_id}"


class EvidenceItem(_FrozenStrictModel):
    """One immutable camera- or user-derived evidence record.

    Detection, OCR, and grounding confidence remain separate fields.  Oracle
    adapters leave all three null; in particular they never manufacture a
    numeric certainty of ``1.0`` for human annotations.
    """

    schema_version: str = EVIDENCE_SCHEMA_VERSION
    evidence_id: str = Field(min_length=1)
    frame_id: str | None = None
    region_id: str | None = None
    bbox: NormalizedBBox | None = None
    content: str = Field(min_length=1)
    content_type: EvidenceContentType
    semantic_role: str | None = None
    physical_source: EvidencePhysicalSource | None = None
    control_class: ControlClass | None = None
    supports_ground_truth: bool | None = None
    detection_confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    ocr_confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    grounding_confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    registry_origin: RegistryOrigin

    # Existing Phase 2 labels are retained explicitly rather than re-labelled
    # as Phase 3 physical source or semantic-role ground truth.
    benchmark_source_label: str | None = None
    content_claimed_authority: str | None = None
    claims: tuple[EvidenceClaim, ...] = ()

    @field_validator(
        "evidence_id",
        "frame_id",
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
            raise ValueError("evidence string fields must not be blank")
        return cleaned

    @model_validator(mode="after")
    def validate_identity_and_origin(self) -> EvidenceItem:
        if self.schema_version != EVIDENCE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be exactly {EVIDENCE_SCHEMA_VERSION!r}"
            )
        if not is_well_formed_evidence_id(self.evidence_id):
            raise ValueError("evidence_id is malformed")

        is_user = self.evidence_id.startswith("USER:")
        if is_user:
            argument_name = self.evidence_id.partition(":")[2]
            if _USER_ARGUMENT_RE.fullmatch(argument_name) is None:
                raise ValueError("USER evidence IDs must be USER:<argument_name>")
            if self.content_type is not EvidenceContentType.USER_INPUT:
                raise ValueError("USER evidence must use content_type='user_input'")
            if self.registry_origin is not RegistryOrigin.USER_PROMPT:
                raise ValueError("USER evidence must have registry_origin='user_prompt'")
            if self.frame_id is not None or self.region_id is not None or self.bbox is not None:
                raise ValueError("USER evidence cannot contain frame, region, or bbox fields")
            if self.detection_confidence is not None or self.ocr_confidence is not None:
                raise ValueError("USER evidence cannot contain perception confidence")
            if self.physical_source not in {
                EvidencePhysicalSource.EXPLICIT_USER,
                EvidencePhysicalSource.USER_INPUT,
            }:
                raise ValueError("USER evidence must identify an explicit user source")
            if self.control_class is not None:
                raise ValueError("USER evidence does not use camera control_class labels")
            return self

        if self.content_type is EvidenceContentType.USER_INPUT:
            raise ValueError("camera evidence cannot use content_type='user_input'")
        if self.physical_source in {
            EvidencePhysicalSource.EXPLICIT_USER,
            EvidencePhysicalSource.USER_INPUT,
        }:
            raise ValueError(
                "camera evidence cannot use a user-input physical source; "
                "represent user values as USER evidence without a region or bbox"
            )
        if self.frame_id is None or self.region_id is None:
            raise ValueError("camera evidence requires frame_id and region_id")
        expected_id = canonical_evidence_id(self.frame_id, self.region_id)
        if self.evidence_id != expected_id:
            raise ValueError(
                f"camera evidence_id must be the canonical frame-scoped ID {expected_id!r}"
            )
        if self.registry_origin is RegistryOrigin.USER_PROMPT:
            raise ValueError("camera evidence cannot have registry_origin='user_prompt'")
        return self


def _stable_user_content(value: object) -> str:
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("user evidence content must not be blank")
        return cleaned
    if value is None:
        raise ValueError("user evidence content must not be null")
    if isinstance(value, (bool, int, float)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    raise TypeError("user evidence values must be strings, booleans, integers, or floats")


def create_user_evidence(argument_name: str, value: object) -> EvidenceItem:
    """Create explicit non-camera evidence for one user-supplied argument."""

    if not isinstance(argument_name, str) or _USER_ARGUMENT_RE.fullmatch(argument_name) is None:
        raise ValueError("argument_name is not valid for a USER evidence ID")
    return EvidenceItem(
        evidence_id=f"USER:{argument_name}",
        content=_stable_user_content(value),
        content_type=EvidenceContentType.USER_INPUT,
        semantic_role=argument_name,
        physical_source=EvidencePhysicalSource.EXPLICIT_USER,
        registry_origin=RegistryOrigin.USER_PROMPT,
    )


def create_user_evidence_items(arguments: Mapping[str, object]) -> tuple[EvidenceItem, ...]:
    """Create stable user evidence in the mapping's insertion order."""

    return tuple(create_user_evidence(name, value) for name, value in arguments.items())


class EvidenceRegistry(Mapping[str, EvidenceItem]):
    """Read-only registry containing one camera frame plus optional user input."""

    __slots__ = ("_frame_id", "_items", "_items_by_id", "_schema_version", "_sealed")

    def __init__(
        self,
        frame_id: str,
        items: Iterable[EvidenceItem | Mapping[str, Any]],
        *,
        schema_version: str = EVIDENCE_SCHEMA_VERSION,
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        if not isinstance(frame_id, str) or _FRAME_ID_RE.fullmatch(frame_id) is None:
            raise ValueError("registry frame_id must be one valid identifier component")
        if schema_version != EVIDENCE_SCHEMA_VERSION:
            raise ValueError(
                f"registry schema_version must be exactly {EVIDENCE_SCHEMA_VERSION!r}"
            )

        validated: list[EvidenceItem] = []
        by_id: dict[str, EvidenceItem] = {}
        for raw_item in items:
            item = (
                raw_item
                if isinstance(raw_item, EvidenceItem)
                else EvidenceItem.model_validate(raw_item)
            )
            if item.evidence_id in by_id:
                raise ValueError(f"duplicate evidence_id {item.evidence_id!r}")
            if item.frame_id is not None and item.frame_id != frame_id:
                raise CrossFrameEvidenceReferenceError(
                    f"evidence {item.evidence_id!r} belongs to frame {item.frame_id!r}, "
                    f"not registry frame {frame_id!r}"
                )
            validated.append(item)
            by_id[item.evidence_id] = item

        object.__setattr__(self, "_frame_id", frame_id)
        object.__setattr__(self, "_schema_version", schema_version)
        object.__setattr__(self, "_items", tuple(validated))
        object.__setattr__(self, "_items_by_id", MappingProxyType(by_id))
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise TypeError("EvidenceRegistry is immutable")
        object.__setattr__(self, name, value)

    @property
    def frame_id(self) -> str:
        return self._frame_id

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def items(self) -> tuple[EvidenceItem, ...]:
        return self._items

    def __getitem__(self, evidence_id: str) -> EvidenceItem:
        return self._items_by_id[evidence_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items_by_id)

    def __len__(self) -> int:
        return len(self._items_by_id)

    def contains(self, evidence_id: object) -> bool:
        return isinstance(evidence_id, str) and evidence_id in self._items_by_id

    def validate_reference(self, evidence_id: object) -> EvidenceItem:
        """Resolve an exact reference or raise a specific deterministic error."""

        if not is_well_formed_evidence_id(evidence_id):
            raise MalformedEvidenceReferenceError(f"malformed evidence reference: {evidence_id!r}")
        assert isinstance(evidence_id, str)  # narrowed by lexical validation
        item = self._items_by_id.get(evidence_id)
        if item is not None:
            return item
        if not evidence_id.startswith("USER:") and not evidence_id.startswith(
            f"{self.frame_id}:"
        ):
            referenced_frame = evidence_id.partition(":")[0]
            raise CrossFrameEvidenceReferenceError(
                f"reference belongs to frame {referenced_frame!r}, not {self.frame_id!r}"
            )
        raise UnknownEvidenceReferenceError(
            f"evidence reference {evidence_id!r} is not present in this registry"
        )

    def require(self, evidence_id: object) -> EvidenceItem:
        return self.validate_reference(evidence_id)

    def resolve(self, evidence_ids: Iterable[object]) -> tuple[EvidenceItem, ...]:
        resolved: list[EvidenceItem] = []
        seen: set[str] = set()
        for evidence_id in evidence_ids:
            item = self.validate_reference(evidence_id)
            if item.evidence_id in seen:
                raise DuplicateEvidenceReferenceError(
                    f"duplicate evidence reference {item.evidence_id!r}"
                )
            seen.add(item.evidence_id)
            resolved.append(item)
        return tuple(resolved)

    def model_dump(self, *, include_dataset_labels: bool = True) -> dict[str, Any]:
        """Return a detached JSON-compatible snapshot of the immutable registry."""

        excluded = set()
        if not include_dataset_labels:
            excluded = {
                "control_class",
                "supports_ground_truth",
                "benchmark_source_label",
                "content_claimed_authority",
                "claims",
                "grounding_confidence",
            }
        serialized = []
        for item in self._items:
            payload = item.model_dump(mode="json", exclude=excluded)
            serialized.append(payload)
        return {
            "schema_version": self.schema_version,
            "frame_id": self.frame_id,
            "items": serialized,
        }

    def as_model_input(self) -> dict[str, Any]:
        """Return the registry view safe to place in an action-model prompt.

        Dataset-only labels and post-hoc grounding confidence are withheld.  A
        caller cannot mutate registry state through the returned dictionaries.
        """

        return self.model_dump(include_dataset_labels=False)


# Explicit alias used in architecture documents and some experiment code.
UserEvidence = EvidenceItem
