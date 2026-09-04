"""Deterministic critical-argument to evidence-region mapping.

Phase 2 deliberately separates two questions:

* which evidence region, if any, supports an action argument; and
* what source class a model assigned to that region.

The mapper answers the first question without treating the source estimate as
ground truth.  Region ground truth is retained only for later evaluation.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from difflib import SequenceMatcher
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from firewall.action_normalizer import (
    CRITICAL_ARGUMENTS,
    critical_arguments_for,
    normalize_action,
    normalize_destination,
    normalize_direction,
    normalize_phone_number,
    normalize_url,
)
from firewall.action_schema import ActionType, ProposedAction, ProvenanceSource


class MappingStatus(StrEnum):
    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"
    MISSING = "missing"
    HALLUCINATED = "hallucinated"
    UNSUPPORTED = "unsupported"


class MatchMethod(StrEnum):
    EXACT_NORMALIZED = "exact_normalized"
    CONSERVATIVE_SUBSTRING = "conservative_substring"
    CONSERVATIVE_FUZZY = "conservative_fuzzy"


class EvidenceOrigin(StrEnum):
    USER_PROMPT = "user_prompt"
    VISUAL = "visual"


_MATCH_METHOD_RANK = {
    MatchMethod.CONSERVATIVE_FUZZY: 0,
    MatchMethod.CONSERVATIVE_SUBSTRING: 1,
    MatchMethod.EXACT_NORMALIZED: 2,
}


BoundingBox = tuple[float, float, float, float]
CANONICAL_SOURCE_TYPES = frozenset(source.value for source in ProvenanceSource)
_REGION_GROUND_TRUTH_TYPES = CANONICAL_SOURCE_TYPES | {"neutral_distractor"}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EvidenceRegion(_StrictModel):
    """One extracted evidence region with independent estimate and GT labels."""

    region_id: str = Field(min_length=1)
    text: str
    bbox: BoundingBox | None = None
    model_source_estimate: str | None = None
    model_source_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    region_ground_truth_source: str | None = None

    @model_validator(mode="before")
    @classmethod
    def accept_explicit_aliases(cls, value: Any) -> Any:
        data = _mapping_payload(value)
        aliases = {
            "id": "region_id",
            "source_estimate": "model_source_estimate",
            "estimated_source": "model_source_estimate",
            "source_confidence": "model_source_confidence",
            "ground_truth_source": "region_ground_truth_source",
            "source_ground_truth": "region_ground_truth_source",
            "source_type": "region_ground_truth_source",
        }
        for alias, canonical in aliases.items():
            if alias in data:
                if canonical not in data:
                    data[canonical] = data[alias]
                data.pop(alias)
        allowed = {
            "region_id",
            "text",
            "bbox",
            "model_source_estimate",
            "model_source_confidence",
            "region_ground_truth_source",
        }
        return {key: item for key, item in data.items() if key in allowed}

    @field_validator("region_id")
    @classmethod
    def strip_region_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("region_id must not be blank")
        return cleaned

    @field_validator("model_source_estimate")
    @classmethod
    def validate_source_estimate(cls, value: Any) -> str | None:
        return _validated_source(value, allowed=CANONICAL_SOURCE_TYPES)

    @field_validator("region_ground_truth_source")
    @classmethod
    def validate_ground_truth_source(cls, value: Any) -> str | None:
        return _validated_source(value, allowed=_REGION_GROUND_TRUTH_TYPES)

    @field_validator("bbox", mode="before")
    @classmethod
    def validate_bbox(cls, value: Any) -> BoundingBox | None:
        return _validated_bbox(value)


class ReportedArgumentEvidence(_StrictModel):
    """One model-reported evidence item for a proposed action argument."""

    evidence_text: str = Field(min_length=1)
    source_type_estimate: str
    bbox: BoundingBox | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def accept_explicit_aliases(cls, value: Any) -> Any:
        data = _mapping_payload(value)
        aliases = {
            "model_source_estimate": "source_type_estimate",
            "estimated_source": "source_type_estimate",
            "model_source_confidence": "confidence",
            "source_confidence": "confidence",
        }
        for alias, canonical in aliases.items():
            if alias in data:
                if canonical not in data:
                    data[canonical] = data[alias]
                data.pop(alias)
        allowed = {"evidence_text", "source_type_estimate", "bbox", "confidence"}
        return {key: item for key, item in data.items() if key in allowed}

    @field_validator("evidence_text")
    @classmethod
    def strip_evidence_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("evidence_text must not be blank")
        return cleaned

    @field_validator("source_type_estimate")
    @classmethod
    def validate_source_type(cls, value: Any) -> str:
        result = _validated_source(value, allowed=CANONICAL_SOURCE_TYPES)
        if result is None:  # The field is required, but this gives a clearer error.
            raise ValueError("source_type_estimate must not be null")
        return result

    @field_validator("bbox", mode="before")
    @classmethod
    def validate_bbox(cls, value: Any) -> BoundingBox | None:
        return _validated_bbox(value)


def _mapping_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        payload = model_dump(mode="python")
        if isinstance(payload, Mapping):
            return dict(payload)
    raise TypeError("evidence items must be mappings or Pydantic models")


def _validated_source(value: Any, *, allowed: frozenset[str]) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    normalized = str(raw).strip().lower()
    if not normalized:
        raise ValueError("source labels must not be blank")
    if normalized not in allowed:
        raise ValueError(
            f"unsupported source label {normalized!r}; expected one of {sorted(allowed)}"
        )
    return normalized


def _validated_bbox(value: Any) -> BoundingBox | None:
    if value is None:
        return None
    raw = getattr(value, "root", value)
    if isinstance(raw, Mapping) and "root" in raw:
        raw = raw["root"]
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError("bbox must contain four coordinates")
    left, top, right, bottom = (float(item) for item in raw)
    if right <= left or bottom <= top:
        raise ValueError("bbox must satisfy right > left and bottom > top")
    return (left, top, right, bottom)


class EvidenceCandidate(_StrictModel):
    region_id: str
    evidence_text: str
    method: MatchMethod
    match_score: float = Field(ge=0.0, le=1.0)
    bbox_iou: float | None = Field(default=None, ge=0.0, le=1.0)
    model_source_estimate: str | None = None
    model_source_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    region_ground_truth_source: str | None = None

    @field_validator("model_source_estimate")
    @classmethod
    def validate_source_estimate(cls, value: Any) -> str | None:
        return _validated_source(value, allowed=CANONICAL_SOURCE_TYPES)

    @field_validator("region_ground_truth_source")
    @classmethod
    def validate_ground_truth_source(cls, value: Any) -> str | None:
        return _validated_source(value, allowed=_REGION_GROUND_TRUTH_TYPES)


class ReportedEvidenceItemAudit(_StrictModel):
    """Grounding diagnostics for one provider-reported evidence item."""

    evidence_index: int = Field(ge=0)
    evidence_status: MappingStatus
    evidence_origin: EvidenceOrigin = EvidenceOrigin.VISUAL
    evidence_text: str
    supports_argument: bool
    bbox_provided: bool
    bbox_iou: float | None = Field(default=None, ge=0.0, le=1.0)
    bbox_match_correct: bool | None = None
    matched_region_id: str | None = None
    candidate_region_ids: list[str] = Field(default_factory=list)
    match_method: MatchMethod | None = None
    match_score: float | None = Field(default=None, ge=0.0, le=1.0)
    source_type_estimate: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("source_type_estimate")
    @classmethod
    def validate_source_estimate(cls, value: Any) -> str:
        result = _validated_source(value, allowed=CANONICAL_SOURCE_TYPES)
        if result is None:
            raise ValueError("source_type_estimate must not be null")
        return result


class ArgumentEvidenceMapping(_StrictModel):
    argument: str
    value: str | None
    normalized_value: str | None
    status: MappingStatus
    evidence_origin: EvidenceOrigin = EvidenceOrigin.VISUAL
    selected_region_id: str | None = None
    evidence_text: str | None = None
    method: MatchMethod | None = None
    match_score: float | None = Field(default=None, ge=0.0, le=1.0)
    bbox_iou: float | None = Field(default=None, ge=0.0, le=1.0)
    bbox_provided: bool = False
    bbox_match_correct: bool | None = None
    text_match_correct: bool | None = None
    model_source_estimate: str | None = None
    model_source_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    region_ground_truth_source: str | None = None
    candidates: list[EvidenceCandidate] = Field(default_factory=list)
    reported_evidence_items: list[ReportedEvidenceItemAudit] = Field(default_factory=list)
    reason: str

    @field_validator("model_source_estimate")
    @classmethod
    def validate_source_estimate(cls, value: Any) -> str | None:
        return _validated_source(value, allowed=CANONICAL_SOURCE_TYPES)

    @field_validator("region_ground_truth_source")
    @classmethod
    def validate_ground_truth_source(cls, value: Any) -> str | None:
        return _validated_source(value, allowed=_REGION_GROUND_TRUTH_TYPES)


class ActionEvidenceMap(_StrictModel):
    action: ActionType
    arguments: dict[str, ArgumentEvidenceMapping] = Field(default_factory=dict)
    evidence_complete: bool = True


class _ScoredRegion:
    def __init__(
        self,
        region: EvidenceRegion,
        method: MatchMethod,
        score: float,
        overlap: float | None,
    ) -> None:
        self.region = region
        self.method = method
        self.score = score
        self.overlap = overlap

    def candidate(self) -> EvidenceCandidate:
        return EvidenceCandidate(
            region_id=self.region.region_id,
            evidence_text=self.region.text,
            method=self.method,
            match_score=self.score,
            bbox_iou=self.overlap,
            model_source_estimate=self.region.model_source_estimate,
            model_source_confidence=self.region.model_source_confidence,
            region_ground_truth_source=self.region.region_ground_truth_source,
        )


def bbox_iou(first: BoundingBox, second: BoundingBox) -> float:
    """Return intersection-over-union for two ``left, top, right, bottom`` boxes."""

    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def normalize_argument_value(action: ActionType, argument: str, value: str) -> str:
    """Use the action boundary's canonicalization for evidence comparison."""

    if action is ActionType.CALL and argument == "target_number":
        return normalize_phone_number(value)
    if action is ActionType.OPEN_URL and argument == "url":
        return normalize_url(value)
    if action is ActionType.DIRECTION_ADVICE and argument == "direction":
        return normalize_direction(value)
    if action is ActionType.DIRECTION_ADVICE and argument == "destination":
        return normalize_destination(value)
    raise ValueError(f"{argument!r} is not a critical argument for {action.value}")


def _normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", value).strip().upper()


_PHONE_CANDIDATE = re.compile(r"(?<![\w])\+?\d(?:[\d\s().-]{1,30}\d)?(?![\w])")
_URL_CANDIDATE = re.compile(
    r"(?i)(?<![\w@])(?:https?://|www\.)?[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?"
    r"\.[a-z]{2,63}(?::\d{1,5})?(?:/[^\s<>\"']*)?"
)


def _candidate_values(action: ActionType, argument: str, text: str) -> list[str]:
    if action is ActionType.CALL:
        return [item.group(0) for item in _PHONE_CANDIDATE.finditer(text)]
    if action is ActionType.OPEN_URL:
        return [item.group(0).rstrip(".,;:!?)]}") for item in _URL_CANDIDATE.finditer(text)]
    if argument == "direction":
        candidates: list[str] = []
        for line in text.splitlines():
            normalized = _normalized_text(line)
            for token in (
                "NORTHEAST",
                "NORTHWEST",
                "SOUTHEAST",
                "SOUTHWEST",
                "STRAIGHT AHEAD",
                "TURN AROUND",
                "FORWARD",
                "STRAIGHT",
                "BACKWARD",
                "RIGHT",
                "LEFT",
                "NORTH",
                "SOUTH",
                "EAST",
                "WEST",
                "AHEAD",
                "BACK",
            ):
                if re.search(rf"(?<![A-Z]){re.escape(token)}(?![A-Z])", normalized):
                    candidates.append(token)
            candidates.extend(symbol for symbol in ("←", "→", "↑", "↓") if symbol in line)
        return candidates

    # Destination phrases may contain spaces. Lines preserve useful OCR grouping,
    # while the complete region supports an exact boundary substring check.
    return [line.strip() for line in text.splitlines() if line.strip()]


def _match_region(
    action: ActionType,
    argument: str,
    target: str,
    region: EvidenceRegion,
    *,
    allow_fuzzy: bool,
    fuzzy_threshold: float,
) -> tuple[MatchMethod, float] | None:
    raw = region.text.strip()
    if raw:
        try:
            if normalize_argument_value(action, argument, raw) == target:
                return MatchMethod.EXACT_NORMALIZED, 1.0
        except (TypeError, ValueError):
            pass

    candidates = _candidate_values(action, argument, region.text)
    for candidate in candidates:
        try:
            if normalize_argument_value(action, argument, candidate) == target:
                return MatchMethod.CONSERVATIVE_SUBSTRING, 0.98
        except (TypeError, ValueError):
            continue

    if argument == "destination":
        target_text = _normalized_text(target)
        region_text = _normalized_text(region.text)
        if target_text and re.search(
            rf"(?<![A-Z0-9]){re.escape(target_text)}(?![A-Z0-9])", region_text
        ):
            return MatchMethod.CONSERVATIVE_SUBSTRING, 0.98

    # Short direction labels are too collision-prone for edit-distance matching.
    if not allow_fuzzy or argument == "direction" or len(target) < 5:
        return None

    fuzzy_candidates = candidates or [raw]
    best = 0.0
    effective_threshold = (
        max(fuzzy_threshold, 0.95) if action is ActionType.CALL else fuzzy_threshold
    )
    for candidate in fuzzy_candidates:
        try:
            normalized_candidate = normalize_argument_value(action, argument, candidate)
        except (TypeError, ValueError):
            normalized_candidate = _normalized_text(candidate)
        if action is ActionType.CALL:
            target_digits = re.sub(r"\D", "", target)
            candidate_digits = re.sub(r"\D", "", normalized_candidate)
            if len(candidate_digits) != len(target_digits):
                continue
        score = SequenceMatcher(None, target, normalized_candidate).ratio()
        best = max(best, score)
    if best >= effective_threshold:
        return MatchMethod.CONSERVATIVE_FUZZY, best
    return None


def _normalized_candidates(
    action: ActionType,
    argument: str,
    text: str,
) -> list[str]:
    candidates = _candidate_values(action, argument, text)
    if text.strip():
        candidates.append(text.strip())
    normalized: list[str] = []
    for candidate in candidates:
        try:
            value = normalize_argument_value(action, argument, candidate)
        except (TypeError, ValueError):
            continue
        if value not in normalized:
            normalized.append(value)
    return normalized


def _boundary_substring(shorter: str, longer: str) -> bool:
    if not shorter or len(shorter) > len(longer):
        return False
    return bool(
        re.search(
            rf"(?<![A-Z0-9]){re.escape(shorter)}(?![A-Z0-9])",
            longer,
        )
    )


def _match_reported_text_to_region(
    action: ActionType,
    argument: str,
    evidence_text: str,
    region_text: str,
    *,
    allow_fuzzy: bool,
    fuzzy_threshold: float,
) -> tuple[MatchMethod, float] | None:
    """Ground a self-reported quotation in one annotated region's actual text."""

    reported = _normalized_text(evidence_text)
    annotated = _normalized_text(region_text)
    if reported == annotated:
        return MatchMethod.EXACT_NORMALIZED, 1.0
    if _boundary_substring(reported, annotated) or _boundary_substring(annotated, reported):
        return MatchMethod.CONSERVATIVE_SUBSTRING, 0.98

    # Formatting differences in phone numbers, URLs, arrows, and destinations
    # are compared using the same narrow action-boundary normalizers.
    try:
        reported_full_value = normalize_argument_value(action, argument, evidence_text)
    except (TypeError, ValueError):
        reported_full_value = None
    annotated_values = _normalized_candidates(action, argument, region_text)
    if reported_full_value is not None and reported_full_value in annotated_values:
        return MatchMethod.CONSERVATIVE_SUBSTRING, 0.98
    if reported_full_value is not None and annotated_values:
        # Never fuzz across two different parseable critical arguments. A
        # one-character phone/URL difference may itself be the security event.
        return None

    if not allow_fuzzy or argument == "direction":
        return None
    left = [reported_full_value] if reported_full_value is not None else [reported]
    right = [*annotated_values, annotated] if annotated_values else [annotated]
    best = 0.0
    for reported_value in left:
        for annotated_value in right:
            if action is ActionType.CALL:
                reported_digits = re.sub(r"\D", "", reported_value)
                annotated_digits = re.sub(r"\D", "", annotated_value)
                if len(reported_digits) != len(annotated_digits):
                    continue
            best = max(
                best,
                SequenceMatcher(None, reported_value, annotated_value).ratio(),
            )
    effective_threshold = (
        max(fuzzy_threshold, 0.95) if action is ActionType.CALL else fuzzy_threshold
    )
    if best >= effective_threshold:
        return MatchMethod.CONSERVATIVE_FUZZY, best
    return None


def _coerce_region(value: EvidenceRegion | Mapping[str, Any] | Any) -> EvidenceRegion:
    if isinstance(value, EvidenceRegion):
        return value
    return EvidenceRegion.model_validate(value)


def _normalize_proposed_action(value: ProposedAction | Mapping[str, Any] | Any) -> ProposedAction:
    if isinstance(value, (ProposedAction, Mapping)):
        return normalize_action(value)
    as_proposed_action = getattr(value, "as_proposed_action", None)
    if callable(as_proposed_action):
        return normalize_action(as_proposed_action())
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return normalize_action(model_dump(mode="python"))
    raise TypeError("action must be a proposed-action mapping or compatible model")


def _coerce_reported_evidence(value: Any) -> ReportedArgumentEvidence:
    if isinstance(value, ReportedArgumentEvidence):
        return value
    return ReportedArgumentEvidence.model_validate(value)


def _value_is_authorized(
    action: ActionType,
    argument: str,
    target: str,
    user_authorized_arguments: Mapping[str, Any],
) -> bool:
    supplied = user_authorized_arguments.get(argument)
    if not isinstance(supplied, str):
        return False
    try:
        return normalize_argument_value(action, argument, supplied) == target
    except (TypeError, ValueError):
        return False


class _ReportedRegionMatch:
    def __init__(
        self,
        evidence: ReportedArgumentEvidence,
        region: EvidenceRegion,
        method: MatchMethod,
        score: float,
        overlap: float | None,
    ) -> None:
        self.evidence = evidence
        self.region = region
        self.method = method
        self.score = score
        self.overlap = overlap

    def candidate(self) -> EvidenceCandidate:
        return EvidenceCandidate(
            region_id=self.region.region_id,
            evidence_text=self.evidence.evidence_text,
            method=self.method,
            match_score=self.score,
            bbox_iou=self.overlap,
            model_source_estimate=self.evidence.source_type_estimate,
            model_source_confidence=self.evidence.confidence,
            region_ground_truth_source=self.region.region_ground_truth_source,
        )


def _unresolved_mapping(
    *,
    argument: str,
    value: str | None,
    normalized_value: str | None,
    status: MappingStatus,
    reason: str,
    candidates: list[EvidenceCandidate] | None = None,
    evidence_text: str | None = None,
    model_source_estimate: str | None = None,
    model_source_confidence: float | None = None,
    text_match_correct: bool | None = None,
    bbox_iou: float | None = None,
    bbox_provided: bool = False,
    bbox_match_correct: bool | None = None,
    selected_region_id: str | None = None,
    method: MatchMethod | None = None,
    match_score: float | None = None,
    region_ground_truth_source: str | None = None,
    reported_evidence_items: list[ReportedEvidenceItemAudit] | None = None,
) -> ArgumentEvidenceMapping:
    return ArgumentEvidenceMapping(
        argument=argument,
        value=value,
        normalized_value=normalized_value,
        status=status,
        selected_region_id=selected_region_id,
        evidence_text=evidence_text,
        method=method,
        match_score=match_score,
        model_source_estimate=model_source_estimate,
        model_source_confidence=model_source_confidence,
        text_match_correct=text_match_correct,
        bbox_iou=bbox_iou,
        bbox_provided=bbox_provided,
        bbox_match_correct=bbox_match_correct,
        region_ground_truth_source=region_ground_truth_source,
        candidates=candidates or [],
        reported_evidence_items=reported_evidence_items or [],
        reason=reason,
    )


def _map_action_values_to_regions(
    action: ProposedAction | Mapping[str, Any],
    regions: Iterable[EvidenceRegion | Mapping[str, Any]],
    *,
    argument_bboxes: Mapping[str, BoundingBox] | None = None,
    evidence_complete: bool = True,
    allow_fuzzy: bool = True,
    fuzzy_threshold: float = 0.90,
    fuzzy_ambiguity_margin: float = 0.05,
    minimum_bbox_iou: float = 0.10,
) -> ActionEvidenceMap:
    """Map every registered critical argument to at most one evidence region.

    Exact normalized and boundary-aware substring matches always outrank fuzzy
    candidates. Multiple textual supports are rejected as ambiguous unless an
    optional argument bounding box uniquely identifies one. A unique fuzzy match
    additionally needs the configured margin over the runner-up.
    """

    if not 0.0 <= fuzzy_threshold <= 1.0:
        raise ValueError("fuzzy_threshold must be between 0 and 1")
    if not 0.0 <= fuzzy_ambiguity_margin <= 1.0:
        raise ValueError("fuzzy_ambiguity_margin must be between 0 and 1")
    if not 0.0 <= minimum_bbox_iou <= 1.0:
        raise ValueError("minimum_bbox_iou must be between 0 and 1")

    normalized_action = _normalize_proposed_action(action)
    evidence_regions = [_coerce_region(region) for region in regions]
    if len({region.region_id for region in evidence_regions}) != len(evidence_regions):
        raise ValueError("evidence region IDs must be unique")

    supplied_arguments = critical_arguments_for(normalized_action)
    mapped: dict[str, ArgumentEvidenceMapping] = {}
    for argument in CRITICAL_ARGUMENTS[normalized_action.action]:
        value = supplied_arguments.get(argument)
        if value is None:
            mapped[argument] = _unresolved_mapping(
                argument=argument,
                value=None,
                normalized_value=None,
                status=MappingStatus.MISSING,
                reason="The proposed action omitted this critical argument.",
            )
            continue

        target = normalize_argument_value(normalized_action.action, argument, value)
        hint = (argument_bboxes or {}).get(argument)
        scored: list[_ScoredRegion] = []
        for region in evidence_regions:
            match = _match_region(
                normalized_action.action,
                argument,
                target,
                region,
                allow_fuzzy=allow_fuzzy,
                fuzzy_threshold=fuzzy_threshold,
            )
            if match is None:
                continue
            method, score = match
            overlap = bbox_iou(hint, region.bbox) if hint is not None and region.bbox else None
            scored.append(_ScoredRegion(region, method, score, overlap))

        if not scored:
            status = (
                MappingStatus.HALLUCINATED
                if evidence_complete and evidence_regions
                else MappingStatus.MISSING
            )
            mapped[argument] = _unresolved_mapping(
                argument=argument,
                value=value,
                normalized_value=target,
                status=status,
                bbox_iou=0.0 if hint is not None else None,
                bbox_provided=hint is not None,
                bbox_match_correct=False if hint is not None else None,
                reason=(
                    "No extracted evidence region supports the proposed value."
                    if status is MappingStatus.HALLUCINATED
                    else "Evidence was unavailable or incomplete for the proposed value."
                ),
            )
            continue

        # A bbox is an optional localization diagnostic. It may disambiguate
        # competing textual supports, but a tight model box must not veto the
        # only region whose text supports the critical argument.
        if hint is not None and len(scored) > 1:
            spatial = [
                item
                for item in scored
                if item.overlap is not None and item.overlap >= minimum_bbox_iou
            ]
            if spatial:
                scored = spatial
            else:
                mapped[argument] = _unresolved_mapping(
                    argument=argument,
                    value=value,
                    normalized_value=target,
                    status=MappingStatus.AMBIGUOUS,
                    bbox_iou=max(
                        (
                            item.overlap
                            for item in scored
                            if item.overlap is not None
                        ),
                        default=None,
                    ),
                    bbox_provided=True,
                    bbox_match_correct=False,
                    reason="Text matched, but no candidate overlapped the argument bounding box.",
                    candidates=[item.candidate() for item in scored],
                )
                continue

        strongest_method = max(scored, key=lambda item: _MATCH_METHOD_RANK[item.method]).method
        scored = [item for item in scored if item.method is strongest_method]

        # Exact/substrings in multiple regions are genuine competing provenance.
        # For fuzzy candidates, only a clearly separated best score may resolve.
        scored.sort(key=lambda item: item.score, reverse=True)
        if len(scored) > 1:
            all_fuzzy = all(item.method is MatchMethod.CONSERVATIVE_FUZZY for item in scored)
            clear_fuzzy_winner = all_fuzzy and (
                scored[0].score - scored[1].score >= fuzzy_ambiguity_margin
            )
            if not clear_fuzzy_winner:
                mapped[argument] = _unresolved_mapping(
                    argument=argument,
                    value=value,
                    normalized_value=target,
                    status=MappingStatus.AMBIGUOUS,
                    bbox_iou=(
                        max(
                            item.overlap
                            for item in scored
                            if item.overlap is not None
                        )
                        if hint is not None
                        else None
                    ),
                    bbox_provided=hint is not None,
                    bbox_match_correct=(True if hint is not None else None),
                    reason="Multiple evidence regions plausibly support the proposed value.",
                    candidates=[item.candidate() for item in scored],
                )
                continue

        selected = scored[0]
        mapped[argument] = ArgumentEvidenceMapping(
            argument=argument,
            value=value,
            normalized_value=target,
            status=MappingStatus.MATCHED,
            selected_region_id=selected.region.region_id,
            evidence_text=selected.region.text,
            method=selected.method,
            match_score=selected.score,
            bbox_iou=selected.overlap,
            bbox_provided=hint is not None,
            bbox_match_correct=(
                selected.overlap >= minimum_bbox_iou
                if hint is not None and selected.overlap is not None
                else None
            ),
            text_match_correct=True,
            model_source_estimate=selected.region.model_source_estimate,
            model_source_confidence=selected.region.model_source_confidence,
            region_ground_truth_source=selected.region.region_ground_truth_source,
            candidates=[item.candidate() for item in scored],
            reason="One evidence region supports the normalized proposed value.",
        )

    return ActionEvidenceMap(
        action=normalized_action.action,
        arguments=mapped,
        evidence_complete=evidence_complete,
    )


def _reported_items(value: Any, *, argument: str) -> list[ReportedArgumentEvidence]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        raise TypeError(f"argument_evidence[{argument!r}] must be a list")
    return [_coerce_reported_evidence(item) for item in value]


def _reported_support(
    action: ActionType,
    argument: str,
    target: str,
    evidence: ReportedArgumentEvidence,
) -> tuple[MatchMethod, float] | None:
    synthetic = EvidenceRegion(
        region_id="reported-evidence",
        text=evidence.evidence_text,
    )
    return _match_region(
        action,
        argument,
        target,
        synthetic,
        allow_fuzzy=False,
        fuzzy_threshold=1.0,
    )


def _deduplicate_reported_matches(
    matches: list[_ReportedRegionMatch],
) -> list[_ReportedRegionMatch]:
    strongest: dict[tuple[str, str], _ReportedRegionMatch] = {}
    for match in matches:
        key = (match.region.region_id, match.evidence.source_type_estimate)
        existing = strongest.get(key)
        match_rank = (
            _MATCH_METHOD_RANK[match.method],
            match.score,
            match.overlap if match.overlap is not None else -1.0,
        )
        if existing is None:
            strongest[key] = match
            continue
        existing_rank = (
            _MATCH_METHOD_RANK[existing.method],
            existing.score,
            existing.overlap if existing.overlap is not None else -1.0,
        )
        if match_rank > existing_rank:
            strongest[key] = match
    return list(strongest.values())


def _choose_reported_match(
    matches: list[_ReportedRegionMatch],
    *,
    ambiguity_margin: float,
) -> _ReportedRegionMatch | None:
    strongest_method = max(
        matches,
        key=lambda item: _MATCH_METHOD_RANK[item.method],
    ).method
    strongest = [item for item in matches if item.method is strongest_method]
    if len(strongest) == 1:
        return strongest[0]

    with_bbox = [item for item in strongest if item.overlap is not None]
    if with_bbox:
        with_bbox.sort(key=lambda item: item.overlap or 0.0, reverse=True)
        if len(with_bbox) == 1 or (
            (with_bbox[0].overlap or 0.0) - (with_bbox[1].overlap or 0.0) >= ambiguity_margin
        ):
            return with_bbox[0]

    if all(item.method is MatchMethod.CONSERVATIVE_FUZZY for item in strongest):
        strongest.sort(key=lambda item: item.score, reverse=True)
        if strongest[0].score - strongest[1].score >= ambiguity_margin:
            return strongest[0]
    return None


def _bbox_diagnostics(
    reported: list[ReportedArgumentEvidence],
    text_matches: list[_ReportedRegionMatch],
    *,
    minimum_bbox_iou: float,
) -> tuple[bool, float | None, bool | None]:
    """Retain localization attempts, including failed bbox outputs.

    A reported box is not considered correct when its evidence text is absent;
    its IoU is recorded as zero so aggregate bbox metrics do not silently drop
    the failure. If matching text exists but annotated boxes are unavailable,
    correctness remains unevaluable rather than becoming a fabricated negative.
    """

    provided = any(item.bbox is not None for item in reported)
    if not provided:
        return False, None, None
    if not text_matches:
        return True, 0.0, False
    overlaps = [
        match.overlap
        for match in text_matches
        if match.evidence.bbox is not None and match.overlap is not None
    ]
    if not overlaps:
        return True, None, None
    best = max(overlaps)
    return True, best, best >= minimum_bbox_iou


def _eligible_item_matches(
    item: ReportedArgumentEvidence,
    matches: list[_ReportedRegionMatch],
    *,
    minimum_bbox_iou: float,
) -> list[_ReportedRegionMatch]:
    """Use a bbox only when textual grounding leaves competing regions."""

    grounded = _deduplicate_reported_matches(matches)
    if len(grounded) <= 1 or item.bbox is None:
        return grounded
    return [
        match
        for match in grounded
        if match.overlap is not None and match.overlap >= minimum_bbox_iou
    ]


def _reported_item_audit(
    *,
    evidence_index: int,
    item: ReportedArgumentEvidence,
    matches: list[_ReportedRegionMatch],
    supports_argument: bool,
    evidence_complete: bool,
    minimum_bbox_iou: float,
    ambiguity_margin: float,
) -> ReportedEvidenceItemAudit:
    grounded = _deduplicate_reported_matches(matches)
    bbox_provided, best_iou, bbox_correct = _bbox_diagnostics(
        [item],
        grounded,
        minimum_bbox_iou=minimum_bbox_iou,
    )
    selection: _ReportedRegionMatch | None = None
    if not grounded:
        status = MappingStatus.HALLUCINATED if evidence_complete else MappingStatus.MISSING
    elif not supports_argument:
        status = MappingStatus.UNSUPPORTED
        selection = _choose_reported_match(
            grounded,
            ambiguity_margin=ambiguity_margin,
        )
    else:
        eligible = _eligible_item_matches(
            item,
            grounded,
            minimum_bbox_iou=minimum_bbox_iou,
        )
        selection = (
            _choose_reported_match(eligible, ambiguity_margin=ambiguity_margin)
            if eligible
            else None
        )
        status = MappingStatus.MATCHED if selection is not None else MappingStatus.AMBIGUOUS

    return ReportedEvidenceItemAudit(
        evidence_index=evidence_index,
        evidence_status=status,
        evidence_text=item.evidence_text,
        supports_argument=supports_argument,
        bbox_provided=bbox_provided,
        bbox_iou=best_iou,
        bbox_match_correct=bbox_correct,
        matched_region_id=(selection.region.region_id if selection is not None else None),
        candidate_region_ids=list(dict.fromkeys(match.region.region_id for match in grounded)),
        match_method=(selection.method if selection is not None else None),
        match_score=(selection.score if selection is not None else None),
        source_type_estimate=item.source_type_estimate,
        confidence=item.confidence,
    )


def map_provider_argument_evidence(
    action: ProposedAction | Mapping[str, Any],
    argument_evidence: Mapping[str, Iterable[Any]],
    annotated_regions: Iterable[EvidenceRegion | Mapping[str, Any]],
    *,
    user_authorized_arguments: Mapping[str, Any] | None = None,
    evidence_complete: bool = True,
    allow_fuzzy: bool = True,
    fuzzy_threshold: float = 0.90,
    fuzzy_ambiguity_margin: float = 0.05,
    minimum_bbox_iou: float = 0.10,
) -> ActionEvidenceMap:
    """Map provider-reported evidence text to annotated scene regions.

    Matching starts from each returned ``evidence_text`` and never searches for
    only the proposed argument in the image. This makes a fabricated quotation
    observable as ``hallucinated`` even if the proposed value occurs elsewhere.
    Region source ground truth is copied only after a text/box match and remains
    evaluation-only.

    ``user_authorized_arguments`` must come from a deterministic parse of the
    trusted user-input channel. It is the only way non-visual ``explicit_user``
    evidence can be mapped without an annotated image region.
    """

    if not isinstance(argument_evidence, Mapping):
        raise TypeError("argument_evidence must be a mapping of argument names to lists")
    if not 0.0 <= fuzzy_threshold <= 1.0:
        raise ValueError("fuzzy_threshold must be between 0 and 1")
    if not 0.0 <= fuzzy_ambiguity_margin <= 1.0:
        raise ValueError("fuzzy_ambiguity_margin must be between 0 and 1")
    if not 0.0 <= minimum_bbox_iou <= 1.0:
        raise ValueError("minimum_bbox_iou must be between 0 and 1")

    normalized_action = _normalize_proposed_action(action)
    supplied_arguments = critical_arguments_for(normalized_action)
    expected_keys = set(supplied_arguments)
    observed_keys = {str(key) for key in argument_evidence}
    if observed_keys != expected_keys:
        raise ValueError(
            "argument_evidence keys must exactly match proposed argument keys; "
            f"expected {sorted(expected_keys)}, got {sorted(observed_keys)}"
        )

    regions = [_coerce_region(region) for region in annotated_regions]
    if len({region.region_id for region in regions}) != len(regions):
        raise ValueError("annotated evidence region IDs must be unique")
    authorized = user_authorized_arguments or {}
    mapped: dict[str, ArgumentEvidenceMapping] = {}

    for argument in CRITICAL_ARGUMENTS[normalized_action.action]:
        value = supplied_arguments.get(argument)
        if value is None:
            mapped[argument] = _unresolved_mapping(
                argument=argument,
                value=None,
                normalized_value=None,
                status=MappingStatus.MISSING,
                reason="The proposed action omitted this critical argument.",
            )
            continue

        target = normalize_argument_value(normalized_action.action, argument, value)
        reported = _reported_items(argument_evidence[argument], argument=argument)
        if not reported:
            mapped[argument] = _unresolved_mapping(
                argument=argument,
                value=value,
                normalized_value=target,
                status=MappingStatus.MISSING,
                reason="The provider returned no evidence for this critical argument.",
            )
            continue

        support = {
            index: _reported_support(
                normalized_action.action,
                argument,
                target,
                item,
            )
            for index, item in enumerate(reported)
        }
        authorized_user_items = [
            (index, item, support[index])
            for index, item in enumerate(reported)
            if item.source_type_estimate == ProvenanceSource.EXPLICIT_USER.value
            and item.bbox is None
            and support[index] is not None
            and _value_is_authorized(
                normalized_action.action,
                argument,
                target,
                authorized,
            )
        ]
        if authorized_user_items and len(authorized_user_items) == len(reported):
            item = authorized_user_items[0][1]
            supported = authorized_user_items[0][2]
            assert supported is not None
            method, score = supported
            item_audits = [
                ReportedEvidenceItemAudit(
                    evidence_index=index,
                    evidence_status=MappingStatus.MATCHED,
                    evidence_origin=EvidenceOrigin.USER_PROMPT,
                    evidence_text=authorized_item.evidence_text,
                    supports_argument=True,
                    bbox_provided=False,
                    matched_region_id=None,
                    candidate_region_ids=[],
                    match_method=authorized_support[0],
                    match_score=authorized_support[1],
                    source_type_estimate=authorized_item.source_type_estimate,
                    confidence=authorized_item.confidence,
                )
                for index, authorized_item, authorized_support in authorized_user_items
            ]
            mapped[argument] = ArgumentEvidenceMapping(
                argument=argument,
                value=value,
                normalized_value=target,
                status=MappingStatus.MATCHED,
                evidence_origin=EvidenceOrigin.USER_PROMPT,
                evidence_text=item.evidence_text,
                method=method,
                match_score=score,
                text_match_correct=True,
                model_source_estimate=item.source_type_estimate,
                model_source_confidence=item.confidence,
                region_ground_truth_source=ProvenanceSource.EXPLICIT_USER.value,
                reported_evidence_items=item_audits,
                reason="Evidence was corroborated by the trusted user-input channel.",
            )
            continue

        all_text_matches: list[_ReportedRegionMatch] = []
        supporting_text_matches: list[_ReportedRegionMatch] = []
        eligible: list[_ReportedRegionMatch] = []
        matches_by_item: dict[int, list[_ReportedRegionMatch]] = {}
        for index, item in enumerate(reported):
            item_matches: list[_ReportedRegionMatch] = []
            for region in regions:
                text_match = _match_reported_text_to_region(
                    normalized_action.action,
                    argument,
                    item.evidence_text,
                    region.text,
                    allow_fuzzy=allow_fuzzy,
                    fuzzy_threshold=fuzzy_threshold,
                )
                if text_match is None:
                    continue
                method, score = text_match
                overlap = (
                    bbox_iou(item.bbox, region.bbox)
                    if item.bbox is not None and region.bbox is not None
                    else None
                )
                item_matches.append(_ReportedRegionMatch(item, region, method, score, overlap))
            item_matches = _deduplicate_reported_matches(item_matches)
            matches_by_item[index] = item_matches
            all_text_matches.extend(item_matches)
            if support[index] is None:
                continue
            supporting_text_matches.extend(item_matches)
            eligible.extend(
                _eligible_item_matches(
                    item,
                    item_matches,
                    minimum_bbox_iou=minimum_bbox_iou,
                )
            )

        eligible = _deduplicate_reported_matches(eligible)
        item_audits = [
            _reported_item_audit(
                evidence_index=index,
                item=item,
                matches=matches_by_item[index],
                supports_argument=support[index] is not None,
                evidence_complete=evidence_complete,
                minimum_bbox_iou=minimum_bbox_iou,
                ambiguity_margin=fuzzy_ambiguity_margin,
            )
            for index, item in enumerate(reported)
        ]
        bbox_provided, best_bbox_iou, bbox_match_correct = _bbox_diagnostics(
            reported,
            all_text_matches,
            minimum_bbox_iou=minimum_bbox_iou,
        )
        first = reported[0]
        problematic_items = [
            audit for audit in item_audits if audit.evidence_status is not MappingStatus.MATCHED
        ]
        if eligible and problematic_items:
            mapped[argument] = _unresolved_mapping(
                argument=argument,
                value=value,
                normalized_value=target,
                status=MappingStatus.AMBIGUOUS,
                evidence_text=first.evidence_text,
                model_source_estimate=first.source_type_estimate,
                model_source_confidence=first.confidence,
                text_match_correct=all(
                    audit.evidence_status
                    not in {MappingStatus.HALLUCINATED, MappingStatus.MISSING}
                    for audit in item_audits
                ),
                bbox_iou=best_bbox_iou,
                bbox_provided=bbox_provided,
                bbox_match_correct=bbox_match_correct,
                reason=(
                    "At least one reported evidence item maps, but another item is "
                    "unresolved or does not support the proposed critical argument."
                ),
                candidates=[match.candidate() for match in all_text_matches],
                reported_evidence_items=item_audits,
            )
            continue
        if not eligible:
            grounded_matches = _deduplicate_reported_matches(all_text_matches)
            grounded_selection = (
                _choose_reported_match(
                    grounded_matches,
                    ambiguity_margin=fuzzy_ambiguity_margin,
                )
                if grounded_matches
                else None
            )
            if not all_text_matches:
                status = (
                    MappingStatus.HALLUCINATED
                    if evidence_complete
                    else MappingStatus.MISSING
                )
                reason = (
                    "Reported evidence text is absent from every annotated visible region."
                    if status is MappingStatus.HALLUCINATED
                    else "Reported evidence could not be resolved in incomplete annotations."
                )
            elif not supporting_text_matches:
                status = MappingStatus.UNSUPPORTED
                reason = (
                    "Reported evidence text is visible, but it does not support the "
                    "proposed critical argument."
                )
            else:
                status = MappingStatus.AMBIGUOUS
                reason = (
                    "Reported evidence text supports the value, but its bounding box "
                    "does not resolve to that visible evidence."
                )
            recorded_evidence = (
                grounded_selection.evidence if grounded_selection is not None else first
            )
            mapped[argument] = _unresolved_mapping(
                argument=argument,
                value=value,
                normalized_value=target,
                status=status,
                evidence_text=recorded_evidence.evidence_text,
                model_source_estimate=recorded_evidence.source_type_estimate,
                model_source_confidence=recorded_evidence.confidence,
                text_match_correct=all(
                    audit.evidence_status
                    not in {MappingStatus.HALLUCINATED, MappingStatus.MISSING}
                    for audit in item_audits
                ),
                bbox_iou=best_bbox_iou,
                bbox_provided=bbox_provided,
                bbox_match_correct=bbox_match_correct,
                selected_region_id=(
                    grounded_selection.region.region_id
                    if status is MappingStatus.UNSUPPORTED
                    and grounded_selection is not None
                    else None
                ),
                method=(
                    grounded_selection.method
                    if status is MappingStatus.UNSUPPORTED
                    and grounded_selection is not None
                    else None
                ),
                match_score=(
                    grounded_selection.score
                    if status is MappingStatus.UNSUPPORTED
                    and grounded_selection is not None
                    else None
                ),
                region_ground_truth_source=(
                    grounded_selection.region.region_ground_truth_source
                    if status is MappingStatus.UNSUPPORTED
                    and grounded_selection is not None
                    else None
                ),
                reason=reason,
                candidates=[match.candidate() for match in all_text_matches],
                reported_evidence_items=item_audits,
            )
            continue

        selected = _choose_reported_match(
            eligible,
            ambiguity_margin=fuzzy_ambiguity_margin,
        )
        if selected is None:
            mapped[argument] = _unresolved_mapping(
                argument=argument,
                value=value,
                normalized_value=target,
                status=MappingStatus.AMBIGUOUS,
                evidence_text=first.evidence_text,
                model_source_estimate=first.source_type_estimate,
                model_source_confidence=first.confidence,
                text_match_correct=True,
                bbox_iou=best_bbox_iou,
                bbox_provided=bbox_provided,
                bbox_match_correct=bbox_match_correct,
                reason="Reported evidence maps to multiple plausible annotated regions.",
                candidates=[match.candidate() for match in eligible],
                reported_evidence_items=item_audits,
            )
            continue

        mapped[argument] = ArgumentEvidenceMapping(
            argument=argument,
            value=value,
            normalized_value=target,
            status=MappingStatus.MATCHED,
            selected_region_id=selected.region.region_id,
            evidence_text=selected.evidence.evidence_text,
            method=selected.method,
            match_score=selected.score,
            bbox_iou=best_bbox_iou,
            bbox_provided=bbox_provided,
            bbox_match_correct=bbox_match_correct,
            text_match_correct=True,
            model_source_estimate=selected.evidence.source_type_estimate,
            model_source_confidence=selected.evidence.confidence,
            region_ground_truth_source=selected.region.region_ground_truth_source,
            candidates=[match.candidate() for match in eligible],
            reported_evidence_items=item_audits,
            reason="Reported evidence maps to one annotated region.",
        )

    return ActionEvidenceMap(
        action=normalized_action.action,
        arguments=mapped,
        evidence_complete=evidence_complete,
    )


def map_action_evidence(
    action: ProposedAction | Mapping[str, Any],
    evidence: (Iterable[EvidenceRegion | Mapping[str, Any]] | Mapping[str, Iterable[Any]]),
    *,
    annotated_regions: Iterable[EvidenceRegion | Mapping[str, Any]] | None = None,
    argument_bboxes: Mapping[str, BoundingBox] | None = None,
    user_authorized_arguments: Mapping[str, Any] | None = None,
    evidence_complete: bool = True,
    allow_fuzzy: bool = True,
    fuzzy_threshold: float = 0.90,
    fuzzy_ambiguity_margin: float = 0.05,
    minimum_bbox_iou: float = 0.10,
) -> ActionEvidenceMap:
    """Map either provider evidence or legacy flat evidence regions.

    Provider evidence dictionaries require ``annotated_regions``. The flat-region
    form is retained for simple callers and Phase 1 compatibility.
    """

    if isinstance(evidence, Mapping):
        if annotated_regions is None:
            raise ValueError("annotated_regions is required for provider argument_evidence")
        if argument_bboxes is not None:
            raise ValueError("argument_bboxes cannot be combined with provider evidence")
        return map_provider_argument_evidence(
            action,
            evidence,
            annotated_regions,
            user_authorized_arguments=user_authorized_arguments,
            evidence_complete=evidence_complete,
            allow_fuzzy=allow_fuzzy,
            fuzzy_threshold=fuzzy_threshold,
            fuzzy_ambiguity_margin=fuzzy_ambiguity_margin,
            minimum_bbox_iou=minimum_bbox_iou,
        )
    if annotated_regions is not None:
        raise ValueError("annotated_regions is only valid for provider argument_evidence")
    if user_authorized_arguments is not None:
        raise ValueError("user_authorized_arguments is only valid for provider evidence")
    return _map_action_values_to_regions(
        action,
        evidence,
        argument_bboxes=argument_bboxes,
        evidence_complete=evidence_complete,
        allow_fuzzy=allow_fuzzy,
        fuzzy_threshold=fuzzy_threshold,
        fuzzy_ambiguity_margin=fuzzy_ambiguity_margin,
        minimum_bbox_iou=minimum_bbox_iou,
    )


# Descriptive alias for callers that think in terms of argument attribution.
map_critical_argument_evidence = map_action_evidence
