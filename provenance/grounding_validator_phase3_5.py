"""Deterministic argument-level grounding for LensGuard Phase 3.5.

Grounding answers whether the immutable evidence records selected by the model
actually support each proposed argument.  It does not decide source authority,
classify maliciousness, infer trust, or alter the proposed value.  Numeric
``grounding_confidence`` remains ``None`` because Phase 3.5 has no calibrated
deterministic confidence estimator.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from firewall.action_normalizer import (
    normalize_destination,
    normalize_direction,
    normalize_phone_number,
    normalize_url,
)
from provenance.reference_validator_phase3_5 import (
    ArgumentReferenceStatus,
    EvidenceReferenceValidation,
    evidence_field,
    registry_get,
    registry_items,
    validate_evidence_references,
)


class GroundingStatus(StrEnum):
    """Required categorical outcomes; these values must never be collapsed."""

    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    AMBIGUOUS = "AMBIGUOUS"
    CONFLICTING = "CONFLICTING"
    MISSING = "MISSING"
    INVALID_REFERENCE = "INVALID_REFERENCE"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArgumentGroundingAssessment(_StrictModel):
    argument_name: str
    argument_value: Any = None
    normalized_argument_value: str | None = None
    status: GroundingStatus
    referenced_evidence_ids: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()
    irrelevant_evidence_ids: tuple[str, ...] = ()
    unreferenced_conflicting_evidence_ids: tuple[str, ...] = ()
    observed_candidate_values: tuple[str, ...] = ()
    grounding_confidence: float | None = None
    reasons: tuple[str, ...] = ()


class GroundingValidationResult(_StrictModel):
    action: str
    frame_id: str | None
    argument_results: dict[str, ArgumentGroundingAssessment] = Field(default_factory=dict)
    reference_validation: EvidenceReferenceValidation

    def for_argument(self, argument_name: str) -> ArgumentGroundingAssessment:
        return self.argument_results[argument_name]

    @property
    def statuses(self) -> dict[str, GroundingStatus]:
        return {name: result.status for name, result in self.argument_results.items()}

    @property
    def all_supported(self) -> bool:
        return all(
            result.status is GroundingStatus.SUPPORTED
            for result in self.argument_results.values()
        )


def _payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python", exclude_none=True)
        if isinstance(dumped, Mapping):
            return dict(dumped)
    return {}


def _mapping(value: Any) -> dict[str, Any]:
    return _payload(value)


def _enum_string(value: Any) -> str:
    return str(getattr(value, "value", value)).strip().upper()


def _plain_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip().upper()


def _normalize_time(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("time must be a string")
    candidate = _plain_text(value).replace(".", "")
    twelve_hour = re.fullmatch(r"(1[0-2]|0?[1-9])(?::([0-5][0-9]))?\s*([AP]M)", candidate)
    if twelve_hour:
        hour = int(twelve_hour.group(1)) % 12
        if twelve_hour.group(3) == "PM":
            hour += 12
        minute = int(twelve_hour.group(2) or 0)
        return f"{hour:02d}:{minute:02d}"
    twenty_four = re.fullmatch(r"([01]?[0-9]|2[0-3]):([0-5][0-9])", candidate)
    if twenty_four:
        return f"{int(twenty_four.group(1)):02d}:{int(twenty_four.group(2)):02d}"
    raise ValueError("time must be an unambiguous 12-hour or 24-hour clock value")


_HAZARD_ALIASES = {
    "STAIR": "STAIRS",
    "STAIRS": "STAIRS",
    "STAIRCASE": "STAIRS",
    "BARRIER": "BARRIER",
    "BARRICADE": "BARRIER",
    "HOLE": "HOLE",
    "POTHOLE": "HOLE",
    "OBSTACLE": "OBSTACLE",
    "OBSTRUCTION": "OBSTACLE",
    "BLOCKED PATH": "BLOCKED_PATH",
    "RESTRICTED PATH": "RESTRICTED_PATH",
    "RESTRICTED AREA": "RESTRICTED_PATH",
    "STEP": "STEP",
    "DROP": "HEIGHT_DIFFERENCE",
    "HEIGHT DIFFERENCE": "HEIGHT_DIFFERENCE",
    "LEDGE": "HEIGHT_DIFFERENCE",
    "NONE": "NONE",
    "NO HAZARD": "NONE",
}


def _normalize_hazard(value: Any) -> str:
    candidate = _plain_text(value).replace("-", " ").replace("_", " ")
    candidate = re.sub(r"\s+", " ", candidate)
    return _HAZARD_ALIASES.get(candidate, candidate.replace(" ", "_"))


def normalize_grounding_value(action: str, argument: str, value: Any) -> str:
    """Narrow, deterministic comparison normalization for one argument."""

    if action in {"CALL", "RESTAURANT_RESERVATION"} and argument == "target_number":
        return normalize_phone_number(value)
    if action == "OPEN_URL" and argument == "url":
        return normalize_url(value)
    if action == "DIRECTION_ADVICE" and argument == "direction":
        return normalize_direction(value)
    if action == "DIRECTION_ADVICE" and argument == "destination":
        return normalize_destination(value)
    if action == "SAFETY_ADVICE" and argument == "safe_to_proceed":
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower()
        raise ValueError("safe_to_proceed must be boolean")
    if action == "SAFETY_ADVICE" and argument == "hazard":
        return _normalize_hazard(value)
    if action == "RESTAURANT_RESERVATION" and argument == "restaurant":
        candidate = _plain_text(value)
        if not candidate:
            raise ValueError("restaurant must not be blank")
        return candidate
    if action == "RESTAURANT_RESERVATION" and argument == "time":
        return _normalize_time(value)
    if action == "RESTAURANT_RESERVATION" and argument == "party_size":
        if isinstance(value, bool):
            raise ValueError("party_size must be an integer")
        number = int(value)
        if number <= 0:
            raise ValueError("party_size must be positive")
        return str(number)
    candidate = _plain_text(value)
    if not candidate:
        raise ValueError("argument must not be blank")
    return candidate


def _claim_values(item: Any, action: str, argument: str) -> list[Any]:
    values: list[Any] = []
    claims = evidence_field(item, "claims", ()) or ()
    for claim in claims:
        claim_action = _enum_string(evidence_field(claim, "action", ""))
        claim_argument = str(evidence_field(claim, "argument", "")).strip()
        if claim_action == action and claim_argument == argument:
            values.append(evidence_field(claim, "value"))
    return values


_PHONE_PATTERN = re.compile(r"(?<![A-Za-z0-9])\+?\d(?:[\d \t().-]{1,}\d)?(?![A-Za-z0-9])")
_URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>{}\[\]\"']+", re.IGNORECASE)
_DIRECTION_TOKENS: tuple[tuple[str, str], ...] = (
    (r"(?:←|⬅(?:️)?)", "LEFT"),
    (r"(?:→|➡(?:️)?)", "RIGHT"),
    (r"(?:↑|⬆(?:️)?)", "STRAIGHT"),
    (r"(?:↓|⬇(?:️)?)", "BACK"),
    (r"\bNORTH\s*EAST\b|\bNORTHEAST\b|\bNE\b", "NORTHEAST"),
    (r"\bSOUTH\s*EAST\b|\bSOUTHEAST\b|\bSE\b", "SOUTHEAST"),
    (r"\bSOUTH\s*WEST\b|\bSOUTHWEST\b|\bSW\b", "SOUTHWEST"),
    (r"\bNORTH\s*WEST\b|\bNORTHWEST\b|\bNW\b", "NORTHWEST"),
    (r"\bLEFT\b", "LEFT"),
    (r"\bRIGHT\b", "RIGHT"),
    (r"\b(?:STRAIGHT(?:\s+AHEAD)?|AHEAD|FORWARD)\b", "STRAIGHT"),
    (r"\b(?:BACK|BACKWARD|REVERSE|TURN\s+AROUND)\b", "BACK"),
    (r"\bNORTH\b", "NORTH"),
    (r"\bEAST\b", "EAST"),
    (r"\bSOUTH\b", "SOUTH"),
    (r"\bWEST\b", "WEST"),
)


def _phone_candidates(content: str) -> list[str]:
    candidates: list[str] = []
    for match in _PHONE_PATTERN.findall(content):
        try:
            candidates.append(normalize_phone_number(match.strip()))
        except (TypeError, ValueError):
            continue
    return candidates


def _url_candidates(content: str) -> list[str]:
    candidates: list[str] = []
    for match in _URL_PATTERN.findall(content):
        try:
            candidates.append(normalize_url(match.rstrip(".,;:!?")))
        except (TypeError, ValueError):
            continue
    return candidates


def _direction_candidates(content: str) -> list[str]:
    normalized = _plain_text(content)
    found: list[str] = []
    for pattern, value in _DIRECTION_TOKENS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            found.append(value)
    return found


def _time_candidates(content: str) -> list[str]:
    results: list[str] = []
    patterns = (
        r"\b(?:1[0-2]|0?[1-9])(?::[0-5][0-9])?\s*[APap]\.?[Mm]\.?(?=\b|\s|$)",
        r"\b(?:[01]?[0-9]|2[0-3]):[0-5][0-9]\b(?!\s*[APap]\.?[Mm])",
    )
    for pattern in patterns:
        for value in re.findall(pattern, content):
            try:
                results.append(_normalize_time(value))
            except (TypeError, ValueError):
                continue
    return results


def _hazard_candidates(content: str, *, content_type: str) -> list[str]:
    normalized = _plain_text(content).replace("-", " ").replace("_", " ")
    candidates: list[str] = []
    for label, canonical in _HAZARD_ALIASES.items():
        if canonical == "NONE":
            continue
        phrase = label.replace("_", " ")
        if not re.search(rf"\b{re.escape(phrase)}S?\b", normalized):
            continue
        # Text saying a hazard does *not* exist is not positive hazard
        # evidence.  Object/spatial detections are not negated by nearby prose.
        if content_type == "text" and re.search(
            rf"\b(?:NO|WITHOUT)\s+(?:[A-Z]+\s+){{0,2}}{re.escape(phrase)}S?\b",
            normalized,
        ):
            continue
        candidates.append(canonical)
    return candidates


def _safety_boolean_candidates(content: str, item: Any) -> list[str]:
    normalized = _plain_text(content)
    content_type = _enum_string(evidence_field(item, "content_type", "text")).lower()
    negative_phrases = (
        r"\bNOT SAFE\b",
        r"\bUNSAFE\b",
        r"\bDO NOT (?:ENTER|PROCEED)\b",
        r"\b(?:PATH|ROUTE|ENTRANCE) BLOCKED\b",
        r"\bRESTRICTED (?:PATH|AREA|ACCESS)\b",
    )
    positive_phrases = (
        r"\bSAFE TO (?:ENTER|PROCEED)\b",
        r"\bPATH CLEAR\b",
        r"\bNO (?:STAIRS?|OBSTACLES?|HAZARDS?)\b",
        r"\bFLAT PATH\b",
    )
    values: list[str] = []
    if any(re.search(pattern, normalized) for pattern in negative_phrases):
        values.append("false")
    if any(re.search(pattern, normalized) for pattern in positive_phrases):
        values.append("true")
    # A detected physical hazard is deterministic support for not proceeding.
    if content_type in {"object", "spatial", "symbol", "other"} and _hazard_candidates(
        content, content_type=content_type
    ):
        values.append("false")
    return values


def _content_candidates(
    item: Any,
    action: str,
    argument: str,
    proposed_normalized: str,
) -> list[str]:
    content = str(evidence_field(item, "content", ""))
    content_type = _enum_string(evidence_field(item, "content_type", "text")).lower()

    if action in {"CALL", "RESTAURANT_RESERVATION"} and argument == "target_number":
        return _phone_candidates(content)
    if action == "OPEN_URL" and argument == "url":
        return _url_candidates(content)
    if action == "DIRECTION_ADVICE" and argument == "direction":
        return _direction_candidates(content)
    if action == "DIRECTION_ADVICE" and argument == "destination":
        return [proposed_normalized] if proposed_normalized in _plain_text(content) else []
    if action == "SAFETY_ADVICE" and argument == "safe_to_proceed":
        return _safety_boolean_candidates(content, item)
    if action == "SAFETY_ADVICE" and argument == "hazard":
        return _hazard_candidates(content, content_type=content_type)
    if action == "RESTAURANT_RESERVATION" and argument == "restaurant":
        return [proposed_normalized] if proposed_normalized in _plain_text(content) else []
    if action == "RESTAURANT_RESERVATION" and argument == "time":
        return _time_candidates(content)
    if action == "RESTAURANT_RESERVATION" and argument == "party_size":
        # Party size is expected to be explicit user evidence.  Read the whole
        # short evidence value, not arbitrary digits embedded in camera text.
        if content_type == "user_input":
            try:
                return [str(int(content.strip()))]
            except (TypeError, ValueError):
                return []
        return []
    return [proposed_normalized] if proposed_normalized in _plain_text(content) else []


def candidate_values_for_evidence(
    item: Any,
    action: str,
    argument: str,
    proposed_normalized: str,
) -> tuple[str, ...]:
    """Extract normalized semantic candidates, preferring annotation claims."""

    raw_values = _claim_values(item, action, argument)
    if not raw_values:
        raw_values = _content_candidates(item, action, argument, proposed_normalized)
    normalized: list[str] = []
    for value in raw_values:
        try:
            candidate = normalize_grounding_value(action, argument, value)
        except (TypeError, ValueError, OverflowError):
            continue
        if candidate not in normalized:
            normalized.append(candidate)
    return tuple(normalized)


def _selected_user_evidence(registry: Any, evidence_ids: list[str]) -> bool:
    for evidence_id in evidence_ids:
        item = registry_get(registry, evidence_id)
        content_type = _enum_string(evidence_field(item, "content_type", "")).lower()
        origin = _enum_string(evidence_field(item, "registry_origin", "")).lower()
        physical_source = _enum_string(evidence_field(item, "physical_source", "")).lower()
        if (
            evidence_id.startswith("USER:")
            and content_type == "user_input"
            and (origin == "user_prompt" or physical_source == "explicit_user")
        ):
            return True
    return False


def _ground_one_argument(
    *,
    action: str,
    argument: str,
    value: Any,
    registry: Any,
    reference_validation: EvidenceReferenceValidation,
) -> ArgumentGroundingAssessment:
    reference_result = reference_validation.argument_results[argument]
    if reference_result.status is ArgumentReferenceStatus.MISSING:
        return ArgumentGroundingAssessment(
            argument_name=argument,
            argument_value=value,
            status=GroundingStatus.MISSING,
            referenced_evidence_ids=reference_result.reference_ids,
            reasons=("critical argument has no complete evidence-reference coverage",),
        )
    if reference_result.status is ArgumentReferenceStatus.INVALID_REFERENCE:
        return ArgumentGroundingAssessment(
            argument_name=argument,
            argument_value=value,
            status=GroundingStatus.INVALID_REFERENCE,
            referenced_evidence_ids=reference_result.reference_ids,
            reasons=("one or more evidence references failed strict validation",),
        )

    try:
        proposed = normalize_grounding_value(action, argument, value)
    except (TypeError, ValueError, OverflowError):
        return ArgumentGroundingAssessment(
            argument_name=argument,
            argument_value=value,
            status=GroundingStatus.UNSUPPORTED,
            referenced_evidence_ids=reference_result.reference_ids,
            reasons=("argument value cannot be deterministically normalized",),
        )

    supporting: list[str] = []
    contradicting: list[str] = []
    irrelevant: list[str] = []
    observed: list[str] = []
    for evidence_id in reference_result.resolved_evidence_ids:
        item = registry_get(registry, evidence_id)
        candidates = candidate_values_for_evidence(item, action, argument, proposed)
        for candidate in candidates:
            if candidate not in observed:
                observed.append(candidate)
        if proposed in candidates:
            supporting.append(evidence_id)
            if any(candidate != proposed for candidate in candidates):
                contradicting.append(evidence_id)
        elif candidates:
            contradicting.append(evidence_id)
        else:
            irrelevant.append(evidence_id)

    # Scan the fixed registry for unresolved competing values.  Explicit user
    # evidence is authoritative for its own supplied value, so unrelated camera
    # candidates do not make that user-derived binding ambiguous.
    unreferenced_conflicts: list[str] = []
    if supporting and not _selected_user_evidence(registry, supporting):
        selected = set(reference_result.resolved_evidence_ids)
        for item in registry_items(registry):
            evidence_id = evidence_field(item, "evidence_id")
            if not isinstance(evidence_id, str) or evidence_id in selected:
                continue
            candidates = candidate_values_for_evidence(item, action, argument, proposed)
            for candidate in candidates:
                if candidate not in observed:
                    observed.append(candidate)
            if candidates and any(candidate != proposed for candidate in candidates):
                unreferenced_conflicts.append(evidence_id)

    reasons: list[str] = []
    if not supporting:
        status = GroundingStatus.UNSUPPORTED
        reasons.append("referenced evidence does not support the proposed argument value")
    elif contradicting or unreferenced_conflicts:
        status = GroundingStatus.CONFLICTING
        reasons.append("supporting evidence coexists with a different candidate value")
    elif irrelevant:
        status = GroundingStatus.AMBIGUOUS
        reasons.append("the reference set mixes supporting and semantically irrelevant evidence")
    else:
        status = GroundingStatus.SUPPORTED
        reasons.append("referenced evidence deterministically supports the proposed argument")

    return ArgumentGroundingAssessment(
        argument_name=argument,
        argument_value=value,
        normalized_argument_value=proposed,
        status=status,
        referenced_evidence_ids=reference_result.reference_ids,
        supporting_evidence_ids=tuple(supporting),
        contradicting_evidence_ids=tuple(dict.fromkeys(contradicting)),
        irrelevant_evidence_ids=tuple(irrelevant),
        unreferenced_conflicting_evidence_ids=tuple(unreferenced_conflicts),
        observed_candidate_values=tuple(observed),
        grounding_confidence=None,
        reasons=tuple(reasons),
    )


def validate_argument_grounding(
    action_output: Any,
    registry: Any,
    *,
    reference_validation: EvidenceReferenceValidation | None = None,
    frame_id: str | None = None,
) -> GroundingValidationResult:
    """Evaluate all critical argument bindings with categorical outcomes."""

    if reference_validation is None:
        reference_validation = validate_evidence_references(
            action_output,
            registry,
            frame_id=frame_id,
        )
    payload = _payload(action_output)
    action = _enum_string(payload.get("action", reference_validation.action))
    arguments = _mapping(payload.get("arguments"))

    results: dict[str, ArgumentGroundingAssessment] = {}
    for argument in reference_validation.expected_arguments:
        value = arguments.get(argument)
        results[argument] = _ground_one_argument(
            action=action,
            argument=argument,
            value=value,
            registry=registry,
            reference_validation=reference_validation,
        )

    return GroundingValidationResult(
        action=action,
        frame_id=reference_validation.frame_id,
        argument_results=results,
        reference_validation=reference_validation,
    )


class GroundingValidator:
    """Reusable deterministic validator bound to one registry."""

    def __init__(self, registry: Any, *, frame_id: str | None = None) -> None:
        self._registry = registry
        self._frame_id = frame_id

    def validate(
        self,
        action_output: Any,
        *,
        reference_validation: EvidenceReferenceValidation | None = None,
    ) -> GroundingValidationResult:
        return validate_argument_grounding(
            action_output,
            self._registry,
            reference_validation=reference_validation,
            frame_id=self._frame_id,
        )


validate_grounding = validate_argument_grounding
