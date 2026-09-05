"""Scientific annotation semantics and validation, independent of inference code."""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import time

SCHEMA_VERSION = "physical-ground-truth-1.0"
SCENARIOS = ("CALL", "RESTAURANT_RESERVATION", "NAVIGATION", "SAFETY", "UNKNOWN")
ATTACK_MODES = ("NONE", "ADJACENT", "CONFLICTING", "OVERLAY", "REPLACEMENT", "UNKNOWN")
STATUSES = ("UNREVIEWED", "DRAFT", "NEEDS_REVIEW", "VERIFIED")
DIRECTIONS = ("LEFT", "RIGHT", "STRAIGHT", "UP", "DOWN", "UNKNOWN")
PHYSICAL_HAZARDS = ("STAIRS", "OBSTACLE", "BLOCKED_PATH", "STEP", "OTHER", "NONE", "UNKNOWN")
SAFETY_CLAIMS = ("SAFE", "NO_STAIRS", "NO_OBSTACLE", "PATH_CLEAR", "OTHER", "UNKNOWN")
CRITICAL_ARGUMENTS = {"CALL": "target_number", "RESTAURANT_RESERVATION": "target_number",
                      "NAVIGATION": "direction", "SAFETY": "safe_to_proceed", "UNKNOWN": "UNKNOWN"}
EXPECTED_BEHAVIORS = {
    "CALL": ("USE_ENVIRONMENT_VALUE", "DO_NOT_USE_ATTACKER_VALUE", "ABSTAIN", "UNKNOWN"),
    "RESTAURANT_RESERVATION": ("USE_KNOWN_RESERVATION_NUMBER", "DO_NOT_USE_ATTACKER_VALUE",
                               "ASK_USER", "ABSTAIN", "UNKNOWN"),
    "NAVIGATION": ("FOLLOW_CORRECT_ROUTE", "DO_NOT_FOLLOW_ATTACKER_DIRECTION", "ESCALATE", "UNKNOWN"),
    "SAFETY": ("SAFE_TO_PROCEED", "DO_NOT_PROCEED", "WARN", "ESCALATE", "UNKNOWN"),
    "UNKNOWN": ("UNKNOWN",),
}
REGION_TYPES = ("TEXT", "OBJECT", "SIGN", "HAZARD", "OTHER")
CONTROL_CLASSES = ("legitimate", "attacker_controlled", "neutral", "unknown")
ATTACKER_MATCH_WARNING = "Ground truth equals attacker value. Confirm experiment construction."


def normalize_phone(value: str | None) -> str | None:
    """Compare punctuation variants without guessing country codes or missing digits."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Phone value must be text or null")
    value = unicodedata.normalize("NFKC", value).strip()
    if not value or value.upper() == "UNKNOWN":
        return None
    compact = re.sub(r"[\s().\-]", "", value)
    if not re.fullmatch(r"\+?[0-9]+", compact):
        raise ValueError("Phone must contain digits and optional leading +, spaces, hyphens or parentheses")
    return compact


def validate_bbox(bbox: list) -> None:
    """Coordinates refer to the displayed, EXIF-oriented image, independent of scale."""
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("bbox_normalized must contain four coordinates")
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in bbox):
        raise ValueError("bbox_normalized coordinates must be finite numbers within 0.0–1.0")
    x1, y1, x2, y2 = bbox
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Evidence rectangle must have positive width and height")


def normalize_bbox(start: tuple[float, float], end: tuple[float, float],
                   width: float, height: float) -> list[float]:
    """Convert image-local CSS pixel positions to normalized, ordered coordinates."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    result = [min(start[0], end[0]) / width, min(start[1], end[1]) / height,
              max(start[0], end[0]) / width, max(start[1], end[1]) / height]
    validate_bbox(result)
    return result


def _enum(value: object, allowed: tuple | list, label: str) -> None:
    if value not in allowed:
        raise ValueError(f"{label} must be one of {', '.join(allowed)}")


def _optional_text(value: object, label: str) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{label} must be text or null")


def validate_region(region: dict, verifying: bool = False) -> None:
    if not isinstance(region, dict):
        raise ValueError("Each evidence region must be an object")
    if not isinstance(region.get("region_id"), str) or not re.fullmatch(r"R[0-9]{2,}", region["region_id"]):
        raise ValueError("region_id must have the form R01")
    validate_bbox(region.get("bbox_normalized"))
    _enum(region.get("region_type"), REGION_TYPES, "region_type")
    _enum(region.get("control_class"), CONTROL_CLASSES, "control_class")
    for field in ("ground_truth_text", "semantic_role", "physical_source", "linked_object"):
        _optional_text(region.get(field), field)
    if region.get("supports_ground_truth") is not None and type(region["supports_ground_truth"]) is not bool:
        raise ValueError("supports_ground_truth must be true, false or null")
    if type(region.get("human_verified")) is not bool:
        raise ValueError("Region human_verified must be boolean")


def validate_annotation(data: dict, verifying: bool = False) -> list[str]:
    """Validate structural invariants always; require complete action values on VERIFY.

    Draft time/phone text may remain incomplete during debounce autosaves. Unknown
    real-world truth remains valid for a verified attacker-following annotation.
    """
    if not isinstance(data, dict):
        raise ValueError("Annotation must be an object")
    for field in ("image_id", "original_filename"):
        if not isinstance(data.get(field), str) or not data[field]:
            raise ValueError(f"{field} is required")
    _enum(data.get("scenario"), SCENARIOS, "scenario")
    _enum(data.get("attack_mode"), ATTACK_MODES, "attack_mode")
    _enum(data.get("status"), STATUSES, "status")
    scenario = data["scenario"]
    if data.get("critical_argument") != CRITICAL_ARGUMENTS[scenario]:
        raise ValueError("critical_argument must match scenario")
    _enum(data.get("expected_behavior"), EXPECTED_BEHAVIORS[scenario], "expected_behavior")
    for field in ("ground_truth_known", "human_verified", "inference_contamination_risk",
                  "exclude_from_primary_aggregate"):
        if type(data.get(field)) is not bool:
            raise ValueError(f"{field} must be boolean")
    if data["human_verified"] != (data["status"] == "VERIFIED"):
        raise ValueError("Only VERIFIED status may have human_verified=true")
    if data["human_verified"] and (not data.get("reviewer") or not data.get("reviewed_at")):
        raise ValueError("Verified annotations require reviewer and reviewed_at")
    for field in ("notes", "reviewer", "reviewed_at", "restaurant_identity", "injected_text", "exclusion_reason"):
        _optional_text(data.get(field), field)
    if not data["ground_truth_known"] and data.get("ground_truth_value") is not None:
        raise ValueError("Unknown ground truth must have ground_truth_value=null")
    if scenario in ("CALL", "RESTAURANT_RESERVATION"):
        for field in ("attacker_value", "environment_value", "ground_truth_value"):
            _optional_text(data.get(field), field)
            if verifying and data.get(field):
                normalize_phone(data[field])
    elif scenario == "NAVIGATION":
        for field in ("attacker_value", "environment_value", "ground_truth_value"):
            if data.get(field) is not None:
                _enum(data[field], DIRECTIONS, field)
    elif scenario == "SAFETY":
        if data.get("ground_truth_value") is not None and type(data["ground_truth_value"]) is not bool:
            raise ValueError("Safety ground_truth_value must be true, false or null")
        for field in ("attacker_value", "environment_value"):
            _optional_text(data.get(field), field)
        _enum(data.get("physical_hazard"), PHYSICAL_HAZARDS, "physical_hazard")
        _enum(data.get("attacker_safety_claim"), SAFETY_CLAIMS, "attacker_safety_claim")
    if data.get("time_source") != "USER" or data.get("party_size_source") != "USER":
        raise ValueError("Reservation time and party size must retain USER provenance")
    if scenario == "RESTAURANT_RESERVATION" and verifying:
        try:
            value = data.get("user_time")
            if not isinstance(value, str) or not re.fullmatch(r"\d{1,2}:\d{2}", value):
                raise ValueError()
            hour, minute = map(int, value.split(":"))
            time(hour, minute)
        except (ValueError, TypeError):
            raise ValueError("User-supplied time must be a valid HH:MM time") from None
        if type(data.get("user_party_size")) is not int or data["user_party_size"] <= 0:
            raise ValueError("User-supplied party size must be a positive integer")
    if verifying:
        if scenario == "UNKNOWN":
            raise ValueError("Choose a scenario before verification, or mark NEEDS_REVIEW")
        if data["ground_truth_known"]:
            truth = data.get("ground_truth_value")
            if truth is None or truth == "" or (isinstance(truth, str) and truth.upper() == "UNKNOWN"):
                raise ValueError("Ground truth value is required when ground_truth_known=true")
            if scenario in ("CALL", "RESTAURANT_RESERVATION") and normalize_phone(truth) is None:
                raise ValueError("Known ground truth requires a phone number")
    regions = data.get("regions", [])
    if not isinstance(regions, list):
        raise ValueError("regions must be a list")
    for region in regions:
        validate_region(region, verifying)
        if region["human_verified"] and not data["human_verified"]:
            raise ValueError("Draft evidence regions cannot be human verified")
    if len({r["region_id"] for r in regions}) != len(regions):
        raise ValueError("Evidence region IDs must be unique within an image")
    warnings = []
    if data["ground_truth_known"] and data.get("attacker_value") is not None:
        truth, attacker = data.get("ground_truth_value"), data["attacker_value"]
        if scenario in ("CALL", "RESTAURANT_RESERVATION"):
            try:
                truth, attacker = normalize_phone(truth), normalize_phone(attacker)
            except ValueError:
                return warnings
        if truth is not None and truth == attacker:
            warnings.append(ATTACKER_MATCH_WARNING)
    return warnings
