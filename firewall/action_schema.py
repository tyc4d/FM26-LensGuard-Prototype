"""Strict, side-effect-free schemas used at the LensGuard action boundary.

The multimodal model is never allowed to return an arbitrary tool invocation.  Its
output must validate as :class:`ProposedAction` before the rest of the dry-run
pipeline sees it.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ActionType(StrEnum):
    CALL = "CALL"
    OPEN_URL = "OPEN_URL"
    DIRECTION_ADVICE = "DIRECTION_ADVICE"
    NONE = "NONE"


class Decision(StrEnum):
    ALLOW = "ALLOW"
    WARN = "WARN"
    CONFIRM = "CONFIRM"
    BLOCK = "BLOCK"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Reversibility(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ProvenanceSource(StrEnum):
    """Source labels supported by the Phase 1 oracle and optional estimator."""

    EXPLICIT_USER = "explicit_user"
    VERIFIED_CONTACTS = "verified_contacts"
    VERIFIED_APPLICATION_DATA = "verified_application_data"
    VERIFIED_NAVIGATION_DATA = "verified_navigation_data"
    OFFICIAL_SIGNAGE = "official_signage"
    CAMERA_UNVERIFIED = "camera_unverified"
    QR_CODE_UNVERIFIED = "qr_code_unverified"
    ADVERTISEMENT = "advertisement"
    HANDWRITTEN_NOTE = "handwritten_note"
    UNVERIFIED_NOTICE = "unverified_notice"
    UNKNOWN_VISUAL_SOURCE = "unknown_visual_source"
    UNKNOWN = "unknown"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class CallArguments(_StrictModel):
    target_number: str = Field(min_length=1)


class OpenURLArguments(_StrictModel):
    url: str = Field(min_length=1)


class DirectionAdviceArguments(_StrictModel):
    direction: str = Field(min_length=1)
    # A model can know the requested destination from the user prompt while only
    # seeing a directional arrow in the image.  Keeping this optional lets the
    # action remain representable when that context is genuinely absent.
    destination: str | None = None


class EmptyArguments(_StrictModel):
    """Arguments for NONE. Extra keys are rejected by ``_StrictModel``."""


ActionArguments = (
    CallArguments | OpenURLArguments | DirectionAdviceArguments | EmptyArguments
)


_ARGUMENT_MODELS: dict[ActionType, type[_StrictModel]] = {
    ActionType.CALL: CallArguments,
    ActionType.OPEN_URL: OpenURLArguments,
    ActionType.DIRECTION_ADVICE: DirectionAdviceArguments,
    ActionType.NONE: EmptyArguments,
}


class ProposedAction(_StrictModel):
    action: ActionType
    arguments: ActionArguments
    reason_summary: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def validate_action_specific_arguments(cls, value: Any) -> Any:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            return value

        raw_action = value.get("action")
        try:
            action = raw_action if isinstance(raw_action, ActionType) else ActionType(raw_action)
        except (TypeError, ValueError):
            # Let Pydantic produce the canonical enum validation error.
            return value

        raw_arguments = value.get("arguments", {})
        argument_model = _ARGUMENT_MODELS[action]
        if not isinstance(raw_arguments, argument_model):
            raw_arguments = argument_model.model_validate(raw_arguments)

        validated = dict(value)
        validated["action"] = action
        validated["arguments"] = raw_arguments
        return validated


class ConsequencePrediction(_StrictModel):
    effects: list[str] = Field(default_factory=list)
    severity: Severity
    reversibility: Reversibility
    uncertainties: list[str] = Field(default_factory=list)

    @field_validator("effects", "uncertainties")
    @classmethod
    def reject_blank_list_items(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            item = value.strip()
            if not item:
                raise ValueError("list entries must not be blank")
            cleaned.append(item)
        return cleaned


class FirewallDecision(_StrictModel):
    decision: Decision
    action: ActionType
    critical_arguments: dict[str, str] = Field(default_factory=dict)
    provenance: dict[str, str] = Field(default_factory=dict)
    predicted_consequences: list[str] = Field(default_factory=list)
    policy_rules_triggered: list[str] = Field(default_factory=list)
    user_message: str

    @field_validator("critical_arguments", "provenance")
    @classmethod
    def reject_blank_mapping_values(cls, values: dict[str, str]) -> dict[str, str]:
        for key, value in values.items():
            if not key.strip() or not value.strip():
                raise ValueError("mapping keys and values must not be blank")
        return values
