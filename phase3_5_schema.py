"""Strict action-model contracts for LensGuard Phase 3.5.

The grounded contract lets the model propose an action and select existing
evidence IDs for each individual argument.  It contains no trust, authority,
maliciousness, policy, confidence, rationale, or bounding-box output fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from phase3_5_constants import CRITICAL_ARGUMENTS, Phase35ActionType
from provenance.evidence_registry_phase3_5 import is_well_formed_evidence_id


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _NonblankArguments(_StrictModel):
    @field_validator("*", mode="before")
    @classmethod
    def reject_blank_strings(cls, value: Any) -> Any:
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                raise ValueError("argument values must not be blank")
            return cleaned
        return value


class CallArguments(_NonblankArguments):
    target_number: str = Field(min_length=1)


class OpenURLArguments(_NonblankArguments):
    url: str = Field(min_length=1)


class DirectionAdviceArguments(_NonblankArguments):
    direction: str = Field(min_length=1)
    destination: str = Field(min_length=1)


class SafetyAdviceArguments(_NonblankArguments):
    safe_to_proceed: bool = Field(strict=True)
    hazard: str = Field(min_length=1)


class RestaurantReservationArguments(_NonblankArguments):
    restaurant: str = Field(min_length=1)
    target_number: str = Field(min_length=1)
    time: str = Field(min_length=1)
    party_size: int = Field(ge=1, strict=True)


class EmptyArguments(_StrictModel):
    pass


Phase35Arguments = (
    CallArguments
    | OpenURLArguments
    | DirectionAdviceArguments
    | SafetyAdviceArguments
    | RestaurantReservationArguments
    | EmptyArguments
)

_ARGUMENT_MODEL_BY_ACTION: dict[Phase35ActionType, type[_StrictModel]] = {
    Phase35ActionType.CALL: CallArguments,
    Phase35ActionType.OPEN_URL: OpenURLArguments,
    Phase35ActionType.DIRECTION_ADVICE: DirectionAdviceArguments,
    Phase35ActionType.SAFETY_ADVICE: SafetyAdviceArguments,
    Phase35ActionType.RESTAURANT_RESERVATION: RestaurantReservationArguments,
    Phase35ActionType.NONE: EmptyArguments,
}


def _validate_action_arguments(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    raw_action = value.get("action")
    try:
        action = (
            raw_action
            if isinstance(raw_action, Phase35ActionType)
            else Phase35ActionType(raw_action)
        )
    except (TypeError, ValueError):
        return value

    argument_model = _ARGUMENT_MODEL_BY_ACTION[action]
    raw_arguments = value.get("arguments", {})
    if not isinstance(raw_arguments, argument_model):
        raw_arguments = argument_model.model_validate(raw_arguments)
    validated = dict(value)
    validated["action"] = action
    validated["arguments"] = raw_arguments
    return validated


class Phase35ActionOutput(_StrictModel):
    """Action-only comparison arm: exactly action plus typed arguments."""

    action: Phase35ActionType
    arguments: Phase35Arguments

    @model_validator(mode="before")
    @classmethod
    def validate_action_specific_arguments(cls, value: Any) -> Any:
        return _validate_action_arguments(value)

    @model_validator(mode="after")
    def validate_critical_argument_shape(self) -> Phase35ActionOutput:
        expected_model = _ARGUMENT_MODEL_BY_ACTION[self.action]
        if not isinstance(self.arguments, expected_model):
            raise ValueError(f"{self.action.value} received the wrong argument schema")
        observed = tuple(self.argument_values())
        expected = CRITICAL_ARGUMENTS[self.action]
        if observed != expected:
            raise ValueError(
                f"{self.action.value} arguments must be exactly {list(expected)!r}"
            )
        return self

    def argument_values(self) -> dict[str, str | bool | int]:
        return self.arguments.model_dump(mode="python", exclude_none=True)

    @property
    def critical_argument_names(self) -> tuple[str, ...]:
        return CRITICAL_ARGUMENTS[self.action]


class GroundedActionOutput(_StrictModel):
    """Action, arguments, and exact references into a pre-built registry."""

    action: Phase35ActionType
    arguments: Phase35Arguments
    argument_evidence_refs: dict[str, tuple[str, ...]]

    @model_validator(mode="before")
    @classmethod
    def validate_action_and_reference_container(cls, value: Any) -> Any:
        validated = _validate_action_arguments(value)
        if not isinstance(validated, Mapping):
            return validated
        raw_refs = validated.get("argument_evidence_refs")
        if not isinstance(raw_refs, Mapping):
            raise ValueError("argument_evidence_refs must be a JSON object")
        normalized_refs: dict[str, tuple[str, ...]] = {}
        for argument_name, raw_argument_refs in raw_refs.items():
            if not isinstance(argument_name, str) or not argument_name.strip():
                raise ValueError("argument_evidence_refs keys must be nonblank strings")
            if not isinstance(raw_argument_refs, (list, tuple)):
                raise ValueError(
                    f"evidence refs for {argument_name!r} must be a JSON array"
                )
            normalized_refs[argument_name] = tuple(raw_argument_refs)
        result = dict(validated)
        result["argument_evidence_refs"] = normalized_refs
        return result

    @field_validator("argument_evidence_refs")
    @classmethod
    def validate_reference_ids(
        cls, references: dict[str, tuple[str, ...]]
    ) -> dict[str, tuple[str, ...]]:
        for argument_name, evidence_ids in references.items():
            if not evidence_ids:
                raise ValueError(
                    f"argument {argument_name!r} requires at least one evidence reference"
                )
            if len(set(evidence_ids)) != len(evidence_ids):
                raise ValueError(
                    f"argument {argument_name!r} contains duplicate evidence references"
                )
            for evidence_id in evidence_ids:
                if not is_well_formed_evidence_id(evidence_id):
                    raise ValueError(f"malformed evidence ID {evidence_id!r}")
        return references

    @model_validator(mode="after")
    def validate_action_specific_contract(self) -> GroundedActionOutput:
        action_output = self.action_output()
        expected = set(action_output.argument_values())
        observed = set(self.argument_evidence_refs)
        if observed != expected:
            raise ValueError(
                "argument_evidence_refs keys must exactly match argument keys; "
                f"expected {sorted(expected)}, got {sorted(observed)}"
            )
        return self

    def action_output(self) -> Phase35ActionOutput:
        return Phase35ActionOutput(action=self.action, arguments=self.arguments)

    def argument_values(self) -> dict[str, str | bool | int]:
        return self.action_output().argument_values()

    @property
    def critical_argument_names(self) -> tuple[str, ...]:
        return CRITICAL_ARGUMENTS[self.action]


# Readable alias used by some reports for the GROUNDED_REGISTRY arm.
GroundedRegistryOutput = GroundedActionOutput
