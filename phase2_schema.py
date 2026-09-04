"""Strict schemas for the Phase 2 action/provenance experiment.

Phase 2 deliberately keeps model-estimated *visual source category* separate
from oracle trust metadata.  These models describe what a provider may emit;
they do not make an authenticity judgment or a firewall decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    computed_field,
    field_validator,
    model_validator,
)

from firewall.action_schema import ActionArguments, ActionType, ProposedAction


class Phase2Arm(StrEnum):
    ACTION_ONLY = "ACTION_ONLY"
    TWO_PASS_PROVENANCE = "TWO_PASS_PROVENANCE"
    INLINE_PROVENANCE = "INLINE_PROVENANCE"
    ORACLE_PROVENANCE = "ORACLE_PROVENANCE"


class Phase2Operation(StrEnum):
    ACTION_ONLY = "action_only"
    INLINE_PROVENANCE = "inline_provenance"
    TWO_PASS_EVIDENCE = "two_pass_evidence"


class SourceTypeEstimate(StrEnum):
    """Benchmark-compatible source estimates, not authenticity determinations."""

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


NormalizedCoordinate = Annotated[float, Field(ge=0.0, le=1.0)]


class NormalizedBBox(
    RootModel[
        tuple[
            NormalizedCoordinate,
            NormalizedCoordinate,
            NormalizedCoordinate,
            NormalizedCoordinate,
        ]
    ]
):
    """A normalized ``[x1, y1, x2, y2]`` box with a top-left origin."""

    @model_validator(mode="after")
    def validate_extent(self) -> NormalizedBBox:
        x1, y1, x2, y2 = self.root
        if x1 >= x2 or y1 >= y2:
            raise ValueError("bbox must satisfy x1 < x2 and y1 < y2")
        return self

    @property
    def x1(self) -> float:
        return self.root[0]

    @property
    def y1(self) -> float:
        return self.root[1]

    @property
    def x2(self) -> float:
        return self.root[2]

    @property
    def y2(self) -> float:
        return self.root[3]


class ArgumentEvidence(_StrictModel):
    evidence_text: str = Field(min_length=1)
    source_type_estimate: SourceTypeEstimate
    bbox: NormalizedBBox | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("evidence_text")
    @classmethod
    def strip_nonblank_evidence(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("evidence_text must not be blank")
        return cleaned


def _validated_action_fields(value: Any) -> dict[str, Any] | Any:
    if not isinstance(value, Mapping):
        return value
    action_payload = {
        "action": value.get("action"),
        "arguments": value.get("arguments", {}),
    }
    validated = ProposedAction.model_validate(action_payload)
    updated = dict(value)
    updated["action"] = validated.action
    updated["arguments"] = validated.arguments
    return updated


class ActionOnlyOutput(_StrictModel):
    """Minimal action response: no rationale and no provenance assertion."""

    action: ActionType
    arguments: ActionArguments

    @model_validator(mode="before")
    @classmethod
    def validate_action_arguments(cls, value: Any) -> dict[str, Any] | Any:
        return _validated_action_fields(value)

    def as_proposed_action(self) -> ProposedAction:
        return ProposedAction(action=self.action, arguments=self.arguments)

    def argument_values(self) -> dict[str, str]:
        return {
            key: str(value)
            for key, value in self.arguments.model_dump(mode="json", exclude_none=True).items()
        }


def _validate_evidence_keys(
    action: ActionOnlyOutput,
    argument_evidence: Mapping[str, list[ArgumentEvidence]],
) -> None:
    expected = set(action.argument_values())
    observed = set(argument_evidence)
    if observed != expected:
        raise ValueError(
            "argument_evidence keys must exactly match proposed argument keys; "
            f"expected {sorted(expected)}, got {sorted(observed)}"
        )


class InlineProvenanceOutput(_StrictModel):
    action: ActionType
    arguments: ActionArguments
    argument_evidence: dict[str, list[ArgumentEvidence]]

    @model_validator(mode="before")
    @classmethod
    def validate_action_arguments(cls, value: Any) -> dict[str, Any] | Any:
        return _validated_action_fields(value)

    @model_validator(mode="after")
    def validate_complete_evidence(self) -> InlineProvenanceOutput:
        _validate_evidence_keys(self.action_output(), self.argument_evidence)
        return self

    def action_output(self) -> ActionOnlyOutput:
        return ActionOnlyOutput(action=self.action, arguments=self.arguments)


class EvidenceOnlyOutput(_StrictModel):
    argument_evidence: dict[str, list[ArgumentEvidence]]


class Phase2TokenUsage(_StrictModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    thought_tokens: int | None = Field(default=None, ge=0)


class Phase2CallMetadata(_StrictModel):
    operation: Phase2Operation
    raw_response: str
    latency_ms: float = Field(ge=0.0)
    attempts: int = Field(ge=1)
    model: str | None = None
    token_usage: Phase2TokenUsage = Field(default_factory=Phase2TokenUsage)
    response_metadata: dict[str, Any] = Field(default_factory=dict)


class Phase2ArmResult(_StrictModel):
    """Arm output plus an auditable record for every model call it used."""

    arm: Phase2Arm
    action_output: ActionOnlyOutput
    argument_evidence: dict[str, list[ArgumentEvidence]] = Field(default_factory=dict)
    calls: list[Phase2CallMetadata]
    reused_action_only: bool = False

    @model_validator(mode="after")
    def validate_arm_shape(self) -> Phase2ArmResult:
        expected_calls = 2 if self.arm is Phase2Arm.TWO_PASS_PROVENANCE else 1
        if len(self.calls) != expected_calls:
            raise ValueError(f"{self.arm.value} requires {expected_calls} recorded model call(s)")

        operations = [call.operation for call in self.calls]
        expected_operations = {
            Phase2Arm.ACTION_ONLY: [Phase2Operation.ACTION_ONLY],
            Phase2Arm.INLINE_PROVENANCE: [Phase2Operation.INLINE_PROVENANCE],
            Phase2Arm.TWO_PASS_PROVENANCE: [
                Phase2Operation.ACTION_ONLY,
                Phase2Operation.TWO_PASS_EVIDENCE,
            ],
            Phase2Arm.ORACLE_PROVENANCE: [Phase2Operation.ACTION_ONLY],
        }[self.arm]
        if operations != expected_operations:
            raise ValueError(
                f"{self.arm.value} call operations must be "
                f"{[item.value for item in expected_operations]}"
            )

        if self.arm in {Phase2Arm.ACTION_ONLY, Phase2Arm.ORACLE_PROVENANCE}:
            if self.argument_evidence:
                raise ValueError(f"{self.arm.value} provider output must not contain evidence")
        else:
            _validate_evidence_keys(self.action_output, self.argument_evidence)

        if self.reused_action_only and self.arm is not Phase2Arm.ORACLE_PROVENANCE:
            raise ValueError("only the ORACLE arm may mark an action-only response as reused")
        return self

    @computed_field
    @property
    def call_count(self) -> int:
        return len(self.calls)

    @computed_field
    @property
    def total_attempts(self) -> int:
        return sum(call.attempts for call in self.calls)

    @computed_field
    @property
    def total_latency_ms(self) -> float:
        return sum(call.latency_ms for call in self.calls)

    @computed_field
    @property
    def aggregate_token_usage(self) -> Phase2TokenUsage:
        def total(field: str) -> int | None:
            values = [getattr(call.token_usage, field) for call in self.calls]
            if not values or any(value is None for value in values):
                return None
            return sum(value for value in values if value is not None)

        return Phase2TokenUsage(
            input_tokens=total("input_tokens"),
            output_tokens=total("output_tokens"),
            total_tokens=total("total_tokens"),
            cached_tokens=total("cached_tokens"),
            thought_tokens=total("thought_tokens"),
        )


def coerce_action_output(
    value: ActionOnlyOutput | ProposedAction | Mapping[str, Any],
) -> ActionOnlyOutput:
    if isinstance(value, ActionOnlyOutput):
        return value
    if isinstance(value, ProposedAction):
        return ActionOnlyOutput(action=value.action, arguments=value.arguments)
    return ActionOnlyOutput.model_validate(value)


def canonical_phase2_arm(value: Phase2Arm | str) -> Phase2Arm:
    """Parse canonical arm names plus the two concise CLI aliases."""

    if isinstance(value, Phase2Arm):
        return value
    normalized = str(value).strip().upper().replace("-", "_")
    normalized = {
        "TWO_PASS": Phase2Arm.TWO_PASS_PROVENANCE.value,
        "ORACLE": Phase2Arm.ORACLE_PROVENANCE.value,
    }.get(normalized, normalized)
    return Phase2Arm(normalized)


def validate_evidence_for_action(
    action: ActionOnlyOutput | ProposedAction | Mapping[str, Any],
    evidence: EvidenceOnlyOutput | Mapping[str, Any],
) -> EvidenceOnlyOutput:
    action_output = coerce_action_output(action)
    evidence_output = (
        evidence
        if isinstance(evidence, EvidenceOnlyOutput)
        else EvidenceOnlyOutput.model_validate(evidence)
    )
    _validate_evidence_keys(action_output, evidence_output.argument_evidence)
    return evidence_output


def token_usage_from_metadata(metadata: Mapping[str, Any] | None) -> Phase2TokenUsage:
    """Normalize Gemini or mock usage metadata without inventing missing counts."""

    raw: Any = (metadata or {}).get("usage")
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(mode="json", exclude_none=True)
    if not isinstance(raw, Mapping):
        return Phase2TokenUsage()

    def integer(*keys: str) -> int | None:
        for key in keys:
            value = raw.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None

    return Phase2TokenUsage(
        input_tokens=integer("total_input_tokens", "input_tokens", "totalInputTokens"),
        output_tokens=integer("total_output_tokens", "output_tokens", "totalOutputTokens"),
        total_tokens=integer("total_tokens", "totalTokens"),
        cached_tokens=integer("total_cached_tokens", "cached_tokens", "totalCachedTokens"),
        thought_tokens=integer("total_thought_tokens", "thought_tokens", "totalThoughtTokens"),
    )


def call_metadata_from_response(
    operation: Phase2Operation,
    response: Any,
) -> Phase2CallMetadata:
    """Convert a Phase 1 ``ProviderResponse`` into a serializable call record."""

    metadata = dict(response.response_metadata)
    return Phase2CallMetadata(
        operation=operation,
        raw_response=response.raw_response,
        latency_ms=response.latency_ms,
        attempts=response.attempts,
        model=response.model,
        token_usage=token_usage_from_metadata(metadata),
        response_metadata=metadata,
    )


def total_tokens(calls: Sequence[Phase2CallMetadata]) -> int | None:
    """Small accounting helper for callers that do not need an arm result."""

    values = [call.token_usage.total_tokens for call in calls]
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)
