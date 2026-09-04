"""Strict uncertainty and user-escalation contracts for LensGuard Phase 3.6."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from phase3_5_constants import CRITICAL_ARGUMENTS, Phase35ActionType
from phase3_6_constants import ESCALATION_SCHEMA_VERSION, UNCERTAINTY_SCHEMA_VERSION


class UncertaintyStatus(StrEnum):
    """Categorical argument outcomes; never collapse these into one unsafe bit."""

    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    AMBIGUOUS = "AMBIGUOUS"
    CONFLICTING = "CONFLICTING"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    AUTHENTICITY_UNKNOWN = "AUTHENTICITY_UNKNOWN"
    MISSING = "MISSING"
    INVALID_REFERENCE = "INVALID_REFERENCE"


class AuthenticityStatus(StrEnum):
    """Whether authenticity was established, unresolved, or irrelevant.

    ``ESTABLISHED`` is a schema capability, not a visual inference rule.  A
    later deterministic assessor must provide an auditable basis before using
    it.  Good detection, OCR, alignment, or grounding alone are never enough.
    """

    ESTABLISHED = "ESTABLISHED"
    UNKNOWN = "UNKNOWN"
    NOT_REQUIRED = "NOT_REQUIRED"
    NOT_ASSESSED = "NOT_ASSESSED"


class EscalationReasonCode(StrEnum):
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    AUTHENTICITY_UNKNOWN = "AUTHENTICITY_UNKNOWN"
    LOW_PERCEPTION_CONFIDENCE = "LOW_PERCEPTION_CONFIDENCE"
    SAFETY_INVARIANT = "SAFETY_INVARIANT"


class UserEscalationOption(StrEnum):
    CONFIRM = "confirm"
    CANCEL = "cancel"
    VERIFY_INDEPENDENTLY = "verify_independently"


Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
CandidateValue = StrictStr | StrictBool | StrictInt


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceConfidenceDimensions(_FrozenStrictModel):
    """Independent confidence channels for one evidence item.

    There is intentionally no ``overall_confidence`` field and no aggregation
    rule.  Missing or uncalibrated values remain ``None``.
    """

    evidence_id: str = Field(min_length=1)
    detection_confidence: Confidence | None = None
    ocr_confidence: Confidence | None = None
    grounding_confidence: Confidence | None = None

    @field_validator("evidence_id")
    @classmethod
    def strip_evidence_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("evidence_id must not be blank")
        return cleaned


class ArgumentUncertaintyAssessment(_FrozenStrictModel):
    """Serializable argument-level uncertainty without a policy decision."""

    schema_version: Literal[UNCERTAINTY_SCHEMA_VERSION] = UNCERTAINTY_SCHEMA_VERSION
    argument: str = Field(min_length=1)
    # Keep malformed proposal values losslessly inspectable. Only normalized
    # candidate_values use the strict primitive contract below.
    argument_value: Any = None
    status: UncertaintyStatus
    authenticity_status: AuthenticityStatus
    authenticity_basis: str | None = None
    candidate_values: tuple[CandidateValue, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    evidence_confidences: tuple[EvidenceConfidenceDimensions, ...] = ()
    grounding_confidence: Confidence | None = None
    reasons: tuple[str, ...] = ()

    @field_validator("argument")
    @classmethod
    def strip_argument(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("argument must not be blank")
        return cleaned

    @field_validator("authenticity_basis")
    @classmethod
    def strip_authenticity_basis(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("authenticity_basis must not be blank")
        return cleaned

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(value.strip() for value in values)
        if any(not value for value in cleaned):
            raise ValueError("evidence_ids must not contain blank values")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("evidence_ids must be unique")
        return cleaned

    @field_validator("reasons")
    @classmethod
    def validate_reasons(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(value.strip() for value in values)
        if any(not value for value in cleaned):
            raise ValueError("reasons must not contain blank values")
        return cleaned

    @model_validator(mode="after")
    def validate_authenticity_state(self) -> ArgumentUncertaintyAssessment:
        if (
            self.status is UncertaintyStatus.AUTHENTICITY_UNKNOWN
            and self.authenticity_status is not AuthenticityStatus.UNKNOWN
        ):
            raise ValueError("AUTHENTICITY_UNKNOWN requires authenticity_status=UNKNOWN")
        if (
            self.status is UncertaintyStatus.SUPPORTED
            and self.authenticity_status is AuthenticityStatus.UNKNOWN
        ):
            raise ValueError("SUPPORTED cannot retain unresolved authenticity")
        if self.status is UncertaintyStatus.AUTHENTICITY_UNKNOWN and (
            not self.candidate_values or not self.evidence_ids
        ):
            raise ValueError(
                "AUTHENTICITY_UNKNOWN requires candidate values and supporting evidence"
            )
        if (
            self.status is UncertaintyStatus.CONFLICTING
            and len(self.candidate_values) < 2
        ):
            raise ValueError("CONFLICTING requires at least two distinct candidates")
        if self.status is UncertaintyStatus.CONFLICTING and not self.evidence_ids:
            raise ValueError("CONFLICTING requires supporting evidence IDs")
        if (
            self.authenticity_status is AuthenticityStatus.ESTABLISHED
            and self.authenticity_basis is None
        ):
            raise ValueError("ESTABLISHED authenticity requires an auditable basis")
        if (
            self.authenticity_status is not AuthenticityStatus.ESTABLISHED
            and self.authenticity_basis is not None
        ):
            raise ValueError("authenticity_basis is only valid when authenticity is ESTABLISHED")
        confidence_ids = [item.evidence_id for item in self.evidence_confidences]
        if len(confidence_ids) != len(set(confidence_ids)):
            raise ValueError("evidence confidence entries must have unique evidence IDs")
        if set(confidence_ids) - set(self.evidence_ids):
            raise ValueError("confidence entries must refer to declared evidence_ids")
        typed_candidates = {(type(value), repr(value)) for value in self.candidate_values}
        if len(typed_candidates) != len(self.candidate_values):
            raise ValueError("candidate_values must contain distinct normalized values")
        return self


class UncertaintyAssessmentReport(_FrozenStrictModel):
    """Schema-level container for per-argument results; it makes no gate decision."""

    schema_version: Literal[UNCERTAINTY_SCHEMA_VERSION] = UNCERTAINTY_SCHEMA_VERSION
    action: Phase35ActionType
    argument_assessments: dict[str, ArgumentUncertaintyAssessment]

    @model_validator(mode="after")
    def validate_argument_map(self) -> UncertaintyAssessmentReport:
        expected = set(CRITICAL_ARGUMENTS[self.action])
        observed = set(self.argument_assessments)
        if observed != expected:
            raise ValueError(
                "argument_assessments keys must exactly match critical arguments; "
                f"expected {sorted(expected)}, got {sorted(observed)}"
            )
        for key, assessment in self.argument_assessments.items():
            if assessment.argument != key:
                raise ValueError("argument assessment keys must match embedded argument names")
        return self


class StructuredEscalation(_FrozenStrictModel):
    """Machine-readable handoff at the Phase 3.6 user-confirmation boundary."""

    schema_version: Literal[ESCALATION_SCHEMA_VERSION] = ESCALATION_SCHEMA_VERSION
    decision: Literal["ESCALATE"] = "ESCALATE"
    reason_code: EscalationReasonCode
    action: Phase35ActionType
    argument: str = Field(min_length=1)
    candidate_values: tuple[CandidateValue, ...] = ()
    message: str = Field(min_length=1)
    user_options: tuple[UserEscalationOption, ...] = (
        UserEscalationOption.CONFIRM,
        UserEscalationOption.CANCEL,
        UserEscalationOption.VERIFY_INDEPENDENTLY,
    )

    @field_validator("argument", "message")
    @classmethod
    def strip_nonblank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("escalation string fields must not be blank")
        return cleaned

    @field_validator("candidate_values")
    @classmethod
    def strip_candidate_strings(
        cls, values: tuple[CandidateValue, ...]
    ) -> tuple[CandidateValue, ...]:
        cleaned: list[CandidateValue] = []
        for value in values:
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    raise ValueError("candidate_values must not contain blank strings")
                cleaned.append(stripped)
            else:
                cleaned.append(value)
        return tuple(cleaned)

    @model_validator(mode="after")
    def validate_reason_payload(self) -> StructuredEscalation:
        typed_values = {(type(value), repr(value)) for value in self.candidate_values}
        if self.reason_code is EscalationReasonCode.CONFLICTING_EVIDENCE:
            if len(typed_values) < 2:
                raise ValueError("CONFLICTING_EVIDENCE requires two distinct candidates")
        if self.reason_code is EscalationReasonCode.AUTHENTICITY_UNKNOWN:
            if not self.candidate_values:
                raise ValueError("AUTHENTICITY_UNKNOWN requires a visible candidate")
        if len(self.user_options) != len(set(self.user_options)):
            raise ValueError("user_options must be unique")
        if UserEscalationOption.CANCEL not in self.user_options:
            raise ValueError("user_options must always include cancel")
        if self.argument not in CRITICAL_ARGUMENTS[self.action]:
            raise ValueError(
                f"argument {self.argument!r} is not valid for action {self.action.value}"
            )
        return self


def dump_jsonable(model: BaseModel) -> dict[str, Any]:
    """Return a JSON-mode payload for fixtures, storage, and user interfaces."""

    return model.model_dump(mode="json")
