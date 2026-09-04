"""Strict Phase 3.5 validation for model-selected evidence references.

The action model is allowed to select identifiers from an already constructed
registry.  It is not allowed to extend that registry, describe a replacement
region, or have an invalid identifier silently repaired.  This module therefore
validates the reference contract without performing fuzzy lookup or nearest-
region matching.

The validator intentionally accepts either the strict ``GroundedActionOutput``
model or an unvalidated mapping.  Accepting the latter lets experiment code
retain and classify malformed model responses instead of losing the reason a
contract failed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from phase3_5_constants import CRITICAL_ARGUMENTS


PHASE3_5_CRITICAL_ARGUMENTS: dict[str, tuple[str, ...]] = {
    action.value: tuple(arguments) for action, arguments in CRITICAL_ARGUMENTS.items()
}

# Camera references are ``<frame-id>:<region-id>``.  A Phase 2 region ID can
# itself contain a colon, so the suffix deliberately permits additional colons.
# User evidence has its own deliberately narrow namespace.
_CAMERA_EVIDENCE_ID = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*:[A-Za-z0-9][A-Za-z0-9._:-]*$"
)
_USER_EVIDENCE_ID = re.compile(r"^USER:[A-Za-z][A-Za-z0-9._-]*$")


class ReferenceIssueCode(StrEnum):
    """Inspectably distinct reasons the reference contract can be rejected."""

    MALFORMED_ACTION = "MALFORMED_ACTION"
    MALFORMED_ARGUMENTS = "MALFORMED_ARGUMENTS"
    MALFORMED_REFERENCE_MAP = "MALFORMED_REFERENCE_MAP"
    MISSING_ARGUMENT = "MISSING_ARGUMENT"
    MISSING_REFERENCES = "MISSING_REFERENCES"
    EXTRA_ARGUMENT_REFERENCES = "EXTRA_ARGUMENT_REFERENCES"
    MALFORMED_REFERENCE_ARRAY = "MALFORMED_REFERENCE_ARRAY"
    MALFORMED_REFERENCE_ID = "MALFORMED_REFERENCE_ID"
    DUPLICATE_REFERENCE = "DUPLICATE_REFERENCE"
    UNKNOWN_REFERENCE = "UNKNOWN_REFERENCE"
    CROSS_FRAME_REFERENCE = "CROSS_FRAME_REFERENCE"
    WRONG_REGISTRY_REFERENCE = "WRONG_REGISTRY_REFERENCE"
    MODEL_CONTRACT_INVALID = "MODEL_CONTRACT_INVALID"


class ArgumentReferenceStatus(StrEnum):
    VALID = "VALID"
    MISSING = "MISSING"
    INVALID_REFERENCE = "INVALID_REFERENCE"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReferenceIssue(_StrictModel):
    code: ReferenceIssueCode
    argument_name: str | None = None
    evidence_id: str | None = None
    message: str


class ArgumentReferenceValidation(_StrictModel):
    argument_name: str
    status: ArgumentReferenceStatus
    reference_ids: tuple[str, ...] = ()
    resolved_evidence_ids: tuple[str, ...] = ()
    issues: tuple[ReferenceIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return self.status is ArgumentReferenceStatus.VALID


class EvidenceReferenceValidation(_StrictModel):
    action: str
    frame_id: str | None
    expected_arguments: tuple[str, ...]
    argument_results: dict[str, ArgumentReferenceValidation] = Field(default_factory=dict)
    issues: tuple[ReferenceIssue, ...] = ()
    contract_valid: bool

    @property
    def valid(self) -> bool:
        return self.contract_valid

    def for_argument(self, argument_name: str) -> ArgumentReferenceValidation:
        return self.argument_results[argument_name]


def _model_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="python", exclude_none=True)
    return value


def _mapping(value: Any) -> dict[str, Any] | None:
    payload = _model_payload(value)
    return dict(payload) if isinstance(payload, Mapping) else None


def _enum_string(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).strip().upper()


def evidence_field(item: Any, name: str, default: Any = None) -> Any:
    """Read one field from a registry record without mutating or coercing it."""

    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def registry_frame_id(registry: Any) -> str | None:
    # EvidenceRegistry implements Mapping over *evidence IDs*, so ``get`` is
    # not the right way to retrieve its frame metadata.  Prefer the explicit
    # attribute and only then support a serialized top-level mapping.
    value = getattr(registry, "frame_id", None)
    if value is None and isinstance(registry, Mapping):
        value = registry.get("frame_id")
    if value is None:
        return None
    return str(value)


def registry_items(registry: Any) -> tuple[Any, ...]:
    """Return immutable registry records from supported registry representations."""

    # The canonical EvidenceRegistry is Mapping-like.  Check Mapping first so a
    # method named ``items`` is not mistaken for the tuple-valued model field.
    if isinstance(registry, Mapping):
        values = tuple(registry.values())
        # A serialized registry is a mapping with a top-level ``items`` field.
        if "items" in registry and not all(
            evidence_field(value, "evidence_id") is not None for value in values
        ):
            raw_items = registry.get("items")
            if isinstance(raw_items, Mapping):
                return tuple(raw_items.values())
            if isinstance(raw_items, Sequence) and not isinstance(raw_items, (str, bytes)):
                return tuple(raw_items)
        return values

    raw_items = getattr(registry, "items", None)
    if callable(raw_items):
        pairs = raw_items()
        return tuple(value for _, value in pairs)
    if isinstance(raw_items, Mapping):
        return tuple(raw_items.values())
    if isinstance(raw_items, Sequence) and not isinstance(raw_items, (str, bytes)):
        return tuple(raw_items)

    dumped = _mapping(registry)
    if dumped is not None:
        return registry_items(dumped)
    raise TypeError("registry must be an EvidenceRegistry or registry mapping")


def registry_get(registry: Any, evidence_id: str) -> Any | None:
    """Perform exact identifier lookup; never normalize or approximate an ID."""

    getter = getattr(registry, "get", None)
    if callable(getter):
        try:
            found = getter(evidence_id)
        except (KeyError, TypeError, ValueError):
            found = None
        if found is not None:
            return found

    for item in registry_items(registry):
        if evidence_field(item, "evidence_id") == evidence_id:
            return item
    return None


def is_well_formed_evidence_id(evidence_id: Any) -> bool:
    if not isinstance(evidence_id, str) or not evidence_id:
        return False
    if evidence_id.startswith("USER:"):
        return _USER_EVIDENCE_ID.fullmatch(evidence_id) is not None
    return _CAMERA_EVIDENCE_ID.fullmatch(evidence_id) is not None


def _reference_sequence(value: Any) -> tuple[tuple[str, ...], list[ReferenceIssue]]:
    if not isinstance(value, (list, tuple)):
        return (), [
            ReferenceIssue(
                code=ReferenceIssueCode.MALFORMED_REFERENCE_ARRAY,
                message="evidence references must be a JSON array of evidence IDs",
            )
        ]

    references: list[str] = []
    issues: list[ReferenceIssue] = []
    for raw_reference in value:
        if not isinstance(raw_reference, str) or not raw_reference:
            issues.append(
                ReferenceIssue(
                    code=ReferenceIssueCode.MALFORMED_REFERENCE_ID,
                    message="each evidence reference must be a non-empty string ID",
                )
            )
            continue
        references.append(raw_reference)
    return tuple(references), issues


def _with_argument(issue: ReferenceIssue, argument_name: str) -> ReferenceIssue:
    return issue.model_copy(update={"argument_name": argument_name})


def _validate_one_argument(
    argument_name: str,
    raw_references: Any,
    registry: Any,
    *,
    frame_id: str | None,
) -> ArgumentReferenceValidation:
    references, initial_issues = _reference_sequence(raw_references)
    issues = [_with_argument(issue, argument_name) for issue in initial_issues]
    resolved: list[str] = []

    if isinstance(raw_references, (list, tuple)) and not raw_references:
        issues.append(
            ReferenceIssue(
                code=ReferenceIssueCode.MISSING_REFERENCES,
                argument_name=argument_name,
                message="a populated argument requires at least one evidence ID",
            )
        )

    seen: set[str] = set()
    for evidence_id in references:
        if evidence_id in seen:
            issues.append(
                ReferenceIssue(
                    code=ReferenceIssueCode.DUPLICATE_REFERENCE,
                    argument_name=argument_name,
                    evidence_id=evidence_id,
                    message="duplicate evidence references are forbidden",
                )
            )
            continue
        seen.add(evidence_id)

        if not is_well_formed_evidence_id(evidence_id):
            issues.append(
                ReferenceIssue(
                    code=ReferenceIssueCode.MALFORMED_REFERENCE_ID,
                    argument_name=argument_name,
                    evidence_id=evidence_id,
                    message="reference is not a canonical camera or USER evidence ID",
                )
            )
            continue

        is_user_evidence = evidence_id.startswith("USER:")
        # A syntactically valid camera ID from a different frame is rejected
        # before lookup.  It must not degrade into a generic unknown-ID label.
        if not is_user_evidence and frame_id is not None and not evidence_id.startswith(
            f"{frame_id}:"
        ):
            issues.append(
                ReferenceIssue(
                    code=ReferenceIssueCode.CROSS_FRAME_REFERENCE,
                    argument_name=argument_name,
                    evidence_id=evidence_id,
                    message=f"camera evidence does not belong to frame {frame_id!r}",
                )
            )
            continue

        item = registry_get(registry, evidence_id)
        if item is None:
            issues.append(
                ReferenceIssue(
                    code=ReferenceIssueCode.UNKNOWN_REFERENCE,
                    argument_name=argument_name,
                    evidence_id=evidence_id,
                    message="evidence ID is absent from the supplied immutable registry",
                )
            )
            continue

        item_id = evidence_field(item, "evidence_id")
        if item_id != evidence_id:
            issues.append(
                ReferenceIssue(
                    code=ReferenceIssueCode.WRONG_REGISTRY_REFERENCE,
                    argument_name=argument_name,
                    evidence_id=evidence_id,
                    message="registry lookup did not return the exact referenced record",
                )
            )
            continue

        item_frame = evidence_field(item, "frame_id")
        if not is_user_evidence and frame_id is not None and item_frame != frame_id:
            issues.append(
                ReferenceIssue(
                    code=ReferenceIssueCode.CROSS_FRAME_REFERENCE,
                    argument_name=argument_name,
                    evidence_id=evidence_id,
                    message="referenced record has a different frame_id",
                )
            )
            continue
        if is_user_evidence and item_frame is not None:
            issues.append(
                ReferenceIssue(
                    code=ReferenceIssueCode.WRONG_REGISTRY_REFERENCE,
                    argument_name=argument_name,
                    evidence_id=evidence_id,
                    message="USER evidence must not masquerade as a camera-frame record",
                )
            )
            continue
        resolved.append(evidence_id)

    if any(issue.code is ReferenceIssueCode.MISSING_REFERENCES for issue in issues):
        status = ArgumentReferenceStatus.MISSING
    elif issues:
        status = ArgumentReferenceStatus.INVALID_REFERENCE
    else:
        status = ArgumentReferenceStatus.VALID

    return ArgumentReferenceValidation(
        argument_name=argument_name,
        status=status,
        reference_ids=references,
        resolved_evidence_ids=tuple(resolved),
        issues=tuple(issues),
    )


def validate_evidence_references(
    action_output: Any,
    registry: Any,
    *,
    frame_id: str | None = None,
    critical_arguments: Iterable[str] | None = None,
) -> EvidenceReferenceValidation:
    """Validate all references in a grounded action without semantic repair.

    Unknown IDs are never mapped to a nearby record.  A foreign-frame ID is
    never imported.  Malformed arrays and duplicate IDs remain explicit
    contract failures for audit and metrics.
    """

    payload = _mapping(action_output)
    issues: list[ReferenceIssue] = []
    if payload is None:
        payload = {}
        issues.append(
            ReferenceIssue(
                code=ReferenceIssueCode.MALFORMED_ACTION,
                message="grounded action output must be a mapping",
            )
        )

    # Reference validity is a subset of the grounded model contract. A payload
    # with forbidden extra fields, wrong argument types, malformed arrays, or
    # duplicate references must never reach ALLOW merely because its surviving
    # IDs resolve. Keep the detailed reference issues below as well.
    try:
        from phase3_5_schema import GroundedActionOutput

        GroundedActionOutput.model_validate(payload)
    except Exception as error:
        issues.append(
            ReferenceIssue(
                code=ReferenceIssueCode.MODEL_CONTRACT_INVALID,
                message=f"grounded action model contract is invalid: {error}",
            )
        )

    action = _enum_string(payload.get("action", ""))
    if action not in PHASE3_5_CRITICAL_ARGUMENTS:
        issues.append(
            ReferenceIssue(
                code=ReferenceIssueCode.MALFORMED_ACTION,
                message="action is absent or not registered for Phase 3.5",
            )
        )

    arguments = _mapping(payload.get("arguments"))
    if arguments is None:
        arguments = {}
        issues.append(
            ReferenceIssue(
                code=ReferenceIssueCode.MALFORMED_ARGUMENTS,
                message="arguments must be a mapping",
            )
        )

    references = _mapping(payload.get("argument_evidence_refs"))
    if references is None:
        references = {}
        issues.append(
            ReferenceIssue(
                code=ReferenceIssueCode.MALFORMED_REFERENCE_MAP,
                message="argument_evidence_refs must be a mapping",
            )
        )

    canonical_frame = registry_frame_id(registry)
    if frame_id is None:
        frame_id = canonical_frame
    elif canonical_frame is not None and frame_id != canonical_frame:
        issues.append(
            ReferenceIssue(
                code=ReferenceIssueCode.WRONG_REGISTRY_REFERENCE,
                message=(
                    f"requested frame {frame_id!r} does not match registry frame "
                    f"{canonical_frame!r}"
                ),
            )
        )

    registered_critical = tuple(
        critical_arguments
        if critical_arguments is not None
        else PHASE3_5_CRITICAL_ARGUMENTS.get(action, ())
    )
    # Every populated argument needs lineage, including future non-critical
    # extension arguments.  Registered critical arguments are mandatory even if
    # the malformed output omitted the argument itself.
    expected = tuple(dict.fromkeys((*registered_critical, *arguments.keys())))

    results: dict[str, ArgumentReferenceValidation] = {}
    for argument_name in expected:
        if argument_name not in arguments:
            issue = ReferenceIssue(
                code=ReferenceIssueCode.MISSING_ARGUMENT,
                argument_name=argument_name,
                message="registered critical argument is absent from the proposal",
            )
            issues.append(issue)
            results[argument_name] = ArgumentReferenceValidation(
                argument_name=argument_name,
                status=ArgumentReferenceStatus.MISSING,
                issues=(issue,),
            )
            continue
        if argument_name not in references:
            issue = ReferenceIssue(
                code=ReferenceIssueCode.MISSING_REFERENCES,
                argument_name=argument_name,
                message="argument has no argument_evidence_refs entry",
            )
            issues.append(issue)
            results[argument_name] = ArgumentReferenceValidation(
                argument_name=argument_name,
                status=ArgumentReferenceStatus.MISSING,
                issues=(issue,),
            )
            continue
        result = _validate_one_argument(
            argument_name,
            references[argument_name],
            registry,
            frame_id=frame_id,
        )
        results[argument_name] = result
        issues.extend(result.issues)

    for extra_name in references.keys() - set(expected):
        issue = ReferenceIssue(
            code=ReferenceIssueCode.EXTRA_ARGUMENT_REFERENCES,
            argument_name=str(extra_name),
            message="evidence references may not be supplied for an absent argument",
        )
        issues.append(issue)

    return EvidenceReferenceValidation(
        action=action,
        frame_id=frame_id,
        expected_arguments=expected,
        argument_results=results,
        issues=tuple(issues),
        contract_valid=not issues and all(result.valid for result in results.values()),
    )


class StrictReferenceValidator:
    """Reusable validator bound to one immutable registry and frame."""

    def __init__(self, registry: Any, *, frame_id: str | None = None) -> None:
        self._registry = registry
        self._frame_id = frame_id

    def validate(
        self,
        action_output: Any,
        *,
        critical_arguments: Iterable[str] | None = None,
    ) -> EvidenceReferenceValidation:
        return validate_evidence_references(
            action_output,
            self._registry,
            frame_id=self._frame_id,
            critical_arguments=critical_arguments,
        )


# An explicit alias makes call sites read naturally while keeping one canonical
# implementation and one set of failure semantics.
validate_reference_contract = validate_evidence_references
