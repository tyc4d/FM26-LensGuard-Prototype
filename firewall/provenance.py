"""Oracle provenance loading for the Phase 1 experiment.

This module does not infer provenance from pixels.  It reads trusted benchmark
metadata, keeping automatic/model-estimated attribution outside the primary
experiment.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .action_normalizer import (
    CRITICAL_ARGUMENTS,
    PRIMARY_CRITICAL_ARGUMENT,
    normalize_action,
    normalize_destination,
    normalize_direction,
    normalize_phone_number,
    normalize_url,
)
from .action_schema import ActionType, ProposedAction, ProvenanceSource


class OracleProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["ORACLE_PROVENANCE"] = "ORACLE_PROVENANCE"
    sources: dict[str, ProvenanceSource] = Field(default_factory=dict)


def _read_metadata(source: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load scenario metadata from {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("scenario metadata must be one JSON object")
    return payload


def _infer_action_type(
    metadata: Mapping[str, Any],
    action: ProposedAction | ActionType | str | Mapping[str, Any] | None,
) -> ActionType:
    if isinstance(action, ProposedAction):
        return action.action
    if isinstance(action, Mapping):
        return ProposedAction.model_validate(action).action
    if action is not None:
        return action if isinstance(action, ActionType) else ActionType(str(action).upper())

    candidate: Any = metadata.get("action_family") or metadata.get("expected_action")
    candidate = candidate or metadata.get("ground_truth_action")
    if isinstance(candidate, Mapping):
        candidate = candidate.get("action")
    if candidate is None:
        raise ValueError("action is required when metadata does not identify an action family")
    return candidate if isinstance(candidate, ActionType) else ActionType(str(candidate).upper())


def _actual_arguments(
    action: ProposedAction | ActionType | str | Mapping[str, Any] | None,
) -> dict[str, str]:
    if isinstance(action, ProposedAction):
        normalized = normalize_action(action)
    elif isinstance(action, Mapping) and "action" in action:
        normalized = normalize_action(action)
    else:
        return {}
    return {
        key: str(value)
        for key, value in normalized.arguments.model_dump(exclude_none=True).items()
    }


def _canonical_argument(action: ActionType, argument: str, value: Any) -> str:
    if not isinstance(value, str):
        return str(value)
    if action is ActionType.CALL and argument == "target_number":
        return normalize_phone_number(value)
    if action is ActionType.OPEN_URL and argument == "url":
        return normalize_url(value)
    if action is ActionType.DIRECTION_ADVICE and argument == "direction":
        return normalize_direction(value)
    if action is ActionType.DIRECTION_ADVICE and argument == "destination":
        return normalize_destination(value)
    return value


def _validate_source(source: Any) -> str:
    raw = source.value if isinstance(source, ProvenanceSource) else str(source)
    try:
        return ProvenanceSource(raw.strip().lower()).value
    except ValueError as exc:
        raise ValueError(f"unsupported provenance source: {source!r}") from exc


def _resolve_source_by_value(
    action_type: ActionType,
    argument: str,
    source_by_value: Mapping[str, Any],
    actual_value: str | None,
) -> str | None:
    if actual_value is None:
        return None
    canonical_actual = _canonical_argument(action_type, argument, actual_value)
    for candidate_value, source in source_by_value.items():
        try:
            canonical_candidate = _canonical_argument(
                action_type, argument, candidate_value
            )
        except (TypeError, ValueError):
            continue
        if canonical_actual == canonical_candidate:
            return _validate_source(source)
    return None


def load_oracle_provenance(
    scenario_metadata: Mapping[str, Any] | str | Path,
    action: ProposedAction | ActionType | str | Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Load argument-level provenance from trusted scenario ground truth.

    Supported metadata encodings are, in precedence order:

    * ``oracle_provenance`` / ``critical_argument_sources`` mappings;
    * ``argument_provenance`` / ``provenance_by_value`` value-to-source maps;
    * the compact ``critical_argument_source`` scalar used by Phase 1 scenes.

    Passing a proposed action enables value-specific attribution when a scenario
    records both official and attacker-controlled values.
    """

    metadata = _read_metadata(scenario_metadata)
    provenance_mode = metadata.get("provenance_mode")
    if provenance_mode is not None and str(provenance_mode).upper() not in {
        "ORACLE",
        "ORACLE_PROVENANCE",
        "ORACLE_PROVENANCE_MODE",
    }:
        raise ValueError(
            "load_oracle_provenance refuses non-oracle provenance metadata"
        )

    action_type = _infer_action_type(metadata, action)
    if action_type is ActionType.NONE:
        return {}
    critical_names = CRITICAL_ARGUMENTS[action_type]
    primary = PRIMARY_CRITICAL_ARGUMENT[action_type]
    actual = _actual_arguments(action)

    direct: Any = metadata.get("oracle_provenance")
    if direct is None:
        direct = metadata.get("critical_argument_sources")
    if direct is None:
        direct = metadata.get("critical_argument_provenance")
    if direct is None and provenance_mode is None:
        # ``provenance`` is accepted for concise scenario fixtures, unless the
        # record explicitly declares another provenance mode.
        direct = metadata.get("provenance")

    result: dict[str, str] = {}
    if isinstance(direct, Mapping):
        for argument in critical_names:
            if argument not in direct:
                continue
            source_spec = direct[argument]
            if isinstance(source_spec, Mapping):
                resolved = _resolve_source_by_value(
                    action_type, argument, source_spec, actual.get(argument)
                )
                if resolved is not None:
                    result[argument] = resolved
            else:
                result[argument] = _validate_source(source_spec)

    value_maps: Any = metadata.get("argument_provenance")
    if value_maps is None:
        value_maps = metadata.get("provenance_by_argument")
    if value_maps is None:
        value_maps = metadata.get("provenance_by_value")
    if isinstance(value_maps, Mapping):
        # The compact dataset form maps primary-argument values directly to
        # sources; the richer form is nested under each argument name.
        if primary is not None and primary not in value_maps:
            value_maps = {primary: value_maps}
        for argument in critical_names:
            mapping = value_maps.get(argument)
            if isinstance(mapping, Mapping):
                resolved = _resolve_source_by_value(
                    action_type, argument, mapping, actual.get(argument)
                )
                # A per-value oracle map is authoritative for that argument. An
                # arbitrary model error must not inherit the scene's compact
                # attacker-source label and be miscounted as attacker influence.
                if resolved is not None:
                    result[argument] = resolved
                elif argument in actual:
                    result[argument] = _validate_source(
                        metadata.get(
                            "unknown_value_source",
                            ProvenanceSource.UNKNOWN_VISUAL_SOURCE.value,
                        )
                    )

    compact_source = metadata.get("critical_argument_source")
    if primary is not None and primary not in result and compact_source is not None:
        if isinstance(compact_source, Mapping):
            if primary in compact_source:
                result[primary] = _validate_source(compact_source[primary])
        else:
            result[primary] = _validate_source(compact_source)

    # The condition itself is trusted benchmark metadata. This fallback makes an
    # explicit user-authority condition unambiguous even in compact fixtures.
    if (
        primary is not None
        and primary not in result
        and str(metadata.get("condition", "")).upper() == "EXPLICIT_USER_OVERRIDE"
    ):
        result[primary] = ProvenanceSource.EXPLICIT_USER.value

    if not result:
        raise ValueError(
            "scenario contains no oracle provenance for the action's critical arguments"
        )
    return result
