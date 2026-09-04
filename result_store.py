"""Crash-safe-enough append persistence and resume helpers for benchmark trials."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence


IDENTITY_FIELDS = (
    "scenario_id",
    "run",
    "provider",
    "model",
    "prompt_version",
    "dataset_version",
    "policy_version",
    "registry_version",
    "selection_scope_id",
    "experiment_config_id",
    "provenance_mode",
)

COHORT_FIELDS = (
    "provider",
    "model",
    "predictor_model",
    "prompt_version",
    "dataset_version",
    "policy_version",
    "registry_version",
    "selection_scope_id",
    "experiment_config_id",
    "provenance_mode",
)

_ANALYSIS_DECISIONS = {"ALLOW", "WARN", "CONFIRM", "BLOCK"}
_ATTACK_CONDITIONS = {
    "BENIGN_UNTRUSTED_SUBSTITUTION",
    "AUTHORITY_IMPERSONATION",
    "OBVIOUS_INJECTION_CONTROL",
}
_COMPLETED_ANALYSIS_FIELDS = {
    "scenario_id",
    "action_family",
    "condition",
    "dataset_partition",
    "run",
    "user_prompt",
    "image_path",
    "ground_truth_action",
    "ground_truth_arguments",
    "proposed_action",
    "proposed_arguments",
    "critical_argument_source",
    "provenance",
    "consequence_prediction",
    "consequence_severity",
    "no_firewall_decision",
    "consequence_only_decision",
    "source_provenance_only_decision",
    "verified_conflict_only_decision",
    "full_firewall_decision",
    "attack_success",
    "action_extraction_correct",
    "critical_argument_extraction_correct",
    "attacker_controlled_influence",
    "security_relevant_influence",
    "security_relevant_influence_no_firewall",
    "security_relevant_influence_consequence_only",
    "security_relevant_influence_full_firewall",
    "policy_rules_triggered",
    "seed_pairing_key",
    "agent_request_seed",
    "predictor_request_seed",
    "latency_agent_ms",
    "latency_predictor_ms",
    "raw_agent_response_path",
    "raw_consequence_only_response_path",
    "raw_consequence_response_path",
    "agent_response_metadata",
    "consequence_only_response_metadata",
    "consequence_response_metadata",
    "timestamp",
}


def trial_identity(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in IDENTITY_FIELDS)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {source}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {source}:{line_number}")
            rows.append(value)
    return rows


def completed_identities(rows: Iterable[dict[str, Any]]) -> set[tuple[str, ...]]:
    return {
        trial_identity(row)
        for row in rows
        if row.get("status") == "completed"
    }


def final_trials_from_attempts(
    rows: Iterable[dict[str, Any]],
    *,
    identity_fields: Sequence[str] = IDENTITY_FIELDS,
) -> list[dict[str, Any]]:
    """Collapse append-only attempts into one scientific result per trial.

    Raw JSONL remains an attempt log.  A retry therefore never erases a quota
    error, malformed response, or other failure.  Scientific metrics, however,
    must not treat retry attempts as independent observations.  For each fully
    identified trial this function selects the last successful result when one
    exists, otherwise the last failed attempt.  Rows from older/unit-test data
    that lack a complete identity are conservatively kept as distinct records.
    """

    records = list(rows)
    grouped: dict[tuple[str, ...], list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(records):
        if all(row.get(field) not in (None, "") for field in identity_fields):
            identity = tuple(str(row[field]) for field in identity_fields)
        else:
            # Missing identity data cannot be safely deduplicated.
            identity = ("__unidentified_attempt__", str(index))
        grouped.setdefault(identity, []).append((index, row))

    selected: list[tuple[int, dict[str, Any]]] = []
    for attempts in grouped.values():
        successful = [item for item in attempts if item[1].get("status") == "completed"]
        selected.append(successful[-1] if successful else attempts[-1])
    selected.sort(key=lambda item: item[0])
    return [dict(row) for _, row in selected]


def attempt_accounting(
    rows: Iterable[dict[str, Any]],
    *,
    identity_fields: Sequence[str] = IDENTITY_FIELDS,
) -> dict[str, int]:
    """Report attempt history separately from deduplicated scientific trials."""

    attempts = list(rows)
    final = final_trials_from_attempts(attempts, identity_fields=identity_fields)
    identified_groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in attempts:
        if all(row.get(field) not in (None, "") for field in identity_fields):
            key = tuple(str(row[field]) for field in identity_fields)
            identified_groups.setdefault(key, []).append(row)
    superseded_errors = sum(
        sum(item.get("status") == "error" for item in group)
        for group in identified_groups.values()
        if any(item.get("status") == "completed" for item in group)
    )
    return {
        "raw_attempts": len(attempts),
        "unique_scientific_trials": len(final),
        "final_completed_trials": sum(row.get("status") == "completed" for row in final),
        "unresolved_error_trials": sum(row.get("status") == "error" for row in final),
        "failed_attempts": sum(row.get("status") == "error" for row in attempts),
        "superseded_error_attempts": superseded_errors,
    }


def validate_single_cohort(rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Return cohort metadata, refusing missing or mixed scientific identities.

    Analysis and reporting must never silently aggregate mock and Gemini rows, or
    trials produced with different models, prompts, datasets, policies, or
    provenance modes. Runs may differ; those are intended repetitions.
    """

    records = list(rows)
    if not records:
        raise ValueError("No result records were found; refusing to analyze an empty cohort.")

    cohort: dict[str, str] = {}
    for field in COHORT_FIELDS:
        if any(row.get(field) in (None, "") for row in records):
            raise ValueError(
                f"Result records are missing required cohort field {field!r}; "
                "refusing to aggregate them."
            )
        observed = {str(row[field]) for row in records}
        if len(observed) != 1:
            raise ValueError(
                f"Result records contain mixed {field} values {sorted(observed)}; "
                "analyze each cohort separately."
            )
        cohort[field] = observed.pop()
    return cohort


def validate_analysis_rows(rows: Iterable[dict[str, Any]]) -> None:
    """Refuse evidence rows whose missing fields could masquerade as defense."""

    for index, row in enumerate(rows, 1):
        status = row.get("status")
        if status not in {"completed", "error"}:
            raise ValueError(
                f"Result row {index} must explicitly declare status 'completed' or 'error'"
            )
        if status == "error":
            missing_error = {"error_type", "error_message"} - set(row)
            if missing_error:
                raise ValueError(
                    f"Error result row {index} is missing {sorted(missing_error)}"
                )
            continue

        missing = _COMPLETED_ANALYSIS_FIELDS - set(row)
        if missing:
            raise ValueError(
                f"Completed result row {index} is missing required evidence fields: "
                f"{sorted(missing)}"
            )
        for field in (
            "no_firewall_decision",
            "consequence_only_decision",
            "source_provenance_only_decision",
            "verified_conflict_only_decision",
            "full_firewall_decision",
        ):
            value = row[field]
            decision = value.get("decision") if isinstance(value, dict) else value
            if decision not in _ANALYSIS_DECISIONS:
                raise ValueError(
                    f"Completed result row {index} has invalid {field}: {decision!r}"
                )
        for field in (
            "attack_success",
            "action_extraction_correct",
            "critical_argument_extraction_correct",
            "attacker_controlled_influence",
            "security_relevant_influence",
            "security_relevant_influence_no_firewall",
            "security_relevant_influence_consequence_only",
            "security_relevant_influence_full_firewall",
        ):
            if not isinstance(row[field], bool):
                raise ValueError(
                    f"Completed result row {index} requires boolean {field}"
                )
        for field in (
            "ground_truth_arguments",
            "proposed_arguments",
            "provenance",
            "consequence_prediction",
            "agent_response_metadata",
            "consequence_only_response_metadata",
            "consequence_response_metadata",
        ):
            if not isinstance(row[field], dict):
                raise ValueError(f"Completed result row {index} requires object {field}")
        if not isinstance(row["policy_rules_triggered"], list):
            raise ValueError(
                f"Completed result row {index} requires list policy_rules_triggered"
            )
        for field in ("latency_agent_ms", "latency_predictor_ms"):
            value = row[field]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"Completed result row {index} has invalid nonnegative {field}"
                )
        for field in (
            "raw_agent_response_path",
            "raw_consequence_only_response_path",
            "raw_consequence_response_path",
        ):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f"Completed result row {index} has invalid {field}")
        if row["condition"] in _ATTACK_CONDITIONS and not row.get("attack_source"):
            raise ValueError(f"Completed attack row {index} is missing attack_source")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write a flattened CSV atomically; nested values remain lossless JSON strings."""

    records = list(rows)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in records for key in row})
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )
    temporary.replace(destination)


def assert_compatible_existing_run(
    rows: Iterable[dict[str, Any]],
    *,
    provider: str,
    provenance_mode: str,
    model: str | None = None,
    predictor_model: str | None = None,
    prompt_version: str | None = None,
    dataset_version: str | None = None,
    policy_version: str | None = None,
    registry_version: str | None = None,
    selection_scope_id: str | None = None,
    experiment_config_id: str | None = None,
) -> None:
    """Refuse to silently combine scientifically incompatible run records."""

    records = list(rows)
    expected = {
        "provider": provider,
        "provenance_mode": provenance_mode,
        "model": model,
        "predictor_model": predictor_model,
        "prompt_version": prompt_version,
        "dataset_version": dataset_version,
        "policy_version": policy_version,
        "registry_version": registry_version,
        "selection_scope_id": selection_scope_id,
        "experiment_config_id": experiment_config_id,
    }
    for field, expected_value in expected.items():
        if expected_value is None:
            continue
        if any(row.get(field) in (None, "") for row in records):
            raise ValueError(
                f"Results are missing required {field!r}; refusing to append. "
                "Choose a different --results-dir."
            )
        observed = {str(row[field]) for row in records}
        if observed and observed != {str(expected_value)}:
            raise ValueError(
                f"Results contain {field} values {sorted(observed)}; refusing to append "
                f"{expected_value!r}. Choose a different --results-dir."
            )
