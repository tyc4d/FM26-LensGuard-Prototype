"""Append-only persistence and cohort guards for Phase 3.5."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from result_store import append_jsonl, read_jsonl, write_csv


PHASE3_5_MODEL_ARTIFACT_NAMES = (
    "raw_generations.jsonl",
    "model_call_records.jsonl",
    "final_trials.csv",
    "analysis.json",
    "report.md",
    "system_info.json",
)

_STREAM_ARTIFACT_NAMES = PHASE3_5_MODEL_ARTIFACT_NAMES[:3]
_SUMMARY_ARTIFACT_NAMES = PHASE3_5_MODEL_ARTIFACT_NAMES[3:]


PHASE3_5_IDENTITY_FIELDS = (
    "experiment_version",
    "scene_id",
    "architecture_arm",
    "run",
    "provider",
    "model_alias",
    "model_revision",
    "prompt_version",
    "dataset_version",
    "evidence_schema_version",
    "model_contract_version",
    "policy_version",
    "action_registry_version",
    "selection_scope_id",
)

PHASE3_5_COHORT_FIELDS = (
    "experiment_version",
    "provider",
    "model_alias",
    "model_id",
    "model_revision",
    "dataset_version",
    "evidence_schema_version",
    "model_contract_version",
    "policy_version",
    "action_registry_version",
    "selection_scope_id",
    "selected_case_count",
    "planned_trial_count",
    "perception_profile",
)


def phase3_5_trial_identity(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in PHASE3_5_IDENTITY_FIELDS)


def validate_phase3_5_rows(rows: Iterable[Mapping[str, Any]]) -> None:
    seen: set[tuple[str, ...]] = set()
    for index, row in enumerate(rows, 1):
        missing = [field for field in PHASE3_5_IDENTITY_FIELDS if row.get(field) in (None, "")]
        if missing:
            raise ValueError(f"Phase 3.5 row {index} is missing identity fields {missing}")
        identity = phase3_5_trial_identity(row)
        if identity in seen:
            raise ValueError(f"Phase 3.5 duplicate scientific identity at row {index}: {identity}")
        seen.add(identity)
        if type(row.get("attempt_index")) is not int or row.get("attempt_index") != 1:
            raise ValueError(f"Phase 3.5 row {index} must be the single scientific attempt")
        if row.get("status") not in {"completed", "error"}:
            raise ValueError(f"Phase 3.5 row {index} has invalid status {row.get('status')!r}")
        if row.get("status") == "error" and not row.get("error_type"):
            raise ValueError(f"Phase 3.5 error row {index} is missing error_type")


def phase3_5_artifact_paths(results_dir: str | Path) -> dict[str, Path]:
    """Return the canonical, version-specific per-model artifact paths."""

    destination = Path(results_dir)
    return {
        name: destination / name for name in PHASE3_5_MODEL_ARTIFACT_NAMES
    }


def validate_phase3_5_artifact_set(results_dir: str | Path) -> dict[str, Path]:
    """Require all six per-model artifacts, without accepting aliases.

    Extra diagnostic files are allowed, but none of the six scientific
    artifacts may be renamed, omitted, or represented only by an alternative
    file. JSON/JSONL contents receive a shallow readability check here; the
    row/cohort validators remain authoritative for their semantic contracts.
    """

    paths = phase3_5_artifact_paths(results_dir)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"Phase 3.5 result directory is missing artifacts {missing}")

    read_jsonl(paths["raw_generations.jsonl"])
    read_jsonl(paths["model_call_records.jsonl"])
    for name in ("analysis.json", "system_info.json"):
        try:
            value = json.loads(paths[name].read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON artifact {paths[name]}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Phase 3.5 artifact {paths[name]} must contain a JSON object")
    return paths


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_phase3_5_summary_artifacts(
    results_dir: str | Path,
    *,
    analysis: Mapping[str, Any],
    report: str,
    system_info: Mapping[str, Any],
) -> dict[str, Path]:
    """Write ``analysis.json``, ``report.md``, and ``system_info.json``.

    The three append/trial artifacts must already exist. This makes finalizing
    an empty or partially named result directory an explicit error and ensures
    that a successful return means the exact six requested artifacts exist.
    """

    if not isinstance(analysis, Mapping):
        raise TypeError("analysis must be a mapping")
    if not isinstance(system_info, Mapping):
        raise TypeError("system_info must be a mapping")
    if not isinstance(report, str) or not report.strip():
        raise ValueError("report must be a non-empty Markdown string")

    paths = phase3_5_artifact_paths(results_dir)
    missing_streams = [name for name in _STREAM_ARTIFACT_NAMES if not paths[name].is_file()]
    if missing_streams:
        raise ValueError(
            "Cannot finalize Phase 3.5 results before trial artifacts exist: "
            f"{missing_streams}"
        )

    _write_text_atomic(
        paths["analysis.json"],
        json.dumps(dict(analysis), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(paths["report.md"], report.rstrip() + "\n")
    _write_text_atomic(
        paths["system_info.json"],
        json.dumps(dict(system_info), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return validate_phase3_5_artifact_set(results_dir)


def validate_phase3_5_cohort(
    rows: Iterable[Mapping[str, Any]],
    *,
    allowed_arms: Sequence[str] | None = None,
) -> dict[str, Any]:
    records = list(rows)
    if not records:
        raise ValueError("No Phase 3.5 records were found")
    validate_phase3_5_rows(records)
    cohort: dict[str, Any] = {}
    for field in PHASE3_5_COHORT_FIELDS:
        values = {str(row.get(field, "")) for row in records}
        if "" in values or len(values) != 1:
            raise ValueError(f"Phase 3.5 cohort has missing or mixed {field}: {sorted(values)}")
        cohort[field] = records[0][field]
    if allowed_arms is not None:
        unexpected = {str(row["architecture_arm"]) for row in records} - set(allowed_arms)
        if unexpected:
            raise ValueError(f"Phase 3.5 cohort contains unexpected arms: {sorted(unexpected)}")
    return cohort


def assert_phase3_5_resume_compatible(
    rows: Iterable[Mapping[str, Any]],
    expected: Mapping[str, Any],
) -> None:
    records = list(rows)
    for field in PHASE3_5_COHORT_FIELDS:
        if field not in expected:
            continue
        observed = {str(row.get(field, "")) for row in records}
        if "" in observed:
            raise ValueError(f"Existing Phase 3.5 records are missing {field}; refusing resume")
        if observed and observed != {str(expected[field])}:
            raise ValueError(
                f"Existing Phase 3.5 records contain incompatible {field} values "
                f"{sorted(observed)}; expected {expected[field]!r}"
            )


def persist_phase3_5_trial(
    results_dir: str | Path,
    row: dict[str, Any],
    call_record: dict[str, Any],
) -> None:
    """Append one trial and its one model call, then refresh the lossless CSV."""

    destination = Path(results_dir)
    raw_path = destination / "raw_generations.jsonl"
    calls_path = destination / "model_call_records.jsonl"
    final_path = destination / "final_trials.csv"
    identity = phase3_5_trial_identity(row)
    existing = read_jsonl(raw_path)
    existing_calls = read_jsonl(calls_path)
    if len(existing) != len(existing_calls):
        raise ValueError(
            "Existing Phase 3.5 raw-generation and model-call logs are not one-to-one; "
            "refusing to append"
        )
    if identity in {phase3_5_trial_identity(item) for item in existing}:
        raise ValueError(f"Refusing a second scientific attempt for {identity}")
    validate_phase3_5_rows([row])
    if call_record.get("trial_identity") != list(identity):
        raise ValueError("model call record is not bound to the trial identity")
    existing_call_identities = {
        tuple(str(value) for value in item.get("trial_identity", ()))
        for item in existing_calls
        if isinstance(item.get("trial_identity"), (list, tuple))
    }
    if identity in existing_call_identities:
        raise ValueError(f"Refusing a second model call for {identity}")

    # Fail before either append for values that cannot be represented in JSON.
    json.dumps(row, ensure_ascii=False, sort_keys=True)
    json.dumps(call_record, ensure_ascii=False, sort_keys=True)
    append_jsonl(raw_path, row)
    append_jsonl(calls_path, call_record)
    refreshed = read_jsonl(raw_path)
    refreshed_calls = read_jsonl(calls_path)
    validate_phase3_5_rows(refreshed)
    if len(refreshed) != len(refreshed_calls):
        raise ValueError("Phase 3.5 persistence did not produce one call record per trial")
    write_csv(final_path, refreshed)


__all__ = [
    "PHASE3_5_COHORT_FIELDS",
    "PHASE3_5_IDENTITY_FIELDS",
    "PHASE3_5_MODEL_ARTIFACT_NAMES",
    "assert_phase3_5_resume_compatible",
    "persist_phase3_5_trial",
    "phase3_5_artifact_paths",
    "phase3_5_trial_identity",
    "validate_phase3_5_artifact_set",
    "validate_phase3_5_cohort",
    "validate_phase3_5_rows",
    "write_phase3_5_summary_artifacts",
]
