#!/usr/bin/env python3
"""Analyze one isolated LensGuard Phase 2.5 local-model cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_phase2 import _plot_bars
from metrics_phase2 import ARMS
from metrics_phase2_5 import compute_phase2_5_metrics
from result_store import read_jsonl
from result_store_phase2_5 import final_phase2_5_trials, validate_phase2_5_attempts


ANALYZER_VERSION = "phase2.5-local-analyzer-v2"
PRIMARY_PHASE2_5_ARMS = frozenset({"ACTION_ONLY", "INLINE_PROVENANCE"})
FULL_PHASE2_5_ARMS = frozenset(
    {"ACTION_ONLY", "INLINE_PROVENANCE", "ORACLE_PROVENANCE"}
)
FULL_PHASE2_CASE_COUNT = 81


def _cohort_value(row: dict[str, Any], field: str) -> Any:
    """Read a cohort selector, with a legacy provider-config compatibility path."""

    value = row.get(field)
    if value not in (None, ""):
        return value
    provider_config = row.get("provider_config")
    if isinstance(provider_config, dict):
        return provider_config.get(field)
    return None


def _cohort(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    if not attempts:
        raise ValueError("No Phase 2.5 attempts were found")
    fields = (
        "provider",
        "model_id",
        "model_revision",
        "dataset_version",
        "zero_shot_prompt_version",
        "schema_transport_version",
        "policy_version",
        "selection_scope_id",
        "benchmark_lock_id",
        "benchmark_lock_sha256",
    )
    result: dict[str, Any] = {}
    for field in fields:
        observed = {str(_cohort_value(row, field) or "") for row in attempts}
        if (
            field == "schema_transport_version"
            and result.get("zero_shot_prompt_version") != "ZERO_SHOT_V2"
            and observed == {""}
        ):
            # Preserve analyzability of already-written V1 artifacts while V2
            # requires a declared transport in the Phase 2.5 validator.
            result[field] = None
            continue
        if "" in observed or len(observed) != 1:
            raise ValueError(
                f"Phase 2.5 attempts have missing or mixed {field}: {sorted(observed)}"
            )
        result[field] = observed.pop()
    result["prompt_versions"] = sorted(
        {str(row.get("prompt_version")) for row in attempts if row.get("prompt_version")}
    )
    return result


def _scope_key(row: dict[str, Any]) -> tuple[str, str, int]:
    run = row.get("run")
    return (
        str(row.get("scene_id", "")),
        str(row.get("condition", "")),
        int(run) if isinstance(run, int) and not isinstance(run, bool) else 0,
    )


def _render_scope(keys: set[tuple[str, str, int]]) -> list[dict[str, Any]]:
    return [
        {"scene_id": scene_id, "condition": condition, "run": run}
        for scene_id, condition, run in sorted(keys)
    ]


def _paired_cohort(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit the exact Action Only/Inline comparison scope, including errors."""

    action_only = {
        _scope_key(row)
        for row in trials
        if row.get("architecture_arm") == "ACTION_ONLY"
    }
    inline = {
        _scope_key(row)
        for row in trials
        if row.get("architecture_arm") == "INLINE_PROVENANCE"
    }
    oracle = {
        _scope_key(row)
        for row in trials
        if row.get("architecture_arm") == "ORACLE_PROVENANCE"
    }
    paired = action_only & inline
    action_only_only = action_only - inline
    inline_only = inline - action_only
    run_values = {
        row.get("run")
        for row in trials
        if isinstance(row.get("run"), int) and not isinstance(row.get("run"), bool)
    }
    selected_values = {
        row.get("selected_case_count")
        for row in trials
        if isinstance(row.get("selected_case_count"), int)
        and not isinstance(row.get("selected_case_count"), bool)
    }
    selected_count = next(iter(selected_values)) if len(selected_values) == 1 else None
    expected_paired = (
        selected_count * len(run_values)
        if isinstance(selected_count, int) and run_values
        else None
    )
    exact_scope_match = bool(action_only and action_only == inline)
    return {
        "identity_fields": ["scene_id", "condition", "run"],
        "action_only_scope_count": len(action_only),
        "inline_provenance_scope_count": len(inline),
        "oracle_provenance_scope_count": len(oracle),
        "paired_scope_count": len(paired),
        "expected_paired_scope_count": expected_paired,
        "exact_scope_match": exact_scope_match,
        "scope_complete": bool(
            exact_scope_match
            and expected_paired is not None
            and len(paired) == expected_paired
        ),
        "oracle_scope_match": bool(action_only and action_only == oracle),
        "missing_from_inline_provenance": _render_scope(action_only_only),
        "missing_from_action_only": _render_scope(inline_only),
    }


def _completion_context(
    attempts: list[dict[str, Any]], metrics: dict[str, Any]
) -> dict[str, Any]:
    trials = final_phase2_5_trials(attempts)
    planned_values = {
        row["planned_trial_count"]
        for row in attempts
        if isinstance(row.get("planned_trial_count"), int)
    }
    planned = next(iter(planned_values)) if len(planned_values) == 1 else None
    selected_case_values = {row.get("selected_case_count") for row in attempts}
    benchmark_case_values = {row.get("benchmark_case_count") for row in attempts}
    selected_case_count = (
        next(iter(selected_case_values))
        if len(selected_case_values) == 1
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in selected_case_values
        )
        else None
    )
    benchmark_case_count = (
        next(iter(benchmark_case_values))
        if len(benchmark_case_values) == 1
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in benchmark_case_values
        )
        else None
    )
    core_counts = metrics["core_phase2_metrics"]["trial_counts"]
    paired = _paired_cohort(trials)
    selection_scope_complete = bool(
        planned is not None
        and len(trials) == planned
    )
    arms = sorted({str(row.get("architecture_arm")) for row in trials})
    primary_arms_complete = PRIMARY_PHASE2_5_ARMS.issubset(arms)
    full_arms_complete = FULL_PHASE2_5_ARMS.issubset(arms)
    paired_comparison_complete = bool(
        selection_scope_complete
        and selected_case_count is not None
        and benchmark_case_count is not None
        and selected_case_count == benchmark_case_count == FULL_PHASE2_CASE_COUNT
        and primary_arms_complete
        and paired["scope_complete"]
    )
    complete = bool(
        paired_comparison_complete
        and full_arms_complete
        and paired["oracle_scope_match"]
    )
    return {
        "dataset_complete": complete,
        "selection_scope_complete": selection_scope_complete,
        "primary_arms_complete": primary_arms_complete,
        "full_arms_complete": full_arms_complete,
        "paired_comparison_complete": paired_comparison_complete,
        "paired_cohort": paired,
        "planned_trial_count": planned,
        "selected_case_count": selected_case_count,
        "benchmark_case_count": benchmark_case_count,
        "selected_arms": arms,
        "attempted_trials": len(trials),
        "completed_trials": core_counts["completed"],
        "unresolved_error_trials": core_counts["unresolved_errors"],
        "result_kind": "local_vlm_zero_shot",
        "eligible_as_complete_local_evidence": complete,
    }


def generate_phase2_5_plots(metrics: dict[str, Any], output: Path) -> None:
    """Generate local-model plots from the unchanged Phase 2 metric outputs."""

    output.mkdir(parents=True, exist_ok=True)
    local = metrics["by_arm"]
    core = metrics["core_phase2_metrics"]["by_arm"]
    selected = [arm for arm in ARMS if core[arm]["usable_trials"]]
    labels = [arm.replace("_PROVENANCE", "").replace("_", " ").title() for arm in selected]
    plots = (
        (
            "provenance_accuracy.png",
            "Critical-argument provenance accuracy",
            [local[arm]["critical_argument_provenance_accuracy"] for arm in selected],
            "Accuracy",
        ),
        (
            "unsafe_execution_by_arm.png",
            "Automatic unsafe execution by architecture",
            [local[arm]["automatic_unsafe_execution_rate"] for arm in selected],
            "Rate",
        ),
        (
            "latency_by_arm.png",
            "Median local inference latency",
            [local[arm]["efficiency"]["p50_inference_latency_ms"] for arm in selected],
            "Milliseconds",
        ),
        (
            "structured_output_by_arm.png",
            "Structured-output parse success",
            [
                local[arm]["structured_output"]["structured_output_parse_success_rate"]
                for arm in selected
            ],
            "Rate",
        ),
        (
            "peak_vram_by_arm.png",
            "Peak allocated GPU memory",
            [
                (
                    value / (1024**3)
                    if (value := local[arm]["efficiency"]["gpu_peak_memory_allocated_bytes"])
                    is not None
                    else None
                )
                for arm in selected
            ],
            "GiB",
        ),
    )
    for filename, title, values, ylabel in plots:
        _plot_bars(
            output,
            filename,
            title,
            labels,
            values,
            ylabel=ylabel,
            mock_only=False,
            incomplete=False,
        )


def analyze_phase2_5(
    input_path: Path,
    output_path: Path,
    plots_dir: Path | None,
    *,
    system_info_path: Path | None = None,
) -> dict[str, Any]:
    attempts = read_jsonl(input_path)
    validate_phase2_5_attempts(attempts)
    trials = final_phase2_5_trials(attempts)
    metrics = compute_phase2_5_metrics(attempts)
    system_info: dict[str, Any] | None = None
    if system_info_path is not None and system_info_path.is_file():
        loaded = json.loads(system_info_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            system_info = loaded
    result = {
        "analyzer_version": ANALYZER_VERSION,
        "source": str(input_path),
        "cohort": _cohort(attempts),
        **_completion_context(attempts, metrics),
        "final_trial_count": len(trials),
        "metrics": metrics,
        "system_info": system_info,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if plots_dir is not None:
        generate_phase2_5_plots(metrics, plots_dir)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("results_phase2_5/gemma3-4b/raw_generations.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results_phase2_5/gemma3-4b/analysis.json")
    )
    parser.add_argument(
        "--plots-dir", type=Path, default=Path("results_phase2_5/gemma3-4b/plots")
    )
    parser.add_argument("--system-info", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    payload = analyze_phase2_5(
        args.input,
        args.output,
        None if args.no_plots else args.plots_dir,
        system_info_path=args.system_info,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ANALYZER_VERSION",
    "PRIMARY_PHASE2_5_ARMS",
    "analyze_phase2_5",
    "generate_phase2_5_plots",
]
