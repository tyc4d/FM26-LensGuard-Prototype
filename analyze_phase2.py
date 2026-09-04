#!/usr/bin/env python3
"""Analyze Phase 2 attempts, materialize plots, and summarize evidence without a verdict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from metrics_phase2 import ARMS, compute_phase2_metrics
from result_store import read_jsonl
from result_store_phase2 import (
    final_phase2_trials,
    phase2_attempt_accounting,
    validate_phase2_attempts,
)

COMMON_COHORT_FIELDS = (
    "provider",
    "model",
    "dataset_version",
    "registry_version",
    "policy_version",
    "selection_scope_id",
    "experiment_config_id",
)


def validate_phase2_cohort(attempts: list[dict[str, Any]]) -> dict[str, str]:
    if not attempts:
        raise ValueError("No Phase 2 attempts were found")
    cohort: dict[str, str] = {}
    for field in COMMON_COHORT_FIELDS:
        observed = {str(row.get(field, "")) for row in attempts}
        if "" in observed or len(observed) != 1:
            raise ValueError(f"Phase 2 attempts have missing or mixed {field}: {sorted(observed)}")
        cohort[field] = observed.pop()
    return cohort


def phase2_completion_context(
    attempts: list[dict[str, Any]], metrics: dict[str, Any]
) -> dict[str, Any]:
    """Describe whether a deduplicated cohort is complete scientific evidence."""

    cohort = validate_phase2_cohort(attempts)
    planned_values = {
        int(row["planned_trial_count"])
        for row in attempts
        if isinstance(row.get("planned_trial_count"), int)
    }
    planned = next(iter(planned_values)) if len(planned_values) == 1 else None
    complete = (
        planned is not None
        and metrics["trial_counts"]["completed"] == planned
        and metrics["trial_counts"]["unresolved_errors"] == 0
    )
    mock_only = cohort["provider"] == "mock"
    return {
        "cohort": cohort,
        "mock_only": mock_only,
        "result_kind": "mock_validation" if mock_only else "gemini_experiment",
        "dataset_complete": complete,
        "planned_trial_count": planned,
        "eligible_as_complete_gemini_evidence": bool(not mock_only and complete),
    }


def summarize_go_nogo(metrics: dict[str, Any]) -> dict[str, Any]:
    by_arm = metrics["by_arm"]
    inline = by_arm["INLINE_PROVENANCE"]
    two_pass = by_arm["TWO_PASS_PROVENANCE"]
    oracle = by_arm["ORACLE_PROVENANCE"]
    action = by_arm["ACTION_ONLY"]
    comparisons = metrics["comparisons"]
    inline_prov = inline.get("provenance") or {}
    two_prov = two_pass.get("provenance") or {}
    return {
        "interpretation": (
            "Evidence indicators only; LensGuard does not reduce Phase 2 to one magical "
            "threshold or issue an automatic GO/NO-GO verdict."
        ),
        "inline_provenance_accuracy": inline_prov.get("critical_argument_provenance_accuracy"),
        "two_pass_provenance_accuracy": two_prov.get("critical_argument_provenance_accuracy"),
        "inline_provenance_coverage": inline_prov.get("provenance_coverage"),
        "inline_visual_argument_units": inline_prov.get("critical_argument_units"),
        "inline_all_origin_provenance_accuracy": (inline_prov.get("all_origins") or {}).get(
            "critical_argument_provenance_accuracy"
        ),
        "inline_hallucinated_evidence_count": (
            inline_prov.get("evidence_status_distribution", {}).get("hallucinated")
        ),
        "action_only_unsafe_execution_rate": action.get("automatic_unsafe_execution_rate"),
        "inline_unsafe_execution_rate": inline.get("automatic_unsafe_execution_rate"),
        "oracle_unsafe_execution_rate": oracle.get("automatic_unsafe_execution_rate"),
        "inline_oracle_unsafe_execution_gap": comparisons.get("inline_oracle_unsafe_execution_gap"),
        "inline_api_call_reduction_vs_two_pass_percent": comparisons.get(
            "inline_api_call_reduction_vs_two_pass_percent"
        ),
        "inline_latency_overhead_vs_action_only_percent": comparisons.get(
            "inline_latency_overhead_vs_action_only_percent"
        ),
        "inline_false_warning_rate": inline.get("false_warning_confirmation_rate"),
        "inline_correct_safe_proposals": inline.get("correct_safe_proposals"),
        "inline_escalated_correct_safe_proposals": inline.get("escalated_correct_safe_proposals"),
        "inline_correct_safe_proposal_escalation_rate": inline.get(
            "correct_safe_proposal_escalation_rate"
        ),
        "inline_resisted_attack_correct_proposals": inline.get("resisted_attack_correct_proposals"),
        "inline_escalated_resisted_attack_correct_proposals": inline.get(
            "escalated_resisted_attack_correct_proposals"
        ),
        "inline_resisted_attack_correct_proposal_escalation_rate": inline.get(
            "resisted_attack_correct_proposal_escalation_rate"
        ),
        "inline_clean_action_accuracy": inline.get("clean_action_accuracy"),
        "action_only_clean_action_accuracy": action.get("clean_action_accuracy"),
        "inline_trusted_user_preservation": inline.get("trusted_user_preservation"),
        "inline_trusted_user_end_to_end_preservation": inline.get(
            "trusted_user_end_to_end_preservation"
        ),
        "inline_trusted_update_preservation": inline.get("trusted_conflicting_update_preservation"),
        "inline_gate_p50_ms": inline["efficiency"].get("p50_thin_gate_latency_ms"),
        "inline_mapping_p50_ms": inline["efficiency"].get("p50_mapping_latency_ms"),
        "inline_physical_requests_per_trial": inline["efficiency"].get(
            "mean_total_gemini_api_calls_per_trial"
        ),
        "two_pass_physical_requests_per_trial": two_pass["efficiency"].get(
            "mean_total_gemini_api_calls_per_trial"
        ),
        "questions_for_human_review": [
            "Does inline evidence map to the value-supporting region across actions and conditions?",
            "Are source-type errors driven by attacker-written authority words?",
            "Does inline authorization approach the oracle arm without excessive escalation?",
            "Does the gate preserve benchmark-correct safe proposals, including when the model "
            "resists attacker-controlled evidence?",
            "Does inline preserve action accuracy and explicitly authorized user targets?",
            "Is its measured latency/token overhead acceptable relative to action-only?",
            "Does one-pass materially reduce calls and latency relative to two-pass?",
            "Do raw hallucinated, ambiguous, and missing evidence cases match their labels?",
        ],
    }


def _plot_bars(
    output: Path,
    filename: str,
    title: str,
    labels: list[str],
    values: list[float | None],
    *,
    ylabel: str,
    mock_only: bool,
    incomplete: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = [(label, value) for label, value in zip(labels, values) if value is not None]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    if pairs:
        bars = ax.bar([item[0] for item in pairs], [item[1] for item in pairs], color="#315D73")
        ax.bar_label(bars, fmt="%.3g", padding=3)
    else:
        ax.text(0.5, 0.5, "No usable data", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=12)
    footer = None
    if mock_only:
        footer = "MOCK VALIDATION ONLY — NOT GEMINI EVIDENCE"
    elif incomplete:
        footer = "INCOMPLETE GEMINI COHORT — DESCRIPTIVE PARTIAL RESULTS"
    if footer:
        fig.text(
            0.5,
            0.01,
            footer,
            ha="center",
            color="#A61B1B",
            weight="bold",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(output / filename, dpi=160)
    plt.close(fig)


def _plot_escalation_rates(
    output: Path,
    labels: list[str],
    clean_rates: list[float | None],
    correct_safe_rates: list[float | None],
    *,
    mock_only: bool,
    incomplete: bool,
) -> None:
    """Plot the narrow clean FWR beside the broader safe-proposal metric."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    width = 0.36
    rendered = False
    clean_label_used = False
    safe_label_used = False
    for index, (clean, safe) in enumerate(zip(clean_rates, correct_safe_rates)):
        if clean is not None:
            bars = ax.bar(
                index - width / 2,
                clean,
                width,
                color="#315D73",
                label="Clean correct proposals" if not clean_label_used else None,
            )
            ax.bar_label(bars, fmt="%.3g", padding=3)
            clean_label_used = True
            rendered = True
        if safe is not None:
            bars = ax.bar(
                index + width / 2,
                safe,
                width,
                color="#C97A40",
                label="All correct safe proposals" if not safe_label_used else None,
            )
            ax.bar_label(bars, fmt="%.3g", padding=3)
            safe_label_used = True
            rendered = True
    if rendered:
        ax.set_xticks(range(len(labels)), labels, rotation=12)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Escalation rate")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No usable data", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("Clean false warnings and escalation of correct safe proposals")
    footer = None
    if mock_only:
        footer = "MOCK VALIDATION ONLY — NOT GEMINI EVIDENCE"
    elif incomplete:
        footer = "INCOMPLETE GEMINI COHORT — DESCRIPTIVE PARTIAL RESULTS"
    if footer:
        fig.text(
            0.5,
            0.01,
            footer,
            ha="center",
            color="#A61B1B",
            weight="bold",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(output / "false_warning_by_arm.png", dpi=160)
    plt.close(fig)


def generate_phase2_plots(
    metrics: dict[str, Any],
    output: Path,
    *,
    mock_only: bool,
    incomplete: bool = False,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    by_arm = metrics["by_arm"]
    labels = [name.replace("_PROVENANCE", "").replace("_", " ").title() for name in ARMS]
    _plot_bars(
        output,
        "provenance_accuracy.png",
        "Critical-argument provenance accuracy",
        labels,
        [
            (by_arm[name].get("provenance") or {}).get("critical_argument_provenance_accuracy")
            for name in ARMS
        ],
        ylabel="Accuracy",
        mock_only=mock_only,
        incomplete=incomplete,
    )
    _plot_bars(
        output,
        "unsafe_execution_by_arm.png",
        "Automatic unsafe execution by architecture",
        labels,
        [by_arm[name].get("automatic_unsafe_execution_rate") for name in ARMS],
        ylabel="Rate",
        mock_only=mock_only,
        incomplete=incomplete,
    )
    _plot_bars(
        output,
        "latency_by_arm.png",
        "Median end-to-end inference and gate latency",
        labels,
        [by_arm[name]["efficiency"].get("p50_end_to_end_latency_ms") for name in ARMS],
        ylabel="Milliseconds",
        mock_only=mock_only,
        incomplete=incomplete,
    )
    _plot_bars(
        output,
        "api_calls_by_arm.png",
        "Physical API requests per trial",
        labels,
        [by_arm[name]["efficiency"].get("mean_total_gemini_api_calls_per_trial") for name in ARMS],
        ylabel="Physical API requests",
        mock_only=mock_only,
        incomplete=incomplete,
    )
    _plot_escalation_rates(
        output,
        labels,
        [by_arm[name].get("false_warning_confirmation_rate") for name in ARMS],
        [by_arm[name].get("correct_safe_proposal_escalation_rate") for name in ARMS],
        mock_only=mock_only,
        incomplete=incomplete,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    points = []
    for arm in ARMS:
        provenance = by_arm[arm].get("provenance") or {}
        accuracy = provenance.get("critical_argument_provenance_accuracy")
        unsafe = by_arm[arm].get("automatic_unsafe_execution_rate")
        if accuracy is not None and unsafe is not None:
            points.append((arm, accuracy, unsafe))
    if points:
        for arm, accuracy, unsafe in points:
            ax.scatter(accuracy, unsafe, s=70)
            ax.annotate(
                arm.replace("_PROVENANCE", ""),
                (accuracy, unsafe),
                xytext=(5, 5),
                textcoords="offset points",
            )
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Critical-argument provenance accuracy")
        ax.set_ylabel("Automatic unsafe execution rate")
    else:
        ax.text(0.5, 0.5, "No usable provenance/security pairs", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("Security versus provenance accuracy")
    footer = None
    if mock_only:
        footer = "MOCK VALIDATION ONLY — NOT GEMINI EVIDENCE"
    elif incomplete:
        footer = "INCOMPLETE GEMINI COHORT — DESCRIPTIVE PARTIAL RESULTS"
    if footer:
        fig.text(
            0.5,
            0.01,
            footer,
            ha="center",
            color="#A61B1B",
            weight="bold",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(output / "security_vs_provenance_accuracy.png", dpi=160)
    plt.close(fig)


def analyze_phase2(input_path: Path, output_path: Path, plots_dir: Path | None) -> dict[str, Any]:
    attempts = read_jsonl(input_path)
    validate_phase2_attempts(attempts)
    trials = final_phase2_trials(attempts)
    metrics = compute_phase2_metrics(attempts)
    context = phase2_completion_context(attempts, metrics)
    result = {
        "source": str(input_path),
        **context,
        "attempt_accounting": phase2_attempt_accounting(attempts),
        "final_trial_count": len(trials),
        "metrics": metrics,
        "go_no_go_evidence": summarize_go_nogo(metrics),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if plots_dir is not None:
        generate_phase2_plots(
            metrics,
            plots_dir,
            mock_only=context["mock_only"],
            incomplete=not context["dataset_complete"],
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results_phase2/raw_attempts.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results_phase2/analysis.json"))
    parser.add_argument("--plots-dir", type=Path, default=Path("results_phase2/plots"))
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    payload = analyze_phase2(
        arguments.input,
        arguments.output,
        None if arguments.no_plots else arguments.plots_dir,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
