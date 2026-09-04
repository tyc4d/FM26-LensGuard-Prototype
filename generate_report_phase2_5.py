#!/usr/bin/env python3
"""Generate per-model and aggregate LensGuard Phase 2.5 reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from metrics_phase2 import ARMS
from result_store import read_jsonl
from result_store_phase2_5 import final_phase2_5_trials, validate_phase2_5_attempts


REPORT_VERSION = "phase2.5-local-report-v3"
MODEL_ORDER = ("gemma3-4b", "qwen3vl-8b", "minicpm-v4.5")
MODEL_LABELS = {
    "gemma3-4b": "Gemma 3 4B",
    "qwen3vl-8b": "Qwen3-VL 8B",
    "minicpm-v4.5": "MiniCPM-V 4.5",
}
MODEL_REPOSITORIES = {
    "gemma3-4b": "google/gemma-3-4b-it",
    "qwen3vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "minicpm-v4.5": "openbmb/MiniCPM-V-4_5",
}


def _pct(value: Any) -> str:
    return f"{value:.1%}" if isinstance(value, (int, float)) else "N/A"


def _ms(value: Any) -> str:
    return f"{value:.1f} ms" if isinstance(value, (int, float)) else "N/A"


def _gib(value: Any) -> str:
    return f"{value / (1024**3):.2f} GiB" if isinstance(value, (int, float)) else "N/A"


def _decimal(value: Any) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "N/A"


def _params(value: Any) -> str:
    if not isinstance(value, int) or isinstance(value, bool):
        return "N/A"
    return f"{value / 1_000_000_000:.3g}B"


def _count(value: Any) -> str:
    return str(value) if isinstance(value, int) and not isinstance(value, bool) else "N/A"


def _rate_with_counts(summary: Any) -> str:
    if not isinstance(summary, dict):
        return "N/A"
    numerator = summary.get("successes")
    denominator = summary.get("assessed_trials")
    rate = summary.get("rate")
    if (
        not isinstance(numerator, int)
        or isinstance(numerator, bool)
        or not isinstance(denominator, int)
        or isinstance(denominator, bool)
    ):
        return "N/A"
    return f"{_pct(rate)} ({numerator}/{denominator})"


def _unsafe_with_counts(summary: Any) -> str:
    if not isinstance(summary, dict):
        return "N/A"
    numerator = summary.get("unsafe_executions")
    denominator = summary.get("gate_assessed_attack_trials")
    rate = summary.get("unsafe_execution_rate")
    if (
        not isinstance(numerator, int)
        or isinstance(numerator, bool)
        or not isinstance(denominator, int)
        or isinstance(denominator, bool)
    ):
        return "N/A"
    return f"{_pct(rate)} ({numerator}/{denominator})"


def _normalization_cell(summary: Any) -> str:
    if not isinstance(summary, dict):
        return "N/A"
    count = summary.get("normalization_count")
    methods = summary.get("normalization_method_counts")
    if not isinstance(count, int) or isinstance(count, bool):
        return "N/A"
    if not isinstance(methods, dict) or not methods:
        return str(count)
    rendered = ", ".join(f"{key}: {value}" for key, value in sorted(methods.items()))
    return f"{count} ({rendered})"


def _cell(value: Any) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return "N/A"
    rendered = str(value).strip()
    if not rendered:
        return "N/A"
    return rendered.replace("|", "\\|").replace("\n", " ")


def _arm_label(arm: str) -> str:
    return {
        "ACTION_ONLY": "Action Only",
        "TWO_PASS_PROVENANCE": "Two Pass",
        "INLINE_PROVENANCE": "Inline Provenance",
        "ORACLE_PROVENANCE": "Oracle",
    }[arm]


def _model_value(analysis: dict[str, Any], field: str) -> Any:
    values = analysis.get("metrics", {}).get("model_metadata_values", {}).get(field, [])
    return values[0] if isinstance(values, list) and len(values) == 1 else None


def _hardware_lines(system_info: dict[str, Any] | None) -> list[str]:
    if not isinstance(system_info, dict):
        return ["System metadata was not available."]
    return [
        f"- GPU: `{system_info.get('gpu_model', 'unknown')}`",
        f"- Total VRAM: `{_gib(system_info.get('vram_total_bytes'))}`",
        f"- NVIDIA driver: `{system_info.get('nvidia_driver_version', 'unknown')}`",
        f"- PyTorch / CUDA runtime: `{system_info.get('torch_version', 'unknown')}` / "
        f"`{system_info.get('cuda_runtime_visible_to_torch', 'unknown')}`",
        f"- Transformers / Python: `{system_info.get('transformers_version', 'unknown')}` / "
        f"`{system_info.get('python_version', 'unknown')}`",
        f"- OS: `{system_info.get('os', 'unknown')}`",
    ]


def build_local_model_report(
    attempts: list[dict[str, Any]],
    analysis: dict[str, Any],
    *,
    source_path: Path,
    system_info: dict[str, Any] | None,
) -> str:
    validate_phase2_5_attempts(attempts)
    trials = final_phase2_5_trials(attempts)
    cohort = analysis["cohort"]
    metrics = analysis["metrics"]
    core = metrics["core_phase2_metrics"]
    local = metrics["by_arm"]
    incomplete = not analysis.get("dataset_complete", False)
    paired = _paired_scope(trials)
    prompt_profile = str(cohort.get("zero_shot_prompt_version", "unknown"))
    schema_transport = str(cohort.get("schema_transport_version", "unknown"))
    failure_counts = metrics.get("contract_quality", {}).get(
        "failure_category_counts", {}
    )
    failure_summary = (
        ", ".join(f"{name}={count}" for name, count in sorted(failure_counts.items()))
        if isinstance(failure_counts, dict) and failure_counts
        else "none"
    )
    errors = [row for row in trials if row.get("status") == "error"]
    hallucinated: list[str] = []
    for row in trials:
        for unit in row.get("provenance_evaluations", []):
            if not isinstance(unit, dict):
                continue
            items = unit.get("reported_evidence_items")
            for item in items if isinstance(items, list) else []:
                if isinstance(item, dict) and item.get("evidence_status") == "hallucinated":
                    hallucinated.append(
                        f"{row.get('scene_id')} / {row.get('architecture_arm')} / "
                        f"{unit.get('argument_name')}: {item.get('evidence_text')!r}"
                    )
    lines = ["# LensGuard Phase 2.5 — Local / Edge VLM Evaluation", ""]
    if incomplete:
        lines += [
            "> **INCOMPLETE LOCAL COHORT.** These are smoke/partial results, not final Phase 2.5 "
            "evidence.",
            "",
        ]
    lines += [
        "## 1. Research Questions",
        "",
        "Can a zero-shot local VLM recover protected actions, critical arguments, observable "
        "supporting evidence, and source categories accurately enough for LensGuard's model-free "
        "Thin Trusted Gate? What quality, security, latency, and VRAM trade-off separates a "
        "4B-class model from stronger local baselines?",
        "",
        "## 2. Benchmark Freeze",
        "",
        f"Lock: `{cohort['benchmark_lock_id']}` "
        f"(`{cohort['benchmark_lock_sha256']}`). Dataset `{cohort['dataset_version']}`, "
        f"prompt cohort `{prompt_profile}` with semantic prompt versions "
        f"`{', '.join(cohort.get('prompt_versions', []))}`, policy `{cohort['policy_version']}`. "
        f"Schema transport `{schema_transport}`. The Phase 2 dataset, attack definition, mapper, "
        "evaluator, registry, policy, and gate semantics remain authoritative.",
        "",
        "## 3. Hardware Environment",
        "",
        *_hardware_lines(system_info),
        "",
        "The RTX 4090 is an evaluation and edge-proxy platform. These measurements do not prove "
        "deployment on current glasses hardware.",
        "",
        "## 4. Models",
        "",
        f"Provider/model: `local` / `{cohort['model_id']}` at revision "
        f"`{cohort['model_revision']}`. Parameter count is `{_params(_model_value(analysis, 'parameter_count'))}`. "
        f"Runtime profile: `{_model_value(analysis, 'model_dtype') or 'unknown'}`, "
        f"quantization `{_model_value(analysis, 'quantization') or 'unknown'}`, attention "
        f"`{_model_value(analysis, 'attention_backend') or 'unknown'}`.",
        "",
        "## 5. Experimental Protocol",
        "",
        f"{prompt_profile}; schema transport {schema_transport}; batch 1; sampling disabled; one "
        "model resident at a time. Model output is parsed conservatively. Only an explicitly "
        "recorded deterministic compatibility normalization may produce the canonical object; no "
        "argument, evidence value, or coordinate is guessed. The deterministic Phase 2 evidence "
        "mapper and model-free Thin Trusted Gate consume the normalized common schema.",
        "",
        f"Source: `{source_path}`. Final trials: {len(trials)}; completed: "
        f"{core['trial_counts']['completed']}; unresolved errors: "
        f"{core['trial_counts']['unresolved_errors']}. Selected cases: "
        f"{analysis.get('selected_case_count', 'unknown')}/"
        f"{analysis.get('benchmark_case_count', 'unknown')}; primary arms complete: "
        f"{analysis.get('primary_arms_complete', False)}.",
        f"Action Only / Inline attempted scope identical: {paired['scope_identical']} "
        f"({paired['paired_trials']} paired; {paired['action_only_trials']} Action Only; "
        f"{paired['inline_trials']} Inline). Errors remain members of the attempted scope.",
        "",
        "## 6. Action Extraction Results",
        "",
        "This table retains the frozen Phase 2 completed-trial denominators. Section 9 reports "
        "independent action and critical-argument correctness for every trial whose action "
        "candidate remained scoreable, including provenance-format failures.",
        "",
        "| Arm | Usable | Clean action accuracy | Action class accuracy | Critical argument accuracy | Exact attacker adoption |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = local[arm]
        lines.append(
            f"| {_arm_label(arm)} | {core['by_arm'][arm]['usable_trials']} | "
            f"{_pct(item['clean_action_accuracy'])} | "
            f"{_pct(item['action_class_extraction_accuracy'])} | "
            f"{_pct(item['critical_argument_accuracy'])} | "
            f"{_pct(item['exact_attacker_target_adoption'])} |"
        )
    lines += [
        "",
        "## 7. Provenance Results",
        "",
        "| Arm | Text match | Region accuracy | Source accuracy | Argument provenance | Coverage | Missing | Ambiguous | Hallucinated items |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = local[arm]
        lines.append(
            f"| {_arm_label(arm)} | {_pct(item['evidence_text_match_accuracy'])} | "
            f"{_pct(item['evidence_region_accuracy'])} | "
            f"{_pct(item['source_type_classification_accuracy'])} | "
            f"{_pct(item['critical_argument_provenance_accuracy'])} | "
            f"{_pct(item['provenance_coverage'])} | "
            f"{_pct(item['missing_provenance_rate'])} | "
            f"{_pct(item['ambiguous_provenance_rate'])} | "
            f"{_pct(item['hallucinated_evidence_rate'])} |"
        )
    lines += [
        "",
        "## 8. Security Results",
        "",
        "| Arm | Unsafe automatic execution | Thin Gate escalation recall | False escalation | Trusted-user preservation |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = local[arm]
        lines.append(
            f"| {_arm_label(arm)} | {_pct(item['automatic_unsafe_execution_rate'])} | "
            f"{_pct(item['thin_gate_escalation_recall'])} | "
            f"{_pct(item['false_escalation_rate'])} | "
            f"{_pct(item['trusted_user_preservation'])} |"
        )
    lines += [
        "",
        "## 9. Structured Output Reliability",
        "",
        "Raw JSON parsing, raw structural-schema validity, normalized acceptance, semantic "
        "provenance, action correctness, and gate outcomes are measured separately. `N/A` means "
        "the stage was not assessed; it is never counted as safe or correct.",
        "",
        "| Arm | Attempted | Completed | Errors | Runtime | Parse success | Raw structural schema valid | Normalized accepted | Normalizations | Contract semantic | Provenance semantic | Action correct | Critical argument correct | Unsafe / assessed attacks | Gate coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[
            _contract_row(_arm_label(arm), local[arm].get("contract_quality", {}))
            for arm in ARMS
        ],
        _contract_row("Overall", metrics.get("contract_quality", {})),
        "",
        "`Raw structural schema valid` measures the parsed JSON payload before any compatibility shape "
        "normalization. Safe transport cleanup (for example, removing one whole-response JSON "
        "fence) belongs to `Parse success`; the exact response is still preserved. `Normalized "
        "accepted` measures whether the canonical schema was valid after the narrowly scoped, "
        "recorded compatibility normalizer. A normalized output remains a raw-schema miss and "
        "increments Normalizations.",
        "",
        "## 10. Latency",
        "",
        "| Arm | p50 inference | p95 inference | p50 end-to-end | p95 end-to-end | p50 mapper | p50 Thin Gate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = local[arm]["efficiency"]
        lines.append(
            f"| {_arm_label(arm)} | {_ms(item['p50_inference_latency_ms'])} | "
            f"{_ms(item['p95_inference_latency_ms'])} | "
            f"{_ms(item['p50_end_to_end_latency_ms'])} | "
            f"{_ms(item['p95_end_to_end_latency_ms'])} | "
            f"{_ms(item['p50_evidence_mapper_latency_ms'])} | "
            f"{_ms(item['p50_thin_gate_latency_ms'])} |"
        )
    lines += [
        "",
        "## 11. VRAM",
        "",
        f"Model load time: {_ms(metrics['efficiency']['model_load_time_ms'])}. Maximum allocated "
        f"VRAM: {_gib(metrics['efficiency']['gpu_peak_memory_allocated_bytes'])}; maximum reserved "
        f"VRAM: {_gib(metrics['efficiency']['gpu_peak_memory_reserved_bytes'])}. Token throughput "
        f"p50: {metrics['efficiency']['p50_tokens_per_second'] or 'N/A'} tokens/s.",
        "",
        "## 12. Model-by-Model Failure Cases",
        "",
        f"Unresolved trial errors: {len(errors)}. Hallucinated reported evidence items: "
        f"{len(hallucinated)}. Recorded failure categories: {failure_summary}.",
        *(
            [f"- {row.get('scene_id')} / {row.get('architecture_arm')}: {row.get('error_type')} — {row.get('error_message')}" for row in errors[:20]]
            or ["- No unresolved trial errors recorded."]
        ),
        *(
            [f"- {description}" for description in hallucinated[:20]]
            or ["- No hallucinated evidence item recorded."]
        ),
        "",
        "## 13. Oracle Gap",
        "",
        f"Inline minus Oracle unsafe-execution rate: "
        f"{_pct(core['comparisons']['inline_oracle_unsafe_execution_gap'])}. Inline minus Oracle "
        f"argument-provenance accuracy: "
        f"{_pct(core['comparisons']['inline_oracle_provenance_accuracy_gap'])}.",
        "",
        "## 14. Local-vs-Cloud Comparison Placeholder",
        "",
        "Gemini and OpenAI rows are intentionally absent until independently completed frozen "
        "Phase 2 cohorts are supplied. Local and cloud raw trials must remain separate and may "
        "only be compared after lock/version compatibility checks.",
        "",
        "## 15. Limitations",
        "",
        "- Evidence is observable self-reported attribution, not chain of thought or causal proof.",
        "- Source-category estimates never create real authority; only separate trusted channels can.",
        "- Synthetic 1200×760 scenes and annotated regions are not physical wearable input.",
        "- Mapper timing excludes production OCR/segmentation and region acquisition.",
        "- Missing model confidence is invalid under the frozen schema; no confidence is fabricated.",
        "- A 4090 result is an edge-proxy measurement, not a glasses-deployment demonstration.",
        "",
        "## 16. Phase 2.5 Go / No-Go",
        "",
        "No aggregate score or automatic verdict is issued. A GO signal requires useful grounded "
        "provenance, materially lower unsafe execution after the Thin Gate, preserved clean/user "
        "utility, manageable hallucination/parse failure, usable event-driven latency, and VRAM "
        "headroom. Phase 2.6 LoRA/SFT is justified only if zero-shot attribution is insufficient "
        "but shows learnable signal, with the physical holdout never used for training. Phase 3 is "
        "justified only after at least one complete local cohort shows grounded provenance across "
        "families, tolerable false escalation, and a small enough Oracle security gap.",
    ]
    return "\n".join(str(line) for line in lines) + "\n"


def _arm_metrics(analysis: dict[str, Any], arm: str) -> dict[str, Any]:
    metrics = analysis.get("metrics")
    by_arm = metrics.get("by_arm") if isinstance(metrics, dict) else None
    item = by_arm.get(arm) if isinstance(by_arm, dict) else None
    return item if isinstance(item, dict) else {}


def _core_arm_metrics(analysis: dict[str, Any], arm: str) -> dict[str, Any]:
    metrics = analysis.get("metrics")
    core = metrics.get("core_phase2_metrics") if isinstance(metrics, dict) else None
    by_arm = core.get("by_arm") if isinstance(core, dict) else None
    item = by_arm.get(arm) if isinstance(by_arm, dict) else None
    return item if isinstance(item, dict) else {}


def _efficiency(analysis: dict[str, Any]) -> dict[str, Any]:
    metrics = analysis.get("metrics")
    item = metrics.get("efficiency") if isinstance(metrics, dict) else None
    return item if isinstance(item, dict) else {}


def _structured_output(analysis: dict[str, Any]) -> dict[str, Any]:
    metrics = analysis.get("metrics")
    item = metrics.get("structured_output") if isinstance(metrics, dict) else None
    return item if isinstance(item, dict) else {}


def _contract_quality(
    analysis: dict[str, Any], arm: str | None = None
) -> dict[str, Any]:
    metrics = analysis.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    if arm is None:
        item = metrics.get("contract_quality")
    else:
        by_arm = metrics.get("by_arm")
        arm_metrics = by_arm.get(arm) if isinstance(by_arm, dict) else None
        item = arm_metrics.get("contract_quality") if isinstance(arm_metrics, dict) else None
    return item if isinstance(item, dict) else {}


def _paired_scope(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe Action-Only/Inline pairing without treating errors as absent cases."""

    def identities(arm: str) -> set[tuple[str, str, str]]:
        return {
            (
                str(row.get("scene_id", "")),
                str(row.get("condition", "")),
                str(row.get("run", "")),
            )
            for row in trials
            if row.get("architecture_arm") == arm
        }

    action = identities("ACTION_ONLY")
    inline = identities("INLINE_PROVENANCE")
    return {
        "action_only_trials": len(action),
        "inline_trials": len(inline),
        "paired_trials": len(action & inline),
        "scope_identical": bool(action and inline and action == inline),
    }


def _contract_row(label: str, quality: dict[str, Any], *, model: str | None = None) -> str:
    prefix = f"| {model} | {label}" if model is not None else f"| {label}"
    return (
        f"{prefix} | {_count(quality.get('attempted_trials'))} | "
        f"{_count(quality.get('completed_trials'))} | "
        f"{_count(quality.get('unresolved_error_trials'))} | "
        f"{_count(quality.get('runtime_error_trials'))} | "
        f"{_rate_with_counts(quality.get('parse'))} | "
        f"{_rate_with_counts(quality.get('raw_schema'))} | "
        f"{_rate_with_counts(quality.get('normalized_schema'))} | "
        f"{_normalization_cell(quality)} | "
        f"{_rate_with_counts(quality.get('contract_semantic'))} | "
        f"{_rate_with_counts(quality.get('provenance_semantic'))} | "
        f"{_rate_with_counts(quality.get('action_correctness'))} | "
        f"{_rate_with_counts(quality.get('critical_argument_correctness'))} | "
        f"{_unsafe_with_counts(quality)} | "
        f"{_pct(quality.get('unsafe_execution_assessment_coverage'))} |"
    )


def _cohort_status(analysis: dict[str, Any] | None) -> str:
    if analysis is None:
        return "Not supplied"
    if analysis.get("dataset_complete") is True:
        return "Complete benchmark cohort"
    if analysis.get("selection_scope_complete") is True:
        return "Smoke/partial (selected scope complete)"
    return "Incomplete smoke/partial cohort"


def _table_row(alias: str, analysis: dict[str, Any] | None) -> str:
    if analysis is None:
        return f"| {MODEL_LABELS[alias]} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
    inline = _arm_metrics(analysis, "INLINE_PROVENANCE")
    action_quality = _contract_quality(analysis, "INLINE_PROVENANCE").get(
        "action_correctness", {}
    )
    efficiency = inline.get("efficiency", {})
    return (
        f"| {MODEL_LABELS[alias]} | {_params(_model_value(analysis, 'parameter_count'))} | "
        f"{_pct(action_quality.get('rate'))} | "
        f"{_pct(inline.get('critical_argument_provenance_accuracy'))} | "
        f"{_pct(inline.get('hallucinated_evidence_rate'))} | "
        f"{_pct(inline.get('automatic_unsafe_execution_rate'))} | "
        f"{_ms(efficiency.get('p50_inference_latency_ms'))} | "
        f"{_gib(efficiency.get('gpu_peak_memory_allocated_bytes'))} |"
    )


def _rate_range(values: list[float]) -> str:
    if not values:
        return "N/A"
    if min(values) == max(values):
        return _pct(values[0])
    return f"{_pct(min(values))}–{_pct(max(values))}"


def _oracle_row(analyses: dict[str, dict[str, Any]]) -> str:
    oracle_rows = [
        _arm_metrics(analysis, "ORACLE_PROVENANCE")
        for analysis in analyses.values()
        if _core_arm_metrics(analysis, "ORACLE_PROVENANCE").get("usable_trials")
    ]
    provenance = [
        value
        for row in oracle_rows
        if isinstance(
            (value := row.get("critical_argument_provenance_accuracy")), (int, float)
        )
        and not isinstance(value, bool)
    ]
    hallucination = [
        value
        for row in oracle_rows
        if isinstance((value := row.get("hallucinated_evidence_rate")), (int, float))
        and not isinstance(value, bool)
    ]
    unsafe = [
        value
        for row in oracle_rows
        if isinstance(
            (value := row.get("automatic_unsafe_execution_rate")), (int, float)
        )
        and not isinstance(value, bool)
    ]
    return (
        "| Oracle (matched cohorts) | N/A | - | "
        f"{_rate_range(provenance)} | {_rate_range(hallucination)} | "
        f"{_rate_range(unsafe)} | - | - |"
    )


def _validate_aggregate_analyses(analyses: dict[str, dict[str, Any]]) -> None:
    """Reject mislabeled or scientifically incomparable local cohorts."""

    comparison_fields = (
        "benchmark_lock_id",
        "benchmark_lock_sha256",
        "dataset_version",
        "zero_shot_prompt_version",
        "schema_transport_version",
        "policy_version",
        "prompt_versions",
        "selection_scope_id",
    )
    expected_comparison: tuple[str, ...] | None = None
    expected_protocol: tuple[str, ...] | None = None
    for alias, analysis in analyses.items():
        if alias not in MODEL_ORDER:
            raise ValueError(f"Unsupported local-model alias in aggregate report: {alias!r}")
        if not isinstance(analysis, dict):
            raise ValueError(f"Analysis for {alias} must be a JSON object")
        if not isinstance(analysis.get("metrics"), dict):
            raise ValueError(f"Analysis for {alias} lacks metrics")
        cohort = analysis.get("cohort")
        if not isinstance(cohort, dict):
            raise ValueError(f"Analysis for {alias} lacks cohort metadata")
        expected_repository = MODEL_REPOSITORIES[alias]
        if cohort.get("model_id") != expected_repository:
            raise ValueError(
                f"Analysis labeled {alias} identifies model {cohort.get('model_id')!r}; "
                f"expected {expected_repository!r}"
            )
        missing = [field for field in comparison_fields if not cohort.get(field)]
        if missing:
            raise ValueError(f"Analysis for {alias} lacks comparison metadata {missing}")
        comparison = tuple(
            json.dumps(cohort[field], sort_keys=True, separators=(",", ":"))
            for field in comparison_fields
        )
        if expected_comparison is None:
            expected_comparison = comparison
        elif comparison != expected_comparison:
            raise ValueError(
                "Local analyses use different benchmark locks, datasets, prompts, schema "
                "transports, policies, or selection scopes"
            )

        protocol_values = (
            analysis.get("selected_arms"),
            analysis.get("selected_case_count"),
            analysis.get("benchmark_case_count"),
            analysis.get("planned_trial_count"),
        )
        if any(value is None for value in protocol_values):
            raise ValueError(f"Analysis for {alias} lacks selection/protocol metadata")
        protocol = tuple(
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            for value in protocol_values
        )
        if expected_protocol is None:
            expected_protocol = protocol
        elif protocol != expected_protocol:
            raise ValueError(
                "Local analyses use different arms, case scopes, or run counts"
            )


def _common_analysis(analyses: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    return next(iter(analyses.values()), None)


def _derived_run_count(analysis: dict[str, Any] | None) -> int | None:
    if analysis is None:
        return None
    cases = analysis.get("selected_case_count")
    arms = analysis.get("selected_arms")
    planned = analysis.get("planned_trial_count")
    if (
        not isinstance(cases, int)
        or isinstance(cases, bool)
        or cases <= 0
        or not isinstance(arms, list)
        or not arms
        or not isinstance(planned, int)
        or isinstance(planned, bool)
    ):
        return None
    denominator = cases * len(arms)
    return planned // denominator if planned > 0 and planned % denominator == 0 else None


def _partial_notice(analyses: dict[str, dict[str, Any]]) -> list[str]:
    partial = [
        MODEL_LABELS[alias]
        for alias, analysis in analyses.items()
        if analysis.get("dataset_complete") is not True
    ]
    if not partial:
        return []
    return [
        "> **SMOKE/PARTIAL COHORTS:** "
        + ", ".join(partial)
        + ". These rows are smoke/partial observations, not full benchmark evidence.",
        "",
    ]


def _action_rows(analyses: dict[str, dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for alias in MODEL_ORDER:
        analysis = analyses.get(alias)
        if analysis is None:
            rows.append(
                f"| {MODEL_LABELS[alias]} | N/A | N/A | N/A | N/A | N/A | N/A |"
            )
            continue
        for arm in ("ACTION_ONLY", "INLINE_PROVENANCE"):
            item = _arm_metrics(analysis, arm)
            core = _core_arm_metrics(analysis, arm)
            rows.append(
                f"| {MODEL_LABELS[alias]} | {_arm_label(arm)} | "
                f"{_count(core.get('usable_trials'))} | "
                f"{_pct(item.get('clean_action_accuracy'))} | "
                f"{_pct(item.get('action_class_extraction_accuracy'))} | "
                f"{_pct(item.get('critical_argument_accuracy'))} | "
                f"{_pct(item.get('exact_attacker_target_adoption'))} |"
            )
    return rows


def _provenance_rows(analyses: dict[str, dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for alias in MODEL_ORDER:
        analysis = analyses.get(alias)
        if analysis is None:
            rows.append(
                f"| {MODEL_LABELS[alias]} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
            )
            continue
        for arm in ("INLINE_PROVENANCE", "ORACLE_PROVENANCE"):
            item = _arm_metrics(analysis, arm)
            rows.append(
                f"| {MODEL_LABELS[alias]} | {_arm_label(arm)} | "
                f"{_pct(item.get('evidence_text_match_accuracy'))} | "
                f"{_pct(item.get('evidence_region_accuracy'))} | "
                f"{_pct(item.get('source_type_classification_accuracy'))} | "
                f"{_pct(item.get('critical_argument_provenance_accuracy'))} | "
                f"{_pct(item.get('provenance_coverage'))} | "
                f"{_pct(item.get('missing_provenance_rate'))} | "
                f"{_pct(item.get('ambiguous_provenance_rate'))} | "
                f"{_pct(item.get('hallucinated_evidence_rate'))} |"
            )
    return rows


def _security_rows(analyses: dict[str, dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for alias in MODEL_ORDER:
        analysis = analyses.get(alias)
        if analysis is None:
            rows.append(
                f"| {MODEL_LABELS[alias]} | N/A | N/A | N/A | N/A | N/A |"
            )
            continue
        for arm in ("ACTION_ONLY", "INLINE_PROVENANCE", "ORACLE_PROVENANCE"):
            item = _arm_metrics(analysis, arm)
            rows.append(
                f"| {MODEL_LABELS[alias]} | {_arm_label(arm)} | "
                f"{_pct(item.get('automatic_unsafe_execution_rate'))} | "
                f"{_pct(item.get('thin_gate_escalation_recall'))} | "
                f"{_pct(item.get('false_escalation_rate'))} | "
                f"{_pct(item.get('trusted_user_preservation'))} |"
            )
    return rows


def _comparison_value(analysis: dict[str, Any], field: str) -> Any:
    metrics = analysis.get("metrics")
    core = metrics.get("core_phase2_metrics") if isinstance(metrics, dict) else None
    comparisons = core.get("comparisons") if isinstance(core, dict) else None
    return comparisons.get(field) if isinstance(comparisons, dict) else None


def build_aggregate_report(analyses: dict[str, dict[str, Any]]) -> str:
    """Build the common report without pooling or filling missing model results."""

    _validate_aggregate_analyses(analyses)
    common = _common_analysis(analyses)
    cohort = common.get("cohort", {}) if common is not None else {}
    selected_arms = common.get("selected_arms") if common is not None else None
    lock_prompts = cohort.get("prompt_versions")
    prompt_text = ", ".join(lock_prompts) if isinstance(lock_prompts, list) else "N/A"
    prompt_profile = _cell(cohort.get("zero_shot_prompt_version"))
    schema_transport = _cell(cohort.get("schema_transport_version"))
    paired_cohort = common.get("paired_cohort", {}) if common is not None else {}
    completed_models = [
        MODEL_LABELS[alias]
        for alias, analysis in analyses.items()
        if analysis.get("dataset_complete") is True
    ]

    lines = [
        "# LensGuard Phase 2.5 — Local Model Comparison",
        "",
        "> Results remain model-separated. `N/A` means no compatible observation was supplied; "
        "missing values are never inferred or imputed.",
        "",
        *_partial_notice(analyses),
        "## 1. Research Questions",
        "",
        "- RQ1: Can small local VLMs perform the frozen action, argument, and evidence task?",
        "- RQ2: How accurately do they map critical arguments to visual evidence?",
        "- RQ3: Does automatic local provenance remain useful to the Thin Trusted Gate?",
        "- RQ4: What gap remains between automatic local and Oracle provenance?",
        "- RQ5: What latency and GPU-memory cost does each local model impose?",
        "- RQ6: Is a 4B-class VLM security-useful, or is a stronger baseline required?",
        "",
        "## 2. Benchmark Freeze",
        "",
        "The generator rejects mixed lock digests, datasets, prompts, policies, arm selections, "
        "case scopes, or planned cohort sizes before comparison.",
        "",
        "| Frozen field | Compatible value |",
        "|---|---|",
        f"| Benchmark lock | {_cell(cohort.get('benchmark_lock_id'))} |",
        f"| Lock SHA-256 | {_cell(cohort.get('benchmark_lock_sha256'))} |",
        f"| Dataset | {_cell(cohort.get('dataset_version'))} |",
        f"| Zero-shot prompt cohort | {prompt_profile} |",
        f"| Schema transport | {schema_transport} |",
        f"| Semantic prompt versions | {_cell(prompt_text)} |",
        f"| Policy | {_cell(cohort.get('policy_version'))} |",
        f"| Selection scope | {_cell(cohort.get('selection_scope_id'))} |",
        "",
        "## 3. Hardware Environment",
        "",
        "| Model | GPU | Total VRAM | Driver | PyTorch / CUDA | Transformers / Python | OS |",
        "|---|---|---:|---|---|---|---|",
    ]
    for alias in MODEL_ORDER:
        analysis = analyses.get(alias)
        info = analysis.get("system_info") if analysis is not None else None
        info = info if isinstance(info, dict) else {}
        lines.append(
            f"| {MODEL_LABELS[alias]} | {_cell(info.get('gpu_model'))} | "
            f"{_gib(info.get('vram_total_bytes'))} | "
            f"{_cell(info.get('nvidia_driver_version'))} | "
            f"{_cell(info.get('torch_version'))} / "
            f"{_cell(info.get('cuda_runtime_visible_to_torch'))} | "
            f"{_cell(info.get('transformers_version'))} / "
            f"{_cell(info.get('python_version'))} | {_cell(info.get('os'))} |"
        )
    lines += [
        "",
        "The RTX 4090 is an evaluation/edge-proxy platform; it does not establish deployment on "
        "current glasses hardware.",
        "",
        "## 4. Models",
        "",
        "Primary comparison (all reported metrics are from `INLINE_PROVENANCE`; Oracle values are "
        "derived only from matched supplied cohorts):",
        "",
        "| Model | Params | Action Acc | Prov Acc | Halluc Prov | Unsafe Exec | p50 Lat | Peak VRAM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *[_table_row(alias, analyses.get(alias)) for alias in MODEL_ORDER],
        _oracle_row(analyses),
        "",
        "| Model | Repository / revision | dtype | Quantization | Attention | Cohort status |",
        "|---|---|---|---|---|---|",
    ]
    for alias in MODEL_ORDER:
        analysis = analyses.get(alias)
        row_cohort = analysis.get("cohort", {}) if analysis is not None else {}
        revision = row_cohort.get("model_revision") if isinstance(row_cohort, dict) else None
        repository_revision = f"{MODEL_REPOSITORIES[alias]} / {_cell(revision)}"
        lines.append(
            f"| {MODEL_LABELS[alias]} | {repository_revision} | "
            f"{_cell(_model_value(analysis, 'model_dtype') if analysis else None)} | "
            f"{_cell(_model_value(analysis, 'quantization') if analysis else None)} | "
            f"{_cell(_model_value(analysis, 'attention_backend') if analysis else None)} | "
            f"{_cohort_status(analysis)} |"
        )
    lines += [
        "",
        "## 5. Experimental Protocol",
        "",
        f"{prompt_profile}, schema transport {schema_transport}, batch size 1, sampling disabled, "
        "and one resident model at a time. The frozen deterministic evidence mapper and model-free "
        "Thin Trusted Gate are reused. No weights are trained. Any compatibility normalization is "
        "deterministic and explicitly counted; missing or malformed semantics are never guessed.",
        "",
        "| Protocol field | Value |",
        "|---|---|",
        f"| Selected arms | {_cell(', '.join(selected_arms) if isinstance(selected_arms, list) else None)} |",
        f"| Selected / benchmark cases | {_count(common.get('selected_case_count') if common else None)} / {_count(common.get('benchmark_case_count') if common else None)} |",
        f"| Runs (derived from planned cohort cardinality) | {_count(_derived_run_count(common))} |",
        f"| Planned trials per model | {_count(common.get('planned_trial_count') if common else None)} |",
        f"| Selection-scope complete | {_cell(common.get('selection_scope_complete') if common else None)} |",
        f"| Primary arms complete | {_cell(common.get('primary_arms_complete') if common else None)} |",
        f"| Full 243-trial arms complete | {_cell(common.get('full_arms_complete') if common else None)} |",
        f"| Action Only / Inline exact scope match | {_cell(paired_cohort.get('exact_scope_match') if isinstance(paired_cohort, dict) else None)} |",
        f"| Action Only / Inline paired trials | {_count(paired_cohort.get('paired_scope_count') if isinstance(paired_cohort, dict) else None)} |",
        f"| Oracle exact scope match | {_cell(paired_cohort.get('oracle_scope_match') if isinstance(paired_cohort, dict) else None)} |",
        "",
        "## 6. Action Extraction Results",
        "",
        "These rows retain the frozen Phase 2 completed-trial denominators. Section 9 includes "
        "independently scoreable actions from provenance-format failures.",
        "",
        "| Model | Arm | Usable | Clean action accuracy | Action-class accuracy | Critical-argument accuracy | Exact attacker adoption |",
        "|---|---|---:|---:|---:|---:|---:|",
        *_action_rows(analyses),
        "",
        "## 7. Provenance Results",
        "",
        "| Model | Arm | Text match | Region accuracy | Source accuracy | Argument provenance | Coverage | Missing | Ambiguous | Hallucinated |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *_provenance_rows(analyses),
        "",
        "## 8. Security Results",
        "",
        "| Model | Arm | Unsafe automatic execution | Gate escalation recall | False escalation | Trusted-user preservation |",
        "|---|---|---:|---:|---:|---:|",
        *_security_rows(analyses),
        "",
        "Source estimates remain predictions, not authority. Only mapped evidence and frozen "
        "policy metadata enter the model-free gate.",
        "",
        "## 9. Structured Output Reliability",
        "",
        "Raw structural-schema validity is shown separately from post-normalization acceptance. "
        "All rates include their assessed denominator; unassessed trials are not implicit failures, "
        "successes, or safe executions.",
        "",
        "| Model | Arm | Attempted | Completed | Errors | Runtime | Parse success | Raw structural schema valid | Normalized accepted | Normalizations | Contract semantic | Provenance semantic | Action correct | Critical argument correct | Unsafe / assessed attacks | Gate coverage |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for alias in MODEL_ORDER:
        analysis = analyses.get(alias)
        if analysis is None:
            lines.append(
                f"| {MODEL_LABELS[alias]} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | "
                "N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
            )
            continue
        for arm in ("ACTION_ONLY", "INLINE_PROVENANCE"):
            lines.append(
                _contract_row(
                    _arm_label(arm),
                    _contract_quality(analysis, arm),
                    model=MODEL_LABELS[alias],
                )
            )
        lines.append(
            _contract_row(
                "Overall", _contract_quality(analysis), model=MODEL_LABELS[alias]
            )
        )
    lines += [
        "",
        "A normalized output remains a raw-schema miss. Invalid structured output remains invalid; "
        "safe fence removal or JSON extraction does not authorize guessing a missing critical "
        "argument, evidence value, argument-to-evidence association, or coordinate.",
        "",
        "## 10. Latency",
        "",
        "| Model | Load | p50 inference | p95 inference | p50 end-to-end | p95 end-to-end | p50 mapper | p50 gate | p50 tokens/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for alias in MODEL_ORDER:
        analysis = analyses.get(alias)
        item = _efficiency(analysis) if analysis is not None else {}
        throughput = item.get("p50_tokens_per_second")
        lines.append(
            f"| {MODEL_LABELS[alias]} | {_ms(item.get('model_load_time_ms'))} | "
            f"{_ms(item.get('p50_inference_latency_ms'))} | "
            f"{_ms(item.get('p95_inference_latency_ms'))} | "
            f"{_ms(item.get('p50_end_to_end_latency_ms'))} | "
            f"{_ms(item.get('p95_end_to_end_latency_ms'))} | "
            f"{_ms(item.get('p50_evidence_mapper_latency_ms'))} | "
            f"{_ms(item.get('p50_thin_gate_latency_ms'))} | "
            f"{_decimal(throughput)} |"
        )
    lines += [
        "",
        "Latency is descriptive for the recorded runtime profile; smoke samples do not provide "
        "stable p95 estimates.",
        "",
        "## 11. VRAM",
        "",
        "| Model | Before inference (max) | Peak allocated | Peak reserved | Total device VRAM |",
        "|---|---:|---:|---:|---:|",
    ]
    for alias in MODEL_ORDER:
        analysis = analyses.get(alias)
        item = _efficiency(analysis) if analysis is not None else {}
        info = analysis.get("system_info") if analysis is not None else None
        info = info if isinstance(info, dict) else {}
        lines.append(
            f"| {MODEL_LABELS[alias]} | "
            f"{_gib(item.get('gpu_memory_allocated_before_inference_bytes_max'))} | "
            f"{_gib(item.get('gpu_peak_memory_allocated_bytes'))} | "
            f"{_gib(item.get('gpu_peak_memory_reserved_bytes'))} | "
            f"{_gib(info.get('vram_total_bytes'))} |"
        )
    lines += [
        "",
        "## 12. Model-by-Model Failure Cases",
        "",
        "The aggregate analysis retains failure counts and rates, not arbitrary reconstructed "
        "examples. Inspect each model's `report.md` and `raw_generations.jsonl` for recorded cases.",
        "",
        "| Model | Cohort | Unresolved errors | Runtime errors | Raw-schema failures | Normalizations | Failure categories | Inline missing provenance | Inline hallucinated evidence | Source |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for alias in MODEL_ORDER:
        analysis = analyses.get(alias)
        quality = _contract_quality(analysis) if analysis is not None else {}
        inline = _arm_metrics(analysis, "INLINE_PROVENANCE") if analysis is not None else {}
        raw_schema = quality.get("raw_schema")
        schema_failures = raw_schema.get("failures") if isinstance(raw_schema, dict) else None
        categories = quality.get("failure_category_counts")
        category_text = (
            ", ".join(f"{name}={count}" for name, count in sorted(categories.items()))
            if isinstance(categories, dict) and categories
            else None
        )
        lines.append(
            f"| {MODEL_LABELS[alias]} | {_cohort_status(analysis)} | "
            f"{_count(analysis.get('unresolved_error_trials') if analysis else None)} | "
            f"{_count(quality.get('runtime_error_trials'))} | "
            f"{_count(schema_failures)} | "
            f"{_count(quality.get('normalization_count'))} | "
            f"{_cell(category_text)} | "
            f"{_pct(inline.get('missing_provenance_rate'))} | "
            f"{_pct(inline.get('hallucinated_evidence_rate'))} | "
            f"{_cell(analysis.get('source') if analysis else None)} |"
        )
    lines += [
        "",
        "## 13. Oracle Gap",
        "",
        "Gaps are `INLINE_PROVENANCE - ORACLE_PROVENANCE`; negative and positive signs are "
        "preserved.",
        "",
        "| Model | Oracle usable | Oracle provenance accuracy | Oracle hallucinated evidence | Oracle unsafe execution | Inline−Oracle provenance gap | Inline−Oracle unsafe gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for alias in MODEL_ORDER:
        analysis = analyses.get(alias)
        oracle = _arm_metrics(analysis, "ORACLE_PROVENANCE") if analysis is not None else {}
        oracle_core = (
            _core_arm_metrics(analysis, "ORACLE_PROVENANCE")
            if analysis is not None
            else {}
        )
        lines.append(
            f"| {MODEL_LABELS[alias]} | {_count(oracle_core.get('usable_trials'))} | "
            f"{_pct(oracle.get('critical_argument_provenance_accuracy'))} | "
            f"{_pct(oracle.get('hallucinated_evidence_rate'))} | "
            f"{_pct(oracle.get('automatic_unsafe_execution_rate'))} | "
            f"{_pct(_comparison_value(analysis, 'inline_oracle_provenance_accuracy_gap') if analysis else None)} | "
            f"{_pct(_comparison_value(analysis, 'inline_oracle_unsafe_execution_gap') if analysis else None)} |"
        )
    lines += [
        "",
        "## 14. Local-vs-Cloud Comparison Placeholder",
        "",
        "Cloud files are optional and are not required to generate this report. They may be "
        "added later only after the same frozen lock and scientific protocol are verified.",
        "",
        "| Provider/model | Status |",
        "|---|---|",
        "| Gemini Flash | Awaiting compatible frozen Phase 2 results |",
        "| OpenAI model | Awaiting compatible frozen Phase 2 results |",
        "",
        "## 15. Limitations",
        "",
        "- Smoke/partial cohorts measure compatibility, not full-benchmark scientific outcomes.",
        "- Evidence is observable attribution, not chain of thought or causal proof.",
        "- A source-type estimate does not establish authority.",
        "- Synthetic scenes and annotated regions do not reproduce physical wearable input.",
        "- Mapper timing excludes production OCR, segmentation, and region acquisition.",
        "- Missing confidence remains invalid under the frozen schema; confidence is not invented.",
        "- RTX 4090 measurements do not prove deployment on current glasses hardware.",
        "",
        "## 16. Phase 2.5 Go / No-Go",
        "",
    ]
    if completed_models:
        lines.append(
            "Complete cohorts available for evidence review: "
            + ", ".join(completed_models)
            + ". This report intentionally does not collapse the evidence into an arbitrary score."
        )
    else:
        lines.append(
            "No GO/NO-GO conclusion is supported yet: no supplied model has a complete benchmark "
            "cohort. Any populated values above are smoke/partial compatibility observations."
        )
    lines += [
        "",
        "GO evidence requires useful grounded provenance, a substantial Thin Gate reduction in "
        "unsafe execution, high clean-action utility, manageable hallucination and parse failure, "
        "usable event-driven latency, and comfortable VRAM headroom. Pivot evidence includes low "
        "coverage, frequent hallucination, unreliable structure, collapsed action accuracy, high "
        "false escalation, impractical latency, or a large automatic-to-Oracle gap.",
        "",
        "Phase 2.6 fine-tuning is justified only if the preserved zero-shot baseline is inadequate "
        "but exhibits learnable provenance signal; the physical holdout must never enter training. "
        "Phase 3 physical experiments are justified when at least one complete local cohort shows "
        "grounded provenance across action families, meaningful gate security gain, preserved "
        "utility, tolerable false escalation, usable runtime, and an acceptable Oracle gap.",
    ]
    return "\n".join(lines) + "\n"


def load_analyses(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Analysis must be a JSON object: {path}")
        model_id = payload.get("cohort", {}).get("model_id")
        alias = next(
            (key for key, repository in MODEL_REPOSITORIES.items() if model_id == repository),
            path.parent.name,
        )
        if alias not in MODEL_ORDER:
            raise ValueError(f"Unsupported local-model analysis {model_id!r}: {path}")
        if alias in result:
            raise ValueError(f"Duplicate analysis for {alias}")
        result[alias] = payload
    _validate_aggregate_analyses(result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, action="append", default=[])
    parser.add_argument(
        "--output", type=Path, default=Path("results_phase2_5/report_local_models.md")
    )
    parser.add_argument("--attempts", type=Path)
    parser.add_argument("--per-model-output", type=Path)
    parser.add_argument("--system-info", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    analyses = load_analyses(args.analysis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_aggregate_report(analyses), encoding="utf-8")
    if args.attempts is not None:
        if len(analyses) != 1 or args.per_model_output is None:
            raise ValueError(
                "--attempts requires exactly one --analysis and --per-model-output"
            )
        analysis = next(iter(analyses.values()))
        system_info = None
        if args.system_info is not None and args.system_info.is_file():
            system_info = json.loads(args.system_info.read_text(encoding="utf-8"))
        report = build_local_model_report(
            read_jsonl(args.attempts),
            analysis,
            source_path=args.attempts,
            system_info=system_info,
        )
        args.per_model_output.parent.mkdir(parents=True, exist_ok=True)
        args.per_model_output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()


__all__ = [
    "MODEL_LABELS",
    "MODEL_ORDER",
    "MODEL_REPOSITORIES",
    "REPORT_VERSION",
    "build_aggregate_report",
    "build_local_model_report",
    "load_analyses",
]
