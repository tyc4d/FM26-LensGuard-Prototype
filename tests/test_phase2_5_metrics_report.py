from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from analyze_phase2_5 import _completion_context, analyze_phase2_5
from benchmark_phase2 import (
    _experiment_config_id,
    _selection_scope_id,
    load_phase2_dataset,
    run_phase2_trial,
)
from firewall.thin_gate import load_action_registry, load_thin_gate_policy
from generate_report_phase2_5 import build_aggregate_report, build_local_model_report
from metrics_phase2_5 import compute_phase2_5_metrics
from phase2_schema import Phase2Arm
from providers.mock_phase2 import MockPhase2Provider
from result_store import append_jsonl
from result_store_phase2_5 import validate_phase2_5_attempts


ROOT = Path(__file__).resolve().parents[1]


def _local_row(tmp_path: Path) -> dict:
    dataset, scenarios = load_phase2_dataset(ROOT / "dataset_phase2/metadata.json")
    scenario = next(
        row
        for row in scenarios
        if row["condition"] == "CLEAN_TRUSTED" and row["action_family"] == "CALL"
    )
    provider = MockPhase2Provider()
    registry = load_action_registry(ROOT / "config/action_registry.yaml")
    policy = load_thin_gate_policy(ROOT / "config/policy_phase2.yaml")
    scope = _selection_scope_id([scenario])
    experiment = _experiment_config_id(
        provider,
        seed=0,
        generation_seed=0,
        runs=1,
        selection_scope_id=scope,
    )
    row = run_phase2_trial(
        scenario=scenario,
        arm=Phase2Arm.INLINE_PROVENANCE,
        run=1,
        provider_name="local",
        provider=provider,
        dataset_version=dataset["dataset_version"],
        registry=registry,
        policy=policy,
        results_dir=tmp_path,
        selection_scope_id=scope,
        experiment_config_id=experiment,
        planned_trial_count=1,
        attempt_index=1,
        selection_seed=0,
        generation_seed=0,
        request_delay=0,
    )
    local_inference = {
        "structured_output_valid": True,
        "preprocessing_latency_ms": 2.0,
        "generation_latency_ms": 3.0,
        "inference_latency_ms": 5.0,
        "input_token_count": row["input_tokens"],
        "output_token_count": row["output_tokens"],
        "generated_tokens": row["output_tokens"],
        "tokens_per_second": row["output_tokens"] / 0.003,
        "gpu_memory_allocated_before_inference_bytes": 100,
        "gpu_peak_memory_allocated_bytes": 200,
        "gpu_peak_memory_reserved_bytes": 300,
        "image_width": 1200,
        "image_height": 760,
        "model_load_time_ms": 10.0,
        "parameter_count": 4_300_079_472,
    }
    row["model_call_records"][0]["response_metadata"]["local_inference"] = local_inference
    row.update(
        model="google/gemma-3-4b-it",
        model_id="google/gemma-3-4b-it",
        model_revision="revision-test",
        processor_revision="revision-test",
        zero_shot_prompt_version="ZERO_SHOT_V2",
        schema_transport_version="phase2.5-local-json-schema-transport-v2",
        benchmark_lock_id="lensguard-phase2-frozen-v1",
        benchmark_lock_sha256="a" * 64,
        selected_case_count=1,
        benchmark_case_count=1,
        structured_output_valid=True,
        parse_success=True,
        schema_valid=True,
        normalization_applied=False,
        normalization_method=None,
        normalized_schema_valid=True,
        contract_semantically_valid=True,
        provenance_semantically_valid=True,
        action_candidate={
            "action": row["proposed_action"],
            "arguments": row["proposed_arguments"],
        },
        action_correct=row["action_extraction_correct"],
        critical_argument_correct=row["critical_argument_extraction_correct"],
        unsafe_execution=row["unsafe_automatic_execution"],
        failure_category=None,
        failure_categories=[],
        model_load_time_ms=10.0,
        preprocessing_latency_ms=2.0,
        generation_latency_ms=3.0,
        inference_latency_ms=5.0,
        evidence_mapper_latency_ms=row["mapping_latency_ms"],
        input_token_count=row["input_tokens"],
        output_token_count=row["output_tokens"],
        generated_tokens=row["output_tokens"],
        tokens_per_second=row["output_tokens"] / 0.003,
        gpu_memory_allocated_before_inference_bytes=100,
        gpu_peak_memory_allocated_bytes=200,
        gpu_peak_memory_reserved_bytes=300,
        model_dtype="bf16",
        quantization="none",
        attention_backend="sdpa",
        image_width=1200,
        image_height=760,
        parameter_count=4_300_079_472,
    )
    return row


def test_phase2_5_metrics_reuse_core_and_add_local_instrumentation(tmp_path: Path) -> None:
    row = _local_row(tmp_path)
    validate_phase2_5_attempts([row])

    metrics = compute_phase2_5_metrics([row])

    inline = metrics["by_arm"]["INLINE_PROVENANCE"]
    assert inline["action_class_extraction_accuracy"] == 1.0
    assert inline["critical_argument_accuracy"] == 1.0
    assert inline["structured_output"]["structured_output_parse_success_rate"] == 1.0
    assert inline["contract_quality"]["parse"]["successes"] == 1
    assert inline["contract_quality"]["raw_schema"]["successes"] == 1
    assert inline["contract_quality"]["normalized_schema"]["successes"] == 1
    assert inline["contract_quality"]["normalization_count"] == 0
    assert inline["efficiency"]["p50_inference_latency_ms"] == 5.0
    assert inline["efficiency"]["gpu_peak_memory_allocated_bytes"] == 200
    assert metrics["model_metadata_values"]["parameter_count"] == [4_300_079_472]


def _diagnostic_row(scene_id: str, **updates: object) -> dict:
    row = {
        "scene_id": scene_id,
        "condition": "AUTHORITY_IMPERSONATION",
        "architecture_arm": "INLINE_PROVENANCE",
        "provider": "local",
        "model": "test/model",
        "model_id": "test/model",
        "model_revision": "revision-test",
        "run": 1,
        "prompt_version": "phase2-inline-provenance-v2",
        "dataset_version": "dataset-test",
        "policy_version": "policy-test",
        "status": "completed",
        "is_attack": True,
        "parse_success": True,
        "schema_valid": True,
        "normalization_applied": False,
        "normalization_method": None,
        "normalized_schema_valid": True,
        "contract_semantically_valid": True,
        "provenance_semantically_valid": True,
        "action_correct": True,
        "critical_argument_correct": True,
        "unsafe_execution": False,
        "failure_category": None,
        "failure_categories": [],
    }
    row.update(updates)
    return row


def test_contract_quality_keeps_format_semantics_and_outcomes_separate() -> None:
    rows = [
        # Valid canonical JSON can still contain semantically invalid provenance.
        _diagnostic_row(
            "canonical-semantic-failure",
            provenance_semantically_valid=False,
            unsafe_execution=True,
            failure_category="provenance_semantic_failure",
            failure_categories=["provenance_semantic_failure"],
        ),
        # Compatibility normalization is accepted but remains a raw-schema miss.
        _diagnostic_row(
            "normalized-list",
            schema_valid=False,
            normalization_applied=True,
            normalization_method="argument_evidence_list_to_object",
            action_correct=False,
            critical_argument_correct=False,
            failure_category="schema_mismatch",
            failure_categories=["schema_mismatch", "action_prediction_failure"],
        ),
        # The action can be scored independently even when canonical validation fails.
        _diagnostic_row(
            "unrepairable-schema",
            status="error",
            schema_valid=False,
            normalized_schema_valid=False,
            contract_semantically_valid=False,
            provenance_semantically_valid=None,
            unsafe_execution=None,
            failure_category="schema_mismatch",
            failure_categories=["schema_mismatch"],
        ),
        _diagnostic_row(
            "malformed-json",
            status="error",
            parse_success=False,
            schema_valid=False,
            normalized_schema_valid=False,
            contract_semantically_valid=False,
            provenance_semantically_valid=None,
            action_correct=None,
            critical_argument_correct=None,
            unsafe_execution=None,
            failure_category="malformed_json",
            failure_categories=["malformed_json"],
        ),
        _diagnostic_row(
            "runtime-error",
            status="error",
            parse_success=False,
            schema_valid=False,
            normalized_schema_valid=False,
            contract_semantically_valid=False,
            provenance_semantically_valid=None,
            action_correct=None,
            critical_argument_correct=None,
            unsafe_execution=None,
            failure_category="inference_runtime",
            failure_categories=["inference_runtime"],
        ),
    ]

    quality = compute_phase2_5_metrics(rows)["contract_quality"]

    assert quality["attempted_trials"] == 5
    assert quality["completed_trials"] == 2
    assert quality["unresolved_error_trials"] == 3
    assert quality["runtime_error_trials"] == 1
    assert quality["parse"] == {
        "assessed_trials": 4,
        "successes": 3,
        "failures": 1,
        "unassessed_trials": 1,
        "rate": 0.75,
        "assessment_coverage": 0.8,
    }
    assert quality["raw_schema"]["rate"] == pytest.approx(1 / 3)
    assert quality["raw_schema"]["assessed_trials"] == 3
    assert quality["normalized_schema"]["rate"] == pytest.approx(2 / 3)
    assert quality["contract_semantic"]["rate"] == 1.0
    assert quality["provenance_semantic"]["rate"] == 0.5
    assert quality["provenance_semantic"]["assessment_coverage"] == 0.4
    assert quality["action_correctness"]["rate"] == pytest.approx(2 / 3)
    assert quality["action_correctness"]["unassessed_trials"] == 2
    assert quality["critical_argument_correctness"]["rate"] == pytest.approx(2 / 3)
    assert quality["normalization_count"] == 1
    assert quality["normalization_method_counts"] == {
        "argument_evidence_list_to_object": 1
    }
    assert quality["unsafe_executions"] == 1
    assert quality["gate_assessed_attack_trials"] == 2
    assert quality["unsafe_execution_rate"] == 0.5
    assert quality["unsafe_execution_assessment_coverage"] == 0.4
    assert quality["failure_category_counts"] == {
        "action_prediction_failure": 1,
        "inference_runtime": 1,
        "malformed_json": 1,
        "provenance_semantic_failure": 1,
        "schema_mismatch": 2,
    }


def test_phase2_5_analysis_and_reports_are_model_separated(tmp_path: Path) -> None:
    row = _local_row(tmp_path)
    raw = tmp_path / "raw_generations.jsonl"
    append_jsonl(raw, row)
    system_info = {
        "gpu_model": "Test GPU",
        "vram_total_bytes": 24 * 1024**3,
        "nvidia_driver_version": "test-driver",
        "torch_version": "test-torch",
        "cuda_runtime_visible_to_torch": "test-cuda",
        "transformers_version": "test-transformers",
        "python_version": "3.12",
        "os": "test-os",
    }
    system_path = tmp_path / "system_info.json"
    system_path.write_text(json.dumps(system_info), encoding="utf-8")

    analysis = analyze_phase2_5(
        raw,
        tmp_path / "analysis.json",
        None,
        system_info_path=system_path,
    )
    report = build_local_model_report(
        [row], analysis, source_path=raw, system_info=system_info
    )
    aggregate = build_aggregate_report({"gemma3-4b": analysis})

    assert analysis["selection_scope_complete"] is True
    assert analysis["dataset_complete"] is False
    assert analysis["primary_arms_complete"] is False
    assert analysis["cohort"]["zero_shot_prompt_version"] == "ZERO_SHOT_V2"
    assert (
        analysis["cohort"]["schema_transport_version"]
        == "phase2.5-local-json-schema-transport-v2"
    )
    assert "## 9. Structured Output Reliability" in report
    assert "Raw structural schema valid" in report
    assert "Normalized accepted" in report
    assert "Action Only / Inline attempted scope identical" in report
    assert "## 16. Phase 2.5 Go / No-Go" in report
    assert "RTX 4090 is an evaluation and edge-proxy platform" in report
    assert "Gemma 3 4B" in aggregate
    assert "Provenance semantic" in aggregate
    assert "Selection scope" in aggregate
    assert "Qwen3-VL 8B | N/A" in aggregate
    assert "smoke/partial observations" in aggregate

    incompatible = deepcopy(analysis)
    incompatible["cohort"]["model_id"] = "Qwen/Qwen3-VL-8B-Instruct"
    incompatible["cohort"]["dataset_version"] = "different-dataset"
    with pytest.raises(ValueError, match="different benchmark locks, datasets"):
        build_aggregate_report(
            {"gemma3-4b": analysis, "qwen3vl-8b": incompatible}
        )


def test_full_completion_requires_full_case_scope_and_all_primary_arms() -> None:
    attempts = [
        {
            "scene_id": f"case-{case_index}",
            "condition": "CLEAN_TRUSTED",
            "run": 1,
            "architecture_arm": arm,
            "planned_trial_count": 243,
            "selected_case_count": 81,
            "benchmark_case_count": 81,
        }
        for case_index in range(81)
        for arm in ("ACTION_ONLY", "INLINE_PROVENANCE", "ORACLE_PROVENANCE")
    ]
    metrics = {
        "core_phase2_metrics": {
            "trial_counts": {"completed": 243, "unresolved_errors": 0}
        }
    }

    context = _completion_context(attempts, metrics)

    assert context["selection_scope_complete"] is True
    assert context["primary_arms_complete"] is True
    assert context["full_arms_complete"] is True
    assert context["paired_comparison_complete"] is True
    assert context["dataset_complete"] is True

    attempts[0]["benchmark_case_count"] = 82
    assert _completion_context(attempts, metrics)["dataset_complete"] is False


def test_aggregate_report_has_all_required_sections_in_order(tmp_path: Path) -> None:
    row = _local_row(tmp_path)
    raw = tmp_path / "raw_generations.jsonl"
    append_jsonl(raw, row)
    analysis = analyze_phase2_5(raw, tmp_path / "analysis.json", None)

    report = build_aggregate_report({"gemma3-4b": analysis})
    headings = [
        "## 1. Research Questions",
        "## 2. Benchmark Freeze",
        "## 3. Hardware Environment",
        "## 4. Models",
        "## 5. Experimental Protocol",
        "## 6. Action Extraction Results",
        "## 7. Provenance Results",
        "## 8. Security Results",
        "## 9. Structured Output Reliability",
        "## 10. Latency",
        "## 11. VRAM",
        "## 12. Model-by-Model Failure Cases",
        "## 13. Oracle Gap",
        "## 14. Local-vs-Cloud Comparison Placeholder",
        "## 15. Limitations",
        "## 16. Phase 2.5 Go / No-Go",
    ]

    positions = [report.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert all(report.count(heading) == 1 for heading in headings)
    assert "**SMOKE/PARTIAL COHORTS:** Gemma 3 4B" in report
    assert "No GO/NO-GO conclusion is supported yet" in report


def test_empty_aggregate_report_never_imputes_unobserved_results() -> None:
    report = build_aggregate_report({})

    assert "missing values are never inferred or imputed" in report
    assert (
        "| Gemma 3 4B | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
        in report
    )
    assert (
        "| Oracle (matched cohorts) | N/A | - | N/A | N/A | N/A | - | - |"
        in report
    )
    assert "100.0%" not in report


def test_aggregate_report_rejects_incompatible_benchmark_locks(
    tmp_path: Path,
) -> None:
    row = _local_row(tmp_path)
    raw = tmp_path / "raw_generations.jsonl"
    append_jsonl(raw, row)
    gemma = analyze_phase2_5(raw, tmp_path / "analysis.json", None)
    qwen = deepcopy(gemma)
    qwen["cohort"]["model_id"] = "Qwen/Qwen3-VL-8B-Instruct"
    qwen["cohort"]["benchmark_lock_sha256"] = "b" * 64

    with pytest.raises(ValueError, match="different benchmark locks"):
        build_aggregate_report({"gemma3-4b": gemma, "qwen3vl-8b": qwen})


@pytest.mark.parametrize(
    ("field", "different_value"),
    [
        ("selection_scope_id", "different-selection-scope"),
        ("schema_transport_version", "different-schema-transport"),
    ],
)
def test_aggregate_report_rejects_incompatible_contract_cohorts(
    tmp_path: Path, field: str, different_value: str
) -> None:
    row = _local_row(tmp_path)
    raw = tmp_path / "raw_generations.jsonl"
    append_jsonl(raw, row)
    gemma = analyze_phase2_5(raw, tmp_path / "analysis.json", None)
    qwen = deepcopy(gemma)
    qwen["cohort"]["model_id"] = "Qwen/Qwen3-VL-8B-Instruct"
    qwen["cohort"][field] = different_value

    with pytest.raises(ValueError, match="schema transports, policies, or selection scopes"):
        build_aggregate_report({"gemma3-4b": gemma, "qwen3vl-8b": qwen})
