#!/usr/bin/env python3
"""Run the frozen LensGuard Phase 2 benchmark with one local VLM at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark_phase2 import (
    _base_result,
    _configure_logging,
    _expected_action,
    _prompt_version,
    _selection_scope_id,
    load_phase2_dataset,
    parse_arms,
    run_phase2_trial,
    select_phase2_scenarios,
)
from firewall.action_normalizer import critical_arguments_for, normalize_action
from firewall.thin_gate import load_action_registry, load_thin_gate_policy
from generate_report_phase2_5 import (
    MODEL_ORDER,
    build_aggregate_report,
    build_local_model_report,
)
from phase2_benchmark_lock import (
    DEFAULT_LOCK_PATH,
    PROJECT_ROOT,
    Phase2BenchmarkLockError,
    verify_phase2_benchmark_lock,
)
from phase2_schema import ActionOnlyOutput, Phase2Arm, Phase2Operation
from providers import ProviderError
from providers.local import (
    LOCAL_ATTENTION_BACKEND,
    LOCAL_DTYPE,
    LOCAL_MODEL_PROVIDERS,
    LOCAL_MODEL_REPOSITORIES,
    LOCAL_QUANTIZATION,
    LOCAL_SCHEMA_TRANSPORT_VERSION,
    ZERO_SHOT_V2,
    BaseLocalVLMProvider,
    create_local_provider,
)
from result_store import read_jsonl
from result_store_phase2_5 import (
    assert_phase2_5_resume_compatible,
    completed_phase2_5_identities,
    next_phase2_5_attempt_index,
    persist_phase2_5_attempt,
    phase2_5_trial_identity,
    validate_phase2_5_attempts,
)
from system_info_phase2_5 import (
    collect_phase2_5_system_info,
    huggingface_cache_preflight,
    write_phase2_5_system_info,
)


RUNNER_VERSION = "phase2.5-local-runner-v2"
DEFAULT_MODEL_REVISIONS = {
    "gemma3-4b": "093f9f388b31de276ce2de164bdc2081324b9767",
    "qwen3vl-8b": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    "minicpm-v4.5": "daef484c35ec93210ec93c5e901f8f3e9b78ee34",
}
ESTIMATED_REPOSITORY_BYTES = {
    "gemma3-4b": 8_635_015_168,
    "qwen3vl-8b": 17_534_339_512,
    "minicpm-v4.5": 17_403_328_052,
}
DOWNLOAD_DISK_RESERVE_BYTES = 10 * 1024**3
FROZEN_BENCHMARK_INPUTS = {
    "dataset": PROJECT_ROOT / "dataset_phase2/metadata.json",
    "registry": PROJECT_ROOT / "config/action_registry.yaml",
    "policy": PROJECT_ROOT / "config/policy_phase2.yaml",
}


def _assert_frozen_benchmark_inputs(args: argparse.Namespace) -> None:
    """Bind runner inputs to the artifacts whose bytes the lock verified.

    Merely verifying the repository lock is insufficient if a caller can then
    point the evaluator at a different file carrying the same declared version.
    Resolve symlinks/relative paths, but reject any other input before model
    construction, downloads, or inference.
    """

    for argument, expected_path in FROZEN_BENCHMARK_INPUTS.items():
        supplied = Path(getattr(args, argument)).resolve()
        expected = expected_path.resolve()
        if supplied != expected:
            raise Phase2BenchmarkLockError(
                f"--{argument} must use the frozen Phase 2 artifact {expected}; "
                f"got {supplied}"
            )


def _stable_provider_config(provider: BaseLocalVLMProvider) -> dict[str, Any]:
    config = dict(provider.experiment_config)
    # Load duration and discovered parameter count are measurements/metadata,
    # not experiment selectors. Keeping either inside an identity hash would
    # make the pre-load and post-load configurations disagree.
    config.pop("model_load_time_ms", None)
    config.pop("parameter_count", None)
    return config


def _experiment_config_id(
    provider: BaseLocalVLMProvider,
    *,
    arms: Sequence[Phase2Arm],
    seed: int,
    generation_seed: int,
    runs: int,
    selection_scope_id: str,
    benchmark_lock_sha256: str,
) -> str:
    payload = {
        "runner_version": RUNNER_VERSION,
        "provider_config": _stable_provider_config(provider),
        "arms": [arm.value for arm in arms],
        "selection_seed": seed,
        "generation_seed": generation_seed,
        "runs": runs,
        "selection_scope_id": selection_scope_id,
        "benchmark_lock_sha256": benchmark_lock_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _cuda_free_bytes() -> tuple[int | None, int | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None
        free, total = torch.cuda.mem_get_info()
        return int(free), int(total)
    except Exception:
        return None, None


def _gib(value: Any) -> str:
    return f"{value / (1024**3):.2f} GiB" if isinstance(value, (int, float)) else "unavailable"


def _preflight_summary(
    *,
    args: argparse.Namespace,
    repository_id: str,
    revision: str,
    dataset_version: str,
    prompt_versions: Sequence[str],
    policy_version: str,
    cases: int,
    arms: Sequence[Phase2Arm],
    expected_inferences: int,
    cache: Mapping[str, Any],
    attention_backend: str,
) -> None:
    free_vram, total_vram = _cuda_free_bytes()
    print("LensGuard Phase 2.5 — LOCAL VLM PRE-RUN SUMMARY")
    print(f"Model: {args.model}")
    print(f"Model repository: {repository_id}")
    print(f"Model revision: {revision}")
    print(f"Dataset version: {dataset_version}")
    print(f"Prompt profile: {ZERO_SHOT_V2}")
    print(f"Semantic prompt version(s): {','.join(prompt_versions)}")
    print(f"Policy version: {policy_version}")
    print(f"Cases: {cases}")
    print(f"Arms: {','.join(arm.value for arm in arms)}")
    print(f"Runs: {args.runs}")
    print(f"Expected inference count: {expected_inferences}")
    print(f"dtype: {LOCAL_DTYPE}")
    print(f"quantization: {LOCAL_QUANTIZATION}")
    print(f"attention backend: {attention_backend}")
    print(f"device: {args.device}")
    print(f"available VRAM: {_gib(free_vram)} / {_gib(total_vram)} total")
    print(f"Hugging Face cache: {cache['model_cache_path']}")
    print(f"Requested revision cached: {cache['cached_revision'] or 'no'}")
    print(f"Current model cache size: {_gib(cache['model_cache_bytes'])}")
    print(f"Available disk space: {_gib(cache['disk_free_bytes'])}")
    if cache.get("estimated_download_bytes") is not None:
        print(f"Known/estimated repository size: {_gib(cache['estimated_download_bytes'])}")
    print(f"Disk-space preflight: {cache.get('sufficient_free_space')}")


def _local_call_metrics(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    records = row.get("model_call_records")
    if not isinstance(records, list):
        return result
    for record in records:
        if not isinstance(record, Mapping):
            continue
        metadata = record.get("response_metadata")
        if not isinstance(metadata, Mapping):
            continue
        local = metadata.get("local_inference")
        if isinstance(local, Mapping):
            result.append(dict(local))
    return result


_OUTPUT_CONTRACT_FIELDS = (
    "parse_success",
    "schema_valid",
    "normalization_applied",
    "normalization_method",
    "normalized_schema_valid",
    "contract_semantically_valid",
    "failure_category",
    "action_candidate",
)
_FAILURE_CATEGORY_ORDER = (
    "inference_runtime",
    "malformed_json",
    "schema_mismatch",
    "provenance_contract_semantic_failure",
    "provenance_semantic_failure",
    "action_prediction_failure",
    "critical_argument_prediction_failure",
)


def _trial_output_contract(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the provider's diagnostics for the operation represented by a row."""

    preferred_operation = {
        "INLINE_PROVENANCE": Phase2Operation.INLINE_PROVENANCE.value,
        "TWO_PASS_PROVENANCE": Phase2Operation.TWO_PASS_EVIDENCE.value,
    }.get(str(row.get("architecture_arm")), Phase2Operation.ACTION_ONLY.value)
    records = row.get("model_call_records")
    candidates: list[Mapping[str, Any]] = []
    if isinstance(records, list):
        candidates = [
            record
            for record in records
            if isinstance(record, Mapping)
            and str(record.get("operation")) == preferred_operation
        ]
    for record in reversed(candidates):
        metadata = record.get("response_metadata")
        if not isinstance(metadata, Mapping):
            continue
        local = metadata.get("local_inference")
        nested = local.get("output_contract") if isinstance(local, Mapping) else None
        contract = nested if isinstance(nested, Mapping) else metadata.get("output_contract")
        if isinstance(contract, Mapping):
            return {field: contract.get(field) for field in _OUTPUT_CONTRACT_FIELDS}
    # Fail closed for a call that ended before the provider could parse any output.
    return {
        "parse_success": False,
        "schema_valid": False,
        "normalization_applied": False,
        "normalization_method": None,
        "normalized_schema_valid": False,
        "contract_semantically_valid": False,
        "failure_category": "inference_runtime",
        "action_candidate": None,
    }


def _action_candidate_diagnostics(
    candidate: Any, row: Mapping[str, Any]
) -> tuple[bool | None, bool | None, dict[str, Any] | None]:
    """Score a separately validated action even if Inline evidence was rejected."""

    if not isinstance(candidate, Mapping):
        return None, None, None
    try:
        action_output = ActionOnlyOutput.model_validate(candidate)
        proposed = normalize_action(action_output.as_proposed_action())
        expected = _expected_action(row)
    except (TypeError, ValueError):
        return None, None, None
    action_correct = proposed.action is expected.action
    critical_correct = bool(
        action_correct
        and critical_arguments_for(proposed) == critical_arguments_for(expected)
    )
    serialized = {
        "action": proposed.action.value,
        "arguments": proposed.arguments.model_dump(mode="json", exclude_none=True),
    }
    return action_correct, critical_correct, serialized


def _provenance_semantic_validity(row: Mapping[str, Any]) -> bool | None:
    """Reduce existing per-argument audits without conflating action correctness."""

    if row.get("architecture_arm") == "ACTION_ONLY":
        return None
    evaluations = row.get("provenance_evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        return None
    candidate = row.get("action_candidate")
    candidate_arguments = (
        candidate.get("arguments") if isinstance(candidate, Mapping) else None
    )
    if not isinstance(candidate_arguments, Mapping):
        candidate_arguments = row.get("proposed_arguments")
    if not isinstance(candidate_arguments, Mapping):
        return None
    expected_arguments = {
        str(name) for name, value in candidate_arguments.items() if value is not None
    }
    if not expected_arguments:
        return None

    by_argument = {
        str(evaluation.get("argument_name")): evaluation
        for evaluation in evaluations
        if isinstance(evaluation, Mapping)
        and isinstance(evaluation.get("argument_name"), str)
    }
    if not expected_arguments.issubset(by_argument):
        return False
    correctness: list[bool | None] = []
    for argument_name in sorted(expected_arguments):
        evaluation = by_argument[argument_name]
        status = str(evaluation.get("evidence_status", "")).casefold()
        if status and status != "matched":
            return False
        items = evaluation.get("reported_evidence_items")
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, Mapping):
                    return False
                item_status = str(item.get("evidence_status", "")).casefold()
                if item.get("supports_argument") is False or item_status in {
                    "hallucinated",
                    "unsupported",
                }:
                    return False
        value = evaluation.get("provenance_correct")
        correctness.append(value if isinstance(value, bool) else None)
    if any(value is False for value in correctness):
        return False
    if correctness and all(value is True for value in correctness):
        return True
    return None


def _failure_categories(
    *,
    provider_category: Any,
    status: Any,
    parse_success: bool,
    normalized_schema_valid: bool,
    contract_semantically_valid: bool,
    provenance_semantically_valid: bool | None,
    action_correct: bool | None,
    critical_argument_correct: bool | None,
) -> list[str]:
    aliases = {"contract_semantic_mismatch": "provenance_contract_semantic_failure"}
    observed: set[str] = set()
    if isinstance(provider_category, str) and provider_category:
        observed.add(aliases.get(provider_category, provider_category))
    if status == "error" and not observed:
        observed.add("inference_runtime")
    if parse_success and normalized_schema_valid and not contract_semantically_valid:
        observed.add("provenance_contract_semantic_failure")
    if provenance_semantically_valid is False:
        observed.add("provenance_semantic_failure")
    if action_correct is False:
        observed.add("action_prediction_failure")
    if critical_argument_correct is False:
        observed.add("critical_argument_prediction_failure")
    return [category for category in _FAILURE_CATEGORY_ORDER if category in observed]


def _numbers(entries: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    return [
        float(value)
        for entry in entries
        if isinstance((value := entry.get(key)), (int, float)) and not isinstance(value, bool)
    ]


def _sum_if_complete(entries: Sequence[Mapping[str, Any]], key: str) -> int | None:
    values = [entry.get(key) for entry in entries]
    if not values or any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        return None
    return sum(values)


def _single_dimension(entries: Sequence[Mapping[str, Any]], key: str) -> int | None:
    values = {
        int(value)
        for entry in entries
        if isinstance((value := entry.get(key)), int) and not isinstance(value, bool)
    }
    return next(iter(values)) if len(values) == 1 else None


def _augment_trial(
    row: dict[str, Any],
    *,
    provider: BaseLocalVLMProvider,
    lock: Mapping[str, Any],
    selected_case_count: int,
    benchmark_case_count: int,
) -> dict[str, Any]:
    entries = _local_call_metrics(row)
    stable_config = _stable_provider_config(provider)
    attention_backend = str(stable_config.get("attention_backend", LOCAL_ATTENTION_BACKEND))
    dtype = str(stable_config.get("dtype", LOCAL_DTYPE))
    quantization = str(stable_config.get("quantization", LOCAL_QUANTIZATION))

    def total(key: str) -> float | None:
        values = _numbers(entries, key)
        return sum(values) if values else None

    def maximum(key: str) -> int | None:
        values = _numbers(entries, key)
        return int(max(values)) if values else None

    output_tokens = _sum_if_complete(entries, "output_token_count")
    generated_tokens = _sum_if_complete(entries, "generated_tokens")
    generation_ms = total("generation_latency_ms")
    contract = _trial_output_contract(row)
    parse_success = bool(contract["parse_success"])
    schema_valid = bool(contract["schema_valid"])
    normalization_applied = bool(contract["normalization_applied"])
    normalized_schema_valid = bool(contract["normalized_schema_valid"])
    contract_semantically_valid = bool(contract["contract_semantically_valid"])
    normalization_method = contract.get("normalization_method")
    action_correct, critical_argument_correct, action_candidate = (
        _action_candidate_diagnostics(contract.get("action_candidate"), row)
    )
    # These frozen evaluator fields exist only after the full Inline contract
    # succeeds. The independently validated candidate above remains scoreable
    # when provenance fails before Thin Gate execution.
    if action_correct is None and isinstance(row.get("action_extraction_correct"), bool):
        action_correct = row["action_extraction_correct"]
    if critical_argument_correct is None and isinstance(
        row.get("critical_argument_extraction_correct"), bool
    ):
        critical_argument_correct = row["critical_argument_extraction_correct"]
    if action_candidate is not None:
        row.setdefault("proposed_action", action_candidate["action"])
        row.setdefault("proposed_arguments", action_candidate["arguments"])
    provenance_semantically_valid = _provenance_semantic_validity(row)
    unsafe_execution = (
        row.get("unsafe_automatic_execution")
        if row.get("gate_decision") is not None
        and isinstance(row.get("unsafe_automatic_execution"), bool)
        else None
    )
    failure_categories = _failure_categories(
        provider_category=contract.get("failure_category"),
        status=row.get("status"),
        parse_success=parse_success,
        normalized_schema_valid=normalized_schema_valid,
        contract_semantically_valid=contract_semantically_valid,
        provenance_semantically_valid=provenance_semantically_valid,
        action_correct=action_correct,
        critical_argument_correct=critical_argument_correct,
    )
    source_width = _single_dimension(entries, "image_width")
    source_height = _single_dimension(entries, "image_height")
    processed_width = _single_dimension(entries, "processed_image_width")
    processed_height = _single_dimension(entries, "processed_image_height")
    local_performance = {
        "model_load_time_ms": provider.model_load_time_ms,
        "preprocessing_latency_ms": total("preprocessing_latency_ms"),
        "generation_latency_ms": generation_ms,
        "inference_latency_ms": total("inference_latency_ms"),
        "thin_gate_latency_ms": row.get("thin_gate_latency_ms", 0.0),
        "evidence_mapper_latency_ms": row.get("mapping_latency_ms", 0.0),
        "input_token_count": _sum_if_complete(entries, "input_token_count"),
        "output_token_count": output_tokens,
        "generated_tokens": generated_tokens,
        "tokens_per_second": (
            generated_tokens / (generation_ms / 1000.0)
            if generated_tokens is not None and generation_ms is not None and generation_ms > 0
            else None
        ),
        "gpu_memory_allocated_before_inference_bytes": maximum(
            "gpu_memory_allocated_before_inference_bytes"
        ),
        "gpu_peak_memory_allocated_bytes": maximum("gpu_peak_memory_allocated_bytes"),
        "gpu_peak_memory_reserved_bytes": maximum("gpu_peak_memory_reserved_bytes"),
        "model_dtype": dtype,
        "quantization": quantization,
        "attention_backend": attention_backend,
        "image_width": source_width,
        "image_height": source_height,
    }
    row.update(local_performance)
    row.update(
        {
            "local_performance": dict(local_performance),
            "model": provider.repository_id,
            "model_id": provider.repository_id,
            "model_alias": provider.model_alias,
            "model_revision": provider.model_revision,
            "processor_revision": provider.processor_revision,
            "parameter_count": provider.parameter_count,
            "zero_shot_prompt_version": ZERO_SHOT_V2,
            "schema_transport_version": str(
                stable_config.get(
                    "schema_transport_version", LOCAL_SCHEMA_TRANSPORT_VERSION
                )
            ),
            "parse_success": parse_success,
            "schema_valid": schema_valid,
            "normalization_applied": normalization_applied,
            "normalization_method": normalization_method,
            "normalized_schema_valid": normalized_schema_valid,
            "contract_semantically_valid": contract_semantically_valid,
            # Backward-compatible name for post-normalization usable output.
            "structured_output_valid": bool(
                normalized_schema_valid and contract_semantically_valid
            ),
            "action_candidate": action_candidate,
            "action_correct": action_correct,
            "critical_argument_correct": critical_argument_correct,
            "provenance_semantically_valid": provenance_semantically_valid,
            "unsafe_execution": unsafe_execution,
            "failure_category": failure_categories[0] if failure_categories else None,
            "failure_categories": failure_categories,
            "processed_image_width": processed_width,
            "processed_image_height": processed_height,
            "actual_model_image_dimensions": {
                "source_width": source_width,
                "source_height": source_height,
                "processed_width": processed_width,
                "processed_height": processed_height,
            },
            "benchmark_lock_id": lock["benchmark_id"],
            "benchmark_lock_sha256": lock["manifest_sha256"],
            "selected_case_count": selected_case_count,
            "benchmark_case_count": benchmark_case_count,
            "phase2_5_runner_version": RUNNER_VERSION,
            "provider_config": stable_config,
        }
    )
    return row


def _template_metadata(
    template: dict[str, Any],
    *,
    provider: BaseLocalVLMProvider,
    lock: Mapping[str, Any],
    selected_case_count: int,
    benchmark_case_count: int,
) -> dict[str, Any]:
    template.update(
        model=provider.repository_id,
        model_id=provider.repository_id,
        model_alias=provider.model_alias,
        model_revision=provider.model_revision,
        processor_revision=provider.processor_revision,
        zero_shot_prompt_version=ZERO_SHOT_V2,
        schema_transport_version=str(
            provider.experiment_config.get(
                "schema_transport_version", LOCAL_SCHEMA_TRANSPORT_VERSION
            )
        ),
        benchmark_lock_id=lock["benchmark_id"],
        benchmark_lock_sha256=lock["manifest_sha256"],
        selected_case_count=selected_case_count,
        benchmark_case_count=benchmark_case_count,
        phase2_5_runner_version=RUNNER_VERSION,
        provider_config=_stable_provider_config(provider),
    )
    return template


def _print_trial_details(row: Mapping[str, Any]) -> None:
    print("  --- raw structured output ---")
    records = row.get("model_call_records")
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, Mapping):
                continue
            raw_path = record.get("raw_response_path") or row.get("raw_error_response_path")
            raw = None
            if isinstance(raw_path, str):
                try:
                    raw = Path(raw_path).read_text(encoding="utf-8")
                except OSError:
                    raw = None
            print(f"  [{record.get('operation')}] {raw if raw is not None else '<unavailable>'}")
    print(f"  parsed action: {row.get('proposed_action')}")
    print(f"  critical arguments: {row.get('proposed_arguments')}")
    evidence = row.get("self_reported_argument_evidence")
    printed_evidence = False
    if isinstance(evidence, Mapping):
        for argument, items in evidence.items():
            if not isinstance(items, list) or not items:
                print(f"  evidence {argument}: <missing>")
                continue
            for item in items:
                if isinstance(item, Mapping):
                    printed_evidence = True
                    print(
                        f"  evidence {argument}: text={item.get('evidence_text')!r}; "
                        f"source={item.get('source_type_estimate')}; bbox={item.get('bbox')}"
                    )
    if not printed_evidence:
        print("  evidence text: <unavailable>")
        print("  source estimate: <unavailable>")
    evaluations = row.get("provenance_evaluations")
    printed_mapping = False
    if isinstance(evaluations, list):
        for item in evaluations:
            if isinstance(item, Mapping):
                printed_mapping = True
                print(
                    f"  mapped {item.get('argument_name')}: region="
                    f"{item.get('matched_region_id')}; status={item.get('evidence_status')}; "
                    f"source_estimate={item.get('source_type_estimate')}"
                )
    if not printed_mapping:
        print("  mapped region: <unavailable>")
    print(f"  Thin Gate decision: {row.get('gate_decision')}")
    print(
        "  output contract: "
        f"parse_success={row.get('parse_success')}; "
        f"schema_valid={row.get('schema_valid')}; "
        f"normalization_applied={row.get('normalization_applied')}; "
        f"normalized_schema_valid={row.get('normalized_schema_valid')}; "
        f"contract_semantically_valid={row.get('contract_semantically_valid')}"
    )
    print(
        "  evaluation diagnostics: "
        f"provenance_semantically_valid={row.get('provenance_semantically_valid')}; "
        f"action_correct={row.get('action_correct')}; "
        f"critical_argument_correct={row.get('critical_argument_correct')}; "
        f"unsafe_execution={row.get('unsafe_execution')}"
    )
    print(f"  failure categories: {row.get('failure_categories')}")
    inference_ms = row.get("inference_latency_ms")
    formatted_inference = (
        f"{inference_ms:.1f} ms"
        if isinstance(inference_ms, (int, float)) and not isinstance(inference_ms, bool)
        else "unavailable"
    )
    print(f"  inference latency: {formatted_inference}")
    print(f"  peak allocated VRAM: {_gib(row.get('gpu_peak_memory_allocated_bytes'))}")


def _write_load_failure(
    results_dir: Path,
    *,
    args: argparse.Namespace,
    repository_id: str,
    revision: str,
    lock: Mapping[str, Any],
    cache: Mapping[str, Any],
    error: BaseException,
    model_load_time_ms: float | None,
    provider_config: Mapping[str, Any],
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "model_load_error",
        "provider": "local",
        "model_alias": args.model,
        "model_id": repository_id,
        "model_revision": revision,
        "dtype": str(provider_config.get("dtype", LOCAL_DTYPE)),
        "quantization": str(
            provider_config.get("quantization", LOCAL_QUANTIZATION)
        ),
        "attention_backend": str(
            provider_config.get("attention_backend", LOCAL_ATTENTION_BACKEND)
        ),
        "device": args.device,
        "model_load_time_ms": model_load_time_ms,
        "benchmark_lock_id": lock["benchmark_id"],
        "benchmark_lock_sha256": lock["manifest_sha256"],
        "error_type": type(error).__name__,
        "error_message": str(error),
        "runtime_fallback_attempted": False,
        "provider_config": dict(provider_config),
        "cache_preflight": dict(cache),
    }
    (results_dir / "load_failure.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_analysis_reports(results_dir: Path, *, root_results_dir: Path) -> dict[str, Any]:
    from analyze_phase2_5 import analyze_phase2_5

    raw_path = results_dir / "raw_generations.jsonl"
    analysis_path = results_dir / "analysis.json"
    system_path = results_dir / "system_info.json"
    analysis = analyze_phase2_5(
        raw_path,
        analysis_path,
        results_dir / "plots",
        system_info_path=system_path,
    )
    attempts = read_jsonl(raw_path)
    system_info = (
        json.loads(system_path.read_text(encoding="utf-8")) if system_path.is_file() else None
    )
    (results_dir / "report.md").write_text(
        build_local_model_report(
            attempts,
            analysis,
            source_path=raw_path,
            system_info=system_info,
        ),
        encoding="utf-8",
    )

    analyses: dict[str, dict[str, Any]] = {}
    for alias in MODEL_ORDER:
        default_path = root_results_dir / alias / "analysis.json"
        if default_path.is_file():
            candidate = json.loads(default_path.read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                analyses[alias] = candidate
    # A custom --results-dir is normally a smoke or compatibility profile.  It
    # gets its own analysis/report above, but must not silently become the
    # canonical cross-model evidence surface.  Only exact
    # <results-root>/<alias>/analysis.json paths participate automatically.
    root_results_dir.mkdir(parents=True, exist_ok=True)
    (root_results_dir / "report_local_models.md").write_text(
        build_aggregate_report(analyses), encoding="utf-8"
    )
    return analysis


def run_benchmark(
    args: argparse.Namespace,
    *,
    provider_factory: Callable[..., BaseLocalVLMProvider] = create_local_provider,
) -> list[dict[str, Any]]:
    if args.provider != "local":
        raise ValueError("Phase 2.5 supports --provider local only")
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")
    arms = parse_arms(args.arms)
    lock = verify_phase2_benchmark_lock(args.benchmark_lock)
    _assert_frozen_benchmark_inputs(args)
    dataset_payload, all_scenarios = load_phase2_dataset(args.dataset)
    scenarios = select_phase2_scenarios(all_scenarios, args.max_cases, seed=args.seed)
    registry = load_action_registry(args.registry)
    policy = load_thin_gate_policy(args.policy)
    repository_id = LOCAL_MODEL_REPOSITORIES[args.model]
    revision = DEFAULT_MODEL_REVISIONS[args.model]
    results_root = Path(args.results_root)
    results_dir = Path(args.results_dir or results_root / args.model)
    raw_path = results_dir / "raw_generations.jsonl"
    final_path = results_dir / "final_trials.csv"
    existing = read_jsonl(raw_path)
    if existing and not args.resume:
        raise ValueError(
            f"{raw_path} already exists. Use --resume or another --results-dir; "
            "append-only evidence will not be overwritten."
        )

    expected_inferences = len(scenarios) * args.runs * (
        len(arms) + sum(arm is Phase2Arm.TWO_PASS_PROVENANCE for arm in arms)
    )
    prompt_versions = sorted({_prompt_version(arm) for arm in arms})
    effective_attention_backend = str(
        LOCAL_MODEL_PROVIDERS[args.model].EFFECTIVE_ATTENTION_BACKEND
    )
    cache = huggingface_cache_preflight(
        repository_id,
        revision=revision,
        estimated_download_bytes=ESTIMATED_REPOSITORY_BYTES[args.model],
        reserve_bytes=DOWNLOAD_DISK_RESERVE_BYTES,
    )
    _preflight_summary(
        args=args,
        repository_id=repository_id,
        revision=revision,
        dataset_version=str(dataset_payload["dataset_version"]),
        prompt_versions=prompt_versions,
        policy_version=str(policy["policy_version"]),
        cases=len(scenarios),
        arms=arms,
        expected_inferences=expected_inferences,
        cache=cache,
        attention_backend=effective_attention_backend,
    )
    if cache.get("cached_revision") is None and cache.get("sufficient_free_space") is False:
        raise ValueError(
            "Insufficient disk headroom for the uncached model and safety reserve; "
            "no download was started"
        )
    if args.preflight_only:
        print("Preflight only: no model was loaded and no inference was run.")
        return existing

    provider = provider_factory(
        args.model,
        revision=revision,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        enable_nvml=not args.no_nvml,
    )
    try:
        selection_scope_id = _selection_scope_id(scenarios)
        experiment_config_id = _experiment_config_id(
            provider,
            arms=arms,
            seed=args.seed,
            generation_seed=args.generation_seed,
            runs=args.runs,
            selection_scope_id=selection_scope_id,
            benchmark_lock_sha256=str(lock["manifest_sha256"]),
        )
        assert_phase2_5_resume_compatible(
            existing,
            provider="local",
            model_id=provider.repository_id,
            model_revision=provider.model_revision,
            dataset_version=str(dataset_payload["dataset_version"]),
            policy_version=str(policy["policy_version"]),
            prompt_versions=prompt_versions,
            model_dtype=str(provider.experiment_config.get("dtype", LOCAL_DTYPE)),
            quantization=str(
                provider.experiment_config.get("quantization", LOCAL_QUANTIZATION)
            ),
            attention_backend=str(
                provider.experiment_config.get("attention_backend", LOCAL_ATTENTION_BACKEND)
            ),
            zero_shot_prompt_version=ZERO_SHOT_V2,
            schema_transport_version=str(
                provider.experiment_config.get(
                    "schema_transport_version", LOCAL_SCHEMA_TRANSPORT_VERSION
                )
            ),
        )
        for field, expected in (
            ("selection_scope_id", selection_scope_id),
            ("experiment_config_id", experiment_config_id),
            ("benchmark_lock_sha256", lock["manifest_sha256"]),
        ):
            observed = {str(row.get(field, "")) for row in existing}
            if observed and observed != {str(expected)}:
                raise ValueError(
                    f"Existing Phase 2.5 attempts have incompatible {field}: "
                    f"{sorted(observed)}; expected {expected!r}"
                )

        _configure_logging(results_dir)
        try:
            provider.load()
        except Exception as error:
            _write_load_failure(
                results_dir,
                args=args,
                repository_id=repository_id,
                revision=revision,
                lock=lock,
                cache=cache,
                error=error,
                model_load_time_ms=provider.model_load_time_ms,
                provider_config=_stable_provider_config(provider),
            )
            raise ValueError(
                f"Local model load failed for {args.model}; no fallback profile was attempted. "
                f"See {results_dir / 'load_failure.json'}: {error}"
            ) from error

        resolved_experiment_config_id = _experiment_config_id(
            provider,
            arms=arms,
            seed=args.seed,
            generation_seed=args.generation_seed,
            runs=args.runs,
            selection_scope_id=selection_scope_id,
            benchmark_lock_sha256=str(lock["manifest_sha256"]),
        )
        if resolved_experiment_config_id != experiment_config_id:
            raise ValueError(
                "Local provider experiment configuration changed during model load; "
                "refusing an ambiguous scientific profile"
            )

        # Resume guards ran before model loading and cohort writes. An
        # incompatible invocation cannot download weights or overwrite an
        # existing system-info artifact before it fails.
        system_info = collect_phase2_5_system_info(
            model_repository_id=provider.repository_id,
            model_revision=provider.model_revision,
            processor_revision=provider.processor_revision,
            dtype=str(provider.experiment_config.get("dtype", LOCAL_DTYPE)),
            quantization=str(
                provider.experiment_config.get("quantization", LOCAL_QUANTIZATION)
            ),
            attention_backend=str(
                provider.experiment_config.get("attention_backend", LOCAL_ATTENTION_BACKEND)
            ),
            device=args.device,
            model=provider.model,
            processor=provider.processor,
            include_nvml=not args.no_nvml,
        )
        system_info.update(
            {
                "model_alias": provider.model_alias,
                "parameter_count": provider.parameter_count,
                "model_load_time_ms": provider.model_load_time_ms,
                "benchmark_lock_id": lock["benchmark_id"],
                "benchmark_lock_sha256": lock["manifest_sha256"],
                "benchmark_baseline_git_commit": lock["baseline_git_commit"],
                "phase2_5_runner_version": RUNNER_VERSION,
                "zero_shot_prompt_version": ZERO_SHOT_V2,
                "schema_transport_version": str(
                    provider.experiment_config.get(
                        "schema_transport_version", LOCAL_SCHEMA_TRANSPORT_VERSION
                    )
                ),
                "selected_case_count": len(scenarios),
                "benchmark_case_count": len(all_scenarios),
                "cache_preflight": dict(cache),
                "provider_config": _stable_provider_config(provider),
            }
        )
        write_phase2_5_system_info(results_dir / "system_info.json", system_info)

        completed = completed_phase2_5_identities(existing) if args.resume else set()
        planned_trial_count = len(scenarios) * args.runs * len(arms)
        templates: list[tuple[dict[str, Any], Phase2Arm, int, dict[str, Any]]] = []
        for run in range(1, args.runs + 1):
            run_order = list(scenarios)
            random.Random(args.seed * 1_000_003 + run).shuffle(run_order)
            for scenario in run_order:
                for arm in arms:
                    template = _base_result(
                        scenario=scenario,
                        arm=arm,
                        run=run,
                        provider_name="local",
                        provider=provider,
                        dataset_version=str(dataset_payload["dataset_version"]),
                        registry_version=str(registry["registry_version"]),
                        policy_version=str(policy["policy_version"]),
                        selection_scope_id=selection_scope_id,
                        experiment_config_id=experiment_config_id,
                        planned_trial_count=planned_trial_count,
                    )
                    templates.append(
                        (
                            scenario,
                            arm,
                            run,
                            _template_metadata(
                                template,
                                provider=provider,
                                lock=lock,
                                selected_case_count=len(scenarios),
                                benchmark_case_count=len(all_scenarios),
                            ),
                        )
                    )
        pending = [item for item in templates if phase2_5_trial_identity(item[3]) not in completed]
        print(f"Pending scientific trials: {len(pending)}")
        print(f"Resolved model revision: {provider.model_revision}")
        print(f"Resolved processor revision: {provider.processor_revision}")
        print(f"Model load time: {provider.model_load_time_ms:.1f} ms")
        print(f"Parameter count: {provider.parameter_count or 'unavailable'}")

        all_attempts = list(existing)
        new_rows: list[dict[str, Any]] = []
        stop_error: dict[str, Any] | None = None
        for scenario, arm, run, template in pending:
            attempt_index = next_phase2_5_attempt_index(all_attempts, template)
            print(
                f"RUN {scenario['scenario_id']} / {arm.value} / run {run} "
                f"(attempt {attempt_index})"
            )
            row = run_phase2_trial(
                scenario=scenario,
                arm=arm,
                run=run,
                provider_name="local",
                provider=provider,
                dataset_version=str(dataset_payload["dataset_version"]),
                registry=registry,
                policy=policy,
                results_dir=results_dir,
                selection_scope_id=selection_scope_id,
                experiment_config_id=experiment_config_id,
                planned_trial_count=planned_trial_count,
                attempt_index=attempt_index,
                selection_seed=args.seed,
                generation_seed=args.generation_seed,
                request_delay=0.0,
            )
            _augment_trial(
                row,
                provider=provider,
                lock=lock,
                selected_case_count=len(scenarios),
                benchmark_case_count=len(all_scenarios),
            )
            persist_phase2_5_attempt(raw_path, final_path, row)
            all_attempts.append(row)
            new_rows.append(row)
            if row["status"] == "completed":
                print(f"  completed / {row['gate_decision']}")
            else:
                print(f"  error / {row.get('error_type')}: {row.get('error_message')}")
                if row.get("error_type") != "ProviderResponseError":
                    stop_error = row
            if args.print_trial_details:
                _print_trial_details(row)
            if stop_error is not None:
                print(
                    "Stopping after a local runtime/configuration failure; no fallback profile "
                    "was attempted.",
                    file=sys.stderr,
                )
                break

        validate_phase2_5_attempts(read_jsonl(raw_path))
        _write_analysis_reports(results_dir, root_results_dir=results_root)
        completed_count = sum(row.get("status") == "completed" for row in new_rows)
        error_count = len(new_rows) - completed_count
        print(
            f"Persisted {len(new_rows)} attempts ({completed_count} completed, "
            f"{error_count} errors)"
        )
        print(f"Raw generations: {raw_path}")
        print(f"Final trials: {final_path}")
        print(f"System info: {results_dir / 'system_info.json'}")
        print(f"Analysis: {results_dir / 'analysis.json'}")
        print(f"Report: {results_dir / 'report.md'}")
        print(f"Aggregate report: {results_root / 'report_local_models.md'}")
        if stop_error is not None:
            raise ValueError(
                f"Local trial {stop_error['scene_id']} / "
                f"{stop_error['architecture_arm']} failed: {stop_error.get('error_message')}"
            )
        return read_jsonl(raw_path)
    finally:
        provider.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("local",), default="local")
    parser.add_argument("--model", choices=tuple(LOCAL_MODEL_PROVIDERS), required=True)
    parser.add_argument(
        "--arms",
        default="action_only,inline_provenance,oracle",
        help="Comma-separated: action_only,two_pass,inline_provenance,oracle",
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset_phase2/metadata.json"))
    parser.add_argument("--registry", type=Path, default=Path("config/action_registry.yaml"))
    parser.add_argument("--policy", type=Path, default=Path("config/policy_phase2.yaml"))
    parser.add_argument("--benchmark-lock", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--results-root", type=Path, default=Path("results_phase2_5"))
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generation-seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-nvml", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--print-trial-details", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    try:
        run_benchmark(parse_args())
    except (Phase2BenchmarkLockError, ProviderError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_MODEL_REVISIONS",
    "ESTIMATED_REPOSITORY_BYTES",
    "FROZEN_BENCHMARK_INPUTS",
    "RUNNER_VERSION",
    "parse_args",
    "run_benchmark",
]
