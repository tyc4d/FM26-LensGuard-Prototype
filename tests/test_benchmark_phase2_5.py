from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from analyze_phase2_5 import _cohort, _completion_context, _paired_cohort
from benchmark_phase2_5 import (
    _print_trial_details,
    _provenance_semantic_validity,
    parse_args,
    run_benchmark,
)
from phase2_schema import Phase2Operation, token_usage_from_metadata
from providers import ProviderResponse, ProviderResponseError
from providers.local import (
    LOCAL_MODEL_REPOSITORIES,
    LOCAL_SCHEMA_TRANSPORT_VERSION,
    ZERO_SHOT_V2,
)
from providers.local.base_local_vlm import attach_local_call_error_context
from providers.mock_phase2 import MockPhase2Provider
from result_store import read_jsonl


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_attempted_cohort(
    *,
    runs: int = 1,
    arms: tuple[tuple[str, str], ...] = (
        ("ACTION_ONLY", "ACTION_V1"),
        ("INLINE_PROVENANCE", "INLINE_V1"),
    ),
) -> list[dict[str, Any]]:
    planned = 81 * runs * len(arms)
    rows: list[dict[str, Any]] = []
    for run in range(1, runs + 1):
        for index in range(81):
            for arm, prompt in arms:
                rows.append(
                    {
                        "scene_id": f"scene-{index:02d}",
                        "condition": "CLEAN_TRUSTED",
                        "architecture_arm": arm,
                        "provider": "local",
                        "model_id": "model/repository",
                        "model_revision": "revision-a",
                        "run": run,
                        "prompt_version": prompt,
                        "dataset_version": "phase2-v1",
                        "policy_version": "phase2-policy-v1",
                        "status": "completed",
                        "planned_trial_count": planned,
                        "selected_case_count": 81,
                        "benchmark_case_count": 81,
                    }
                )
    return rows


class _MockLocalRunnerProvider:
    """Scientific-pipeline fake: Phase 2 fixture outputs plus local telemetry."""

    def __init__(
        self,
        alias: str,
        *,
        revision: str,
        max_new_tokens: int,
        fail_parse: bool = False,
        fail_schema: bool = False,
        **_: Any,
    ) -> None:
        self.model_alias = alias
        self.repository_id = LOCAL_MODEL_REPOSITORIES[alias]
        self.model_identifier = self.repository_id
        self.model_revision = revision
        self.processor_revision = revision
        self.max_new_tokens = max_new_tokens
        self.attention_backend = (
            "llm_sdpa_vision_eager" if alias == "minicpm-v4.5" else "sdpa"
        )
        self.model_load_time_ms = 4.0
        self.parameter_count = 4_000_000_000
        self.model = SimpleNamespace(config=SimpleNamespace(_commit_hash=revision))
        self.processor = SimpleNamespace(_commit_hash=revision)
        self.delegate = MockPhase2Provider(latency_ms_per_call=0)
        self.fail_parse = fail_parse
        self.fail_schema = fail_schema
        self.closed = False

    @property
    def experiment_config(self) -> dict[str, Any]:
        return {
            "provider": "local",
            "model_alias": self.model_alias,
            "model_repository_id": self.repository_id,
            "model_revision": self.model_revision,
            "processor_revision": self.processor_revision,
            "generation_config": {
                "do_sample": False,
                "max_new_tokens": self.max_new_tokens,
                "batch_size": 1,
            },
            "dtype": "bfloat16",
            "quantization": "none",
            "attention_backend": self.attention_backend,
            "prompt_profile": ZERO_SHOT_V2,
            "schema_transport_version": LOCAL_SCHEMA_TRANSPORT_VERSION,
        }

    def load(self) -> _MockLocalRunnerProvider:
        return self

    def close(self) -> None:
        self.closed = True

    def set_request_seed(self, seed: int) -> None:
        self.delegate.set_request_seed(seed)

    def _local_response(
        self,
        response: ProviderResponse[Any],
        *,
        action_candidate: Any | None = None,
    ) -> ProviderResponse[Any]:
        metadata = dict(response.response_metadata)
        metadata["requested_model"] = self.repository_id
        metadata["returned_model"] = self.repository_id
        usage = token_usage_from_metadata(metadata)
        output_tokens = usage.output_tokens
        parsed = response.parsed
        if action_candidate is None:
            action_output = getattr(parsed, "action_output", None)
            action_candidate = action_output() if callable(action_output) else parsed
        candidate_dump = getattr(action_candidate, "model_dump", None)
        candidate = (
            candidate_dump(mode="json", exclude_none=True)
            if callable(candidate_dump)
            else None
        )
        contract = {
            "parse_success": True,
            "schema_valid": True,
            "normalization_applied": False,
            "normalization_method": None,
            "normalized_schema_valid": True,
            "contract_semantically_valid": True,
            "failure_category": None,
            "action_candidate": candidate,
        }
        metadata["local_inference"] = {
            "structured_output_valid": True,
            "preprocessing_latency_ms": 1.0,
            "generation_latency_ms": 2.0,
            "inference_latency_ms": 3.0,
            "input_token_count": usage.input_tokens,
            "output_token_count": output_tokens,
            "generated_tokens": output_tokens,
            "tokens_per_second": (
                output_tokens / 0.002 if output_tokens is not None else None
            ),
            "gpu_memory_allocated_before_inference_bytes": 8_000_000_000,
            "gpu_peak_memory_allocated_bytes": 8_500_000_000,
            "gpu_peak_memory_reserved_bytes": 9_000_000_000,
            "image_width": 1200,
            "image_height": 760,
            "processed_image_width": 1200,
            "processed_image_height": 760,
            "output_contract": contract,
        }
        metadata["output_contract"] = dict(contract)
        return replace(
            response,
            model=self.repository_id,
            response_metadata=metadata,
        )

    def _raise_parse_error(self, operation: Phase2Operation) -> None:
        raw = "```json\n{bad}\n```"
        local = {
            "structured_output_valid": False,
            "preprocessing_latency_ms": 1.0,
            "generation_latency_ms": 2.0,
            "inference_latency_ms": 3.0,
            "input_token_count": 10,
            "output_token_count": 2,
            "generated_tokens": 2,
            "tokens_per_second": 1_000.0,
            "gpu_memory_allocated_before_inference_bytes": 8_000_000_000,
            "gpu_peak_memory_allocated_bytes": 8_500_000_000,
            "gpu_peak_memory_reserved_bytes": 9_000_000_000,
            "image_width": 1200,
            "image_height": 760,
            "output_contract": {
                "parse_success": False,
                "schema_valid": False,
                "normalization_applied": False,
                "normalization_method": None,
                "normalized_schema_valid": False,
                "contract_semantically_valid": False,
                "failure_category": "malformed_json",
                "action_candidate": None,
            },
        }
        metadata = {
            "status": "error",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
                "cached_tokens": 0,
                "thought_tokens": 0,
            },
            "local_inference": local,
            "output_contract": dict(local["output_contract"]),
        }
        error = ProviderResponseError(
            "mock malformed local JSON",
            raw_response=raw,
            response_metadata=metadata,
        )
        attach_local_call_error_context(
            error,
            operation=operation,
            latency_ms=3.0,
            model=self.repository_id,
            response_metadata=metadata,
            raw_response=raw,
        )
        raise error

    def _raise_schema_error(
        self, operation: Phase2Operation, action_candidate: Any
    ) -> None:
        candidate = action_candidate.model_dump(mode="json", exclude_none=True)
        argument_name = next(iter(candidate["arguments"]), "target")
        raw = json.dumps(
            {
                **candidate,
                "argument_evidence": [
                    {
                        "argument": argument_name,
                        "source": "official_signage",
                        "evidence": "synthetic invalid aliases",
                    }
                ],
            }
        )
        contract = {
            "parse_success": True,
            "schema_valid": False,
            "normalization_applied": False,
            "normalization_method": None,
            "normalized_schema_valid": False,
            "contract_semantically_valid": False,
            "failure_category": "schema_mismatch",
            "action_candidate": candidate,
        }
        local = {
            "structured_output_valid": False,
            "preprocessing_latency_ms": 1.0,
            "generation_latency_ms": 2.0,
            "inference_latency_ms": 3.0,
            "input_token_count": 10,
            "output_token_count": 2,
            "generated_tokens": 2,
            "tokens_per_second": 1_000.0,
            "gpu_memory_allocated_before_inference_bytes": 8_000_000_000,
            "gpu_peak_memory_allocated_bytes": 8_500_000_000,
            "gpu_peak_memory_reserved_bytes": 9_000_000_000,
            "image_width": 1200,
            "image_height": 760,
            "output_contract": contract,
        }
        metadata = {
            "status": "error",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
                "cached_tokens": 0,
                "thought_tokens": 0,
            },
            "local_inference": local,
            "output_contract": dict(contract),
        }
        error = ProviderResponseError(
            "mock local schema mismatch",
            raw_response=raw,
            response_metadata=metadata,
        )
        attach_local_call_error_context(
            error,
            operation=operation,
            latency_ms=3.0,
            model=self.repository_id,
            response_metadata=metadata,
            raw_response=raw,
        )
        raise error

    def action_only(self, *args: Any, **kwargs: Any) -> ProviderResponse[Any]:
        if self.fail_parse:
            self._raise_parse_error(Phase2Operation.ACTION_ONLY)
        return self._local_response(self.delegate.action_only(*args, **kwargs))

    def inline_provenance(self, *args: Any, **kwargs: Any) -> ProviderResponse[Any]:
        if self.fail_parse:
            self._raise_parse_error(Phase2Operation.INLINE_PROVENANCE)
        response = self.delegate.inline_provenance(*args, **kwargs)
        if self.fail_schema:
            self._raise_schema_error(
                Phase2Operation.INLINE_PROVENANCE, response.parsed.action_output()
            )
        return self._local_response(response)

    def two_pass_evidence(self, *args: Any, **kwargs: Any) -> ProviderResponse[Any]:
        if self.fail_parse:
            self._raise_parse_error(Phase2Operation.TWO_PASS_EVIDENCE)
        return self._local_response(
            self.delegate.two_pass_evidence(*args, **kwargs),
            action_candidate=args[2] if len(args) >= 3 else kwargs.get("proposed_action"),
        )


def _args(results_root: Path, *extra: str):
    return parse_args(
        [
            "--model",
            "gemma3-4b",
            "--results-root",
            str(results_root),
            "--max-cases",
            "1",
            "--no-nvml",
            *extra,
        ]
    )


def test_trial_detail_output_keeps_failed_provenance_fields_explicit(capsys: Any) -> None:
    _print_trial_details(
        {
            "proposed_action": None,
            "proposed_arguments": None,
            "self_reported_argument_evidence": None,
            "provenance_evaluations": None,
            "gate_decision": None,
            "inference_latency_ms": 3.0,
            "gpu_peak_memory_allocated_bytes": 1024,
        }
    )

    output = capsys.readouterr().out
    assert "evidence text: <unavailable>" in output
    assert "source estimate: <unavailable>" in output
    assert "mapped region: <unavailable>" in output
    assert "Thin Gate decision: None" in output


def test_provenance_semantics_ignore_an_absent_optional_argument_only() -> None:
    base = {
        "architecture_arm": "INLINE_PROVENANCE",
        "proposed_arguments": {"direction": "LEFT"},
        "provenance_evaluations": [
            {
                "argument_name": "direction",
                "evidence_status": "matched",
                "provenance_correct": True,
                "reported_evidence_items": [
                    {"evidence_status": "matched", "supports_argument": True}
                ],
            },
            {
                "argument_name": "destination",
                "evidence_status": "missing",
                "provenance_correct": None,
                "reported_evidence_items": [],
            },
        ],
    }
    assert _provenance_semantic_validity(base) is True

    base["proposed_arguments"] = {
        "direction": "LEFT",
        "destination": "EMERGENCY EXIT",
    }
    assert _provenance_semantic_validity(base) is False


def test_completion_counts_error_trials_as_attempted_and_requires_exact_pairing() -> None:
    rows = _synthetic_attempted_cohort(
        runs=2,
        arms=(
            ("ACTION_ONLY", "ACTION_V1"),
            ("INLINE_PROVENANCE", "INLINE_V1"),
            ("ORACLE_PROVENANCE", "ACTION_V1"),
        ),
    )
    rows[-1]["status"] = "error"
    metrics = {
        "core_phase2_metrics": {
            "trial_counts": {"completed": len(rows) - 1, "unresolved_errors": 1}
        }
    }

    completion = _completion_context(rows, metrics)

    assert completion["dataset_complete"] is True
    assert completion["attempted_trials"] == 486
    assert completion["completed_trials"] == 485
    assert completion["unresolved_error_trials"] == 1
    assert completion["paired_cohort"]["paired_scope_count"] == 162
    assert completion["paired_cohort"]["expected_paired_scope_count"] == 162
    assert completion["paired_cohort"]["scope_complete"] is True
    assert completion["paired_cohort"]["oracle_scope_match"] is True

    unpaired = [
        row
        for row in rows
        if not (
            row["scene_id"] == "scene-80"
            and row["run"] == 2
            and row["architecture_arm"] == "INLINE_PROVENANCE"
        )
    ]
    pairing = _paired_cohort(unpaired)
    assert pairing["exact_scope_match"] is False
    assert pairing["scope_complete"] is False
    assert pairing["missing_from_inline_provenance"] == [
        {"scene_id": "scene-80", "condition": "CLEAN_TRUSTED", "run": 2}
    ]


def test_analyzer_keeps_legacy_v1_cohort_without_schema_transport() -> None:
    row = _synthetic_attempted_cohort()[0]
    row.update(
        zero_shot_prompt_version="ZERO_SHOT_V1",
        selection_scope_id="scope-a",
        benchmark_lock_id="lock-a",
        benchmark_lock_sha256="a" * 64,
    )

    cohort = _cohort([row])

    assert cohort["zero_shot_prompt_version"] == "ZERO_SHOT_V1"
    assert cohort["schema_transport_version"] is None


def test_preflight_does_not_construct_model_or_write_results(tmp_path: Path) -> None:
    root = tmp_path / "results_phase2_5"

    def forbidden_factory(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("preflight must not construct a provider")

    rows = run_benchmark(
        _args(root, "--preflight-only"),
        provider_factory=forbidden_factory,
    )

    assert rows == []
    assert not root.exists()


def test_runner_persists_separated_completed_trials_and_marks_smoke_partial(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results_phase2_5"
    created: list[_MockLocalRunnerProvider] = []

    def factory(alias: str, **kwargs: Any) -> _MockLocalRunnerProvider:
        provider = _MockLocalRunnerProvider(alias, **kwargs)
        created.append(provider)
        return provider

    rows = run_benchmark(
        _args(root, "--arms", "action_only,inline_provenance"),
        provider_factory=factory,
    )
    model_dir = root / "gemma3-4b"

    assert len(rows) == 2
    assert {row["architecture_arm"] for row in rows} == {
        "ACTION_ONLY",
        "INLINE_PROVENANCE",
    }
    assert all(row["status"] == "completed" for row in rows)
    assert all(row["model_id"] == "google/gemma-3-4b-it" for row in rows)
    assert all(row["selected_case_count"] == 1 for row in rows)
    assert all(row["benchmark_case_count"] == 81 for row in rows)
    assert created[0].closed is True
    for relative in (
        "raw_generations.jsonl",
        "final_trials.csv",
        "system_info.json",
        "analysis.json",
        "report.md",
    ):
        assert (model_dir / relative).is_file()
    analysis = json.loads((model_dir / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["selection_scope_complete"] is True
    assert analysis["dataset_complete"] is False
    assert analysis["primary_arms_complete"] is True
    assert analysis["full_arms_complete"] is False
    assert analysis["paired_cohort"]["exact_scope_match"] is True
    assert all(row["zero_shot_prompt_version"] == ZERO_SHOT_V2 for row in rows)
    assert all(
        row["schema_transport_version"] == LOCAL_SCHEMA_TRANSPORT_VERSION
        for row in rows
    )
    assert all(row["parse_success"] is True for row in rows)
    assert all(row["schema_valid"] is True for row in rows)
    assert all(row["action_correct"] is True for row in rows)
    aggregate = (root / "report_local_models.md").read_text(encoding="utf-8")
    assert "smoke/partial observations" in aggregate

    resumed = run_benchmark(
        _args(root, "--arms", "action_only,inline_provenance", "--resume"),
        provider_factory=factory,
    )
    assert len(resumed) == 2
    assert len(read_jsonl(model_dir / "raw_generations.jsonl")) == 2
    before_incompatible_resume = (model_dir / "system_info.json").read_text(
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match="experiment_config_id"):
        run_benchmark(
            _args(
                root,
                "--arms",
                "action_only,inline_provenance",
                "--resume",
                "--max-new-tokens",
                "99",
            ),
            provider_factory=factory,
        )
    assert (model_dir / "system_info.json").read_text(
        encoding="utf-8"
    ) == before_incompatible_resume


def test_custom_smoke_directory_does_not_enter_canonical_aggregate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results_phase2_5"
    smoke = root / "gemma3-4b-smoke"

    def factory(alias: str, **kwargs: Any) -> _MockLocalRunnerProvider:
        return _MockLocalRunnerProvider(alias, **kwargs)

    run_benchmark(
        _args(
            root,
            "--results-dir",
            str(smoke),
            "--arms",
            "action_only,inline_provenance",
        ),
        provider_factory=factory,
    )

    assert (smoke / "analysis.json").is_file()
    assert (smoke / "report.md").is_file()
    aggregate = (root / "report_local_models.md").read_text(encoding="utf-8")
    assert "| Gemma 3 4B | N/A | N/A | N/A | N/A | N/A | N/A | N/A |" in aggregate
    assert "**SMOKE/PARTIAL COHORTS:** Gemma 3 4B" not in aggregate

    run_benchmark(
        _args(root, "--arms", "action_only,inline_provenance"),
        provider_factory=factory,
    )
    canonical = (root / "report_local_models.md").read_text(encoding="utf-8")
    assert "**SMOKE/PARTIAL COHORTS:** Gemma 3 4B" in canonical


def test_runner_records_malformed_output_as_error_trial(tmp_path: Path) -> None:
    root = tmp_path / "results_phase2_5"

    def factory(alias: str, **kwargs: Any) -> _MockLocalRunnerProvider:
        return _MockLocalRunnerProvider(alias, fail_parse=True, **kwargs)

    rows = run_benchmark(
        _args(root, "--arms", "action_only"),
        provider_factory=factory,
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["error_type"] == "ProviderResponseError"
    assert rows[0]["structured_output_valid"] is False
    assert rows[0]["parse_success"] is False
    assert rows[0]["schema_valid"] is False
    assert rows[0]["failure_category"] == "malformed_json"
    assert rows[0]["failure_categories"] == ["malformed_json"]
    assert rows[0]["action_correct"] is None
    assert rows[0]["critical_argument_correct"] is None
    assert rows[0]["unsafe_execution"] is None
    assert rows[0]["raw_error_response_path"]
    assert Path(rows[0]["raw_error_response_path"]).read_text(encoding="utf-8").startswith(
        "```json"
    )
    analysis = json.loads(
        (root / "gemma3-4b/analysis.json").read_text(encoding="utf-8")
    )
    assert analysis["metrics"]["structured_output"][
        "structured_output_parse_success_rate"
    ] == 0.0
    assert analysis["metrics"]["efficiency"]["p50_thin_gate_latency_ms"] is None
    assert analysis["metrics"]["efficiency"][
        "p50_evidence_mapper_latency_ms"
    ] is None


def test_runner_preserves_schema_error_and_scores_strict_action_candidate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results_phase2_5"

    def factory(alias: str, **kwargs: Any) -> _MockLocalRunnerProvider:
        return _MockLocalRunnerProvider(alias, fail_schema=True, **kwargs)

    rows = run_benchmark(
        _args(root, "--arms", "inline_provenance"),
        provider_factory=factory,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "error"
    assert row["parse_success"] is True
    assert row["schema_valid"] is False
    assert row["normalized_schema_valid"] is False
    assert row["structured_output_valid"] is False
    assert row["action_correct"] is True
    assert row["critical_argument_correct"] is False
    assert row["unsafe_execution"] is None
    assert row["provenance_semantically_valid"] is None
    assert row["failure_category"] == "schema_mismatch"
    assert row["failure_categories"] == [
        "schema_mismatch",
        "critical_argument_prediction_failure",
    ]
    assert row["action_candidate"]["action"] == row["ground_truth_action"]
    assert row["action_candidate"]["arguments"] != row["ground_truth_arguments"]
    raw_path = Path(row["raw_error_response_path"])
    assert raw_path.is_file()
    raw = raw_path.read_text(encoding="utf-8")
    assert json.loads(raw)["argument_evidence"]
    assert row["model_call_records"][0]["raw_response_path"] == str(raw_path)


def test_runner_rejects_unlocked_cli_artifact_before_provider_construction(
    tmp_path: Path,
) -> None:
    alternate = tmp_path / "metadata.json"
    shutil.copyfile(ROOT / "dataset_phase2/metadata.json", alternate)
    constructed = False

    def factory(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal constructed
        constructed = True
        raise AssertionError

    args = _args(tmp_path / "results", "--dataset", str(alternate))
    with pytest.raises(ValueError, match="must use the frozen Phase 2 artifact"):
        run_benchmark(args, provider_factory=factory)
    assert constructed is False
    assert not (tmp_path / "results").exists()


def test_minicpm_preflight_and_load_failure_record_effective_attention_profile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "results_phase2_5"
    args = _args(root, "--arms", "action_only")
    args.model = "minicpm-v4.5"

    class _LoadFailureProvider(_MockLocalRunnerProvider):
        def load(self) -> _MockLocalRunnerProvider:
            self.model_load_time_ms = 12.5
            raise RuntimeError("synthetic model-load failure")

    with pytest.raises(ValueError, match="no fallback profile was attempted"):
        run_benchmark(
            args,
            provider_factory=lambda alias, **kwargs: _LoadFailureProvider(
                alias, **kwargs
            ),
        )

    output = capsys.readouterr().out
    assert "attention backend: llm_sdpa_vision_eager" in output
    failure = json.loads(
        (root / "minicpm-v4.5/load_failure.json").read_text(encoding="utf-8")
    )
    assert failure["attention_backend"] == "llm_sdpa_vision_eager"
    assert failure["provider_config"]["attention_backend"] == (
        "llm_sdpa_vision_eager"
    )
    assert failure["model_load_time_ms"] == 12.5
    assert failure["runtime_fallback_attempted"] is False
