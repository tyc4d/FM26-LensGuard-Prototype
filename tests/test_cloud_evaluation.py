"""Offline scientific-parity tests using unchanged, frozen local outputs."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from benchmark_phase2 import load_phase2_dataset
from benchmark_phase3_5 import _build_registry, _expected_evidence_refs
from cloud_baseline_evaluation import (
    estimate_token_cost,
    normalize_response,
    summarize_records,
)
from providers.base_cloud_vlm import CloudResponse


ROOT = Path(__file__).resolve().parents[1]
LOCAL_RESULTS = ROOT / "results_phase3_5/grounded-provenance-v1"
REPLAY_RESULTS = (
    ROOT / "results_phase3_6/uncertainty-aware-v1/replay_phase3_5"
)
MODELS = ("gemma3-4b", "minicpm-v4.5", "qwen3vl-8b")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _gate_semantics(gate: dict | None) -> dict | None:
    if gate is None:
        return None
    # Pydantic embeds the input dict's insertion order in diagnostic prose.
    # Frozen JSONL sorts object keys; raw model text need not. Preserve every
    # scientific gate field while excluding only that incidental issue prose.
    return {
        **gate,
        "reference_issues": [
            {key: value for key, value in issue.items() if key != "message"}
            for issue in gate["reference_issues"]
        ],
    }


def _response(output_text: str, *, provider: str = "openai") -> CloudResponse:
    return CloudResponse(
        provider=provider,
        model="offline-fixture-model",
        model_id="offline-fixture-model",
        api_interface="offline-fixture",
        timestamp_utc="2026-09-05T00:00:00+00:00",
        raw_response={"output_text": output_text},
        output_text=output_text,
        completed=True,
        error_type=None,
        error_detail=None,
        http_status=200,
        api_status="completed",
        request_id="offline-request",
        latency_ms=125.0,
        usage={"input_tokens": 20, "output_tokens": 10},
        provider_config={"fixture": True},
        transport_attempts=1,
    )


@pytest.fixture(scope="module")
def cases() -> dict[str, dict]:
    _, records = load_phase2_dataset(ROOT / "dataset_phase2/metadata.json")
    return {record["scenario_id"]: record for record in records}


@pytest.fixture(scope="module")
def registries(cases):
    return {case_id: _build_registry(case) for case_id, case in cases.items()}


@pytest.fixture(scope="module")
def local_rows() -> dict[str, list[dict]]:
    return {
        model: [
            row
            for row in _read_jsonl(LOCAL_RESULTS / model / "raw_generations.jsonl")
            if row["architecture_arm"] in {"ACTION_ONLY", "GROUNDED_REGISTRY"}
        ]
        for model in MODELS
    }


@pytest.fixture(scope="module")
def normalized_local_rows(cases, registries, local_rows) -> dict[str, list[dict]]:
    return {
        model: [
            normalize_response(
                response=_response(row["raw_response"]),
                scenario=cases[row["scene_id"]],
                arm=(
                    "ACTION_ONLY"
                    if row["architecture_arm"] == "ACTION_ONLY"
                    else "GROUNDED"
                ),
                registry=registries[row["scene_id"]],
                experiment_id="offline-frozen-parity",
                raw_response_path=f"offline/{model}/{row['architecture_arm']}/{row['scene_id']}.json",
            )
            for row in rows
        ]
        for model, rows in local_rows.items()
    }


@pytest.mark.parametrize("model", MODELS)
def test_frozen_local_utility_schema_and_selection_metrics_are_preserved(
    model, local_rows, normalized_local_rows
) -> None:
    records = normalized_local_rows[model]
    assert len(records) == 162
    assert len({(row["case_id"], row["arm"]) for row in records}) == 162
    for original, normalized in zip(local_rows[model], records, strict=True):
        assert normalized["architecture_arm"] == original["architecture_arm"]
        assert normalized["scientific_attempt"] == 1
        assert normalized["completed"] is True
        assert normalized["schema_valid"] == original["schema_valid"]
        assert normalized["status"] == original["status"]
        assert normalized["action_correct"] == original["action_correct"]
        assert (
            normalized["critical_arguments_correct"]
            == original["critical_arguments_correct"]
        )
        assert (
            normalized["evidence_selection_records"]
            == original["evidence_selection_records"]
        )

    summary = summarize_records(records, planned_trials=162)
    assert summary["planned_trials"] == summary["recorded_trials"] == 162
    assert summary["completed_trials"] == 162
    assert summary["incomplete"] is False
    frozen = json.loads((LOCAL_RESULTS / model / "analysis.json").read_text())
    for arm in ("ACTION_ONLY", "GROUNDED_REGISTRY"):
        actual_arm = summary["metrics"]["by_arm"][arm]
        frozen_arm = frozen["metrics"]["by_arm"][arm]
        assert actual_arm["utility"] == frozen_arm["utility"]
        assert actual_arm["structural"] == frozen_arm["structural"]
        for field in (
            "correct_evidence_selection",
            "correct_evidence_region_selection",
            "correct_user_evidence_selection",
            "unknown_or_invented_evidence_id_count",
            "malformed_reference_container_issue_count",
        ):
            assert (
                actual_arm["evidence_selection"][field]
                == frozen_arm["evidence_selection"][field]
            )


@pytest.mark.parametrize("model", MODELS)
def test_gate_matches_every_frozen_phase3_6_replay_record(
    model, normalized_local_rows
) -> None:
    frozen = {
        record["scene"]["scene_id"]: record
        for record in _read_jsonl(REPLAY_RESULTS / "replay_records.jsonl")
        if record["source"]["identity"]["model_alias"] == model
    }
    grounded = [
        record for record in normalized_local_rows[model] if record["arm"] == "GROUNDED"
    ]
    assert len(grounded) == len(frozen) == 81
    for record in grounded:
        reference = frozen[record["case_id"]]
        assert _gate_semantics(record["phase3_6"]) == _gate_semantics(reference["phase3_6"])
        assert (
            record["phase3_6_expected_decision"]
            == reference["expected"]["phase3_6_required_decision"]
        )

    decisions = Counter(
        record["phase3_6"]["decision"] if record["phase3_6"] else "NOT_EVALUABLE"
        for record in grounded
    )
    analysis = json.loads((REPLAY_RESULTS / "analysis.json").read_text())
    assert dict(decisions) == {
        decision: count
        for decision, count in analysis["by_model"][model]["decision_distribution"].items()
        if count
    }


@pytest.mark.parametrize("provider", ("openai", "gemini"))
def test_malformed_reference_container_remains_a_usable_unrepaired_proposal(
    provider, cases, registries
) -> None:
    case_id = "p2_call_hotel__clean_trusted"
    scenario, registry = cases[case_id], registries[case_id]
    references = _expected_evidence_refs(scenario, registry)
    scalar_id = references["target_number"][0]
    payload = {
        "action": "CALL",
        "arguments": dict(scenario["ground_truth_arguments"]),
        "argument_evidence_refs": {"target_number": scalar_id},
    }
    response = _response(json.dumps(payload), provider=provider)
    record = normalize_response(
        response=response,
        scenario=scenario,
        arm="GROUNDED",
        registry=registry,
        experiment_id="offline-malformed",
        raw_response_path="offline/malformed.json",
    )
    assert record["completed"] is True
    assert record["parse_success"] is True
    assert record["schema_valid"] is False
    assert record["status"] == "completed"
    assert record["critical_arguments_correct"] is True
    assert record["argument_evidence_refs"] == payload["argument_evidence_refs"]
    assert record["phase3_6"]["decision"] == "BLOCK"
    assert record["phase3_6"]["reason_code"] == "INVALID_REFERENCE"
    assert response.output_text == json.dumps(payload)
    metrics = summarize_records([record], planned_trials=1)["metrics"]
    selection = metrics["by_arm"]["GROUNDED_REGISTRY"]["evidence_selection"]
    assert selection["unknown_or_invented_evidence_id_count"] == 0
    assert selection["malformed_reference_container_issue_count"] == 1


def test_unknown_id_is_counted_without_being_repaired(cases, registries) -> None:
    case_id = "p2_call_hotel__clean_trusted"
    scenario = cases[case_id]
    payload = {
        "action": "CALL",
        "arguments": dict(scenario["ground_truth_arguments"]),
        "argument_evidence_refs": {"target_number": [f"{case_id}:invented-region"]},
    }
    record = normalize_response(
        response=_response(json.dumps(payload)),
        scenario=scenario,
        arm="GROUNDED",
        registry=registries[case_id],
        experiment_id="offline-unknown-id",
        raw_response_path="offline/unknown.json",
    )
    assert record["schema_valid"] is True
    assert record["argument_evidence_refs"] == payload["argument_evidence_refs"]
    assert record["phase3_6"]["decision"] == "BLOCK"
    metrics = summarize_records([record], planned_trials=1)["metrics"]
    selection = metrics["by_arm"]["GROUNDED_REGISTRY"]["evidence_selection"]
    assert selection["unknown_or_invented_evidence_id_count"] == 1
    assert selection["malformed_reference_container_issue_count"] == 0


@pytest.mark.parametrize("arm", ("ACTION_ONLY", "GROUNDED"))
def test_transport_failure_is_unassessed_even_with_parseable_partial_output(
    arm, cases, registries
) -> None:
    case_id = "p2_call_hotel__authority_impersonation"
    scenario, registry = cases[case_id], registries[case_id]
    payload = {
        "action": "CALL",
        "arguments": dict(scenario["ground_truth_arguments"]),
    }
    if arm == "GROUNDED":
        payload["argument_evidence_refs"] = _expected_evidence_refs(scenario, registry)
    response = replace(
        _response(json.dumps(payload)),
        completed=False,
        error_type="RATE_LIMIT_EXHAUSTED",
        api_status="RESOURCE_EXHAUSTED",
        http_status=429,
        transport_attempts=5,
        rate_limit_events=5,
        total_backoff_seconds=450.0,
    )
    record = normalize_response(
        response=response,
        scenario=scenario,
        arm=arm,
        registry=registry,
        experiment_id="offline-transport-error",
        raw_response_path="offline/transport-error.json",
    )
    assert record["completed"] is False
    assert record["status"] == "error"
    assert record["schema_valid"] is None
    assert record["error_type"] == "RATE_LIMIT_EXHAUSTED"
    assert record["scientific_attempt"] == 1
    assert record["transport_attempts"] == 5
    assert record["phase3_6"] is None
    assert record["attack_success"] is None
    assert record["unsafe_automatic_execution"] is None

    summary = summarize_records([record], planned_trials=2)
    assert summary["recorded_trials"] == 1
    assert summary["completed_trials"] == 0
    assert summary["incomplete"] is True
    arm_name = "ACTION_ONLY" if arm == "ACTION_ONLY" else "GROUNDED_REGISTRY"
    metrics = summary["metrics"]["by_arm"][arm_name]
    unsafe = metrics["security"]["automatic_unsafe_execution"]
    assert (
        unsafe["numerator"], unsafe["denominator"],
        unsafe["eligible_count"], unsafe["unassessed_count"],
    ) == (0, 0, 1, 1)
    assert unsafe["rate"] is None
    e2e = metrics["utility"]["critical_argument_accuracy_end_to_end"]
    assert (e2e["numerator"], e2e["denominator"]) == (0, 1)
    if arm == "GROUNDED":
        phase3_6 = summary["phase3_6"]["security_and_abstention"]
        for name in ("unsafe_auto_execution_rate", "escalation_recall"):
            metric = phase3_6[name]
            assert (
                metric["numerator"], metric["denominator"],
                metric["eligible_count"], metric["unassessed_count"],
            ) == (0, 0, 1, 1)


def test_malformed_json_is_a_model_failure_with_a_preserved_api_completion(
    cases, registries
) -> None:
    case_id = "p2_call_hotel__clean_trusted"
    record = normalize_response(
        response=_response('{"action": "CALL",'),
        scenario=cases[case_id],
        arm="GROUNDED",
        registry=registries[case_id],
        experiment_id="offline-json-error",
        raw_response_path="offline/json-error.json",
    )
    assert record["completed"] is True
    assert record["parse_success"] is False
    assert record["schema_valid"] is False
    assert record["status"] == "error"
    assert record["phase3_6"] is None
    assert record["scientific_attempt"] == record["transport_attempts"] == 1
    summary = summarize_records([record], planned_trials=1)
    assert summary["completed_trials"] == 1
    assert summary["metrics"]["completed_trials"] == 0


@pytest.mark.parametrize(
    ("provider", "model", "usage", "expected_cost", "billed_output_tokens"),
    (
        (
            "openai",
            "gpt-5.6-sol",
            {"input_tokens": 1000, "cached_input_tokens": 100, "output_tokens": 50},
            (900 * 4 + 100 * 0.4 + 50 * 20) / 1_000_000,
            50,
        ),
        (
            "gemini",
            "gemini-3.1-flash-lite",
            {
                "input_tokens": 1000,
                "cached_input_tokens": 100,
                "output_tokens": 50,
                "reasoning_tokens": 10,
                "total_tokens": 1060,
            },
            (900 * 0.25 + 100 * 0.025 + 60 * 1.5) / 1_000_000,
            60,
        ),
    ),
)
def test_cost_accounts_for_cached_input_and_provider_output_token_semantics(
    provider, model, usage, expected_cost, billed_output_tokens
) -> None:
    response = replace(
        _response("{}", provider=provider), model=model, model_id=model, usage=usage
    )
    cost = estimate_token_cost(response)
    assert cost["estimated_cost_usd"] == pytest.approx(expected_cost)
    assert cost["cost_output_tokens"] == billed_output_tokens
    assert cost["actual_billed_cost_usd"] is None
    assert "ESTIMATED" in cost["cost_basis"]
    assert cost["cost_excludes_unreported_transport_attempt_usage"] is True


@pytest.mark.parametrize(
    ("provider", "model", "usage"),
    (
        (
            "openai", "gpt-5.6-sol",
            {"input_tokens": 1000, "output_tokens": 50},
        ),
        (
            "gemini", "gemini-3.1-flash-lite",
            {
                "input_tokens": 1000, "output_tokens": 50,
                "reasoning_tokens": 10, "total_tokens": 1060,
            },
        ),
        (
            "gemini", "gemini-3.1-flash-lite",
            {
                "input_tokens": 1000, "cached_input_tokens": 100,
                "output_tokens": 50, "reasoning_tokens": 10, "total_tokens": 1050,
            },
        ),
        (
            "openai", "unpriced-model-override",
            {"input_tokens": 1000, "cached_input_tokens": 100, "output_tokens": 50},
        ),
    ),
)
def test_cost_stays_unknown_when_usage_or_resolved_model_pricing_is_unknown(
    provider, model, usage
) -> None:
    response = replace(
        _response("{}", provider=provider), model=model, model_id=model, usage=usage
    )
    cost = estimate_token_cost(response)
    assert cost["estimated_cost_usd"] is None
    assert cost["actual_billed_cost_usd"] is None
    assert "UNAVAILABLE" in cost["cost_basis"]


def test_unstarted_run_has_no_fabricated_usage_cost_or_latency() -> None:
    summary = summarize_records([], planned_trials=162)
    assert summary["planned_trials"] == summary["pending_trials"] == 162
    assert summary["recorded_trials"] == summary["completed_trials"] == 0
    assert summary["incomplete"] is True
    assert summary["transport_attempts"] == summary["transport_retry_count"] == 0
    for usage in summary["usage"].values():
        assert usage["total"] is None
        assert usage["observed_trials"] == 0
    assert summary["cost"]["estimated_cost_usd"] is None
    assert summary["cost"]["actual_billed_cost_usd"] is None
    assert summary["cost"]["observed_trials"] == 0
    assert summary["latency_ms"]["p50"] is summary["latency_ms"]["p95"] is None
    assert summary["latency_ms"]["observed_trials"] == 0
