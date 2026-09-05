"""Benchmark execution and preservation tests; no live inference."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

import benchmark_cloud_phase3_6 as benchmark
from cloud_baseline_contracts import load_cases
from cloud_baseline_store import CloudResultStore, read_json, write_new
from providers.base_cloud_vlm import CloudResponse


class FakeProvider:
    provider = "openai"
    model = "gpt-5.6-sol"
    provider_config = {"tools": [], "test_transport": True}

    def __init__(self, *, fail=False):
        self.calls = []
        self.fail = fail

    def infer(self, request):
        self.calls.append(request)
        payload = {"action": "NONE", "arguments": {}}
        if request.arm == "GROUNDED":
            payload["argument_evidence_refs"] = {}
        return CloudResponse(
            provider=self.provider, model=self.model, model_id=self.model,
            api_interface="mock", timestamp_utc="2026-09-05T00:00:00+00:00",
            raw_response={"output_text": json.dumps(payload)}, output_text=json.dumps(payload),
            completed=not self.fail, error_type="RATE_LIMIT_EXHAUSTED" if self.fail else None,
            error_detail=None, http_status=429 if self.fail else 200,
            api_status="RESOURCE_EXHAUSTED" if self.fail else "completed", request_id="mock-id",
            latency_ms=10.0, usage={}, provider_config=self.provider_config,
            transport_attempts=1, rate_limit_events=int(self.fail),
            stop_provider=self.fail, hard_quota=self.fail,
        )


@pytest.fixture
def cases():
    return load_cases(smoke=True)


@pytest.fixture
def run_root(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    return tmp_path / "run"


def run(provider, cases, root, resume=False):
    return benchmark.run_provider(provider, cases, root, "test-cloud", smoke=True, resume=resume)


def test_full_dry_run_plans_324_and_constructs_no_clients(monkeypatch, capsys):
    class NeverClient:
        default_model = "test-model"
        model_environment = "UNSET_TEST_CLOUD_MODEL"

        def __init__(self):
            pytest.fail("Dry run constructed an API client")

    monkeypatch.setattr(benchmark, "PROVIDERS", {"openai": NeverClient, "gemini": NeverClient})
    assert benchmark.main(["--provider", "all", "--full", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert output.count("case_count=81 arm_count=2 planned_request_count=162") == 2
    assert "total_planned_request_count=324" in output
    assert "API_KEY present" not in output


def test_smoke_exact_trial_set_raw_preservation_and_resume_no_calls(cases, run_root):
    provider = FakeProvider()
    manifest = run(provider, cases, run_root)
    assert manifest["planned_trials"] == manifest["completed_trials"] == len(provider.calls) == 6
    assert not manifest["pending_trial_ids"] and not manifest["failed_trial_ids"]
    assert len(list((run_root / "raw").glob("*/*/*.json"))) == 6
    assert len((run_root / "openai_normalized.jsonl").read_text().splitlines()) == 6
    before = {p: p.read_bytes() for p in (run_root / "raw").glob("*/*/*.json")}
    resumed = FakeProvider()
    run(resumed, cases, run_root, resume=True)
    assert resumed.calls == []
    assert all(p.read_bytes() == content for p, content in before.items())
    with pytest.raises(FileExistsError):
        run(FakeProvider(), cases, run_root)


def test_hard_quota_records_failure_and_resumes_only_missing(cases, run_root):
    failed = FakeProvider(fail=True)
    manifest = run(failed, cases, run_root)
    assert len(failed.calls) == 1
    assert manifest["incomplete_due_to_quota"] and len(manifest["pending_trial_ids"]) == 5
    assert len(manifest["failed_trial_ids"]) == 1
    resumed = FakeProvider()
    manifest = run(resumed, cases, run_root, resume=True)
    assert len(resumed.calls) == 5
    assert manifest["recorded_trials"] == 6 and manifest["completed_trials"] == 5
    assert len(manifest["failed_trial_ids"]) == 1 and not manifest["pending_trial_ids"]


def test_model_unavailable_resume_cannot_continue_provider(cases, run_root):
    provider = FakeProvider(fail=True)
    original = provider.infer
    provider.infer = lambda request: replace(original(request), error_type="MODEL_UNAVAILABLE", hard_quota=False)
    run(provider, cases, run_root)
    resumed = FakeProvider()
    run(resumed, cases, run_root, resume=True)
    assert resumed.calls == []


def test_changed_plan_or_tampered_raw_stops_before_inference(cases, run_root):
    run(FakeProvider(), cases, run_root)
    changed = FakeProvider()
    changed.model = "different-model"
    with pytest.raises(ValueError, match="Resume plan differs"):
        run(changed, cases, run_root, resume=True)
    raw = next((run_root / "raw").glob("*/*/*.json"))
    raw.write_text("{}")
    same = FakeProvider()
    with pytest.raises(ValueError, match="artifact changed"):
        run(same, cases, run_root, resume=True)
    assert same.calls == changed.calls == []


def test_interrupted_send_is_not_resent(cases, run_root):
    provider = FakeProvider()
    plan = benchmark.build_plan("openai", provider.model, provider.provider_config, cases, "test-cloud", smoke=True)
    store = CloudResultStore(run_root, "openai", plan)
    store.begin(cases[0]["scenario_id"], "ACTION_ONLY", "request-hash", "timestamp")
    with pytest.raises(FileExistsError, match="already started"):
        run(provider, cases, run_root, resume=True)
    assert provider.calls == []
    assert store.manifest()["interrupted_trial_ids"]


def test_raw_response_survives_evaluator_failure_and_resume_uses_no_extra_call(cases, run_root, monkeypatch):
    original = benchmark.normalize_response
    monkeypatch.setattr(benchmark, "normalize_response", lambda **kwargs: (_ for _ in ()).throw(ValueError("test evaluator failure")))
    provider = FakeProvider()
    with pytest.raises(ValueError, match="test evaluator failure"):
        run(provider, cases, run_root)
    assert len(provider.calls) == 1
    assert len(list((run_root / "raw").glob("*/*/*.json"))) == 1
    monkeypatch.setattr(benchmark, "normalize_response", original)
    resumed = FakeProvider()
    manifest = run(resumed, cases, run_root, resume=True)
    assert len(resumed.calls) == 5 and manifest["completed_trials"] == 6


def test_write_new_never_overwrites(tmp_path):
    path = tmp_path / "scientific.json"
    write_new(path, {"response": "original"})
    with pytest.raises(FileExistsError):
        write_new(path, {"response": "replacement"})
    assert read_json(path) == {"response": "original"}


def test_resumed_gemini_obeys_configured_spacing(cases, run_root):
    provider = FakeProvider(fail=True)
    provider.provider = "gemini"
    provider.model = "gemini-3.1-flash-lite"
    original = provider.infer
    def response(request):
        result = original(request)
        if len(provider.calls) == 1:
            return replace(result, completed=True, error_type=None, stop_provider=False, hard_quota=False)
        return result
    provider.infer = response
    run(provider, cases, run_root)
    resumed = FakeProvider()
    resumed.provider, resumed.model = provider.provider, provider.model
    resumed.request_delay_seconds = 12
    sleeps = []
    resumed._bounded_sleep = sleeps.append
    run(resumed, cases, run_root, resume=True)
    assert sleeps == [14.0] and len(resumed.calls) == 4
