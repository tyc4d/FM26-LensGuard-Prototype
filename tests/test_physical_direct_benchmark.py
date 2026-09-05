"""Physical DIRECT orchestration regressions, with no API or model execution."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

import benchmark_physical_direct as benchmark
from cloud_baseline_store import CloudResultStore, read_json


VALID_TEXT = '{"action":"NONE","arguments":{},"decision_text":"Need more information."}'


class FakeProvider:
    def __init__(self, alias="openai", *, malformed=False, failed_call=None,
                 stop=False, response_model=None):
        self.provider = alias
        self.model = benchmark.MODEL_IDS[alias]
        self.provider_config = {"mock_transport": True, "tools": []}
        if alias == "gemini":
            self.provider_config["request_delay_seconds"] = 8
        self.calls = []
        self.malformed, self.failed_call, self.stop = malformed, failed_call, stop
        self.response_model = response_model
        self.sleeps = []
        self.request_delay_seconds = 8
        self._bounded_sleep = self.sleeps.append

    def infer(self, request):
        self.calls.append(request)
        failed = self.failed_call == len(self.calls)
        raw = "malformed scientific output" if self.malformed else VALID_TEXT
        return {
            "provider": self.provider, "model": self.response_model or self.model,
            "model_id": self.response_model or self.model,
            "api_interface": "mock-direct", "timestamp_utc": benchmark.utcnow(),
            "raw_response": {"exact_text": raw}, "output_text": raw,
            "completed": not failed, "error_type": "LOCAL_RUNTIME_ERROR" if failed else None,
            "error_detail": None, "http_status": 503 if failed else 200,
            "api_status": "FAILED" if failed else "completed", "request_id": "fixture-id",
            "latency_ms": 10.0, "usage": {}, "provider_config": copy.deepcopy(self.provider_config),
            "scientific_attempt": 1, "transport_attempts": 1,
            "stop_provider": failed and self.stop, "hard_quota": False,
            "rate_limit_events": 0, "total_backoff_seconds": 0,
            "model_revision": benchmark.LOCAL_MODELS.get(self.provider, {}).get("revision"),
        }


@pytest.fixture(autouse=True)
def never_load_actual_credentials(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)


@pytest.fixture
def fixture_run(tmp_path, monkeypatch):
    """Four software images carrying deliberately private review-only metadata."""
    rows, paths, sleeps = [], {}, []
    monkeypatch.setattr(benchmark.time, "sleep", sleeps.append)
    for index, case in enumerate(read_json(benchmark.SMOKE_CONFIG_PATH)["cases"]):
        filename = case["filename"]
        path = tmp_path / filename
        Image.new("RGB", (12, 8), (index * 30, 80, 90)).save(path)
        paths[filename] = path
        rows.append({
            "image_id": filename, "original_filename": filename,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "scenario_family": case["scenario"], "quality_class": "REVIEW_QUALITY_ONLY",
            "inference_contamination_risk": False,
            "review_notes": "DO_NOT_SEND_REVIEW_ANNOTATION",
            "provisional_visible_evidence": {"phone": "DO_NOT_SEND_CANDIDATE_PHONE"},
            "provisional_attack_mode": "DO_NOT_SEND_ATTACK_LABEL",
            "scene_group": "DO_NOT_SEND_SCENE_GROUP",
        })
    manifest = {"records": rows}
    manifest_path = tmp_path / "input_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(benchmark, "INPUT_MANIFEST", manifest_path)
    monkeypatch.setattr(benchmark, "load_manifest", lambda: copy.deepcopy(manifest))
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    # Reporting is outside this runner test. Keep structural parsing real.
    reporter = SimpleNamespace(
        estimate_cost=lambda response: {"estimated_cost_usd": None},
        summarize_records=lambda records, planned: {
            "planned_trials": planned, "recorded_trials": len(records),
            "completed_trials": sum(row["completed"] for row in records),
        },
        render_model_report=lambda summary: json.dumps(summary),
    )
    monkeypatch.setitem(sys.modules, "physical_direct_reporting", reporter)
    return SimpleNamespace(root=tmp_path / "run", paths=paths, rows=rows, sleeps=sleeps)


def run(provider, fixtures, *, resume=False):
    return benchmark.run_provider(
        provider.provider, provider, "smoke", root=fixtures.root,
        image_paths=fixtures.paths, resume=resume,
    )


@pytest.mark.parametrize("mode,count", [("--full", 54), ("--smoke", 4)])
def test_dry_run_reports_real_counts_and_constructs_no_provider(mode, count, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run attempted provider construction, extraction or subprocess execution")

    monkeypatch.setattr(benchmark, "construct_provider", forbidden)
    monkeypatch.setattr(benchmark, "extract_originals", forbidden)
    monkeypatch.setattr(benchmark.subprocess, "run", forbidden)
    monkeypatch.setattr(sys, "argv", ["benchmark_physical_direct.py", "--model", "all", mode, "--dry-run"])
    assert benchmark.main() == 0
    lines = capsys.readouterr().out.splitlines()
    plans = [json.loads(line) for line in lines if line.startswith("{")]
    assert len(plans) == 5
    assert {p["model_alias"]: p["model"] for p in plans} == benchmark.MODEL_IDS
    assert all(p["case_count"] == p["planned_requests"] == count and p["arm_count"] == 1 for p in plans)
    assert f"Total planned requests: {5 * count}; network requests: 0" in lines


def test_all_models_receive_identical_bytes_prompt_schema_and_no_review_labels(fixture_run):
    observed = []
    for alias in benchmark.MODEL_IDS:
        provider = FakeProvider(alias)
        summary = run(provider, fixture_run)
        assert summary["completed_trials"] == len(provider.calls) == 4
        observed.append([
            (request.case_id, request.image_path.read_bytes(), request.prompt, request.response_schema)
            for request in provider.calls
        ])
        assert all(request.arm == "DIRECT" for request in provider.calls)
        for request in provider.calls:
            assert not hasattr(request, "registry")
            assert "DO_NOT_SEND" not in request.prompt
            assert "REVIEW_QUALITY_ONLY" not in request.prompt
        plan = read_json(fixture_run.root / "plans" / f"{alias}.json")
        assert plan["arms"] == ["DIRECT"] and plan["scientific_attempts_per_trial"] == 1
        assert plan["ground_truth_frozen"] is False
        assert "DO_NOT_SEND" not in json.dumps(plan)
        assert benchmark.validate_provider(fixture_run.root, alias, require_complete=True)["valid"]
    assert all(requests == observed[0] for requests in observed[1:])


def test_malformed_response_is_preserved_once_and_never_repaired(fixture_run):
    provider = FakeProvider(malformed=True)
    run(provider, fixture_run)
    assert len(provider.calls) == 4
    rows = [json.loads(line) for line in (fixture_run.root / "openai_normalized.jsonl").read_text().splitlines()]
    assert all(row["completed"] and not row["schema_valid"] for row in rows)
    assert all(row["error_type"] == "MALFORMED_JSON" for row in rows)
    assert all(row["scientific_attempt"] == row["transport_attempts"] == 1 for row in rows)
    for path in (fixture_run.root / "raw").glob("*/*/*.json"):
        assert read_json(path)["response"]["raw_response"]["exact_text"] == "malformed scientific output"


def test_completed_and_failed_records_are_immutable_and_both_skipped_on_resume(fixture_run):
    first = FakeProvider(failed_call=2)
    run(first, fixture_run)
    immutable = {p: p.read_bytes() for folder in ("raw", "records", "started", "plans")
                 for p in (fixture_run.root / folder).rglob("*.json")}
    resumed = FakeProvider()
    summary = run(resumed, fixture_run, resume=True)
    assert resumed.calls == [] and summary["recorded_trials"] == 4
    assert summary["completed_trials"] == 3
    assert all(path.read_bytes() == content for path, content in immutable.items())
    manifest = read_json(fixture_run.root / "openai_manifest.json")
    assert len(manifest["failed_trial_ids"]) == 1 and not manifest["pending_trial_ids"]
    with pytest.raises(FileExistsError, match="already exists"):
        run(FakeProvider(), fixture_run)


def test_raw_preserved_before_parse_crash_and_resume_recovers_without_resending(fixture_run, monkeypatch):
    original = benchmark.parse_output
    first = FakeProvider()

    def crash_after_preservation(text):
        paths = list((fixture_run.root / "raw" / "openai" / "direct").glob("*.json"))
        assert len(paths) == 1 and read_json(paths[0])["response"]["output_text"] == text
        raise RuntimeError("Simulated parser interruption")

    monkeypatch.setattr(benchmark, "parse_output", crash_after_preservation)
    with pytest.raises(RuntimeError, match="parser interruption"):
        run(first, fixture_run)
    assert len(first.calls) == 1
    raw = next((fixture_run.root / "raw").glob("*/*/*.json"))
    original_bytes = raw.read_bytes()
    assert not (fixture_run.root / ".openai.running").exists()
    monkeypatch.setattr(benchmark, "parse_output", original)
    resumed = FakeProvider()
    summary = run(resumed, fixture_run, resume=True)
    assert summary["completed_trials"] == 4 and len(resumed.calls) == 3
    assert first.calls[0].case_id not in {r.case_id for r in resumed.calls}
    assert raw.read_bytes() == original_bytes


def test_started_without_raw_is_ambiguous_and_never_resent(fixture_run):
    provider = FakeProvider()
    plan = benchmark.make_plan("openai", "smoke", provider.provider_config)
    store = CloudResultStore(fixture_run.root, "openai", plan)
    case = plan["cases"][0]
    store.begin(case["case_id"], "DIRECT", benchmark.request_hash(plan, case), benchmark.utcnow())
    with pytest.raises(FileExistsError, match="already started"):
        run(provider, fixture_run, resume=True)
    assert provider.calls == []
    assert store.manifest()["interrupted_trial_ids"] == [case["case_id"] + "/DIRECT"]


def test_preserved_fatal_stop_refuses_remaining_requests_and_resume(fixture_run):
    provider = FakeProvider(failed_call=1, stop=True)
    summary = run(provider, fixture_run)
    assert len(provider.calls) == summary["recorded_trials"] == 1
    resumed = FakeProvider()
    with pytest.raises(RuntimeError, match="preserved stop condition"):
        run(resumed, fixture_run, resume=True)
    assert resumed.calls == []
    manifest = read_json(fixture_run.root / "openai_manifest.json")
    assert len(manifest["pending_trial_ids"]) == 3
    assert manifest["stop_reasons"] == ["LOCAL_RUNTIME_ERROR"]


def test_image_hash_mismatch_stops_before_any_inference(fixture_run):
    next(iter(fixture_run.paths.values())).write_bytes(b"changed fixture bytes")
    provider = FakeProvider()
    with pytest.raises(ValueError, match="image hash mismatch"):
        run(provider, fixture_run)
    assert provider.calls == []
    assert not list((fixture_run.root / "raw").glob("*/*/*.json"))


@pytest.mark.parametrize("delay", ["0", "7.99", "-1", "nan", "inf"])
def test_gemini_rejects_less_than_eight_or_nonfinite_delay(delay, monkeypatch):
    monkeypatch.setenv("GEMINI_REQUEST_DELAY_SECONDS", delay)
    monkeypatch.setattr(benchmark, "require_secret_safety", lambda: None)
    constructor = Mock(side_effect=AssertionError("Invalid pacing constructed a provider"))
    monkeypatch.setitem(sys.modules, "providers.gemini_vlm", SimpleNamespace(GeminiProvider=constructor))
    with pytest.raises(ValueError, match="at least 8 seconds"):
        benchmark.construct_provider("gemini")
    constructor.assert_not_called()


@pytest.mark.parametrize("alias", ["openai", "gemini"])
def test_cloud_construction_pins_requested_model_id_despite_environment(alias, monkeypatch):
    monkeypatch.setattr(benchmark, "require_secret_safety", lambda: None)
    monkeypatch.setenv("GEMINI_REQUEST_DELAY_SECONDS", "8")
    monkeypatch.setenv(alias.upper() + "_MODEL", "unapproved-replacement-model")
    constructor = Mock(return_value=object())
    name = "OpenAIProvider" if alias == "openai" else "GeminiProvider"
    monkeypatch.setitem(sys.modules, f"providers.{alias}_vlm", SimpleNamespace(**{name: constructor}))
    benchmark.construct_provider(alias)
    constructor.assert_called_once_with(model=benchmark.MODEL_IDS[alias])


def test_wrong_provider_model_is_rejected_before_inference(fixture_run):
    provider = FakeProvider()
    provider.model = "unapproved-replacement-model"
    with pytest.raises(ValueError, match="[Mm]odel"):
        run(provider, fixture_run)
    assert provider.calls == []


def test_wrong_response_model_is_preserved_and_rejected_without_relabeling(fixture_run):
    provider = FakeProvider(response_model="unapproved-replacement-model")
    with pytest.raises(ValueError, match="[Mm]odel"):
        run(provider, fixture_run)
    assert len(provider.calls) == 1
    raw = next((fixture_run.root / "raw").glob("*/*/*.json"))
    assert read_json(raw)["response"]["model"] == "unapproved-replacement-model"
    assert not list((fixture_run.root / "records").glob("*/*/*.json"))


def test_gemini_resume_restores_minimum_spacing_before_first_new_request(fixture_run, monkeypatch):
    original = benchmark.parse_output
    monkeypatch.setattr(benchmark, "parse_output", lambda text: (_ for _ in ()).throw(RuntimeError("pause")))
    with pytest.raises(RuntimeError, match="pause"):
        run(FakeProvider("gemini"), fixture_run)
    monkeypatch.setattr(benchmark, "parse_output", original)
    fixture_run.sleeps.clear()
    resumed = FakeProvider("gemini")
    infer = resumed.infer

    def check_spacing(request):
        assert sum(fixture_run.sleeps) >= 8
        return infer(request)

    resumed.infer = check_spacing
    run(resumed, fixture_run, resume=True)
    assert len(resumed.calls) == 3
