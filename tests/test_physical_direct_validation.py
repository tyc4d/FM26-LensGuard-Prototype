"""Validate partial evidence without inference or mutations of scientific files."""

import json

import pytest

import validate_physical_direct as validation


def test_current_validation_reports_absent_models_without_requiring_completion(tmp_path, monkeypatch):
    root = tmp_path / "results"
    plan = root / "plans/gemma.json"
    plan.parent.mkdir(parents=True)
    plan.write_text(json.dumps({"provider": "gemma"}))
    before = plan.read_bytes()
    monkeypatch.setattr(validation, "RESULT_ROOT", root)
    original = {"image_id": "image.jpg", "sha256": "hash"}
    monkeypatch.setattr(validation, "load_manifest", lambda: {"records": [original]})
    monkeypatch.setattr(validation, "inspect_archive", lambda _: [original])
    calls = []

    def validate_provider(path, alias):
        calls.append((path, alias))
        return {"valid": True, "recorded_trials": 1, "missing_trials": 53}

    monkeypatch.setattr(validation, "validate_provider", validate_provider)
    report = validation.validate_current()
    assert report["valid"] is True
    assert report["completeness_required"] is False
    assert report["full_recorded_trials"] == 1
    assert report["full_missing_trials"] == 269
    assert report["full_models_without_plan"] == ["minicpm", "qwen", "openai", "gemini"]
    assert calls == [(root, "gemma")]
    assert plan.read_bytes() == before
    assert list(root.rglob("*.json")) == [plan]


def test_current_validation_rejects_archive_metadata_mismatch(monkeypatch):
    monkeypatch.setattr(validation, "load_manifest", lambda: {"records": [{"width": 100}]})
    monkeypatch.setattr(validation, "inspect_archive", lambda _: [{"width": 200}])
    with pytest.raises(ValueError, match="Archive metadata differs"):
        validation.validate_current()
