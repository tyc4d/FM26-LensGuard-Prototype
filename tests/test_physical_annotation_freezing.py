"""Freeze tests use temporary annotation directories, never scientific outputs."""

import copy
import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from physical_annotation.dataset import load_dataset
from physical_annotation.freezing import export_csv, export_jsonl, freeze, validate_dataset
from physical_annotation.storage import AnnotationStore, RevisionConflict, atomic_write, canonical_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def store(tmp_path):
    dataset = load_dataset(ROOT)
    # Synthetic test reviewer decisions, stored exclusively below pytest tmp_path.
    for row in dataset["annotations"]:
        row.update(status="VERIFIED", human_verified=True, reviewer="TEST FIXTURE",
                   reviewed_at="2026-01-01T00:00:00+00:00", updated_at="2026-01-01T00:00:00+00:00")
    result = AnnotationStore(ROOT, dataset, directory=tmp_path / "annotations")
    atomic_write(result.draft_path,
                 b"".join(canonical_json({**row, "_revision": 1}) for row in dataset["annotations"]))
    return result


def change(store, **fields):
    state = store.state()
    row = copy.deepcopy(state["annotations"][0])
    row.update(fields)
    return store.save(row["image_id"], row, state["revision"])


def test_exports_are_deterministic_and_include_all_54_images(store, tmp_path):
    state = store.state()
    payload = export_jsonl(state)
    reversed_state = {**state, "revision": 999, "annotations": list(reversed(state["annotations"]))}
    assert export_jsonl(reversed_state) == payload
    assert len(payload.splitlines()) == 54
    assert all("_revision" not in json.loads(line) for line in payload.splitlines())
    first = freeze(store, 1, confirm=True)
    other = AnnotationStore(ROOT, store.dataset, directory=tmp_path / "independent")
    second = freeze(other, 0, confirm=True)
    expected = hashlib.sha256(payload).hexdigest()
    assert first["sha256"] == second["sha256"] == expected
    assert Path(first["files"]["jsonl"]).read_bytes() == payload
    assert Path(first["files"]["sha256"]).read_text() == f"{expected}  ground_truth_v1.jsonl\n"


def test_csv_preserves_nested_fields_and_prevents_spreadsheet_formulas(store):
    state = store.state()
    row = state["annotations"][0]
    row.update(notes="  =HYPERLINK(\"https://example.invalid\")", attacker_value="+886225856661",
               restaurant_identity="餐廳,分店", regions=[{"ground_truth_text": "@value"}])
    exported = list(csv.DictReader(io.StringIO(export_csv(state).decode("utf-8"))))
    actual = next(item for item in exported if item["image_id"] == row["image_id"])
    assert actual["notes"].startswith("'  =")
    assert actual["attacker_value"].startswith("'+")
    assert actual["restaurant_identity"] == "餐廳,分店"
    assert json.loads(actual["regions"]) == row["regions"]
    assert set(actual) == set(row)
    assert json.loads(export_jsonl(state).splitlines()[0])["notes"] != "'" + row["notes"]


@pytest.mark.parametrize("confirm", [False, None, 1, "true"])
def test_freeze_requires_explicit_boolean_confirmation(store, confirm):
    before = set(store.directory.iterdir())
    with pytest.raises(ValueError, match="explicit human confirmation"):
        freeze(store, 1, confirm=confirm)
    assert set(store.directory.iterdir()) == before


def test_freeze_rejects_stale_preview(store):
    change(store, notes="human edit")
    with pytest.raises(RevisionConflict, match="confirm again"):
        freeze(store, 1, confirm=True)
    assert not list(store.directory.glob("ground_truth_v*"))


def test_unreviewed_and_draft_cannot_be_acknowledged_away(store, tmp_path):
    fresh = AnnotationStore(ROOT, load_dataset(ROOT), directory=tmp_path / "fresh")
    with pytest.raises(ValueError, match="Freeze blocked"):
        freeze(fresh, 0, confirm=True, acknowledge_unresolved=True)
    change(store, notes="unfinished")
    report = validate_dataset(store)
    assert report["draft"] == report["unresolved"] == 1
    assert not report["can_freeze"]
    with pytest.raises(ValueError, match="Freeze blocked"):
        freeze(store, 2, confirm=True, acknowledge_unresolved=True)


def test_needs_review_is_explicitly_acknowledged_and_remains_unverified(store):
    state = change(store, status="NEEDS_REVIEW", notes="Route independently unknown")
    report = validate_dataset(store)
    assert report["verified"] == 53
    assert report["unresolved"] == report["needs_review"] == 1
    assert report["can_freeze"] and report["requires_unresolved_acknowledgement"]
    with pytest.raises(ValueError, match="Explicitly acknowledge"):
        freeze(store, state["revision"], confirm=True)
    result = freeze(store, state["revision"], confirm=True, acknowledge_unresolved=True)
    assert result["unresolved_acknowledged"] is True
    assert result["verified_count"] == 53
    frozen = [json.loads(line) for line in Path(result["files"]["jsonl"]).read_bytes().splitlines()]
    unresolved = next(row for row in frozen if row["status"] == "NEEDS_REVIEW")
    assert unresolved["human_verified"] is False
    assert unresolved["ground_truth_value"] is None


def test_exclusion_requires_reason_and_does_not_verify_labels(store):
    state = change(store, exclude_from_primary_aggregate=True, exclusion_reason="")
    report = validate_dataset(store)
    assert report["excluded"] == 1
    assert report["errors"][0]["error"].startswith("Exclusion requires")
    with pytest.raises(ValueError, match="Freeze blocked"):
        freeze(store, state["revision"], confirm=True)
    state = change(store, exclusion_reason="Human reviewer excludes experiment-screen contamination")
    result = freeze(store, state["revision"], confirm=True)
    assert result["excluded_count"] == 1
    assert result["verified_count"] == 53
    assert result["unresolved_count"] == 0
    assert store.state()["annotations"][0]["status"] == "DRAFT"


def test_v1_is_immutable_and_duplicate_freeze_rejected(store):
    result = freeze(store, 1, confirm=True)
    before = {key: Path(path).read_bytes() for key, path in result["files"].items()}
    for path in result["files"].values():
        assert Path(path).stat().st_mode & 0o222 == 0
    with pytest.raises(ValueError, match="change_reason"):
        freeze(store, 1, confirm=True)
    with pytest.raises(ValueError, match="No annotation changes"):
        freeze(store, 1, confirm=True, change_reason="Repeated button click")
    assert {key: Path(path).read_bytes() for key, path in result["files"].items()} == before
    assert not (store.directory / "ground_truth_v2.jsonl").exists()


def test_v2_records_parent_reason_and_only_changed_image_ids(store):
    first = freeze(store, 1, confirm=True)
    first_payload = Path(first["files"]["jsonl"]).read_bytes()
    state = store.state()
    corrected = copy.deepcopy(state["annotations"][0])
    corrected["notes"] = "Human correction: independently established reservation contact is still unknown."
    state = store.save(corrected["image_id"], corrected, state["revision"],
                       verify=True, reviewer="TEST FIXTURE")
    with pytest.raises(ValueError, match="change_reason"):
        freeze(store, state["revision"], confirm=True)
    second = freeze(store, state["revision"], confirm=True, change_reason="Clarify independently known facts")
    assert second["version"] == "v2" and second["parent_version"] == "v1"
    assert second["changed_image_ids"] == [corrected["image_id"]]
    assert second["change_reason"] == "Clarify independently known facts"
    assert Path(first["files"]["jsonl"]).read_bytes() == first_payload
    assert second["sha256"] != first["sha256"]


def test_reverification_timestamp_alone_is_not_a_scientific_correction(store):
    freeze(store, 1, confirm=True)
    state = store.state()
    row = state["annotations"][0]
    state = store.save(row["image_id"], row, 1, verify=True, reviewer="TEST FIXTURE")
    with pytest.raises(ValueError, match="No annotation changes"):
        freeze(store, state["revision"], confirm=True, change_reason="Reverified same labels")


def test_interrupted_publish_reserves_version_and_never_overwrites_partial(store, monkeypatch):
    import physical_annotation.freezing as module

    original_link = module.os.link

    def interrupted_link(source, target):
        if str(target).endswith(".sha256"):
            raise OSError("simulated interrupted disk publication")
        original_link(source, target)

    monkeypatch.setattr(module.os, "link", interrupted_link)
    with pytest.raises(OSError, match="simulated interrupted"):
        freeze(store, 1, confirm=True)
    partial = store.directory / "ground_truth_v1.jsonl"
    original = partial.read_bytes()
    assert not (store.directory / "ground_truth_v1_manifest.json").exists()
    assert (store.directory / ".ground_truth_v1.reserve").exists()
    monkeypatch.setattr(module.os, "link", original_link)
    with pytest.raises(ValueError, match="Incomplete freeze v1 reserves this version"):
        freeze(store, 1, confirm=True)
    assert partial.read_bytes() == original
    assert not (store.directory / "ground_truth_v2.jsonl").exists()


def test_tampered_frozen_version_blocks_corrections(store):
    result = freeze(store, 1, confirm=True)
    path = Path(result["files"]["jsonl"])
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="integrity check failed"):
        freeze(store, 1, confirm=True, change_reason="Must not overwrite tampered history")
