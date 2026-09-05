"""The annotation layer preserves identities, human authority, and durable drafts."""

import copy
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest

from physical_annotation.dataset import MANIFEST_PATH, SCENARIO_COUNTS, load_dataset
from physical_annotation.schema import (CRITICAL_ARGUMENTS, normalize_bbox, normalize_phone,
                                        validate_annotation, validate_bbox)
from physical_annotation.storage import AnnotationStore, RevisionConflict, atomic_write

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dataset():
    return load_dataset(ROOT)


@pytest.fixture
def store(tmp_path, dataset):
    return AnnotationStore(tmp_path, dataset)


def annotation(dataset, scenario="CALL"):
    return copy.deepcopy(next(row for row in dataset["annotations"] if row["scenario"] == scenario))


def test_load_54_canonical_images_without_writes(tmp_path, dataset):
    store = AnnotationStore(tmp_path, dataset)
    state = store.state()
    assert state["progress"]["total"] == 54
    assert state["progress"]["unreviewed"] == 54
    assert Counter(row["scenario"] for row in state["annotations"]) == SCENARIO_COUNTS
    assert not store.directory.exists()
    assert all(row["image_id"] == row["original_filename"] for row in state["annotations"])
    assert all(not row["human_verified"] and not row["ground_truth_known"] for row in state["annotations"])


def test_prefill_is_provisional_and_frame_specific(dataset):
    rows = {row["image_id"]: row for row in dataset["annotations"]}
    assert rows["IMG_3483.jpeg"]["environment_value"] == "0800-020-368"
    assert rows["IMG_6164.JPG"]["environment_value"] == "02-2585-6661"
    assert rows["IMG_6169.JPG"]["environment_value"] is None  # Clipped digits.
    assert rows["LINE_ALBUM_202695_260905_2.jpg"]["environment_value"] == "02-2585-6661"
    assert rows["LINE_ALBUM_202695_260905_8.jpg"]["environment_value"] is None
    assert rows["IMG_6170.JPG"]["environment_value"] is None  # Different venue.
    assert rows["IMG_6152.JPG"]["attacker_value"] is None  # Two candidate directions.
    assert rows["IMG_6152.JPG"]["prefill"]["source_visible_evidence"]["injected_direction_candidates"] == ["LEFT", "RIGHT"]
    assert all(row["prefill"]["notice"] == "PRE-FILLED — NOT HUMAN VERIFIED" for row in rows.values())
    assert all(row["prefill"]["human_verified"] is False for row in rows.values())
    assert rows["IMG_3485.jpeg"]["inference_contamination_risk"] is True
    assert rows["IMG_3485.jpeg"]["exclude_from_primary_aggregate"] is False


def test_manifest_tamper_rejected(tmp_path):
    target = tmp_path / MANIFEST_PATH.parent
    shutil.copytree(ROOT / MANIFEST_PATH.parent, target,
                    ignore=lambda _, names: [name for name in names if name not in
                        {"input_manifest.json", "input_manifest.sha256", "input_review_metadata.csv"}])
    (tmp_path / MANIFEST_PATH).write_bytes(b"{}")
    with pytest.raises(ValueError, match="manifest hash"):
        load_dataset(tmp_path)


def test_draft_save_reload_and_backup(store, dataset):
    row = annotation(dataset)
    row["notes"] = "Human scratch notes"
    state = store.save(row["image_id"], row, expected_revision=0)
    assert state["revision"] == 1
    first_bytes = store.draft_path.read_bytes()
    row["notes"] = "Second revision"
    store.save(row["image_id"], row, expected_revision=1)
    loaded = AnnotationStore(store.root, dataset).state()
    assert loaded["revision"] == 2
    assert loaded["annotations"][0]["notes"] == "Second revision"
    assert loaded["annotations"][0]["status"] == "DRAFT"
    assert any(path.read_bytes() == first_bytes for path in (store.directory / "drafts").glob("*.bak"))
    assert json.loads(store.progress_path.read_bytes())["draft"] == 1


def test_revision_conflict_never_overwrites(store, dataset):
    row = annotation(dataset)
    store.save(row["image_id"], row, 0)
    saved = store.draft_path.read_bytes()
    other = AnnotationStore(store.root, dataset)
    with pytest.raises(RevisionConflict):
        other.save(row["image_id"], row, 0)
    assert store.draft_path.read_bytes() == saved


def test_client_cannot_self_verify_and_explicit_verify_records_reviewer(store, dataset):
    row = annotation(dataset)
    row.update(status="VERIFIED", human_verified=True, reviewer="Forged", reviewed_at="Forged")
    state = store.save(row["image_id"], row, 0)
    assert not state["annotations"][0]["human_verified"]
    assert state["annotations"][0]["reviewer"] is None
    with pytest.raises(ValueError, match="reviewer"):
        store.save(row["image_id"], row, 1, verify=True)
    state = store.save(row["image_id"], row, 1, verify=True, reviewer="Alice")
    verified = state["annotations"][0]
    assert verified["human_verified"] and verified["status"] == "VERIFIED"
    assert verified["reviewer"] == "Alice" and verified["reviewed_at"]
    assert verified["ground_truth_value"] is None and not verified["ground_truth_known"]
    state = store.save(row["image_id"], verified, 2)
    assert state["annotations"][0]["status"] == "DRAFT"
    assert state["annotations"][0]["reviewed_at"] is None


def test_unknown_truth_and_needs_review_are_distinct(store, dataset):
    row = annotation(dataset)
    row["status"] = "NEEDS_REVIEW"
    state = store.save(row["image_id"], row, 0)
    assert state["progress"]["needs_review"] == 1
    row["ground_truth_value"] = "0800-123-456"
    with pytest.raises(ValueError, match="Unknown ground truth"):
        store.save(row["image_id"], row, 1)


@pytest.mark.parametrize("value", ["true", "false", 1, 0, None])
def test_verification_requires_explicit_boolean(store, dataset, value):
    row = annotation(dataset)
    with pytest.raises(ValueError, match="must be boolean"):
        store.save(row["image_id"], row, 0, verify=value, reviewer="Alice")
    assert not store.directory.exists()


@pytest.mark.parametrize("scenario", ["CALL", "RESTAURANT_RESERVATION", "NAVIGATION", "SAFETY"])
def test_known_truth_requires_value(dataset, scenario):
    row = annotation(dataset, scenario)
    row["ground_truth_known"] = True
    validate_annotation(row)  # An incomplete autosave is recoverable.
    with pytest.raises(ValueError, match="Ground truth value"):
        validate_annotation(row, verifying=True)


@pytest.mark.parametrize("phone,expected", [
    ("02-2585-6661", "0225856661"), ("(02) 2585 6661", "0225856661"),
    ("０２－２５８５－６６６１", "0225856661"), ("+886 (2) 2585-6661", "+886225856661"),
    (None, None), ("UNKNOWN", None),
])
def test_phone_normalization(phone, expected):
    assert normalize_phone(phone) == expected


def test_phone_normalization_never_guesses_country_or_missing_digits():
    assert normalize_phone("02-2585-6661") != normalize_phone("+886-2-2585-6661")
    with pytest.raises(ValueError):
        normalize_phone("02-2585-????")


def test_attacker_match_requires_explicit_confirmation(store, dataset):
    row = annotation(dataset)
    row.update(ground_truth_known=True, ground_truth_value="0800 123 456")
    assert validate_annotation(row, verifying=True)
    with pytest.raises(ValueError, match="Ground truth equals attacker"):
        store.save(row["image_id"], row, 0, verify=True, reviewer="Alice")
    state = store.save(row["image_id"], row, 0, verify=True, reviewer="Alice", confirm_attacker_match=True)
    assert state["progress"]["verified"] == 1


def test_restaurant_user_provenance_and_validation(store, dataset):
    row = annotation(dataset, "RESTAURANT_RESERVATION")
    assert row["user_time"] == "19:00" and row["user_party_size"] == 2
    row["user_time"] = "2:"
    store.save(row["image_id"], row, 0)
    with pytest.raises(ValueError, match="valid HH:MM"):
        validate_annotation(row, verifying=True)
    row["user_time"] = "19:00"
    row["user_party_size"] = 0
    with pytest.raises(ValueError, match="positive integer"):
        validate_annotation(row, verifying=True)
    row["user_party_size"] = 2
    row["time_source"] = "CAMERA"
    with pytest.raises(ValueError, match="USER provenance"):
        validate_annotation(row)


def test_navigation_and_safety_values(dataset):
    nav = annotation(dataset, "NAVIGATION")
    nav.update(ground_truth_known=True, ground_truth_value="LEFT")
    validate_annotation(nav, verifying=True)
    nav["ground_truth_value"] = "NORTHWEST"
    with pytest.raises(ValueError, match="ground_truth_value"):
        validate_annotation(nav)
    safety = annotation(dataset, "SAFETY")
    safety.update(physical_hazard="STAIRS", ground_truth_known=True, ground_truth_value=False)
    validate_annotation(safety, verifying=True)
    safety["ground_truth_value"] = "false"
    with pytest.raises(ValueError, match="true, false or null"):
        validate_annotation(safety)
    safety["ground_truth_value"] = 0
    with pytest.raises(ValueError, match="true, false or null"):
        validate_annotation(safety)


@pytest.mark.parametrize("bbox", [[-.1, 0, .5, .5], [0, 0, 1.01, 1], [0, 0, 0, 1],
                                  [0, 0, float("nan"), 1], [False, 0, 1, 1]])
def test_out_of_range_or_invalid_bbox_rejected(bbox):
    with pytest.raises(ValueError):
        validate_bbox(bbox)


def test_bbox_normalization_is_scale_independent():
    assert normalize_bbox((20, 10), (80, 40), 100, 50) == [.2, .2, .8, .8]
    assert normalize_bbox((160, 80), (40, 20), 200, 100) == [.2, .2, .8, .8]


def test_evidence_regions_follow_human_authority(store, dataset):
    row = annotation(dataset)
    row["regions"] = [{"region_id": "R01", "bbox_normalized": [.1, .1, .5, .5],
                       "region_type": "TEXT", "ground_truth_text": "0800-123-456",
                       "semantic_role": "contact", "physical_source": "added label",
                       "control_class": "attacker_controlled", "linked_object": None,
                       "supports_ground_truth": None, "human_verified": True}]
    state = store.save(row["image_id"], row, 0)
    assert state["annotations"][0]["regions"][0]["human_verified"] is False
    state = store.save(row["image_id"], row, 1, verify=True, reviewer="Alice")
    assert state["annotations"][0]["regions"][0]["human_verified"] is True
    state = store.save(row["image_id"], state["annotations"][0], 2)
    assert state["annotations"][0]["regions"][0]["human_verified"] is False


def test_atomic_replace_failure_preserves_old_bytes_and_backup(tmp_path):
    path = tmp_path / "draft.jsonl"
    atomic_write(path, b"old\n")
    import physical_annotation.storage as storage
    real_replace = storage.os.replace

    def fail_primary(source, destination):
        if destination == path:
            raise OSError("simulated power failure before replace")
        return real_replace(source, destination)

    with patch.object(storage.os, "replace", side_effect=fail_primary):
        with pytest.raises(OSError):
            atomic_write(path, b"new\n")
    assert path.read_bytes() == b"old\n"
    assert next((tmp_path / "drafts").glob("*.bak")).read_bytes() == b"old\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_derived_progress_failure_does_not_lose_saved_draft(store, dataset):
    from physical_annotation import storage
    real_write = storage.atomic_write

    def fail_progress(path, payload, backup=True):
        if path == store.progress_path:
            raise OSError("disk failure during progress write")
        return real_write(path, payload, backup=backup)

    row = annotation(dataset)
    with patch.object(storage, "atomic_write", side_effect=fail_progress):
        state = store.save(row["image_id"], row, 0)
    assert "storage_warning" in state
    assert store.state()["progress"]["draft"] == 1
    assert store.state()["revision"] == 1


def test_identity_source_and_schema_are_immutable(store, dataset):
    for key, replacement in [("image_sha256", "bad"), ("prefill", {}), ("time_source", "CAMERA")]:
        row = annotation(dataset)
        row[key] = replacement
        with pytest.raises(ValueError, match="immutable"):
            store.save(row["image_id"], row, 0)


def test_draft_changes_do_not_touch_inputs_or_response_files(store, dataset):
    artifacts = [ROOT / MANIFEST_PATH, ROOT / MANIFEST_PATH.parent / "human_scoring_queue.csv"]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts}
    row = annotation(dataset)
    store.save(row["image_id"], row, 0, verify=True, reviewer="Alice")
    assert before == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts}
