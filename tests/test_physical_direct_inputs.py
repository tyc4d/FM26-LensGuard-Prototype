"""Input freeze is bound to reviewed bytes, without granting review labels authority."""

import copy
import hashlib

import pytest

import physical_direct_inputs as inputs


def test_frozen_review_identity_and_contamination():
    manifest = inputs.load_manifest()
    assert len(manifest["records"]) == 54
    flagged = [r for r in manifest["records"] if r["inference_contamination_risk"]]
    assert [r["original_filename"] for r in flagged] == ["IMG_3485.jpeg"]
    assert all(not r["provisional_visible_evidence"]["human_verified"] for r in manifest["records"])


@pytest.mark.parametrize("mutation", ["duplicate", "authority", "contamination", "scenario"])
def test_reject_review_reinterpretation(mutation):
    manifest = copy.deepcopy(inputs.load_manifest())
    if mutation == "duplicate":
        manifest["records"][1]["image_id"] = manifest["records"][0]["image_id"]
    elif mutation == "authority":
        manifest["records"][0]["human_verified"] = True
    elif mutation == "contamination":
        manifest["records"][0]["inference_contamination_risk"] = True
    else:
        manifest["records"][0]["scenario_family"] = "SAFETY"
    with pytest.raises(ValueError):
        inputs.validate_manifest(manifest)


def test_extract_preserves_original_bytes_and_refuses_modified_cache(tmp_path):
    archive = inputs.ROOT / "TestData.zip"
    if not archive.exists():
        pytest.skip("User-supplied original archive is not distributed in Git")
    paths = inputs.extract_originals(archive, tmp_path)
    assert len(paths) == 54
    for row in inputs.load_manifest()["records"]:
        assert hashlib.sha256(paths[row["image_id"]].read_bytes()).hexdigest() == row["sha256"]
    next(iter(paths.values())).write_bytes(b"changed cache")
    with pytest.raises(ValueError, match="Cached original"):
        inputs.extract_originals(archive, tmp_path)


def test_archive_change_stops_before_extraction(tmp_path):
    archive = tmp_path / "wrong.zip"
    archive.write_bytes(b"not the reviewed archive")
    with pytest.raises(ValueError, match="Archive differs"):
        inputs.inspect_archive(archive)
