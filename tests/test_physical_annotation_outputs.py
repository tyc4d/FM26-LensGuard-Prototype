"""Offline checks for lazy, immutable model-output previews and blind review."""

import ast
import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from physical_annotation.dataset import MANIFEST_PATH, load_dataset
from physical_annotation.model_outputs import MODEL_NAMES, outputs_for_image

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / MANIFEST_PATH.parent


def verified(image_id="IMG_3483.jpeg"):
    return {"image_id": image_id, "status": "VERIFIED", "human_verified": True,
            "reviewer": "synthetic-test-reviewer", "reviewed_at": "2026-09-05T00:00:00+00:00"}


@pytest.mark.parametrize("state", [
    {"status": "UNREVIEWED", "human_verified": False},
    {"status": "DRAFT", "human_verified": False},
    {"status": "NEEDS_REVIEW", "human_verified": False},
    {"status": "VERIFIED", "human_verified": False},
    {"status": "DRAFT", "human_verified": True},
])
def test_blind_default_denies_before_reading_any_files(state, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Blind denied request attempted filesystem access")
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    with pytest.raises(PermissionError, match="Blind annotation mode"):
        outputs_for_image(ROOT, "IMG_3483.jpeg", {"image_id": "IMG_3483.jpeg", **state})


def test_manual_blind_override_allows_preview_without_verifying_annotation():
    annotation = {"image_id": "IMG_3483.jpeg", "status": "DRAFT", "human_verified": False}
    before = copy.deepcopy(annotation)
    result = outputs_for_image(ROOT, annotation["image_id"], annotation, blind_mode=False)
    assert len(result["outputs"]) == 5
    assert result["blind_mode"] is False
    assert "reviewer bias" in result["warning"]
    assert annotation == before


def test_initial_dataset_load_does_not_read_model_outputs(monkeypatch):
    original_read = Path.read_bytes
    def checked_read(path):
        assert not ({"records", "raw"} & set(path.parts))
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", checked_read)
    dataset = load_dataset(ROOT)
    assert len(dataset["annotations"]) == 54
    assert all("output_text" not in json.dumps(row) for row in dataset["annotations"])


def test_all_270_outputs_match_preserved_records_and_remain_read_only():
    paths = sorted([*RESULTS.glob("records/*/direct/*.json"), *RESULTS.glob("raw/*/direct/*.json")])
    assert len(paths) == 540
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    for image in load_dataset(ROOT)["records"]:
        image_id = image["image_id"]
        result = outputs_for_image(ROOT, image_id, verified(image_id))
        assert [(r["model_alias"], r["model_name"]) for r in result["outputs"]] == list(MODEL_NAMES)
        for preview in result["outputs"]:
            source = json.loads((ROOT / preview["provenance"]["record_path"]).read_bytes())
            assert preview["action"] == source["action"]
            assert preview["output_text"] == source["output_text"]
            assert preview["parse_valid"] == source["parse_valid"]
            assert preview["schema_valid"] == source["schema_valid"]
            assert preview["completed"] == source["completed"]
            assert preview["provenance"]["image_sha256"] == image["sha256"]
            args = source["arguments"] or {}
            assert preview["critical_argument"] == args.get(preview["critical_argument_name"])
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths} == before


def test_malformed_output_is_not_repaired():
    preview = outputs_for_image(ROOT, "IMG_3483.jpeg", verified())["outputs"][0]
    assert preview["model_name"] == "Gemma"
    assert preview["parse_valid"] is False
    assert preview["output_text"].startswith("```json")
    assert preview["action"] is None
    assert preview["arguments"] is None
    assert preview["critical_argument"] is None


@pytest.fixture
def copied_sources(tmp_path):
    destination = tmp_path / MANIFEST_PATH.parent
    destination.mkdir(parents=True)
    for name in ["input_manifest.json", "input_manifest.sha256", "input_review_metadata.csv", "manifest.json"]:
        shutil.copyfile(RESULTS / name, destination / name)
    for alias, _ in MODEL_NAMES:
        for kind in ["records", "raw"]:
            relative = Path(kind) / alias / "direct" / "IMG_3483.jpeg.json"
            (destination / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(RESULTS / relative, destination / relative)
    return tmp_path


@pytest.mark.parametrize("kind", ["records", "raw"])
def test_modified_preserved_source_is_rejected(copied_sources, kind):
    path = copied_sources / MANIFEST_PATH.parent / kind / "gemma/direct/IMG_3483.jpeg.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        outputs_for_image(copied_sources, "IMG_3483.jpeg", verified())


def test_inventory_must_reference_canonical_manifest(copied_sources):
    path = copied_sources / MANIFEST_PATH.parent / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["immutable_source_sha256"]["input_manifest.json"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="canonical input manifest"):
        outputs_for_image(copied_sources, "IMG_3483.jpeg", verified())


def test_noncanonical_or_mismatched_identity_is_rejected():
    with pytest.raises(ValueError, match="identity"):
        outputs_for_image(ROOT, "IMG_3484.jpeg", verified())
    with pytest.raises(ValueError, match="Unknown canonical"):
        outputs_for_image(ROOT, "../../secret", verified("../../secret"))


def test_annotation_construction_labels_never_enter_output_structures():
    annotation = verified()
    annotation.update(control_class="attacker_controlled", physical_source="secret-construction-source",
                      supports_ground_truth=True, ground_truth_value="SECRET-GROUND-TRUTH")
    annotation["regions"] = [{"control_class": "attacker_controlled"}]
    text = json.dumps(outputs_for_image(ROOT, annotation["image_id"], annotation))
    for forbidden in ["control_class", "supports_ground_truth", "secret-construction-source", "SECRET-GROUND-TRUTH"]:
        assert forbidden not in text


def test_annotation_package_has_no_action_runtime_or_provider_imports():
    forbidden = {"torch", "transformers", "openai", "google", "providers", "provenance", "firewall",
                 "prototype_demo_server", "physical_direct_local", "physical_direct_openai",
                 "benchmark_physical_direct"}
    for path in (ROOT / "physical_annotation").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            assert not {name.split(".")[0] for name in names} & forbidden, path
