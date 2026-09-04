import json
from collections import Counter
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_generated_dataset_has_balanced_semantic_scenarios_and_conditions():
    payload = json.loads((PROJECT_ROOT / "dataset/metadata.json").read_text())
    records = payload["records"]
    core = [row for row in records if row["dataset_partition"] == "CORE"]
    assert payload["semantic_scenario_count"] == 30
    assert payload["dataset_version"] == "lensguard-phase1-dataset-v1.2.0"
    assert payload["generator_version"] == "1.2.0"
    assert payload["record_count"] == 166
    assert payload["image_count"] == 166
    assert len({row["base_scenario_id"] for row in records}) == 30
    assert Counter(row["action_family"] for row in core) == {
        "CALL": 50,
        "OPEN_URL": 50,
        "DIRECTION_ADVICE": 50,
    }
    assert set(Counter(row["condition"] for row in core).values()) == {30}


def test_matched_source_authority_subset_has_required_coverage_and_controls():
    payload = json.loads((PROJECT_ROOT / "dataset/metadata.json").read_text())
    variants = [
        row
        for row in payload["records"]
        if row["dataset_partition"] == "SOURCE_AUTHORITY_MATCHED"
    ]
    assert len(variants) == 16
    required = {
        "official_signage",
        "advertisement",
        "handwritten_note",
        "unverified_notice",
        "camera_unverified",
    }
    for family in ("CALL", "OPEN_URL", "DIRECTION_ADVICE"):
        family_rows = [row for row in variants if row["action_family"] == family]
        assert required <= {row["attack_source"] for row in family_rows}
        assert {row["condition"] for row in family_rows} == {
            "BENIGN_UNTRUSTED_SUBSTITUTION"
        }
        assert len({row["base_scenario_id"] for row in family_rows}) == 1
        assert len({row["attacker_target"] for row in family_rows}) == 1
        assert len({row["render_spec_id"] for row in family_rows}) == 1
        assert all(row["rendered_source_cue"] for row in family_rows)

    qr_rows = [
        row
        for row in variants
        if row["action_family"] == "OPEN_URL"
        and row["attack_source"] == "qr_code_unverified"
    ]
    assert len(qr_rows) == 1
    assert qr_rows[0]["rendered_source_cue"] == "SOURCE TYPE: UNVERIFIED QR CODE"


def test_matched_source_variants_only_change_pixels_in_source_cue_region():
    payload = json.loads((PROJECT_ROOT / "dataset/metadata.json").read_text())
    for family in ("CALL", "OPEN_URL", "DIRECTION_ADVICE"):
        variants = [
            row
            for row in payload["records"]
            if row["dataset_partition"] == "SOURCE_AUTHORITY_MATCHED"
            and row["action_family"] == family
        ]
        baseline_path = PROJECT_ROOT / variants[0]["image_path"]
        with Image.open(baseline_path) as baseline_image:
            baseline = baseline_image.copy()
        for row in variants[1:]:
            with Image.open(PROJECT_ROOT / row["image_path"]) as candidate:
                difference = _difference_bbox(baseline, candidate)
            assert difference is not None
            left, top, right, bottom = difference
            assert 626 <= left < right <= 1142
            assert 570 <= top < bottom <= 595


def _difference_bbox(first: Image.Image, second: Image.Image):
    from PIL import ImageChops

    return ImageChops.difference(first, second).getbbox()


def test_dataset_images_and_oracle_value_maps_exist():
    payload = json.loads((PROJECT_ROOT / "dataset/metadata.json").read_text())
    sample = payload["records"][1]
    mapping = sample["argument_provenance"][sample["critical_argument_name"]]
    assert mapping[sample["official_value"]] == sample["official_source"]
    assert mapping[sample["attacker_target"]] == sample["attack_source"]

    with Image.open(PROJECT_ROOT / sample["image_path"]) as image:
        assert image.size == (1200, 760)
        assert image.mode == "RGB"


def test_core_records_have_neutral_source_footers_in_metadata():
    payload = json.loads((PROJECT_ROOT / "dataset/metadata.json").read_text())
    core = [row for row in payload["records"] if row["dataset_partition"] == "CORE"]
    assert all(row["rendered_source_cue"] is None for row in core)
