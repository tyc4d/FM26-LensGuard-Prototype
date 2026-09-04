import json
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "dataset_phase2/metadata.json"
STANDARD_CONDITIONS = {
    "CLEAN_TRUSTED",
    "BENIGN_UNTRUSTED_SUBSTITUTION",
    "AUTHORITY_IMPERSONATION",
    "OBVIOUS_INJECTION_CONTROL",
    "EXPLICIT_USER_OVERRIDE",
}
SPECIAL_CONDITIONS = {
    "NO_VERIFIED_GROUND_TRUTH",
    "TRUSTED_BUT_CONFLICTING_UPDATE",
}


def _payload():
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def test_phase2_exact_counts_and_balance():
    payload = _payload()
    records = payload["records"]
    assert payload["dataset_version"] == "lensguard-phase2-dataset-v1.1.0"
    assert payload["generator_version"] == "1.1.0"
    assert payload["semantic_scenario_count"] == 15
    assert payload["record_count"] == payload["image_count"] == 81
    assert payload["partition_counts"] == {"STANDARD": 75, "SPECIAL_CONTROL": 6}
    assert len({row["base_scenario_id"] for row in records}) == 15
    assert Counter(row["action_family"] for row in records) == {
        "CALL": 27,
        "OPEN_URL": 27,
        "DIRECTION_ADVICE": 27,
    }


def test_phase2_standard_and_special_condition_coverage():
    records = _payload()["records"]
    standard = [row for row in records if row["dataset_partition"] == "STANDARD"]
    special = [row for row in records if row["dataset_partition"] == "SPECIAL_CONTROL"]
    assert Counter(row["condition"] for row in standard) == {
        condition: 15 for condition in STANDARD_CONDITIONS
    }
    for family in ("CALL", "OPEN_URL", "DIRECTION_ADVICE"):
        family_standard = [row for row in standard if row["action_family"] == family]
        assert Counter(row["condition"] for row in family_standard) == {
            condition: 5 for condition in STANDARD_CONDITIONS
        }
        family_special = [row for row in special if row["action_family"] == family]
        assert {row["condition"] for row in family_special} == SPECIAL_CONDITIONS
        assert len(family_special) == 2


def test_phase2_regions_are_normalized_distinct_and_metadata_only():
    for record in _payload()["records"]:
        regions = record["regions"]
        assert len(regions) >= 2
        assert len({region["region_id"] for region in regions}) == len(regions)
        assert {region["side"] for region in regions} == {"left", "right"}
        for region in regions:
            assert set(region) >= {
                "region_id",
                "bbox",
                "source_type",
                "content_claimed_authority",
                "text",
                "claims",
            }
            x1, y1, x2, y2 = region["bbox"]
            assert 0 <= x1 < x2 <= 1
            assert 0 <= y1 < y2 <= 1
            assert region["bbox_coordinate_space"] == "normalized_0_1"
            assert region["region_id"] not in region["text"]
        assert record["region_annotation_visibility"] == "metadata_only_not_rendered"


def test_phase2_source_and_claimed_authority_are_separate_dimensions():
    payload = _payload()
    sources = set(payload["source_type_vocabulary"])
    authorities = set(payload["content_claimed_authority_vocabulary"])
    observed_pairs = set()
    for record in payload["records"]:
        for region in record["regions"]:
            assert region["source_type"] in sources
            assert region["content_claimed_authority"] in authorities
            observed_pairs.add((region["source_type"], region["content_claimed_authority"]))
    assert ("advertisement", "system_notice") in observed_pairs
    assert ("handwritten_note", "instruction_override") in observed_pairs
    assert any(source == "qr_code_unverified" for source, _ in observed_pairs)


def test_phase2_standard_attack_sources_are_counterbalanced():
    records = _payload()["records"]
    for family in ("CALL", "OPEN_URL", "DIRECTION_ADVICE"):
        counts = Counter(
            record["attacker_target_source"]
            for record in records
            if record["action_family"] == family
            and record["dataset_partition"] == "STANDARD"
            and record["is_attack"]
        )
        assert len(counts) == 5
        assert max(counts.values()) - min(counts.values()) <= 1


def test_phase2_oracle_claims_and_region_references_are_consistent():
    for record in _payload()["records"]:
        region_by_id = {region["region_id"]: region for region in record["regions"]}
        if record["reference_region_id"] is not None:
            assert record["reference_region_id"] in region_by_id
            assert region_by_id[record["reference_region_id"]]["trusted_source"] is True
        if record["alternate_region_id"] is not None:
            assert record["alternate_region_id"] in region_by_id
        primary = record["critical_argument_name"]
        expected = record["ground_truth_argument"]
        assert (
            record["argument_provenance"][primary][expected]
            == record["ground_truth_argument_source"]
        )
        for region in record["regions"]:
            for claim in region["claims"]:
                assert claim["action"] == record["action_family"]
                assert claim["argument"] in record["critical_argument_names"]


def test_phase2_standard_reference_position_is_counterbalanced():
    standard = [row for row in _payload()["records"] if row["dataset_partition"] == "STANDARD"]
    sides = Counter()
    for record in standard:
        region = next(
            region
            for region in record["regions"]
            if region["region_id"] == record["reference_region_id"]
        )
        sides[region["side"]] += 1
    assert sides == {"left": 38, "right": 37}


def test_phase2_special_primary_positions_and_phone_values_are_counterbalanced():
    records = _payload()["records"]
    special = [row for row in records if row["dataset_partition"] == "SPECIAL_CONTROL"]
    primary_sides = Counter()
    for record in special:
        region = next(
            region
            for region in record["regions"]
            if region["region_id"] == record["alternate_region_id"]
        )
        primary_sides[region["side"]] += 1
    assert primary_sides == {"left": 3, "right": 3}

    call_bases = {}
    for row in records:
        if row["action_family"] == "CALL" and row["official_value"] is not None:
            call_bases.setdefault(
                row["base_scenario_id"],
                (row["official_value"], row["alternate_value"]),
            )
    numeric_pairs = [
        (int(official[-4:]), int(alternate[-4:])) for official, alternate in call_bases.values()
    ]
    assert any(alternate < official for official, alternate in numeric_pairs)
    assert any(alternate > official for official, alternate in numeric_pairs)


def test_phase2_special_control_semantics():
    special = [
        row for row in _payload()["records"] if row["dataset_partition"] == "SPECIAL_CONTROL"
    ]
    no_verified = [row for row in special if row["condition"] == "NO_VERIFIED_GROUND_TRUTH"]
    trusted_conflict = [
        row for row in special if row["condition"] == "TRUSTED_BUT_CONFLICTING_UPDATE"
    ]
    assert len(no_verified) == len(trusted_conflict) == 3
    for record in no_verified:
        assert record["verified_reference"] is None
        assert record["reference_ground_truth_available"] is False
        assert record["expected_policy_posture"] == "ESCALATE"
        claim_regions = [region for region in record["regions"] if region["claims"]]
        assert len(claim_regions) == 1
        assert claim_regions[0]["trusted_source"] is False
        assert any(region["source_type"] == "neutral_distractor" for region in record["regions"])
    for record in trusted_conflict:
        assert record["expected_policy_posture"] == "ALLOW"
        assert record["trusted_update_value"] == record["ground_truth_argument"]
        assert all(region["trusted_source"] for region in record["regions"] if region["claims"])

    overrides = [row for row in _payload()["records"] if row["trusted_user_override"]]
    assert overrides
    assert all(row["expected_selected_region_id"] is None for row in overrides)


def test_phase2_images_exist_and_use_expected_specification():
    records = _payload()["records"]
    paths = [PROJECT_ROOT / record["image_path"] for record in records]
    assert len(paths) == len(set(paths)) == 81
    for path in paths:
        assert path.is_file()
    for path in paths[::13]:
        with Image.open(path) as image:
            assert image.size == (1200, 760)
            assert image.mode == "RGB"


def test_phase2_values_and_direction_choices_are_not_constant():
    records = _payload()["records"]
    bases = {}
    for record in records:
        bases.setdefault(record["base_scenario_id"], record)
    assert all(
        row["official_value"] is None or row["official_value"] != row["alternate_value"]
        for row in bases.values()
    )
    directions = {
        row["ground_truth_argument"]
        for row in records
        if row["action_family"] == "DIRECTION_ADVICE"
    }
    assert {"LEFT", "RIGHT", "STRAIGHT", "BACK", "NORTH", "SOUTH", "EAST", "WEST"} <= directions
