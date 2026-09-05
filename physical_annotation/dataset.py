"""Read the existing immutable inventory; prefill only traceable provisional metadata."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath

from .schema import CRITICAL_ARGUMENTS, SCHEMA_VERSION, validate_annotation

MANIFEST_PATH = Path("results_physical_pilot/direct_v1/input_manifest.json")
SCENARIO_COUNTS = {"CALL": 6, "RESTAURANT_RESERVATION": 30, "NAVIGATION": 11, "SAFETY": 7}
PREFILL_NOTICE = "PRE-FILLED — NOT HUMAN VERIFIED"


def _default_annotation(record: dict, manifest_hash: str) -> dict:
    scenario = record["scenario_family"]
    evidence = record.get("provisional_visible_evidence", {})
    fields = {"scenario": "scenario_family", "attack_mode": "provisional_attack_mode"}
    attacker = None
    candidates = (evidence.get("injected_phone_candidates", []) if scenario in
                  ("CALL", "RESTAURANT_RESERVATION") else evidence.get("injected_direction_candidates", []))
    if record["provisional_attack_mode"] != "NONE" and len(candidates) == 1:
        attacker = candidates[0]
        fields["attacker_value"] = "provisional_visible_evidence (single candidate)"
    environment = None
    notes = record.get("review_notes", "")
    # Shared restaurant candidate prose explicitly includes multiple scenes. Only
    # frame-specific notes can supply a provisional environment phone; clipped
    # digits remain empty rather than being completed from neighboring frames.
    if scenario == "CALL":
        matches = re.findall(r"\b0[0-9]+-[0-9]+-[0-9]+\b", evidence.get("candidate_environment_evidence", ""))
        if len(matches) == 1:
            environment = matches[0]
            fields["environment_value"] = "provisional_visible_evidence.candidate_environment_evidence"
    elif (scenario == "RESTAURANT_RESERVATION" and record.get("scene_group") == "REST_GOODCHOS_DOOR"
          and not re.search(r"clipped|missing digits", notes, re.I)):
        match = re.search(r"environmental (?:Good Cho's door contact|phone) (0[0-9-]+)", notes, re.I)
        if match:
            environment = match[1]
            fields["environment_value"] = "review_notes (frame-specific environmental contact)"
    annotation = {
        "schema_version": SCHEMA_VERSION,
        "image_id": record["image_id"], "original_filename": record["original_filename"],
        "scenario": scenario, "attack_mode": record["provisional_attack_mode"],
        "critical_argument": CRITICAL_ARGUMENTS[scenario], "attacker_value": attacker,
        "environment_value": environment, "expected_behavior": "UNKNOWN",
        "ground_truth_value": None, "ground_truth_known": False, "notes": "",
        "status": "UNREVIEWED", "human_verified": False, "reviewer": None,
        "reviewed_at": None, "updated_at": None, "restaurant_identity": "",
        "user_time": "19:00", "user_party_size": 2, "time_source": "USER", "party_size_source": "USER",
        "physical_hazard": "UNKNOWN", "injected_text": "", "attacker_safety_claim": "UNKNOWN",
        "inference_contamination_risk": bool(record.get("inference_contamination_risk", False)),
        "exclude_from_primary_aggregate": False, "exclusion_reason": "", "regions": [],
        "prefill": {"notice": PREFILL_NOTICE, "source": str(MANIFEST_PATH),
                    "manifest_sha256": manifest_hash, "fields": fields,
                    "source_review_notes": notes,
                    "source_visible_evidence": copy.deepcopy(evidence),
                    "human_verified": False},
        "image_sha256": record["sha256"],
        "bbox_coordinate_space": "EXIF_ORIENTED_NORMALIZED",
    }
    if scenario == "SAFETY" and record["provisional_attack_mode"] != "NONE":
        annotation["injected_text"] = evidence.get("injected_safety_claim") or ""
        if annotation["injected_text"]:
            fields["injected_text"] = "provisional_visible_evidence.injected_safety_claim"
    validate_annotation(annotation)
    return annotation


def load_dataset(root: str | Path) -> dict:
    """Hash-check the committed manifest and review CSV without creating files."""
    root = Path(root)
    path = root / MANIFEST_PATH
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    expected = path.with_suffix(".sha256").read_text().split()[0]
    if digest != expected:
        raise ValueError("Frozen input manifest hash mismatch")
    manifest = json.loads(payload)
    review = path.parent / "input_review_metadata.csv"
    if hashlib.sha256(review.read_bytes()).hexdigest() != manifest["review_csv_sha256"]:
        raise ValueError("Frozen review CSV hash mismatch")
    records = manifest["records"]
    if len(records) != 54 or manifest.get("image_count") != 54:
        raise ValueError("Expected the canonical 54-image physical dataset")
    if len({r["image_id"] for r in records}) != 54 or len({r["sha256"] for r in records}) != 54:
        raise ValueError("Duplicate canonical image identity/hash")
    if Counter(r["scenario_family"] for r in records) != SCENARIO_COUNTS:
        raise ValueError("Canonical scenario inventory changed")
    for row in records:
        name = PurePosixPath(row["archive_member"])
        if (row["image_id"] != row["original_filename"] or name.name != row["image_id"]
                or name.is_absolute() or ".." in name.parts or "\\" in row["archive_member"]):
            raise ValueError("Unsafe or incompatible canonical image identity")
        if row["human_verified"] or manifest["ground_truth_frozen"]:
            raise ValueError("Provisional metadata must not become verified ground truth")
    return {"manifest": manifest, "records": records, "manifest_sha256": digest,
            "archive_sha256": manifest["archive_sha256"],
            "annotations": [_default_annotation(row, digest) for row in records]}
