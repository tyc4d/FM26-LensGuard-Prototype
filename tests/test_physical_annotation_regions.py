"""Evidence geometry and authority round-trip through the real local HTTP layer."""

from pathlib import Path

from fastapi.testclient import TestClient

from physical_annotation.schema import normalize_bbox
from physical_annotation.server import create_app

ROOT = Path(__file__).resolve().parents[1]


class FakeImages:
    def read(self, image_id):
        return b"local-test-image"


def test_region_round_trip_scaled_geometry_and_explicit_authority(tmp_path):
    app = create_app(ROOT, annotation_directory=tmp_path / "labels", images=FakeImages())
    with TestClient(app) as client:
        state = client.get("/api/state").json()
        row = state["annotations"][0]
        row["regions"] = [{"region_id": "R01", "bbox_normalized": normalize_bbox((60, 40), (240, 160), 300, 200),
            "region_type": "TEXT", "ground_truth_text": "Human transcription", "semantic_role": "contact",
            "physical_source": "added paper", "control_class": "attacker_controlled", "linked_object": None,
            "supports_ground_truth": None, "human_verified": False}]
        headers = {"X-Annotation-Token": state["token"]}
        path = "/api/annotations/" + row["image_id"]
        response = client.post(path, json={"annotation": row, "expected_revision": 0}, headers=headers)
        assert response.status_code == 200
        saved = response.json()["annotations"][0]
        assert saved["regions"][0]["bbox_normalized"] == [.2, .2, .8, .8]
        assert not saved["regions"][0]["human_verified"]
        response = client.post(path, json={"annotation": saved, "expected_revision": 1,
            "verify": True, "reviewer": "Human reviewer"}, headers=headers)
        assert response.status_code == 200
        verified = response.json()["annotations"][0]
        assert verified["regions"][0]["human_verified"]
        assert verified["bbox_coordinate_space"] == "EXIF_ORIENTED_NORMALIZED"
        verified["regions"][0]["bbox_normalized"][2] = 1.2
        assert client.post(path, json={"annotation": verified, "expected_revision": 2}, headers=headers).status_code == 422
        assert client.get("/api/state").json()["annotations"][0]["human_verified"]


def test_evidence_assets_are_local_and_direct_scoring_needs_no_rectangles(tmp_path):
    with TestClient(create_app(ROOT, annotation_directory=tmp_path / "labels", images=FakeImages())) as client:
        response = client.get("/static/regions.js")
        assert response.status_code == 200
        state = client.get("/api/state").json()
        row = state["annotations"][0]
        response = client.post("/api/annotations/" + row["image_id"], headers={"X-Annotation-Token": state["token"]},
            json={"annotation": row, "expected_revision": 0, "verify": True, "reviewer": "Human reviewer"})
        assert response.status_code == 200
        assert response.json()["annotations"][0]["regions"] == []
