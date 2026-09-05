"""Atomic local drafts with explicit verification and optimistic concurrency."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import tempfile
import threading
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .schema import SCHEMA_VERSION, SCENARIOS, STATUSES, validate_annotation


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def atomic_write(path: Path, payload: bytes, backup: bool = True) -> None:
    """fsync a sibling temp file, preserve the old bytes, then atomically replace.

    Any interrupted write leaves either the complete previous state or complete
    next state. The JSONL is authoritative; progress is a reconstructible cache.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if backup and path.exists():
            old = path.read_bytes()
            digest = hashlib.sha256(old).hexdigest()[:16]
            target = path.parent / "drafts" / f"{path.name}.{digest}.bak"
            if not target.exists():
                atomic_write(target, old, backup=False)
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def progress_for(annotations: list[dict]) -> dict:
    statuses = Counter(row["status"] for row in annotations)
    scenarios = {scenario: {"total": sum(row["scenario"] == scenario for row in annotations),
                            "verified": sum(row["scenario"] == scenario and row["human_verified"]
                                            for row in annotations)} for scenario in SCENARIOS}
    return {"total": len(annotations), "verified": statuses["VERIFIED"],
            "draft": statuses["DRAFT"], "needs_review": statuses["NEEDS_REVIEW"],
            "unreviewed": statuses["UNREVIEWED"],
            "excluded": sum(row["exclude_from_primary_aggregate"] for row in annotations),
            "contamination_risk": sum(row["inference_contamination_risk"] for row in annotations),
            "scenarios": scenarios}


class RevisionConflict(ValueError):
    """Another browser or process saved this dataset after the submitted revision."""


class AnnotationStore:
    def __init__(self, root: str | Path, dataset: dict, directory: str | Path | None = None):
        self.root = Path(root)
        self.dataset = copy.deepcopy(dataset)
        self.directory = Path(directory) if directory is not None else self.root / "data/physical_pilot_v1/annotations"
        self.draft_path = self.directory / "ground_truth_draft.jsonl"
        self.progress_path = self.directory / "annotation_progress.json"
        self._thread_lock = threading.RLock()
        self._defaults = {row["image_id"]: row for row in self.dataset["annotations"]}

    @contextmanager
    def locked(self, create: bool = False):
        """Serialize writes across threads/processes; fresh startup creates no files."""
        with self._thread_lock:
            if create:
                self.directory.mkdir(parents=True, exist_ok=True)
            if not self.directory.exists():
                yield
                return
            with (self.directory / ".annotation.lock").open("a+b") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def _read_state(self) -> dict:
        if not self.draft_path.exists():
            annotations = copy.deepcopy(self.dataset["annotations"])
            revision = 0
        else:
            annotations = [json.loads(line) for line in self.draft_path.read_text().splitlines() if line]
            if len(annotations) != len(self._defaults) or {r["image_id"] for r in annotations} != set(self._defaults):
                raise ValueError("Draft identities do not match the canonical image inventory")
            revisions = {row.pop("_revision", None) for row in annotations}
            if len(revisions) != 1 or type(next(iter(revisions))) is not int:
                raise ValueError("Draft transaction revision is missing or inconsistent")
            revision = next(iter(revisions))
            for row in annotations:
                self._check_identity(row)
                validate_annotation(row, verifying=row["human_verified"])
            by_id = {row["image_id"]: row for row in annotations}
            annotations = [by_id[image_id] for image_id in self._defaults]
        return {"schema_version": SCHEMA_VERSION, "revision": revision,
                "annotations": annotations, "progress": progress_for(annotations)}

    def state(self) -> dict:
        with self.locked():
            return self._read_state()

    def _check_identity(self, row: dict) -> None:
        original = self._defaults.get(row.get("image_id"))
        if original is None:
            raise ValueError("Unknown canonical image ID")
        for key in ("image_id", "original_filename", "image_sha256", "schema_version", "prefill",
                    "bbox_coordinate_space", "time_source", "party_size_source"):
            if row.get(key) != original[key]:
                raise ValueError(f"Cannot change immutable source/provenance field: {key}")

    def save(self, image_id: str, annotation: dict, expected_revision: int,
             verify: bool = False, reviewer: str | None = None,
             confirm_attacker_match: bool = False) -> dict:
        if type(verify) is not bool or type(confirm_attacker_match) is not bool:
            raise ValueError("verify and confirm_attacker_match must be boolean")
        if not isinstance(annotation, dict):
            raise ValueError("Annotation must be an object")
        with self.locked(create=True):
            state = self._read_state()
            if type(expected_revision) is not int or expected_revision != state["revision"]:
                raise RevisionConflict("Annotation revision changed. Reload before saving to avoid losing another review.")
            if image_id not in self._defaults or annotation.get("image_id") != image_id:
                raise ValueError("Unknown or mismatched canonical image ID")
            row = copy.deepcopy(annotation)
            self._check_identity(row)
            row.pop("_revision", None)
            # Verification is a distinct explicit operation. Client-provided
            # authority flags never promote a draft, including evidence regions.
            row["status"] = "NEEDS_REVIEW" if row.get("status") == "NEEDS_REVIEW" else "DRAFT"
            row["human_verified"] = False
            row["reviewer"] = None
            row["reviewed_at"] = None
            row["updated_at"] = utc_now()
            if not isinstance(row.get("regions"), list) or any(not isinstance(r, dict) for r in row["regions"]):
                raise ValueError("regions must be a list of objects")
            for region in row["regions"]:
                region["human_verified"] = False
            warnings = validate_annotation(row, verifying=verify)
            if verify:
                if not isinstance(reviewer, str) or not reviewer.strip():
                    raise ValueError("Enter a reviewer name before verification")
                if warnings and not confirm_attacker_match:
                    raise ValueError(" ".join(warnings))
                row.update(status="VERIFIED", human_verified=True, reviewer=reviewer.strip(),
                           reviewed_at=row["updated_at"])
                for region in row["regions"]:
                    region["human_verified"] = True
                validate_annotation(row, verifying=True)
            annotations = [row if old["image_id"] == image_id else old for old in state["annotations"]]
            revision = state["revision"] + 1
            payload = b"".join(canonical_json({**item, "_revision": revision}) for item in annotations)
            atomic_write(self.draft_path, payload)
            result = {"schema_version": SCHEMA_VERSION, "revision": revision,
                      "annotations": annotations, "progress": progress_for(annotations)}
            # A failed derived cache write must not misreport a committed draft
            # as lost. state() reconstructs this summary from the complete JSONL.
            try:
                atomic_write(self.progress_path, canonical_json({
                    "schema_version": SCHEMA_VERSION, "revision": revision,
                    "updated_at": row["updated_at"], **result["progress"]}))
            except OSError:
                result["storage_warning"] = "Draft saved; progress cache will be reconstructed on reload."
            return result
