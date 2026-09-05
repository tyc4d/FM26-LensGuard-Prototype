"""Explicit human freeze, deterministic exports, and immutable local versions.

These files are benchmark annotations. This module never constructs action-model
inputs, starts inference, or commits scientific data to version control.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path

from .schema import SCHEMA_VERSION, validate_annotation
from .storage import AnnotationStore, RevisionConflict, canonical_json, utc_now

_VERSION_FILE = re.compile(r"^\.?ground_truth_v([1-9][0-9]*)(?:\.|_)")
_TRANSIENT_FIELDS = {"_revision", "updated_at", "reviewed_at"}


def _rows(state: dict) -> list[dict]:
    return sorted(({key: value for key, value in row.items() if key != "_revision"}
                   for row in state["annotations"]), key=lambda row: row["image_id"])


def export_jsonl(state: dict) -> bytes:
    """Canonical UTF-8 bytes; repeated exports of the same state have one hash."""
    return b"".join(canonical_json(row) for row in _rows(state))


def _csv_value(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    # CSV is for review. Prefix possible formulas, including whitespace-obscured
    # ones, instead of permitting spreadsheet execution. JSONL retains exact text.
    if value.startswith(("\t", "\r", "\n")) or value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def export_csv(state: dict) -> bytes:
    """Export every annotation field; nested regions/provenance use JSON cells."""
    rows = _rows(state)
    keys = set().union(*(row.keys() for row in rows)) if rows else set()
    leading = [key for key in ("image_id", "original_filename", "scenario", "status") if key in keys]
    columns = leading + sorted(keys.difference(leading))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: _csv_value(row.get(key)) for key in columns} for row in rows)
    return output.getvalue().encode("utf-8")


def _validate_state(state: dict) -> dict:
    rows = state["annotations"]
    errors = []
    blocked = set()
    unresolved = set()
    needs_review = []
    for row in rows:
        image_id = row["image_id"]
        excluded = row.get("exclude_from_primary_aggregate") is True
        status = row.get("status")
        try:
            validate_annotation(row, verifying=status == "VERIFIED")
            if excluded and (status == "UNREVIEWED" or not isinstance(row.get("exclusion_reason"), str)
                             or not row["exclusion_reason"].strip()):
                raise ValueError("Exclusion requires an explicit human edit and an exclusion reason")
        except ValueError as exc:
            errors.append({"image_id": image_id, "error": str(exc)})
            blocked.add(image_id)
            unresolved.add(image_id)
        if status == "NEEDS_REVIEW" and not excluded:
            needs_review.append(image_id)
            unresolved.add(image_id)
        elif status != "VERIFIED" and not excluded:
            blocked.add(image_id)
            unresolved.add(image_id)
    return {
        "revision": state["revision"], "schema_version": SCHEMA_VERSION,
        "total": len(rows), "verified": sum(row["status"] == "VERIFIED" for row in rows),
        "excluded": sum(row["exclude_from_primary_aggregate"] for row in rows),
        "needs_review": sum(row["status"] == "NEEDS_REVIEW" for row in rows),
        "draft": sum(row["status"] == "DRAFT" for row in rows),
        "unreviewed": sum(row["status"] == "UNREVIEWED" for row in rows),
        "unresolved": len(unresolved), "blocked_image_ids": sorted(blocked),
        "needs_review_image_ids": sorted(needs_review), "errors": errors,
        "can_freeze": not blocked,
        "requires_unresolved_acknowledgement": bool(needs_review),
    }


def validate_dataset(store: AnnotationStore) -> dict:
    """Describe readiness without changing a draft or verifying any image."""
    with store.locked():
        return _validate_state(store._read_state())


def _version_paths(directory: Path, number: int) -> dict[str, Path]:
    stem = f"ground_truth_v{number}"
    return {"jsonl": directory / f"{stem}.jsonl",
            "sha256": directory / f"{stem}.sha256",
            "manifest": directory / f"{stem}_manifest.json"}


def _history(directory: Path) -> list[tuple[int, dict, bytes]]:
    """Manifest is the commit marker; partial or tampered versions need an audit."""
    numbers = sorted({int(match.group(1)) for path in directory.iterdir()
                      if (match := _VERSION_FILE.match(path.name))})
    history = []
    for number in numbers:
        paths = _version_paths(directory, number)
        if not all(path.is_file() for path in paths.values()):
            raise ValueError(
                f"Incomplete freeze v{number} reserves this version. Preserved artifacts in {directory} "
                "must be audited before another freeze; the tool will never overwrite them."
            )
        payload = paths["jsonl"].read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        try:
            manifest = json.loads(paths["manifest"].read_bytes())
            checksum = paths["sha256"].read_text(encoding="utf-8").strip()
            valid = (manifest["sha256"] == digest and manifest["version"] == f"v{number}"
                     and checksum == f"{digest}  {paths['jsonl'].name}")
        except (ValueError, KeyError, TypeError):
            valid = False
        if not valid:
            raise ValueError(f"Frozen v{number} integrity check failed. Preserve files and audit before proceeding.")
        history.append((number, manifest, payload))
    return history


def _meaningful_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in _TRANSIENT_FIELDS}


def _changed_image_ids(previous: bytes, state: dict) -> list[str]:
    before = {row["image_id"]: _meaningful_row(row)
              for line in previous.splitlines() if line for row in [json.loads(line)]}
    after = {row["image_id"]: _meaningful_row(row) for row in state["annotations"]}
    return sorted(image_id for image_id in before.keys() | after.keys()
                  if before.get(image_id) != after.get(image_id))


def _sync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _exclusive_write(path: Path, payload: bytes) -> None:
    """Reserve a version even if the process exits before publishing its files."""
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _sync_directory(path.parent)


def _publish(directory: Path, number: int, payloads: dict[str, bytes]) -> dict[str, Path]:
    """Stage complete files, then link exclusively; manifest is published last.

    Hard links on the same local filesystem are atomic and fail if a target
    exists. An interruption leaves a reserved, auditable partial version rather
    than overwriting scientific data or pretending the freeze completed.
    """
    paths = _version_paths(directory, number)
    with tempfile.TemporaryDirectory(prefix=f".ground_truth_v{number}.stage-", dir=directory) as staging:
        for key, payload in payloads.items():
            path = Path(staging) / paths[key].name
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            path.chmod(0o444)
        _sync_directory(Path(staging))
        for key in ("jsonl", "sha256", "manifest"):
            os.link(Path(staging) / paths[key].name, paths[key])
            _sync_directory(directory)
    return paths


def freeze(store: AnnotationStore, expected_revision: int, confirm: bool,
           acknowledge_unresolved: bool = False, change_reason: str | None = None) -> dict:
    """Publish one immutable version only after explicit human confirmation.

    NEEDS_REVIEW records remain unverified in frozen output and require a second
    explicit acknowledgement. Included UNREVIEWED/DRAFT records always block.
    """
    if confirm is not True:
        raise ValueError("Freeze requires explicit human confirmation")
    with store.locked(create=True):
        state = store._read_state()
        if type(expected_revision) is not int or expected_revision != state["revision"]:
            raise RevisionConflict("Annotations changed after the freeze preview. Validate and confirm again.")
        validation = _validate_state(state)
        if not validation["can_freeze"]:
            raise ValueError("Freeze blocked: verify, explicitly mark NEEDS_REVIEW, or document exclusion for: "
                             + ", ".join(validation["blocked_image_ids"]))
        if validation["requires_unresolved_acknowledgement"] and acknowledge_unresolved is not True:
            raise ValueError("Unresolved NEEDS_REVIEW images remain. Explicitly acknowledge before freezing; "
                             "these records will not become verified ground truth.")
        history = _history(store.directory)
        number = history[-1][0] + 1 if history else 1
        parent_version = history[-1][1]["version"] if history else None
        changed_ids = []
        if history:
            if not isinstance(change_reason, str) or not change_reason.strip():
                raise ValueError("A correction version requires change_reason")
            if history[-1][1].get("dataset_zip_sha256") != store.dataset["archive_sha256"]:
                raise ValueError("Dataset ZIP identity differs from the frozen parent version")
            changed_ids = _changed_image_ids(history[-1][2], state)
            if not changed_ids:
                raise ValueError("No annotation changes since the frozen parent version")
        payload = export_jsonl(state)
        digest = hashlib.sha256(payload).hexdigest()
        paths = _version_paths(store.directory, number)
        manifest = {
            "version": f"v{number}", "parent_version": parent_version,
            "change_reason": change_reason.strip() if isinstance(change_reason, str) else None,
            "changed_image_ids": changed_ids, "timestamp_utc": utc_now(),
            "annotation_schema_version": SCHEMA_VERSION,
            "dataset_zip_sha256": store.dataset["archive_sha256"],
            "input_manifest_sha256": store.dataset.get("manifest_sha256"),
            "annotation_count": validation["total"], "verified_count": validation["verified"],
            "excluded_count": validation["excluded"], "needs_review_count": validation["needs_review"],
            "unresolved_count": validation["unresolved"],
            "unresolved_acknowledged": validation["requires_unresolved_acknowledgement"],
            "needs_review_image_ids": validation["needs_review_image_ids"],
            "source_revision": state["revision"], "sha256": digest,
            "annotation_file": paths["jsonl"].name,
            "control_class_usage": "BENCHMARK GROUND TRUTH ONLY; never provide to action model or Thin Gate",
        }
        _exclusive_write(store.directory / f".ground_truth_v{number}.reserve",
                         canonical_json({"version": manifest["version"],
                                         "reserved_at": manifest["timestamp_utc"],
                                         "source_revision": state["revision"]}))
        paths = _publish(store.directory, number, {
            "jsonl": payload,
            "sha256": f"{digest}  {paths['jsonl'].name}\n".encode("ascii"),
            "manifest": canonical_json(manifest),
        })
        return {**manifest, "files": {key: str(path.resolve()) for key, path in paths.items()}}
