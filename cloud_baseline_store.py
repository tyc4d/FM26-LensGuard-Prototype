"""Exclusive scientific writes and resumable operational manifests.

Raw envelopes, trial records, request plans and start markers are immutable.
JSONL, manifests, hashes and summaries are derived indexes and may be rebuilt.
An interrupted send with no preserved envelope is never automatically resent.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from providers.base_cloud_vlm import redact_secrets


def encoded(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(encoded(redact_secrets(value)))
        handle.flush()
        os.fsync(handle.fileno())


def write_index(path: Path, value: Any, *, text: bool = False) -> None:
    """Replace only derived operational files, never scientific source files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = redact_secrets(value)
    temporary.write_bytes(payload.encode() if text else encoded(payload))
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def trial_id(case_id: str, arm: str) -> str:
    return f"{case_id}/{arm}"


class CloudResultStore:
    def __init__(self, root: Path, provider: str, plan: dict, *, resume: bool = False):
        self.root, self.provider, self.plan = Path(root), provider, plan
        self.plan_path = self.root / "plans" / f"{provider}.json"
        self.record_root = self.root / "records" / provider
        self.raw_root = self.root / "raw" / provider
        self.started_root = self.root / "started" / provider
        self.expected = {
            trial_id(case["case_id"], arm)
            for case in plan["cases"] for arm in plan["arms"]
        }
        if self.plan_path.exists():
            if not resume:
                raise FileExistsError("Provider run already exists; use --resume for missing trials")
            if read_json(self.plan_path) != redact_secrets(plan):
                raise ValueError("Resume plan differs in model, config, cases, images, prompts or schema")
        else:
            # Refuse orphaned data rather than claiming a fresh scientific run.
            if any(p.exists() for p in (self.record_root, self.raw_root, self.started_root)):
                raise FileExistsError("Orphaned scientific output exists without a plan")
            write_new(self.plan_path, plan)
        self.validate()

    def path(self, kind: str, case_id: str, arm: str) -> Path:
        if trial_id(case_id, arm) not in self.expected:
            raise ValueError("Unknown trial identity")
        return self.root / kind / self.provider / arm.lower() / f"{case_id}.json"

    def records(self) -> list[dict]:
        rows = [read_json(p) for p in sorted(self.record_root.glob("*/*.json"))]
        return sorted(rows, key=lambda row: trial_id(row["case_id"], row["arm"]))

    def validate(self, *, require_complete: bool = False) -> dict:
        hashes_path = self.root / f"{self.provider}_hashes.json"
        if hashes_path.exists():
            for relative, expected_hash in read_json(hashes_path).items():
                path = self.root / relative
                if not path.resolve().is_relative_to(self.root.resolve()):
                    raise ValueError("Invalid hash catalog path")
                if not path.is_file() or file_digest(path) != expected_hash:
                    raise ValueError("Preserved scientific artifact changed")
        rows = self.records()
        identities = [trial_id(row["case_id"], row["arm"]) for row in rows]
        if len(set(identities)) != len(identities) or set(identities) - self.expected:
            raise ValueError("Duplicate or unknown scientific trial identities")
        for row in rows:
            raw = self.path("raw", row["case_id"], row["arm"])
            started = self.path("started", row["case_id"], row["arm"])
            if not raw.is_file() or file_digest(raw) != row["raw_response_sha256"]:
                raise ValueError("Missing or modified raw response")
            if not started.is_file() or read_json(started)["request_sha256"] != row["request_sha256"]:
                raise ValueError("Missing or mismatched pre-inference request marker")
            envelope = read_json(raw)
            if envelope["request_sha256"] != row["request_sha256"]:
                raise ValueError("Raw response bound to a different request")
            if row["scientific_attempt"] != 1 or envelope["response"]["scientific_attempt"] != 1:
                raise ValueError("Scientific attempt policy violated")
            if row["provider"] != self.provider or row["model"] != self.plan["model"]:
                raise ValueError("Provider/model changed within scientific run")
        for kind in ("raw", "started", "records"):
            for path in (self.root / kind / self.provider).glob("*/*.json"):
                identity = trial_id(path.stem, path.parent.name.upper())
                if identity not in self.expected:
                    raise ValueError("Unknown scientific artifact identity")
        if require_complete and set(identities) != self.expected:
            raise ValueError("Scientific run has missing trials")
        return {"valid": True, "planned_trials": len(self.expected),
                "recorded_trials": len(rows), "missing_trials": len(self.expected - set(identities))}

    def begin(self, case_id: str, arm: str, request_sha256: str, timestamp: str) -> None:
        if any(self.path(kind, case_id, arm).exists() for kind in ("records", "raw", "started")):
            raise FileExistsError("Scientific trial already started; never automatically resend")
        write_new(self.path("started", case_id, arm), {
            "case_id": case_id, "arm": arm, "request_sha256": request_sha256,
            "timestamp_utc": timestamp, "scientific_attempt": 1,
        })

    def save_raw(self, case_id: str, arm: str, response: dict, request_sha256: str) -> Path:
        path = self.path("raw", case_id, arm)
        write_new(path, {"case_id": case_id, "arm": arm,
                         "request_sha256": request_sha256, "response": response})
        return path

    def save_record(self, record: dict) -> None:
        write_new(self.path("records", record["case_id"], record["arm"]), record)

    def manifest(self) -> dict:
        rows = self.records()
        completed = [trial_id(r["case_id"], r["arm"]) for r in rows if r["completed"]]
        failed = [trial_id(r["case_id"], r["arm"]) for r in rows if not r["completed"]]
        pending = sorted(self.expected - set(completed) - set(failed))
        unresolved = [identity for identity in pending
                      if self.path("started", *identity.split("/")).exists()]
        return {
            "experiment_id": self.plan["experiment_id"], "provider": self.provider,
            "model": self.plan["model"], "planned_trials": len(self.expected),
            "recorded_trials": len(rows), "completed_trials": len(completed),
            "completed_case_ids": sorted({r["case_id"] for r in rows if r["completed"]}),
            "failed_case_ids": sorted({r["case_id"] for r in rows if not r["completed"]}),
            "pending_case_ids": sorted({identity.split("/")[0] for identity in pending}),
            "completed_trial_ids": completed, "failed_trial_ids": failed,
            "pending_trial_ids": pending, "interrupted_trial_ids": unresolved,
            "incomplete": len(completed) != len(self.expected),
            "incomplete_due_to_quota": any(r.get("hard_quota") or r.get("error_type") == "RATE_LIMIT_EXHAUSTED"
                                           for r in rows) and len(completed) != len(self.expected),
            "transport_retry_count": sum(max(r["transport_attempts"] - 1, 0) for r in rows),
            "rate_limit_events": sum(r.get("rate_limit_events", 0) for r in rows),
            "total_backoff_seconds": sum(r.get("total_backoff_seconds", 0) for r in rows),
            "stop_reasons": sorted({r["error_type"] for r in rows if r.get("stop_provider")}),
            "resume_policy": "Only missing trials; completed and failed records are immutable. "
                             "An interrupted send with no raw envelope requires manual audit.",
        }

    def refresh_indexes(self) -> dict:
        self.validate()
        rows = self.records()
        manifest = self.manifest()
        write_index(self.root / f"{self.provider}_manifest.json", manifest)
        lines = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
        write_index(self.root / f"{self.provider}_normalized.jsonl", lines, text=True)
        files = [self.plan_path, *self.record_root.glob("*/*.json"),
                 *self.raw_root.glob("*/*.json"), *self.started_root.glob("*/*.json")]
        write_index(self.root / f"{self.provider}_hashes.json", {
            path.relative_to(self.root).as_posix(): file_digest(path) for path in sorted(files)
        })
        return manifest
