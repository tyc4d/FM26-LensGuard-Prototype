"""Verify historical bytes and any locally preserved cloud scientific artifacts."""

import subprocess

from cloud_baseline_contracts import FROZEN_HEAD, ROOT
from cloud_baseline_store import CloudResultStore, read_json


def test_all_preexisting_scientific_files_unchanged():
    original = set(subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", FROZEN_HEAD], cwd=ROOT, text=True,
    ).splitlines())
    changed = set(subprocess.check_output(
        ["git", "diff", "--name-only", FROZEN_HEAD, "--"], cwd=ROOT, text=True,
    ).splitlines())
    # Only shared dependency declarations are extended by this additive experiment.
    assert changed & original <= {"pyproject.toml", "uv.lock"}


def test_preserved_cloud_result_integrity_when_present():
    for path in (ROOT / "results_cloud_baseline").glob("**/plans/*.json"):
        plan = read_json(path)
        root = path.parent.parent
        store = CloudResultStore(root, plan["provider"], plan, resume=True)
        manifest = store.manifest()
        result = store.validate(require_complete=not manifest["incomplete"])
        assert result["recorded_trials"] == manifest["recorded_trials"]
        assert result["planned_trials"] == (6 if plan["mode"] == "smoke" else 162)
