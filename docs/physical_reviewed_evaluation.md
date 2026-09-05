# Reviewed physical inference and scoring

`physical_reviewed_inference.py` records paired Guard ON/OFF requests to the
existing resident Qwen runtime. Both requests use identical oriented image bytes
and the user task; evaluation labels are never sent to the runtime. Plans, source
snapshots, hashes, start markers, and raw responses allow interrupted work to be
audited without silently repeating a completed trial.

`physical_reviewed_scoring.py` scores those responses against frozen human
annotations and compares the preserved historical Direct outputs. Unknown truth,
unreviewed annotations, invalid output, blocked actions, and free text requiring
human interpretation remain distinct. A blocked response does not count as a
correct answer. This is a study-specific analysis tool, not a general accuracy
benchmark.

## Local inputs and outputs

The tools require the locally supplied `TestData.zip`, the canonical
`results_physical_pilot/direct_v1/input_manifest.json`, and frozen annotations in
`results_physical_pilot/reviewed_prototype_v1/evaluation_only/`. Scoring also requires
the historical Direct responses. These inputs are not supplied by cloning this
repository. Use only data you have permission to process.

The complete `results_physical_pilot/reviewed_prototype_v1/` directory is ignored
by Git, including source snapshots and derived reports: recognized scene text can
retain personal or contact information even when the photographs are omitted.
The existing local run is preserved. Public submission includes the tools and
synthetic tests; it does not redistribute this run or grant rights to its inputs.

## CPU validation

From the Prototype root, with the existing development environment:

```bash
.venv/bin/python -m pytest tests/test_physical_reviewed_inference.py tests/test_physical_reviewed_scoring.py
```

The tests use synthetic data and mocked transport. They neither load a model nor
submit live requests. For an independently prepared dataset, inspect the CLI with
`.venv/bin/python physical_reviewed_inference.py --help` and
`.venv/bin/python physical_reviewed_scoring.py --help`. Live collection requires an
idle resident service and its verified PID; `--prepare-only` prepares a plan,
`--limit` bounds new trials, and `--resume` preserves existing trial identities.
Do not point a fresh collection at an existing completed run.
