# Phase 3 physical pilot: DIRECT v1

Experiment: `lensguard-phase3-physical-direct-v1`. This records raw model behavior on the 54 reviewed physical photographs. Human-reviewed ground truth is not frozen. Completion, JSON structure, emitted arguments, latency and usage are descriptive observations; they are not accuracy, attack success, defense success or unsafe execution measurements.

The frozen input manifest and copied review metadata live in `results_physical_pilot/direct_v1/`. `IMG_3485.jpeg` remains included and carries `inference_contamination_risk=true` because laptop screens contain experiment/model-related text. Reports must show that frame separately and identify denominators when excluding it.

The archive hash is `b6aadea97982cdb7d8383948a912cc334e349c2b880ff05a0d09ea06d186d25a`. Originals remain in the archive and a temporary byte-identical cache. They are neither renamed nor recompressed nor committed. Reproduction requires the same `TestData.zip` at the repository root; the immutable manifest verifies every original filename, hash and dimension.

`config/physical_direct_prompts_v1.yaml` fixes one task prompt per scenario and one common output schema before inference. The model receives only image bytes/pixels, the scenario task and the shared JSON instructions. No review notes, control labels, injected candidates, region annotations, provenance or expected answers enter its input. No evidence registry, grounding validator, conflict evaluator, deterministic gate or external verification runs in this experiment. Tool lists are empty for both cloud providers.

Models are the frozen Gemma 3 4B, MiniCPM-V 4.5 and Qwen3-VL 8B revisions specified in `physical_direct_local.py`, plus `gpt-5.6-sol` and `gemini-3.1-flash-lite`. Local runs use their previously frozen Python environments, BF16, CUDA, batch one, greedy generation and 1,024 maximum new tokens. Cloud native configuration is preserved in every request plan and response. OpenAI uses Responses; Gemini uses Google GenAI Interactions. Model fallback is forbidden.

Source image bytes are never changed. Local decoding applies EXIF orientation in memory before the frozen family processor; cloud APIs receive original JPEG-family bytes and use provider-native decoding. Cloud EXIF handling and internal resizing are not independently observable. Local output is constrained by the shared prompt; cloud APIs additionally receive native JSON schema syntax. These implementation differences are recorded and limit claims of identical decoding conditions.

The initial OpenAI smoke request exceeded its native patch limit and produced no model output. The physical-only adapter consequently sets the supported `detail: high` for every OpenAI image, retaining the original image bytes. Its four active smoke requests use a separate compatibility namespace; the other models' smoke responses are not repeated. See [the preserved compatibility audit](../results_physical_pilot/direct_v1/api_compatibility_note.md). The one rejected diagnostic request is additional to the planned 20 active smoke and 270 full trials.

Each image/model/run-type identity gets one semantic attempt. The four predeclared smoke images are `IMG_3483.jpeg`, `IMG_6164.JPG`, `IMG_6152.JPG` and `IMG_6157.JPG`. Their 20 responses remain separate from 270 full responses. Malformed responses remain malformed. Only identical cloud transport requests may retry: OpenAI uses 1/2/4/8-second delays and Gemini 30/60/120/240 seconds, at most four transport retries. Gemini is sequential with at least eight seconds after a successful request plus optional jitter; a fresh process also waits the configured interval before its first request. Local runtime failures stop that model without fallback or generation retry.

Plans, start markers, raw envelopes and normalized per-trial records are exclusive immutable writes. Raw envelopes are saved before parsing. Resume skips existing records and can reconstruct a missing normalized record from its raw envelope. An interrupted send without a raw envelope is not automatically repeated. Summary JSONL, hash catalogs and reports are derived indexes that may be rebuilt. Scientific response files must never be edited after inference.

Reproduction and validation:

```bash
uv run python benchmark_physical_direct.py --model all --smoke --dry-run
uv run python benchmark_physical_direct.py --model all --full --dry-run
# These two commands make real calls; existing identities cannot be repeated.
uv run python benchmark_physical_direct.py --model all --smoke
uv run python benchmark_physical_direct.py --model all --full
# Existing completed trials are never rerun by --resume.
uv run python benchmark_physical_direct.py --model all --full --resume
uv run python validate_physical_direct.py --current
uv run pytest
```

The `all` runner invokes the separately pinned local environments sequentially. The cloud environment is the repository `.venv`; credentials are loaded only from environment after verifying `.env` is ignored and untracked. Do not put credentials in commands or configuration.

Cost fields, where calculable, use the documented 2026-09-05 pricing snapshot from the frozen cloud baseline and available usage metadata. They are estimates, never exact billed totals. Local GPU inference latency excludes model loading; cloud latency includes API/network transport and transport retries, but excludes pacing. These are different timing scopes and do not establish that a cloud model is faster than a local model.

The final descriptive reports and human-scoring queue are under `results_physical_pilot/direct_v1/`. Candidate matches against the pre-inference visual review are explicitly provisional and require human adjudication. Restaurant phone ownership and navigation route correctness must not be inferred from appearance. Safety output booleans are reported without claiming hazard-veto effectiveness. After human annotation freeze, existing immutable responses can be scored without another inference call.

This is a 54-image pilot, not the planned 112-image controlled corpus. Visually reviewed attack modes include no replacement samples; the captures do not establish controlled C0–C6 conditions. No Oracle or automatic registry experiment begins as part of this work.

## Contributor quick start

Run commands from the repository root. Install the locked CPU/cloud dependencies
with `uv sync --group dev`; `uv run python` selects that environment (a global
`python` executable is not required). The CLI uses **`--model`**, not `--provider`,
and requires either `--smoke` or `--full`.

The archive contains 54 JPEG-family originals: CALL (6), restaurant reservation
(30), navigation (11), and safety (7). Its compressed size is 161,465,722 bytes;
the original image payload totals 162,802,872 bytes. Obtain the exact archive from
the dataset owner and place it at `TestData.zip`. No public dataset download is
currently provided. Git LFS is not configured. Keep the archive and extraction
caches outside Git; do not resize, recompress, or create another input freeze.

```bash
# Offline enumeration; neither models nor archive extraction are required.
uv run python benchmark_physical_direct.py --model all --smoke --dry-run
uv run python benchmark_physical_direct.py --model all --full --dry-run
# Check the frozen metadata; then verify originals and all existing response identities.
uv run python physical_direct_inputs.py
uv run python validate_physical_direct.py --current
```

`--current` verifies available artifacts and reports missing trial counts. It does
not require all five full runs, repair records, write reports, or perform scoring.
The validator without `--current` requires all 270 full and 20 active smoke trials;
it is expected to fail while the experiment is incomplete.

## Local and cloud execution reference

The following commands perform inference and are documentation for a separately
authorized run. This cleanup did not execute them. The CLI uses fixed result
namespaces and refuses existing identities; it has no `--output-dir` option.
Do not delete evidence to make a command run again. `--resume` preserves completed
and failed attempts; any started request without a raw envelope needs manual audit.

Local execution requires CUDA/BF16 support and offline cached model/processor
revisions from `physical_direct_local.py`. Gemma and Qwen require Transformers
5.16.1; MiniCPM requires 4.51.0 and its trusted remote model code. The repository
CPU/cloud environment alone does not install these GPU runtimes. Keep separate
compatible environments and model caches; exact model revisions are mandatory.
For the `--model all` launcher, override interpreter locations as needed:

```bash
export LENSGUARD_GEMMA_PYTHON="$HOME/venvs/lensguard-vlm/bin/python"
export LENSGUARD_MINICPM_PYTHON="$HOME/venvs/lensguard-minicpm/bin/python"
export LENSGUARD_QWEN_PYTHON="$HOME/venvs/lensguard-qwen/bin/python"
# A single local model runs in the interpreter you invoke.
"$LENSGUARD_QWEN_PYTHON" benchmark_physical_direct.py --model qwen --smoke
"$LENSGUARD_GEMMA_PYTHON" benchmark_physical_direct.py --model gemma --smoke
"$LENSGUARD_MINICPM_PYTHON" benchmark_physical_direct.py --model minicpm --smoke
# Requires OPENAI_API_KEY or GEMINI_API_KEY respectively in the environment / ignored .env.
uv run python benchmark_physical_direct.py --model openai --smoke
uv run python benchmark_physical_direct.py --model gemini --smoke
```

The defaults for local interpreters are the same paths under the current user's
home directory. Interpreter overrides do not change prompts or generation settings.
An existing immutable plan may contain its original machine path: retain that
historical metadata verbatim. Changing runtime configuration is not permission to
resume under a different request plan. Cloud model IDs are fixed by this harness.
`GEMINI_REQUEST_DELAY_SECONDS` defaults to 8 and must be finite and at least 8.
Keys must never be placed in a command, tracked config, report, or commit.
Transport retry limits and latency caveats are described above; malformed semantic
outputs never trigger another attempt.

## Artifact map and review trail

All paths below are relative to `results_physical_pilot/direct_v1/`:

| Path | Purpose |
| --- | --- |
| `input_manifest.json`, `input_manifest.sha256`, `input_review_metadata.csv` | Frozen original identities and provisional review metadata |
| `plans/<model>.json` | Immutable exact prompts, settings, and planned identities |
| `started/<model>/direct/*.json` | One-attempt request markers; retain even when no response exists |
| `raw/<model>/direct/*.json` | Immutable original response envelopes |
| `records/<model>/direct/*.json` | Immutable structurally parsed records bound to raw hashes |
| `<model>_hashes.json`, `<model>_manifest.json`, `<model>_normalized.jsonl` | Existing per-model hashes, completion inventory, and record index |
| `<model>_summary.json`, `<model>_summary.md` | Existing descriptive summaries where available |
| `smoke/` | Separate active smoke and preserved OpenAI compatibility diagnostic |
| `cleanup_index.json`, `cleanup_validation.json` | Static cleanup snapshot and read-only validation of stable committed evidence |

The cleanup index links each artifact by repository-relative path, byte count,
and SHA-256. It is a snapshot, not a claim of full-run completion. The Qwen run
was already active when cleanup began and was suspended; another session subsequently resumed it and changed its live indexes. The entire
live Qwen full-run bundle is therefore excluded from this cleanup snapshot. Its
local evidence remains untouched. A process lock is operational state and is ignored.
Gemma and MiniCPM summaries are preserved; absent summaries are not fabricated.

Review exports include `raw_output_text` verbatim, including malformed JSON and
Markdown fences. They do not repair or extract JSON from invalid responses, and
provisional matches do not establish scientific correctness. No response, metric,
input label, or historical Phase 2 / 2.5 / 3.5 / 3.6 / cloud result is rewritten.
Any future annotation or scoring work must preserve these originals and identify
its separate version. Source hashes and completeness checks are distinct from
scientific ground-truth validation.

```bash
# Non-GPU tests use fake transports/models; no inference is needed.
uv run pytest tests/test_physical_direct*.py
uv run pytest
uv run python phase2_benchmark_lock.py
uv run python verify_phase3_5_frozen_baselines.py
uv run pytest tests/test_phase3_6_replay.py tests/test_cloud_integrity.py
uv run python validate_physical_direct.py --current
```
