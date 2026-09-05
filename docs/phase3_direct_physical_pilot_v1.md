# Phase 3 physical pilot: DIRECT v1

Experiment: `lensguard-phase3-physical-direct-v1`. This records raw model behavior on the 54 reviewed physical photographs. Human-reviewed ground truth is not frozen. Completion, JSON structure, emitted arguments, latency and usage are descriptive observations; they are not accuracy, attack success, defense success or unsafe execution measurements.

The frozen input manifest and copied review metadata live in `results_physical_pilot/direct_v1/`. `IMG_3485.jpeg` remains included and carries `inference_contamination_risk=true` because laptop screens contain experiment/model-related text. Reports must show that frame separately and identify denominators when excluding it.

The archive hash is `b6aadea97982cdb7d8383948a912cc334e349c2b880ff05a0d09ea06d186d25a`. Originals remain in the archive and a temporary byte-identical cache. They are neither renamed nor recompressed nor committed. Reproduction requires the same `TestData.zip` at the repository root; the immutable manifest verifies every original filename, hash and dimension.

`config/physical_direct_prompts_v1.yaml` fixes one task prompt per scenario and one common output schema before inference. The model receives only image bytes/pixels, the scenario task and the shared JSON instructions. No review notes, control labels, injected candidates, region annotations, provenance or expected answers enter its input. No evidence registry, grounding validator, conflict evaluator, deterministic gate or external verification runs in this experiment. Tool lists are empty for both cloud providers.

Models are the frozen Gemma 3 4B, MiniCPM-V 4.5 and Qwen3-VL 8B revisions specified in `physical_direct_local.py`, plus `gpt-5.6-sol` and `gemini-3.1-flash-lite`. Local runs use their previously frozen Python environments, BF16, CUDA, batch one, greedy generation and 1,024 maximum new tokens. Cloud native configuration is preserved in every request plan and response. OpenAI uses Responses; Gemini uses Google GenAI Interactions. Model fallback is forbidden.

Source image bytes are never changed. Local decoding applies EXIF orientation in memory before the frozen family processor; cloud APIs receive original JPEG-family bytes and use provider-native decoding. Cloud EXIF handling and internal resizing are not independently observable. Local output is constrained by the shared prompt; cloud APIs additionally receive native JSON schema syntax. These implementation differences are recorded and limit claims of identical decoding conditions.

Each image/model/run-type identity gets one semantic attempt. The four predeclared smoke images are `IMG_3483.jpeg`, `IMG_6164.JPG`, `IMG_6152.JPG` and `IMG_6157.JPG`. Their 20 responses remain separate from 270 full responses. Malformed responses remain malformed. Only identical cloud transport requests may retry: OpenAI uses 1/2/4/8-second delays and Gemini 30/60/120/240 seconds, at most four transport retries. Gemini is sequential with at least eight seconds after a successful request plus optional jitter; a fresh process also waits the configured interval before its first request. Local runtime failures stop that model without fallback or generation retry.

Plans, start markers, raw envelopes and normalized per-trial records are exclusive immutable writes. Raw envelopes are saved before parsing. Resume skips existing records and can reconstruct a missing normalized record from its raw envelope. An interrupted send without a raw envelope is not automatically repeated. Summary JSONL, hash catalogs and reports are derived indexes that may be rebuilt. Scientific response files must never be edited after inference.

Reproduction and validation:

```bash
uv run python benchmark_physical_direct.py --model all --smoke --dry-run
uv run python benchmark_physical_direct.py --model all --full --dry-run
# These two commands make real calls; run once in a fresh result namespace only.
uv run python benchmark_physical_direct.py --model all --smoke
uv run python benchmark_physical_direct.py --model all --full
# Existing completed trials are never rerun by --resume.
uv run python benchmark_physical_direct.py --model all --full --resume
uv run python validate_physical_direct.py
uv run pytest
```

The `all` runner invokes the separately pinned local environments sequentially. The cloud environment is the repository `.venv`; credentials are loaded only from environment after verifying `.env` is ignored and untracked. Do not put credentials in commands or configuration.

Cost fields, where calculable, use the documented 2026-09-05 pricing snapshot from the frozen cloud baseline and available usage metadata. They are estimates, never exact billed totals. Local GPU inference latency excludes model loading; cloud latency includes API/network transport and transport retries, but excludes pacing. These are different timing scopes and do not establish that a cloud model is faster than a local model.

The final descriptive reports and human-scoring queue are under `results_physical_pilot/direct_v1/`. Candidate matches against the pre-inference visual review are explicitly provisional and require human adjudication. Restaurant phone ownership and navigation route correctness must not be inferred from appearance. Safety output booleans are reported without claiming hazard-veto effectiveness. After human annotation freeze, existing immutable responses can be scored without another inference call.

This is a 54-image pilot, not the planned 112-image controlled corpus. Visually reviewed attack modes include no replacement samples; the captures do not establish controlled C0–C6 conditions. No Oracle or automatic registry experiment begins as part of this work.
