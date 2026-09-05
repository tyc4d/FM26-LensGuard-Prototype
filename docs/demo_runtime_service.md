# Local Demo Runtime Service

The independent `FM26-LensGuard-Demo` presentation app calls this repository over loopback HTTP. No repository merge, cross-repository Python imports, model code copies, or editable package installation is needed. Benchmark modules/configuration are unchanged.

## Start

First inspect `nvidia-smi` and experiment processes. Phase 3.6 takes priority; never stop, pause, restart or renice it for the demo. No other compute process may be active on GPU 0. Leave at least 14,000 MiB free for first load (BF16 weights plus conservative input/generation reserve); subsequent requests require 4,096 MiB free. The preflight checks process ownership and logs GPU name/total/used/free. This is a point-in-time check, not a GPU reservation against future experiment launches.

```bash
cd /home/tyc4d/FM26-LensGuard-Prototype
source /home/tyc4d/venvs/lensguard-vlm/bin/activate
# Only web packages, after confirming existing AnyIO/Pydantic/Click/h11 dependencies:
python -m pip install --no-deps -r requirements/demo-runtime.txt
python -m prototype_demo_server --model gemma3-4b --host 127.0.0.1 --port 8010
```

This uses the verified PyTorch 2.10.0+cu128 / Transformers 5.16.1 environment. No torch/transformers/CUDA upgrades are part of setup. The service validates those versions before loading. Downloads are disabled, and Gemma's existing pinned model/processor revision is reused from the physical-local profile: `093f9f388b31de276ce2de164bdc2081324b9767`. The model cache must already exist. Do not run multiple workers, reload, or a second VLM family. Check `/health` before starting another service.

`create_local_provider('gemma3-4b')` uses the existing Gemma3Provider. `invoke_phase3_5(... ACTION_ONLY ...)` supplies the frozen prompt, native image preprocessing, deterministic BF16/SDPA generation (1024 token limit), and existing JSON extraction / Phase35ActionOutput parser. The trusted user request and visual input remain distinct; scenario IDs never add attacker text or dictate answers. Model loading is lazy and idempotent; the same provider remains resident across requests.

## API

- `GET /health`: status `unloaded`, `loading`, `processing`, `ready` or `error`, model_loaded, profile, CUDA device, component limitations and load error. Health does not load a model.
- `POST /v1/analyze`: multipart `image` JPEG/PNG <=10 MiB and 16 MP, `user_request` <=4000 characters, optional `scenario_id`, `mode=action_only`. Other modes are rejected.
- `POST /warmup`: optional one-time blank-image inference, protected by the same GPU preflight. Repeated successful warmup returns already_warmed. Never warm up during experiment contention.

Response `contract_version=lensguard-demo-v1` carries model identity, input acknowledgement, raw_text, parsed, proposed_action, native_action, parser diagnostics, provenance, policy and actual timing. Invalid model output is HTTP 200 with parsed=false and null proposed_action/policy, preserving raw text. Invalid input is 422/413, busy/loading contention 409, GPU/runtime failures 503. No model retry or mock fallback is performed.

Temporary images are deleted after inference; cancellation waits for the worker before releasing the model lock or deleting its image. Timing includes inference, generation, parsing-plus-metadata, deterministic policy, and total request handling; first total includes model loading. Raw images and outputs are never written into benchmark results.

## Authorization boundary and limitations

The VLM only proposes. Existing `firewall.thin_gate.evaluate_thin_gate` handles CALL, OPEN_URL, DIRECTION_ADVICE and NONE with honestly missing evidence. Its native CONFIRM/WARN/BLOCK means no automatic execution; the Demo renders BLOCKED and preserves native details. Unsupported policy actions fail closed.

The separate `DEMO_SCOPED_CARD_CALL_DELEGATION_V1` rule permits one proposed CALL only for the exact trusted request `幫我撥打這張名片上的電話`. This is a narrowly scoped demo authorization, **not** proof the number was visually grounded on the card. It does not claim general natural-language delegation or authenticity. Scenario metadata, model-generated policy text and environmental text cannot activate the rule. It does not modify benchmark policy.

Automatic perception is currently an abstract backend interface; complete Phase 3.5/3.6 grounding requires an independently constructed registry. Thus provenance here is actual transport lineage only, with semantic_grounding=unavailable. No fake OCR, regions, model-estimated authority or oracle scenario annotation is supplied. Full grounding is future work inside this service. All external effects stay simulated in the Demo.

## CPU tests

Install web dependencies into the repository's ordinary test environment, not a replacement CUDA environment, and run:

```bash
.venv/bin/python -m pytest tests/test_demo_runtime_service.py tests/test_local_vlm_providers.py tests/test_benchmark_phase3_5.py tests/test_phase2_thin_gate.py tests/test_physical_direct_local.py
```

The service factory accepts a fake adapter for tests only; the CLI exposes no fake mode. Tests require no GPU or model load.

## Validation observed 2026-09-05

No Phase 3.6 process was visible during initial inspection or immediately before smoke inference. RTX 4090 initially used 15 / 24,564 MiB. No existing process was stopped or changed. Only web transport packages were added to the verified model environment, with dependency resolution disabled; torch/transformers versions stayed unchanged.

Real smoke used a temporary 1280×800 business-card image with ABC Bistro and 02-2345-6789. Model output was a fenced JSON CALL with target_number=02-2345-6789; the existing parser accepted it. First request total 7,982 ms, inference 1,550 ms, model load 4,928 ms. Peak allocated memory 8,866,027,008 bytes; resident nvidia-smi use about 9,013 MiB. Subsequent requests reused the loaded model (~951–977 ms inference) and verified ALLOW, withheld/CONFIRM, and Demo Guard OFF simulation through SSE. This is an integration smoke, not attack-defense evaluation. No physical webcam was used.

After validation, only this task's temporary Demo/Prototype servers were stopped to release VRAM for research. Final GPU usage returned to 15 MiB with no compute processes. Existing mock development servers were preserved. The affected CPU regression suite passed 92 tests.

### Invalid but syntactically decoded actions

The response additionally exposes `output.candidate_action`, taken directly from the existing invocation's decoded JSON payload. It is not an authorized or schema-valid proposal. `parsed=false` and null proposed_action/policy remain unchanged when validation fails. For example, RESTAURANT_RESERVATION requires a positive integer party_size; the model's string "N/A" remains invalid and is never repaired or invented. This lets the Demo show actual candidate fields and parser diagnostics while terminating safely. Benchmark prompts and schemas remain unchanged.

### Parsed but unusable action arguments

The demo boundary separately checks CALL, OPEN_URL, DIRECTION_ADVICE and NONE with the existing action normalizer before authorization or scoped delegation. A string such as `direction: "未知"` can pass the frozen parser while still being unusable as a direction. These responses preserve `parsed=true`, parser diagnostics, raw text, native action and candidate, and return an explicit `output.validation_error` with null proposed_action/policy. They do not report an infrastructure policy failure or substitute a guessed direction. The Demo displays the candidate as invalid and terminates without authorization or simulated execution, even with Guard OFF. Unexpected policy failures remain separately reported in `output.policy_error`.

The demo adapter recognizes explicit traditional/simplified Chinese direction aliases, such as `向右` → `RIGHT`, `向左` → `LEFT`, `直走` → `STRAIGHT`, and `向後` → `BACK`, plus cardinal/intercardinal names. Only a separate copy passed to the gate is canonicalized; raw text, native/candidate actions and displayed proposals retain the original Chinese. The policy's native critical arguments expose the canonical direction, and navigation identifies `navigate.direction` regardless of argument order. Matching is exact after Unicode/whitespace and terminal punctuation normalization: unknown, negated or ambiguous phrases are rejected rather than interpreted. Language normalization grants no authority; missing semantic evidence still requires confirmation. The frozen benchmark normalizer and model prompts remain unchanged.
