# LensGuard Phase 2.5 — Local Model Comparison

> Results remain model-separated. `N/A` means no compatible observation was supplied; missing values are never inferred or imputed.

## 1. Research Questions

- RQ1: Can small local VLMs perform the frozen action, argument, and evidence task?
- RQ2: How accurately do they map critical arguments to visual evidence?
- RQ3: Does automatic local provenance remain useful to the Thin Trusted Gate?
- RQ4: What gap remains between automatic local and Oracle provenance?
- RQ5: What latency and GPU-memory cost does each local model impose?
- RQ6: Is a 4B-class VLM security-useful, or is a stronger baseline required?

## 2. Benchmark Freeze

The generator rejects mixed lock digests, datasets, prompts, policies, arm selections, case scopes, or planned cohort sizes before comparison.

| Frozen field | Compatible value |
|---|---|
| Benchmark lock | lensguard-phase2-frozen-v1 |
| Lock SHA-256 | 4262f6d6186ac02f49168543a80093130de53ba12764eddd2283502326b12c4f |
| Dataset | lensguard-phase2-dataset-v1.1.0 |
| Zero-shot prompt cohort | ZERO_SHOT_V2 |
| Schema transport | phase2.5-local-json-schema-transport-v2 |
| Semantic prompt versions | phase2-action-v1, phase2-inline-provenance-v2 |
| Policy | phase2-thin-gate-v2 |
| Selection scope | d54b7949091f9940e4dda8425f22b7c71ff8c1e90fb71107c3a9ca8107353c47 |

## 3. Hardware Environment

| Model | GPU | Total VRAM | Driver | PyTorch / CUDA | Transformers / Python | OS |
|---|---|---:|---|---|---|---|
| Gemma 3 4B | NVIDIA GeForce RTX 4090 | 23.54 GiB | 610.43.02 | 2.10.0+cu128 / 12.8 | 5.16.1 / 3.12.3 | Linux-7.0.0-30-generic-x86_64-with-glibc2.39 |
| Qwen3-VL 8B | NVIDIA GeForce RTX 4090 | 23.54 GiB | 610.43.02 | 2.10.0+cu128 / 12.8 | 5.16.1 / 3.12.3 | Linux-7.0.0-30-generic-x86_64-with-glibc2.39 |
| MiniCPM-V 4.5 | NVIDIA GeForce RTX 4090 | 23.54 GiB | 610.43.02 | 2.10.0+cu128 / 12.8 | 4.51.0 / 3.12.3 | Linux-7.0.0-30-generic-x86_64-with-glibc2.39 |

The RTX 4090 is an evaluation/edge-proxy platform; it does not establish deployment on current glasses hardware.

## 4. Models

Primary comparison (all reported metrics are from `INLINE_PROVENANCE`; Oracle values are derived only from matched supplied cohorts):

| Model | Params | Action Acc | Prov Acc | Halluc Prov | Unsafe Exec | p50 Lat | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma 3 4B | 4.3B | 100.0% | 21.3% | 1.1% | 0.0% | 2465.7 ms | 8.46 GiB |
| Qwen3-VL 8B | 8.77B | 100.0% | 18.5% | 0.0% | 0.0% | 2309.4 ms | 17.00 GiB |
| MiniCPM-V 4.5 | 8.7B | 100.0% | 9.3% | 0.0% | 0.0% | 3245.3 ms | 18.61 GiB |
| Oracle (matched cohorts) | N/A | - | 75.9%–90.4% | 0.0% | 0.0% | - | - |

| Model | Repository / revision | dtype | Quantization | Attention | Cohort status |
|---|---|---|---|---|---|
| Gemma 3 4B | google/gemma-3-4b-it / 093f9f388b31de276ce2de164bdc2081324b9767 | bfloat16 | none | sdpa | Complete benchmark cohort |
| Qwen3-VL 8B | Qwen/Qwen3-VL-8B-Instruct / 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b | bfloat16 | none | sdpa | Complete benchmark cohort |
| MiniCPM-V 4.5 | openbmb/MiniCPM-V-4_5 / daef484c35ec93210ec93c5e901f8f3e9b78ee34 | bfloat16 | none | llm_sdpa_vision_eager | Complete benchmark cohort |

## 5. Experimental Protocol

ZERO_SHOT_V2, schema transport phase2.5-local-json-schema-transport-v2, batch size 1, sampling disabled, and one resident model at a time. The frozen deterministic evidence mapper and model-free Thin Trusted Gate are reused. No weights are trained. Any compatibility normalization is deterministic and explicitly counted; missing or malformed semantics are never guessed.

| Protocol field | Value |
|---|---|
| Selected arms | ACTION_ONLY, INLINE_PROVENANCE, ORACLE_PROVENANCE |
| Selected / benchmark cases | 81 / 81 |
| Runs (derived from planned cohort cardinality) | 1 |
| Planned trials per model | 243 |
| Selection-scope complete | True |
| Primary arms complete | True |
| Full 243-trial arms complete | True |
| Action Only / Inline exact scope match | True |
| Action Only / Inline paired trials | 81 |
| Oracle exact scope match | True |

## 6. Action Extraction Results

These rows retain the frozen Phase 2 completed-trial denominators. Section 9 includes independently scoreable actions from provenance-format failures.

| Model | Arm | Usable | Clean action accuracy | Action-class accuracy | Critical-argument accuracy | Exact attacker adoption |
|---|---|---:|---:|---:|---:|---:|
| Gemma 3 4B | Action Only | 81 | 73.3% | 100.0% | 67.9% | 12.5% |
| Gemma 3 4B | Inline Provenance | 65 | 54.5% | 100.0% | 58.5% | 11.9% |
| Qwen3-VL 8B | Action Only | 81 | 86.7% | 100.0% | 87.7% | 8.3% |
| Qwen3-VL 8B | Inline Provenance | 81 | 93.3% | 100.0% | 86.4% | 8.3% |
| MiniCPM-V 4.5 | Action Only | 81 | 66.7% | 100.0% | 70.4% | 8.3% |
| MiniCPM-V 4.5 | Inline Provenance | 45 | 66.7% | 100.0% | 84.4% | 4.2% |

## 7. Provenance Results

| Model | Arm | Text match | Region accuracy | Source accuracy | Argument provenance | Coverage | Missing | Ambiguous | Hallucinated |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 3 4B | Inline Provenance | 98.5% | 84.6% | 32.3% | 21.3% | 68.5% | 27.0% | 3.4% | 1.1% |
| Gemma 3 4B | Oracle | 100.0% | 100.0% | 100.0% | 75.9% | 75.9% | 24.1% | 0.0% | 0.0% |
| Qwen3-VL 8B | Inline Provenance | 100.0% | 65.3% | 42.6% | 18.5% | 93.5% | 6.5% | 0.0% | 0.0% |
| Qwen3-VL 8B | Oracle | 100.0% | 100.0% | 100.0% | 90.4% | 90.4% | 9.6% | 0.0% | 0.0% |
| MiniCPM-V 4.5 | Inline Provenance | 100.0% | 66.0% | 20.0% | 9.3% | 74.1% | 13.0% | 13.0% | 0.0% |
| MiniCPM-V 4.5 | Oracle | 100.0% | 100.0% | 100.0% | 75.9% | 75.9% | 24.1% | 0.0% | 0.0% |

## 8. Security Results

| Model | Arm | Unsafe automatic execution | Gate escalation recall | False escalation | Trusted-user preservation |
|---|---|---:|---:|---:|---:|
| Gemma 3 4B | Action Only | 12.5% | 0.0% | 0.0% | 100.0% |
| Gemma 3 4B | Inline Provenance | 0.0% | 100.0% | 0.0% | 100.0% |
| Gemma 3 4B | Oracle | 0.0% | 100.0% | 0.0% | 100.0% |
| Qwen3-VL 8B | Action Only | 8.3% | 0.0% | 0.0% | 100.0% |
| Qwen3-VL 8B | Inline Provenance | 0.0% | 100.0% | 0.0% | 100.0% |
| Qwen3-VL 8B | Oracle | 0.0% | 100.0% | 0.0% | 100.0% |
| MiniCPM-V 4.5 | Action Only | 8.3% | 0.0% | 0.0% | 100.0% |
| MiniCPM-V 4.5 | Inline Provenance | 0.0% | 100.0% | 0.0% | 100.0% |
| MiniCPM-V 4.5 | Oracle | 0.0% | 100.0% | 0.0% | 100.0% |

Source estimates remain predictions, not authority. Only mapped evidence and frozen policy metadata enter the model-free gate.

## 9. Structured Output Reliability

Raw structural-schema validity is shown separately from post-normalization acceptance. All rates include their assessed denominator; unassessed trials are not implicit failures, successes, or safe executions.

| Model | Arm | Attempted | Completed | Errors | Runtime | Parse success | Raw structural schema valid | Normalized accepted | Normalizations | Contract semantic | Provenance semantic | Action correct | Critical argument correct | Unsafe / assessed attacks | Gate coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 3 4B | Action Only | 81 | 81 | 0 | 0 | 100.0% (81/81) | 100.0% (81/81) | 100.0% (81/81) | 0 | 100.0% (81/81) | N/A (0/0) | 100.0% (81/81) | 67.9% (55/81) | 12.5% (6/48) | 100.0% |
| Gemma 3 4B | Inline Provenance | 81 | 65 | 16 | 0 | 100.0% (81/81) | 0.0% (0/81) | 80.2% (65/81) | 65 (single_argument_list: 65) | 100.0% (65/65) | 29.2% (19/65) | 100.0% (68/68) | 60.3% (41/68) | 0.0% (0/42) | 87.5% |
| Gemma 3 4B | Overall | 243 | 227 | 16 | 0 | 100.0% (243/243) | 66.7% (162/243) | 93.4% (227/243) | 65 (single_argument_list: 65) | 100.0% (227/227) | 68.5% (100/146) | 100.0% (230/230) | 65.7% (151/230) | 4.3% (6/138) | 95.8% |
| Qwen3-VL 8B | Action Only | 81 | 81 | 0 | 0 | 100.0% (81/81) | 100.0% (81/81) | 100.0% (81/81) | 0 | 100.0% (81/81) | N/A (0/0) | 100.0% (81/81) | 87.7% (71/81) | 8.3% (4/48) | 100.0% |
| Qwen3-VL 8B | Inline Provenance | 81 | 81 | 0 | 0 | 100.0% (81/81) | 100.0% (81/81) | 100.0% (81/81) | 0 | 100.0% (81/81) | 4.9% (4/81) | 100.0% (81/81) | 86.4% (70/81) | 0.0% (0/48) | 100.0% |
| Qwen3-VL 8B | Overall | 243 | 243 | 0 | 0 | 100.0% (243/243) | 100.0% (243/243) | 100.0% (243/243) | 0 | 100.0% (243/243) | 52.5% (85/162) | 100.0% (243/243) | 87.2% (212/243) | 2.8% (4/144) | 100.0% |
| MiniCPM-V 4.5 | Action Only | 81 | 81 | 0 | 0 | 100.0% (81/81) | 100.0% (81/81) | 100.0% (81/81) | 0 | 100.0% (81/81) | N/A (0/0) | 100.0% (81/81) | 70.4% (57/81) | 8.3% (4/48) | 100.0% |
| MiniCPM-V 4.5 | Inline Provenance | 81 | 45 | 36 | 0 | 79.0% (64/81) | 70.3% (45/64) | 70.3% (45/64) | 0 | 100.0% (45/45) | 11.1% (5/45) | 100.0% (64/64) | 89.1% (57/64) | 0.0% (0/24) | 50.0% |
| MiniCPM-V 4.5 | Overall | 243 | 207 | 36 | 0 | 93.0% (226/243) | 91.6% (207/226) | 91.6% (207/226) | 0 | 100.0% (207/207) | 68.3% (86/126) | 100.0% (226/226) | 75.7% (171/226) | 3.3% (4/120) | 83.3% |

A normalized output remains a raw-schema miss. Invalid structured output remains invalid; safe fence removal or JSON extraction does not authorize guessing a missing critical argument, evidence value, argument-to-evidence association, or coordinate.

## 10. Latency

| Model | Load | p50 inference | p95 inference | p50 end-to-end | p95 end-to-end | p50 mapper | p50 gate | p50 tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 3 4B | 5272.4 ms | 815.2 ms | 2915.7 ms | 817.2 ms | 2917.4 ms | 0.2 ms | 0.3 ms | 53.37 |
| Qwen3-VL 8B | 8623.5 ms | 848.6 ms | 3107.3 ms | 850.2 ms | 3109.9 ms | 0.2 ms | 0.3 ms | 36.09 |
| MiniCPM-V 4.5 | 7826.5 ms | 991.9 ms | 3415.9 ms | 993.5 ms | 3420.0 ms | 0.2 ms | 0.3 ms | 29.36 |

Latency is descriptive for the recorded runtime profile; smoke samples do not provide stable p95 estimates.

## 11. VRAM

| Model | Before inference (max) | Peak allocated | Peak reserved | Total device VRAM |
|---|---:|---:|---:|---:|
| Gemma 3 4B | 8.03 GiB | 8.46 GiB | 8.54 GiB | 23.54 GiB |
| Qwen3-VL 8B | 16.36 GiB | 17.00 GiB | 17.41 GiB | 23.54 GiB |
| MiniCPM-V 4.5 | 17.39 GiB | 18.61 GiB | 18.95 GiB | 23.54 GiB |

## 12. Model-by-Model Failure Cases

The aggregate analysis retains failure counts and rates, not arbitrary reconstructed examples. Inspect each model's `report.md` and `raw_generations.jsonl` for recorded cases.

| Model | Cohort | Unresolved errors | Runtime errors | Raw-schema failures | Normalizations | Failure categories | Inline missing provenance | Inline hallucinated evidence | Source |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| Gemma 3 4B | Complete benchmark cohort | 16 | 0 | 81 | 65 | critical_argument_prediction_failure=79, provenance_semantic_failure=46, schema_mismatch=16 | 27.0% | 1.1% | results_phase2_5/contract-v2-full/gemma3-4b/raw_generations.jsonl |
| Qwen3-VL 8B | Complete benchmark cohort | 0 | 0 | 0 | 0 | critical_argument_prediction_failure=31, provenance_semantic_failure=77 | 6.5% | 0.0% | results_phase2_5/contract-v2-full/qwen3vl-8b/raw_generations.jsonl |
| MiniCPM-V 4.5 | Complete benchmark cohort | 36 | 0 | 19 | 0 | critical_argument_prediction_failure=55, malformed_json=17, provenance_semantic_failure=40, schema_mismatch=19 | 13.0% | 0.0% | results_phase2_5/contract-v2-full/minicpm-v4.5/raw_generations.jsonl |

## 13. Oracle Gap

Gaps are `INLINE_PROVENANCE - ORACLE_PROVENANCE`; negative and positive signs are preserved.

| Model | Oracle usable | Oracle provenance accuracy | Oracle hallucinated evidence | Oracle unsafe execution | Inline−Oracle provenance gap | Inline−Oracle unsafe gap |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 3 4B | 81 | 75.9% | 0.0% | 0.0% | -54.5% | 0.0% |
| Qwen3-VL 8B | 81 | 90.4% | 0.0% | 0.0% | -71.9% | 0.0% |
| MiniCPM-V 4.5 | 81 | 75.9% | 0.0% | 0.0% | -66.6% | 0.0% |

## 14. Local-vs-Cloud Comparison Placeholder

Cloud files are optional and are not required to generate this report. They may be added later only after the same frozen lock and scientific protocol are verified.

| Provider/model | Status |
|---|---|
| Gemini Flash | Awaiting compatible frozen Phase 2 results |
| OpenAI model | Awaiting compatible frozen Phase 2 results |

## 15. Limitations

- Smoke/partial cohorts measure compatibility, not full-benchmark scientific outcomes.
- Evidence is observable attribution, not chain of thought or causal proof.
- A source-type estimate does not establish authority.
- Synthetic scenes and annotated regions do not reproduce physical wearable input.
- Mapper timing excludes production OCR, segmentation, and region acquisition.
- Missing confidence remains invalid under the frozen schema; confidence is not invented.
- RTX 4090 measurements do not prove deployment on current glasses hardware.

## 16. Phase 2.5 Go / No-Go

Complete cohorts available for evidence review: Gemma 3 4B, MiniCPM-V 4.5, Qwen3-VL 8B. This report intentionally does not collapse the evidence into an arbitrary score.

GO evidence requires useful grounded provenance, a substantial Thin Gate reduction in unsafe execution, high clean-action utility, manageable hallucination and parse failure, usable event-driven latency, and comfortable VRAM headroom. Pivot evidence includes low coverage, frequent hallucination, unreliable structure, collapsed action accuracy, high false escalation, impractical latency, or a large automatic-to-Oracle gap.

Phase 2.6 fine-tuning is justified only if the preserved zero-shot baseline is inadequate but exhibits learnable provenance signal; the physical holdout must never enter training. Phase 3 physical experiments are justified when at least one complete local cohort shows grounded provenance across action families, meaningful gate security gain, preserved utility, tolerable false escalation, usable runtime, and an acceptable Oracle gap.
