# LensGuard Phase 3.5 — Grounded Provenance Local Models

> The action model selects only pre-existing evidence IDs. Authorization, grounding validation, task policy, and gate decisions are deterministic and model-free.
> All current registries use benchmark annotations (ORACLE PERCEPTION); these results do not measure OCR, detection, or physical-scene perception.

## Primary 81-case comparison

Every percentage retains its numerator/denominator. A `coverage` suffix means eligible trials were unassessed; in particular, runtime errors are not counted as successful security defenses.

Assessed utility conditions on a usable action proposal. End-to-end utility uses all 81 trials, so contract failures cannot appear as perfect utility.

| Model | Arm | Completed | Action assessed | Action E2E | Critical args assessed | Critical args E2E | Exact all evidence | Exact camera region | Unknown/invented IDs | Unsafe execution | Escalation recall | Inference p50 / p95 | Peak allocated / reserved |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 3 4B | ACTION_ONLY | 81/81 | 100.0% (81/81) | 100.0% (81/81) | 92.6% (75/81) | 92.6% (75/81) | N/A | N/A | N/A | 10.4% (5/48) | N/A (0/0) | 913.7 ms / 969.9 ms | 8.26 GiB / 8.61 GiB |
| Gemma 3 4B | GROUNDED_REGISTRY | 50/81 | 100.0% (50/50; coverage 50/81) | 61.7% (50/81) | 92.0% (46/50; coverage 50/81) | 56.8% (46/81) | 50.0% (54/108) | 48.5% (32/66) | 0.0% (0/74) | 0.0% (0/29; coverage 29/48) | 100.0% (4/4) | 1904.4 ms / 2828.5 ms | 8.43 GiB / 8.61 GiB |
| Gemma 3 4B | ORACLE | 81/81 | 100.0% (81/81) | 100.0% (81/81) | 92.6% (75/81) | 92.6% (75/81) | 93.5% (101/108) | 92.4% (61/66) | 0.0% (0/106) | 6.2% (3/48) | 40.0% (2/5) | 911.8 ms / 969.8 ms | 8.26 GiB / 8.61 GiB |
| MiniCPM-V 4.5 | ACTION_ONLY | 81/81 | 100.0% (81/81) | 100.0% (81/81) | 95.1% (77/81) | 95.1% (77/81) | N/A | N/A | N/A | 8.3% (4/48) | N/A (0/0) | 982.8 ms / 1009.5 ms | 18.61 GiB / 18.95 GiB |
| MiniCPM-V 4.5 | GROUNDED_REGISTRY | 81/81 | 100.0% (81/81) | 100.0% (81/81) | 92.6% (75/81) | 92.6% (75/81) | 66.7% (72/108) | 83.3% (55/66) | 0.0% (0/136) | 4.2% (2/48) | 66.7% (4/6) | 1596.9 ms / 2269.5 ms | 18.61 GiB / 18.95 GiB |
| MiniCPM-V 4.5 | ORACLE | 81/81 | 100.0% (81/81) | 100.0% (81/81) | 95.1% (77/81) | 95.1% (77/81) | 96.3% (104/108) | 93.9% (62/66) | 0.0% (0/107) | 6.2% (3/48) | 25.0% (1/4) | 982.4 ms / 1009.5 ms | 18.61 GiB / 18.95 GiB |
| Qwen3-VL 8B | ACTION_ONLY | 81/81 | 100.0% (81/81) | 100.0% (81/81) | 95.1% (77/81) | 95.1% (77/81) | N/A | N/A | N/A | 8.3% (4/48) | N/A (0/0) | 823.0 ms / 865.7 ms | 16.83 GiB / 17.38 GiB |
| Qwen3-VL 8B | GROUNDED_REGISTRY | 81/81 | 100.0% (81/81) | 100.0% (81/81) | 96.3% (78/81) | 96.3% (78/81) | 96.3% (104/108) | 95.5% (63/66) | 0.0% (0/108) | 6.2% (3/48) | 0.0% (0/3) | 1847.3 ms / 2219.6 ms | 16.97 GiB / 17.38 GiB |
| Qwen3-VL 8B | ORACLE | 81/81 | 100.0% (81/81) | 100.0% (81/81) | 95.1% (77/81) | 95.1% (77/81) | 96.3% (104/108) | 93.9% (62/66) | 0.0% (0/108) | 6.2% (3/48) | 25.0% (1/4) | 824.5 ms / 865.8 ms | 16.83 GiB / 17.38 GiB |

## Grounded Registry gate behavior

Recall counts both ESCALATE and BLOCK as successful intervention on an adopted attacker target. False escalation is measured on correct proposals in non-attack cases. End-to-end preservation counts an unusable trial as not preserved.

| Model | Escalation recall | False escalation | Clean preservation (conditional / E2E) | Trusted-user preservation (conditional / E2E) | Decision distribution |
|---|---:|---:|---:|---:|---|
| Gemma 3 4B | 100.0% (4/4) | 31.6% (6/19) | 76.9% (10/13) / 66.7% (10/15) | 50.0% (3/6) / 20.0% (3/15) | ALLOW=13, BLOCK=14, ESCALATE=23 |
| MiniCPM-V 4.5 | 66.7% (4/6) | 3.2% (1/31) | 100.0% (15/15) / 100.0% (15/15) | 100.0% (15/15) / 100.0% (15/15) | ALLOW=32, BLOCK=7, ESCALATE=42 |
| Qwen3-VL 8B | 0.0% (0/3) | 3.3% (1/30) | 100.0% (15/15) / 100.0% (15/15) | 93.3% (14/15) / 93.3% (14/15) | ALLOW=32, ESCALATE=49 |

## Answers to the Phase 3.5 comparison questions

The closest Phase 2.5 and Phase 3.5 provenance measures still expose different contracts. The table therefore retains both denominators and treats percentage-point comparisons as directional, not as a formally identical metric.

| Model | P2.5 Inline trial semantic | P2.5 Inline argument provenance | P3.5 Grounded exact all evidence | P3.5 Grounded exact camera | P3.5 Oracle exact all evidence | P3.5 Oracle exact camera | Oracle − Grounded gap (all / camera) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma 3 4B | 29.2% (19/65; coverage 65/81) | 21.3% (19/89) | 50.0% (54/108) | 48.5% (32/66) | 93.5% (101/108) | 92.4% (61/66) | +43.5 percentage points / +43.9 percentage points |
| MiniCPM-V 4.5 | 11.1% (5/45; coverage 45/81) | 9.3% (5/54) | 66.7% (72/108) | 83.3% (55/66) | 96.3% (104/108) | 93.9% (62/66) | +29.6 percentage points / +10.6 percentage points |
| Qwen3-VL 8B | 4.9% (4/81) | 18.5% (20/108) | 96.3% (104/108) | 95.5% (63/66) | 96.3% (104/108) | 93.9% (62/66) | +0.0 percentage points / -1.5 percentage points |

### 1. Does evidence-ID selection improve semantic provenance over Inline Provenance?

Answer: **YES, DIRECTIONALLY FOR EVERY AVAILABLE MODEL.** This is a directional comparison because the historical and current contracts are not identical.

- Gemma 3 4B: YES, DIRECTIONALLY HIGHER — Grounded exact all-argument selection 50.0% (54/108) versus Inline argument provenance 21.3% (19/89).
- MiniCPM-V 4.5: YES, DIRECTIONALLY HIGHER — Grounded exact all-argument selection 66.7% (72/108) versus Inline argument provenance 9.3% (5/54).
- Qwen3-VL 8B: YES, DIRECTIONALLY HIGHER — Grounded exact all-argument selection 96.3% (104/108) versus Inline argument provenance 18.5% (20/108).

### 2. Does it reduce hallucinated provenance?

Answer: **The free-form hallucination channel is eliminated by contract.** The VLM cannot emit authoritative evidence text, bbox, source labels, semantic roles, or confidence. The remaining empirical analogue is selection of an unknown or invented ID:

- Gemma 3 4B: Inline hallucinated evidence 1.1% (1/92); Grounded unknown/invented IDs 0.0% (0/74).
- MiniCPM-V 4.5: Inline hallucinated evidence 0.0% (0/60); Grounded unknown/invented IDs 0.0% (0/136).
- Qwen3-VL 8B: Inline hallucinated evidence 0.0% (0/108); Grounded unknown/invented IDs 0.0% (0/108).

### 3. Does it reduce unknown or invented evidence?

Answer: **NOT DIRECTLY COMPARABLE ACROSS PHASES.** Inline Provenance had no pre-built ID universe. In Phase 3.5, unknown/invented evidence IDs are rejected without repair. Observed rates are listed below; malformed reference containers are structural failures and are not mislabeled as invented IDs.

- Gemma 3 4B: unknown/invented IDs 0.0% (0/74); all invalid IDs 0.0% (0/74); malformed reference-container trials 37.0% (30/81).
- MiniCPM-V 4.5: unknown/invented IDs 0.0% (0/136); all invalid IDs 0.0% (0/136); malformed reference-container trials 7.4% (6/81).
- Qwen3-VL 8B: unknown/invented IDs 0.0% (0/108); all invalid IDs 0.0% (0/108); malformed reference-container trials 0.0% (0/81).

### 4. Does it reduce unsafe execution?

Answer: The primary comparison is Grounded Registry versus the ungated Action Only arm. Unsafe execution is reported over execution-assessed attack trials; coverage remains visible, so an unresolved model error is not credited as a defense. Historical Inline rates are secondary and use the frozen Phase 2.5 gate; an improvement over Action Only does not imply an improvement over that historical gate.

- Gemma 3 4B: versus Action Only, **LOWER AMONG ASSESSED TRIALS; FULL-COHORT INCONCLUSIVE** — Action Only 10.4% (5/48), Grounded 0.0% (0/29; coverage 29/48). Versus historical Inline: **FULL-COHORT INCONCLUSIVE DUE TO PARTIAL ASSESSMENT** — Inline 0.0%.
- MiniCPM-V 4.5: versus Action Only, **YES, LOWER** — Action Only 8.3% (4/48), Grounded 4.2% (2/48). Versus historical Inline: **NO, HIGHER** — Inline 0.0%.
- Qwen3-VL 8B: versus Action Only, **YES, LOWER** — Action Only 8.3% (4/48), Grounded 6.2% (3/48). Versus historical Inline: **NO, HIGHER** — Inline 0.0%.

### 5. Does it preserve critical-argument accuracy?

Answer: Assessed and end-to-end results are both shown. End-to-end uses all 81 cases and therefore exposes any contract failures.

- Gemma 3 4B: **NO, LOWER END-TO-END.** Action Only assessed 92.6% (75/81), end-to-end 92.6% (75/81); Grounded assessed 92.0% (46/50; coverage 50/81), end-to-end 56.8% (46/81); Oracle end-to-end 92.6% (75/81); historical Inline assessed 58.5% (38/65; coverage 65/81), all-trial 46.9% (38/81).
- MiniCPM-V 4.5: **NO, LOWER END-TO-END.** Action Only assessed 95.1% (77/81), end-to-end 95.1% (77/81); Grounded assessed 92.6% (75/81), end-to-end 92.6% (75/81); Oracle end-to-end 95.1% (77/81); historical Inline assessed 84.4% (38/45; coverage 45/81), all-trial 46.9% (38/81).
- Qwen3-VL 8B: **YES, AND HIGHER.** Action Only assessed 95.1% (77/81), end-to-end 95.1% (77/81); Grounded assessed 96.3% (78/81), end-to-end 96.3% (78/81); Oracle end-to-end 95.1% (77/81); historical Inline assessed 86.4% (70/81), all-trial 86.4% (70/81).

### 6. Does Qwen still show perfect structure but poor semantic grounding?

Answer: **NO.** Grounded schema validity 100.0% (81/81), evidence-reference contract 100.0% (81/81), exact all-argument selection 96.3% (104/108), and exact camera-region selection 95.5% (63/66). Grounding statuses were SUPPORTED 54.6% (59/108) and CONFLICTING 45.4% (49/108).

`CONFLICTING` is not automatically a semantic selection error: it can record correctly selected evidence in a registry that also contains a contradictory candidate.

### 7. How large is the gap to Oracle?

Answer: The exact-selection gaps are reported independently for all argument provenance channels and for camera regions only. Oracle assigns references to its unchanged Action Only proposal; it does not correct that proposal. Grounded uses a different model contract, so Oracle is not a mathematical ceiling and a small negative gap is possible:

- Gemma 3 4B: all-argument gap +43.5 percentage points; camera-region gap +43.9 percentage points.
- MiniCPM-V 4.5: all-argument gap +29.6 percentage points; camera-region gap +10.6 percentage points.
- Qwen3-VL 8B: all-argument gap +0.0 percentage points; camera-region gap -1.5 percentage points.

## Corpus scope

The compatible corpus contains 27 CALL, 27 OPEN_URL, and 27 DIRECTION_ADVICE cases. It contains no physical safety or restaurant-reservation task.

- SAFETY_ADVICE: **NOT MEASURABLE IN CURRENT CORPUS**
- RESTAURANT_RESERVATION: **NOT MEASURABLE IN CURRENT CORPUS**
- Physical C0–C6 perception: **NOT MEASURABLE IN CURRENT CORPUS**

The software schema covers all 16 planned base scenes and seven capture conditions (112 image records), while real collection and automatic-perception validation remain future work.

Report version: `phase3.5-report-v1`.
