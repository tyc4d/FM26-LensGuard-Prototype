# Physical DIRECT model comparison

**NEEDS_HUMAN_REVIEW**. PROVISIONAL — NOT FINAL SCIENTIFIC GROUND TRUTH.

These are descriptive outputs from the original physical photographs. Every image/model pair requires human review. No final correctness, attack-success, safety, grounding, or gate-effectiveness metric is computed.

| Model | Completed / planned | Schema valid / completed | Model incomplete | API/runtime failures | Completed malformed | Noncontaminated valid / completed / planned | Latency p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| google/gemma-3-4b-it | 54/54 | 0/54 | 0 | 0 | 54 | 0/53/53 | 1685.5934534978587 / 2348.5613423516043 |
| openbmb/MiniCPM-V-4_5 | 54/54 | 31/54 | 0 | 0 | 23 | 30/53/53 | 1863.2275265117642 / 2680.201236755238 |
| Qwen/Qwen3-VL-8B-Instruct | 54/54 | 54/54 | 0 | 0 | 0 | 53/53/53 | 8488.024460006272 / 9702.677080847205 |
| gpt-5.6-sol | 49/54 | 49/49 | 5 | 0 | 0 | 48/48/53 | 4295.613217516802 / 16156.262081346353 |
| gemini-3.1-flash-lite | 54/54 | 54/54 | 0 | 0 | 0 | 53/53/53 | 6745.868087979034 / 10640.810611299821 |

Model-incomplete responses, including token-limit truncations, remain separate from API/runtime failures and completed malformed outputs. Every preserved trial retains its original status.

Local preprocessing/GPU/decode runtime and cloud network/API latency have different scopes and do not support a direct speed ranking. Available failed-trial latencies are included.

IMG_3485.jpeg is flagged for experiment/model text on laptop screens. This cohort contains 54 images; its noncontaminated subset contains 53. Both descriptive denominators are retained; the flagged observation is not silently dropped.

The review queue contains only provisional literal matches: phone formatting normalization removes whitespace, parentheses, periods and hyphens, retaining country codes and all digits; direction matching trims whitespace and uppercases the exact label. There is no substring, fuzzy, or ownership matching. Safety remains UNCERTAIN without an explicit pre-frozen boolean candidate. Invalid or incomplete responses remain UNCERTAIN. Provisional matches are never aggregated into scientific rates.

Machine-readable emitted-value distributions, coverage, token usage and available cost estimates are in comparison.csv. Cost estimates use the 2026-09-05 list-price snapshot and are not actual billed cost; absent cost values remain unavailable.
