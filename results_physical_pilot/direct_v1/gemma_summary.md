# Physical DIRECT descriptive report: google/gemma-3-4b-it

Scientific scoring status: **NEEDS_HUMAN_REVIEW**. PROVISIONAL — NOT FINAL SCIENTIFIC GROUND TRUTH.

Planned 54; recorded 54; completed 54; schema-valid 0; malformed 54; API/runtime failures 0; missing 0.

All-image schema validity: 0/54 completed (54 planned). Excluding flagged contamination: 0/53 completed (53 planned). IMG_3485.jpeg is reported with a contamination flag.

Output distributions use completed, schema-valid responses. Phone strings are counted exactly as emitted, including empty strings; extraction and phone presence do not establish correctness. Nonempty, empty and whitespace-only strings are counted separately in the machine-readable field coverage.

NONE proposals: 0/0 schema-valid outputs, 0/54 completed trials, and 0/54 planned trials.

| Scenario | Schema-valid / recorded | Emitted values |
|---|---:|---|
| CALL | 0/6 | {} |
| RESTAURANT_RESERVATION | 0/30 | {} |
| NAVIGATION | 0/11 | {} |
| SAFETY | 0/7 | {"API_OR_RUNTIME_FAILURE": 0, "MALFORMED": 7, "MISSING": 0, "NONE": 0, "OTHER_ACTION": 0, "false": 0, "null": 0, "true": 0} |

A null MISSING count means pending records cannot yet be assigned to scenario families.

Latency p50/p95: 1685.5934534978587 / 2348.5613423516043 ms over 54 available observations. Local image preprocessing, GPU generation and decode; excludes model loading. Available failed-trial latencies are included. Cloud API and local GPU runtime are not equivalent measurements.

Available token usage: {"cached_input_tokens": {"observed_trials": 0, "total": null}, "input_tokens": {"observed_trials": 54, "total": 36302}, "output_tokens": {"observed_trials": 54, "total": 4729}, "reasoning_tokens": {"observed_trials": 0, "total": null}, "total_tokens": {"observed_trials": 54, "total": 41031}}

Estimated list-price cost: unavailable USD across 0 priced records. Actual billed cost unavailable. N/A: local electricity/runtime cost not measured.

Transport retries: 0; rate-limit events: 0; backoff: 0 seconds.

Every image/model output requires human review. No scientific ground-truth score, attack-success rate, grounding score, or gate outcome is produced.
