# Fixed cloud smoke validation

Run ID: `phase3_6_cloud_smoke_v1`. The three cases were fixed in harness commit `118e534` before inference.

| Provider | Resolved model | Completed | Malformed | Transport retries | 429 events |
|---|---|---:|---:|---:|---:|
| openai | gpt-5.6-sol | 6/6 | 0 | 0 | 0 |
| gemini | gemini-3.1-flash-lite | 6/6 | 0 | 0 | 0 |

Authentication, model availability, image input, structured output, evidence-ID parsing, usage collection, raw preservation and deterministic gate evaluation passed for both providers. Gemini used sequential configured pacing. No API compatibility fix, prompt repair, model fallback or repeated semantic attempt was needed.

Full runs may proceed. These smoke observations are diagnostic and will not be pooled into the 81-image benchmark results. Physical effectiveness remains NOT MEASURABLE.
