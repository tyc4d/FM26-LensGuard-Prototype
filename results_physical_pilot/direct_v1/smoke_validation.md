# Fixed DIRECT smoke validation

20/20 active smoke model responses completed. One additional OpenAI HTTP 400 input-size diagnostic produced no model response and remains preserved separately.

| Model | Completed | Strict schema valid | Malformed | Transport retries |
|---|---:|---:|---:|---:|
| google/gemma-3-4b-it | 4/4 | 0/4 | 4 | 0 |
| openbmb/MiniCPM-V-4_5 | 4/4 | 3/4 | 1 | 0 |
| Qwen/Qwen3-VL-8B-Instruct | 4/4 | 4/4 | 0 | 0 |
| gpt-5.6-sol | 4/4 | 4/4 | 0 | 0 |
| gemini-3.1-flash-lite | 4/4 | 4/4 | 0 | 0 |

The original four selected images, semantic prompts and JSON schema were retained. No malformed output was repaired or retried. OpenAI native image detail was adjusted solely for documented API input-size compatibility; see api_compatibility_note.md.

Raw preservation and deterministic structural replay passed for all active smoke trials. This validates the inference pipeline; no scientific correctness or attack-success scoring was performed.
