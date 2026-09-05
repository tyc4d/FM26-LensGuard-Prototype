# OpenAI physical image input compatibility

The first OpenAI smoke request used the frozen cloud adapter's omitted image-detail setting with the unchanged original `IMG_3483.jpeg`. On 2026-09-05 at 06:31 UTC, the Responses API rejected the request with HTTP 400 before producing model output: 47,628 required patches exceeded the 30,000-patch rejection limit. The request had one transport attempt. Its raw error, normalized failed record, start marker, plan and hashes remain immutable in `smoke/physical_direct_smoke_v1/` under OpenAI.

The official [OpenAI Images and vision documentation](https://developers.openai.com/api/docs/guides/images-vision), checked on 2026-09-05, specifies that GPT-5.6 Sol supports `detail: high`, which bounds native image processing to 2,048 pixels and 2,500 patches. Original detail does not shrink an oversized image to satisfy the separate rejection limit.

The physical-only adapter now sets `detail: high` on **every OpenAI image**, including all four fixed smoke cases and all 54 full cases. It records this native configuration in plans and responses. The original JPEG bytes, model ID, prompt, shared JSON schema, generation settings and empty tools list remain unchanged. Historical cloud code/results are untouched. Provider-native resampling remains a comparison limitation.

The four OpenAI smoke requests using this compatible setting are preserved separately in `smoke/physical_direct_smoke_v1_openai_high_detail/`. The completed 12 local smoke responses are retained; none is repeated or repaired. Gemini's four fixed smoke requests use the original smoke namespace. Active smoke coverage is 20 model responses when complete, plus **one separately reported API rejection diagnostic with no model response**. This is an API compatibility repair, not a semantic retry or model fallback.

The local smoke returned malformed responses under the strict JSON-only contract (Gemma 4/4 and MiniCPM 1/4). Those responses remain malformed and do not motivate any prompt/schema/decoding change. Full runs will use the same frozen semantic contracts.
