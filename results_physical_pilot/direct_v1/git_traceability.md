# DIRECT physical pilot Git traceability

Branch: `phase3-direct-physical-pilot-v1`. Baseline: `752668b995ab297484d3acc963810d0b54dfa358`. Every commit below is present in the remote tracking branch. No commits were squashed or amended.

Concurrent cleanup commits are identified explicitly. Their per-checkpoint test execution was not observed by the inference session; their final state is covered by the 900-test non-GPU suite, frozen baseline checks, and final response-integrity validation.

| Commit | Subject | Validation evidence | Push |
|---|---|---|---|
| `377910cf364c518e5c8f50505f5d49ec20c748e2` | feat(physical-direct): freeze reviewed 54-image input manifest | 7 focused tests; 700-test baseline; archive/hash checks | PUSHED |
| `a9dce8451d11446befeef3b32a7c71ddf959a9ae` | feat(physical-direct): freeze direct prompts and smoke selection | 100 contract tests | PUSHED |
| `768d59749407d3076ffc9a2efe992c0576a3b2c6` | feat(physical-direct): add immutable five-model inference harness | 864-test non-GPU suite; 62 focused tests | PUSHED |
| `3c1588aaeb7c9584b5a53dfcad11252096f4fbcc` | fix(physical-direct): bound OpenAI native image detail for large originals | 26 compatibility/harness tests | PUSHED |
| `21eb2abd00b7168038881657f0871da5f7e83dad` | test(physical-direct): validate five-model physical smoke inference | 69 focused tests; 20 active smoke records validated | PUSHED |
| `1532311fe32c9841ee86aeec7730d892bc8ce219` | fix(physical-pilot): preserve raw text in review exports | Concurrent cleanup; final 900-test suite and integrity checks | PUSHED |
| `dd22f52a5c8179dd509351422683872e7aaf65f5` | fix(physical-pilot): make local runtime paths configurable | Concurrent cleanup; final 900-test suite and integrity checks | PUSHED |
| `d641ab69421c120492a1ade2a27b6cbd20c15d64` | feat(physical-pilot): validate incomplete artifacts without inference | Concurrent cleanup; final 900-test suite and integrity checks | PUSHED |
| `ee08945d2894561e6ef31268fc45189c36c8c166` | test(physical-pilot): add preservation and partial integrity checks | Concurrent cleanup; final 900-test suite and integrity checks | PUSHED |
| `32a6ff86bbfe29f2e1174ea3f9d23f692294954b` | chore(repo): ignore physical-pilot temporary artifacts | Concurrent cleanup; final 900-test suite and integrity checks | PUSHED |
| `2f5897b99017830d038a29090b634f1780f31526` | docs(physical-pilot): improve reproducibility and contributor guidance | Concurrent cleanup; final 900-test suite and integrity checks | PUSHED |
| `69ba4b8cf97348b307e0d1ad520ce6fb051ee43b` | chore(physical-pilot): preserve existing Gemma direct artifacts | Concurrent cleanup; final 900-test suite and integrity checks | PUSHED |
| `de725ef635c1627cd48438dbdf5cf120c3d8de24` | chore(physical-pilot): preserve existing MiniCPM direct artifacts | Concurrent cleanup; final 900-test suite and integrity checks | PUSHED |
| `35d9d95cf2ba1b50573bfd7ed900ebcc3bfb4dc7` | chore(physical-pilot): index stable experiment artifacts | Concurrent cleanup; final 900-test suite and integrity checks | PUSHED |
| `fe4cb75381fff4cc0c3730a882ac53199a05e579` | test(physical-direct): add local VLM real-image inference | 162 local response identities and raw hashes validated | PUSHED |
| `015d8019d513664ce997d5ec6d66a9d006c3612e` | feat(physical-direct): report raw literals and incomplete model outputs | 46 reporting tests | PUSHED |
| `343f39cb55c4193ff58281e78a5ed4ae002d017e` | test(physical-direct): add GPT-5.6 Sol real-image inference | 54 OpenAI trial records validated (49 complete, 5 token-limited) | PUSHED |
| `8f8a851a60a2ef141c589d23fb317f100ac9d4d7` | test(physical-direct): add Gemini real-image inference | 54 Gemini records; 53 pacing intervals validated | PUSHED |
| `2fe3f72ae0b86dce2dae223843196b6524b119ab` | fix(physical-direct): disclose audited support-file maintenance | 4 validator tests; 290 full/smoke identities; final 900-test suite | PUSHED |

The final `test(physical-direct): add descriptive physical VLM comparison` commit contains this trace, the descriptive comparison, review queue, final validation, and execution notes. Its own hash is intentionally resolved from Git rather than embedded recursively. Its validation is exact report reproduction, 290 unique full/smoke identities, the final 900-test suite, frozen baseline checks, and a zero-match secret scan. Push and clean/upstream verification are performed after the commit.
