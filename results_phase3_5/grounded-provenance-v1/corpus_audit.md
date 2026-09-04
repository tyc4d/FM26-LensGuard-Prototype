# Phase 3.5 frozen-corpus compatibility audit

The Phase 2 benchmark lock passed before this audit read or wrote any corpus artifact.

- Benchmark: `lensguard-phase2-frozen-v1`
- Lock SHA-256: `4262f6d6186ac02f49168543a80093130de53ba12764eddd2283502326b12c4f`
- Metadata SHA-256: `3e56d80240152d00ddb961c2745462591d0ef3441ad0b85d116a48bf66cf48ed`
- Image-tree SHA-256: `a7dcd80f480088de5192f564b3892307d51b4509b62ae2e11aded201511aa5f5`
- Records/images: 81/81
- Semantic base scenes: 15
- Annotated region occurrences: 162

## Existing compatible actions

| Action | Records | Critical arguments |
| --- | --- | --- |
| CALL | 27 | target_number |
| DIRECTION_ADVICE | 27 | direction, destination |
| OPEN_URL | 27 | url |

## Existing synthetic conditions

| Current condition | Records |
| --- | --- |
| AUTHORITY_IMPERSONATION | 15 |
| BENIGN_UNTRUSTED_SUBSTITUTION | 15 |
| CLEAN_TRUSTED | 15 |
| EXPLICIT_USER_OVERRIDE | 15 |
| NO_VERIFIED_GROUND_TRUTH | 3 |
| OBVIOUS_INJECTION_CONTROL | 15 |
| TRUSTED_BUT_CONFLICTING_UPDATE | 3 |

## Phase 3.5 measurability

| Phase 3.5 action | Status |
| --- | --- |
| CALL | MEASURABLE IN CURRENT CORPUS |
| OPEN_URL | MEASURABLE IN CURRENT CORPUS |
| DIRECTION_ADVICE | MEASURABLE IN CURRENT CORPUS |
| SAFETY_ADVICE | NOT MEASURABLE IN CURRENT CORPUS |
| RESTAURANT_RESERVATION | NOT MEASURABLE IN CURRENT CORPUS |

| Future physical condition | Status |
| --- | --- |
| C0 | NOT MEASURABLE IN CURRENT CORPUS |
| C1 | NOT MEASURABLE IN CURRENT CORPUS |
| C2 | NOT MEASURABLE IN CURRENT CORPUS |
| C3 | NOT MEASURABLE IN CURRENT CORPUS |
| C4 | NOT MEASURABLE IN CURRENT CORPUS |
| C5 | NOT MEASURABLE IN CURRENT CORPUS |
| C6 | NOT MEASURABLE IN CURRENT CORPUS |

The current condition labels are synthetic semantic/security conditions, not the future C0-C6 physical capture conditions.

## Region identity finding

The 162 region occurrences use 51 distinct legacy IDs; 33 legacy IDs occur in more than one image frame. Frame-scoped IDs are therefore required.

The JSON keeps the complete frozen metadata payload under `phase2_corpus`. Frame IDs and evidence IDs appear only under `compatibility_derived` and in the CSV column explicitly suffixed `_derived`.

## Annotation limits

Legacy `source_type` and `content_claimed_authority` labels are preserved. No `physical_source`, `semantic_role`, `control_class`, perception confidence, or grounding confidence is inferred. SAFETY and RESTAURANT_RESERVATION are not represented by experimental cases in this corpus.
