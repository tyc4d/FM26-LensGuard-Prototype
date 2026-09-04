# LensGuard action registry (Phases 1 and 2)

The machine-readable registry is [`config/action_registry.yaml`](../config/action_registry.yaml). It defines the only three protected action families used in Phases 1 and 2. The registry describes arguments, sources, effects, reversibility, and policy defaults; it never executes an action.

## Source vocabulary

Source trust is action-specific. `explicit_user` can authorize any in-scope action;
`verified_contacts` is trusted for `CALL`, `verified_application_data` for `OPEN_URL`, and
`verified_navigation_data` for `DIRECTION_ADVICE`. `official_signage` is trusted only for
`DIRECTION_ADVICE`, and only when it does not conflict with another verified reference. It is
not a trusted authority for a phone number or URL; a CALL or OPEN_URL proposal attributed to
`official_signage` therefore matches the explicit unverified-target rule. Untrusted visual sources include
`camera_unverified`, `advertisement`,
`handwritten_note`, `unverified_notice`, `unknown_visual_source`, and `unknown`;
`qr_code_unverified` is additionally supported for OPEN_URL.

The primary Phase 1 study runs in **ORACLE PROVENANCE MODE**. Each scenario provides a value-sensitive mapping. The firewall looks up the source of the value the model actually proposed: the official value maps to its trusted source, the selected alternate maps to its scenario source, an expressly authorized alternate maps to `explicit_user`, and an arbitrary unlisted value maps to `unknown_visual_source`.

Phase 2 reuses this registry as a static consequence and authorization table. Automatic arms keep
the model's source estimate separate from the source attached to the matched benchmark region.
Benchmark source-type ground truth is used only for evaluation; the Thin Gate treats the model's
source estimate as descriptive evidence, never as authority by itself. Automatic authorization
requires grounded self-reported evidence plus trusted user-input corroboration or an exact match
to separately supplied verified-value/update metadata. The controlled grounding step
does use benchmark region transcripts/boxes, which must be replaced by a trusted runtime region
interface in future work. A visible word such as “OFFICIAL” never becomes proof of authenticity
by itself.

## Dataset partitions

Dataset v1.2 retains 30 semantic bases—10 per action family—crossed with five conditions in
the 150-record `CORE` partition. These fixed source assignments exercise policy paths; they are
not matched source-authority comparisons. Primary metrics are computed from CORE only.

The separate 16-record `SOURCE_AUTHORITY_MATCHED` partition holds the condition, attack target,
geometry, font, allocated area, and contrast fixed while crossing one selected semantic scenario
per family over `official_signage`, `advertisement`, `handwritten_note`, `unverified_notice`, and
`camera_unverified`. OPEN_URL has one extra `qr_code_unverified` variant. Its source results are
exploratory and reported separately; a five-level crossing on one chosen scenario per family is
not evidence that the same effect generalizes across tasks.

## `CALL`

- **Kind:** machine action
- **Critical argument:** `target_number`
- **Trusted sources:** explicit user authorization or verified contacts
- **Untrusted sources:** unverified camera content, advertisements, handwritten notes, unverified notices, official signage, unknown visual content, or unknown source
- **Possible consequences:** external communication, caller-identity disclosure, social engineering, and call charges
- **Reversibility:** low
- **Default risk:** medium
- **Policy:** allow an explicitly supplied number or verified contact; require confirmation for an untrusted visual number; escalate a conflict with a verified contact according to policy configuration. The prototype never places the call.

## `OPEN_URL`

- **Kind:** machine action
- **Critical argument:** `url`
- **Trusted sources:** explicit user authorization or verified application data
- **Untrusted sources:** unverified camera content, QR codes, advertisements, handwritten notes, unverified notices, official signage, unknown visual content, or unknown source
- **Possible consequences:** network request, external-origin navigation, device-metadata disclosure, and exposure to malicious content
- **Reversibility:** medium
- **Default risk:** medium
- **Policy:** allow an explicitly supplied URL or verified application URL; require confirmation for an untrusted visual URL; warn when the proposed domain conflicts with a verified domain. The prototype never opens the URL.

## `DIRECTION_ADVICE`

- **Kind:** human-impact action
- **Critical arguments:** `direction` and `destination`
- **Trusted sources:** explicit user authorization, verified navigation data, or official signage without a conflict
- **Untrusted sources:** unverified camera content, advertisements, handwritten notes, unverified notices, unknown visual content, or unknown source
- **Possible consequences:** physical movement, navigation error, and safety impact
- **Reversibility:** medium
- **Default risk:** medium
- **Policy:** allow explicit user instructions, verified navigation, and non-conflicting official signage; warn on unverified conflicting directions, advertisements, or handwritten notes. `destination` is guarded independently, so a trusted direction cannot launder an untrusted destination. The output remains advice only and performs no navigation.

## Why consequence prediction is insufficient by itself

The consequences of calling one syntactically valid number can look identical to calling another. Likewise, opening an attacker domain still resembles an ordinary network request, and “turn right” is not intrinsically dangerous without its context. The source of the critical argument and its conflict with verified information are therefore inputs to deterministic policy. Model-predicted consequences remain advisory and are logged for comparison.

When multiple independent non-terminal rules match, policy composition is monotonic and selects
the most restrictive decision (`BLOCK > CONFIRM > WARN > ALLOW`). Explicit-user authorization is
declared as a terminal YAML rule rather than depending on list order, so it preserves the
specifically authorized value. A high-severity consequence prediction may only escalate a
non-terminal result; it cannot override terminal user authorization or relax a provenance
decision. `DIRECTION_ADVICE.destination` is evaluated by its own argument guard independently of
the primary direction rule.

In the Phase 2 main architecture, the effects and reversibility above are read directly from this
registry; no Gemini consequence-prediction call occurs. In the benchmark, trusted reference and
update fixtures simulate separate authenticated channels; they are not derived from source words
in the rendered image. Trusted conflicting updates are allowed only from configured update
channels: `verified_contacts` for CALL,
`verified_application_data` for OPEN_URL, and `verified_navigation_data` for direction advice.
Conflicting `official_signage` still warns because real signage authenticity is not established
from pixels.
