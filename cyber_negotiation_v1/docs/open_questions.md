# Open Questions

These items remain intentionally documented instead of guessed.

## Evaluation Defaults Needing User Confirmation

1. Public message character bounds
- Current conservative default: `350..1200` characters.
- This replaces the earlier word-based targeting because the updated spec explicitly asks for char-based validation.

2. Leakage policy strength
- Current implementation only flags obvious forbidden-token leakage in the public message.
- If you want stronger leakage detection later, it should be specified explicitly rather than inferred.

3. Sign-off gate scope
- Current implementation applies the new `accept/block_reason` gate to exact-agreement metrics only.
- Type-only agreement metrics still use plain label unanimity.
- If you want sign-off to also gate type-only agreement metrics later, that should be specified explicitly rather than inferred.
4. Condition-level aggregation workflow
- The patched runner emits per-run outputs plus a single-run condition aggregate.
- If you want a built-in folder-level reducer over many `history*.json` files, that should be added explicitly rather than inferred.

5. Single-agent baseline instruction policy
- The table requires C1/C2 single-model baselines, but the original cyber role instructions were defined for the 3-agent R/C/K negotiation setup.
- Current implementation uses a new combined-lens baseline instruction file.

6. C5 total outputs ambiguity
- Your table shows about `13` outputs for C5, but the scheduler enforces equal public-turn counts across 3 agents, so total public messages must be divisible by 3.
- Current runnable default for C5 is `15`.

## Documented Differences From The Earlier Cyber V1 Patch

1. Headline metrics reduced to the new 6-metric table
- Older broad metrics such as `FlipCountType`, `LateDriftType`, `JsonFailureRate`, and `MessageLengthViolations` are no longer headline outputs.
- Where still useful, they have been removed or demoted to derived/debug reporting.

2. Citation and public-message checks are now non-fatal validators
- Earlier code rejected some citation/message issues during parsing.
- Updated behavior matches the new spec: only JSON/schema-invalid outputs trigger retries.

3. Committee ties now use the explicit string `"NoConsensus"`
- Earlier code used `null` plus status fields.
- Updated behavior follows the new evaluation spec exactly.

4. Structured validation remains stricter than the original polynomial runner
- The cyber project still enforces a structured assessment block for evaluation, but validation is now manual inside `cyber_utils.py` instead of using `pydantic`.
- The outer JSON contract and polynomial-style file layout remain unchanged.

5. Mock-mode scaffolding and local runtime guardrails were removed
- The cyber project no longer carries the earlier mock-response path or the local timeout/wallclock placeholder controls.

6. Invalid-output policy after retry exhaustion
- Current implementation now aborts the run and records the failure instead of emitting a fallback assessment.
- If you later want a different policy, it should be specified explicitly because fabricated substitute outputs are intentionally excluded.

7. Azure API path choice
- Current implementation now uses the same Azure chat-completions integration style as the polynomial runner instead of a separate Responses-API adapter layer.
- This matches the user's existing working Azure setup more closely, but it is a deliberate deviation from the earlier cyber scaffold that used a standalone provider wrapper.
