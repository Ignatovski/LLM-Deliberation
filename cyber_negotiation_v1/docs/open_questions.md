# Open Questions (Ambiguities + Deviations from Polynomial JSON Runner)

This document records unresolved items and any intentional implementation differences from the prior `main_polynomial_json.py` approach.

## Required User Clarifications (Not Blocking V1 Scaffold)

1. Exact C1-C7 condition table definitions
- Requirement says support the exact user-defined condition matrix via config only.
- The exact C1-C7 content was not provided.
- V1 includes `configs/conditions/c1_c7_template.yaml` placeholders and a runnable `mock_demo.yaml`.

2. Final committee tie behavior policy
- For 3 agents, majority usually exists, but 3-way splits are possible.
- V1 default: committee output becomes `null` with `no_majority`.
- Confirm if you want a forced tie-break rule.

3. Exact `FlipCountType` definition
- V1 default: count label transitions in committee majority Top-1 label trajectory across public turns (including transitions to/from no-majority).
- Confirm if flips should ignore no-majority states.

4. Exact `SeverityVarianceAcrossRounds` definition
- V1 default: variance of committee majority Top-1 severity ordinal across public turns where majority exists.
- Alternative interpretations exist (agent-level variance, exact-majority-only, etc.).

5. Final-turn announcement wording
- V1 uses a conservative generic announcement string in prompt rendering.
- Confirm desired wording/policy (only final turn vs final N turns).

6. Azure OpenAI Responses API payload details in your environment
- SDK/API combinations differ across versions and Azure deployments.
- V1 adapter is scaffolded and integrated, but exact `responses.create(...)` payload may need a config-compatible tweak after first live test.

7. Anthropic structured-output mode preference
- V1 uses prompt-contract + strict local JSON validation/retry.
- Confirm whether you want tool-use or a provider-specific structured mode when available.

8. Siemens expert review column set finalization
- V1 exports all required fields plus optional placeholders.
- Confirm if additional columns (reviewer ID/date/etc.) are needed.

## Intentional Differences vs `main_polynomial_json.py` (Documented Per Request)

1. Strict JSON parser policy (no fallback parsing)
- Polynomial JSON runner has robustness logic tailored to mixed outputs and `<VALUE>` extraction.
- V1 uses strict JSON only (`json.loads`), schema validation, bounded retries, and logs every failed attempt/error.

2. Config-driven experiment conditions (no embedded condition logic)
- Polynomial code embeds game mechanics and some fixed assumptions in runner code.
- V1 moves conditions, priors, label sets, roles, prompts, and scenarios into config files.

3. Provider abstraction layer
- Polynomial project uses a shared `Agent` wrapper with mixed provider handling.
- V1 separates providers (`base`, `azure_responses`, `anthropic`, `mock_provider`) from orchestration and validation for auditability and mock testing.

4. Cyber-specific schemas and committee metrics
- Polynomial outputs center on numeric `x`, utilities, thresholds, and acceptance.
- V1 outputs center on finding labels, severities, citations, committee snapshots, and trust/drift/calibration metrics.

5. Private/public separation as first-class log model
- Polynomial histories store prompt/full/public answer and plans.
- V1 explicitly models/logs `private_notes`, `private_plan`, `public_message`, and structured assessment with visibility separation.

6. Round semantics
- Polynomial runs use turn loops and a final review step.
- V1 treats Round 0 as a separate independent phase and counts “rounds” as public negotiation answers only.

7. No temperature setting in V1
- Polynomial code supports temperature.
- V1 omits temperature from provider calls/config defaults (provider defaults only).

8. Mock-first runnable mode
- Polynomial framework is API-centric.
- V1 must run offline with no keys; mock provider is a primary tested path.

## Deferred to Phase 2 (Non-Blocking for V1 Scaffold)

- Azure/Anthropic token-usage capture coverage parity across all paths
- cost accounting (placeholder in V1)
- dashboard polish and Siemens overview templates beyond machine-readable exports + basic charts

