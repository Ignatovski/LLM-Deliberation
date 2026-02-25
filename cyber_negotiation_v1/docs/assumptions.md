# Assumptions (Conservative Defaults Used in V1)

This file documents assumptions made where the user request was intentionally open-ended or ambiguous.

## Repository Adaptation

- The existing repository already contains a separate polynomial negotiation project.
- Conservative adaptation: implement this framework as a self-contained subproject under `cyber_negotiation_v1/`.
- Package namespace is `cyberneg` to avoid stdlib/package collisions (notably `io`).

## Runtime Defaults

- Default mock provider is used unless a provider/model is selected in config.
- Default public negotiation message count in example configs: `6` (must be divisible by 3).
- Default JSON retry max retries: `3`.
- Default per-call timeout: `90s` (overridable via config/env).
- Default final-turn announcement window: only the final public turn is explicitly marked.
- Default execution mode: sequential.
- Optional parallel execution exists at experiment level and is disabled by default.

## JSON Validation Defaults

- Strict JSON means `json.loads` only (no regex/text fallback parsing).
- Invalid JSON/schema outputs are retried with a validation-error correction prompt.
- All failed attempts and validation errors are logged in machine-readable JSON.
- Public message length target (soft validation): `80-150` words.
- Public message hard cap (schema/business validation): `220` words in example configs.

## Label Set / Taxonomy Defaults

- Default label set is `default_findings_v1` in `configs/label_sets/default_findings_v1.yaml`.
- Included labels (default): `BruteForce`, `CommandInjection`, `CSRF`, `FileInclusion`, `FileUpload`, `InsecureCaptcha`, `SQLInjection`, `BlindSQLInjection`, `WeakSessionIDs`, `XSS_DOM`, `XSS_Reflected`, `XSS_Stored`, `CSPBypass`, `BrokenAccessControl`, `AuthBypass`, `ExposedSecret`, `NoFinding`, `Other`
- Alias normalization is supported through config and logged when applied.

## Committee / Tie Behavior Defaults (V1)

- Final agreement metrics (`FinalAgreementType`, `FinalAgreementExact`) require all 3 agents to agree.
- Committee type-only output is computed by majority vote on Top-1 label.
- Committee exact output is computed by majority vote on `(Top-1 label, severity)` pair.
- If no majority exists (possible 3-way split), committee output is `null` and status `no_majority`.

## Metric Definition Defaults (Open to Revision)

- `FlipCountType`: count of changes in committee majority Top-1 label across public turns (including transitions to/from no-majority).
- `SeverityVarianceAcrossRounds`: population variance of committee majority Top-1 severity ordinal across public turns where a majority severity exists.
- `ConsensusLatencyType/Exact`: first public turn index where full agreement is reached and remains unchanged through final public turn.

## Priors

- Priors are instruction-only and applied in Round 0 only by default.
- Rounds 1+ do not re-inject role-specific instructions unless condition config explicitly enables it.

## Provider Adapter Status

- Mock provider is fully runnable in V1 Phase 1.
- Azure Responses API and Anthropic adapters are scaffolded with config/env support and strict-JSON pipeline integration.
- Exact request payload details may require environment-specific adjustment after first live run.

