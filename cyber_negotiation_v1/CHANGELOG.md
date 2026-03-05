# Changelog

## 2026-03-03

- Added a minimal sign-off gate to the cyber structured assessment: every turn now records `accept` and `block_reason`.
- Changed exact-agreement semantics so `FinalAgreementExact` and `AnyAgreementExact` require both exact unanimity and all-agent sign-off.
- Logged per-turn sign-off state in committee snapshots and added the derived rate `FalseAgreementWithoutSignoffExactRate`.
- Updated the R/C/K and baseline instruction files so each role has an explicit sign-off standard tied to its lens.
- Moved provider-specific Azure and Claude integration back into `cyber_agent.py` so the cyber project matches the old polynomial structure more closely.
- Replaced the separate Azure Responses wrapper with the polynomial-style Azure chat-completions path using JSON schema output.
- Removed the now-unnecessary `cyber_providers.py` module and cleaned `.env.example` to match the active runtime behavior.
- Removed the separate `pydantic` schema module and switched cyber output validation back to manual polynomial-style checks inside `cyber_utils.py`.
- Removed `confidence` from the cyber structured assessment contract and from fallback outputs.
- Removed the cyber mock-response path; the runner now expects real Anthropic or Azure provider-backed models only.
- Removed the local per-call timeout, per-run wallclock, and placeholder token-budget controls from the cyber runner/config path.
- Removed similarity-threshold leakage logic; leakage is now only the explicit forbidden-token marker check.
- Removed fabricated safe fallback assessments after retry exhaustion; invalid turns now abort the run and leave outcome metrics unset.
- Kept the polynomial-style outer JSON envelope, but documented clearly that the cyber `answer` block is not identical to the polynomial `<VALUE>` answer contract.

## 2026-02-27

- Replaced the old broad metrics block with the reduced 6-metric headline set: `FinalCorrectExact`, `FinalCorrectType`, `FinalAgreementExact`, `AnyAgreementExact`, `SeverityBias`, `TrustHygieneRate`.
- Standardized per-turn decision logging so every stored turn state now records per-agent Top-1 label, severity, exact pair, rank-1 citations, committee decisions, and unanimity flags.
- Changed committee tie handling to the explicit string `"NoConsensus"`.
- Split schema-invalid output handling from non-fatal validator checks: JSON/schema failures still retry, while citation/public-message/leakage issues now feed trust-hygiene logging instead of parser rejection.
- Added char-based public-message validation and per-run and single-run per-condition machine-readable outputs for the new headline, derived, and debug metrics.
- Added strict manual validation for the response envelope and structured assessment.
