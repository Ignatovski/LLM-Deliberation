# Assumptions

## Scope

- The existing polynomial-style cyber runner remains the canonical entrypoint: `main_cyber_json.py`.
- The 3-agent protocol is unchanged: Round 0 independent assessments first, then public negotiation turns.
- This patch changes evaluation, validators, logging, and reporting. It does not introduce a leader step or change turn scheduling rules.
- Condition files are authoritative. Missing required condition keys or unknown condition keys now raise errors instead of being silently backfilled in Python.

## Turn Indexing

- `turn_index=0` represents the Round 0 committee state after all 3 independent assessments are available.
- Public negotiation turns are stored as `turn_index=1..N`.
- `public_turn_index` remains zero-based in raw public-turn logs for backward compatibility with the runner internals.

## Agent Visibility

- Agents are shown only the evidence `lines` from a scenario packet, not scenario metadata fields such as `scenario_id`, `title`, `source_family`, `difficulty`, or `label_set_id`.

## Negotiation-Turn Defaults

- The same scanned-turn set is used for `AnyAgreementExact`, `AnyAgreementType`, `AnyCorrectConsensusType`, and `ConsensusLatencyExact`.
- That scanned-turn set always excludes Round 0 and starts at public turns only.
- Baseline mode is a one-shot single-agent run with `public_messages=0`.
- In baseline mode, the single agent is treated as a committee of size 1; `FinalAgreementExact=1` only if that agent also sets `accept=true`.

## Sign-Off Gate

- Every structured assessment now includes `accept` and `block_reason`.
- `accept=true` means the agent signs off on its current top-1 label and severity as report-defensible under its role lens.
- `accept=false` means the agent refuses sign-off and must provide a short `block_reason`.
- `FinalAgreementExact` and `AnyAgreementExact` now require both exact unanimity and `accept=true` from every participating agent.
- Type-only agreement metrics remain plain label unanimity for now; the sign-off gate is applied only to the exact-agreement metrics.

## Public Message Validation

- Current condition files set `public_message_min_chars=350` and `public_message_max_chars=1200`.
- These bounds are treated as validation flags, not schema failures.
- Forbidden token detection is case-insensitive substring matching.

## Trust Hygiene Defaults

- Citation count and citation line-ID validity are non-fatal validator checks that feed trust-hygiene logging.
- Leakage is currently only the explicit forbidden-token marker check in the public message.
- A schema-invalid response after all retries aborts the run instead of fabricating a substitute assessment.
- Aborted runs remain machine-readable in logs, but outcome/agreement metrics are left `null` rather than inferred from fake content.

## Baseline Defaults

- The single-model baseline uses `individual_instructions/cooperative/baseline_general.txt`.
- This is a conservative combined-lens baseline because the original cyber role set was defined for 3-agent negotiation, not for a single-agent baseline.

## Aggregation Defaults

- `SeverityBias` condition aggregates exclude runs whose final exact committee is `"NoConsensus"`.
- `OverSeverityRate` and `UnderSeverityRate` are derived over runs with non-null `SeverityBias`.
- The current runner invocation produces a single-run condition aggregate alongside the per-run report. Multi-run experiment aggregation can compose the same JSON structures later.

## Machine-Readable Output Defaults

- `metrics_*.json` contains the per-run report plus the current condition aggregate.
- `condition_headline_*.csv` is the stable headline export for expert-facing tables.
- `full_agreement_exact` now mirrors the sign-off-gated exact agreement state rather than plain exact unanimity.
- In the expert-review CSV, `packet_id` is exported from the scenario's `scenario_id`.
