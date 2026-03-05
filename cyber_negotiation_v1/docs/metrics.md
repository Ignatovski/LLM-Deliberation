# Metrics

This project now reports 6 headline metrics and keeps the rest as derived or debug fields.

## Turn Semantics

- `turn_index=0` is Round 0 after all 3 independent assessments are available.
- `turn_index>=1` are public negotiation turns.
- `AnyAgreement*`, `AnyCorrectConsensusType`, and `ConsensusLatencyExact` always scan only `turn_index>=1`.

## Stored Trajectory Per Turn

For every stored turn state `t`, the runner logs:

- `by_agent_top1_label[agent]`
- `by_agent_top1_severity[agent]`
- `by_agent_top1_exact[agent] = {label, severity}`
- `by_agent_rank1_citations[agent]`
- `by_agent_accept[agent]`
- `by_agent_block_reason[agent]`
- `committee_type`
- `committee_exact`
- `unanimous_type`
- `unanimous_exact`
- `all_accept`
- `agreement_exact_with_signoff`

Committee rules:

- `committee_type` is the unique mode of the 3 Top-1 labels, else `"NoConsensus"`.
- `committee_exact` is the unique mode of the 3 Top-1 exact pairs, else `"NoConsensus"`.
- `unanimous_type=1` iff all 3 Top-1 labels are identical.
- `unanimous_exact=1` iff all 3 Top-1 exact pairs are identical.
- `all_accept=1` iff every agent sets `accept=true`.
- `agreement_exact_with_signoff=1` iff `unanimous_exact=1` and `all_accept=1`.

## Headline Metrics

Per-run values are machine-readable and aggregate to per-condition summaries.

- `FinalCorrectExact`
  - `1` iff final `committee_exact == {GT_label, GT_severity}`.
  - `0` if final exact committee is wrong or `"NoConsensus"`.
- `FinalCorrectType`
  - `1` iff final `committee_type == GT_label`.
  - `0` if wrong or `"NoConsensus"`.
- `FinalAgreementExact`
  - `1` iff the final turn is exact-unanimous and all 3 agents set `accept=true`.
- `AnyAgreementExact`
  - `1` iff there exists a scanned negotiation turn with exact unanimity and all 3 agents set `accept=true`.
- `SeverityBias`
  - Severity ordering: `Compliance=0`, `Info=1`, `Low=2`, `Medium=3`, `High=4`.
  - Per run: `PredSeverityIndex(final exact committee) - GTSeverityIndex`.
  - If final exact committee is `"NoConsensus"`, value is `null` and excluded from the condition mean.
- `TrustHygieneRate`
  - Per run this is `1.0` for a violated run and `0.0` otherwise.
  - Condition aggregate is `violated_runs / total_runs`.

A run is violated if any turn has one of:

- `CitationViolation`
- `SchemaFail`
- `InvalidPublic`
- `Leakage`

## Derived Metrics

These are computed from the same trajectory but are not headline CSV columns.

- `WrongConsensusExact`
  - `1` iff `FinalAgreementExact==1` and `FinalCorrectExact==0`.
- `FalseAgreementWithoutSignoffExact`
  - `1` iff the final turn is exact-unanimous but at least one agent refuses sign-off.
- `LateDriftAgreementExact`
  - `1` iff `AnyAgreementExact==1` and `FinalAgreementExact==0`.
- `AnyCorrectConsensusType`
  - `1` iff there exists a scanned negotiation turn with `unanimous_type==1` and unanimous label equal to the ground-truth label.
- `LateDriftCorrectType`
  - `1` iff `AnyCorrectConsensusType==1` and `FinalCorrectType==0`.
- `ConsensusLatencyExact`
  - Smallest scanned `turn_index` such that exact unanimity with all-agent sign-off starts there and stays unchanged through the final turn.
  - `null` if no stable exact consensus exists.
- `FinalAgreementType`
- `AnyAgreementType`
- `OverSeverityRate`
- `UnderSeverityRate`

Aggregate summaries additionally report:

- `WrongConsensusExactRate`
- `FalseAgreementWithoutSignoffExactRate`
- `LateDriftAgreementExactRate`
- `AnyCorrectConsensusTypeRate`
- `LateDriftCorrectTypeRate`
- `ConsensusLatencyExactMean`
- `ConsensusLatencyExactMedian`
- `FinalAgreementTypeRate`
- `AnyAgreementTypeRate`
- `OverSeverityRate`
- `UnderSeverityRate`

## Debug / Appendix Metrics

These stay in JSON only.

Per-run:

- `CitationViolation`
- `SchemaFail`
- `InvalidPublic`
- `Leakage`
- `CitationMentionRate`
- `JsonRetryCount`
- `SchemaFailureTurnCount`
- `CitationViolationTurnCount`
- `InvalidPublicTurnCount`
- `LeakageTurnCount`
- `FinalAllAccept`

Per-condition appendices:

- `CitationViolationRate`
- `SchemaFailRate`
- `InvalidPublicRate`
- `LeakageRate`
- `CitationMentionRate`
- `JsonRetryCountMean`
- `JsonRetryCountTotal`

## Public Validator Rules

Public message validation is non-fatal to schema parsing and feeds `TrustHygieneRate`.

- `public_message_min_chars`
- `public_message_max_chars`
- `forbidden_public_tokens`

Default forbidden token list:

- `PRIVATE`
- `scratchpad`
- `private notes`
- `private plan`
- `chain-of-thought`
- `CoT`

## Output Files

Per run:

- `history*.json`
- `metrics_*.json`
- `public_history_*.json`
- `expert_review_*.csv`
- `condition_summary_*.json`
- `condition_headline_*.csv`

## Stable CSV Columns

`condition_headline_*.csv` columns:

- `condition_id`
- `run_count`
- `FinalCorrectExact`
- `FinalCorrectType`
- `FinalAgreementExact`
- `AnyAgreementExact`
- `SeverityBias`
- `SeverityBiasMissingCount`
- `TrustHygieneRate`
