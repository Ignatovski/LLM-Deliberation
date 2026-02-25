# Metrics (V1)

V1 metrics cover four layers:

1. Outcome quality
2. Negotiation trust quality
3. Calibration quality
4. Auditability / validation quality

## Outcome Quality

- `FinalCorrectType`
- `FinalCorrectSeverity`
- `FinalCorrectExact`
- `OverSeverityRate`
- `UnderSeverityRate`

## Negotiation Trust Quality

- `FinalAgreementType`
- `FinalAgreementExact`
- `WrongConsensusType`
- `WrongConsensusExact`
- `NoConsensus`
- `LateDriftType`
- `LateDriftExact`
- `FlipCountType`
- `ConsensusLatencyType`
- `ConsensusLatencyExact`

## Calibration Quality

- `SeverityVarianceAcrossRounds`
- `ExactSeverityDisagreementRateAtFinal`

## Validation / Auditability

- `JsonRetryCount`
- `JsonFailureRate`
- `SchemaValidationFailureRate`
- `CitationValidityRate`
- `CitationCountViolations`
- `MessageLengthViolations`

