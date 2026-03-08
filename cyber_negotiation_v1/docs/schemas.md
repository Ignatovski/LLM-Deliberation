# Schemas

## Response Envelope

Top-level response contract stays aligned with the polynomial JSON runner:

```json
{
  "scratchpad": "<SCRATCHPAD>...</SCRATCHPAD>",
  "answer": "<ANSWER>public message</ANSWER>\n<ASSESSMENT>{...}</ASSESSMENT>",
  "plan": "<PLAN>...</PLAN>"
}
```

Validation:

- top-level object only
- keys `scratchpad`, `answer`, `plan` required
- strict JSON parsing only
- no text fallback parser

## Structured Assessment

`<ASSESSMENT>{...}</ASSESSMENT>` is validated with manual checks in `cyber_utils.py` (`_validate_structured_assessment` + label normalization).

```json
{
  "ranked_findings": [
    {
      "rank": 1,
      "label": "XSS_Reflected",
      "severity": "Medium",
      "citations": ["L003", "L004"],
      "rationale": "..."
    },
    {
      "rank": 2,
      "label": "XSS_Stored",
      "severity": "Low",
      "citations": [],
      "rationale": "..."
    },
    {
      "rank": 3,
      "label": "NoFinding",
      "severity": "Info",
      "citations": [],
      "rationale": "..."
    }
  ],
  "decision_summary": "...",
  "accept": true,
  "block_reason": null
}
```

Schema rules:

- required keys are exactly: `ranked_findings`, `decision_summary`, `accept`, `block_reason`
- exactly 3 ranked findings
- ranks must be `1,2,3`
- severity required for every ranked finding
- severity must be one of `Compliance`, `Info`, `Low`, `Medium`, `High`
- label must validate against the configured label set after alias normalization
- each finding may include only `rank`, `label`, `severity`, `citations`, `rationale`
- each finding must include a `citations` list
- rank-1 citations must contain 1-2 items
- `decision_summary` must be a non-empty string
- `accept` must be boolean
- if `accept=false`, `block_reason` must be a non-empty string
- if `accept=true`, `block_reason` must be null or empty

Validator-only rules:

- rank-1 citation IDs must exist in the evidence packet
- public message must satisfy configured length and forbidden-token rules
- leakage currently means explicit forbidden-token leakage markers in the public message
- validator-only issues are tracked as trust-hygiene flags and do not trigger schema retries by themselves

## Trajectory Snapshot

Every stored turn state contains:

```json
{
  "turn_index": 1,
  "phase": "public",
  "public_turn_index": 0,
  "speaker": "agent_a",
  "by_agent_top1_label": {"agent_a": "XSS_Reflected", "agent_b": "XSS_Reflected", "agent_c": "NoFinding"},
  "by_agent_top1_severity": {"agent_a": "Medium", "agent_b": "Medium", "agent_c": "Info"},
  "by_agent_top1_exact": {"agent_a": {"label": "XSS_Reflected", "severity": "Medium"}},
  "by_agent_rank1_citations": {"agent_a": ["L003", "L004"]},
  "committee_type": "XSS_Reflected",
  "committee_exact": {"label": "XSS_Reflected", "severity": "Medium"},
  "unanimous_type": false,
  "unanimous_exact": false
}
```

Tie rule:

- If no unique mode exists, `committee_type` or `committee_exact` is `"NoConsensus"`.
