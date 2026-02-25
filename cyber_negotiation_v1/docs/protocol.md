# V1 Protocol (Cyber Negotiation)

## Agents

- Exactly 3 agents in V1: `R`, `C`, `K`
- No leader
- No leader summary step
- Agents communicate directly in public negotiation turns

## Visibility Model

Agents can see:
- full evidence packet (from Round 0 onward)
- all previous public messages
- their own previous private notes
- their own previous private plan

Agents cannot see:
- other agents’ private notes
- other agents’ private plans

## Phases

### Phase A: Round 0 Independent Assessment (separate phase)

- All three agents independently analyze the same full evidence packet.
- Each produces strict JSON output.
- Round 0 is logged and evaluated, but not counted as a public negotiation round.

### Phase B: Public Negotiation (Rounds 1..N in reporting)

- `N` is the configured number of public messages for the run.
- Public speaking order is randomized with constraints:
  - no same agent twice in a row
  - equal number of public messages per agent by end of run
- Final public turn is explicitly announced to the active agent.

## Strict JSON Contract

- Provider responses must be valid JSON and schema-valid.
- Invalid outputs trigger bounded retries with validation-error feedback.
- No text fallback parser.
- All failed attempts are logged.

