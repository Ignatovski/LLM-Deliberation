# Cyber Negotiation Framework V1

This subproject keeps the polynomial-style build-up and now uses the reduced evaluation design from the updated pentest-finding spec.

## Canonical Entry Point

- runner: `cyber_negotiation_v1/main_cyber_json.py`
- game folder: `cyber_negotiation_v1/games_descriptions/cyber_game/`

## Layout

```text
cyber_negotiation_v1/
  main_cyber_json.py
  cyber_agent.py
  cyber_save_utils.py
  cyber_utils.py
  docs/
  games_descriptions/
    cyber_game/
      config*.txt
      global_instructions.txt
      initial_deal.txt
      individual_instructions/cooperative/
      conditions/
      label_sets/
      scenarios/
      ground_truth/
      output/
```

## JSON Contract

```json
{
  "scratchpad": "<SCRATCHPAD>...</SCRATCHPAD>",
  "answer": "<ANSWER>public message</ANSWER>\n<ASSESSMENT>{...}</ASSESSMENT>",
  "plan": "<PLAN>...</PLAN>"
}
```

This matches the polynomial runner at the outer envelope level: the top-level keys are `scratchpad`, `answer`, and `plan`, and the runner still requires one JSON object per turn. The cyber runner differs inside `answer`: instead of a polynomial `<VALUE>...</VALUE>`, it carries a hidden `<ASSESSMENT>{...}</ASSESSMENT>` block.

## Headline Metrics

The stable headline table is now:

- `FinalCorrectExact`
- `FinalCorrectType`
- `FinalAgreementExact`
- `AnyAgreementExact`
- `SeverityBias`
- `TrustHygieneRate`

Everything else is emitted as derived or debug JSON.

## Quickstart

From `cyber_negotiation_v1/`:

```bash
python main_cyber_json.py \
  --game_dir ./games_descriptions/cyber_game \
  --condition_file conditions/C3.txt \
  --scenario_file placeholder_webapp_001.json \
  --azure \
  --exp_name run1
```

Azure is wired in the same style as the polynomial runner:
- provider-specific Azure and Claude calling logic lives in `cyber_agent.py`
- Azure uses the `AzureOpenAI` chat-completions path with JSON schema output
- the same Azure env keys are reused: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`

## Output Files

Each run writes into `games_descriptions/cyber_game/output/<exp_name>/`:

- `history*.json`
- `metrics_*.json`
- `condition_summary_*.json`
- `condition_headline_*.csv`
- `public_history_*.json`
- `expert_review_*.csv`

## Condition Auto-Config

If a condition file includes `config_file=...`, the runner will use that config automatically when you leave `--config_file` at its default.

Examples:

- `conditions/C1.txt` -> single GPT-5 baseline
- `conditions/C2.txt` -> single Claude baseline
- `conditions/C3.txt` -> 3-agent GPT-5 negotiation
- `conditions/C4.txt` -> 3-agent Claude negotiation
- `conditions/C5.txt` -> mixed-model negotiation
- `conditions/C6.txt` -> GPT-5 negotiation with LLM prior wording
- `conditions/C7.txt` -> Claude negotiation with human prior wording

## Docs

- `docs/metrics.md`
- `docs/assumptions.md`
- `docs/open_questions.md`
- `docs/schemas.md`
- `docs/protocol.md`
- `docs/expert_review.md`
