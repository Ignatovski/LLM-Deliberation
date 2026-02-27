# Cyber Negotiation Framework V1

This subproject now follows the old polynomial project more closely.

Primary workflow:
- runner-first entrypoint: `cyber_negotiation_v1/main_cyber_json.py`
- polynomial-style game layout: `cyber_negotiation_v1/games_descriptions/cyber_game/`
- simple text config files: `config.txt`, `config_claude.txt`, `config_azure.txt`, `config_mix_all_diff.txt`
- shared instruction file: `global_instructions.txt`
- role-specific instruction files: `individual_instructions/cooperative/*.txt`
- simple condition files: `conditions/*.txt`
- scenario packets and ground truth: `scenarios/*.json`, `ground_truth/*.json`

The earlier `src/cyberneg/` package scaffold is still present, but it is now support infrastructure rather than the main interface.

## Canonical Layout

```text
cyber_negotiation_v1/
  main_cyber_json.py
  cyber_agent.py
  cyber_utils.py
  cyber_save_utils.py
  games_descriptions/
    cyber_game/
      config.txt
      config_claude.txt
      config_azure.txt
      config_mix_all_diff.txt
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

The outer JSON contract now mirrors `main_polynomial_json.py`:

```json
{
  "scratchpad": "<SCRATCHPAD>...</SCRATCHPAD>",
  "answer": "<ANSWER>public message</ANSWER>\n<ASSESSMENT>{...}</ASSESSMENT>",
  "plan": "<PLAN>...</PLAN>"
}
```

Notes:
- `scratchpad` remains private.
- `plan` remains private.
- `answer` contains the public message plus a hidden structured assessment block.
- public history is built from `<ANSWER>...</ANSWER>` only.
- structured assessment is parsed strictly from `<ASSESSMENT>{...}</ASSESSMENT>`.

## Quickstart (Mock Mode)

From `cyber_negotiation_v1/`:

```bash
python main_cyber_json.py \
  --game_dir ./games_descriptions/cyber_game \
  --config_file config.txt \
  --condition_file conditions/C1.txt \
  --scenario_file placeholder_webapp_001.json \
  --exp_name mock_run
```

This writes a polynomial-style history file plus companion metrics/export files into:
- `cyber_negotiation_v1/games_descriptions/cyber_game/output/mock_run/`

## Provider Modes

- `mock-cyber` in `config*.txt`: offline runnable, no keys required
- `claude-*` in `config*.txt`: Anthropic adapter
- `gpt-*` with `--azure`: Azure Responses adapter

## Important Docs

- `docs/open_questions.md`
- `docs/assumptions.md`
- `docs/protocol.md`
- `docs/schemas.md`
- `docs/metrics.md`
- `docs/expert_review.md`
