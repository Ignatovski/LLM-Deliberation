# Cyber Negotiation Framework V1 (Scaffold-First)

This is a new cyber-only subproject scaffolded inside the existing repository.

Adaptation note:
- The existing repository is focused on polynomial negotiation experiments.
- To avoid breaking that codebase, this V1 framework is implemented as a self-contained subproject under `cyber_negotiation_v1/`.
- The Python package is namespaced as `cyberneg` (instead of top-level `core`, `io`, etc.) to avoid import collisions (notably Python stdlib `io`).

## V1 Scope

Implemented scaffold-first components:
- strict JSON schemas and validation (pydantic v2)
- mock/offline provider + mock runnable baseline and 3-agent negotiation
- scheduler with fairness/no-repeat constraints
- private/public logging separation
- committee aggregation + metrics + CSV exports
- basic visualizations (machine-readable metrics + PNGs)
- Streamlit local dashboard (lightweight)
- config-driven conditions/roles/prompts/scenarios/label sets
- provider adapter scaffolds for Azure OpenAI (Responses API) and Anthropic

## Quickstart (Mock Mode)

From `cyber_negotiation_v1/`:

```bash
python -m cyberneg.cli.main validate-config --config examples/configs/mock_experiment.yaml
python -m cyberneg.cli.main dry-run --config examples/configs/mock_experiment.yaml
python -m cyberneg.cli.main run-scenario --config examples/configs/mock_experiment.yaml
python -m cyberneg.cli.main run-baseline --config examples/configs/mock_experiment.yaml
python -m cyberneg.cli.main compute-metrics --run-dir outputs/latest_mock_run
python -m cyberneg.cli.main export-expert-csv --run-dir outputs/latest_mock_run
```

Or install editable:

```bash
pip install -e .
cyberneg validate-config --config examples/configs/mock_experiment.yaml
```

## CLI Commands (V1)

- `validate-config`
- `dry-run`
- `run-scenario`
- `run-baseline`
- `run-condition`
- `run-experiment`
- `compute-metrics`
- `export-expert-csv`
- `launch-dashboard`
- `scaffold-scenario`
- `scaffold-condition`

## Important Docs

- `docs/open_questions.md` (all unresolved ambiguities + deviations vs polynomial JSON runner)
- `docs/assumptions.md` (conservative defaults used in V1)
- `docs/protocol.md`
- `docs/schemas.md`
- `docs/metrics.md`
- `docs/expert_review.md`

