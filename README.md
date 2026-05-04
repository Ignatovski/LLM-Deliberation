# LLM Deliberation

Repository accompanying my master's thesis on evaluating multi-agent LLM systems in negotiation-style settings.

The project contains two main experimental tracks:

- a polynomial negotiation game used to study deliberation, coordination, adversarial behavior, and agreement dynamics
- a cybersecurity committee framework in which multiple LLM agents deliberate over vulnerability findings, severity, and final agreement

This repository is maintained as a thesis artifact repository. It keeps the code, configurations, saved outputs, summaries, and thesis figures needed to document and reproduce the reported analyses.

## Main Entry Points

- `main_polynomial.py`
  Main runner for the polynomial negotiation experiments.
- `cyber_negotiation_v1/main_cyber_json.py`
  Main runner for the cybersecurity committee experiments.

## Top-Level Structure

```text
LLM-Deliberation/
  main_polynomial.py
  main_polynomial_json.py
  main_polynomial_xyz.py
  polynomial/
    outputs/
    archive/
    reference_images/
  polynomial_utils.py
  visualize_polynomial.py
  games_descriptions/
  summarys/
  viewer/plots/thesis/
  scripts/
  cyber_negotiation_v1/
    main_cyber_json.py
    cyber_agent.py
    cyber_utils.py
    docs/
    tests/
    games_descriptions/cyber_game/
```

## What Is Kept Here

- experiment code used in the thesis
- prompt, game, and scenario assets
- saved run outputs needed for traceability
- summary JSON files used in analysis
- scripts that regenerate thesis metrics and figures

Local editor state, temporary helper scripts, setup residue, and unrelated development clutter have been removed.

## Key Directories

- `polynomial/`
  Consolidated home for polynomial raw outputs, archived backups, and reference images.
- `games_descriptions/`
  Polynomial negotiation game definitions, prompts, and configurations.
- `cyber_negotiation_v1/games_descriptions/cyber_game/`
  Cybersecurity scenarios, committee conditions, labels, ground truth, and saved outputs.
- `logs/`, `summarys/`
  Saved summaries and supporting artifacts retained as thesis evidence.
- `viewer/plots/thesis/`
  Thesis-ready figures exported from the analysis pipeline.
- `scripts/`
  Retained analysis and figure-generation scripts relevant to thesis reproduction.

## Reproducing Core Thesis Artifacts

The retained analysis pipeline is centered on:

- `scripts/build_metrics_summary.py`
- `scripts/build_dynamics_summary.py`
- `scripts/regenerate_thesis_root_plots.py`
- `scripts/thesis_style_trust_risks.py`

Typical workflow:

```powershell
py -3 scripts/rebuild_polynomial_summaries.py
py -3 scripts/regenerate_thesis_root_plots.py
```

`scripts/rebuild_polynomial_summaries.py` regenerates the baseline metrics summary, the dynamics summary, the history manifest, and the adversarial polynomial summaries from the consolidated `polynomial/outputs/` tree.

Some cyber-specific and leakage-specific analyses use additional scripts in `scripts/` and documentation in `cyber_negotiation_v1/docs/`.

## Notes

- API credentials are only needed if the experiments are rerun against external model providers.
- Historical outputs are intentionally retained because they are part of the thesis evidence base.
- Polynomial run artifacts have been consolidated under `polynomial/outputs/` to keep thesis evidence in one place.

## License

The repository keeps the existing `LICENSE` file. Reuse should respect both the repository license and the licensing terms of any external model providers used in the experiments.
