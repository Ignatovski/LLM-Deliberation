import runpy, sys
from pathlib import Path

root = Path(r"D:\Drive\OneDrive\git\LLM-Deliberation")
script = root / "leakage_fix" / "eval_leakage_invalid_llm.py"

histories = sorted(
    str(p) for p in
    (root / "games_descriptions" / "polynomial_game_adversarial" / "output" / "obstructive").rglob("history*.json")
)

sys.argv = [
    str(script),
    "--azure",
    "--model", "gpt-5",
    "--env-file", str(root / "cyber_negotiation_v1" / ".env"),
    "--overwrite",
    "--out", str(root / "summarys" / "leakage" / "eval_adversarial_obstructive.json"),
    "--history",
    *histories,
]

runpy.run_path(str(script), run_name="__main__")
