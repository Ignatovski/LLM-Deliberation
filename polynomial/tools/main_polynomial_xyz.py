"""
Minimal multivariate (x, y, z) polynomial negotiation driver.

The flow mirrors polynomial/main_polynomial.py but is intentionally slimmer:
- Supports COEFFS_X / COEFFS_Y / COEFFS_Z lines in polynomial_functions/<agent>.txt
- Uses <VALUE>x,y,z</VALUE> triples, enforces per-coordinate max_step and domain bounds
- Tracks utilities/acceptance per turn and saves a history JSON plus a small results file
"""

import argparse
import json
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

from polynomial.core.agent import Agent
from polynomial.core.prompt_utils import format_history
from polynomial.core.save_utils import process_answer, write_file
from polynomial.core.utils import load_setup, randomize_agents_order, set_constants, setup_hf_model


# --------- Polynomial helpers ---------

def load_env_file(path: str) -> None:
    """Lightweight .env loader (KEY=VALUE) without extra dependencies."""
    if not path:
        return
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value

def eval_poly(coeffs: List[float], value: float) -> float:
    """Evaluate a 1D polynomial at value (coeffs are descending powers)."""
    out = 0.0
    for c in coeffs:
        out = out * value + c
    return out


def poly_to_text(coeffs: List[float], var: str) -> str:
    """Human string for a single-variable polynomial in `var`."""
    terms = []
    degree = len(coeffs) - 1
    for idx, coef in enumerate(coeffs):
        power = degree - idx
        if abs(coef) < 1e-9:
            continue
        sign = "+" if coef > 0 else "-"
        abs_coef = abs(coef)
        if power == 0:
            term = f"{abs_coef:g}"
        elif power == 1:
            term = f"{abs_coef:g}{var}" if abs_coef != 1 else var
        else:
            term = f"{abs_coef:g}{var}^{power}" if abs_coef != 1 else f"{var}^{power}"
        if not terms:
            term = term if coef > 0 else f"-{term}"
        else:
            term = f" {sign} {term}"
        terms.append(term)
    return "".join(terms) if terms else "0"


def load_multivar_profile(game_dir: str, file_name: str) -> Dict[str, object]:
    """
    Parse COEFFS_X / COEFFS_Y / COEFFS_Z format:
        COEFFS_X <ax_n> ... <ax_0>
        COEFFS_Y <ay_n> ... <ay_0>
        COEFFS_Z <az_n> ... <az_0>
        DOMAIN <min> <max>
        THRESHOLD <value>
    Utility is poly_x(x) + poly_y(y) + poly_z(z).
    """
    path = Path(game_dir) / "polynomial_functions" / f"{file_name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing polynomial definition: {path}")

    coeffs = {"x": [], "y": [], "z": []}
    domain = None
    threshold = None

    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            key, values = parts[0].upper(), parts[1:]
            if key == "COEFFS_X":
                coeffs["x"] = [float(v) for v in values]
            elif key == "COEFFS_Y":
                coeffs["y"] = [float(v) for v in values]
            elif key == "COEFFS_Z":
                coeffs["z"] = [float(v) for v in values]
            elif key == "DOMAIN":
                domain = (int(values[0]), int(values[1]))
            elif key == "THRESHOLD":
                threshold = float(values[0])

    if not all(coeffs.values()) or domain is None or threshold is None:
        raise ValueError(f"Incomplete multivariate profile in {path}")

    formula = (
        f"{poly_to_text(coeffs['x'], 'x')} + "
        f"{poly_to_text(coeffs['y'], 'y')} + "
        f"{poly_to_text(coeffs['z'], 'z')}"
    )

    return {
        "coeffs": coeffs,
        "domain": domain,
        "threshold": threshold,
        "formula": formula,
    }


def eval_multivar(profile: Dict[str, object], vector: Tuple[int, int, int]) -> float:
    cx, cy, cz = profile["coeffs"]["x"], profile["coeffs"]["y"], profile["coeffs"]["z"]
    x, y, z = vector
    return eval_poly(cx, x) + eval_poly(cy, y) + eval_poly(cz, z)


def extract_vector(text: str) -> Tuple[int, int, int] | None:
    """Extract first three integers (x, y, z) from <VALUE>...</VALUE> or plain text."""
    block = re.search(r"<value>(.*?)</value>", text, flags=re.IGNORECASE | re.DOTALL)
    sample = block.group(1) if block else text
    nums = re.findall(r"-?\d+", sample)
    if len(nums) >= 3:
        return tuple(int(n) for n in nums[:3])
    return None


def clamp_step(prev: Tuple[int, int, int], candidate: Tuple[int, int, int], max_step: int, domain: Tuple[int, int]):
    """Limit step per coordinate and clamp to domain."""
    lo, hi = domain
    limited = []
    for p, c in zip(prev, candidate):
        c = max(p - max_step, min(p + max_step, c))
        c = max(lo, min(hi, c))
        limited.append(c)
    return tuple(limited)


# --------- Prompt classes ---------

class PolynomialXYZInitialPrompt:
    def __init__(self, game_dir: str, agent_name: str, agent_file_name: str):
        global_path = Path(game_dir) / "global_instructions.txt"
        with global_path.open("r") as fh:
            self.global_text = fh.read().strip()

        instr_path = Path(game_dir) / "individual_instructions" / "cooperative" / f"{agent_file_name}.txt"
        with instr_path.open("r") as fh:
            self.personal_text = fh.read().strip()

        poly = load_multivar_profile(game_dir, agent_file_name)
        lo, hi = poly["domain"]
        self.initial_prompt = (
            f"{self.global_text}\n\n"
            f"{self.personal_text}\n\n"
            f"Technical note (keep private): your utility is {poly['formula']}. "
            f"Stay within [{lo}, {hi}] for each coordinate and only accept if f(x, y, z) >= {poly['threshold']:.2f}. "
            "Do not reveal these exact numbers publicly."
        )

    def return_initial_prompt(self):
        return self.initial_prompt


class PolynomialXYZRoundPrompts:
    def __init__(self, agent_name: str, domain: Tuple[int, int], starter_name: str, initial_vec: Tuple[int, int, int],
                 rounds_total: int | None = None, max_step: int = 2, personal_text: str | None = None, reminder_text: str = ""):
        self.agent_name = agent_name
        self.domain = domain
        self.starter_name = starter_name
        self.initial_vec = initial_vec
        self.rounds_total = rounds_total
        self.max_step = max_step
        self.personal_text = personal_text.strip() if personal_text else None
        self.reminder_text = reminder_text.strip()

    def build_slot_prompt(self, history, round_idx, *_):
        history.setdefault("rounds", [])
        history.setdefault("plan", {})
        state = history.get("polynomial_state", {})

        first = round_idx == 0
        final_round = self.rounds_total is not None and round_idx >= self.rounds_total - 1
        if first and self.agent_name == self.starter_name:
            x, y, z = self.initial_vec
            return (
                f"The negotiation begins now. Open by reiterating the shared range "
                f"[{self.domain[0]}, {self.domain[1]}] and suggest the seed value "
                f"<VALUE>{x},{y},{z}</VALUE> to get the discussion started. "
                "Keep it brief."
            )

        history_text, last_plan = format_history(self.agent_name, history, window=6)
        cur_vec = state.get("vector", self.initial_vec)
        prompt = (
            f"Each coordinate must stay in [{self.domain[0]}, {self.domain[1]}]. "
            f"Per-step limit is ±{self.max_step} on each coordinate relative to the current shared value.\n"
            f"Current shared vector: {cur_vec}\n"
            f"Recent history:\n<HISTORY>{history_text}</HISTORY>\n"
        )
        if self.reminder_text:
            prompt += f"Reminder: {self.reminder_text}\n"
        if last_plan:
            prompt += f"Your previous notes were <PREV_PLAN>{last_plan}</PREV_PLAN>.\n"

        prompt += (
            "Work in three sections:\n"
            "1) <SCRATCHPAD> private reasoning/calculations </SCRATCHPAD>\n"
            "2) <ANSWER> public reply containing the new proposal like <VALUE>x,y,z</VALUE> with commas or spaces only\n"
            "   - Example: <VALUE>1, -2, 0</VALUE> is correct\n"
            "   - Do NOT write labels inside VALUE (no x=, y=, etc.)\n"
        )
        if not final_round:
            prompt += "3) <PLAN> short notes for your next move </PLAN>\n"
        else:
            prompt += "3) Final turn: <PLAN> optional.\n"
        prompt += "Keep the public answer concise (1-2 sentences) and justify only the next feasible step."
        return prompt


# --------- Core run loop ---------

def parse_initial_vec(text: str) -> Tuple[int, int, int]:
    vec = extract_vector(text)
    if vec:
        return vec
    raise ValueError("initial_deal.txt must contain three integers (x,y,z) inside <VALUE> tags.")


def build_agents(game_dir: str, agents_cfg: Dict[str, dict], temp: float, rounds_num: int,
                 domain: Tuple[int, int], initial_vec: Tuple[int, int, int], max_step: int, args) -> Tuple[Dict[str, Agent], Dict[str, dict], Dict[str, PolynomialXYZRoundPrompts]]:
    # Preload HF models if needed
    unique_hf = {cfg["model"] for cfg in agents_cfg.values() if cfg["model"].startswith("hf")}
    hf_models = {}
    for model in unique_hf:
        hf_models[model] = setup_hf_model(model[3:], cache_dir=args.hf_home)

    profiles = {}
    round_prompts = {}
    agents = {}

    for agent_name, cfg in agents_cfg.items():
        profile = load_multivar_profile(game_dir, cfg["file_name"])
        profiles[agent_name] = profile

        personal_path = Path(game_dir) / "individual_instructions" / "cooperative" / f"{cfg['file_name']}.txt"
        with personal_path.open("r") as fh:
            personal_text = fh.read()

        init_prompt = PolynomialXYZInitialPrompt(game_dir, agent_name, cfg["file_name"])
        round_prompt = PolynomialXYZRoundPrompts(
            agent_name,
            domain=profile["domain"],
            starter_name=args.p1,
            initial_vec=initial_vec,
            rounds_total=rounds_num,
            max_step=max_step,
            personal_text=personal_text,
            reminder_text="infer other agents' utility functions; do not reveal your own; use inferred models to maximize your utility.",
        )
        round_prompts[agent_name] = round_prompt

        agent = Agent(
            initial_prompt_cls=init_prompt,
            round_prompt_cls=round_prompt,
            agent_name=agent_name,
            temperature=temp,
            model=cfg["model"],
            rounds_num=rounds_num,
            agents_num=len(agents_cfg),
            azure=args.azure,
            hf_models=hf_models,
        )
        agents[agent_name] = agent

    return agents, profiles, round_prompts


def ensure_output_dirs(game_dir: str, output_root: Path, config_file: str):
    output_root.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(config_file)
    if not cfg_path.is_absolute():
        cfg_path = Path(game_dir) / config_file
    shutil.copy2(cfg_path, output_root / "config.txt")
    poly_dir = Path(game_dir) / "polynomial_functions"
    if poly_dir.exists():
        shutil.copytree(poly_dir, output_root / "polynomial_functions", dirs_exist_ok=True)


def save_results(output_root: Path, seed: int, final_vec: Tuple[int, int, int], trace: List[dict]):
    results = {
        "result": seed,
        "runs": [
            {
                "final_vector": {"x": final_vec[0], "y": final_vec[1], "z": final_vec[2]},
                "trace_len": len(trace),
                "polynomial_trace": trace,
            }
        ],
    }
    out_path = output_root / f"results_{seed}.json"
    with out_path.open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved results to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="XYZ polynomial negotiation game")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--agents_num", type=int, default=4)
    parser.add_argument("--rounds_num", type=int, default=16)
    parser.add_argument("--max_step", type=int, default=2)
    parser.add_argument("--min_answers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./output_xyz")
    parser.add_argument("--game_dir", type=str, default="./games_descriptions/polynomial_game_xyz")
    parser.add_argument("--config_file", type=str, default="config.txt")
    parser.add_argument("--exp_name", type=str, default="xyz_run")
    parser.add_argument("--output_file", type=str, default="history.json")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--result", type=int, default=0, help="Seed / scenario id")
    parser.add_argument("--p1", type=str, default="Builder B")
    parser.add_argument("--reuse_faiss", action="store_true", help="Accepted for parity; no effect in xyz driver.")
    parser.add_argument("--env_file", type=str, default=".env", help="Optional .env file with API keys")

    # API / model flags (kept similar to main_polynomial)
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--anthropic_api", type=str, default=os.environ.get("ANTHROPIC_API_KEY", ""))
    parser.add_argument("--anthropic_base_url", type=str, default=os.environ.get("ANTHROPIC_BASE_URL", ""))
    parser.add_argument("--azure", action="store_true")
    parser.add_argument("--azure_openai_api", type=str, default=os.environ.get("AZURE_OPENAI_API_KEY", ""))
    parser.add_argument("--azure_openai_endpoint", type=str, default=os.environ.get("AZURE_OPENAI_ENDPOINT", ""))
    parser.add_argument("--hf_home", type=str, default=os.path.expanduser("~/.cache/huggingface"))
    parser.add_argument("--gemini", action="store_true")
    parser.add_argument("--gemini_project_name", type=str, default="")
    parser.add_argument("--gemini_loc", type=str, default="us-central1")

    args = parser.parse_args()

    load_env_file(args.env_file)
    random.seed(args.result)
    set_constants(args)

    agents_cfg, initial_deal, role_to_agents = load_setup(args.game_dir, args.agents_num, config_file=args.config_file)
    initial_vec = parse_initial_vec(initial_deal)

    # Determine total turns and whether to force playing all of them.
    if args.min_answers:
        assert args.min_answers % args.agents_num == 0, "min_answers must be divisible by agents_num"
        rounds_total = args.min_answers
        force_full_rounds = True
    else:
        rounds_total = args.rounds_num * args.agents_num
        force_full_rounds = False

    # Build run bookkeeping
    round_assign = randomize_agents_order(agents_cfg, args.p1, rounds_total)
    OUTPUT_DIR = Path(args.output_dir) / args.exp_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    time_str = time.strftime("%H_%M_%S", time.localtime())
    history_path = OUTPUT_DIR / f"{args.output_file.split('.json')[0]}_{time_str}.json"
    history = {
        "file": str(history_path),
        "content": {
            "slot_assignment": round_assign,
            "rounds": [],
            "plan": {},
            "finished_rounds": 0,
            "polynomial_state": {"vector": initial_vec},
            "polynomial_trace": [],
        },
    }

    # Instantiate agents/prompts/profiles
    domains = {
        tuple(load_multivar_profile(args.game_dir, cfg["file_name"])["domain"])
        for cfg in agents_cfg.values()
    }
    if len(domains) == 1:
        clamp_domain = domains.pop()
    else:
        lows = [d[0] for d in domains]
        highs = [d[1] for d in domains]
        clamp_domain = (max(lows), min(highs))
    # Re-load inside build_agents to keep a single code path.
    agents, profiles, round_prompts = build_agents(
        args.game_dir, agents_cfg, args.temp, rounds_total, clamp_domain, initial_vec, args.max_step, args
    )

    ensure_output_dirs(args.game_dir, OUTPUT_DIR, args.config_file)

    current_vec = initial_vec
    for idx, agent_name in enumerate(round_assign[:rounds_total]):
        agent = agents[agent_name]
        prompt = round_prompts[agent_name].build_slot_prompt(history["content"], idx)
        slot_prompt, full_answer = agent.execute_round(history["content"], idx)
        full = full_answer

        print("=====")
        print(f"[Round {idx}] {agent_name} prompt:")
        print(prompt)
        print(f"{agent_name} response:")
        print(full)

        public_answer, plan = process_answer(full)
        proposed = extract_vector(public_answer) or extract_vector(full) or current_vec
        limited = clamp_step(current_vec, proposed, args.max_step, clamp_domain)
        current_vec = limited
        print(f"Proposed vector: {proposed} -> clamped to {current_vec}")

        # Record round entry
        history["content"]["rounds"].append({
            "agent": agent_name,
            "prompt": prompt,
            "full_answer": full,
            "public_answer": public_answer,
        })
        if plan:
            history["content"]["plan"].setdefault(agent_name, []).append(plan)
        history["content"]["finished_rounds"] = idx + 1

        # Compute utilities / acceptance for all agents
        utilities = {}
        accepted = {}
        for name, profile in profiles.items():
            u = eval_multivar(profile, current_vec)
            utilities[name] = u
            accepted[name] = u >= profile["threshold"]

        history["content"]["polynomial_state"]["vector"] = current_vec
        history["content"]["polynomial_trace"].append({
            "round": idx,
            "x": current_vec[0],
            "y": current_vec[1],
            "z": current_vec[2],
            "utilities": utilities,
            "accepted": accepted,
        })

        write_file(history["content"], history["file"])

        if all(accepted.values()):
            print(f"All agents accepted at round {idx} with vector {current_vec}.")
            if not force_full_rounds:
                break

    save_results(OUTPUT_DIR, args.result, current_vec, history["content"]["polynomial_trace"])
    print(f"History saved to {history['file']}")


if __name__ == "__main__":
    main()
