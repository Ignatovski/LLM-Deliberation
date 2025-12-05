# Polynomial negotiation driver.
# ------------------------------
# This file keeps the scaffolding of main.py (CLI options, logging, agent orchestration)
# but swaps the discrete-option deal for a single shared integer x. Each agent maximizes
# its own polynomial utility f_i(x) and accepts once that value crosses a threshold.


# Reuses the registration/CLI pattern from the original main.py but drives a single-variable game.
import argparse
import json
import os
import random
import re
import shutil
import time
import uuid
from pathlib import Path
from faiss_utility import AnswerComparator
from typing import Dict, List, Tuple

from agent import Agent
from prompt_utils import format_history
from save_utils import create_outfiles, save_conversation, write_file
from utils import load_setup, randomize_agents_order, set_constants, setup_hf_model
from evaluation.plot_similarity import (
    load_embeddings as load_sim_embeddings,
    compute_agent_matrices,
    plot_matrices,
)
from visualize_polynomial import load_trace as load_poly_trace, load_thresholds
import matplotlib.pyplot as plt

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


def generate_similarity_image(embeddings_path: Path, save_path: Path):
    """Create a combined similarity heatmap PNG from the FAISS embeddings/metadata."""
    embeddings, metadata = load_sim_embeddings(embeddings_path)
    agent_mats = compute_agent_matrices(embeddings, metadata)
    if not agent_mats:
        print("No agents found for similarity plot; skipping heatmap.")
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_matrices(agent_mats, save_path)


def generate_polynomial_image(history_path: Path, save_path: Path):
    """Create a polynomial negotiation visualization (x over time, utilities, acceptance strip)."""
    trace, _ = load_poly_trace(history_path)
    thresholds = load_thresholds(history_path.parent)

    steps = list(range(len(trace)))
    labels = [str(entry.get("round", idx)) for idx, entry in enumerate(trace)]
    x_values = [entry.get("x", 0) for entry in trace]

    utility_series = {}
    for entry in trace:
        utilities = entry.get("utilities", {})
        for agent, value in utilities.items():
            utility_series.setdefault(agent, []).append(value)
        missing_agents = set(utility_series.keys()) - set(utilities.keys())
        for agent in missing_agents:
            utility_series[agent].append(float("nan"))

    agent_names = sorted(utility_series.keys())

    fig, (ax_x, ax_u, ax_state) = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [2, 3, 1]}
    )

    ax_x.plot(steps, x_values, marker="o", color="black")
    ax_x.set_ylabel("x")
    ax_x.set_title("Negotiated value of x over time")
    ax_x.grid(True, linestyle="--", alpha=0.4)

    for agent, values in utility_series.items():
        ax_u.plot(steps, values, marker="o", label=agent)
        if agent in thresholds:
            ax_u.axhline(
                thresholds[agent],
                linestyle="--",
                color=ax_u.lines[-1].get_color(),
                alpha=0.5,
                label=f"{agent} threshold",
            )

    ax_u.set_ylabel("Utility f_i(x)")
    ax_u.set_xlabel("Negotiation step")
    ax_u.set_title("Agent utilities vs. thresholds")
    ax_u.grid(True, linestyle="--", alpha=0.4)
    ax_u.legend(loc="best")
    ax_u.set_xticks(steps)
    ax_u.set_xticklabels(labels, rotation=45, ha="right")

    directions = []
    prev_x = None
    for value in x_values:
        if prev_x is None:
            directions.append("•")
        else:
            if value > prev_x:
                directions.append("↑")
            elif value < prev_x:
                directions.append("↓")
            else:
                directions.append("→")
        prev_x = value

    ax_state.set_title("Move direction & acceptance per round")
    ax_state.set_yticks(range(len(agent_names)))
    ax_state.set_yticklabels(agent_names)
    ax_state.set_xlabel("Negotiation step")
    ax_state.set_xlim(-0.5, len(steps) - 0.5)
    ax_state.set_ylim(-0.5, len(agent_names) + 0.5)
    ax_state.grid(True, linestyle=":", alpha=0.3, axis="x")

    for idx, arrow in enumerate(directions):
        ax_state.text(idx, len(agent_names) + 0.1, arrow, ha="center", va="bottom", fontsize=12)

    for agent_idx, agent in enumerate(agent_names):
        for step_idx, entry in enumerate(trace):
            accepted = entry.get("accepted", {}).get(agent, False)
            color = "#2ca02c" if accepted else "#d62728"
            ax_state.scatter(step_idx, agent_idx, color=color, s=60, marker="s")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved polynomial visualization to {save_path}")


def ensure_text(response) -> str:
    """
    TODO: do we really need it
    Improvement of the original agent.py
    Convert various AI model API response formats into a plain string.
    Handles responses from different SDKs (OpenAI, Azure, Gemini, HF, etc.) that might return:
    - Plain strings
    - Objects with a .content attribute (like OpenAI's ChatCompletionMessage)
    - Lists of message parts (like Gemini's chunked responses)
    - Other objects that can be stringified with str()
    """
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
    return str(response)



def polynomial_to_text(coeffs: List[float]) -> str:
    """
    Convert a list of polynomial coefficients into a human-readable string representation.
    The coefficients are expected in descending order of powers.
    Args:
        coeffs: List of coefficients where coeffs[0] is for the highest degree term.
                Example: [3, 0, -2, 1] represents 3x³ - 2x + 1
    Returns:
        A formatted string representation of the polynomial.
        Examples:
            [1, -2, 1]   → "x² - 2x + 1"
            [2, 0, -1]   → "2x² - 1"
            [1, 0, 0, 1] → "x³ + 1"
            [0, 1]       → "x"
            [0, 0, 0]    → "0"
    """
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
            if abs_coef == 1:
                term = "x"
            else:
                term = f"{abs_coef:g}x"
        else:
            if abs_coef == 1:
                term = f"x^{power}"
            else:
                term = f"{abs_coef:g}x^{power}"
        if not terms:
            term = term if coef > 0 else f"-{term}"
        else:
            term = f" {sign} {term}"
        terms.append(term)
    return "".join(terms) if terms else "0"


def evaluate_polynomial(coeffs: List[float], x_value: int) -> float:
    """
    Evaluate a polynomial at a given point using Horner's method.

    Args:
        coeffs: List of polynomial coefficients in descending order of degree.
                Example: [2, 3, 1] for 2x² + 3x + 1
        x_value: The point at which to evaluate the polynomial.

    Returns:
        The value of the polynomial at x = x_value.
        Example: evaluate_polynomial([2, 3, 1], 2) returns 15.0 (2*2² + 3*2 + 1 = 15)
    """
    result = 0.0
    for coef in coeffs:
        result = result * x_value + coef
    return result


def extract_value(text: str) -> int | None:
    """
    Extract an integer value from text, optionally wrapped in <VALUE> tags.

    This function handles multiple formats:
    1. Text containing <VALUE>42</VALUE> (case-insensitive)
    2. Text containing <VALUE> x = 42 </VALUE> (case-insensitive, with optional whitespace)
    3. Plain text containing a number

    Args:
        text: Input text potentially containing a numeric value

    Returns:
        The first integer found in the text, or None if no integer is found.
        The search is case-insensitive and handles negative numbers.

    Example:
        extract_value("The value is <VALUE>42</VALUE>")  # Returns: 42
        extract_value("<VALUE> x = -5 </VALUE>")         # Returns: -5
        extract_value("Proposing x = 3")                 # Returns: 3
        extract_value("No number here")                  # Returns: None
    """
    # First try to extract content between <VALUE> tags
    block = re.search(r"<value>(.*?)</value>", text, flags=re.IGNORECASE | re.DOTALL)
    if block:
        sample = block.group(1).strip()
        # Try to match "x = 42" pattern first
        x_equal_match = re.search(r'x\s*=\s*(-?\d+)', sample, re.IGNORECASE)
        if x_equal_match:
            return int(x_equal_match.group(1))
        # If no "x = " pattern, look for a standalone number
        match = re.search(r"-?\d+", sample)
        if match:
            return int(match.group(0))
    
    # If no <VALUE> tags or no number found within them, search the entire text
    match = re.search(r"-?\d+", text)
    return int(match.group(0)) if match else None


def parse_initial_x(initial_line: str) -> int:
    """
    Parse the initial value of x from the first line of initial_deal.txt by using extract_value function.
    """
    value = extract_value(initial_line)
    if value is None:
        match = re.search(r"-?\d+", initial_line)
        if match:
            return int(match.group(0))
        raise ValueError("initial_deal.txt must contain an integer x.")
    return value



def load_polynomial_profile(game_dir: str, file_name: str) -> Dict[str, object]:
    """
    Load a polynomial profile from a configuration file.

    The configuration file should be in the format:
        COEFFS <a_n> <a_n-1> ... <a_0>  # Polynomial coefficients in descending order
        DOMAIN <min> <max>              # Valid range for x
        THRESHOLD <value>               # Minimum acceptable value for f(x)

    Args:
        game_dir: Directory containing the polynomial_functions folder
        file_name: Name of the polynomial profile file (without .txt extension)

    Returns:
        A dictionary containing:
        - 'coeffs': List[float] - Polynomial coefficients [a_n, a_n-1, ..., a_0]
        - 'domain': Tuple[int, int] - (min_x, max_x) range
        - 'threshold': float - Minimum acceptable value
        - 'formula': str - Human-readable polynomial string

    Raises:
        FileNotFoundError: If the polynomial definition file doesn't exist
        ValueError: If the file is malformed or incomplete

    Example:
        # For a file containing:
        # COEFFS 1 -2 1
        # DOMAIN -10 10
        # THRESHOLD 0.5
        profile = load_polynomial_profile("game_data", "parabola")
        # Returns {
        #     'coeffs': [1.0, -2.0, 1.0],
        #     'domain': (-10, 10),
        #     'threshold': 0.5,
        #     'formula': 'x² - 2x + 1'
        # }
    """
    path = os.path.join(game_dir, "polynomial_functions", f"{file_name}.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing polynomial definition: {path}")
    coeffs: List[float] = []
    domain: Tuple[int, int] | None = None
    threshold: float | None = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            key = parts[0].upper()
            values = parts[1:]
            if key == "COEFFS":
                coeffs = [float(v) for v in values]
            elif key == "DOMAIN":
                if len(values) != 2:
                    raise ValueError(f"DOMAIN line must have two numbers in {path}")
                domain = (int(values[0]), int(values[1]))
            elif key == "THRESHOLD":
                threshold = float(values[0])
    if not coeffs or domain is None or threshold is None:
        raise ValueError(f"Incomplete polynomial profile in {path}")
    return {
        "coeffs": coeffs,
        "domain": domain,
        "threshold": threshold,
        "formula": polynomial_to_text(coeffs),
    }



class PolynomialInitialPrompt:
    """
    A class to generate the initial prompt for polynomial negotiation agents.

    Combines global instructions, individual role instructions, and private polynomial
    information into a single prompt for the agent.

    The prompt consists of three main parts:
    1. Global instructions (shared across all agents)
    2. Personal instructions (specific to this agent's role)
    3. Private technical note (polynomial details, not to be shared)

    Args:
        game_dir: Directory containing the game configuration files
        agent_name: Display name of the agent (e.g., "Analyst")
        agent_file_name: Base filename for the agent's configuration (e.g., "analyst_a")

    Attributes:
        global_text (str): Text from global_instructions.txt
        personal_text (str): Text from the agent's individual instructions file
        initial_prompt (str): The complete formatted prompt for the agent

    Example:
        # Typical usage in the polynomial negotiation game
        from main_polynomial import PolynomialInitialPrompt
        
        # Initialize a prompt for the Analyst role
        prompt = PolynomialInitialPrompt(
            "games_descriptions/polynomial_game",  # Path to game directory
            "Analyst",                             # Agent's display name
            "analyst_a"                            # Matches files: analyst_a.txt in polynomial_functions/ and individual_instructions/cooperative/
        )
        
        # Get the complete prompt
        print(prompt.return_initial_prompt())
    """
    def __init__(self, game_dir, agent_name, agent_file_name):
        global_text_path = os.path.join(game_dir, "global_instructions.txt")
        with open(global_text_path, "r") as f:
            self.global_text = f.read().strip()

        instruction_path = os.path.join(
            game_dir, "individual_instructions", "cooperative", f"{agent_file_name}.txt"
        )
        with open(instruction_path, "r") as f:
            self.personal_text = f.read().strip()

        poly_profile = load_polynomial_profile(game_dir, agent_file_name)
        domain_low, domain_high = poly_profile["domain"]
        formula = poly_profile["formula"]
        threshold = poly_profile["threshold"]
        self.initial_prompt = (
            f"{self.global_text}\n\n"
            f"{self.personal_text}\n\n"
            f"Technical note (keep private): your polynomial is {formula}. "
            f"Stay within [{domain_low}, {domain_high}] and only accept if f(x) >= {threshold:g}. "
            "Do not reveal these exact numbers publicly."
        )

    def return_initial_prompt(self):
        return self.initial_prompt



class PolynomialRoundPrompts:
    """
    Slimmed-down equivalent of RoundPrompts for the polynomial game.
    Responsibilities:
        - Provide windowed conversation history.
        - Remind agents to work inside <SCRATCHPAD> before sharing <ANSWER>.
        - Enforce the <VALUE> tag (replacing <DEAL>).
        - Keep private planning in <PLAN>, just like the original implementation.
    """

    def __init__(self, agent_name, domain_low, domain_high, starter_name, initial_x, rounds_total=None, reminder_text: str = ""):
        self.agent_name = agent_name
        self.domain_low = domain_low
        self.domain_high = domain_high
        self.starter_name = starter_name
        self.initial_x = initial_x
        self.rounds_total = rounds_total
        self.reminder_text = reminder_text

    def build_slot_prompt(self, history, round_idx, *_):
        # Ensure history has expected keys to avoid KeyError on fresh runs
        history.setdefault("rounds", [])
        history.setdefault("plan", {})
        state = history.get("polynomial_state", {})

        first = round_idx == 0
        final_round = self.rounds_total is not None and round_idx >= self.rounds_total - 1
        if first and self.agent_name == self.starter_name:
            return (
                f"The negotiation begins now. Open by reiterating the shared range "
                f"[{self.domain_low}, {self.domain_high}] and suggest the seed value "
                f"<VALUE>{self.initial_x}</VALUE> to get the discussion started. "
                "Keep it brief."
            )

        history_text, last_plan = format_history(self.agent_name, history, window=6)
        prompt = (
            f"The shared integer x must stay in [{self.domain_low}, {self.domain_high}]. "
            "Review the latest discussion:\n"
            f"<HISTORY>{history_text}</HISTORY>\n"
        )
        if self.reminder_text:
            prompt += f"Reminder: {self.reminder_text.strip()}\n"
        if last_plan:
            prompt += f"Your previous notes were <PREV_PLAN>{last_plan}</PREV_PLAN>.\n"

        last_x = history["rounds"][-1]["public_answer"] if history["rounds"] else ""
        current_x_hint = f"Current shared x (after limits) is {state.get('x', self.initial_x)}."
        prompt += (
            "Work in three sections:\n"
            f"1. Use <SCRATCHPAD>...</SCRATCHPAD> for private reasoning/calculations. Begin by noting the current x: {current_x_hint}\n"
            "2. In <ANSWER>...</ANSWER>, provide your public response with a numeric proposal like: <VALUE>42</VALUE>\n"
            "   - Only include the number between the VALUE tags, no other text or symbols\n"
            "   - Example: <VALUE>42</VALUE> is correct\n"
            "   - Example: <VALUE>x = 42</VALUE> is incorrect\n"
            "   - IMPORTANT: keep your proposal within the current <VALUE> plus or minus the configured step size "
            "(`--max_step`, default 2)\n"
        )
        if not final_round:
            prompt += (
                "3. After <ANSWER>, log your follow-up options for the next turn inside <PLAN>...</PLAN> "
                "(keep it short and tactical).\n"
            )
        else:
            prompt += (
                "3. Since this is the final turn, you may omit <PLAN> unless you want to note follow-up ideas.\n"
            )

        prompt += (
            "Keep your public answer to two concise sentences and justify only the next feasible step.\n"
        )
        return prompt


def main():
    # CLI mirrors main.py: temp, rounds, model flags, restart knobs.
    # Differences: default to 4 agents, include --max_step (step size for x),
    # and set the default game_dir to the new polynomial description.
    parser = argparse.ArgumentParser(description="Polynomial negotiation game")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--agents_num", type=int, default=4)
    parser.add_argument("--rounds_num", type=int, default=16)
    parser.add_argument("--max_step", type=int, default=2)
    parser.add_argument(
        "--min_answers",
        type=int,
        default=None,
        help="Total minimum number of answers across the run; must be divisible by agents_num. "
             "If set, overrides rounds_num. Example: 16 with 4 agents -> each speaks exactly 4 times.",
    )

    parser.add_argument("--output_dir", type=str, default="./output/")
    parser.add_argument("--game_dir", type=str, default="./games_descriptions/polynomial_game")
    parser.add_argument("--config_file", type=str, default="config.txt", help="Config file name or path (default: config.txt in game_dir)")
    parser.add_argument("--exp_name", type=str, default="poly_demo")

    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--output_file", type=str, default="history.json")

    parser.add_argument("--embedding_model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding_cache_dir", type=str, default="")
    parser.add_argument(
        "--reuse_faiss",
        action="store_true",
        help="Reuse existing FAISS index at <output_dir>/faiss_index instead of creating a new timestamped folder",
    )
    parser.add_argument(
        "--result",
        type=str,
        default=None,
        help="Optional result tag for this run (e.g., final agreed value). Stored in a per-tag results_<value>.json file.",
    )
    parser.add_argument("--hf_home", type=str, default="")
    parser.add_argument("--gemini", action="store_true")
    parser.add_argument("--gemini_project_name", type=str, default="")
    parser.add_argument("--gemini_loc", type=str, default="")
    parser.add_argument("--gemini_model", type=str, default="gemini-1.0-pro-001")
    parser.add_argument("--azure", action="store_true")
    parser.add_argument("--azure_openai_api", default="")
    parser.add_argument("--azure_openai_endpoint", default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--anthropic_api", type=str, default="", help="Anthropic API key for Claude models")
    parser.add_argument("--anthropic_base_url", type=str, default="", help="Anthropic base URL (set to Azure Anthropic endpoint for Azure Claude)")
    parser.add_argument("--env-file", type=str, default=".env", help="Optional .env file with API keys/endpoints")

    args = parser.parse_args()

    # Load .env before consuming args/env combos.
    load_env_file(args.env_file)
    # Fill args from env if not passed explicitly.
    env_overrides = {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "azure_openai_api": os.getenv("AZURE_OPENAI_API_KEY", ""),
        "azure_openai_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        "anthropic_api": os.getenv("ANTHROPIC_API_KEY", ""),
        "anthropic_base_url": os.getenv("ANTHROPIC_BASE_URL", ""),
    }
    for field, env_val in env_overrides.items():
        if getattr(args, field) == "" and env_val:
            setattr(args, field, env_val)

    # configure OpenAI/Azure/Gemini/HF env vars.
    set_constants(args)

    force_full_rounds = args.min_answers is not None
    effective_rounds = args.min_answers if force_full_rounds else args.rounds_num
    if args.min_answers is not None and args.min_answers % args.agents_num != 0:
        raise ValueError("--min_answers must be divisible by --agents_num so each agent gets the same number of turns.")
    per_agent_quota = None
    if args.min_answers is not None:
        per_agent_quota = args.min_answers // args.agents_num

    output_root = os.path.join(args.game_dir, args.output_dir, args.exp_name)
    agent_round_assignment, start_round_idx, history = create_outfiles(args, output_root)

    # Prefer explicit embedding cache dir; fall back to hf_home/env if provided and writable to avoid unwritable defaults.
    cache_dir = None
    cache_candidates = [
        args.embedding_cache_dir,
        args.hf_home,
        os.environ.get("HF_HOME"),
        os.environ.get("TRANSFORMERS_CACHE"),
    ]
    for candidate in cache_candidates:
        if not candidate:
            continue
        try:
            os.makedirs(candidate, exist_ok=True)
        except OSError:
            continue
        if os.access(candidate, os.W_OK):
            cache_dir = candidate
            break
    answer_comparator = None
    run_id = str(uuid.uuid4())
    run_success = False
    try:
        answer_comparator = AnswerComparator(
            os.path.join(args.output_dir, "faiss_index"),
            model_name=args.embedding_model_name,
            cache_dir=cache_dir,
            reuse_existing=args.reuse_faiss,
        ) 

        os.makedirs(output_root, exist_ok=True)
        cfg_src = args.config_file
        if not os.path.isabs(cfg_src):
            cfg_src = os.path.join(args.game_dir, cfg_src)
        cfg_dst = os.path.join(output_root, os.path.basename(cfg_src))
        shutil.copyfile(cfg_src, cfg_dst)
        poly_dir = os.path.join(args.game_dir, "polynomial_functions")
        shutil.copytree(poly_dir, os.path.join(output_root, "polynomial_functions"), dirs_exist_ok=True)
        
        # Load setups of agents from config file. File should contain names, file names, roles, incentives, and models 
        # Also load initial deal file and return a dict of role to agent names
        agents, initial_line, role_to_agent_names = load_setup(args.game_dir, args.agents_num, args.config_file)
       
        # Load HF models 
        hf_models = {}

        current_x = parse_initial_x(initial_line)

        if args.restart:
            saved_state = history["content"].get("polynomial_state", {})
            if "x" in saved_state:
                current_x = saved_state["x"]

        polynomial_profiles = {}
        for name, info in agents.items():
            profile = load_polynomial_profile(args.game_dir, info["file_name"])
            polynomial_profiles[name] = profile

        agent_names = list(agents.keys())
        starter_agent = role_to_agent_names.get("p1", agent_names[0])
        global_low, global_high = polynomial_profiles[starter_agent]["domain"]

        # Instaniate agents (initial prompt, polynomial round prompt, agent class)
        for name, details in agents.items():
            if "hf" in details["model"] and details["model"] not in hf_models:
                hf_models[details["model"]] = setup_hf_model(
                    details["model"].split("hf_")[-1], cache_dir=args.hf_home
                )

            init_prompt = PolynomialInitialPrompt(
                args.game_dir, name, details["file_name"]
            )
            round_prompt = PolynomialRoundPrompts(
                name,
                polynomial_profiles[name]["domain"][0],
                polynomial_profiles[name]["domain"][1],
                starter_agent,
                current_x,
                rounds_total=effective_rounds,
                reminder_text="infer other agents’ utility functions; do not reveal your own; use inferred models to maximize your utility.",
            )
            agent_instance = Agent(
                init_prompt,
                round_prompt,
                name,
                args.temp,
                model=details["model"],
                azure=args.azure,
                hf_models=hf_models,
            )
            agents[name]["instance"] = agent_instance

        def build_balanced_schedule(agent_names, quota):
            schedule = []
            for _ in range(quota):
                block = list(agent_names)
                random.shuffle(block)
                schedule.extend(block)
            return schedule

        if not args.restart:
            if per_agent_quota is not None:
                agent_round_assignment = build_balanced_schedule(agent_names, per_agent_quota)
            else:
                agent_round_assignment = randomize_agents_order(
                    agents, starter_agent, effective_rounds
                )

        # keep x inside [-10,10] and force moves to be incremental (no counterpart in main.py).
        def clamp_value(value: int) -> int:
            return max(global_low, min(global_high, value))

        def limit_step(current: int, proposed: int) -> int:
            if abs(proposed - current) <= args.max_step:
                return proposed
            if proposed > current:
                return current + args.max_step
            return current - args.max_step


        # For each agent:
        # 1. Calculates the utility by evaluating their polynomial at x_value
        # 2. Checks if the utility meets or exceeds their individual threshold
        # Returns:
        #   - Dictionary mapping agent names to their calculated utilities
        #   - Dictionary mapping agent names to boolean acceptance status
        # Note: An agent accepts if utility >= their threshold, rejects otherwise
        def evaluate_all(x_value: int):
            utilities = {}
            accepted = {}
            for name, profile in polynomial_profiles.items():
                util = evaluate_polynomial(profile["coeffs"], x_value)
                utilities[name] = util
                accepted[name] = util >= profile["threshold"]
            return utilities, accepted


        def record_state(round_label, utilities=None, accepted=None):
            """
            Tracks and logs the negotiation state at each round for analysis and visualization.

            Updates two main structures in history["content"]:
            1. polynomial_state: Snapshot of current state with:
               - x: Current x value being considered
               - utilities: Dictionary of {agent_name: utility} if provided
               - accepted: Dictionary of {agent_name: bool} indicating acceptance if provided

            2. polynomial_trace: List of all states, appending a new entry with:
                - round: Current round label
                - x: Current x value
                - utilities: Copy of current utilities (or empty dict if None)
                - accepted: Copy of acceptance status (or empty dict if None)

            This enables both real-time monitoring and post-hoc analysis of the negotiation.
            """
            state = history["content"].setdefault("polynomial_state", {})
            state["x"] = current_x
            if utilities is not None:
                state["utilities"] = utilities
            if accepted is not None:
                state["accepted"] = accepted
            trace = history["content"].setdefault("polynomial_trace", [])
            trace.append(
                {
                    "round": round_label,
                    "x": current_x,
                    "utilities": dict(utilities) if utilities is not None else {},
                    "accepted": dict(accepted) if accepted is not None else {},
                }
            )

        agreement = False
        '''Main negotiation loop - handles agent turns, response processing, and agreement checking.

            For each round:
            1. Agent Selection:
            - First round: Uses the designated starter agent
            - Subsequent rounds: Follows the predefined agent_round_assignment order

            2. Agent Execution:
            - Generates a prompt for the current agent based on conversation history
            - Executes the agent's response using the appropriate model
            - Converts the response to text format
            - Saves the conversation history with the agent's response

            3. Response Processing:
            - Extracts the proposed x value from the agent's response
            - If no valid x is found, defaults to the current x
            - Ensures the proposed x is within the global domain bounds
            - Limits the step size to ensure gradual changes
            - Updates the current x with the processed value

            4. Evaluation and State Tracking:
            - Calculates utilities for all agents at the new x value
            - Determines if each agent accepts the current x (utility >= threshold)
            - Records the current state for analysis and visualization
            - Prints a summary of the current negotiation state

            5. Termination Check:
            - If all agents accept the current x, sets agreement to True and exits the loop
            - Otherwise, continues to the next round

            The loop continues until either:
            - All agents accept the current x value (agreement reached)
            - The maximum number of rounds is reached

            After the loop, if no agreement was reached, a final round is conducted where
            the first agent makes one last proposal that all agents must accept or reject.
            '''
        for round_idx in range(start_round_idx, effective_rounds):
            if round_idx == 0:
                current_agent = starter_agent
                slot_prompt, agent_response = agents[current_agent]["instance"].execute_round(
                    history["content"], round_idx
                )
                response_text = ensure_text(agent_response)
                history = save_conversation(
                    history,
                    current_agent,
                    response_text,
                    slot_prompt,
                    round_assign=agent_round_assignment,
                    initial=True,
                )
            else:
                current_agent = agent_round_assignment[round_idx]
                slot_prompt, agent_response = agents[current_agent]["instance"].execute_round(
                    history["content"], round_idx
                )
                response_text = ensure_text(agent_response)
                history = save_conversation(history, current_agent, response_text, slot_prompt)
            
            # Add Answer to FAISS index
            public_answer = history["content"]["rounds"][-1]["public_answer"]
            answer_comparator.add_answer (
                answer = public_answer,
                agent_name = current_agent,
                round_num = round_idx,
                run_id = run_id   
            )


            proposed = extract_value(response_text)
            if proposed is None:
                proposed = current_x
            proposed = clamp_value(proposed)
            proposed = limit_step(current_x, proposed)
            current_x = proposed
            utilities, accepted = evaluate_all(current_x)
            record_state(round_idx, utilities, accepted)
            write_file(history["content"], history["file"])
            print("=====")
            print(f"{current_agent} response: {response_text}")
            summary = [
                f"{name}: f(x)={utilities[name]:.2f} (>= {polynomial_profiles[name]['threshold']:.2f}) -> "
                f"{'ACCEPT' if accepted[name] else 'hold'}"
                for name in agents.keys()
            ]
            print("  Current x:", current_x)
            print("  " + " | ".join(summary))

            if all(accepted.values()):
                agreement = True
                if not force_full_rounds:
                    break


        '''Final proposal and agreement resolution phase.

        This section handles the endgame of the negotiation:
        1. If no agreement was reached in the main loop (agreement is False):
        - Announces a final review round
        - Selects the first agent (p1) to make one last proposal
        - Executes the agent's response and processes the proposed x value
        - Ensures the proposal respects constraints (domain bounds and step limits)
        - Evaluates the final proposal against all agents' utility functions
        - Records the final state for analysis
        - Prints the final response and summary of agent acceptances
        - Determines if the final proposal was accepted by all agents

        2. If an agreement was reached in the main loop:
        - Announces early agreement
        - Reports the final agreed-upon x value

        The final proposal is a last-ditch effort to reach consensus when the main
        negotiation rounds complete without agreement. This gives agents one final
        opportunity to adjust their positions and find a mutually acceptable solution.

        The outcome is either:
        - A successful agreement (all agents accept the final proposal)
        - A failed negotiation (at least one agent rejects the final proposal)
        - An early agreement (reached during the main negotiation loop)
        '''

        if not agreement:
            print("Final review round.")
            p1 = list(agents.keys())[0]
            slot_prompt, agent_response = agents[p1]["instance"].execute_round(
                history["content"], effective_rounds
            )
            response_text = ensure_text(agent_response)
            history = save_conversation(history, p1, response_text, slot_prompt)
            proposed = extract_value(response_text)
            if proposed is not None:
                proposed = limit_step(current_x, clamp_value(proposed))
                current_x = proposed
            utilities, accepted = evaluate_all(current_x)
            record_state("final", utilities, accepted)
            write_file(history["content"], history["file"])
            print("=====")
            print(f"{p1} response: {response_text}")
            summary = [
                f"{name}: f(x)={utilities[name]:.2f} (>= {polynomial_profiles[name]['threshold']:.2f}) -> "
                f"{'ACCEPT' if accepted[name] else 'hold'}"
                for name in agents.keys()
            ]
            print("  Current x:", current_x)
            print("  " + " | ".join(summary))
            if all(accepted.values()):
                print("Agreement reached at the final vote.")
            else:
                print("Polynomial negotiation ended without unanimous acceptance.")
        else:
            print("Early agreement reached.")
            print(f"Final x: {current_x}")

        print("\n=== Similarity Analysis ===")
        similarity_report = {
            "run_id": run_id,
            "timestamp": int(time.time()),
            "output_root": output_root,
            "agents": [],
        }
        for agent_name in agents.keys():
            print(f"\nComparing {agent_name} 's answers across runs")
            similar = answer_comparator.compare_agent_answers(agent_name, round_num=0)
            similarity_report["agents"].append(
                {"agent_name": agent_name, "results": similar}
            )
            for reasult in similar:
                print(f"  Similarity result: {reasult}")
                print(f"Run ID: {reasult['run_id']}")
                print(f" Answer: {reasult['answer']}")
        similarity_path = os.path.join(output_root, f"similarity_{run_id}.json")
        with open(similarity_path, "w") as sim_file:
            json.dump(similarity_report, sim_file, indent=2)
        print(f"Saved similarity analysis to {similarity_path}")

        if args.result:
            safe_tag = re.sub(r"[^A-Za-z0-9_-]+", "_", args.result)
            result_path = os.path.join(output_root, f"results_{safe_tag}.json")
            payload = {"result": args.result, "runs": []}
            if os.path.exists(result_path):
                try:
                    with open(result_path, "r") as res_file:
                        payload = json.load(res_file)
                    if "runs" not in payload:
                        payload["runs"] = []
                except (OSError, json.JSONDecodeError):
                    payload = {"result": args.result, "runs": []}
            run_label = f"r_{len(payload['runs']) + 1}"
            payload["runs"].append(
                {
                    "label": run_label,
                    "run_id": run_id,
                    "timestamp": int(time.time()),
                    "final_x": current_x,
                    "output_root": output_root,
                }
            )
            with open(result_path, "w") as res_file:
                json.dump(payload, res_file, indent=2)
            print(f"Recorded run {run_label} to {result_path}")

        # Generate visualizations on success
        try:
            images_dir = os.path.join(output_root, "images")
            sim_img_path = Path(images_dir) / "similarity.png"
            generate_similarity_image(Path(answer_comparator.embeddings_file), sim_img_path)

            history_path = Path(history["file"])
            poly_img_path = Path(images_dir) / "polynomial.png"
            generate_polynomial_image(history_path, poly_img_path)
        except Exception as viz_err:
            print(f"Visualization generation failed: {viz_err}")

        run_success = True
    finally:
        if not run_success and answer_comparator is not None:
            removed = answer_comparator.remove_run(run_id)
            if removed:
                print(f"Run failed; removed {removed} FAISS entries for run_id {run_id}.")
            else:
                print("Run failed; no FAISS entries to remove for this run.")
        if not run_success:
            try:
                if os.path.isdir(output_root):
                    shutil.rmtree(output_root, ignore_errors=True)
                    print(f"Run failed; removed output folder at {output_root}.")
            except Exception:
                print(f"Run failed; could not remove output folder at {output_root}.")

if __name__ == "__main__":
    main()
