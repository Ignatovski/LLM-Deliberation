import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

from agent import Agent
from polynomial_utils import (
    clamp_value,
    evaluate_all_agents,
    extract_value,
    format_history,
    limit_step,
    load_polynomial,
    load_polynomial_profile,
    print_polynomial_profiles,
    read_config,
)
from save_utils import create_outfiles, save_conversation, write_file


JSON_PROMPT = """You must respond with ONE JSON object, nothing else:
{
  "scratchpad": "<SCRATCHPAD>…</SCRATCHPAD>",
  "answer": "<ANSWER>Your brief public message (1–2 sentences) with exactly one <VALUE>n</VALUE> integer.</ANSWER>",
  "plan": "<PLAN>…</PLAN>"
}

Rules:
- scratchpad: private reasoning only; keep it brief.
- answer: public text inside <ANSWER>…</ANSWER>; exactly one <VALUE>n</VALUE> with an integer n; you may include up to two short sentences before/after the VALUE; no other numbers or tags.
- plan: next-steps notes; omit or use "<PLAN></PLAN>" if none.
- Range: n must be within [-10, 10] and within ±2 of the previous public value (if given).
- Do not include any utilities, scores, thresholds, or coefficients in answer or plan.
- Output must be valid JSON; no extra text, comments, or Markdown.
If you cannot comply, reformat to valid JSON with the fields above; never return other text.
"""


def load_env_file(path: str) -> None:
    """Load KEY=VALUE lines from an env file if they are not already in os.environ."""
    if not os.path.exists(path):
        return
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, val = stripped.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val


def set_constants_light(args):
    # Minimal env setup without importing heavy deps.
    if getattr(args, "anthropic_api", None):
        os.environ["ANTHROPIC_API_KEY"] = args.anthropic_api
    if getattr(args, "anthropic_base_url", None):
        os.environ["ANTHROPIC_BASE_URL"] = args.anthropic_base_url
    if getattr(args, "api_key", None):
        os.environ["OPENAI_API_KEY"] = args.api_key
    if getattr(args, "azure_openai_api", None):
        os.environ["AZURE_OPENAI_API_KEY"] = args.azure_openai_api
    if getattr(args, "azure_openai_endpoint", None):
        os.environ["AZURE_OPENAI_ENDPOINT"] = args.azure_openai_endpoint
    if getattr(args, "hf_home", None):
        os.environ["HF_HOME"] = args.hf_home
        os.environ["TRANSFORMERS_CACHE"] = args.hf_home


class PolynomialInitialPromptJSON:
    """
    Initial prompt builder that loads global + individual instructions and (optionally)
    includes a private polynomial note.
    """

    def __init__(self, game_dir: str, agent_name: str, agent_file_name: str, behavior_pack: str, poly_profile: Optional[Dict] = None):
        global_text_path = os.path.join(game_dir, "global_instructions.txt")
        with open(global_text_path, "r") as f:
            self.global_text = f.read().strip()

        instruction_path = os.path.join(game_dir, behavior_pack, f"{agent_file_name}.txt")
        with open(instruction_path, "r") as f:
            self.personal_text = f.read().strip()

        private_note = ""
        if poly_profile:
            domain_low, domain_high = poly_profile.get("domain", (-10, 10))
            formula = poly_profile.get("formula", "")
            threshold = poly_profile.get("threshold", 0)
            private_note = (
                f"Technical note (keep private): your polynomial is {formula}. "
                f"Stay within [{domain_low}, {domain_high}] and only accept if f(x) >= {threshold:g}. "
                "Do not reveal these exact numbers publicly."
            )

        parts = [self.global_text, self.personal_text]
        if private_note:
            parts.append(private_note)
        self.initial_prompt = "\n\n".join(parts)

    def return_initial_prompt(self):
        return self.initial_prompt


class PolynomialSlotPromptJSON:
    """
    Same as the original slot prompt, but requests structured JSON output.
    """

    def __init__(
        self,
        agent_name,
        domain_low,
        domain_high,
        starter_name,
        initial_x,
        rounds_total=None,
        reminder_text: str = "",
    ):
        self.agent_name = agent_name
        self.domain_low = domain_low
        self.domain_high = domain_high
        self.starter_name = starter_name
        self.initial_x = initial_x
        self.rounds_total = rounds_total
        self.reminder_text = reminder_text

    def build_slot_prompt(self, history, round_idx, *_):
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
                "Keep it brief.\n\n"
                + JSON_PROMPT
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

        current_x_hint = f"Current shared x (after limits) is {state.get('x', self.initial_x)}."
        prompt += (
            f"Work within these constraints: {current_x_hint} "
            "and keep step size within ±2 of the previous public value. "
            "Respond in the required JSON format only.\n\n"
            + JSON_PROMPT
        )
        if final_round:
            prompt += "This is the final turn; plan may be empty.\n"
        return prompt


def ensure_text(response) -> str:
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


def parse_structured_response(
    resp_text: str,
    prev_value: Optional[int],
    max_step: int,
    domain_low: int,
    domain_high: int,
) -> Optional[Dict[str, str]]:
    """
    Parse JSON response and validate structure/rules.
    Returns dict with scratchpad/answer/plan if valid, else None.
    """
    def try_load(text: str) -> Optional[Dict[str, str]]:
        # Strip code fences if present
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            # Remove leading 'json' if present
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].lstrip()
        try:
            return json.loads(stripped)
        except Exception:
            pass
        # Fallback: grab first JSON object via regex
        import re
        m = re.search(r"\{.*\}", stripped, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

    parsed = try_load(resp_text)
    if not parsed:
        return None
    if not isinstance(parsed, dict):
        return None
    for key in ("scratchpad", "answer", "plan"):
        if key not in parsed or not isinstance(parsed[key], str):
            return None
    answer = parsed["answer"]
    if "<ANSWER>" not in answer or "</ANSWER>" not in answer:
        return None
    # extract VALUE
    start = answer.find("<VALUE>")
    end = answer.find("</VALUE>")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        val_str = answer[start + len("<VALUE>"):end].strip()
        val = int(val_str)
    except Exception:
        return None
    # range check
    if val < domain_low or val > domain_high:
        return None
    if prev_value is not None and abs(val - prev_value) > max_step:
        return None
    # basic cleanliness: only one VALUE tag
    if answer.lower().count("<value>") != 1 or answer.lower().count("</value>") != 1:
        return None
    return parsed


def validate_structured_response(
    resp_text: str,
    prev_value: Optional[int],
    max_step: int,
    domain_low: int,
    domain_high: int,
) -> tuple[Optional[Dict[str, str]], str]:
    """
    Like parse_structured_response, but also returns a short failure reason for logging.
    """
    if not (resp_text or "").strip():
        return None, "empty response"
    parsed = parse_structured_response(
        resp_text,
        prev_value=prev_value,
        max_step=max_step,
        domain_low=domain_low,
        domain_high=domain_high,
    )
    if parsed is None:
        return None, "invalid JSON or rule violation"
    return parsed, ""


def save_state(history, agent_name, prompt, parsed_obj):
    """
    Build a full_answer from structured parts and persist via save_conversation.
    """
    scratch = parsed_obj.get("scratchpad", "")
    ans = parsed_obj.get("answer", "")
    plan = parsed_obj.get("plan", "")
    full_answer = "\n".join(part for part in [scratch, ans, plan] if part)
    history = save_conversation(history, agent_name, full_answer, prompt)
    return history


def main():
    parser = argparse.ArgumentParser(description="Polynomial negotiation game (JSON enforced)")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--agents_num", type=int, default=4)
    parser.add_argument("--rounds_num", type=int, default=16)
    parser.add_argument("--max_step", type=int, default=2)
    parser.add_argument(
        "--min_answers",
        type=int,
        default=None,
        help="Total minimum number of answers across the run; must be divisible by agents_num.",
    )
    parser.add_argument("--output_dir", type=str, default="./output/")
    parser.add_argument("--game_dir", type=str, default="./games_descriptions/polynomial_game")
    parser.add_argument("--config_file", type=str, default="config.txt", help="Config file name or path")
    parser.add_argument("--exp_name", type=str, default="poly_demo_json")
    parser.add_argument("--restart", action="store_true", help="Restart flag.")
    parser.add_argument("--output_file", type=str, default="history.json")
    parser.add_argument("--azure", action="store_true", help="Use Azure OpenAI")
    parser.add_argument("--model", type=str, default="gpt-5")
    parser.add_argument("--claude", action="store_true", help="Use Claude API")
    parser.add_argument("--gemini", action="store_true", help="Use Gemini API")
    parser.add_argument("--hf", action="store_true", help="Use HF model")
    parser.add_argument("--hf_name", type=str, default="gpt2", help="HF model name")
    parser.add_argument("--hf_temp", type=float, default=0.7, help="HF temperature")
    parser.add_argument("--hf_tokens", type=int, default=256, help="HF max tokens")
    parser.add_argument("--faiss", action="store_true", help="Use FAISS similarity checker")
    parser.add_argument("--api_key", type=str, default="", help="OpenAI API key (or set via .env)")
    parser.add_argument("--azure_openai_api", type=str, default="", help="Azure OpenAI API key")
    parser.add_argument("--azure_openai_endpoint", type=str, default="", help="Azure OpenAI endpoint")
    parser.add_argument("--anthropic_api", type=str, default="", help="Anthropic API key")
    parser.add_argument("--anthropic_base_url", type=str, default="", help="Anthropic base URL (optional)")
    parser.add_argument("--gemini_project_name", type=str, default="", help="Gemini project name")
    parser.add_argument("--gemini_loc", type=str, default="", help="Gemini location")
    parser.add_argument("--hf_home", type=str, default=os.getenv("HF_HOME", ""), help="HF cache dir")
    parser.add_argument("--embedding_cache_dir", type=str, default="", help="Embedding cache dir for FAISS")
    parser.add_argument("--env_file", type=str, default=".env", help="Optional .env file with credentials")
    parser.add_argument(
        "--max_attempts_per_turn",
        type=int,
        default=5,
        help=(
            "Max attempts per agent turn to obtain valid structured JSON. "
            "If exceeded, the turn is skipped (x is unchanged) and the run continues. "
            "Set 0 for infinite retries."
        ),
    )
    parser.add_argument(
        "--retry_sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between retries when a model response is invalid.",
    )
    args = parser.parse_args()

    load_env_file(args.env_file)
    set_constants_light(args)

    config_path = os.path.join(args.game_dir, args.config_file)
    config = read_config(config_path)
    # Ensure initial_x parsed with robust extractor
    init_file = os.path.join(args.game_dir, "initial_deal.txt")
    if os.path.exists(init_file):
        val = extract_value(Path(init_file).read_text())
        if val is not None:
            config["initial_x"] = val

    AGENTS = config["agents"]
    if args.agents_num != len(AGENTS):
        raise SystemExit("agents_num must match number of agents in config")

    random.seed(config.get("seed", 42))

    # Load polynomial profiles keyed by display name
    polynomial_profiles = {}
    for agent in AGENTS:
        profile = load_polynomial_profile(args.game_dir, agent["file_name"], config["polynomials"])
        polynomial_profiles[agent["name"]] = profile
    print_polynomial_profiles(polynomial_profiles)

    hf_models = {}
    agents = {}
    for agent in AGENTS:
        name = agent["name"]
        model_name = args.model if not agent.get("model") else agent["model"]
        behavior_pack = agent.get("incentive", "cooperative") if isinstance(agent, dict) else "cooperative"
        init_prompt = PolynomialInitialPromptJSON(
            args.game_dir,
            name,
            agent["file_name"],
            behavior_pack=behavior_pack,
            poly_profile=polynomial_profiles.get(name, {}),
        )
        round_prompt = PolynomialSlotPromptJSON(
            name,
            config["domain_low"],
            config["domain_high"],
            config["starter"],
            config["initial_x"],
            rounds_total=args.rounds_num,
            reminder_text="infer other agents’ utility functions; do not reveal your own; use inferred models to maximize your utility.",
        )
        agent_obj = Agent(
            init_prompt,
            round_prompt,
            name,
            args.temp,
            model=model_name,
            azure=args.azure,
            hf_models=hf_models,
        )
        agents[name] = {"instance": agent_obj}

    # Output setup
    OUTPUT_DIR = os.path.join(args.output_dir, args.exp_name)
    round_assign, round_start, history = create_outfiles(args, OUTPUT_DIR)
    round_assign = round_assign or config["round_assign"]
    history.setdefault("content", {})
    history["content"].setdefault("rounds", [])
    history["content"].setdefault("plan", {})
    history["content"].setdefault("finished_rounds", 0)

    # Assign starter
    current_x = config["initial_x"]
    max_step = args.max_step

    # similarity optional
    answer_comparator = None
    if args.faiss:
        from faiss_utility import AnswerComparator  # lazy import to avoid faiss dependency when unused
        answer_comparator = AnswerComparator()

    # run loop
    force_full_rounds = args.min_answers is not None
    effective_rounds = args.min_answers if force_full_rounds else args.rounds_num
    start_round_idx = round_start
    agreement = False
    structured_failures = 0

    def evaluate_all(x_val: int):
        utilities, accepted = evaluate_all_agents(polynomial_profiles, x_val)
        history["content"]["polynomial_state"] = {"x": x_val}
        return utilities, accepted

    def record_state(step, utilities, accepted):
        state = history["content"].setdefault("polynomial_trace", [])
        state.append(
            {
                "round": step,
                "x": current_x,
                "utilities": utilities,
                "accepted": accepted,
            }
        )

    # main rounds
    if not round_assign:
        agent_names = list(agents.keys())
        # simple cyclic schedule starting with starter
        if config["starter"] in agent_names:
            start_idx = agent_names.index(config["starter"])
            agent_names = agent_names[start_idx:] + agent_names[:start_idx]
        # repeat to cover effective_rounds
        reps = (effective_rounds + len(agent_names) - 1) // len(agent_names)
        round_assign = (agent_names * reps)[:effective_rounds]

    for round_idx in range(start_round_idx, effective_rounds):
        current_agent = round_assign[round_idx]

        prev_public_value = None
        if history["content"]["rounds"]:
            last_pa = history["content"]["rounds"][-1]["public_answer"]
            prev_public_value = extract_value(last_pa)

        # Get response with retries/parsing (no fallback: keep retrying until valid, or abort if max_attempts_per_turn > 0)
        parsed: Optional[Dict[str, str]] = None
        slot_prompt = ""
        attempts = 0
        while True:
            attempts += 1
            slot_prompt, agent_response = agents[current_agent]["instance"].execute_round(
                history["content"], round_idx
            )
            response_text = ensure_text(agent_response)
            if not response_text.strip():
                nudged_prompt = (
                    slot_prompt
                    + "\n\nReminder: return ONLY the JSON object with keys scratchpad/answer/plan. "
                    + "The answer must contain exactly one integer <VALUE>n</VALUE> within the allowed step/range."
                )
                response_text = ensure_text(agents[current_agent]["instance"].prompt("user", nudged_prompt))

            print("RAW LLM RESPONSE:", response_text)
            parsed, err = validate_structured_response(
                response_text,
                prev_value=prev_public_value,
                max_step=max_step,
                domain_low=config["domain_low"],
                domain_high=config["domain_high"],
            )
            if parsed:
                break
            print(
                f"[{current_agent}] invalid structured response (attempt {attempts}) -> {err}",
                flush=True,
            )
            if args.max_attempts_per_turn and attempts >= args.max_attempts_per_turn:
                structured_failures += 1
                print(
                    f"[{current_agent}] FAILED structured output after {attempts} attempts; "
                    f"skipping turn (keeping x={current_x}).",
                    flush=True,
                )
                parsed = {
                    "scratchpad": "<SCRATCHPAD>Structured output failure; skipping turn to avoid leaking raw content.</SCRATCHPAD>",
                    "answer": f"<ANSWER><VALUE>{current_x}</VALUE></ANSWER>",
                    "plan": "<PLAN></PLAN>",
                }
                break
            if args.retry_sleep:
                time.sleep(args.retry_sleep)

        history = save_state(history, current_agent, slot_prompt, parsed)
        public_answer = parsed["answer"]

        if answer_comparator:
            answer_comparator.add_answer(
                answer=public_answer,
                agent_name=current_agent,
                round_num=round_idx,
                run_id=args.exp_name,
            )

        proposed = extract_value(public_answer)
        if proposed is None:
            proposed = current_x
        proposed = clamp_value(proposed, config["domain_low"], config["domain_high"])
        proposed = limit_step(current_x, proposed, max_step=max_step)
        current_x = proposed
        utilities, accepted = evaluate_all(current_x)
        record_state(round_idx, utilities, accepted)
        write_file(history["content"], history["file"])
        print("=====")
        print(f"{current_agent} response (parsed): {parsed}")
        print(f"{current_agent} public answer: {public_answer}")
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

    # final round if no agreement
    if not agreement:
        print("Final review round.")
        p1 = list(agents.keys())[0]
        prev_public_value = extract_value(history["content"]["rounds"][-1]["public_answer"]) if history["content"]["rounds"] else None
        parsed: Optional[Dict[str, str]] = None
        slot_prompt = ""
        attempts = 0
        while True:
            attempts += 1
            slot_prompt, agent_response = agents[p1]["instance"].execute_round(
                history["content"], effective_rounds
            )
            response_text = ensure_text(agent_response)
            if not response_text.strip():
                nudged_prompt = (
                    slot_prompt
                    + "\n\nReminder: return ONLY the JSON object with keys scratchpad/answer/plan. "
                    + "The answer must contain exactly one integer <VALUE>n</VALUE> within the allowed step/range."
                )
                response_text = ensure_text(agents[p1]["instance"].prompt("user", nudged_prompt))

            print("RAW LLM RESPONSE:", response_text)
            parsed, err = validate_structured_response(
                response_text,
                prev_value=prev_public_value,
                max_step=max_step,
                domain_low=config["domain_low"],
                domain_high=config["domain_high"],
            )
            if parsed:
                break
            print(
                f"[{p1}] invalid structured response (attempt {attempts}) -> {err}",
                flush=True,
            )
            if args.max_attempts_per_turn and attempts >= args.max_attempts_per_turn:
                structured_failures += 1
                print(
                    f"[{p1}] FAILED structured output after {attempts} attempts; "
                    f"skipping final turn (keeping x={current_x}).",
                    flush=True,
                )
                parsed = {
                    "scratchpad": "<SCRATCHPAD>Structured output failure; skipping turn to avoid leaking raw content.</SCRATCHPAD>",
                    "answer": f"<ANSWER><VALUE>{current_x}</VALUE></ANSWER>",
                    "plan": "<PLAN></PLAN>",
                }
                break
            if args.retry_sleep:
                time.sleep(args.retry_sleep)

        history = save_state(history, p1, slot_prompt, parsed)
        public_answer = parsed["answer"]
        proposed = extract_value(public_answer)
        if proposed is not None:
            proposed = limit_step(current_x, clamp_value(proposed, config["domain_low"], config["domain_high"]), max_step=max_step)
            current_x = proposed
        utilities, accepted = evaluate_all(current_x)
        record_state("final", utilities, accepted)
        write_file(history["content"], history["file"])
        print("=====")
        print(f"{p1} response (parsed): {parsed}")
        print(f"{p1} public answer: {public_answer}")
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
    if structured_failures:
        print(f"Structured-output failures (skipped turns): {structured_failures}")

    if answer_comparator:
        print("\n=== Similarity Analysis ===")
        similarity_report = {
            "run_id": args.exp_name,
            "timestamp": int(time.time()),
            "output_root": OUTPUT_DIR,
            "agents": [],
        }
        for agent_name in agents.keys():
            print(f"\nComparing {agent_name} 's answers across runs")
            similar = answer_comparator.compare_agent_answers(agent_name, round_num=0, run_id=args.exp_name)
            similarity_report["agents"].append(
                {"agent_name": agent_name, "results": similar}
            )
            for reasult in similar:
                print(f"  Similarity result: {reasult}")
                print(f"Run ID: {reasult['run_id']}")
                print(f" Answer: {reasult['answer']}")
        similarity_path = os.path.join(OUTPUT_DIR, f"similarity_{args.exp_name}.json")
        with open(similarity_path, "w") as sim_file:
            json.dump(similarity_report, sim_file, indent=2)
        print(f"Saved similarity report to {similarity_path}")


if __name__ == "__main__":
    main()


"""
python main_polynomial_json.py \
  --game_dir games_descriptions/polynomial_game \
  --config_file config.txt \
  --output_dir output \
  --exp_name poly_demo_json \
  --model gpt-5 --azure

"""
