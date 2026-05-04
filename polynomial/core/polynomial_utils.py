import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


def read_config(config_path: str) -> Dict:
    """
    Parse a simple config file with lines:
        DisplayName,file_key,role,incentive,model
    Returns a dict with agents list and basic metadata.
    """
    cfg_path = Path(config_path)
    game_dir = cfg_path.parent
    agents: List[Dict] = []
    with cfg_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 5:
                raise ValueError(f"Malformed config line: {line}")
            name, file_key, role, incentive, model = parts
            agents.append(
                {
                    "name": name,
                    "file_name": file_key,
                    "role": role,
                    "incentive": incentive,
                    "model": model,
                }
            )
    # initial x from initial_deal.txt
    init_file = game_dir / "initial_deal.txt"
    initial_x = 0
    if init_file.exists():
        txt = init_file.read_text()
        m = re.search(r"<VALUE>\\s*([-+]?\\d+)", txt, re.IGNORECASE)
        if not m:
            m = re.search(r"([-+]?\\d+)", txt)
        if m:
            initial_x = int(m.group(1))
    starter = next((a["name"] for a in agents if a["role"] == "p1"), agents[0]["name"] if agents else "")
    return {
        "agents": agents,
        "starter": starter,
        "initial_x": initial_x,
        "polynomials": "polynomial_functions",
        "domain_low": -10,
        "domain_high": 10,
        "round_assign": [],
        "seed": 42,
    }


def print_polynomial_profiles(profiles: Dict[str, Dict]) -> None:
    for name, prof in profiles.items():
        print(f"{name}: {prof.get('formula','')} thresh={prof.get('threshold')}")


def load_polynomial(game_dir: str, poly_dir_name: str) -> Dict[str, Dict]:
    base = Path(game_dir) / poly_dir_name
    profiles: Dict[str, Dict] = {}
    for p in base.glob("*.txt"):
        name = p.stem
        profiles[name] = load_polynomial_profile(game_dir, name, poly_dir_name)
    return profiles


def load_polynomial_profile(game_dir: str, file_key: str, poly_dir_name: str = "polynomial_functions") -> Dict[str, object]:
    """
    Parse a polynomial profile file with lines like:
        # f_A(x) = 2x + 10
        COEFFS 2 10
        DOMAIN -10 10
        THRESHOLD 6
    """
    path = Path(game_dir) / poly_dir_name / f"{file_key}.txt"
    coeffs: List[float] = []
    domain: Tuple[int, int] = (-10, 10)
    threshold: float = 0.0
    formula = ""
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if line.startswith("#"):
                    formula = line.lstrip("#").strip()
                continue
            parts = line.split()
            tag = parts[0].upper()
            if tag == "COEFFS":
                coeffs = [float(x) for x in parts[1:]]
            elif tag == "DOMAIN" and len(parts) >= 3:
                domain = (int(parts[1]), int(parts[2]))
            elif tag == "THRESHOLD" and len(parts) >= 2:
                threshold = float(parts[1])
    return {
        "coeffs": coeffs,
        "domain": domain,
        "threshold": threshold,
        "formula": formula,
    }


def evaluate_polynomial(coeffs: List[float], x_value: int) -> float:
    out = 0.0
    for c in coeffs:
        out = out * x_value + c
    return out


def evaluate_all_agents(polynomial_profiles: Dict[str, Dict], x_value: int):
    utilities = {}
    accepted = {}
    for name, prof in polynomial_profiles.items():
        util = evaluate_polynomial(prof["coeffs"], x_value)
        utilities[name] = util
        accepted[name] = util >= prof["threshold"]
    return utilities, accepted


def clamp_value(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def limit_step(current: int, proposed: int, max_step: int = 2) -> int:
    if abs(proposed - current) <= max_step:
        return proposed
    return current + max_step if proposed > current else current - max_step


def extract_value(text: str) -> int | None:
    if not text:
        return None
    m = re.search(r"<VALUE>\s*([-+]?\d+)\s*</VALUE>", text, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    m2 = re.search(r"([-+]?\\d+)", text)
    if m2:
        try:
            return int(m2.group(1))
        except ValueError:
            return None
    return None


def format_history(agent_name, history, window=6):
    # Reuse prompt_utils style formatting
    last_plan = ""
    personalized_history = []
    for slot in history["rounds"][-window:]:
        if agent_name == slot["agent"]:
            slot_str = f". You ({slot['agent']}): {slot['public_answer']}"
        else:
            slot_str = f". {slot['agent']}: {slot['public_answer']}"
        personalized_history.append(slot_str)
    personalized_history_string = " \n ".join(personalized_history)
    if agent_name in history.get("plan", {}):
        last_plan = history["plan"][agent_name][-1]
    return personalized_history_string, last_plan
