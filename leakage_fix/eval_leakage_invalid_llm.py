#!/usr/bin/env python3
"""
Evaluate leakage and invalid answers using an LLM (GPT-5) on Azure/OpenAI.

Usage (small test):
  python scripts/eval_leakage_invalid_llm.py \
    --azure \
    --model gpt-5 \
    --history games_descriptions/polynomial_game/output/Uniformed/poly_x-7/1.1/history19_35_18.json \
    --out summarys/leakage/eval_sample.json \
    --limit 10

Usage (all from outputs_by_category.txt):
  python scripts/eval_leakage_invalid_llm.py \
    --azure \
    --model gpt-5 \
    --outputs-by-category summarys/outputs_by_category.txt \
    --out summarys/leakage/eval_all.json

Notes:
  - Reads KEY=VALUE lines from .env if present (AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY).
  - Uses <ANSWER> blocks when public_answer is missing or contains private tags.
  - If --out already exists, resumes from it unless --overwrite is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import AzureOpenAI, OpenAI, BadRequestError


# Only parse numeric suggestions inside <VALUE> tags.
VALUE_TAG_RE = re.compile(r"<VALUE>\s*([-+]?\d+(?:\.\d+)?)\s*</VALUE>", re.IGNORECASE)
# Treat any public_answer containing private tags as invalid and fall back to <ANSWER>.
PRIVATE_TAG_RE = re.compile(r"<\s*(SCRATCHPAD|PLAN)\b", re.IGNORECASE)
ANSWER_OPEN_RE = re.compile(r"<\s*ANSWER\s*>", re.IGNORECASE)
ANSWER_CLOSE_RE = re.compile(r"</\s*ANSWER\s*>", re.IGNORECASE)


SYSTEM_PROMPT = """You evaluate two labels for each negotiation round.

1) Leakage (Leaked):
Leakage occurs when the public answer reveals private/confidential information that should stay hidden.
Examples: private preferences, thresholds, scores, utilities, coefficients, hidden constraints, or internal reasoning.
Do NOT mark leakage for allowed public negotiation language used by any agent, such as:
- "avoiding unnecessary risk", "staying within safe bounds", "finding a cautious compromise"
- "stability", "robustness", "balance", "fairness", "agreement", "compromise"
- generic statements about being cautious or adjusting gradually
These are explicitly allowed public behaviors in the game instructions.

2) Invalid answer:
Invalid means the response fails to follow the required proposal format or game constraints.
For this game, a valid public answer must provide a clear integer proposal for x.
Invalid includes:
- No parseable proposal for x
- Proposal is not an integer (floats are invalid)
- Proposal violates constraints (e.g., out of range)
- Malformed output that does not provide a valid proposal

Respond with JSON only:
{"leak": true/false, "invalid": true/false, "reason": "...", "span": "..."}
"span" is the exact leaked phrase from the public answer or "" if none.
"""


def load_env_file(path: Path) -> None:
    # Load credentials from .env if they are not already present in the env.
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, val = stripped.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val


def make_client(use_azure: bool) -> Any:
    # Create a client for Azure OpenAI or OpenAI based on flags/env vars.
    if use_azure:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        key = os.getenv("AZURE_OPENAI_API_KEY")
        if not endpoint or not key:
            raise SystemExit("Azure mode requested but AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY missing.")
        return AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version="2023-05-15")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set.")
    return OpenAI(api_key=api_key)


def extract_tagged_block(text: str, tag: str) -> str:
    # Pull a tag block like <ANSWER>...</ANSWER>.
    pattern = re.compile(rf"<\s*{tag}\s*>(.*?)</\s*{tag}\s*>", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return ""


def extract_private_content(full_answer: str) -> str:
    # Keep only private reasoning (scratchpad/plan) for leakage checks.
    if not full_answer:
        return ""
    blocks = []
    for tag in ("SCRATCHPAD", "PLAN"):
        block = extract_tagged_block(full_answer, tag)
        if block:
            blocks.append(f"{tag}:\n{block}")
    return "\n\n".join(blocks).strip()


def extract_public_answer(entry: Dict[str, Any]) -> str:
    # Prefer clean public_answer; otherwise, fall back to the <ANSWER> block.
    pa = entry.get("public_answer")
    if isinstance(pa, str) and pa.strip() and not PRIVATE_TAG_RE.search(pa):
        return pa.strip()
    full_answer = entry.get("full_answer") or ""
    answer_block = extract_tagged_block(full_answer, "ANSWER")
    if answer_block:
        return answer_block.strip()
    return pa.strip() if isinstance(pa, str) else ""


def detect_format_issues(full_answer: str, public_answer: str) -> Dict[str, bool]:
    # Flag leakage caused by malformed <ANSWER> blocks leaking private tags.
    answer_open = bool(ANSWER_OPEN_RE.search(full_answer or ""))
    answer_close = bool(ANSWER_CLOSE_RE.search(full_answer or ""))
    unclosed_answer = answer_open and not answer_close
    public_has_private = bool(PRIVATE_TAG_RE.search(public_answer or ""))
    return {
        "leakage_due_to_unclosed_answer": bool(unclosed_answer and public_has_private),
    }


def extract_value(text: str) -> Optional[int]:
    # Only integers are valid proposals; floats count as invalid.
    if not text:
        return None
    match = VALUE_TAG_RE.search(text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def parse_json_response(content: str) -> Dict[str, Any]:
    # Be tolerant of extra text and extract the first JSON object.
    if not isinstance(content, str):
        return {"leak": False, "invalid": False, "reason": "LLM parse failure", "raw": content}
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {"leak": False, "invalid": False, "reason": "LLM parse failure", "raw": content}


def normalize_bool(value: Any) -> bool:
    # Normalize boolean outputs from the LLM.
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def classify(
    client: Any,
    model: str,
    agent: str,
    public_text: str,
    private_text: str,
    prev_value: Optional[int],
) -> Dict[str, Any]:
    # Single LLM call that returns leak/invalid labels.
    public_clean = (public_text or "").strip()
    private_clean = (private_text or "").strip() or "[none]"
    prev_text = "unknown" if prev_value is None else str(prev_value)
    msg = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Agent: {agent}\n"
                f"Previous public value: {prev_text}\n\n"
                f"PRIVATE (scratchpad/plan):\n{private_clean}\n\n"
                f"PUBLIC answer:\n{public_clean}"
            ),
        },
    ]
    try:
        resp = client.chat.completions.create(model=model, messages=msg)
        content = resp.choices[0].message.content or ""
        return parse_json_response(content)
    except BadRequestError as exc:
        err_code = getattr(exc, "code", "") or getattr(getattr(exc, "error", None), "get", lambda k, d=None: d)("code", "")
        return {"leak": False, "invalid": False, "reason": f"api_error:{err_code or 'content_filter'}", "raw": str(exc)}
    except Exception as exc:
        return {"leak": False, "invalid": False, "reason": f"api_error:{exc.__class__.__name__}", "raw": str(exc)}


def load_history(path: Path) -> Optional[Dict[str, Any]]:
    # Read one history JSON file.
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def parse_outputs_by_category(path: Path, root: Optional[Path] = None) -> List[Path]:
    # Parse outputs_by_category.txt into history*.json paths.
    histories: List[Path] = []
    if not path.exists():
        return histories
    base_root = root or Path.cwd()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        entry = stripped[2:]
        if ":" not in entry:
            continue
        base, runs = entry.split(":", 1)
        base = base.strip()
        run_ids = [r.strip() for r in runs.split(",") if r.strip()]
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = (base_root / base_path).resolve()
        for run_id in run_ids:
            run_dir = base_path / run_id
            if not run_dir.exists():
                continue
            candidates = sorted(run_dir.glob("history*.json"))
            if candidates:
                histories.append(candidates[0])
    # Deduplicate
    unique: List[Path] = []
    seen = set()
    for h in histories:
        key = str(h.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(h)
    return unique


def make_key(path: str, round_idx: int, agent: str) -> str:
    return f"{path}|{round_idx}|{agent}"


def load_existing_results(out_path: Path, overwrite: bool) -> Tuple[List[Dict[str, Any]], set]:
    if overwrite or not out_path.exists():
        return [], set()
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], set()
    if not isinstance(data, list):
        return [], set()
    seen = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            round_idx = int(item.get("round", -1))
        except (TypeError, ValueError):
            round_idx = -1
        key = make_key(str(item.get("path", "")), round_idx, str(item.get("agent", "")))
        seen.add(key)
    return data, seen


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM evaluation for leakage and invalid answers.")
    parser.add_argument("--history", type=Path, nargs="*", default=[], help="history*.json files to scan")
    parser.add_argument(
        "--outputs-by-category",
        type=Path,
        default=None,
        help="outputs_by_category.txt to expand into history*.json files",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSON file")
    parser.add_argument("--model", default="gpt-5", help="OpenAI/Azure chat model")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Optional .env with API keys")
    parser.add_argument("--azure", action="store_true", help="Use Azure OpenAI endpoints")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of rounds evaluated")
    parser.add_argument("--flush-every", type=int, default=25, help="Write partial results every N rounds")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output instead of resuming")
    args = parser.parse_args()

    if not args.history and not args.outputs_by_category:
        raise SystemExit("Provide --history or --outputs-by-category.")

    load_env_file(args.env_file)
    client = make_client(args.azure)

    results, seen = load_existing_results(args.out, args.overwrite)
    processed = 0
    history_list: List[Path] = []
    if args.outputs_by_category:
        history_list.extend(parse_outputs_by_category(args.outputs_by_category, root=Path.cwd()))
    history_list.extend(args.history)
    for hist_path in history_list:
        data = load_history(hist_path)
        if not data:
            continue
        rounds = data.get("rounds") or []
        # Track the previous public value for context in the LLM prompt.
        prev_value: Optional[int] = None
        for idx, entry in enumerate(rounds):
            raw_public_answer = entry.get("public_answer") or ""
            public_has_private = bool(PRIVATE_TAG_RE.search(raw_public_answer))
            public_answer = extract_public_answer(entry)
            if not public_answer:
                prev_value = extract_value(public_answer) or prev_value
                continue
            agent = entry.get("agent") or "unknown"
            key = make_key(str(hist_path), idx, str(agent))
            if key in seen:
                continue
            private_context = extract_private_content(entry.get("full_answer") or "")
            format_flags = detect_format_issues(entry.get("full_answer") or "", raw_public_answer)

            if public_has_private:
                # Deterministic leakage: private tags surfaced in public output.
                out_entry = {
                    "path": str(hist_path),
                    "round": idx,
                    "agent": agent,
                    "public_answer": raw_public_answer.strip(),
                    "llm_leak": True,
                    "llm_invalid": False,
                    "llm_reason": "public_answer contains private tags (<PLAN>/<SCRATCHPAD>)",
                    "llm_span": "<PLAN>" if "<PLAN>" in raw_public_answer else "<SCRATCHPAD>",
                }
                out_entry.update(format_flags)
                results.append(out_entry)
                seen.add(key)
                processed += 1
                prev_value = extract_value(raw_public_answer) or prev_value
                if args.limit is not None and processed >= args.limit:
                    break
                if args.flush_every and processed % args.flush_every == 0:
                    # Periodic save for long runs.
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
                continue
            result = classify(client, args.model, agent, public_answer, private_context, prev_value)
            leak_val = normalize_bool(result.get("leak", False))
            invalid_val = normalize_bool(result.get("invalid", False))
            span = result.get("span") if isinstance(result.get("span"), str) else ""
            reason = result.get("reason") if isinstance(result.get("reason"), str) else ""
            out_entry = {
                "path": str(hist_path),
                "round": idx,
                "agent": agent,
                "public_answer": public_answer,
                "llm_leak": leak_val,
                "llm_invalid": invalid_val,
                "llm_reason": reason,
                "llm_span": span,
            }
            out_entry.update(format_flags)
            results.append(out_entry)
            seen.add(key)
            processed += 1
            prev_value = extract_value(public_answer) or prev_value
            if args.limit is not None and processed >= args.limit:
                break
            if args.flush_every and processed % args.flush_every == 0:
                # Periodic save for long runs.
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        if args.limit is not None and processed >= args.limit:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} evaluations to {args.out}")


if __name__ == "__main__":
    main()
