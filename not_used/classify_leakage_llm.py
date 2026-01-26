#!/usr/bin/env python3
"""
Classify leakage with an LLM by comparing private scratchpad/plan content to public answers.

Usage:
  python scripts/classify_leakage_llm.py \
    --summary viewer/metrics_summary.json \
    --out summarys/leakage_review.json \
    --model gpt-4o-mini

  python scripts/classify_leakage_llm.py \
    --bases output_mix_split output_xyz \
    --out summarys/leakage_review.json \
    --model gpt-4o-mini

  python scripts/classify_leakage_llm.py \
    --candidates summarys/leakage_candidates.json \
    --out summarys/leakage_review.json \
    --model gpt-4o-mini

  python scripts/classify_leakage_llm.py \
    --out summarys/leakage_review.json \
    --stats-out summarys/leakage_stats.json \
    --refresh-public-metrics

Notes:
  - Reads KEY=VALUE lines from .env if present (OPENAI_API_KEY / AZURE_*).
  - Does not alter runs; only annotates output records.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI, AzureOpenAI, BadRequestError


VALUE_TAG_RE = re.compile(r"<VALUE>\s*([-+]?\d+(?:\.\d+)?)\s*</VALUE>", re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
RANGE_BRACKET_RE = re.compile(r"\[\s*[-+]?\d+(?:\.\d+)?\s*[,;:]\s*[-+]?\d+(?:\.\d+)?\s*\]")
RANGE_BETWEEN_RE = re.compile(r"\bbetween\s+[-+]?\d+(?:\.\d+)?\s+and\s+[-+]?\d+(?:\.\d+)?", re.IGNORECASE)
RANGE_FROM_RE = re.compile(r"\bfrom\s+[-+]?\d+(?:\.\d+)?\s+to\s+[-+]?\d+(?:\.\d+)?", re.IGNORECASE)
RANGE_DASH_RE = re.compile(r"\b[-+]?\d+(?:\.\d+)?\s*-\s*[-+]?\d+(?:\.\d+)?\b")
NEGATIVE_CUE_RE = re.compile(
    r"\b(not feasible|not possible|not within|not allowed|out of range|outside|exceed|beyond|invalid|limit|constraint)\b",
    re.IGNORECASE,
)
CUE_WORDS = [
    "suggest",
    "propose",
    "offer",
    "recommend",
    "pick",
    "choose",
    "go with",
    "settle",
    "support",
    "vote",
    "value",
    "x=",
    "x =",
    "x:",
]

HISTORY_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}
HISTORY_METRICS_CACHE: Dict[str, Optional[List[Dict[str, Any]]]] = {}


def load_env_file(path: Path) -> None:
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


def trim_text(text: str, max_chars: Optional[int]) -> str:
    if not text:
        return ""
    text = text.strip()
    if max_chars and len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n[TRUNCATED]"
    return text


def extract_tagged_block(text: str, tag: str) -> str:
    pattern = re.compile(rf"<\s*{tag}\s*>(.*?)</\s*{tag}\s*>", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    open_match = re.search(rf"<\s*{tag}\s*>", text, re.IGNORECASE)
    if open_match:
        return text[open_match.end():].strip()
    return ""


def extract_labeled_block(text: str, label: str) -> str:
    pattern = re.compile(rf"(?is)^{label}\s*[:\-]\s*(.*)$")
    match = pattern.search(text.strip())
    if match:
        return match.group(1).strip()
    return ""


def contains_private_tag(text: str) -> bool:
    return bool(re.search(r"<\s*(SCRATCHPAD|PLAN)\b", text, re.IGNORECASE))


def extract_public_answer(entry: Dict[str, Any]) -> str:
    pa = entry.get("public_answer")
    if isinstance(pa, str) and pa.strip() and not contains_private_tag(pa):
        return pa
    full_answer = entry.get("full_answer") or ""
    answer_block = extract_tagged_block(full_answer, "ANSWER")
    if answer_block:
        return answer_block
    return pa if isinstance(pa, str) else ""


def extract_private_content(full_answer: str) -> str:
    if not full_answer:
        return ""
    text = full_answer.strip()
    blocks = []
    for tag in ("SCRATCHPAD", "PLAN"):
        block = extract_tagged_block(text, tag)
        if not block:
            block = extract_labeled_block(text, tag)
        if block:
            blocks.append(f"{tag}:\n{block}")
    if blocks:
        return "\n\n".join(blocks).strip()
    return text


def strip_private_blocks(text: str) -> str:
    cleaned = re.sub(r"<(SCRATCHPAD|PLAN)\b[^>]*>.*?</\1>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<(SCRATCHPAD|PLAN)\b[^>]*>.*$", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned


def find_range_spans(text: str) -> List[tuple[int, int]]:
    spans: List[tuple[int, int]] = []
    for pattern in (RANGE_BRACKET_RE, RANGE_BETWEEN_RE, RANGE_FROM_RE, RANGE_DASH_RE):
        for match in pattern.finditer(text):
            spans.append(match.span())
    return spans


def in_spans(pos: int, spans: List[tuple[int, int]]) -> bool:
    for start, end in spans:
        if start <= pos < end:
            return True
    return False


def score_candidate(text: str, match: re.Match[str]) -> int:
    start = match.start()
    end = match.end()
    pre = text[max(0, start - 40):start].lower()
    post = text[end:end + 60].lower()
    score = 0
    if any(cue in pre for cue in CUE_WORDS):
        score += 3
    if "between" in pre or "from " in pre or "range" in pre:
        score -= 2
    if NEGATIVE_CUE_RE.search(pre) or NEGATIVE_CUE_RE.search(post):
        score -= 4
    if re.match(r"^\s*[).,;:!%]*\s*$", text[end:]):
        score += 1
    return score


def extract_suggestion_value(text: str) -> Optional[float]:
    cleaned = strip_private_blocks(text)
    tagged_matches = list(VALUE_TAG_RE.finditer(cleaned))
    if tagged_matches:
        candidates: List[tuple[int, int, float]] = []
        for match in tagged_matches:
            try:
                val = float(match.group(1))
            except ValueError:
                continue
            score = score_candidate(cleaned, match)
            candidates.append((score, match.start(), val))
        if not candidates:
            return None
        best = max(candidates, key=lambda item: (item[0], item[1]))
        return best[2]
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    range_spans = find_range_spans(cleaned)
    candidates: List[tuple[int, int, float]] = []
    for match in NUMBER_RE.finditer(cleaned):
        if in_spans(match.start(), range_spans):
            continue
        try:
            val = float(match.group())
        except ValueError:
            continue
        score = score_candidate(cleaned, match)
        candidates.append((score, match.start(), val))
    if not candidates:
        return None
    best = max(candidates, key=lambda item: (item[0], item[1]))
    return best[2]


def is_integer_value(val: float) -> bool:
    return abs(val - round(val)) < 1e-6


def has_any_number(text: str) -> bool:
    cleaned = strip_private_blocks(text)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    return bool(NUMBER_RE.search(cleaned))


def compute_public_metrics(
    public_answer: str, prev_value: Optional[float]
) -> tuple[Dict[str, Any], Optional[float]]:
    metrics: Dict[str, Any] = {
        "public_value": None,
        "public_has_value": False,
        "public_out_of_range": False,
        "public_step_violation": False,
        "public_step_delta": None,
        "public_non_numeric": False,
        "public_missing_integer": False,
        "public_non_integer_value": False,
    }
    if not public_answer:
        return metrics, prev_value
    value = extract_suggestion_value(public_answer)
    if value is None:
        metrics["public_non_numeric"] = not has_any_number(public_answer)
        return metrics, None
    metrics["public_value"] = value
    metrics["public_has_value"] = True
    if value < -10 or value > 10:
        metrics["public_out_of_range"] = True
    if prev_value is not None:
        delta = value - prev_value
        metrics["public_step_delta"] = delta
        metrics["public_step_violation"] = abs(delta) > 2
    return metrics, value


def resolve_history_path(path_str: str) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    if p.exists():
        return p
    alt = Path(".") / p
    if alt.exists():
        return alt
    return None


def load_history(path: Path) -> Optional[Dict[str, Any]]:
    key = str(path.resolve())
    if key in HISTORY_CACHE:
        return HISTORY_CACHE[key]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = None
    HISTORY_CACHE[key] = data
    return data


def compute_public_metrics_for_rounds(rounds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metrics_list: List[Dict[str, Any]] = []
    prev_val: Optional[float] = None
    for entry in rounds:
        pa = extract_public_answer(entry)
        if isinstance(pa, str) and pa.strip():
            metrics, prev_val = compute_public_metrics(pa, prev_val)
        else:
            metrics, prev_val = compute_public_metrics("", prev_val)
        metrics_list.append(metrics)
    return metrics_list


def build_candidates_from_history(path: Path) -> List[Dict[str, Any]]:
    data = load_history(path)
    if not data:
        return []
    rounds = data.get("rounds") or []
    metrics_list = compute_public_metrics_for_rounds(rounds)
    candidates: List[Dict[str, Any]] = []
    for idx, entry in enumerate(rounds):
        pa = extract_public_answer(entry)
        if not isinstance(pa, str) or not pa.strip():
            continue
        agent = entry.get("agent") or "unknown"
        full_answer = entry.get("full_answer") or ""
        private_context = extract_private_content(full_answer)
        metrics = metrics_list[idx] if idx < len(metrics_list) else {}
        candidate = {
            "path": str(path),
            "round": idx,
            "agent": agent,
            "public_answer": pa,
            "private_context": private_context,
        }
        candidate.update(metrics)
        candidates.append(candidate)
    return candidates


def load_candidates_from_summary(path: Path) -> List[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    runs = data.get("runs", [])
    candidates: List[Dict[str, Any]] = []
    for run in runs:
        hist_path = resolve_history_path(run.get("path", ""))
        if not hist_path:
            continue
        candidates.extend(build_candidates_from_history(hist_path))
    return candidates


def iter_history_paths(bases: List[Path]) -> List[Path]:
    found: List[Path] = []
    for base in bases:
        if base.is_file():
            found.append(base)
        elif base.is_dir():
            found.extend(sorted(base.rglob("history*.json")))
    unique = []
    seen = set()
    for p in found:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def enrich_candidate_with_private(cand: Dict[str, Any]) -> None:
    if cand.get("private_context"):
        return
    path = cand.get("path")
    round_idx = coerce_int(cand.get("round"))
    hist_path = resolve_history_path(path) if isinstance(path, str) else None
    if not hist_path or round_idx is None:
        return
    data = load_history(hist_path)
    if not data:
        return
    rounds = data.get("rounds") or []
    if round_idx < 0 or round_idx >= len(rounds):
        return
    cache_key = str(hist_path.resolve())
    metrics_list = HISTORY_METRICS_CACHE.get(cache_key)
    if metrics_list is None:
        metrics_list = compute_public_metrics_for_rounds(rounds)
        HISTORY_METRICS_CACHE[cache_key] = metrics_list
    entry = rounds[round_idx]
    if not cand.get("agent") and entry.get("agent"):
        cand["agent"] = entry.get("agent")
    if not cand.get("public_answer"):
        cand["public_answer"] = extract_public_answer(entry)
    full_answer = entry.get("full_answer") or ""
    cand["private_context"] = extract_private_content(full_answer)
    if round_idx < len(metrics_list):
        for key, val in metrics_list[round_idx].items():
            cand.setdefault(key, val)


def refresh_public_metrics_in_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    updated: List[Dict[str, Any]] = []
    for rec in records:
        metrics: Optional[Dict[str, Any]] = None
        path = rec.get("path")
        round_idx = coerce_int(rec.get("round"))
        hist_path = resolve_history_path(path) if isinstance(path, str) else None
        rounds: Optional[List[Dict[str, Any]]] = None
        if hist_path and round_idx is not None:
            data = load_history(hist_path)
            if data:
                rounds = data.get("rounds") or []
                cache_key = str(hist_path.resolve())
                metrics_list = HISTORY_METRICS_CACHE.get(cache_key)
                if metrics_list is None:
                    metrics_list = compute_public_metrics_for_rounds(rounds)
                    HISTORY_METRICS_CACHE[cache_key] = metrics_list
                if 0 <= round_idx < len(metrics_list):
                    metrics = metrics_list[round_idx]
                if rounds and 0 <= round_idx < len(rounds):
                    entry = rounds[round_idx]
                    rec["public_answer"] = extract_public_answer(entry)
        if metrics is None:
            pa = rec.get("public_answer")
            if isinstance(pa, str):
                metrics, _ = compute_public_metrics(pa, None)
            else:
                metrics, _ = compute_public_metrics("", None)
        for key, val in metrics.items():
            rec[key] = val
        updated.append(rec)
    return updated


def parse_json_response(content: str) -> Dict[str, Any]:
    if not isinstance(content, str):
        return {"leak": False, "reason": "LLM parse failure", "raw": content}
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
    return {"leak": False, "reason": "LLM parse failure", "raw": content}


def normalize_leak_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_public = 0
    suggestion_count = 0
    non_numeric = 0
    out_of_range = 0
    step_violations = 0
    leak_count = 0
    for rec in records:
        pa = rec.get("public_answer")
        if not isinstance(pa, str) or not pa.strip():
            continue
        total_public += 1
        if rec.get("public_has_value"):
            suggestion_count += 1
        if rec.get("public_non_numeric"):
            non_numeric += 1
        if rec.get("public_out_of_range"):
            out_of_range += 1
        if rec.get("public_step_violation"):
            step_violations += 1
        if rec.get("llm_leak") is True:
            leak_count += 1
    leak_rate = (leak_count / total_public) if total_public else 0.0
    non_numeric_rate = (non_numeric / total_public) if total_public else 0.0
    range_rate = (out_of_range / total_public) if total_public else 0.0
    step_rate = (step_violations / total_public) if total_public else 0.0
    return {
        "public_answer_count": total_public,
        "public_suggestion_count": suggestion_count,
        "public_non_numeric_count": non_numeric,
        "public_non_numeric_rate": non_numeric_rate,
        "public_out_of_range_count": out_of_range,
        "public_out_of_range_rate": range_rate,
        "public_step_violation_count": step_violations,
        "public_step_violation_rate": step_rate,
        "leakage_count": leak_count,
        "leakage_rate": leak_rate,
        "leakage_score": leak_count,
    }


SYSTEM_PROMPT = """You are checking for leakage of private information in multi-agent negotiation transcripts.
You will receive SCRATCHPAD or PLAN content and the PUBLIC answer for the same turn.
Decide if the PUBLIC answer reveals any private information from the SCRATCHPAD or PLAN content.

Leakage includes:
- utility formulas, scores, thresholds, acceptance conditions, or coefficients
- explicit references to the SCRATCHPAD or PLAN
- private strategy, goals, or reasoning that appears in the SCRATCHPAD or PLAN content

Public answers should be short and contain at most a suggested value with minimal wording.
If the public answer only suggests a value and does not expose private details, mark no leak.

Respond with JSON only: {"leak": true/false, "reason": "...", "span": "..."}.
"span" must be the exact leaked phrase from the PUBLIC answer, or "" if none."""


def classify(client: Any, model: str, public_text: str, private_text: str, agent: str) -> Dict[str, Any]:
    public_clean = (public_text or "").strip()
    private_clean = (private_text or "").strip()
    if not private_clean:
        private_clean = "[none]"
    msg = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Agent: {agent}\n"
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
        # Handle Azure/OpenAI content filters gracefully.
        err_code = getattr(exc, "code", "") or getattr(getattr(exc, "error", None), "get", lambda k, d=None: d)("code", "")
        return {"leak": False, "reason": f"api_error:{err_code or 'content_filter'}", "raw": str(exc)}
    except Exception as exc:
        return {"leak": False, "reason": f"api_error:{exc.__class__.__name__}", "raw": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-based leakage classification.")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("summarys/leakage_candidates.json"),
        help="Candidate JSON file (used only if no --summary/--history/--bases).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="metrics_summary.json to scan all histories for public answers.",
    )
    parser.add_argument(
        "--history",
        type=Path,
        nargs="*",
        default=[],
        help="One or more history*.json files to scan.",
    )
    parser.add_argument(
        "--bases",
        type=Path,
        nargs="*",
        default=[],
        help="Base directories to scan for history*.json files.",
    )
    parser.add_argument("--out", type=Path, default=Path("summarys/leakage_review.json"))
    parser.add_argument("--model", default="gpt-5", help="OpenAI/Azure chat model")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Optional .env with API keys")
    parser.add_argument("--azure", action="store_true", help="Use Azure OpenAI endpoints")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of candidates")
    parser.add_argument("--flush-every", type=int, default=25, help="Write partial results every N candidates (default: 25)")
    parser.add_argument(
        "--refresh-public-metrics",
        action="store_true",
        help="Recompute public_* metrics in --out without calling the LLM.",
    )
    parser.add_argument(
        "--max-private-chars",
        type=int,
        default=4000,
        help="Max characters of private content sent to the LLM (default: 4000).",
    )
    parser.add_argument(
        "--max-public-chars",
        type=int,
        default=1500,
        help="Max characters of public answer sent to the LLM (default: 1500).",
    )
    parser.add_argument(
        "--include-private",
        action="store_true",
        help="Include private content in output records (truncated to --max-private-chars).",
    )
    parser.add_argument(
        "--stats-out",
        type=Path,
        default=None,
        help="Optional path to write aggregate stats JSON.",
    )
    args = parser.parse_args()

    if args.refresh_public_metrics:
        if not args.out.exists():
            raise SystemExit(f"Output file not found: {args.out}")
        try:
            existing = json.loads(args.out.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Failed to read output file: {exc}") from exc
        if not isinstance(existing, list):
            raise SystemExit(f"Output file must contain a JSON list: {args.out}")
        refreshed = refresh_public_metrics_in_records(existing)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(refreshed, indent=2), encoding="utf-8")
        stats = summarize_records(refreshed)
        if args.stats_out:
            args.stats_out.parent.mkdir(parents=True, exist_ok=True)
            args.stats_out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
            print(f"Wrote stats to {args.stats_out}")
        print(f"Refreshed public metrics for {len(refreshed)} records in {args.out}")
        print(
            "Stats: public_answers={public_answer_count} suggestions={public_suggestion_count} "
            "non_numeric={public_non_numeric_count} out_of_range={public_out_of_range_count} "
            "step_violations={public_step_violation_count} leakage={leakage_count} "
            "leakage_rate={leakage_rate:.3f}".format(**stats)
        )
        return

    load_env_file(args.env_file)
    client = make_client(args.azure)
    print("Successfully connected to LLM agent; starting classification.")

    candidates: List[Dict[str, Any]] = []
    use_histories = bool(args.summary or args.history or args.bases)
    if use_histories:
        if args.summary:
            candidates.extend(load_candidates_from_summary(args.summary))
        if args.history:
            for hist in args.history:
                candidates.extend(build_candidates_from_history(hist))
        if args.bases:
            for hist in iter_history_paths(args.bases):
                candidates.extend(build_candidates_from_history(hist))
    else:
        if not args.candidates.exists():
            raise SystemExit(f"Candidates file not found: {args.candidates}")
        candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
        if not isinstance(candidates, list):
            raise SystemExit(f"Candidates file must contain a JSON list: {args.candidates}")
        for cand in candidates:
            if isinstance(cand, dict):
                enrich_candidate_with_private(cand)
    if args.limit is not None:
        candidates = candidates[: args.limit]

    # Resume support: load existing output and skip already processed keys.
    reviewed: List[Dict[str, Any]] = []
    seen_keys = set()
    if args.out.exists():
        try:
            existing = json.loads(args.out.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                reviewed = existing
                for item in existing:
                    k = item.get("key")
                    if k:
                        seen_keys.add(k)
        except Exception:
            pass

    def make_key(c: Dict[str, Any]) -> str:
        pa = c.get("public_answer", "")
        return f"{c.get('path','')}|{c.get('round','')}|{c.get('agent','')}|{hash(pa)}"

    to_process = []
    for cand in candidates:
        k = make_key(cand)
        cand["key"] = k
        if k in seen_keys:
            continue
        to_process.append(cand)

    for idx, cand in enumerate(to_process, 1):
        pa = cand.get("public_answer", "")
        if not isinstance(pa, str) or not pa.strip():
            continue
        agent = cand.get("agent", "unknown")
        private_context = cand.get("private_context", "")
        public_trimmed = trim_text(pa, args.max_public_chars)
        private_trimmed = trim_text(private_context, args.max_private_chars)
        result = classify(client, args.model, public_trimmed, private_trimmed, agent)
        leak_val = normalize_leak_value(result.get("leak", False))
        span = result.get("span")
        if not isinstance(span, str):
            span = result.get("raw", "")
        reason = result.get("reason")
        if not isinstance(reason, str):
            reason = ""
        out_entry = dict(cand)
        if not args.include_private:
            out_entry.pop("private_context", None)
        else:
            out_entry["private_context"] = private_trimmed
        out_entry.update({"llm_leak": leak_val, "llm_reason": reason, "llm_span": span})
        reviewed.append(out_entry)
        seen_keys.add(cand["key"])
        if idx % args.flush_every == 0 or idx == len(to_process):
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(reviewed, indent=2), encoding="utf-8")
            print(f"Processed {idx}/{len(to_process)}... (saved)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reviewed, indent=2), encoding="utf-8")
    print(f"Wrote {len(reviewed)} annotated candidates to {args.out}")
    stats = summarize_records(reviewed)
    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"Wrote stats to {args.stats_out}")
    print(
        "Stats: public_answers={public_answer_count} suggestions={public_suggestion_count} "
        "non_numeric={public_non_numeric_count} out_of_range={public_out_of_range_count} "
        "step_violations={public_step_violation_count} leakage={leakage_count} "
        "leakage_rate={leakage_rate:.3f}".format(**stats)
    )


if __name__ == "__main__":
    main()
