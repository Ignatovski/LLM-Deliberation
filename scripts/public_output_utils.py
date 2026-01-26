#!/usr/bin/env python3
"""
Shared utilities for deterministic (non-LLM) public-output checks.

These helpers are used by:
  - scripts/check_unclosed_answer_leakage.py
  - scripts/check_invalid_public_values.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Only parse numeric suggestions inside <VALUE> tags.
VALUE_TAG_RE = re.compile(r"<VALUE>\s*([-+]?\d+(?:\.\d+)?)\s*</VALUE>", re.IGNORECASE)
# Treat any public_answer containing private tags as leakage (public shows private tags).
PRIVATE_TAG_RE = re.compile(r"<\s*(SCRATCHPAD|PLAN)\b", re.IGNORECASE)
ANSWER_OPEN_RE = re.compile(r"<\s*ANSWER\s*>", re.IGNORECASE)
ANSWER_CLOSE_RE = re.compile(r"</\s*ANSWER\s*>", re.IGNORECASE)


def extract_tagged_block(text: str, tag: str) -> str:
    """Pull a tag block like <ANSWER>...</ANSWER>."""
    if not text:
        return ""
    pattern = re.compile(rf"<\s*{tag}\s*>(.*?)</\s*{tag}\s*>", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return ""


def extract_public_answer(entry: Dict[str, Any]) -> str:
    """
    Prefer clean public_answer; otherwise fall back to the <ANSWER> block
    inside full_answer.
    """
    pa = entry.get("public_answer")
    if isinstance(pa, str) and pa.strip() and not PRIVATE_TAG_RE.search(pa):
        return pa.strip()

    full_answer = entry.get("full_answer") or ""
    answer_block = extract_tagged_block(full_answer, "ANSWER")
    if answer_block:
        return answer_block.strip()

    return pa.strip() if isinstance(pa, str) else ""


def public_has_private_tags(text: str) -> bool:
    return bool(PRIVATE_TAG_RE.search(text or ""))


def unclosed_answer_tag(full_answer: str) -> bool:
    full = full_answer or ""
    return bool(ANSWER_OPEN_RE.search(full)) and not bool(ANSWER_CLOSE_RE.search(full))


def parse_value_tags(text: str) -> Tuple[List[int], bool]:
    """
    Returns (ints, has_non_integer_value).
      - ints: list of successfully parsed ints from <VALUE> tags
      - has_non_integer_value: True if a <VALUE> tag exists but is not an int (e.g. 1.1)
    """
    if not text:
        return [], False
    matches = VALUE_TAG_RE.findall(text)
    ints: List[int] = []
    has_non_int = False
    for raw in matches:
        raw = raw.strip()
        try:
            if re.fullmatch(r"[-+]?\d+", raw):
                ints.append(int(raw))
            else:
                # It looked numeric but not an integer (e.g. 1.1)
                float(raw)  # validate numeric
                has_non_int = True
        except ValueError:
            # Not even numeric; ignore here.
            has_non_int = True
    return ints, has_non_int


def load_history(path: Path) -> Optional[Dict[str, Any]]:
    """Read one history JSON file."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def parse_outputs_by_category(path: Path, root: Optional[Path] = None) -> List[Path]:
    """Parse outputs_by_category.txt into history*.json paths."""
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
    seen: Set[str] = set()
    for h in histories:
        key = str(h.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(h)
    return unique


def make_key(path: str, round_idx: int, agent: str) -> str:
    return f"{path}|{round_idx}|{agent}"


def load_existing_results(out_path: Path, overwrite: bool) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Resume helper: if output exists, load it and return (results, seen_keys)."""
    if overwrite or not out_path.exists():
        return [], set()
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], set()
    if not isinstance(data, list):
        return [], set()
    seen: Set[str] = set()
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


def flush_results(out_path: Path, results: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

