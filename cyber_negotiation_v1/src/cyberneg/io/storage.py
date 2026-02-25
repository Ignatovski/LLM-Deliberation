from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_ts_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, data: Any) -> Path:
    p = ensure_dir(Path(path).parent) / Path(path).name
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    p = ensure_dir(Path(path).parent) / Path(path).name
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


def write_text(path: str | Path, text: str) -> Path:
    p = ensure_dir(Path(path).parent) / Path(path).name
    p.write_text(text, encoding="utf-8")
    return p


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    p = ensure_dir(Path(path).parent) / Path(path).name
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return p


def prepare_run_dir(output_root: str | Path, experiment_id: str, condition_id: str, scenario_id: str, run_id: str) -> Path:
    root = ensure_dir(output_root)
    ts_root = ensure_dir(root / experiment_id / utc_ts_compact())
    return ensure_dir(ts_root / condition_id / scenario_id / run_id)


def write_latest_pointer(output_root: str | Path, target_dir: str | Path) -> Path:
    p = ensure_dir(output_root) / "latest_mock_run"
    p.write_text(str(Path(target_dir).resolve()), encoding="utf-8")
    return p

