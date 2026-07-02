from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from scripts.rag_eval.common import json_dumps


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """将逐题评测结果写入 CSV。"""
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    """将汇总指标写入 CSV。"""
    rows = []
    for group, metrics in summary.items():
        rows.append({"group": group, **metrics})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """将逐题 judge 明细写入 JSONL，便于人工排查。"""
    if not rows:
        return
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json_dumps(row) + "\n")
