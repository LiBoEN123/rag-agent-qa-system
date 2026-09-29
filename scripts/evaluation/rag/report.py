"""RAG 评测专用的 CSV 写入工具。

用途：把已计算好的逐题结果和汇总指标落盘，方便复核与归档。
输入：目标路径，以及字段已经固定的结果字典列表或汇总字典。
输出：UTF-8/UTF-8-BOM 编码的 CSV 文件。
边界：只负责序列化，不加载输入集、不判分，也不调用模型或网络。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List


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
