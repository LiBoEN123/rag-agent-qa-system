"""Evaluation Baseline v2 的运行结果对比入口。

用途：比较基线运行与候选运行的指标、逐题变化和回归阈值。
输入：两个完整的评测输出目录，以及可选的阈值配置。
输出：结构化差异文件、Markdown 报告，并以退出码表达阈值是否通过。
边界：本文件只装配命令行入口；对比规则由 ``scripts.evaluation.comparison`` 维护。
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.comparison import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
