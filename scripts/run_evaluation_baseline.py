"""Evaluation Baseline v2 的正式命令行入口。

用途：统一启动 Agent 与 RAG 基线评测，或只做离线数据合同校验。
输入：命令行参数、版本化评测集，以及运行时 Settings 快照。
输出：由 ``scripts.evaluation.runner`` 生成的清单、逐题结果和汇总报告。
边界：本文件只负责定位项目根目录并转交 CLI；不在入口层实现评测逻辑。
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.runner import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
