# Evaluation datasets

本目录保存版本化评测数据，不存放知识库正文、索引、数据库或运行输出。

| 文件 | 当前用途 |
| --- | --- |
| `agent_eval_cases_v2.json` | 25 个 Core Agent case + 3 个可选 Online Web case |
| `rag_eval_questions_v2_semantic_fixed.json` | 主 RAG 数据集，65 题，生成 Prediction Schema v3 |
| `rag_eval_table_v1.json` | 独立表格 diagnostics，14 个受控问题，不参与主 Baseline |
| `performance_cases.json` | HTTP 性能测试输入，不参与质量 Baseline |
| `evaluation_thresholds_v2.example.json` | 比较工具示例阈值，不是 CI 门禁 |

离线校验 Agent 与 RAG 数据：

```bash
.venv/bin/python scripts/run_evaluation_baseline.py --validate-only
```

Agent case 必须使用唯一 ID、`version: 2`、`core` 或 `online` suite、合法 status/工具名，
单轮 `question` 和多轮 `turns` 只能选择一种。RAG 题目必须有非空且唯一的 qid 和问题。
Ground truth 必须来自已核验文档或确定性生产合同。

主指标为 prediction coverage、retrieval Gold Recall@K、reference Gold Recall 与可选
Judge 的 key point recall。运行与产物说明见
[Evaluation Baseline](../../docs/evaluation/baseline-v2.md)。
