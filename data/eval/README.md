# Evaluation datasets

本目录只存放可版本化的离线评测数据，不存放知识库正文、Chroma、SQLite 或真实
运行输出。

| 文件 | 版本与用途 |
| --- | --- |
| `agent_eval_cases_nmgut_v1.json` | 历史 v1 cases，仅保留数据，不提供旧 CLI |
| `agent_eval_cases_v2.json` | Evaluation Baseline v2 Agent cases |
| `rag_eval_questions_nmgut_graduate_v1.json` | 旧 RAG 数据 |
| `rag_eval_questions_v2_semantic_fixed.json` | 当前 RAG gold 数据 |
| `evaluation_thresholds_v2.example.json` | 比较工具示例阈值，不是 CI 门禁 |

校验：

```bash
python scripts/run_evaluation_baseline.py --validate-only
```

新增 Agent case 时必须使用唯一 ID、`version: 2`、合法 suite/status/工具名，并在
单轮 `question` 和多轮 `turns` 中只选一种。Ground truth 必须来自当前文档、已有
gold 数据或确定性生产合同；不能凭猜测增加来源、关键词或工具序列。

详细运行、输出、比较和敏感信息边界见
[`docs/evaluation/baseline-v2.md`](../../docs/evaluation/baseline-v2.md)。
