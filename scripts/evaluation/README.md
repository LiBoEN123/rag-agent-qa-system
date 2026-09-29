# Evaluation 启动说明

本目录是可导入实现。正式 CLI 为 `scripts/run_evaluation_baseline.py` 和
`scripts/compare_evaluation_runs.py`，从仓库根目录执行。
完整合同、指标和产物见 [Evaluation Baseline](../../docs/evaluation/baseline-v2.md)。

## 选择运行方式

| 目的 | 参数 | 调用真实模型 | 真实 Web |
| --- | --- | --- | --- |
| 校验 Agent 与 65 题 RAG 数据 | `--validate-only` | 否 | 否 |
| Core Agent + RAG 基线 | 默认 | 是 | 否 |
| 只运行 Agent 题集 | `--agent-only` | 是 | 默认否 |
| 只运行 RAG 题集 | `--rag-only` | 是 | 否 |
| 包含 Online Web case | `--agent-only --include-online` | 是 | 是 |
| 可选语义 Judge | `--rag-only --use-llm-judge` | 是 | 否 |
| 比较两次已有运行 | `compare_evaluation_runs.py` | 否 | 否 |

Agent-only 的知识库问题仍会检索。RAG-only 同样经过生产 Agent，只是不执行 Agent case 题集。
`--include-online` 只影响 Agent case；`--use-llm-judge` 只影响 RAG 答案评分。
评测 Settings 始终禁用 MCP，保留生产 Source/ingestion 的独立能力。

## 离线检查

```bash
.venv/bin/python scripts/run_evaluation_baseline.py --validate-only
.venv/bin/python scripts/run_evaluation_baseline.py --help
```

validate-only 检查 schema、ID、工具名和覆盖率，不创建输出目录、不加载模型或索引。

## 正式基线

需要已配置模型、已有本地索引、Embedding/Rerank 缓存、干净工作区与全新输出目录。

```bash
.venv/bin/python scripts/run_evaluation_baseline.py \
  --output-dir outputs/evaluation_baseline_v2/core-run \
  --retrieval-k 5
```

默认使用 `agent_eval_cases_v2.json`（25 Core + 3 Online）和
`rag_eval_questions_v2_semantic_fixed.json`（65 题）。Online 默认跳过。
`--allow-dirty` 只用于带 dirty 标记的实验；`--force-rebuild` 会写索引，正常评测不需要。

限定完整 Agent case：

```bash
.venv/bin/python scripts/run_evaluation_baseline.py \
  --agent-only --case-id D008 --case-id agent-followup-001 \
  --output-dir outputs/evaluation_baseline_v2/targeted-run
```

使用 `--agent-cases` / `--rag-eval-file` 可指定数据。表格专项题集与
`table_diagnostics.py` 属于 diagnostics，Runner 没有表格专属逻辑，不计入主 Baseline。

## 评分与结果

Agent 检查最终 action/status、必要工具、答案关键词、来源和本轮成功 KB evidence grounding；
仅 multi-tool/recovery case 约束工具顺序。多轮按每轮最终结果评分，summary 只按
suite/category 分组。指标定义与适用 turns 分母见 [Baseline 合同](../../docs/evaluation/baseline-v2.md)。

RAG 正式输入只支持 Prediction v3：`retrieved_chunk_ids` 为有序去重 KB 观察，
`reference_chunk_ids` 为最终 grounded 引用。`--retrieval-k` 仅截取观察前 K 项评分，
不会变成 Agent 工具参数。

headline 指标为 prediction coverage、retrieval Gold Recall@K、reference Gold Recall，
以及可选的 `avg_answer_correctness`。`llm_judge_coverage` 单独表示判分覆盖率。
旧 Prediction v1/v2 不再读取。

批量 runtime failure 单独进入 `rag/prediction_failures.jsonl`；对应缺失预测降低
coverage、召回计零，不调用 Judge。Judge 关闭、失败或 prediction 缺失时，
`answer_correctness=None`；均值只计算成功判分的记录，无结果时为 `None`。结构异常仍 fail closed。

`deepeval==3.9.6` 承担确定性 Tool Correctness 与可选 GEval Answer Correctness。
Tool Correctness 使用禁用模型的占位，不调用模型；GEval 使用固定评价步骤，读取问题、
实际回答与 Gold 规范（expected behavior、参考答案、关键点、forbidden claims），
通过项目 Judge factory/Gateway 调用已配置模型。无需 Cloud 登录，telemetry/tracing 关闭，
框架文件系统只读，不启动其 pytest runner。

输出包括 manifest、覆盖矩阵、summary、逐题 JSONL/CSV、失败报告和 Markdown。
数据、工具合同、Prompt 与索引 hash 用于审查可复现性。

```bash
.venv/bin/python scripts/compare_evaluation_runs.py \
  --baseline outputs/evaluation_baseline_v2/run-a \
  --candidate outputs/evaluation_baseline_v2/run-b
```

Online 内容与外部服务会变化，应与 Core 分开解释。模型运行失败时检查失败 sidecar；
不能把部分结果作为完整正式基线，也不能用旧 workflow 分数直接归因当前改动。
