# Evaluation Baseline v2

统一入口是 `scripts/run_evaluation_baseline.py`；比较入口是
`scripts/compare_evaluation_runs.py`。实现位于 `scripts/evaluation/`，不改变生产
Agent、RAG、工具或 API 行为。Baseline/Agent 数据集版本仍为 v2；RAG Prediction
输入合同固定为 **Schema v3**。

## 当前评测层与数据

| 层 | 数据集 | 默认行为 |
| --- | --- | --- |
| Core Agent | `data/eval/agent_eval_cases_v2.json` 中 25 个 Core case | 执行 |
| RAG | `data/eval/rag_eval_questions_v2_semantic_fixed.json`，65 题 | 执行 |
| Online Web | Agent 数据集中的 3 个 `web_search`、`web_fetch`、`web_pdf_fetch` case | 仅 `--include-online` 执行 |
| 可选 LLM Judge | GEval Answer Correctness：答案与 Gold 规范 | 仅 `--use-llm-judge` 启用 |

Agent case 必须有唯一 ID、`version: 2`，suite 只接受 `core` 或 `online`。
单轮 `question` 和至少两轮的 `turns` 只能选择一种。状态来自 `AgentResultStatus`，
工具名必须属于当前本地 ToolSpec catalog。

`rag_eval_table_v1.json` 与 `table_diagnostics.py` 是独立的专项 diagnostics，
不参与主 Baseline，Runner 没有导入或自动运行它们。历史受控结果见
[表格诊断记录](table-aware-eval.md)，不能作为当前 65 题基线。

## 离线校验

从仓库根目录执行：

```bash
.venv/bin/python scripts/run_evaluation_baseline.py --validate-only
```

校验 Agent schema、ID、注册工具与覆盖率，以及 RAG 题目格式和唯一 qid，并报告
`rag_questions: 65`。不会初始化模型、RAG、工具运行时或访问网络，也不创建输出目录。

## 真实运行

以下命令会调用已配置模型，要求已有可用索引。默认执行 Core Agent 与 RAG，
Online 与 Judge 默认关闭：

```bash
.venv/bin/python scripts/run_evaluation_baseline.py \
  --output-dir outputs/evaluation_baseline_v2/core-run \
  --retrieval-k 5
```

使用 `--agent-only` 或 `--rag-only` 只运行相应题集。Agent-only 的知识库问题仍会
通过 Agent 调用检索。限定 Agent case 时，使用可重复的 `--case-id`，保持多轮 case 完整：

```bash
.venv/bin/python scripts/run_evaluation_baseline.py \
  --agent-only --case-id D008 \
  --output-dir outputs/evaluation_baseline_v2/d008-diagnostic
```

这类运行的 manifest 标记为 `targeted_agent_check`，不是完整基线。

显式启用 Online Web 或 RAG Judge：

```bash
.venv/bin/python scripts/run_evaluation_baseline.py \
  --agent-only --include-online \
  --output-dir outputs/evaluation_baseline_v2/online-run

.venv/bin/python scripts/run_evaluation_baseline.py \
  --rag-only --use-llm-judge --judge-model deepseek-chat \
  --output-dir outputs/evaluation_baseline_v2/rag-judge-run
```

正式运行要求工作区干净且输出目录为空。`--allow-dirty` 只生成带 dirty 标记的实验结果。
`--force-rebuild` 会重建本地索引；通常使用既有索引即可。

## 执行与隔离

Agent 复用真实 `ChatService.stream()`，注入独立内存 session 与无写入的 metrics/
bad-case writer。评测 Settings 固定禁用 MCP 与 tool audit，备用路径指向本轮输出目录。
Core 权限只有 `knowledge_base:read`；Online case 额外获得 `web:search`、`web:fetch`。
生产 MCP Source/ingestion 能力不受此评测边界影响。

工具输出捕获只保留字段名、调用状态和安全证据身份；不把捕获的完整正文写入报告。
正文题的 `valid_tool_paths` 必须包含成功返回正文 evidence 的工具，最终引用须来自本轮
真实 evidence。发现文档目录不能替代正文证据。比较题仍核对双方正文和最终来源。

工具包含关系和完整序列由固定的 `deepeval==3.9.6` `ToolCorrectnessMetric`
确定性评分，不调用 Judge。首工具备选、状态、来源、关键词和 Query Rewrite 由项目检查。
恢复 case 用等长的 `tool_sequence` / `tool_status_sequence` 明确预期失败与成功。

## Prediction v3 与指标

RAG Prediction 复用 `ChatService.stream → stream_agent → create_agent`，每题空 session，
使用 `RAG_ONLY` 与 `knowledge_base:read`。每轮 query/top_k 由 Agent 自主决定。

| 字段 | 语义 |
| --- | --- |
| `retrieved_chunk_ids` | 成功 KB evidence 工具输出，按实际观察顺序稳定去重 |
| `reference_chunk_ids` | 最终通过 grounding 的引用 chunk IDs |
| `status` | 合法答案、无证据或澄清状态 |
| `retrieval_call_count` / `tool_call_count` / `llm_call_count` | 实际调用次数 |

生成器保存上述 ID 的数量及安全元数据。CSV ID 为逗号分隔整数，JSON/JSONL 也可用整数列表。
评分器只接受显式 v3，拒绝无版本、v1/v2 及缺失 evidence 字段的输入。

正式 headline metrics：

| 指标 | 定义 |
| --- | --- |
| `prediction_coverage` | 成功生成预测的题数 / 全部题数 |
| `retrieval_gold_chunk_recall_at_k` | 前 K 个去重观察覆盖的 Gold chunk 比例 |
| `reference_gold_chunk_recall` | 全部最终引用覆盖的同一组 Gold chunk 比例 |
| `avg_answer_correctness` | DeepEval GEval 答案正确性，只对成功判分记录求均值；无结果为 `None` |

Gold 优先通过证据文本在当前索引对齐，找不到匹配时沿用题集 ID。
`--retrieval-k` 仅为离线评分 cutoff，不控制工具 top_k。逐题保留 Gold、观察和引用 IDs，
以便定位检索遗漏与观察到 Gold 后未引用的问题。

`llm_judge_coverage` 为成功判分题数 / 全部题数，是运行完整性指标，不是质量分。
Judge 未启用、失败或缺少 prediction 时，`answer_correctness=None`、reason 为空、
`judge_used=False`；失败原因保留在 `judge_error`。没有任何结果时均值为 `None`、coverage 为 0。

GEval 使用固定 `evaluation_steps`，evaluation params 为 `INPUT`、`ACTUAL_OUTPUT`、
`EXPECTED_OUTPUT`。Expected output 包含 expected behavior、gold answer、gold key points
和 forbidden points。评价语义事实覆盖、遗漏、矛盾与错误断言；partial answer 要回答已知部分并
说明限制，正确 abstain 不因拒答扣分，同义措辞不扣分。数据集内容与 schema 保持不变。
项目只读取 GEval 的 score/reason，不维护逐关键点判分或综合权重。

薄 `DeepEvalBaseLLM` 经 `get_judge_model()` 与 `ModelGateway.invoke_text()` 返回文本，
由 DeepEval 3.9.6 负责解析。沿用 `JUDGE_*` 配置与项目模型默认值，Runner 支持
`--use-llm-judge`、`--judge-model`。直接调用 metric.measure 并禁用云端记录；
telemetry/tracing 关闭，框架文件系统只读，Core Tool Correctness 仍不调用模型。

`run_rag_evaluation(judge_cache_report=...)` 可只读复用 `per_question_report.csv` 中的
qid、answer、answer_correctness、reason。仅当前 qid 和完全一致的 answer 可复用；
缺行、无分数或答案变化记录 judge_error，保持空分数，绝不回退调用 LLM。当前统一 CLI
没有 cache 参数。复用前需由调用者确认 Gold 和 Judge 配置一致。

批量生成遇到 `MODEL_ERROR`、`MODEL_BAD_OUTPUT`、`TOOL_ERROR` 或 `INTERNAL_ERROR`，
将安全失败记录写到 `prediction_failures.jsonl`，继续后续题目；缺失预测计入 coverage 分母，
召回为零，语义分为 `None`，不调用 Judge。重复调用 ID、无法对齐的 KB ID 等结构错误仍中止生成。

## 产物与可复现性

```text
outputs/evaluation_baseline_v2/<run_id>/
├── manifest.json
├── baseline_summary.json
├── baseline_report.md
├── coverage_report.json
├── failure_cases.jsonl
├── agent/
│   ├── per_case.jsonl
│   ├── per_case.csv
│   └── summary.json
└── rag/
    ├── predictions.csv
    ├── prediction_failures.jsonl
    ├── per_question_report.csv
    ├── summary_report.csv
    └── summary_report.json
```

未运行的层可以没有子目录，summary 记录 skipped。
Manifest 保留 Git/运行环境、数据/索引/Prompt/工具合同 hash、模型配置、Agent 预算、
v3 版本、评分 K、评测权限和 Judge 配置。只写显式白名单；敏感字段和 SecretStr 被拒绝。

Agent 按 suite/category/status/tool/tags 分组；Core 与 Online pass rate 分开解释。
RAG 确定性失败分类包括 `runtime_error`、`rag_chunk_miss`、`rag_reference_gold_miss`。
答案语义问题通过逐题 correctness/reason 观察，不再用启发式规则生成拒答或关键点标签。

## 比较运行

```bash
.venv/bin/python scripts/compare_evaluation_runs.py \
  --baseline outputs/evaluation_baseline_v2/run-a \
  --candidate outputs/evaluation_baseline_v2/run-b
```

比较当前合同下的 Agent case 变化、覆盖率、检索/引用 Recall、Answer Correctness、延迟和调用次数。
未运行的层可显示 unavailable；correctness 任一侧为空或 Judge coverage 不一致/缺失时，
delta 为 `None`，不据此认定质量回退。coverage 相同也需人工核对判分题目集合。
不支持历史 Prediction 的兼容评分。比较前须核对数据、
工具、Prompt、模型、预算和索引是否一致，历史 workflow 分数不能直接归因于当前修改。

默认只报告差异。只有同时指定 `--thresholds data/eval/evaluation_thresholds_v2.example.json`
与 `--fail-on-regression` 才按显式阈值返回非零；示例阈值不是正式 CI 门禁。

性能测试、在线连通性 smoke 与模型 Tool Calling smoke 是其他入口，不参与主 Baseline。
