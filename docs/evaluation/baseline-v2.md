# Evaluation Baseline v2

Evaluation Baseline v2 是产品质量测量入口，不改变 Agent、RAG、工具或 API 的生产
行为。它把现有 Agent evaluator、RAG predictions 和语义 evaluator 组合成一次可
版本化、可重复、可比较、可诊断的运行。

## 三个评测层

| 层 | 默认 | 外部依赖 | 统计口径 |
| --- | --- | --- | --- |
| Core Agent + 本地 RAG | 启用 | 已配置模型和本地索引 | 稳定基线 |
| Online tools | 关闭；`--include-online` 开启 | 搜索、网页、PDF 和网络 | 与 Core 分开 |
| LLM Judge | 关闭；`--use-llm-judge` 开启 | Judge 模型 | 仅辅助答案语义质量 |

Judge 不替代 Router、status、工具序列、来源、gold chunk 或确定性关键词检查。
Online 搜索结果和网页内容会变化，因此不应成为稳定 CI 门禁。

## 数据集

- Agent v2：`data/eval/agent_eval_cases_v2.json`
- RAG：`data/eval/rag_eval_questions_v2_semantic_fixed.json`
- 示例阈值：`data/eval/evaluation_thresholds_v2.example.json`

v1 文件保留。Agent v2 的每条 case 必须包含 `version: 2`，并且只能选择
`question` 单轮结构或至少两轮的 `turns` 结构。`suite` 只能是 `core` 或
`online`。合法 status 来自 `AgentResultStatus`，工具名来自当前
`ToolRegistry`。

只校验数据合同：

```bash
python scripts/run_evaluation_baseline.py --validate-only
```

此命令不创建输出目录，不初始化模型或 RAG，不创建 ToolRuntime，不访问网络，也
不写数据库。

## 运行

默认运行 Core Agent 和本地 RAG，不运行 Online 或 Judge：

```bash
python scripts/run_evaluation_baseline.py \
  --agent-cases data/eval/agent_eval_cases_v2.json \
  --rag-eval-file data/eval/rag_eval_questions_v2_semantic_fixed.json \
  --output-dir outputs/evaluation_baseline_v2/initial-core \
  --top-k 5
```

只运行一个层：

```bash
python scripts/run_evaluation_baseline.py --agent-only \
  --output-dir outputs/evaluation_baseline_v2/agent-only

python scripts/run_evaluation_baseline.py --rag-only \
  --output-dir outputs/evaluation_baseline_v2/rag-only
```

显式启用不稳定或有额外成本的能力：

```bash
python scripts/run_evaluation_baseline.py \
  --include-online \
  --use-llm-judge \
  --judge-model deepseek-chat \
  --output-dir outputs/evaluation_baseline_v2/online-with-judge
```

工作区不干净时，Runner 默认拒绝生成正式基线。`--allow-dirty` 只用于实验运行，
且 `manifest.json` 会记录 `git_dirty: true`，报告顶部也会标记。
`--force-rebuild` 会重建本地 RAG 索引，除非明确需要，否则不要使用。

## 隔离边界

多轮 Agent case 使用真实 `ChatService` 流程，但注入内存 session adapter 和
no-op metrics/bad-case writer。每个 case 使用独立 session，结束后清空。
评测 Settings 快照关闭 LangGraph Checkpointer、Agent trace store 和 tool audit
写入，并把所有备用路径限定在本次 output 目录下。因此评测不会写生产 SQLite、
metrics、bad case、trace 或工具审计文件。

生产 tool trace 只保留按 ToolSpec 脱敏后的输出摘要。为校验 `tool_output_keys`，
Runner 仅在当前评测调用的内存中捕获真实工具输出的字段名；字段值不会进入评测
记录、报告或 manifest，调用结束后立即丢弃。

Core case 的工具权限只有本地知识库读取；只有 Online case 才开放 `web:search`
和 `web:fetch`。这使 Core 运行不会意外访问真实 Web Search、Web Fetch 或 Web
PDF Fetch。

## 输出

一次完整运行的目录为：

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
    ├── per_question_report.csv
    ├── summary_report.csv
    ├── summary_report.json
    └── llm_judge_details.jsonl
```

未启用的层可以没有对应子目录；`baseline_summary.json` 和 Markdown 报告会记录
`skipped` 及原因。`outputs/` 整体被 Git 忽略。

## Manifest

`manifest.json` 记录：

- commit、branch、dirty 状态、run ID、时间、Python 和平台；
- Agent/RAG 数据路径及 SHA-256、case/suite/category 数量；
- 索引 manifest 路径及 SHA-256、索引来源、文档/chunk 数量、embedding、
  rerank、chunk size 和 overlap；
- 回答、Query Rewrite、Decompose 模型名和开关、温度、token、top-k 和 Agent
  预算；
- 当前八个工具名、完整 Tool contract hash；
- Prompt、Query Rewrite、Decompose、Generation、Compare 和工具描述所在源文件
  的 SHA-256；
- Judge 是否启用、模型和缓存来源。

Manifest 使用显式白名单，禁止完整 `settings.model_dump()`。递归安全检查拒绝
API key、Authorization、Cookie、password、secret 字段和 `SecretStr` 值。报告也
不保存完整敏感工具输入、网页/PDF 正文或私密会话。

## 指标与失败原因

Agent 指标按执行 case 计算；跳过的 Online case 不进入分子或分母。延迟 P50/P95
使用排序后线性插值。工具成功率按真实 `tool_call` trace 中成功调用数除以全部调用
数。LLM call 数使用 ChatService 汇总指标，包含实际发生的 Query Rewrite 调用。

Agent 结果按 suite、category、expected status、expected first tool 和 tags 分组。
Core 与 Online 在 `groups.suite` 中独立。

RAG 指标直接复用 `scripts/rag_eval/evaluation.py`，包括 prediction coverage、
gold chunk hit/recall、all-gold hit、doc hit、behavior、key-point、forbidden 和
最终分。公式未被统一 Runner 修改；key-point/forbidden 语义指标是否有效取决于
LLM Judge 是否启用。

Judge 关闭时，拒答行为只用高精度规则识别整段独立拒答；长答案中出现“未说明”
或“未提供”等局部证据限制，不会被直接当作整段拒答。语义关键点覆盖和 forbidden
判断仍明确标记为未评判，不能把对应的 `0` 当作模型语义质量结论。

每个失败 case 可以有多个 `failure_reasons`，例如：

```text
runtime_error
timeout
invalid_result_contract
decision_mismatch
action_mismatch
status_mismatch
first_tool_mismatch
tool_sequence_mismatch
tool_call_failed
keyword_missing
forbidden_keyword_hit
source_missing
tool_output_schema_missing
rag_document_miss
rag_chunk_miss
rag_answer_keypoint_miss
rag_unexpected_refusal
rag_missing_refusal
query_rewrite_mismatch
unexpected_clarification
unexpected_no_evidence
```

## 比较两轮运行

```bash
python scripts/compare_evaluation_runs.py \
  --baseline outputs/evaluation_baseline_v2/run-a \
  --candidate outputs/evaluation_baseline_v2/run-b
```

默认在 candidate 目录写 `comparison.json` 和 `comparison.md`，只报告 case
增删、新增失败、已修复失败、发生变化的 case、总体和 category 指标、RAG 指标、
P50/P95、平均模型及工具调用，不因下降返回失败退出码。

只有同时显式提供版本化阈值和开关时，回退才返回非零：

```bash
python scripts/compare_evaluation_runs.py \
  --baseline outputs/evaluation_baseline_v2/run-a \
  --candidate outputs/evaluation_baseline_v2/run-b \
  --thresholds data/eval/evaluation_thresholds_v2.example.json \
  --fail-on-regression
```

示例阈值不是当前 CI 门禁。至少完成一次真实基线和人工审查后，才应决定是否建立
正式阈值。

## 新增 case

1. 从当前知识库文档、已有 RAG gold 或确定性 Router/Tool contract 核验 ground
   truth。
2. 选择 `core` 或 `online`，填写唯一 ID、category、问题和确定性 expected。
3. 工具期望写真实 `tool_call` 名，不使用 Router 的旧 `tool_name`。
4. 来源和关键词只写已核验内容；不能可靠确定时留空或把 case 标为 disabled 并写
   明原因。
5. 多轮 case 使用独立 `session_id`；Query Rewrite 期望放在对应 turn 的
   `rewrite_keywords`、`rewritten` 或 `rewrite_llm_used`。
6. 运行 `--validate-only`，检查 coverage matrix 和 gap。

CI 只运行数据校验，不运行真实模型、索引或网络。真实 baseline 属于受控、本地的
产品质量流程。
