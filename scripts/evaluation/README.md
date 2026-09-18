# Evaluation Baseline v2 启动说明

本目录保存 Evaluation Baseline v2 的可导入实现，包括 Agent 执行与判分、RAG
预测与评测、Manifest、报告和两轮结果比较。这里不是命令行入口目录，不要直接执行
`runner.py`、`agent.py`、`rag_predictions.py` 或 `rag_evaluator.py`。

请始终从仓库根目录使用以下两个正式入口：

```text
scripts/run_evaluation_baseline.py
scripts/compare_evaluation_runs.py
```

本文重点说明如何选择启动方式、各模式实际会运行什么、输出如何区分，以及当前分支
仍需注意的评测口径限制。完整数据合同和指标定义另见
`docs/evaluation/baseline-v2.md`。

工具包含关系和完整工具序列由固定版本 `deepeval==3.9.6` 的
`ToolCorrectnessMetric` 在本地确定性评分；首工具备选、状态、权限、审批、MCP
安全和 RAG baseline 合同继续由项目维护。不要求 Cloud 登录，不自动上传评测内容，
不启动 DeepEval pytest runner；继续使用下列项目入口。版本兼容性、迁移范围及
离线验证见 [FRAMEWORK-SLIM-4 审计与结果](../../docs/evaluation/framework-slim-4.md)。

## 一、先选择正确的启动方式

| 目的 | 推荐命令 | 是否调用真实模型 | 是否运行 Agent | 是否运行独立 RAG 题集 | 是否访问真实 Web |
| --- | --- | --- | --- | --- | --- |
| 只检查数据和工具合同 | `--validate-only` | 否 | 否 | 否 | 否 |
| 生成正式 Core 基线 | 默认 Combined | 是 | 是 | 是 | 否 |
| 只诊断 Agent | `--agent-only` | 是 | 是 | 否 | 默认否 |
| 只诊断 RAG | `--rag-only` | 是 | 否 | 是 | 否 |
| 验证 Online 工具 | `--agent-only --include-online` | 是 | 是 | 否 | 是 |
| 验证 MCP 工具 | `--agent-only --include-mcp` + Extended 数据集 | 是 | 是 | 否 | 是 |
| 验证 Mixed 路由 | `--agent-only --include-mixed` + Extended 数据集 | 是 | 是 | 否 | 是 |
| 增加 RAG 语义判断 | 默认或 `--rag-only` 加 `--use-llm-judge` | 是 | 取决于模式 | 是 | 不调用 Web 工具，但调用 Judge 模型 |
| 比较两次完整运行 | `compare_evaluation_runs.py` | 否 | 否 | 否 | 否 |

默认 Combined 已经同时生成 `agent/` 和 `rag/` 两部分结果，是正式基线的主运行。
`Agent-only` 和 `RAG-only` 是出现异常、首次验证 Runner 或需要隔离耗时时使用的诊断
模式，不要求每次正式评测都额外运行。

需要特别区分：

- `Agent-only` 不是“完全不用 RAG”。Agent 遇到知识库问题时仍会调用本地
  `search_knowledge_base`，只是不会执行独立的 65 题 RAG 数据集。
- `RAG-only` 只执行 RAG 数据集，不创建 Agent case 结果。
- `--include-online` 只影响 Agent suite；与 `--rag-only` 一起使用没有实际意义。
- `--include-mcp` 与 `--include-mixed` 才会初始化 MCP；二者都不改变
  `--include-online` 的既有 Web-only 语义。
- `--use-llm-judge` 只影响 RAG 语义指标；与 `--agent-only` 一起使用没有实际意义。

## 二、运行前准备

### 1. 必须从仓库根目录启动

```bash
cd /home/wabce94123/projects/rag-agent-qa-system
```

使用项目虚拟环境中的 Python，并先确认入口可以加载：

```bash
python scripts/run_evaluation_baseline.py --help
python scripts/compare_evaluation_runs.py --help
```

### 2. 正式运行前检查 Git 状态

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git diff --check
```

正式基线要求工作区干净。Runner 在工作区有未提交修改时会拒绝运行，目的是确保
Manifest 中记录的 commit、数据 SHA、Prompt SHA 和工具合同能够复现结果。

不要为了绕过检查而默认添加 `--allow-dirty`。它只适合临时实验，不能替代正式
基线。

### 3. 确认真实运行依赖

Combined、Agent-only 和 RAG-only 都可能需要：

- 可以构造当前 `AppSettings`；
- 回答模型已经配置并可以连接；
- Query Rewrite 启用时相应模型可以连接；
- 知识库源文件、索引 Manifest 和 Chroma 持久化目录存在；
- Embedding 与 Rerank 模型缓存可用。

默认 Core 不开放 Web Search、Web Fetch 或 Web PDF Fetch，但调用配置的回答模型、
Query Rewrite 模型仍可能需要访问模型服务。模型服务网络失败不等于
RAG 索引失败；正式 Agent 返回的错误状态作为当前题目的 runtime failure 计零分。
初始化或评测执行抛出的结构/基础设施异常仍中止整批。

## 三、推荐启动流程

### 第一步：只做离线合同校验

```bash
python scripts/run_evaluation_baseline.py --validate-only
```

此模式只检查：

- Agent v2 数据 Schema；
- case ID 是否重复；
- Core/Online、单轮/多轮和 category 覆盖；
- expected status、工具名和当前本地 ToolSpec catalog 是否一致；
- `tool_sequence`、`tool_status_sequence` 等合同是否合法。

它不会初始化 RAG、模型或工具执行器，不访问网络，不创建输出目录，也不写数据库。
建议每次修改评测数据或评测合同后先运行它。

### 第二步：运行正式 Combined Core

默认模式同时执行独立 RAG 题集和 Agent Core case：

```bash
python scripts/run_evaluation_baseline.py \
  --agent-cases data/eval/agent_eval_cases_v2.json \
  --rag-eval-file data/eval/rag_eval_questions_v2_semantic_fixed.json \
  --output-dir outputs/evaluation_baseline_v2/core-<short-sha> \
  --retrieval-k 5
```

Combined 的执行顺序是：

```text
独立 RAG 数据集
    -> 生成 rag/ 结果
Agent Core 数据集
    -> 生成 agent/ 结果
统一汇总
    -> manifest、summary、report、failure_cases
```

这是正式结果目录。正常情况下，不需要为了得到“更完整的正式分数”再重复执行
Agent-only 和 RAG-only。

### 第三步：只在需要时运行单层诊断

RAG 初始化、检索、生成或 RAG evaluator 出现问题时：

```bash
python scripts/run_evaluation_baseline.py \
  --rag-only \
  --rag-eval-file data/eval/rag_eval_questions_v2_semantic_fixed.json \
  --output-dir outputs/evaluation_baseline_v2/rag-only-<short-sha> \
  --retrieval-k 5
```

ChatService、Agent 工具选择、状态映射、多轮或工具序列出现问题时：

```bash
python scripts/run_evaluation_baseline.py \
  --agent-only \
  --agent-cases data/eval/agent_eval_cases_v2.json \
  --output-dir outputs/evaluation_baseline_v2/agent-core-<short-sha>
```

首次引入新 Runner 时，可以按 `RAG-only -> Agent-only -> Combined` 的顺序验证三层
是否一致。Runner 已稳定后，日常正式运行直接执行 Combined，只有异常时再补单层诊断。

单层与 Combined 是不同的真实模型运行。自然语言答案和检索顺序可能波动，不要求逐字
一致；应比较 case 集合、状态、工具合同、数据/索引/Prompt 哈希以及同口径指标。

## 四、Online 工具的启动方式

Core suite 只给 Agent `knowledge_base:read` 权限。Online suite 才额外开放
`web:search` 和 `web:fetch`，并由 Web PDF 工具复用受控网页读取能力。

运行 Core 加 Online Agent case：

```bash
python scripts/run_evaluation_baseline.py \
  --agent-only \
  --include-online \
  --output-dir outputs/evaluation_baseline_v2/agent-online-<short-sha>
```

注意：

- `--include-online` 会执行 Core 和 Online Agent case，不是 Online-only 过滤器。
- Online 结果依赖实时网络、搜索结果、网页内容、代理和第三方站点，不能与 Core pass
  rate 混为稳定质量门禁。
- Core 中的天气、股票等问题在 Web 权限关闭时只能测“能力不可用时是否安全处理”，
  不能证明真实联网能力。
- 股票推荐属于安全/建议边界，不应因为开放 Web 权限就要求模型给出买入建议；如果要
  测联网查询，应另设“查询公开行情”case。

若只想检查 Web 工具连通性，而不是执行 Agent Baseline，应使用仓库根目录的
`scripts/smoke_test_online_tools.py`，不要把 Smoke Test 结果计入 Core Baseline。

### MCP 与 Mixed Extended Evaluation

Extended live cases 使用独立数据集，先执行零网络校验，再分别运行 MCP 与 Mixed：

```bash
python scripts/run_evaluation_baseline.py \
  --agent-cases data/eval/agent_eval_cases_extended_v2.json \
  --validate-only

python scripts/run_evaluation_baseline.py --agent-only \
  --agent-cases data/eval/agent_eval_cases_extended_v2.json \
  --include-mcp \
  --output-dir outputs/evaluation_baseline_v2/extended-mcp-<short-sha>

python scripts/run_evaluation_baseline.py --agent-only \
  --agent-cases data/eval/agent_eval_cases_extended_v2.json \
  --include-mixed \
  --output-dir outputs/evaluation_baseline_v2/extended-mixed-<short-sha>
```

Runner 只给每条 case 声明的 Server 增加 `mcp:<server>` 权限。正式运行会在评测
Settings scope 内完成 discovery、runtime catalog preflight 和成对 cleanup；Server
或工具缺失会作为 `PRECHECK_FAILED` 停止，不记成普通 Agent failure。Manifest 只写
安全 catalog 字段，MCP 正文不会进入逐 case trace，结束后还会用本地配置解析出的
Secret 值逐值扫描文本产物。真实外部运行不属于普通 CI 门禁。

## 五、LLM Judge 的启动方式

默认关闭 Judge：

```text
llm_judge_coverage = 0
```

此时 retrieval、拒答规则等确定性指标仍可用，但 `key_point_recall`、语义 forbidden、
contradiction 和依赖这些字段的综合 Final Score 不能解释为完整答案质量。

显式启用 Judge：

```bash
python scripts/run_evaluation_baseline.py \
  --rag-only \
  --use-llm-judge \
  --judge-model deepseek-chat \
  --output-dir outputs/evaluation_baseline_v2/rag-judge-<short-sha> \
  --retrieval-k 5
```

也可以在 Combined 中启用，但建议首次先用 RAG-only 验证 Judge 配置、成本、错误处理和
报告字段，再决定是否运行完整 Combined。

Judge 是辅助语义判断，不替代以下确定性检查：

- Agent status、首个工具和工具序列；
- 工具输出 Schema；
- 来源绑定；
- Gold 文档和 Gold chunk；
- Online 权限边界。

## 六、常用参数说明

| 参数 | 含义 | 使用建议 |
| --- | --- | --- |
| `--agent-cases` | Agent v2 数据集路径 | 默认使用 `data/eval/agent_eval_cases_v2.json` |
| `--rag-eval-file` | RAG 数据集路径 | 默认使用 `data/eval/rag_eval_questions_v2_semantic_fixed.json` |
| `--output-dir` | 完整输出目录 | 正式运行使用全新目录；非空目录会被拒绝 |
| `--run-name` | Manifest 中的 run ID | 未提供时使用输出目录名或 UTC 时间 |
| `--retrieval-k` | Agent 有序去重观察证据的评分 cutoff | 不控制每轮工具 top_k |
| `--agent-only` | 跳过独立 RAG 题集 | Agent 的知识库工具仍可使用 RAG |
| `--rag-only` | 跳过 Agent case | 不受 `--include-online` 影响 |
| `--include-online` | 在 Core Agent 之外增加 Online Agent case | 只在显式联网运行中使用 |
| `--include-mcp` | 增加 MCP suite 并初始化隔离 runtime | 只与 Extended 数据集 live run 一起使用 |
| `--include-mixed` | 增加 Mixed suite，开放 Web 和 case 声明的 MCP Server | 只与 Extended 数据集 live run 一起使用 |
| `--use-llm-judge` | 启用 RAG 关键点和 forbidden 语义判断 | 会增加模型调用和成本 |
| `--judge-model` | 指定 Judge 模型 | 只有同时启用 Judge 才有实际意义 |
| `--allow-dirty` | 允许脏工作区实验 | 不用于正式基线 |
| `--force-rebuild` | 强制重建本地索引 | 会修改持久化索引，只在确认索引无效后使用 |

如果不传 `--output-dir`，Runner 会使用 UTC 时间作为 run ID，并写入：

```text
outputs/evaluation_baseline_v2/<UTC-run-id>/
```

为便于人工识别和避免覆盖，推荐正式运行显式使用包含短 commit SHA 的新目录。

## 七、输出目录怎么读

Combined 成功后主要文件为：

```text
outputs/evaluation_baseline_v2/<run-id>/
├── manifest.json
├── baseline_summary.json
├── baseline_report.md
├── coverage_report.json
├── failure_cases.jsonl
├── secret_scan.json
├── agent/
│   ├── per_case.jsonl
│   ├── per_case.csv
│   └── summary.json
└── rag/
    ├── predictions.csv
    ├── prediction_failures.jsonl
    ├── per_question_report.csv
    ├── summary_report.csv
    ├── summary_report.json
    └── llm_judge_details.jsonl
```

推荐阅读顺序：

1. `manifest.json`：先确认 commit、dirty、数据/索引/Prompt/工具合同哈希和实际模式。
2. `coverage_report.json`：确认 case 数量、Core/Online、工具和 category 没有缺口。
3. `baseline_summary.json`：读取机器可比较的 Agent 与 RAG 汇总指标。
4. `failure_cases.jsonl`：查看确定性 evaluator 捕获的失败。
5. `agent/per_case.jsonl`：查看每个 Agent case 的状态、工具、关键词、来源和 trace。
6. `rag/per_question_report.csv`：对照问题、Gold、召回 chunk 和实际答案。
7. `baseline_report.md`：用于人工汇报，不应代替原始 JSON/CSV 审计。

`failure_cases.jsonl` 只包含现有分类器能够确定的失败。Judge 关闭时，部分回答缺关键点、
跨文档答案遗漏、语义 forbidden 和事实矛盾可能不会进入该文件，仍需人工审查逐题报告。

## 八、比较两次运行

候选运行完成后：

```bash
python scripts/compare_evaluation_runs.py \
  --baseline outputs/evaluation_baseline_v2/core-<baseline-sha> \
  --candidate outputs/evaluation_baseline_v2/core-<candidate-sha>
```

默认把 `comparison.json` 和 `comparison.md` 写到 candidate 目录。比较器只读取已有产物，
不会重新调用模型、RAG 或网络。

只有在阈值已经过人工确认后，才使用：

```bash
python scripts/compare_evaluation_runs.py \
  --baseline outputs/evaluation_baseline_v2/core-<baseline-sha> \
  --candidate outputs/evaluation_baseline_v2/core-<candidate-sha> \
  --thresholds data/eval/evaluation_thresholds_v2.example.json \
  --fail-on-regression
```

不要把同一个目录同时作为 baseline 和 candidate。那只能验证比较器可以读取格式，不能
提供任何运行一致性或质量变化证据。

## 九、参数与证据口径

### 1. Agentic RAG 预测合同

Agent 与 RAG Evaluation 共用 `run_evaluation_chat_turn → ChatService.stream → stream_agent → create_agent`。
每题 RAG 预测使用空 session、`RAG_ONLY` 与唯一权限 `knowledge_base:read`。
模型可见且可执行的 KB 证据工具为 `search_knowledge_base`、`compare_documents`、`get_chunk_text`；
Web、MCP 和文档列表不参与该 baseline。每轮 query/top_k 由 Agent 自主决定。

Prediction v3 字段为版本、qid、question、answer、status、retrieved/reference chunk IDs 与数量，
以及 retrieval/tool/LLM call count。retrieval_call_count 计成功返回 evidence 字段的 KB 工具调用，
包括成功但 evidence 为空的检索；tool_call_count 取生产工具 trace，LLM count 取正式结果。

`retrieved_chunk_ids` 从真实工具输出的 KB Document metadata 提取，按工具观察顺序稳定去重；
`reference_chunk_ids` 只从最终通过 grounding 的 references 投影，允许是观察证据的子集。
缺少有效 chunk_id 或不在当前评测索引时生成失败；模型、工具和内部错误也使正式预测失败。
NO_RELEVANT_INFO 与 NEED_CLARIFICATION 正常参与行为评分。

批量生成只隔离 Agent 正常返回的 `MODEL_ERROR`、`MODEL_BAD_OUTPUT`、`TOOL_ERROR`、
`INTERNAL_ERROR`：该题不写 prediction row，而写入 `rag/prediction_failures.jsonl`，然后继续下一题。
`predictions.csv` 只保存完成 Prediction v3 合同的结果，Schema 仍为 3。
单题诊断 API `run_rag_once_with_result` 对这些错误状态仍抛出 `RuntimeError`。

Failure sidecar 字段为 `qid`、`question`、`status`、`reason`、`retrieved_chunk_ids`、
`reference_chunk_ids`、`retrieval_call_count`、`tool_call_count`、`llm_call_count`，以及
`requested_evidence_ids`、`available_evidence_ids`、`unavailable_evidence_ids`。
缺失值使用空列表、0 或空字符串。只投影已有公开状态、ID、计数和数据集问题；不保存失败答案、
证据正文、完整 metadata、ToolMessage、Prompt 或隐藏推理。
三个 grounding ID 列表来自 `compose_final_answer` 的 mismatch trace；生产成功路径不附这些字段。
未知证据仍被完整拒绝，不过滤后放行，也不新增 retry。

Missing prediction 的 `has_prediction=false`，行为、关键点、检索及最终分均为 0，
Judge 不被调用，失败分类仅为 `runtime_error`。Sidecar 中的实际检索 IDs 只用于归因，
不会输入正式 retrieval score。题目仍进入整体和 category 的 coverage 分母。
`baseline_summary.json` 的 `rag` 部分增加 `prediction_failure_count` 和
`prediction_failure_status_counts`；用 `qid` 将 sidecar 的 status/reason 与
`failure_cases.jsonl` 的 `case_id` 对齐。

CSV header、每条成功 row 与每条失败 JSONL 都即时 flush；完整 loop 必须满足
`successful_prediction_count + case_runtime_failure_count == evaluation_question_count`。
全题失败时 CSV 仅含 header，coverage 为 0，Runner 仍可完成 evaluator 和报告。
缺失/非法 chunk ID、工具捕获与 trace 无法对齐、重复 call ID、损坏/重复题目、初始化失败、
序列化或文件写入异常仍中止整批。进程崩溃或 KeyboardInterrupt 也不构成成功运行。
部分产物只供诊断；accepted baseline 必须完成所有题目、正式 evaluator、manifest 和 reports，
并满足既有 clean-worktree 要求，无需 100% prediction coverage。

### 2. 评分 cutoff 与历史兼容

`--retrieval-k` 只对完整有序观察证据的前 K 项计算 retrieval metrics，绝不传给检索工具。
Manifest 记录 `prediction_schema_version=3`、`retrieval_metric_k`、索引与模型信息，
Prompt hash 仅保留 `prompts.agent`。不再保留 requested/effective retrieval 参数。

v3 没有独立 context 阶段，context 指标显示 unavailable / not applicable，不计为零。
历史 v1/v2 CSV 可以继续纯数据判分；其 context 语义不用于当前 Agent。
Reference 指标、Judge/cache 和 Final Score 原权重保留。

Prediction Schema v3 is a new baseline generation contract.
Raw scores are not strictly comparable to historical v2 workflow scores.

## 十、仍存在的限制

### 1. Core 不评测真实联网能力

Core Agent 只开放本地知识库权限。天气、股票等 Core case 只能检查 Web 权限关闭时的
安全退化和结果状态，不能评判有网络时的工具选择或实时答案质量。联网能力必须通过
显式 Online case 单独评测，安全拒答问题也不应强制要求无意义的本地知识库调用。

### 2. Judge 关闭时语义答案质量不完整

不启用 `--use-llm-judge` 时，关键点覆盖和语义 forbidden 判断不可用；确定性
Retrieval/Reference 指标不能替代完整答案质量判断。

### 3. Online 结果不适合作为稳定 CI 门禁

Online 工具受网络、搜索结果和外部网页变化影响，应与 Core 分开记录和人工复核。

## 十一、常见阻塞与处理

### 工作区不干净

```text
工作区不干净，拒绝生成正式 baseline
```

这是复现保护，不是参数错误。先审查并提交预期修改，再从干净工作区生成正式结果；
只有明确的实验才使用 `--allow-dirty`。

### 输出目录已经存在且非空

Runner 不覆盖已有结果。保留原目录作为证据，为重试使用新的 `-r1`、`-r2` 或新 SHA
目录。

### `APIConnectionError` 或模型调用失败

先区分Agent、Rewrite 或 Judge 模型服务。索引已经成功加载时，模型网络错误
不是检索质量失败，也不应通过伪造结果、自动换模型或降低 Gold 处理。

### 索引缺失或无效

先检查知识库路径、索引 Manifest、源文件变化、Embedding/Rerank 模型和缓存。只有确认
索引确实需要重建时才使用 `--force-rebuild`；不要为了提高分数重建或调参。

### Online case 失败

检查是否显式使用 `--include-online`，再区分权限不可见、搜索/代理失败、网页 SSRF
拒绝、远端内容变化和模型工具选择。Online 失败不得混入稳定 Core 质量结论。

## 十二、建议的完整质量门

修改评测实现、数据或文档后，至少执行：

```bash
python scripts/run_evaluation_baseline.py --validate-only
ruff check app tests scripts
python -m compileall -q app scripts tests
git diff --check
```

准备提交或生成正式基线前，再执行项目完整门禁：

```bash
lint-imports
pytest -q
python scripts/inspect_agent_runtime.py --check
python -m pip check
```

自动化测试不得访问真实模型、RAG、网络、代理或生产持久化。真实 Combined、Online 和
Judge 运行属于显式、本地、受控的产品质量流程。
