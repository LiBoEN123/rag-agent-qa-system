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

## 一、先选择正确的启动方式

| 目的 | 推荐命令 | 是否调用真实模型 | 是否运行 Agent | 是否运行独立 RAG 题集 | 是否访问真实 Web |
| --- | --- | --- | --- | --- | --- |
| 只检查数据和工具合同 | `--validate-only` | 否 | 否 | 否 | 否 |
| 生成正式 Core 基线 | 默认 Combined | 是 | 是 | 是 | 否 |
| 只诊断 Agent | `--agent-only` | 是 | 是 | 否 | 默认否 |
| 只诊断 RAG | `--rag-only` | 是 | 否 | 是 | 否 |
| 验证 Online 工具 | `--agent-only --include-online` | 是 | 是 | 否 | 是 |
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
- Query Rewrite 或 Decompose 启用时，相应模型可以连接；
- 知识库源文件、索引 Manifest 和 Chroma 持久化目录存在；
- Embedding 与 Rerank 模型缓存可用。

默认 Core 不开放 Web Search、Web Fetch 或 Web PDF Fetch，但调用配置的回答模型、
Query Rewrite 模型和 Decompose 模型仍可能需要访问模型服务。模型服务网络失败不等于
RAG 索引失败，也不能计入质量分。

## 三、推荐启动流程

### 第一步：只做离线合同校验

```bash
python scripts/run_evaluation_baseline.py --validate-only
```

此模式只检查：

- Agent v2 数据 Schema；
- case ID 是否重复；
- Core/Online、单轮/多轮和 category 覆盖；
- expected status、工具名和当前 `ToolRegistry` 是否一致；
- `tool_sequence`、`tool_status_sequence` 等合同是否合法。

它不会初始化 RAG、模型或 ToolRuntime，不访问网络，不创建输出目录，也不写数据库。
建议每次修改评测数据或评测合同后先运行它。

### 第二步：运行正式 Combined Core

默认模式同时执行独立 RAG 题集和 Agent Core case：

```bash
python scripts/run_evaluation_baseline.py \
  --agent-cases data/eval/agent_eval_cases_v2.json \
  --rag-eval-file data/eval/rag_eval_questions_v2_semantic_fixed.json \
  --output-dir outputs/evaluation_baseline_v2/core-<short-sha> \
  --top-k 5 \
  --decompose-top-k 5
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
  --top-k 5 \
  --decompose-top-k 5
```

ChatService、Agent 工具选择、状态映射、多轮或工具序列出现问题时：

```bash
python scripts/run_evaluation_baseline.py \
  --agent-only \
  --agent-cases data/eval/agent_eval_cases_v2.json \
  --output-dir outputs/evaluation_baseline_v2/agent-core-<short-sha> \
  --top-k 5 \
  --decompose-top-k 5
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
  --output-dir outputs/evaluation_baseline_v2/agent-online-<short-sha> \
  --top-k 5 \
  --decompose-top-k 5
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
  --top-k 5 \
  --decompose-top-k 5
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
| `--top-k` | 单查询最终排序数量和 evaluator 的 `Recall@k` | 比较运行时必须保持一致 |
| `--decompose-top-k` | 多查询合并后的目标数量 | 比较运行时必须保持一致；当前分支限制见下文 |
| `--agent-only` | 跳过独立 RAG 题集 | Agent 的知识库工具仍可使用 RAG |
| `--rag-only` | 跳过 Agent case | 不受 `--include-online` 影响 |
| `--include-online` | 在 Core Agent 之外增加 Online Agent case | 只在显式联网运行中使用 |
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

## 九、当前分支已知限制

以下限制来自当前代码和真实首轮产物。修复后必须同步更新本节。

### 1. 独立 RAG 路径没有传递 `decompose_top_k`

Runner 会解析并在 Manifest 中记录 `--decompose-top-k`，但当前
`run_rag_baseline -> run_predictions -> run_rag_subgraph` 调用链只传递 `top_k`。
RAG 子图因此回退到 `AppSettings.decompose_top_k`。

这会导致命令、Manifest 和实际多查询 `final_chunks` 数量不一致。在该问题修复并重跑
前，不应把现有 RAG-only/Combined 结果当作指定 `decompose_top_k` 下的正式可复现
结果。Agent 调用链会通过 `ChatServiceRequest` 显式传递该值，不受同一遗漏影响。

### 2. `Recall@k` 不等于回答模型实际上下文覆盖

当前 evaluator 对 `predictions.csv` 中的 `retrieved_chunk_ids` 直接取前 `k` 个计算
`Recall@k`。多查询路径可能返回更多 `final_chunks`，而回答上下文还会受
`max_context_chars` 截断。

因此需要区分：

```text
基础召回候选
最终排序前 k 个
完整 final_chunks
真正进入 Prompt 的 context chunks
```

当前产物没有单独保存最后一项。`Recall@5` 可以解释为前五排序质量，但不能直接代表
回答模型实际看到了多少 Gold 证据。

### 3. Core 不评测真实联网能力

Core Agent 只开放本地知识库权限。天气、股票等 Core case 只能检查 Web 权限关闭时的
安全退化和结果状态，不能评判有网络时的工具选择或实时答案质量。联网能力必须通过
显式 Online case 单独评测，安全拒答问题也不应强制要求无意义的本地知识库调用。

## 十、常见阻塞与处理

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

先区分回答、Rewrite、Decompose 或 Judge 模型服务。索引已经成功加载时，模型网络错误
不是检索质量失败，也不应通过伪造结果、自动换模型或降低 Gold 处理。

### 索引缺失或无效

先检查知识库路径、索引 Manifest、源文件变化、Embedding/Rerank 模型和缓存。只有确认
索引确实需要重建时才使用 `--force-rebuild`；不要为了提高分数重建或调参。

### Online case 失败

检查是否显式使用 `--include-online`，再区分权限不可见、搜索/代理失败、网页 SSRF
拒绝、远端内容变化和模型工具选择。Online 失败不得混入稳定 Core 质量结论。

## 十一、建议的完整质量门

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
