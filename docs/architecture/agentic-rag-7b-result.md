# AGENTIC-RAG-7B RESULT

## STATUS

**FAIL — Gate 1 在 q045 因最终引用校验失败而中止，正式 Prediction v3 baseline 尚未建立。**

Gate 0 通过；正式运行 q001–q044 均返回 `OK`。q045 返回 `MODEL_ERROR`，原因是
`Agent structured response referenced unavailable evidence`。Runner 按现有 fail-fast
合同退出，退出码为 **1**。q046–q065 未执行，正式 predictions 和评分产物没有生成。

仅对 q045 额外复跑 **2 次**，均为 `OK`，未复现非法 evidence ID。当前归因是
**运行时 grounding 拒绝 + MODEL_VARIANCE / underlying cause UNCERTAIN**，尚未证明
新的确定性代码 blocker。复跑没有替换正式结果，也没有触发另一轮 65 题运行。
按 Gate 依赖要求，Agent Core、Selective Judge 均未执行。

## BASELINE

- branch: `refactor/tool-output-models`
- head: `bce52c0e1f35084676b001945dabd4ea24e1a530`
- worktree_before: **clean**；所有真实调用完成后、写本报告前仍 **clean**。
- rag_output: `outputs/evaluation_baseline_v2/agentic-rag-7b-rag-bce52c0/`；新建空目录，正式运行失败后没有正式文件。
- agent_output: 计划为 `outputs/evaluation_baseline_v2/agentic-rag-7b-agent-bce52c0/`，**NOT RUN / 未创建**。
- judge_output: 计划为 RAG 目录下 `analysis/judge/`，**NOT RUN / 未创建**。
- diagnostics: [agentic-rag-7b-diagnostics-bce52c0](../../outputs/evaluation_baseline_v2/agentic-rag-7b-diagnostics-bce52c0/)。
- 正式运行时间：2026-09-18 15:55:53–16:05:47，Asia/Shanghai；两次诊断于 16:08:27 前完成。
- Python: 项目 `.venv/bin/python`，3.12.3。

实际正式命令如下；两次运行均使用 `set -o pipefail` 和 `2>&1 | tee` 保存完整输出：

```bash
.venv/bin/python scripts/run_evaluation_baseline.py --validate-only

.venv/bin/python -u scripts/run_evaluation_baseline.py \
  --rag-only \
  --rag-eval-file data/eval/rag_eval_questions_v2_semantic_fixed.json \
  --output-dir outputs/evaluation_baseline_v2/agentic-rag-7b-rag-bce52c0 \
  --retrieval-k 5
```

[Gate 0 日志](../../outputs/evaluation_baseline_v2/agentic-rag-7b-diagnostics-bce52c0/validate-only.log)、
[完整 Gate 1 stdout/stderr](../../outputs/evaluation_baseline_v2/agentic-rag-7b-diagnostics-bce52c0/rag-baseline.log)。
未使用 `--allow-dirty`、`--force-rebuild`、`--use-llm-judge` 或 Online/MCP/Mixed 开关；未运行 Combined。

## V3 BASELINE DECLARATION

**The first official real-model Agentic RAG Prediction v3 baseline HAS NOT been established.**

**Historical v2 raw scores are not strictly comparable.** v3 使用生产 Agent 自主工具调用、
累计观察证据及最终引用，与历史 v2 workflow 执行语义不同。本轮不作 v2/v3 分数差比较，
也不建立回归阈值。

以下 11 个必需正式产物均未生成：

```text
rag/predictions.csv
rag/per_question_report.csv
rag/summary_report.csv
rag/summary_report.json
rag/llm_judge_details.jsonl
manifest.json
baseline_summary.json
baseline_report.md
failure_cases.jsonl
coverage_report.json
secret_scan.json
```

不存在可接受的 65 行 Prediction v3 CSV。没有手工补行、跳题、拼接旧预测，或使用
Phase 7A smoke/本轮复跑补题。[产物检查及源码完整性](../../outputs/evaluation_baseline_v2/agentic-rag-7b-diagnostics-bce52c0/source-integrity-before-report.json)
记录了每个文件的存在性。诊断目录中的配置快照、日志投影及归因矩阵均不冒充正式产物。

## REAL-MODEL CONFIG

配置来自当前 checkout 的安全白名单快照及诊断初始化后的
[targeted-manifest.json](../../outputs/evaluation_baseline_v2/agentic-rag-7b-diagnostics-bce52c0/targeted-manifest.json)。
该文件是诊断 manifest，**不是成功的 Gate 1 manifest**。

| 项目 | 实际值 |
| --- | --- |
| answer model | `qwen-plus` |
| query rewrite | enabled=true，`deepseek-chat`；本轮空 session RAG 日志均为 no history，未调用 rewrite LLM |
| judge | 配置为 `qwen-plus`；本轮未调用 |
| embedding | `Qwen/Qwen3-Embedding-0.6B` |
| rerank | `BAAI/bge-reranker-v2-m3` |
| temperature / max tokens | 0.1 / 1024 |
| budgets | iterations=10；tool_calls=10；tool_errors=3；连续同工具调用/错误上限均为 3；total_cost_ms=120000 |
| index docs/chunks | Chroma，16 documents / 493 chunks |
| prediction schema / producer | 3 / `production_chat_service` |
| retrieval metric k | 5；是离线累计证据评分口径，不是固定工具 top_k |
| tool mode / permission | `rag_only` / `knowledge_base:read` |
| MCP | production_enabled=true；evaluation_enabled=false；`core_disabled` |

正式与诊断初始化均复用现有索引；正式日志记录 `num_added=0, num_updated=0,
num_skipped=423, num_deleted=0`。没有强制重建或改动索引算法。早期未初始化的
`config-snapshot.json` 中 14/0 是磁盘 manifest 的局部信息及缺省计数，不用它代替运行时的 16/493。

主要指纹：

- index manifest SHA-256: `148159a7077c5d5a340dcb35db5fa8b27f1575d788c57207923539c6ec0f3ead`
- RAG dataset SHA-256: `c5951e65e31f9b131379eb4ac99fd67a84aaa6bc6d6c947b634277d2ff25ccf0`
- Agent dataset SHA-256: `87d864e8c3dfffc783bb6602e4c9c1ae9e71b47600bacf0cb3589a5f7ab1fb0c`
- Agent prompt inputs combined SHA-256: `b579ffd1c6d46f1474a4434c79d7bce7096ccc0fe3f93c95712ae8c96a854661`
- tool contract SHA-256: `91ad843241a28c52773e34b85f473280395301cade9c8e5ba61b054051cc1c54`

## RAG V3 OVERALL

| 指标 | 正式结果 |
| --- | --- |
| questions | 计划 65；实际尝试 45；正常返回 44；runtime failure 1；未执行 20 |
| prediction coverage | **NOT EVALUATED**：CSV 未生成；不能把 44 次正常返回当作落盘覆盖率 |
| status counts | 仅日志观测：OK=44，MODEL_ERROR=1（q045） |
| retrieval doc hit@5 | NOT EVALUATED |
| gold chunk hit@5 / recall@5 / all-gold hit@5 | NOT EVALUATED |
| reference gold hit / recall / doc hit | NOT EVALUATED |
| avg retrieved chunks / references | 正式 Prediction v3 聚合 NOT EVALUATED |
| avg retrieval calls / tool calls / LLM calls | 正式 Prediction v3 聚合 NOT EVALUATED；日志观测见下表 |
| key_point_recall / Final Score semantic component | NOT EVALUATED；Judge 未运行，不代表完整答案质量 |

原运行日志共有 **61 次工具调用：55 次成功 search、2 次成功 compare、4 次 compare
错误**；共 **107 次 LLM 调用**。这些是已尝试 45 题的调用事实，不是 65 题质量指标。

## CATEGORY BREAKDOWN

所有类别的 doc hit@5、gold chunk hit@5/recall@5、reference gold hit/recall 均为
**NOT EVALUATED**。下表仅用日志中的成功 Search/Compare/Get 调用计数描述运行行为；
正式 Prediction 的 `retrieval_call_count` 未落盘，不补造该字段。
未执行类别的分布标记 N/A，不把未执行题计为 0 次检索。

| 类别 | 计划/尝试 | 状态分布 | 平均成功证据调用 | 平均工具/LLM 调用 | 0/1/2+ 次成功证据调用题数 |
| --- | --- | --- | --- | --- | --- |
| single_doc | 40/40 | OK=40 | 1.175 | 1.200 / 2.150 | 0 / 33 / 7 |
| cross_doc | 10/5 | OK=4，MODEL_ERROR=1 | 2.000 | 2.600 / 4.200 | 0 / 1 / 4 |
| partial_answer | 6/0 | NOT RUN | N/A | N/A | N/A |
| not_in_doc | 5/0 | NOT RUN | N/A | N/A | N/A |
| trap | 4/0 | NOT RUN | N/A | N/A | N/A |

来源：[逐题日志投影](../../outputs/evaluation_baseline_v2/agentic-rag-7b-diagnostics-bce52c0/log-case-observations.jsonl)、
[中止运行汇总](../../outputs/evaluation_baseline_v2/agentic-rag-7b-diagnostics-bce52c0/aborted-run-summary.json)。

## DYNAMIC DECOMPOSITION

原运行已尝试的 cross_doc 中，**4/5（80%）**有 2+ 次成功证据调用，single_doc 为
**7/40（17.5%）**。cross_doc 其余 5 题未运行，不能写成完整 10 题的比例；partial_answer
完全未执行。q044 一次 compare 正常完成，调用少不等于没有比较双文档。

正式 stdout 只保存工具顺序、状态和耗时，没有逐调用 query/top_k/返回 IDs，因此正式运行
“第二次检索新增 gold 次数”与有效多 query 案例数均 **NOT OBSERVABLE**。
不能从调用次数直接推断 effective dynamic decomposition。

只有 q045 获得两次独立诊断 trace，结论限于这两个样本：

| 样本 | 首次 search | 后续 search queries / top_k | 后续新增 gold |
| --- | --- | --- | --- |
| rerun 1 | `学生证丢失补办流程` / 5 | `内蒙古工业大学学生证补办管理办法` / 5；`内蒙古工业大学教务处你问我答 学生证补办` / 5 | 0、0 |
| rerun 2 | 同上 / 5 | `管理办法 学生证补办规定` / 3；`教务问答 学生证丢失补办` / 3 | 0、0 |

两次首次检索均已取得全部 gold `[365,387,388]`。rerun 1 后两次检索分别新增一个非 gold
chunk 325、零新 chunk；rerun 2 后两次均没有新增 chunk。Agent 确实修改了 query，
并在 rerun 2 将 search top_k 从 5 改为 3；但 **4 次后续 search 都没有新增 gold**，属于
`AGENT_PLANNING_REDUNDANT_RETRIEVAL` 的诊断效率信号，不能称为补齐子问题证据的成功例子。
本轮没有足够 trace 提供已确认的 effective dynamic decomposition 成功案例。

## Q049

- 正式结果：**NOT RUN**，Gate 1 在 q045 已停止。
- status、retrieval_call_count、tool_call_count、llm_call_count、reference_chunk_ids：N/A。
- targeted rerun：0 次；未预跑，未因历史波动增加预算。
- 归因：本轮未评估，不能沿用历史 `max_iterations_reached` 冒充当前结果。

## Q052

- 正式结果 / retrieved / referenced / answer / retrieval_call_count：**NOT RUN / N/A**。
- Judge：NOT RUN；targeted rerun：0 次。
- 归因：本轮不能验证过度拒答、partial-answer weakness 或其是否改善。

## Q054

- 正式结果：**NOT RUN**；targeted rerun：0 次。
- locator confusion / tool errors / redundant chunk reread：本题均 **NOT OBSERVED**，不写成零错误通过。
- 结论：本轮中止原因与 q054 旧 locator blocker 不同。Phase 7A 的修复未被本轮证据推翻，
  但本轮没有完成 q054 的正式回归确认；不引用旧 smoke 补充正式结果。

## EVIDENCE FUNNEL

| 漏斗问题 | 正式 65 题 | 独立 q045 诊断 |
| --- | --- | --- |
| gold evidence 未进入累计 retrieved | NOT EVALUATED；case IDs 未知 | 0/2 次；两次全部 gold 均已观察 |
| gold observed but not referenced（至少一个 gold 未引用） | NOT EVALUATED；case IDs 未知 | rerun 2 遗漏 gold chunk 388；rerun 1 全部引用 |
| 已观察 gold，但最终完全没有 gold 引用 | NOT EVALUATED；case IDs 未知 | 0/2 次正常诊断结果；不得把原 runtime failure 的空引用算入质量失败 |
| gold referenced but Judge key point missing | NOT EVALUATED | NOT EVALUATED；未送 Judge |

按原 evaluator 的 `resolve_gold_chunk_ids` / `score_evidence_chunks` 只读分析当前 Chroma，
q045 gold 为 `[365,387,388]`。rerun 1 observed=`[365,355,389,387,388,325]`，
referenced=`[389,387,388,365]`；rerun 2 observed=`[365,355,389,387,388]`，
referenced=`[365,387,389]`。两次累计 gold recall 都为 1；reference gold recall 分别为
1 与 2/3，两次引用均覆盖两份 gold 文档。这些是诊断样本指标，未写入正式 CSV。

来源：[q045-attribution.json](../../outputs/evaluation_baseline_v2/agentic-rag-7b-diagnostics-bce52c0/q045-attribution.json)。
引用遗漏不自动等于答案遗漏；没有 Judge 或明确人工语义核验，不宣称 answer synthesis failure。

## AGENT CORE

**NOT RUN — Gate 1 未完成。** Gate 0 实际校验当前 Agent 数据集为 28 个 active cases，
其中 Core=25、Online=3，覆盖 gaps=[]。

core executed case count=0；pass rate、first tool accuracy、status accuracy、tool sequence
accuracy、tool call success rate、avg tool calls、avg LLM calls、latency 均 **NOT EVALUATED**。
search、compare、get_chunk_text、list_documents、direct answer、clarify、no_evidence
的 Core 行为均未测量。RAG 中的工具调用不能代替 Agent Core baseline。

## SELECTIVE JUDGE

- status: **NOT RUN — BLOCKED_MISSING_BASELINE_PREDICTIONS**。
- case count: 0；category distribution: 未生成 subset。
- key point recall / missing / partial / contradicted / forbidden hits / behavior_label：NOT EVALUATED。
- 未创建 judge subset、未重新生成 predictions，未将 q045 runtime error 或诊断答案送 Judge。
- 停止原因是 Gate 1 失败，不是已验证的 Judge 服务错误。

## FAILURE ATTRIBUTION

[65 行归因矩阵](../../outputs/evaluation_baseline_v2/agentic-rag-7b-diagnostics-bce52c0/failure-attribution-matrix.csv)
逐题区分尝试、未执行、原运行状态与诊断次数；所有正式质量归因均保留 NOT EVALUATED。

### INFRASTRUCTURE / RUNTIME

**case IDs: q045（正式运行）。**

实际链为 `search(success) → compare(error) → search(success) → search(success)`，
工具调用 4 次、成功证据调用 3 次、LLM 调用 8 次，latency=137797ms；最终 references=0。
结束原因明确为 `Agent structured response referenced unavailable evidence`，没有
`max_iterations_reached`、`max_tool_errors_reached` 或 locator confusion 记录。
虽然耗时超过配置中的 120000ms，本次终止理由不是 budget exhaustion，不能用时间反推根因。

当前 [results.py](../../app/agent/results.py) 在 typed structured response 的 `evidence_ids`
不属于本次成功 ToolMessages 的 evidence 集合时 fail closed；普通生成文本没有被当作合法答案放行。
随后 [rag_predictions.py](../../scripts/evaluation/rag_predictions.py) 对 `MODEL_ERROR` 抛错，
且只在全部题目完成后写 CSV，故本次没有预测文件。两条边界均按当前合同工作。

正式日志没有保留失败的具体 evidence ID 和完整可用 ID 集合，因此不能进一步断言是
ID 拼写/截断、模型编造、结构化输出修正过程或 evidence 状态传递问题。
两次复跑保存了对应公开 ID 集合，非法 ID 数都为 0。

未见 API 认证、网络、Embedding、Reranker 或索引不可用的致命证据。
`tool audit write failed` 告警同样出现在成功调用；它不是本次终止原因。

### AGENT PLANNING

**正式 case IDs: q035、q042、q043、q045，限 compare 错误后 search fallback 信号。**
原日志没有参数，不能确认这四题都来自同一种文档名错误。

**诊断 case IDs: q045，`AGENT_PLANNING_REDUNDANT_RETRIEVAL`。** 两次都在首轮
已取得全部 gold 后继续检索。compare 的名称定位问题见效率章节。
没有确认 `AGENT_PLANNING_UNDER_RETRIEVAL` 或导致 gold miss 的
`AGENT_PLANNING_BAD_QUERY`。不据单题重复检索建议恢复第二 Planner 或调整规划 Prompt。

### RETRIEVAL

**正式 case IDs: NOT EVALUATED。** 没有确认
`RETRIEVER_MISS_DESPITE_REASONABLE_QUERY`。q045 两次诊断的 gold 均在首次 search
进入 observed，不能把本次运行阻断归于 Retriever；不调整 BM25、Embedding、rerank、top_k 或 chunking。

### EVIDENCE SELECTION

**正式 case IDs: NOT EVALUATED。诊断 case IDs: q045 rerun 2（部分 gold 未引用）。**
chunk 388 已观察但未引用，属于 `EVIDENCE_OBSERVED_BUT_NOT_REFERENCED` 的部分遗漏信号。
未证明它造成语义答案缺失，也未证明为跨题系统性问题。原 q045 的运行错误和空引用优先归 Runtime。

### ANSWER SYNTHESIS

**case IDs: NOT EVALUATED。** 未运行 Judge，也未进行可支持缺失/矛盾结论的完整人工语义评审。
不能由 reference recall、最终 `OK` 或答案长度推断答案质量。

### REFUSAL / PARTIAL ANSWER

**case IDs: NOT EVALUATED。** q051–q056、所有 not_in_doc/trap 均未执行。
不能判断已知部分回答、未知部分声明、整体过拒或 unsupported detail 的系统表现。

### MODEL VARIANCE

**case IDs: q045。** 同一 HEAD、Prompt、工具合同、模型配置和预算下，正式运行失败，
两次额外复跑成功。非法引用失败没有稳定复现，分类为 **MODEL_VARIANCE / UNCERTAIN**。
这是有限样本的行为差异，不是已识别底层机制，也不是可靠故障概率。

四种候选解释的证据边界：new deterministic blocker **未证实**；MODEL_VARIANCE
**已观察到**；external model/network blocker **无明确证据**；evaluation harness issue
**未发现**，现有 producer 正常执行 fail-fast 合同。

## EFFICIENCY FINDINGS

### REDUNDANT GET_CHUNK_TEXT

- 全部 65 题 redundant_chunk_reread_count：**NOT EVALUATED**，20 题未执行。
- 已尝试 q001–q045：get_chunk_text 调用 **0**，重复读取 **0**，affected qids=`[]`，额外工具/LLM 调用均为 0。
- q045 两次诊断：同样没有 get_chunk_text。
- q054 未执行，不能声称它的重复读取现象已消失。

### COMPARE_DOCUMENTS FALLBACK

正式已尝试的 45 题中：**attempts=6，success=2，errors=4**。
成功 qids=`[q006,q044]`；fallback-to-search qids=`[q035,q042,q043,q045]`。
其中 q035/q042/q043 最终 `OK`，q045 最终因引用校验失败中止；正式 evidence miss 未评估，
不能把 fallback 或 `OK` 直接解释为证据质量好坏。

q045 两次诊断各有一次 compare 错误：`doc_a="管理办法"`，`doc_b="教务问答"`。
在当前 493 个 chunk 上调用现有只读匹配函数，doc_a 匹配 4 份文档共 40 chunks，
doc_b 匹配 **0 chunks**；现有代码因此返回 `execution_error`。
后续 search 仍覆盖全部 gold，两次均正常结束，说明该名称定位错误可恢复；
没有证据表明它直接导致原运行的非法引用。

两次诊断合计 4 次后续 search 未新增 gold。公开模型计数显示每次均有一轮发出两次后续 search；
不能把两次工具调用等同于两次额外 LLM turn。没有反事实运行，无法精确量化可节省的 LLM turns。
本轮未优化 Prompt 或匹配算法。

## TARGETED RERUNS

仅执行 **1 个 qid × 2 次额外运行**；没有其他真实模型诊断或质量复跑。

| qid / run | status / reason | retrieval / tool / LLM calls | latency | reference chunk IDs |
| --- | --- | --- | --- | --- |
| q045 正式运行 | MODEL_ERROR / unavailable evidence | 3（日志） / 4 / 8 | 137.797s | 无有效引用；具体非法 ID 未保留 |
| q045 rerun 1 | OK / structured answer | 3 / 4 / 3 | 27.631s | 389,387,388,365 |
| q045 rerun 2 | OK / structured answer | 3 / 4 / 3 | 21.821s | 365,387,389 |

复用 `initialize_rag_for_predictions()`、`run_rag_once_with_result()` 和既有隔离设置，
空 session、RAG_ONLY；未绕过 producer 的错误检查。诊断进程仅在结果归一化边界加
透明观测 wrapper：原函数照常执行、原结果原样返回，保存工具参数/状态、evidence IDs、
chunk IDs、公开引用、模型调用计数。没有保存隐藏推理或重写答案，没有源码文件修改。

证据：[rerun 1](../../outputs/evaluation_baseline_v2/agentic-rag-7b-diagnostics-bce52c0/q045-rerun-1.json)、
[rerun 2](../../outputs/evaluation_baseline_v2/agentic-rag-7b-diagnostics-bce52c0/q045-rerun-2.json)、
[完整复跑日志](../../outputs/evaluation_baseline_v2/agentic-rag-7b-diagnostics-bce52c0/q045-targeted-reruns.log)、
[复跑汇总](../../outputs/evaluation_baseline_v2/agentic-rag-7b-diagnostics-bce52c0/targeted-summary.json)。

## CODE CHANGES

```text
Production code changes: NONE
Evaluation code changes: NONE
Prompt changes: NONE
Config changes: NONE
Dataset changes: NONE
Historical report changes: NONE
```

唯一新增的仓库报告为本文件；所有临时分析和诊断位于 Git 忽略的 `outputs/`。
真实调用结束后核对 **351 个原有 tracked 文件 SHA-256 全部不变**，HEAD 不变、
工作区仍 clean；随后才写报告。没有 commit、push、新建/切换分支、reset、rebase 或 merge。
没有改预算、恢复 Decompose LLM 或放宽 grounding。

## TESTS

| 检查 | 结果 |
| --- | --- |
| validate-only | PASS，退出码 0；当前 Agent 数据集 Core=25、Online=3，gaps=[] |
| RAG baseline | FAIL，退出码 1；q045 runtime failure |
| Agent Core | NOT RUN：Gate 1 未完成 |
| Judge | NOT RUN：没有正式 predictions |
| targeted reruns | q045 两次，OK / OK；诊断进程退出码 0 |
| 源码/数据/Prompt 完整性 | 原有 351 tracked 文件均不变；索引 manifest hash 不变 |
| git diff --check | PASS |
| 诊断产物、完整日志及本报告敏感值扫描 | PASS，未发现已配置敏感值泄漏 |
| full pytest / full lint / frontend tests | NOT RUN |
| Phase 7A focused tests / Phase 6 tests | NOT RERUN |
| Online / MCP / Mixed / Combined | NOT RUN |

Gate 0 的现有 `--validate-only` 只进行 Agent 数据合同校验，不冒充完整 RAG 评分。
最终核对见 [report-verification.json](../../outputs/evaluation_baseline_v2/agentic-rag-7b-diagnostics-bce52c0/report-verification.json)，
敏感值扫描见 [diagnostic-secret-scan.json](../../outputs/evaluation_baseline_v2/agentic-rag-7b-diagnostics-bce52c0/diagnostic-secret-scan.json)；
它们仅验证诊断产物与报告，不改变 Gate 1 的 FAIL 状态。
诊断配置快照准备中曾将 judge 属性名误写为不存在的 `rag_eval_judge_model_name`，
修正一次性命令为当前 `settings.judge_model` 后完成；未修改配置或源码，未触发模型调用，
与正式 q045 错误无关。

## FINDINGS

1. 当前正式运行被 **1 个 grounding runtime failure** 阻断，首个完整 v3 baseline 仍缺失。
2. q045 非法引用在两次受限复跑中均未复现；不支持立即进行确定性代码修复或增加预算。
3. 两次 q045 诊断均会修改 query，其中一次改变 search top_k；后续检索均没有新增 gold。
4. 原运行有 4 次 compare 错误后 search fallback；q045 诊断确认其中一种原因是文档别名匹配不到目标。
5. q045 rerun 2 存在部分 gold 已观察却未引用的信号，不能据此推断全量 Evidence Selection 或 Answer Synthesis 问题。
6. q049/q052/q054 和所有拒答/陷阱类别未执行，不能声称它们本轮通过或失败。

## DOMINANT FAILURE MODE

**RUNTIME_STABILITY**。

这里指本轮阻止 baseline 成立的首要失败层，**不是已经证明跨题系统性高频故障**。
支持证据为正式尝试 45 题中 q045 的 1 次 `MODEL_ERROR`（1/45），导致 20 题未运行、
65 行正式 Prediction CSV 未生成；同题 2 次诊断均 `OK`，提示波动而非稳定复现。
由于没有全量 predictions/Judge，不能选择 Retrieval、Evidence Selection、Answer Synthesis
或 Refusal 为主导质量失败模式。

## RECOMMENDED NEXT CHANGE

**NO PRODUCTION CHANGE YET.** 当前只支持继续定位最终 structured evidence ID 的运行稳定性。
下一次获授权的测量应在保持返回值及 fail-closed 语义不变的前提下，保留失败时公开的请求 ID
与可用 evidence ID 集合，以区分模型引用不稳定和 evidence 状态传递问题；本轮未实施这项改动。

不修改 planning/refusal Prompt、Retriever、文档匹配算法、预算或评测 fail-fast 合同。
两次诊断已达到本轮单题额度，不继续试到通过或自动重启 65 题。

## NEXT PHASE

**先解决 baseline 接受条件，暂不开始 AGENTIC-RAG-8。** 后续仅围绕 q045 最终引用
稳定性取得足以支持最小改动的证据，再在独立目录完成正式 Gate 1；之后才依次执行
Agent Core、复用正式 predictions 的 Selective Judge 和必要归因。没有 accepted v3 baseline，
本轮不制定回归阈值，也不进行 Phase 8 质量调优。
