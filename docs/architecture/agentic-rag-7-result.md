# AGENTIC-RAG-7 RESULT

## STATUS

**FAIL — 正式 65 题基线在 q054 中止，首个官方 Prediction v3 baseline 尚未建立。**

离线校验与单题真实模型 smoke 均通过。正式运行到 q054 时，Agent 一次检索后连续三次调用 `get_chunk_text` 失败，触发 `max_tool_errors_reached`，返回 `MODEL_ERROR`。Runner 按现有合同立即退出；此前 53 题的预测仅在进程内存中，未生成 `predictions.csv`，也没有正式评分与 manifest。

这次失败属于 **A. RUNTIME / TOOL USAGE FAILURE**，是本地工具定位参数混淆造成的运行阻断；没有证据把它归为 API 连接、认证、Embedding、Reranker 或索引不可用。不计入模型答案质量分。

本报告保留原运行事实与独立诊断结果，不以诊断结果补写或替换正式 baseline。Agent Core 和 Judge 的前置 Gate 未满足，均未执行。

## BASELINE

- branch: `refactor/tool-output-models`
- head: `2b638117037ea45c0a2affc7295f89591668e3c3`
- short SHA: `2b63811`
- Python: `3.12.3`，项目 `.venv/bin/python`
- worktree_before: **clean**；全部真实调用结束后、写报告前仍 clean。
- rag_output: `outputs/evaluation_baseline_v2/agentic-rag-7-rag-2b63811/`。正式运行退出码 **1**；目前仅有独立 `analysis/`，没有正式生成的 manifest、predictions、per-question report、summary 或 failure_cases。
- agent_output: 计划为 `outputs/evaluation_baseline_v2/agentic-rag-7-agent-2b63811/`，**未运行、未创建**。
- diagnostics: `outputs/evaluation_baseline_v2/agentic-rag-7-diagnostics-2b63811/`。
- 运行日期：2026-09-18，Asia/Shanghai。原运行日志从 09:43:18 的索引初始化到 09:53:09 的 q054 中止。

实际执行的正式命令：

```bash
.venv/bin/python scripts/run_evaluation_baseline.py --validate-only

.venv/bin/python -u scripts/run_evaluation_baseline.py \
  --rag-only \
  --rag-eval-file data/eval/rag_eval_questions_v2_semantic_fixed.json \
  --output-dir outputs/evaluation_baseline_v2/agentic-rag-7-rag-2b63811 \
  --retrieval-k 5
```

正式运行 stdout/stderr 保存在 [rag_baseline.log](../../outputs/evaluation_baseline_v2/agentic-rag-7-diagnostics-2b63811/rag_baseline.log)。未使用 `--allow-dirty`、`--force-rebuild`、`--use-llm-judge` 或任何 Online/MCP/Mixed 开关；未运行 Combined。

Runner 在单题错误状态处抛错，并且只在整批预测完成后写 CSV，详见 [rag_predictions.py](/home/wabce94123/projects/rag-agent-qa-system/scripts/evaluation/rag_predictions.py:104) 的错误保护与第 153–177 行的写入顺序。正式 manifest 的构建位于 RAG 返回之后，见 [runner.py](/home/wabce94123/projects/rag-agent-qa-system/scripts/evaluation/runner.py:400)。没有尝试修改这个合同、跳过 q054、手工补预测或重跑 65 题。

## REAL-MODEL CONFIG

配置来自现有 `build_manifest()` 自动生成的 [smoke_manifest.json](../../outputs/evaluation_baseline_v2/agentic-rag-7-diagnostics-2b63811/smoke_manifest.json)，明确属于 Gate 0 配置快照，不冒充 Gate 1 正式 manifest。

| 项目 | 实际配置 |
| --- | --- |
| answer model | `qwen-plus` |
| query rewrite model | `deepseek-chat`，enabled=true；本轮空 session 的 RAG 调用未触发 rewrite |
| judge model | `qwen-plus`，已配置；Judge 未调用 |
| embedding model | `Qwen/Qwen3-Embedding-0.6B` |
| rerank model | `BAAI/bge-reranker-v2-m3` |
| prediction schema | 3；producer=`production_chat_service` |
| retrieval metric K | 5，仅离线评分截取前 K 个有序去重证据，不控制工具 top_k |
| temperature / max tokens | 0.1 / 1024 |
| Agent budgets | iterations=10，tool_calls=10，tool_errors=3，max_total_cost_ms=120000 |
| index | Chroma，16 documents / 493 chunks；启动记录 force_rebuild=False、corpus unchanged |
| MCP | production_enabled=true，evaluation_enabled=false，`core_disabled` |

指纹：

- index manifest SHA-256: `148159a7077c5d5a340dcb35db5fa8b27f1575d788c57207923539c6ec0f3ead`
- Agent prompt inputs combined SHA-256: `f79ee3708589ffa16f1afb7b9593b2a6a3889f213a987a7310e27a6e369d524a`
- tool contract SHA-256: `8c90a417d618dda5e86f84ad6b3dccf5c9966ceb42fa761720a1363698f06d2d`
- RAG dataset SHA-256: `c5951e65e31f9b131379eb4ac99fd67a84aaa6bc6d6c947b634277d2ff25ccf0`

[source_integrity.json](../../outputs/evaluation_baseline_v2/agentic-rag-7-diagnostics-2b63811/source_integrity.json) 确认报告前 HEAD、工作区、索引 manifest hash 与所有 manifest 中的 Prompt 输入文件均保持一致。没有修改生产配置、Prompt、工具 schema、retrieval 参数、grounding 或预算。

## RAG V3 SUMMARY

| 指标 | 正式结果 |
| --- | --- |
| questions | 计划 65；实际尝试 54；正常结果 53；运行失败 1；未执行 11 |
| prediction coverage | **NOT EVALUATED**：正式 predictions 未生成，不能以正常返回数冒充文件覆盖率 |
| status counts | 原运行日志：OK=52，NO_RELEVANT_INFO=1（q052），MODEL_ERROR=1（q054） |
| retrieval doc hit@5 | NOT EVALUATED |
| gold chunk hit@5 / recall@5 | NOT EVALUATED |
| reference gold hit | NOT EVALUATED |
| average retrieval / tool / LLM calls | 正式 Prediction v3 聚合未生成；下节仅提供日志观测 |
| Final Score / key-point metrics | NOT EVALUATED；没有 Judge，不作完整答案质量解释 |

原运行日志可核对到 80 次工具调用：71 次成功 search、1 次成功 compare、5 次 compare 错误、3 次 get_chunk_text 错误；129 次 LLM 调用。它们是已尝试的 54 题的运行事实，不是 65 题质量指标。

## CATEGORY BREAKDOWN

以下调用分布来自原运行的公开 `tool_call_chain` 日志。`成功证据调用` 统计成功的 search/compare/get_chunk_text，不等同于已生成的 Prediction 字段；正式 predictions 不存在。分母仅限实际执行的题，未执行题不按 0 次检索计算。各类 retrieval hit、reference hit 均 **NOT EVALUATED**。

| category | 计划/已尝试 | 状态 | 成功证据调用分布 | 平均/中位/最大 | 0/1/2+ 次题数 | 平均工具/LLM 调用 |
| --- | --- | --- | --- | --- | --- | --- |
| single_doc | 40/40 | OK=40 | 1次×34题, 2次×6题 | 1.150/1/2 | 0/34/6 | 1.150/2.125 |
| cross_doc | 10/10 | OK=10 | 2次×8题, 1次×1题, 3次×1题 | 2.000/2/3 | 0/1/9 | 2.500/3.200 |
| partial_answer | 6/4 | OK=2, NO_RELEVANT_INFO=1, MODEL_ERROR=1 | 2次×2题, 1次×2题 | 1.500/1.5/2 | 0/2/2 | 2.250/3.000 |
| not_in_doc | 5/0 | NOT EXECUTED | N/A | N/A | N/A | N/A |
| trap | 4/0 | NOT EXECUTED | N/A | N/A | N/A | N/A |

完整日志投影见 [aborted_run_summary.json](../../outputs/evaluation_baseline_v2/agentic-rag-7-rag-2b63811/analysis/aborted_run_summary.json) 与 [log_case_observations.jsonl](../../outputs/evaluation_baseline_v2/agentic-rag-7-rag-2b63811/analysis/log_case_observations.jsonl)。

## MULTI-SUBQUESTION ANALYSIS

真实 Agent **会使用多轮检索**。原运行 10 道 cross_doc 中，9 道有 2 次以上成功证据调用；q044 用 1 次 compare 返回双文档证据，不应因只有 1 次调用判失败。single_doc 的对应观测为 6/40。partial_answer 只执行了 4/6，其中 2 题有 2 次成功证据调用；不能推断完整 6 题表现。

由于原 predictions 未落盘，不能回答原运行“10 道 cross_doc 中有多少未覆盖全部 gold docs”。独立复跑的 6 个 cross_doc 样本均观察到全部目标文档；5 个正常返回样本中，4 个引用覆盖全部目标文档，q041 没有引用目标《缓考、免考办理流程通知》，而引用了另一个汇编文档。q049 复跑达到迭代上限，其空引用属于运行中止，不参与引用质量判断。

成功补齐的具体路径包括 q042 分别检索 2023 年通知与 2018 年办法、q043 分别检索学业警示办法与教务问答。这证明真实模型能按不同信息需求更换 query；不等同于证明全部复杂问题答案完整。

## AGENT CORE SUMMARY

**NOT RUN — Gate 1 未成功。** `--validate-only` 确认当前 Agent 数据集为 28 个 active cases，其中 core=25、online=3。按要求未越过 Gate 1 直接运行 Agent-only。

pass rate、first-tool accuracy、status accuracy、tool-sequence accuracy、tool success、avg tool calls、avg llm calls、latency 全部 **NOT EVALUATED**。不能把 RAG 题中的工具使用统计当作 Agent Core 基线。

## SELECTIVE JUDGE

- judge status: **NOT EVALUATED — BLOCKED_MISSING_BASELINE_PREDICTIONS**。
- case count: 0；计划的 25–35 题子集未生成、未评分。
- categories / key-point recall / forbidden results / missing / partial / contradicted key points / behavior_label: NOT EVALUATED。
- 原因是 Gate 1 没有 `rag/predictions.csv`，不是 Judge 配置缺失或已验证的外部服务错误。
- 未调用 `run_predictions()` 重新生成 Judge 输入；也未用诊断答案冒充 Gate 1 predictions。

## FAILURE ATTRIBUTION

[65 行 Failure Attribution Matrix](../../outputs/evaluation_baseline_v2/agentic-rag-7-rag-2b63811/analysis/failure_attribution_matrix.csv) 保留全部 qid：原运行 q054 为 A；其余已执行题的质量归因未评估，q055–q065 未执行。下面的进一步归因均单独标明诊断证据，未替换原运行。

### INFRASTRUCTURE

**case IDs: q054（原运行与两次复跑），q049（仅诊断复跑）。**

q054：原运行 1 次 search 成功后 3 次 get_chunk_text 失败，`MODEL_ERROR / max_tool_errors_reached`。两次复跑均完全复现该链，均为 4 次工具、4 次 LLM 调用。

两次复跑 query 均为 `学业警示 通知书 模板`，top_k=5，Observed=`[382,239,383,384,385]`。随后分别以 Document.id 调用 get_chunk_text，policy_decision 全为 allow，error_code 全为 execution_error。现有只读查找函数的核对结果：

| 错用的 Document.id（前缀） | 对应 metadata.chunk_id | 用 Document.id 查找 | 用 metadata.chunk_id 查找 |
| --- | --- | --- | --- |
| knowledge_base:ef463ed6… | 382 | found=false | found=true |
| knowledge_base:105ada17… | 239 | found=false | found=true |
| knowledge_base:0ea8b6af… | 383 | found=false | found=true |

源代码 [get_chunk_full_text](/home/wabce94123/projects/rag-agent-qa-system/app/rag/debug.py:37) 按 metadata.chunk_id 比较；[chunk_to_evidence_document](/home/wabce94123/projects/rag-agent-qa-system/app/tools/knowledge_base.py:75) 另外生成稳定 Document.id。模型在实际调用中混淆了这两个标识。现有合同未承诺用 Document.id 定位 chunk，因此不能仅凭这一现象宣称 handler 回归，也不能误报为 API 服务故障。[结果映射](/home/wabce94123/projects/rag-agent-qa-system/app/agent/results.py:145) 会将这类预算停止映射为 MODEL_ERROR。

证据：[复跑 1](../../outputs/evaluation_baseline_v2/agentic-rag-7-diagnostics-2b63811/q054-rerun-1.json)、[复跑 2](../../outputs/evaluation_baseline_v2/agentic-rag-7-diagnostics-2b63811/q054-rerun-2.json)、[定位核对](../../outputs/evaluation_baseline_v2/agentic-rag-7-diagnostics-2b63811/q054-locator-diagnosis.json)。完整 ID 在文件中。诊断期没有修改函数、注入替代 ID 或执行修复。

q049：原运行 OK，诊断为 `MODEL_ERROR / max_iterations_reached`，LLM=10、工具=3，耗时约 170 秒。已取得两份目标文档，gold chunk recall=3/4；运行中止，不把空引用或安全停止文案判成答案质量失败。未保存足够的结构化响应重试细节来断言迭代耗尽根因，不能凭此提高预算。

日志中另有 `tool audit write failed` 噪声，但成功 search 同样出现此告警；q054 的实际工具错误为 execution_error。不能把这些审计告警当成已确认的致命原因。

### RETRIEVAL MISS

**正式 65 题：NOT EVALUATED。诊断中未确认 B 类“完全未命中 gold”的 case。**

q054 复跑命中 gold `[382,383]`，遗漏 `[386]`，现有 evaluator 的 full-observation gold recall=2/3；q049 复跑遗漏 `[380]`。这只证明局部证据未齐，尚未验证“合理 query、多轮检索仍稳定找不到”，不足以建议修改 BM25、Embedding、reranker、chunking 或 top_k。

### AGENT PLANNING MISS

**case IDs: q042、q043、q049（定位失败/额外调用）；尚未确认稳定的 under-retrieval。**

原运行 q042/q043/q045/q049 共 5 次 compare 错误。诊断 q042/q043/q049 各 1 次 compare 错误，传入的 doc_a、doc_b 在现有文档匹配函数中均匹配到 0 个 chunk；之后能改 query 检索。q042/q043 最终观察到全部 gold chunk。问题属于已观察到的文档定位和调用效率现象，不是“Agent 不会分解问题”。

不能把 q049 缺少一个 gold chunk或 q054 中止前只检索一次直接归成规划漏检；运行失败优先归 A。没有足够证据满足修改 retrieval-completeness Prompt 的多 case 稳定缺陷条件。

### EVIDENCE OBSERVED BUT NOT REFERENCED

**诊断 case IDs: q052；q041 为跨文档引用覆盖候选问题。**

q052 的 gold `[375,376]` 全部 observed，reference=[]，属于 evidence observed but not referenced；主要归因是下述整体拒答。

q041 observed 包含全部 gold `[2,361]`，但 references=`[361,232]`，缺少目标通知的 chunk 2。两份目标文档均观察到，最终只引用其中一份目标文档加汇编文档。属于严格 gold/source 选择缺口，不应归给 Retriever；尚未据此断言最终答案错误或形成“大量 case 的引用弱点”结论。

q042/q044/q046 也没有引用全部 gold chunk，但分别引用了全部目标文档；gold 可能含替代/重叠证据，少引用不自动等于错误。q049/q054 的运行中止空引用不纳入 D 类质量判断。

### ANSWER SYNTHESIS

**case IDs: NOT EVALUATED。** 没有可复用的 Gate 1 predictions 和 Judge 结果，也未完成逐关键点人工验证。不能用 raw gold chunk recall 或关键词缺失下 synthesis 结论。

### REFUSAL / PARTIAL ANSWER

**诊断 case ID: q052，REFUSAL_POLICY_WEAKNESS（限定单题证据）。**

原运行 q052 为 NO_RELEVANT_INFO；一次独立复跑仍为该状态。复跑两次 search（top_k 均 5）观察到全部 gold `[375,376]`，最终只答“我无法根据当前取得的证据确定答案。”且无引用。

人工对照原 gold 及其证据：已知部分包括学院制定实施细则、面试/实操/笔试形式以及适用类别的 30%+70% 综合成绩规则；未知部分是具体面试题和评分表。复跑没有回答任何已知部分，因此可以确认该样本整体拒答过多。它不是 retrieval miss，也没有证据显示它编造未知面试题。没有将此人工判断伪装成 Judge key-point 数值。

not_in_doc 5 题、trap 4 题未执行；完整 partial_answer 6 题结论 **NOT EVALUATED**。

### MODEL VARIANCE

**case ID: q049，MODEL_VARIANCE / UNCERTAIN。** 原运行 OK，诊断复跑达到 10 次模型调用上限。确认同版本存在结果变化，未证明概率、根因或稳定的应用回归。q054 原运行与两次复跑均失败，不应以一次性波动解释。

## CROSS-DOC TRACE SAMPLES

选择了原运行“有工具错误”的 q042/q043/q049 与“无工具错误”的 q041/q044/q046 各 3 题，全部只复跑一次。这是**工具执行分组**，不是由缺失的正式评分捏造出 3 个答案质量 PASS 与 3 个 FAIL。正式 success/failure 样本要求因此仍未完整满足。

以下只记录公开参数、结果 metadata、evidence IDs 和最终状态。observed/reference recall 复用现有 `score_evidence_chunks()`，对诊断的全部 observed IDs 评分，不是正式 hit@5，也没有新增评分公式。

### q042

[完整诊断](../../outputs/evaluation_baseline_v2/agentic-rag-7-rag-2b63811/analysis/trace-q042.json)。最终状态 `OK`；reason=`Agent model produced structured answer`。

- Call 1: `compare_documents`，args=`{"doc_a": "2023年本科教学管理调整通知", "doc_b": "2018年本科学生学籍管理办法", "query": "普通本科学生学习年限规定"}`；top_k 未显式传入，schema 默认每文档 3；status=`error`，error_code=`execution_error`。
- Call 2: `search_knowledge_base`，args=`{"query": "2023年普通本科学生学习年限调整通知", "top_k": 5}`；status=`success`，error_code=`<none>`；returned chunk IDs=[9, 7, 371, 228, 8]。
- Call 3: `search_knowledge_base`，args=`{"query": "2018年本科学生学籍管理办法 学习年限", "top_k": 5}`；status=`success`，error_code=`<none>`；returned chunk IDs=[228, 42, 7, 6, 310]。

Observed: `[9, 7, 371, 228, 8, 42, 6, 310]`。Referenced: `[7, 371, 310]`。

诊断 observed gold chunk recall=100.00%；reference gold chunk recall=75.00%。全部目标文档 observed=是；全部目标文档 referenced=是。答案语义 PASS/FAIL 未评分。

### q043

[完整诊断](../../outputs/evaluation_baseline_v2/agentic-rag-7-rag-2b63811/analysis/trace-q043.json)。最终状态 `OK`；reason=`Agent model produced structured answer`。

- Call 1: `compare_documents`，args=`{"doc_a": "学业警示办法", "doc_b": "教务问答", "query": "学业警示的定义与等级说明"}`；top_k 未显式传入，schema 默认每文档 3；status=`error`，error_code=`execution_error`。
- Call 2: `search_knowledge_base`，args=`{"query": "学业警示办法", "top_k": 5}`；status=`success`，error_code=`<none>`；returned chunk IDs=[381, 382, 386, 383, 385]。
- Call 3: `search_knowledge_base`，args=`{"query": "教务问答 学业警示 定义 等级", "top_k": 5}`；status=`success`，error_code=`<none>`；returned chunk IDs=[346, 381, 382, 383, 385]。

Observed: `[381, 382, 386, 383, 385, 346]`。Referenced: `[381, 346]`。

诊断 observed gold chunk recall=100.00%；reference gold chunk recall=100.00%。全部目标文档 observed=是；全部目标文档 referenced=是。答案语义 PASS/FAIL 未评分。

### q049

[完整诊断](../../outputs/evaluation_baseline_v2/agentic-rag-7-rag-2b63811/analysis/trace-q049.json)。最终状态 `MODEL_ERROR`；reason=`max_iterations_reached`。

- Call 1: `compare_documents`，args=`{"doc_a": "2025年本科转专业管理办法", "doc_b": "教务问答", "query": "转专业条件、流程、时间节点、特殊情形处理"}`；top_k 未显式传入，schema 默认每文档 3；status=`error`，error_code=`execution_error`。
- Call 2: `search_knowledge_base`，args=`{"query": "2025年本科转专业管理办法", "top_k": 3}`；status=`success`，error_code=`<none>`；returned chunk IDs=[375, 376, 377]。
- Call 3: `search_knowledge_base`，args=`{"query": "教务问答 转专业", "top_k": 3}`；status=`success`，error_code=`<none>`；returned chunk IDs=[360, 340, 39]。

Observed: `[375, 376, 377, 360, 340, 39]`。Referenced: `[]`。

诊断 observed gold chunk recall=75.00%；reference gold chunk recall=0.00%。全部目标文档 observed=是；全部目标文档 referenced=否。运行失败，引用数值仅描述停止结果，不参与质量归因。

### q041

[完整诊断](../../outputs/evaluation_baseline_v2/agentic-rag-7-rag-2b63811/analysis/trace-q041.json)。最终状态 `OK`；reason=`Agent model produced structured answer`。

- Call 1: `search_knowledge_base`，args=`{"query": "学生缓考办理条件和流程", "top_k": 5}`；status=`success`，error_code=`<none>`；returned chunk IDs=[2, 3, 4, 5, 361]。
- Call 2: `search_knowledge_base`，args=`{"query": "哪些课程不能办理缓考", "top_k": 3}`；status=`success`，error_code=`<none>`；returned chunk IDs=[361, 232, 349]。

Observed: `[2, 3, 4, 5, 361, 232, 349]`。Referenced: `[361, 232]`。

诊断 observed gold chunk recall=100.00%；reference gold chunk recall=50.00%。全部目标文档 observed=是；全部目标文档 referenced=否。答案语义 PASS/FAIL 未评分。

### q044

[完整诊断](../../outputs/evaluation_baseline_v2/agentic-rag-7-rag-2b63811/analysis/trace-q044.json)。最终状态 `OK`；reason=`Agent model produced structured answer`。

- Call 1: `compare_documents`，args=`{"doc_a": "学生管理规定", "doc_b": "纪律处分实施办法", "query": "学生处分设置期限和解除处分后权益恢复"}`；top_k 未显式传入，schema 默认每文档 3；status=`success`，error_code=`<none>`。

Observed: `[280, 279, 275, 289, 284, 287]`。Referenced: `[280, 289]`。

诊断 observed gold chunk recall=100.00%；reference gold chunk recall=50.00%。全部目标文档 observed=是；全部目标文档 referenced=是。答案语义 PASS/FAIL 未评分。

### q046

[完整诊断](../../outputs/evaluation_baseline_v2/agentic-rag-7-rag-2b63811/analysis/trace-q046.json)。最终状态 `OK`；reason=`Agent model produced structured answer`。

- Call 1: `search_knowledge_base`，args=`{"query": "研究生学位论文复制比检测到论文评审的前置要求", "top_k": 5}`；status=`success`，error_code=`<none>`；returned chunk IDs=[401, 402, 400, 398, 399]。
- Call 2: `search_knowledge_base`，args=`{"query": "研究生学位论文答辩评审前的前置条件和流程", "top_k": 3}`；status=`success`，error_code=`<none>`；returned chunk IDs=[401, 407, 412]。

Observed: `[401, 402, 400, 398, 399, 407, 412]`。Referenced: `[401, 407]`。

诊断 observed gold chunk recall=100.00%；reference gold chunk recall=66.67%。全部目标文档 observed=是；全部目标文档 referenced=是。答案语义 PASS/FAIL 未评分。

## TOKEN / CALL EFFICIENCY

- 原运行已尝试 54 题：成功证据调用=72，工具调用=80，LLM 调用=129；平均分别为 1.333、1.481、2.389。包含 q054 运行失败，不能作为 65 题正式平均值。
- 原运行 cross_doc：成功证据调用平均 2.0、工具平均 2.5、LLM 平均 3.2；single_doc 对应 1.15、1.15、2.125。
- 诊断执行 8 个不同 case、9 次调用（q054 两次，其余各一次）：工具调用=24、LLM 调用=37。没有任何 case 超过 2 次追加运行，没有重复整套 baseline。
- 包括 smoke 的本次可核对 LLM 调用总数：2+129+37=168。没有保存可靠 token usage，token 总量 **NOT RECORDED**，不从调用次数估算 token 或费用。
- 明确的额外调用：原运行 5 次 compare 错误；q054 三次定位失败后达到错误预算。q049 诊断 10 次 LLM 调用却仅 3 次工具调用，属于需保留的效率/运行稳定性信号。
- 已保存诊断的 cross_doc search top_k：5 出现 6 次，3 出现 4 次；compare 4 次都省略 top_k（当前 schema 默认 3），其中 3 次工具失败。partial_answer 诊断 search top_k=5 共 4 次，包含 q054 的两次复跑。
- 原运行没有持久化逐题 tool args；smoke 依要求只保留安全摘要。因此 single_doc 的实际 top_k 分布与全 65 题 top_k 分布 **NOT OBSERVED**，不能把返回 5 个 chunk当作模型显式选择 top_k=5。
- q042/q043 等两轮 query 不同且补充新 gold evidence，不应算无意义重复检索。当前没有证据支持“复杂问题总取很小 top_k”或“简单问题持续取 10”。

详情见 [diagnostic_summary.json](../../outputs/evaluation_baseline_v2/agentic-rag-7-rag-2b63811/analysis/diagnostic_summary.json)。

## V3 BASELINE DECLARATION

**The first official real-model Agentic RAG Prediction v3 baseline has NOT been established by this run.**

本阶段目标仍是建立首个官方 v3 真实模型基线；不能在缺少正式 predictions、manifest 与完整评分时声明已经建立。

Historical v2 raw scores are not strictly comparable. 本报告没有计算 v2→v3 分数差异、regression 百分比或任何新的通过阈值。

## CODE CHANGES

Production code changes: **NONE**。

配置、Prompt、tool schema、retrieval 参数、budgets、grounding、数据集及评估 Python 源码修改：**NONE**。没有新增 smoke script，也没有改生产 trace 或 Prediction schema。

唯一新增的版本化候选 artifact：`docs/architecture/agentic-rag-7-result.md`。其余产物均位于 Git 忽略的 `outputs/`。未 commit、push、切换/创建分支、reset、rebase 或 merge。

## TESTS

| 检查 | 实际结果 |
| --- | --- |
| validate-only | PASS；28 active / 25 Core / 3 Online，coverage gaps=[] |
| single-case real-model smoke | PASS；q001，status=OK，retrieved=[2,1,4,3,5]，references=[2]，retrieval/tool/LLM=1/1/2 |
| real-model RAG baseline | FAIL；q054 中止，退出码 1，53 个正常返回但无正式 predictions |
| real-model Agent Core | NOT RUN，Gate 1 failed |
| selective Judge | NOT EVALUATED，没有 Gate 1 predictions |
| targeted reruns | 8 cases / 9 executions；q054 两次复现；6 cross_doc trace；q052 一次拒答诊断 |
| read-only locator check | 3 个 Document.id 查找失败，对应 3 个 metadata.chunk_id 查找成功；索引 manifest 未变 |
| source integrity | 报告前 clean；HEAD、Prompt 输入和索引 manifest hash 未变 |
| report-only verification | 本地链接、JSON/CSV 结构、65 个 qid、诊断次数和 git diff --check |

没有重跑 Phase 6 的 268 tests、full pytest、frontend tests 或全仓 lint；没有运行 Online、MCP、Mixed、Combined，也没有重建索引或调参。

## FINDINGS

1. **INFRASTRUCTURE_BLOCKER / runtime execution blocker：q054。** 原运行和两次诊断均触发 max_tool_errors；两次 trace 与只读定位核对确认 Document.id 被当作 chunk_id。阻断了整批正式产物生成，优先级高于质量调优。
2. **真实多轮检索能力已观测，但完整基线质量未知。** 原运行 cross_doc 9/10 有多次成功证据调用；q042/q043 诊断按不同 query 补齐 gold，不能据此默认需要恢复独立 Decompose LLM。
3. **REFUSAL_POLICY_WEAKNESS：q052 的局部证据。** 诊断 gold 已全部观察，仍整体拒答；原运行也为 NO_RELEVANT_INFO。保留为后续有效基线中的重点观察项，不把单题直接扩展成主导系统问题。
4. **引用与运行波动仍需完整产物。** q041 诊断存在目标来源引用缺口；q049 从原运行 OK 变为迭代耗尽。样本范围和运行错误优先级已分别记录，未据此宣称稳定的 synthesis/selection regression。

## RECOMMENDED NEXT CHANGE

**INFRASTRUCTURE**，此处具体指恢复正式基线可完成的执行条件，而不是泛指外部模型服务故障。

支持证据：q054 连续三次运行中止；两次复跑公开 tool args 可重现 Document.id/chunk_id 混淆；只读查询证明数字 chunk locator 存在。现有 Runner 遇该状态整批退出且不保留已完成预测，导致后续 Gate 缺少输入。

下一步应在另行明确的修复/执行任务中优先处理这一已测量的工具定位调用阻断，再完成 Phase 7 正式评估。本阶段未指定采用 Prompt、schema 或 handler 的哪一种修改，也未实际修复；现有证据不足以把检索召回、Prompt completeness、引用或答案合成定为全系统主导弱点。不能靠增大预算、跳过失败题或放宽 dirty 保护绕过它。

## NEXT PHASE

**继续完成 Phase 7 的前置工作；本次不启动 AGENTIC-RAG-8。**

缺少完整 v3 baseline，因此既不进入质量调优，也不建立 v3 regression thresholds。保留本次中断目录与独立诊断；后续正式尝试须使用新的空输出目录，不能覆盖或拼装本次原始记录。
