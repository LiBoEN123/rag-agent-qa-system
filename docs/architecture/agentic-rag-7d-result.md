# AGENTIC-RAG-7D RESULT

## STATUS

**PASS_WITH_FINDINGS**。首个正式 Prediction v3 real-model baseline 已建立。65 题全部完成并生成 prediction，runtime failures=0。正式评价产物、manifest 和 reports 均完成。系统质量发现单独记录，不能把 baseline 成立解释为每道题质量通过。

## BASELINE

- branch: `refactor/tool-output-models`
- head: `b6f018a0763039e3896398e3cd83d790af8e3595`
- worktree_before: **clean**；正式 RAG 与 Agent Core manifests 均为 `git_dirty=false`。
- rag_output: [agentic-rag-7d-rag-b6f018a](../../outputs/evaluation_baseline_v2/agentic-rag-7d-rag-b6f018a/)
- agent_output: [agentic-rag-7d-agent-b6f018a](../../outputs/evaluation_baseline_v2/agentic-rag-7d-agent-b6f018a/)
- judge_output: [analysis/judge](../../outputs/evaluation_baseline_v2/agentic-rag-7d-rag-b6f018a/analysis/judge/)
- diagnostics: [agentic-rag-7d-diagnostics-b6f018a](../../outputs/evaluation_baseline_v2/agentic-rag-7d-diagnostics-b6f018a/)
- 运行目录首次创建时均不存在；没有复用或覆盖历史输出。RAG 于 2026-09-18 21:11–21:25 完成；Agent Core 于 2026-09-19 09:42–09:46 完成（Asia/Shanghai）。
- 继续任务时核实 RAG 已正常完成，没有重跑或拼接 65 题。两次 manifest 的 code、rag、agent、mcp、prompts 部分完全一致。

## V3 BASELINE DECLARATION

**Official Prediction v3 baseline: ESTABLISHED.**

This is the first official real-model Agentic RAG Prediction v3 baseline.

**Historical v2 raw scores are not strictly comparable.** 不计算 v3 − v2 性能变化，不在本阶段建立 CI threshold。

| 项目 | 结果 |
| --- | --- |
| Successful predictions | 65 |
| Runtime failures | 0 |
| Total accounted / per-question rows | 65 / 65 |
| Prediction coverage | 100% |

12 个必需文件全部存在：`rag/predictions.csv`、`rag/prediction_failures.jsonl`（空文件）、`rag/per_question_report.csv`、`rag/summary_report.csv`、`rag/summary_report.json`、`rag/llm_judge_details.jsonl`、`manifest.json`、`baseline_summary.json`、`baseline_report.md`、`failure_cases.jsonl`、`coverage_report.json`、`secret_scan.json`。已验证 qid 唯一、65 题集合完整、schema=3；正式文件指纹见 [official-integrity.json](../../outputs/evaluation_baseline_v2/agentic-rag-7d-rag-b6f018a/analysis/official-integrity.json)。

## REAL-MODEL CONFIG

| 项目 | 配置 |
| --- | --- |
| Answer model | qwen-plus |
| Query rewrite | enabled；deepseek-chat；RAG 空 session 未调用 rewrite LLM；Core 多轮题实际调用 |
| Selective Judge | qwen-plus，temperature=0，max_tokens=2048 |
| Embedding | Qwen/Qwen3-Embedding-0.6B |
| Rerank | BAAI/bge-reranker-v2-m3 |
| Answer temperature / max_tokens | 0.1 / 1024 |
| Budgets | iterations=10；tool_calls=10；tool_errors=3；连续同工具调用/错误上限均为3；total_cost_ms=120000 |
| Index documents / chunks | Chroma，16 / 493 |
| Prediction schema / producer | 3 / production_chat_service |
| Retrieval metric k | 5；累计去重观察证据的前5项，不控制工具 top_k |
| RAG tool mode / permission | rag_only / knowledge_base:read |
| MCP | production=true；evaluation=false；core_disabled |

RAG、Core 均复用当前索引；初始化同步日志为 added=0、updated=0、skipped=423、deleted=0。没有强制重建、修改 Chroma 配置或检索参数。Judge 与诊断复用现有 Settings，不改变生产配置。

## RUNTIME STABILITY

| 状态 | count | rate / 65 |
| --- | --- | --- |
| MODEL_ERROR | 0 | 0% |
| MODEL_BAD_OUTPUT | 0 | 0% |
| TOOL_ERROR | 0 | 0% |
| INTERNAL_ERROR | 0 | 0% |

prediction_failure_count=0；runtime_failure_rate=0/65=0%；failure case IDs=NONE。正式状态分布：OK=58，NO_RELEVANT_INFO=7。后者是有效 prediction，不等于 runtime failure。Failure sidecar 平均调用数为 N/A（0 个样本）。

## GROUNDING FAILURES

**NONE**。本轮 sidecar 为空，没有 requested / available / unavailable ID mismatch 可供 A–E 形态分类；不虚构分类。q045 的历史错误未复现，MODEL_VARIANCE 仍是合理可能，不能据此证明旧错误机制已被确定或修复。

## RAG V3 OVERALL

| 指标 | 结果 |
| --- | --- |
| Questions | 65 |
| Prediction coverage | 100.00% |
| Retrieval doc hit@5 | 98.46% |
| Retrieval gold chunk hit@5 | 98.46% |
| Retrieval gold chunk recall@5 | 89.87% |
| Retrieval all-gold hit@5 | 78.46% |
| Reference gold hit | 90.77% |
| Reference gold recall | 81.03% |
| Reference doc hit | 90.77% |
| Avg retrieved chunks | 5.7538 |
| Avg references | 1.7385 |
| Avg retrieval calls | 1.4000 |
| Avg tool calls | 1.4769 |
| Avg LLM calls | 2.4308 |

调用数总计：retrieval=91、tool=96、LLM=158。调用均值从 **successful predictions** 计算，本轮该分母恰好等于全65题；并未删除失败题抬高 coverage。现有 summary 不提供三个调用均值，以上为 CSV 算术汇总，没有修改 evaluator 公式。

Judge-off final score=43.65%，behavior score=83.62%。该阶段 key_point_recall=0 和 forbidden_rate=0 为无 Judge 的保守兜底，不能解释为真实语义召回为零或已证明没有编造。`not_in_doc` 的空 gold 在检索指标中默认命中；其100%不代表检索到了支持答案的 gold。

## CATEGORY BREAKDOWN

所有类别 coverage=100%，runtime failure count=0。检索指标分母为本类别全部题目；reference 和调用均值使用可用 prediction（本轮无缺失）。

| 类别 | count | coverage | runtime failures | doc hit@5 | gold hit@5 | gold recall@5 | reference gold recall | avg retrieval/tool/LLM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| single_doc | 40 | 100% | 0 | 97.50% | 97.50% | 90.00% | 86.25% | 1.1000 / 1.1250 / 2.0750 |
| cross_doc | 10 | 100% | 0 | 100.00% | 100.00% | 80.83% | 78.33% | 1.9000 / 2.3000 / 3.1000 |
| partial_answer | 6 | 100% | 0 | 100.00% | 100.00% | 94.44% | 33.33% | 1.6667 / 1.6667 / 2.8333 |
| not_in_doc | 5 | 100% | 0 | 100.00% | 100.00% | 100.00% | 100.00% | 2.8000 / 2.8000 / 3.8000 |
| trap | 4 | 100% | 0 | 100.00% | 100.00% | 91.67% | 83.33% | 1.0000 / 1.0000 / 2.0000 |

## DYNAMIC DECOMPOSITION

| 类别 | 0 retrieval | 1 retrieval | 2+ retrieval |
| --- | ---: | ---: | ---: |
| single_doc | 0 | 36 | 4（10%） |
| cross_doc | 0 | 2 | 8（80%） |
| partial_answer | 0 | 3 | 3（50%） |

cross_doc 明显更常多次获取证据。`retrieval_call_count` 是成功返回 evidence 字段的调用数（含 compare/get），不是严格的串行 Model 轮数；并行工具调用也可能计为2。q044 一次 compare 就观察到全部 gold，所以少调用不等于未分解。partial_answer 的 q052/q055/q056 有补检索，但是否新增 gold 必须由逐调用观测判断。

正式日志只有工具名、顺序、状态和耗时，CSV 只有累计IDs，没有每次 query/top_k/返回IDs。因此全65题的“第二次新增 chunk/gold”“相似 query 重复”精确计数 **NOT OBSERVABLE**。不把定向复跑的调用轨迹回填成正式运行事实。

以下仅是**独立诊断样本**，不代表正式65题每次调用的轨迹。只深入查看3个cross_doc qid（q047/q049/q050，共4次复跑），没有检查65题完整trace；保存的投影仅为工具参数、状态、IDs、计数及公开答案，不记录隐藏推理。

成功补检索：q047第二次search改变问题焦点，新增gold407；q050第二次search新增gold423。可疑cross_doc：q049两次都通过第二次search新增gold360，但仍缺380；正式Judge关键点完整，不能把该ID缺口升级成已确认语义失败。

| 诊断 | status | 工具 / query / top_k / observed IDs | reference IDs |
| --- | --- | --- | --- |
| q047 #1 | OK | search_knowledge_base (success)；研究生未完成注册对培养和学位论文环节的影响；k=5；[416, 420, 264, 227, 253]<br>search_knowledge_base (success)；研究生学位论文评审要求与注册状态的关系；k=5；[407, 409, 408, 416, 411] | [416, 407, 409, 408, 411] |
| q049 #1 | OK | compare_documents (error)；本科转专业政策差异、2025级学生适用注意事项；k=3（schema默认）；[]<br>search_knowledge_base (success)；2025年本科转专业修订办法；k=3；[373, 375, 376]<br>search_knowledge_base (success)；教务问答 本科转专业；k=3；[39, 360, 256] | [375, 376, 360, 256] |
| q049 #2 | OK | compare_documents (error)；转专业条件、流程、时间节点、限制条款；k=3（schema默认）；[]<br>search_knowledge_base (success)；2025年本科转专业管理办法；k=3；[375, 376, 377]<br>search_knowledge_base (success)；教务处常见问题解答 转专业；k=3；[360, 340, 256] | [375, 376, 377, 360, 340] |
| q050 #1 | OK | search_knowledge_base (success)；本科教务问答 学历证书 遗失 补办；k=5；[355, 242, 44, 329, 271]<br>search_knowledge_base (success)；研究生学籍细则 学位证书 遗失 补办；k=5；[423, 271, 262, 242, 44]<br>compare_documents (error)；学历证书或学位证书遗失后如何处理；k=3（schema默认）；[]<br>search_knowledge_base (success)；本科教务问答 学历证书 遗失 补办 证明书；k=3；[355, 242, 44]<br>search_knowledge_base (success)；研究生学籍细则 学位证书 遗失 补办 证明书；k=3；[423, 271, 242] | [355, 423, 44] |

4次cross_doc复跑的第二次成功证据调用均新增gold（q047:+407，q049两次:+360，q050:+423）。新增gold不保证全部gold齐全，q049仍缺380。q050随后compare失败再重复search两次，新增chunk/gold均为0，是同时存在有效分解与冗余后续调用的实例。

partial_answer诊断：q052两次首检都已取得全部gold375/376，第二次仅新增非gold，结局却一拒答一回答；q056两次首检都已有gold399，补检索只新增非gold且仍拒答；q054一次补检索返回已见383，另一次不补检索，均拒答。这些轨迹不支持“所有失败都因Agent不继续检索”。

## Q045

- Official status: **OK**；retrieval/tool/LLM=1/2/3。
- Retrieved `[365,355,389,387,388]`；referenced `[365,389,387]`；gold `[365,387,388]`。
- Failure sidecar: NONE；grounding mismatch IDs: N/A。
- Judge key-point recall=100%，missing/partial/contradicted=0，forbidden_hit=false。
- Targeted reruns: **0**。本轮正常且Judge无语义遗漏，不为历史失败额外刷题。
- Attribution: **previous failure not reproduced → MODEL_VARIANCE remains plausible**。Reference少一个精确gold ID不等于本题语义失败。

## Q049

- Official: **OK**；retrieval/tool/LLM=2/3/4；无max_iterations_reached。
- Retrieved `[375,376,377,360,340,256]`；referenced `[375,376,360]`；gold `[360,375,376,380]`，380未观察到。
- Judge recall=100%，forbidden=false。保留gold-ID召回缺口，不将其直接等同答案关键点失败。
- 复跑结果与实际query/top_k见TARGETED RERUNS；未调高任何预算。

## Q052

- Official: **NO_RELEVANT_INFO**，answer=`我无法根据当前取得的证据确定答案。`。
- Retrieved `[375,377,376,378,379,374,418]`；referenced `[]`；gold `[375,376]`全部观察到。
- Retrieval/tool/LLM=2/2/3。
- Judge recall=25%；已知三点missing，只覆盖“未提供具体题目和评分细则”。
- Attribution: **REFUSAL_POLICY / whole-answer over-refusal**，并表现为observed gold未引用；不是缺乏可回答证据。复跑独立衡量该行为的波动。

## Q054

- Official: **NO_RELEVANT_INFO**，answer=`我无法根据当前取得的证据确定答案。`。
- Retrieved `[382,239,383,384,385]`；referenced `[]`；gold `[382,383,386]`，386未观察到。
- Retrieval/tool/LLM=1/1/3。Get_chunk_text calls=0；tool errors=0；chunk_id args=N/A；redundant reread=0。
- Locator regression: **NOT OBSERVED**，本轮没有 `Document.id → get_chunk_text`；7A blocker继续视为关闭，同时明确正式题未走原文读取路径。
- Judge recall=33.33%，通知书/告知书存在性与送达归档要求missing，只承认未知完整模板。
- Attribution: **REFUSAL_POLICY**为整段不回答已知内容的主要解释；另有386的局部retrieval缺口。诊断中386仍未补齐，已有382/383可确认的内容仍被整体拒答，见TARGETED RERUNS。

## EVIDENCE FUNNEL

以下均排除runtime failure；本轮runtime failure为0。集合以**全部累计观察IDs**计算，不只看前5项。各层可重叠，不能相加充当独立失败题数。

### RETRIEVAL MISS

- 至少一个gold始终未观察到：**10题**，q005、q011、q012、q013、q014、q016、q019、q049、q054、q062。
- 整题完全无gold：**1题，q012**。
- 不归因到synthesis；还须检查query/top_k和同义来源。q005/q013/q016/q019/q049/q062的Judge recall均为100%，说明精确gold-ID缺口不自动等于信息缺失。

### OBSERVED BUT NOT REFERENCED

- 至少一个已观察gold未引用：**13题**，q013、q015、q038、q041、q042、q044、q045、q046、q052、q053、q054、q056、q062。
- 已观察gold但最后完全无gold引用：**5题**，q013、q052、q053、q054、q056。
- q013引用学生手册中的同义注册规则，Judge recall=100%；其指标缺口不能直接证明错误。q052/q054/q056是拒答丢弃已知内容；q053则是可定位到422→271的来源选择冲突。
- q015已观察gold316但未引用，Judge指出“缓考课程不设补考”被矛盾表述；q042未引用gold9，Judge指出缺少“以新通知为准”。保留selection与合成的边界，不越过前层直接判全部synthesis。

### REFERENCED BUT JUDGE KEYPOINT MISS

- 宽条件“至少有gold观察且引用、Judge recall<1”：**5题**，q011、q014、q015、q042、q055。
- q011/q014仍有gold未检索到；q015/q042仍有gold未引用。因此这4题无法排除更早漏斗层。
- 排除上述混杂后，全部gold都已观察且引用的synthesis candidate：**1题，q055**。Judge将委托代办额外材料评为partial，recall=87.5%；没有足够病例据此认定ANSWER_SYNTHESIS主导。

逐题集合、官方调用链和Judge关联见 [failure-attribution-matrix.json](../../outputs/evaluation_baseline_v2/agentic-rag-7d-rag-b6f018a/analysis/failure-attribution-matrix.json)；进一步语义边界见 [manual-evidence-checks.json](../../outputs/evaluation_baseline_v2/agentic-rag-7d-rag-b6f018a/analysis/manual-evidence-checks.json) 与 [synthesis-candidate-boundaries.json](../../outputs/evaluation_baseline_v2/agentic-rag-7d-rag-b6f018a/analysis/synthesis-candidate-boundaries.json)。

## AGENT CORE

Gate 2 完整执行25个 Core case（26个 turn）；3个 Online case按要求跳过。原始结果保留，不做修正后重算。

| 指标 | 结果 |
| --- | --- |
| Core case count | 25 |
| Pass rate | 7/25 = 28% |
| First-tool accuracy | 10/23 = 43.48%（仅有首工具断言的case） |
| Status accuracy | 25/25 = 100% |
| Tool-sequence accuracy | 2/2 = 100%（仅有精确序列断言的case） |
| Tool-call success | 39/40 = 97.5% |
| Avg tool / LLM calls | 1.6000 / 2.5600 |
| P50 / P95 latency | 8.20719s / 17.51327s |

唯一工具错误是 recovery case 主动读取不存在的 `chunk_id=999999`；随后正确读取2。没有未预期工具执行错误、wrong status 或 sequence mismatch。Clarify、direct answer 与 query rewrite 断言通过。Query rewrite 的 native structured-output不支持回退由既有 Gateway完成，本次实际改写成功。

18个失败case如下；failure reasons并不互斥：

| case | 原始failure reasons | 实际工具链 |
| --- | --- | --- |
| D001 | first_tool_mismatch | list_documents → search_knowledge_base |
| D002 | first_tool_mismatch | list_documents → search_knowledge_base |
| D003 | first_tool_mismatch | list_documents → search_knowledge_base → search_knowledge_base |
| D004 | first_tool_mismatch | list_documents → search_knowledge_base |
| D006 | first_tool_mismatch | list_documents → search_knowledge_base |
| D008 | first_tool_mismatch, source_missing | list_documents → search_knowledge_base |
| D009 | first_tool_mismatch | list_documents → search_knowledge_base |
| D010 | first_tool_mismatch | list_documents → search_knowledge_base |
| D012 | first_tool_mismatch | list_documents → search_knowledge_base |
| D013 | first_tool_mismatch | list_documents → search_knowledge_base |
| T001 | keyword_missing | list_documents |
| T003 | tool_output_schema_missing | get_chunk_text |
| compare_documents_explicit_001 | tool_output_schema_missing | compare_documents |
| N001 | keyword_missing | search_knowledge_base |
| N002 | first_tool_mismatch, keyword_missing | list_documents → search_knowledge_base |
| F002 | first_tool_mismatch, keyword_missing | list_documents → search_knowledge_base |
| agent-followup-001 | first_tool_mismatch | list_documents → search_knowledge_base |
| agent-tool-recovery-001 | tool_output_schema_missing | get_chunk_text → get_chunk_text |

归因：

- 13个 first_tool_mismatch 均先 `list_documents` 再检索；其中10个只违反首工具断言，剩余状态、关键词、source等断言通过。这是工具选择/额外调用信号，不足以证明答案架构失败。Core使用默认工具模式，正式RAG使用rag_only，不能把两个工具可见面直接混同。
- `T001` 正确返回当前16个文档，但冻结数据集仍要求关键词14。
- `T003`、`agent-tool-recovery-001` 仍要求旧输出键 `chunk_id/chunk`；当前合同是 `count/evidence`。`compare_documents_explicit_001` 仍要求 `doc_a_evidence/doc_b_evidence/comparison/conflict_detected`，当前工具只返回确定性证据和计数。这3个为明确的评测合同漂移，实际工具链成功；本阶段未改数据集或evaluator。
- `N001/N002/F002` 的实际状态均正确拒答，但不包含固定旧措辞“无法根据提供的文档确定答案”。N002/F002另有首工具不匹配。不能将 `no_evidence_accuracy=0` 解释为三题都生成了无依据答案。
- `D008` 另有source_missing：引用学生手册而非指定2018学籍管理办法，需保留为来源选择发现，不能与runtime错误混合。
- `agent-followup-001` 的query rewrite、关键词与source断言通过，仅第二轮先list而失败；`agent-multi-tool-001`通过；recovery的error→success序列正确，仅旧schema断言失败。

原始自动classification中的MODEL_VARIANCE不是复跑证明，本报告不直接沿用该标签作为稳定性结论。详见 [agent-failure-attribution.json](../../outputs/evaluation_baseline_v2/agentic-rag-7d-rag-b6f018a/analysis/agent-failure-attribution.json) 与 [agent-safe-observations.json](../../outputs/evaluation_baseline_v2/agentic-rag-7d-rag-b6f018a/analysis/agent-safe-observations.json)。

## SELECTIVE JUDGE

**COMPLETED**；34题=全部25道非single_doc + 9道符合固定可疑条件的single_doc。没有为凑满35而选普通题。Single_doc选择为 q005、q011、q012、q013、q014、q015、q016、q019、q038；规则和各题理由见 [judge-selection.json](../../outputs/evaluation_baseline_v2/agentic-rag-7d-rag-b6f018a/analysis/judge-selection.json)。

`judge-predictions.csv`严格按qid过滤正式CSV；每个字段保持一致，没有重新运行Agent，也没有补prediction。直接调用现有 `scripts.evaluation.rag_evaluator.run_rag_evaluation(..., k=5, use_llm_judge=True)`。

| 指标 | 结果 |
| --- | --- |
| Eval case count / actual judged cases | 34 / 34 |
| Judge coverage | 100%（仅该34题子集） |
| Actual Judge gateway calls | 49 = 34 key-point + 15 forbidden；日志未见重试或错误 |
| Runtime-failure skips | 0；没有missing prediction，未实际触发7C跳过分支 |
| Macro key-point recall | 85.42% |
| Covered / missing / partial / contradicted points | 118 / 11 / 6 / 5（共140点） |
| Forbidden hit | 1/34 = 2.94%；q061，3个forbidden points |
| Subset final / behavior score | 76.90% / 73.24%（保留原始公式） |

该子集有意富集可疑single_doc，85.42%不是全65题语义成绩。未改变正式Judge-off summary，也不将两组分数作性能提升比较。

| 类别 | Judge题数 | key-point recall | 原始behavior_label分布 |
| --- | ---: | ---: | --- |
| single_doc | 9 | 87.50% | answered=9 |
| cross_doc | 10 | 98.33% | answered=8, answered_with_supported_limitation=2 |
| partial_answer | 6 | 55.56% | answered_known_part_but_no_refusal=2, poor_partial=3, correct_partial=1 |
| not_in_doc | 5 | 80.00% | should_abstain_but_answered=5 |
| trap | 4 | 100.00% | answered=4 |

逐题关键点异常（完整34题明细见 [judge-case-findings.json](../../outputs/evaluation_baseline_v2/agentic-rag-7d-rag-b6f018a/analysis/judge-case-findings.json)）：

| qid | recall | missing | partial | contradicted |
| --- | ---: | --- | --- | --- |
| q011 | 75.00% | NONE | 送达之日起10日内；逾期不再受理 | NONE |
| q012 | 62.50% | 不得少于规定期限一半 | 违纪处理委员会决定 | NONE |
| q014 | 70.00% | 原则不超过两个月 | 八至十五日学生工作副院长 | NONE |
| q015 | 80.00% | NONE | NONE | 缓考课程不设补考 |
| q042 | 83.33% | 以新通知为准 | NONE | NONE |
| q052 | 25.00% | 学院制定实施细则；可面试/实操/笔试；综合成绩30%+70% | NONE | NONE |
| q053 | 62.50% | NONE | 导师同意学院批准研究生院审核 | 不得超过最长学习年限 |
| q054 | 33.33% | 有通知书和告知书；有送达和归档要求 | NONE | NONE |
| q055 | 87.50% | NONE | 可委托代办需材料 | NONE |
| q056 | 25.00% | 学位论文相似度检测系统；研究生院分配学院子账号；固定IP机器运行 | NONE | NONE |
| q061 | 0.00% | NONE | NONE | 未提供研究生奖助学金等级；未提供金额；未提供评定比例 |

## REFUSAL ANALYSIS

**behavior_label是evaluator的确定性文本规则结果，不是LLM直接给出的行为分类。** 只读调用原有 `is_refusal` / `is_standalone_refusal` 确认，两者都不识别当前统一句子“我无法根据当前取得的证据确定答案。”。因此以下原始标签存在可复现解释偏差；本阶段只记录，不改规则或分数。见 [refusal-classifier-observation.json](../../outputs/evaluation_baseline_v2/agentic-rag-7d-rag-b6f018a/analysis/refusal-classifier-observation.json)。

| 类别 | 实际可支持的结论 | 原始标签与边界 |
| --- | --- | --- |
| partial_answer | 整体过度拒答3/6：q052、q054、q056；均已观察到可回答的gold内容，Judge确认分别遗漏3、2、3个已知关键点 | 原始label均为poor_partial，而非refused_too_much，受拒答短语识别影响 |
| partial_answer | q055原始correct_partial，recall=87.5%；q051覆盖全部关键点；q053选中了另一套来源规定，recall=62.5%，有1个contradicted | 原始correct_partial仅1/6；q051/q053均为answered_known_part_but_no_refusal，不能用该标签直接判定语义拒答边界 |
| partial_answer | forbidden_hit=0；q053的“最长学习年限”是Judge-confirmed冲突，不能因forbidden=0就说全部受支持 | source selection与过度拒答分别归因 |
| not_in_doc | 正确整段拒答4/5：q057–q060，均NO_RELEVANT_INFO且Judge recall=1；错误具体回答1/5：q061 | 原始标签将5题全部列为should_abstain_but_answered，前4题是确定性识别误差 |
| trap | q062–q065均OK，Judge recall=1，forbidden_hit=0；4/4给出受支持的纠错/限制或标准 | 原始label均answered；unsupported detail=0（本次Judge范围内） |

q061在“研究生奖助学金”问题下列出800/500/350/200元和3%/7%/15%/100%，同时又声明适用对象未明确。Judge认为限定语没有消除将未确认适用的具体金额作为答案主体的问题；3个禁用点均命中。该题归REFUSAL_POLICY/适用范围边界，不归无gold的retrieval miss。

q052的Judge结果：学院实施细则、面试/实操/笔试、30%+70%三点missing；未知具体题目/评分细则这一点covered，recall=25%。这不是合格的已知部分回答。q054和q056也只承认未知部分，分别遗漏书面通知/送达归档，以及系统名称/学院子账号/固定IP等可确认内容。

## EFFICIENCY

### REDUNDANT RETRIEVAL

正式全65题的逐次query及返回IDs未落盘，精确冗余检索数为 **NOT OBSERVABLE**。以下是固定9题/13次诊断样本，不能外推为正式全量比例：

- 后续成功证据调用共12次：新增gold 5次；只新增非gold 4次；无新增证据 3次。
- 完全相同query且无新增证据：0次；语义相近、query细化但无新增证据：3次，q050 #1后两次search、q054 #1第二次search。
- 新query但无新增gold：7次，q050两次、q052两次、q054一次、q056两次；其中q052/q056仍增加非gold，不因缺少新gold就自动认定无用。
- q050 #1首次两次search已收齐gold355/423，后续compare错误加两次search不扩充证据，直接构成3次额外工具调用。

### REDUNDANT GET_CHUNK_TEXT

正式RAG：affected qids=NONE，get_chunk_text calls=0，redundant reread=0，extra calls=0。13次诊断也均未调用get_chunk_text。q054两次复跑仍只有search，未出现Document.id作为chunk_id的regression；这一结果不冒充本次已覆盖原文读取路径。Core中显式get_chunk_text和纠错路径另有真实执行且行为正确。

### COMPARE_DOCUMENTS

正式：attempts=7（q035/q042/q043/q044/q045/q049/q050），success=2（q044/q050），errors=5，fallback-to-search=5。

- 错误后最终观察到全部gold：4题，q035/q042/q043/q045。
- 错误后仍缺部分gold：1题，q049缺380；但其正式Judge recall=100%，不能宣称compare错误已导致答案质量失败。
- 是否由fallback首次新增gold在正式逐调用层面不完全可见；上述统计只描述fallback后的最终累计集合。
- 诊断compare共3次、全部错误（q049两次、q050一次）；均继续search。q049两次恢复部分gold但缺380；q050在compare前已齐gold，后续search无新增。参数中“教务问答”“教务处常见问题解答”等简称/别名与文件标签匹配存在具体失败实例，但没有足够数据证明这是主导质量来源，未修改alias或工具。

审计日志反复出现`tool audit write failed ... RuntimeError`，与此同时工具实际成功。没有把这类日志告警等同TOOL_ERROR或归作本轮runtime failure；本阶段保留为诊断噪声finding。

## TARGETED RERUNS

三道正式Gate完成后，按 [targeted-plan.json](../../outputs/evaluation_baseline_v2/agentic-rag-7d-rag-b6f018a/analysis/targeted-plan.json) 固定9个qid、13次调用，每题最多2次。未针对稳定失败反复刷分，未送诊断答案重新做Judge。

| qid | 正式status | 额外次数与结果 | 诊断结论 |
| --- | --- | --- | --- |
| q012 | OK | 1：OK | 第二次新query新增gold289并引用；半期限信息出现在诊断答案。正式缺口保留，query/模型规划波动仍可能。 |
| q047 | OK | 1：OK | 第二次检索新增gold407，证实有效补检索实例。 |
| q049 | OK | 2：OK / OK | 两次均retrieval/tool/LLM=2/3/4；compare失败后检索，380仍缺；未复现iteration exhaustion。 |
| q050 | OK | 1：OK | 前两次search已齐gold；随后compare错误+两次search无新增证据。 |
| q052 | NO_RELEVANT_INFO | 2：NO_RELEVANT_INFO / OK | 两次首检同query、同gold；一次NO_RELEVANT_INFO，一次OK并引用375/376，存在模型行为波动。 |
| q053 | OK | 1：OK | 仍观察422却引用271；另一套来源的1年延期规定再次成为答案主体，selection问题重复。 |
| q054 | NO_RELEVANT_INFO | 2：NO_RELEVANT_INFO / NO_RELEVANT_INFO | 两次NO_RELEVANT_INFO，无get_chunk_text；都已有382/383可回答内容，386仍缺。 |
| q056 | NO_RELEVANT_INFO | 2：NO_RELEVANT_INFO / NO_RELEVANT_INFO | 两次NO_RELEVANT_INFO，均首检已有399；继续检索也未回答已知部分。 |
| q061 | OK | 1：OK | 仍OK，引用181并列出未确认适用于研究生的金额/比例；仅人工核对诊断答案，未重跑Judge。 |

全部诊断runtime failures=0。q045没有额外复跑。每次实际query、top_k（省略时标明schema默认）、observed/reference IDs、新增gold和status均见 [targeted-analysis.json](../../outputs/evaluation_baseline_v2/agentic-rag-7d-rag-b6f018a/analysis/targeted-analysis.json)。

## FAILURE ATTRIBUTION

这些是测量支持的层级归因，不是互斥分桶；尤其过度拒答也会产生空引用，不能重复算成两个独立系统问题。全部65题的原始集合与Judge结果保存在归因矩阵中。

| 层级 | case IDs / 支持边界 |
| --- | --- |
| RUNTIME_STABILITY | NONE：正式65题和额外13次诊断均无runtime failure；无grounding mismatch可分类。 |
| AGENT_RETRIEVAL_PLANNING | q012正式缺gold、诊断换query后补齐，原因仍涉及query选择与模型波动；没有足够正式逐调用记录把缺口全部归给规划。Core的13个list-first偏离是独立工具选择/效率信号。cross_doc已有有效补检索反例，不支持“不继续检索”为主导问题。 |
| RETRIEVAL | 正式精确ID缺口10题见漏斗；其中q011/q012/q014同时有Judge关键点缺口，仍需区分query与retriever。q049三次都缺380，但正式Judge=100%；q012诊断补齐289。因此未证明“合理且充分检索后反复漏gold”的广泛稳定retriever机制。 |
| EVIDENCE_SELECTION | q053最明确：422已观察，却两次引用271并使用另一套延期规定；q015/q042有观察未引用且Judge异常。其余精确ID遗漏见13题列表，不能把全部语义通过题都算成实际答案错误。 |
| ANSWER_SYNTHESIS | q055为排除前层缺口后的1个candidate：全部gold观察并引用，但委托代办材料只partial。q011/q014/q015/q042存在前层混杂，不强行归到synthesis。 |
| REFUSAL_POLICY | **q052、q054、q056**：已有可回答证据却整段拒答；**q061**：适用对象未确认仍给出具体研究生奖助标准。前3题形成同类重复模式，q061是另一端的适用范围/拒答边界问题。 |
| MODEL_VARIANCE | **q052**：正式拒答，复跑拒答/OK，且两次复跑首检query与全部gold一致。q012检索覆盖发生变化，q050工具路径变化；q045历史failure本轮未复现，仅保留plausible。q049历史iteration failure本轮及两次复跑均未复现，不能写成当前稳定runtime缺陷。 |
| EFFICIENCY_ONLY | q050诊断在gold齐全后仍compare失败并重复search，3次额外工具调用；Core中仅first-tool断言失败的10题见Core章节。q035/q042/q043/q045的compare错误后恢复是效率信号，未证明这些错误导致主要语义失败。q054虽有冗余补检索，整题主要质量归因仍为REFUSAL_POLICY。 |

另有 **EVALUATION_CONTRACT / CLASSIFIER_FINDINGS**：Core固定旧字段、数量和措辞断言；RAG确定性拒答识别漏掉当前统一拒答句。它们不属于生产层级修复证据。本阶段保留原始分数和标签，不将这类假失败混入dominant生产质量结论。

## DOMINANT FAILURE MODE

**REFUSAL_POLICY**。

- Supporting case count：核心同类模式 **3题（q052/q054/q056，占partial_answer 3/6）**；另有q061的无依据适用范围回答，共4个不同qid支持拒答边界问题。
- Supporting metrics：partial_answer retrieval gold hit@5=100%、recall@5=94.44%，但reference gold recall=33.33%、Judge key-point recall=55.56%。q052/q056已观察全部gold；q054虽然缺386，已有382/383能确认的内容也没有回答。它们共遗漏8个已知关键点。cross_doc Judge recall=98.33%，与“多子问题检索全面失效”不一致。
- Supporting traces：q052两次首检同query且取得375/376，之后无新增gold，结局一拒答一回答；q054两次已有382/383仍拒答；q056两次首检已有399、再次检索后仍拒答。q054/q056的正式加诊断共6次均拒答。q061复跑继续引用181并列出未确认适用于研究生的金额和比例。
- Attribution limit：q052存在模型波动，不声称每次都必然失败；q054存在局部retrieval缺口；q061复跑未做额外Judge。主导结论来自重复的已知部分丢弃和适用范围边界证据，**不依赖有误的behavior_label识别结果**，也不靠Core原始28%通过率推断。

Runtime stability、Retrieval、Evidence Selection和Synthesis都有独立观测边界；当前证据不足以将它们提升为主导系统性问题。

## CODE CHANGES

```text
Production code changes: NONE
Evaluation code changes: NONE
Prompt changes: NONE
Config changes: NONE
Dataset changes: NONE
```

仅在`outputs/`保留一次性分析/诊断脚本及产物，并在所有真实模型调用结束后新增本报告。没有修改`app/`、`scripts/`、`tests/`、`data/eval/`，没有调整预算、schema、embedding/rerank/BM25、chunk参数或compare alias。

创建报告前，全部已有tracked文件SHA-256与开始时一致，worktree仍clean；HEAD未变。正式12个文件的指纹保持一致，Judge CSV逐字段等于正式CSV的子集，65题日志工具数与CSV逐题相等。诊断前后索引逻辑内容指纹一致（16 documents / 493 chunks）。详见 [source-integrity-before-report.json](../../outputs/evaluation_baseline_v2/agentic-rag-7d-rag-b6f018a/analysis/source-integrity-before-report.json)。不把嵌入式Chroma打开过程说成物理文件字节完全不变。

未commit、push、切换或创建分支；未启动Phase 8。

## TESTS / RUNS

| Gate / 检查 | 实际结果 |
| --- | --- |
| validate-only | PASS |
| Official RAG | COMPLETE；65 predictions + 0 failures；65 report rows；12个必需产物齐全 |
| Agent Core | COMPLETE；25 cases / 26 turns；原始7 passed、18 failed；exit 0 |
| Selective Judge | COMPLETE；34 cases、49 gateway calls；无Judge error；exit 0 |
| Targeted reruns | COMPLETE；9 qids、13次；无runtime failure；exit 0 |
| Source/artifact integrity | PASS；既有tracked内容不变、正式产物未覆盖、Judge只过滤原CSV |
| git diff --check | PASS；另检查新增报告的空白与本地链接 |
| full pytest | NOT RUN |
| 7C focused tests | NOT RERUN |
| Online / MCP / Mixed / Combined | NOT RUN |
| Frontend tests / 268 tests | NOT RUN |

实际正式命令：

```bash
.venv/bin/python scripts/run_evaluation_baseline.py --validate-only

.venv/bin/python -u scripts/run_evaluation_baseline.py \
  --rag-only \
  --rag-eval-file data/eval/rag_eval_questions_v2_semantic_fixed.json \
  --output-dir outputs/evaluation_baseline_v2/agentic-rag-7d-rag-b6f018a \
  --retrieval-k 5

.venv/bin/python -u scripts/run_evaluation_baseline.py \
  --agent-only \
  --agent-cases data/eval/agent_eval_cases_v2.json \
  --output-dir outputs/evaluation_baseline_v2/agentic-rag-7d-agent-b6f018a
```

三个命令均使用`set -o pipefail`及`2>&1 | tee`保存诊断日志。未使用`--allow-dirty`、`--force-rebuild`、RAG重新生成式`--use-llm-judge`，或Online/MCP/Mixed开关。

补充敏感信息扫描覆盖本轮RAG、Core、diagnostics及最终报告；只输出凭据值数量和命中文件路径，不输出密钥。正式Runner的core-only secret_scan未装载MCP凭据，不把它的零命中当成全部模型凭据均已扫描。补充结果与报告验证见 [secret-scan-final.json](../../outputs/evaluation_baseline_v2/agentic-rag-7d-rag-b6f018a/analysis/secret-scan-final.json)、[report-verification.json](../../outputs/evaluation_baseline_v2/agentic-rag-7d-rag-b6f018a/analysis/report-verification.json)。

## FINDINGS

1. 正式v3基线已成立，runtime failure rate本次为0%；7C缺失prediction错误分支没有在本次真实调用中触发，不冒充该分支的在线验证。
2. Agent确实会多次获取跨文档证据，存在改变query并新增gold的诊断实例。未发现恢复Decompose LLM或另建RAG workflow的证据需求。
3. 已知部分因整体拒答被丢弃，是当前最一致的生产质量信号；q052同时有模型方差。
4. 精确gold-ID召回、source选择和答案语义不同：q013可用另一来源答对；q053则选了冲突来源。不能用引用ID遗漏直接替代答案错误判定。
5. q055只提供一个较弱的grounded synthesis candidate；不足以认定synthesis主导。
6. Core原始通过率受list-first策略、旧schema/数量/措辞断言影响；RAG拒答短语识别也存在偏差。均原样记录，没有据此修改生产系统或美化分数。
7. compare失败回退与重复检索有测量证据，但大多恢复gold或不影响已确认语义结果，当前不应成为优先生产改动。

## RECOMMENDED NEXT CHANGE

仅针对 **REFUSAL_POLICY** 提出下一阶段候选：在现有唯一Agent的回答指导中，明确“先回答已被证据确认的部分，再限定未被证据确认的具体细节和适用对象”。候选应保留现有grounding fail-closed，不因某个细节缺失就丢弃全部已知内容，也不把适用对象未确认的具体金额作为答案主体。

用q052/q054/q056及q061和本轮正式v3产物评估该最小候选；明确保留当前evaluator措辞/合同finding的解释边界。本阶段没有实施候选，没有调整检索或预算，没有设置CI threshold。

## NEXT PHASE

**AGENTIC-RAG-8 should address only measured partial-answer/refusal behavior.**

本报告完成AGENTIC-RAG-7D；Phase 8尚未开始。
