# AGENTIC-RAG-7C RESULT

## STATUS

**PASS_WITH_FINDINGS**。单题 Agent 错误状态已从批处理异常改为可测量的失败结果；
生产 grounding 继续 fail closed。169 个 focused tests 通过，工程检查通过。
本阶段只验证离线确定性合同，未建立或重跑真实模型 baseline。

## BASELINE

- branch: `refactor/tool-output-models`
- head_before: `09dae2d8e4c60908afc674fffb7e73ec4a663f3b`
- head_after: `09dae2d8e4c60908afc674fffb7e73ec4a663f3b`
- worktree_before: clean
- worktree_after: 11 个原有文件修改，新增本报告；未 commit、push、切换或创建分支。

## PROBLEM

[Phase 7](agentic-rag-7-result.md) 在 q054 遇到工具 locator blocker，
此前 53 题只在内存中，没有生成 predictions.csv。
[Phase 7A](agentic-rag-7a-result.md) 已修复 locator contract；
[Phase 7B](agentic-rag-7b-result.md) 又在 q045 遇到 unavailable evidence 的
MODEL_ERROR，前 44 题结果未落盘，后续 evaluator、Agent Core 和 Judge 未执行。
单题 runtime/model failure 被提升为整批异常，使正式测量产物无法形成。

q045 两次独立复跑均为 OK；这些历史证据没有证明确定性的生产 grounding bug。
本阶段未复跑这些真实模型案例。

## PRODUCTION GROUNDING

- unknown evidence IDs: still fail closed，仍为 MODEL_ERROR，references=[]。
- unsupported answer: still not exposed，不回退到普通 AIMessage，不过滤错误 ID 后放行。
- retry: not added。ToolStrategy、FinalAgentResponse 和 grounding 决策不变。
- 动态 ID 存在性校验继续位于 Result Builder，reason 保持
  `Agent structured response referenced unavailable evidence`。

## GROUNDING DIAGNOSTICS

仅 dynamic grounding mismatch 的 `compose_final_answer` trace 新增：

- `requested_evidence_ids`：结构化最终响应请求的 ID，保留顺序。
- `available_evidence_ids`：本轮已收集的 Document.id，保留收集顺序。
- `unavailable_evidence_ids`：请求列表中不在 available 集合的 ID。

OK、NO_RELEVANT_INFO、NEED_CLARIFICATION 的 trace 不新增这些字段。
具体 ID 不拼接进 reason、answer 或 tool_result_summary。测试证明错误结果不包含
`PRIVATE SECRET BODY`、完整私有 metadata 或被拒绝的最终答案。

## EVALUATION BEFORE

`MODEL_ERROR → exception → entire batch abort → no predictions.csv`。
Runner 还要求至少一条成功 prediction，因此全失败批次无法评估。

## EVALUATION AFTER

`MODEL_ERROR → case failure → prediction_failures.jsonl → next case continues`。
MODEL_BAD_OUTPUT、TOOL_ERROR、INTERNAL_ERROR 使用相同隔离规则。

`_run_rag_once` 复用唯一 ChatService 调用并投影安全 trace；
`run_rag_once_with_result` 仍对 Agent 错误状态抛出 RuntimeError，保留单题诊断语义。
批量生成调用同一底层实现，结构校验通过后才将错误状态记录为 case failure。

## FAILURE ARTIFACT CONTRACT

文件：`rag/prediction_failures.jsonl`，每条失败题目一行。

| 字段 | 内容 |
| --- | --- |
| qid / question | 原评测题标识和题目 |
| status / reason | 公开 Agent 状态和已有稳定 reason |
| retrieved_chunk_ids / reference_chunk_ids | 实际捕获的 KB chunk IDs 和实际公开引用 IDs |
| retrieval_call_count / tool_call_count / llm_call_count | 已观察到的调用计数 |
| requested_evidence_ids / available_evidence_ids / unavailable_evidence_ids | grounding mismatch 的 ID 诊断 |

不可获取的数据使用 []、0 或空字符串。不保存失败答案、证据正文、完整 metadata、
ToolMessage、Prompt、隐藏推理、凭据或完整 trace。
成功批次也创建空 sidecar，Runner 直接从 sidecar 汇总
`prediction_failure_count` 和 `prediction_failure_status_counts`，写入 baseline summary 的 rag 部分。

## PREDICTION CONTRACT

- Schema: **3**，成功 row 字段不变。
- successful rows: only completed predictions；包含合法的 OK、NO_RELEVANT_INFO、NEED_CLARIFICATION。
- failed cases: not written as fake answers，不在 CSV 中写空答案或 error status 伪预测。
- 完成时强制 `successful_prediction_count + case_runtime_failure_count == evaluation_question_count`。
- 数据集入口拒绝空/损坏题目和重复 qid，防止按 qid 加载 CSV 时静默覆盖题目。

## MISSING PREDICTION SCORING

- has_prediction=false，behavior_label=runtime_error。
- final_score=0，behavior_score=0，key_point_recall=0，正式 retrieval metrics=0。
- 即使 Gold 为空也不给检索默认命中分，不给 clean/forbidden 奖励。
- judge not called；两个 judge_used 标志为 false，使用现有 unavailable results，原因为
  `prediction unavailable due runtime failure`。
- failure classification=runtime_error，不叠加 document/chunk/reference/answer miss。
- retrieved_chunk_ids=[]；sidecar 内失败前观察到的 IDs 不输入正式评分。
- 整体及 category coverage 分母包含失败题目。

## STRUCTURAL FAIL-FAST

只隔离 Agent 正常返回的错误状态。缺失/不可评估/不在索引内的 chunk_id、重复捕获 call IDs、
工具输出与 trace 无法对齐、非法状态、损坏/重复题目、RAG 初始化失败、索引异常、序列化和
文件写入异常仍抛出并中断整批。KeyboardInterrupt 和进程崩溃不会被吞掉。

回归测试在 q1 成功、q2 返回 MODEL_ERROR 且存在结构损坏时证明：q2 抛出、q3 不执行、
已写 q1 保留、q2 不被错误记入 runtime sidecar。

## INCREMENTAL PERSISTENCE

CSV 打开后立即写 header 并 flush；每条 successful row 和每条 failure JSONL 都立即 flush。
测试在下一题进入 ChatService 前重新打开两份文件，确认上一题结果已可读取。

部分 CSV/JSONL 仅用于诊断。Accepted baseline 要求完整 dataset loop、每题有明确去向、
正式 evaluator、manifest 和 reports 全部完成，并满足既有 clean-worktree 要求。
不要求 100% prediction coverage；本阶段 dirty worktree 下只运行离线测试。

## DETERMINISTIC MINI-BATCH

FakeToolCallingModel 经过真实 ChatService/create_agent/governed tools：

| case | result | artifact |
| --- | --- | --- |
| q1 | OK | prediction row |
| q2 | MODEL_ERROR，非法 evidence ID | failure row，保留 observed [1,2]，references=[] |
| q3 | OK | prediction row |

结果：all 3 attempted，2 prediction rows，1 failure row；unsafe answer 不出现在两份文件中。
另有 Runner 离线集成测试覆盖 2/3 与 0/3 coverage，运行真实 producer/evaluator/report 代码，
仅替换 ChatService 结果、索引依赖和 manifest 环境信息；全失败时 CSV 只有 header，
仍生成逐题报告、category summary、manifest、baseline report 和 failure_cases。
这不是真实模型或正式 baseline。

## TESTS

最终执行命令：

```bash
.venv/bin/pytest -q \
  tests/test_agent_results.py \
  tests/test_evaluation_rag_predictions_v3.py \
  tests/test_evaluation_rag_behavior.py \
  tests/test_evaluation_rag_evidence_funnel.py \
  tests/test_evaluation_runner.py \
  tests/test_evaluation_smoke.py::test_minimal_rag_smoke_judge_disabled
```

**169 passed，1 warning**（jieba/pkg_resources DeprecationWarning）。
测试范围由实际 consumer scan 确定，额外只运行直接调用 producer/evaluator 的单个 RAG smoke。
Judge enabled 分支只使用 Mock，不访问真实 Judge。

首轮 168 passed / 1 failed 为新增 mini-batch fixture 把每轮获取模型误当作新题目；
改在单题 ChatService 边界检查 flush 后，定向测试和上述完整 focused 集合均通过。

## TESTS NOT RUN

- full pytest
- 65-question real-model baseline
- Agent Core
- Judge（真实模型）
- Online
- MCP
- Mixed
- Combined

未使用 --allow-dirty 运行 baseline，未调用真实模型制造错误，未开始 AGENTIC-RAG-7D。

## ENGINEERING CHECKS

- lint-imports: PASS，7 kept / 0 broken。
- runtime inspection: PASS，`.venv/bin/python scripts/inspect_agent_runtime.py --check`。
- Ruff: PASS，仅检查本阶段修改的 10 个 Python 文件。
- git diff --check: PASS。

## FILES CHANGED

| 文件 | 修改 |
| --- | --- |
| app/agent/results.py | mismatch-only ID 诊断 |
| scripts/evaluation/rag_predictions.py | 共用执行实现、批次隔离、sidecar、逐题 flush、完整性校验 |
| scripts/evaluation/rag/evaluation.py | missing prediction 归零与 Judge 跳过 |
| scripts/evaluation/report.py | runtime failure 单一分类 |
| scripts/evaluation/runner.py | 接受零成功预测，从 sidecar 汇总失败 |
| tests/test_agent_results.py | 混合真/假 ID、正文隐私、成功 trace 无冗余字段 |
| tests/test_evaluation_rag_predictions_v3.py | 四类错误、fail-fast API、mini-batch、flush、结构失败、损坏数据 |
| tests/test_evaluation_rag_behavior.py | 三类期望行为、空/非空 Gold、Judge enabled/disabled 的 missing 归零 |
| tests/test_evaluation_rag_evidence_funnel.py | runtime_error 不产生其他归因 |
| tests/test_evaluation_runner.py | 部分/全部失败时完整生成离线报告 |
| scripts/evaluation/README.md | 当前运行合同、产物、失败边界、接受条件 |
| docs/architecture/agentic-rag-7c-result.md | 本报告 |

## UNCHANGED

- Agent Prompt
- Retrieval（BM25、Embedding、Chroma、CrossEncoder、top_k、chunking、index）
- FinalAgentResponse
- Grounding decision
- budgets
- compare_documents
- Document identity
- Tool contracts / ToolStrategy
- evaluation dataset
- 历史 Phase 7 / 7A / 7B 报告与 docs/history

## FINDINGS

- q045 underlying model behavior remains uncertain；一次正式失败与两次成功复跑不足以确定根因。
- this phase does not claim to fix q045 model variance。
- it makes future occurrences measurable instead of batch-fatal；安全 ID 诊断补足后续归因信息。
- q049/q052/q054 quality findings remain pending full baseline。重复检索、get_chunk_text 效率、
  compare alias/fallback 等历史现象未在本阶段修复或重新评估。
- 全部验证为确定性离线证据，不代表模型稳定性、线上质量或正式 baseline 已通过。

## NEXT PHASE

AGENTIC-RAG-7D should run the clean-worktree official 65-question Prediction v3 baseline on the committed failure-isolation contract, accept explicit per-case runtime failures as measured zero-score outcomes, then complete Agent Core, selective Judge, and failure attribution before any production quality tuning.
