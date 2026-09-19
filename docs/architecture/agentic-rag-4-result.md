# AGENTIC-RAG-4 RESULT

## STATUS

PASS_WITH_FINDINGS

`compare_documents` 已从内部调用 LLM 的比较工具改为双文档证据检索工具。
最终 focused run 为 **251 passed, 1 failed**；唯一失败是 Docling 的既有导入副作用，
在未修改的当前 HEAD 快照中复现。Compare、grounding、合同和其余定向检查通过。

## BASELINE

```text
branch: refactor/tool-output-models
head_before: aa4949bc7c2e01622e30e97fe285a4dbc1c9686d
head_after: aa4949bc7c2e01622e30e97fe285a4dbc1c9686d
worktree_before: clean
worktree_after: 17 modified files + 1 new report, all uncommitted
commit / push / branch switch / new branch: NOT PERFORMED
```

先扫描真实生产消费者及测试，再修改代码。基线复现使用 `git archive HEAD` 导出
临时目录，使用同一 `.venv` 解释器，没有切换或回退当前工作区。

## ARCHITECTURE BEFORE

```text
Agent
→ compare_documents
→ custom keyword ranking
→ Compare LLM
→ comparison / conflict_detected / summary
→ Agent
```

## ARCHITECTURE AFTER

```text
唯一 create_agent Agent
→ compare_documents
→ KB readiness
→ deterministic document filter
→ safe candidate copy
→ existing CrossEncoder rerank（两份文档分别执行）
→ Top-K → chunk_to_evidence_document → LangChain Document evidence
→ ToolMessage
→ 同一 Agent 自行比较
→ ToolStrategy FinalAgentResponse
→ validated evidence IDs
→ grounded references
```

保留名称标准化、去扩展名、顶层/metadata 文件名提取、精确优先与包含匹配。
仅复制候选 dict 与 metadata，未引入第二个 Agent、Graph、scorer 或 evidence 管理框架。

## NESTED LLM REMOVAL

- compare tool internal LLM removed: **YES**。
- get_compare_model removed: **YES**；唯一生产消费者是旧 compare handler，工厂、导出、
  cache clear 项及专属工厂测试一起删除。
- CompareDocumentsLLMOutput removed: **YES**；唯一生产消费者是旧 compare handler，
  内部 schema、比较提示词、调用/解析测试一起删除。
- ModelGateway compare role removed: **YES**；删除 compare 的网关构造与结构化调用。

`ModelGateway` 本身保留：`app/agent/query_rewrite.py`、`app/rag/generation.py`、
`app/rag/decomposition.py`、`scripts/evaluation/rag/judge.py` 仍有真实消费者。
Compare handler 不再接收只用于内部模型的 trace_id；既有治理审计继续记录请求 trace_id。

## FRAMEWORK REUSE

- LangChain `Document` / `model_copy` / 原生 Pydantic JSON transport。
- existing `chunk_to_evidence_document`，统一 KB identity 与来源 metadata。
- existing `pipeline.rerank` / CrossEncoder，各文档分别重排后取 Top-K。
- existing `ToolSpec` / `ProjectStructuredTool`，权限、参数/输出校验与审计。
- existing `ToolMessage` 与唯一 `create_agent` 循环。
- existing `ToolStrategy` structured response。
- existing evidence-ID validation 与引用投影。
- existing `validate_evidence_documents` / `retrieval_evidence_metadata`。

新增第三方依赖：**NO**。

## COMPARE OUTPUT CONTRACT

| 字段 | 合同 |
| --- | --- |
| `query` | 本轮工具查询 |
| `doc_a` | 传入的第一份文档名 |
| `doc_b` | 传入的第二份文档名 |
| `doc_a_count` | 第一份文档返回数，非负 |
| `doc_b_count` | 第二份文档返回数，非负 |
| `count` | 等于 evidence 长度及两侧数量之和 |
| `evidence` | `list[Document]`；doc_a Top-K 后接 doc_b Top-K |

`comparison` / `conflict_detected` / `summary` 以及两套旧 evidence dict 字段已删除，
输出 schema 拒绝这些旧字段。每个 Document 使用既有 evidence 校验。

返回视图追加 `metadata.comparison_side = doc_a / doc_b` 和
`metadata.comparison_document = 对应传入文档名`，保留原文件、来源、chunk、位置、页码和重排分数。
这些展示 metadata 不参与 ID 构造。

归一化后同名文档仍返回工具错误；doc_a 或 doc_b 不存在仍返回工具错误。
KB readiness 为 false 时抛出标准 `ToolExecutionError(DEPENDENCY_UNAVAILABLE)`，
由治理边界转成 `error_code=dependency_unavailable` 的错误 ToolMessage。
存在文档时不因关键词零重合拒绝候选，交由 CrossEncoder 排序及 Agent 判断证据是否足够。

## EVIDENCE IDENTITY

`test_search_get_chunk_and_compare_share_identity` 通过真实 StructuredTool 调用 search、
get_chunk 和 compare，仅替换底层召回与 CrossEncoder 分数。三条路径返回的同一 chunk
具有相同 Document.id、相同正文，整数 chunk_id=0 保留。

测试覆盖有 source_id 和缺 source_id、使用 source URI + position 的两种路径。
get_chunk 使用真实 `pipeline.get_chunk_full_text`；search/get_chunk 不带比较分组 metadata。
Compare 直接调用公共转换函数，没有重写 ID、URI 或 metadata normalization。

## AGENT GROUNDING

真实 `FakeToolCallingModel → AgentRuntimeHarness → create_agent → StructuredTool` 测试：

```text
Model 1 → compare_documents
ToolMessage → A1 / A2 / B1 / B2 Documents
Model 2 → FinalAgentResponse(evidence_ids=[A1.id, B2.id])
Result → OK，references 只有 A1 / B2
```

实际 chunk IDs 为 `学生手册1`、`学生手册2`、`学籍规定1`、`学籍规定2`；最终只引用
`学生手册1` 与 `学籍规定2`。结论等于唯一 Agent 的 structured_response.answer，
工具不返回比较结论。真实 Agent Model 调用数为 2。

`compare_documents` 只需加入 `_EVIDENCE_TOOLS`，即可沿用：
`Document → collect_evidence_from_messages → evidence_ids → references`。
`app/agent/results.py` 无修改，没有 compare-specific grounding branch。

同一真实循环还验证：

- 已取得 compare evidence，但 answer 的 evidence_ids 为空：MODEL_ERROR，references=[]。
- 声明 `knowledge_base:fake`：MODEL_ERROR，references=[]，候选答案不对外泄漏。
- Agent 判断证据不足：NO_RELEVANT_INFO，references=[]。
- 缺文档/KB 未就绪：模型读到标准错误 ToolMessage，继续返回 no_evidence。

原非 evidence 兼容测试移除 compare，继续覆盖目录、web_search 和普通 MCP 输出。

## SHARED STATE SAFETY

`pipeline.chunks` **没有被 compare rerank 意外修改**。
两份文档的原始候选顺序故意不按分数排列；真实 rerank 实现消费 fake CrossEncoder 分数，
重排后再取各自 Top-K。调用前后深比较全量原 chunks，结果完全相同。

另一个测试让 rerank 写入 candidate metadata，并捕获公共转换函数返回的 Document；
原 chunks 仍不变，比较返回视图是独立 Document/metadata，源 Document 没有分组字段。

审计复用 `retrieval_evidence_metadata`，只记录文档名、计数及最多十条 chunk ID/重排分数。
真实审计事件与公开 tool trace 不包含 Document 正文。Trace 摘要改为“双文档证据检索完成”。

## FILES CHANGED

| 文件 | 变更 |
| --- | --- |
| `app/tools/compare_documents.py` | 保留名称匹配；删除内部 LLM、关键词 scorer、片段格式化与结论字段；加入 readiness、复制、重排、Document 返回 |
| `app/tools/schemas.py` | CompareDocumentsOutput 改为统一 evidence transport，校验正文与两侧/总数 |
| `app/tools/catalog.py` | 更新工具描述和 side metadata 语义，删除无消费者的内部模型 trace 参数传递 |
| `app/tools/audit.py` | 使用公共 metadata 读取函数，输出有限 IDs/分数/计数 |
| `app/agent/final_response.py` | 仅在统一 evidence 工具集合加入 compare |
| `app/agent/tool_events.py` | 更新 compare 完成摘要 |
| `app/prompts.py` | Agent 自行比较、保留适用条件与例外、证据不足及实际引用规则 |
| `app/llm/factory.py` | 删除 Compare Model 工厂与缓存清理项 |
| `app/llm/__init__.py` | 删除 Compare Model 导入和公开导出 |
| `app/llm/schemas.py` | 删除 CompareDocumentsLLMOutput |
| `tests/test_compare_documents_tool.py` | 替换内部 LLM 测试；覆盖双文档重排、ID、共享状态、schema、audit 和真实 Agent grounding |
| `tests/test_agent_results.py` | 从非 evidence 工具参数中移除 compare |
| `tests/test_tool_contracts.py` | 复用 evidence 校验测试覆盖 CompareDocumentsOutput |
| `tests/test_llm_factory.py` | 删除无消费者的 Compare Model 专属配置测试 |
| `tests/test_config_import_safety.py` | 删除旧 Compare Model cache 断言 |
| `docs/architecture/current-system.md` | 更新五类 evidence 工具、compare 检索与 grounding 链路 |
| `docs/agent-runtime-maintenance.md` | 更新 compare 不再调用内部 LLM 的当前状态 |
| `docs/architecture/agentic-rag-4-result.md` | 本验收报告 |

Python 变更合计净删除 **148 行**；其中生产代码净删除 **146 行**。
没有修改 retrieval 实现、KB 转换器、Evidence identity、Result Builder、ToolStrategy、SSE、
API、前端、应用持久化或依赖文件。历史 Phase 3 验收报告保留其当时结论。

## TESTS RUN

首轮 compare 与通用 tool contracts 定向测试：**104 passed, 1 warning**。
完成剩余更新后的最终 focused run：

```bash
.venv/bin/python -m pytest -q \
  tests/test_compare_documents_tool.py \
  tests/test_agent_results.py \
  tests/test_agent_knowledge_tool_integration.py \
  tests/test_tool_contracts.py \
  tests/test_tool_registry.py \
  tests/test_llm_factory.py \
  tests/test_tool_schema.py \
  tests/test_create_agent_builder.py::test_builder_passes_retrieval_guidance_through_native_system_prompt \
  tests/test_config_import_safety.py::test_primary_consumers_import_without_key_or_runtime_side_effects
```

结果：**251 passed, 1 failed, 1 warning**。未重新运行整套 Phase 3 测试。
Warning 是 jieba 的 pkg_resources 弃用提示。

唯一失败：`test_primary_consumers_import_without_key_or_runtime_side_effects`。
Docling 导入时执行 `docling_core.utils.settings.CoreSettings()`，被测试对
`BaseSettings.__init__` 的全局禁止捕获。在未修改的当前 HEAD 快照中单独执行该测试，
同样 **1 failed**，调用栈落在相同 Docling 行。基线复现日志：
`/tmp/agentic-rag-4-baseline-import.log`。没有修改测试禁令或扩大范围修复 Docling。

工程验证：

- `.venv/bin/lint-imports`：**7 kept, 0 broken**。
- `.venv/bin/python scripts/inspect_agent_runtime.py --check`：**PASS**。
- `.venv/bin/ruff check` 本阶段全部 15 个修改的 Python 文件：**PASS**。
- `git diff --check`：**PASS**。
- 受保护路径 `git diff --exit-code`：**PASS**，包括 app/rag、KB/evidence 转换器、
  Result Builder、framework Agent/middleware、streaming、ModelGateway、API、contracts、
  services、config、storage、frontend 和 requirements 文件。

以上是确定性模型与底层召回/CrossEncoder 分数替身下的测试，真实框架、工具治理、
rerank 实现和引用校验被执行；不表示在线模型或真实 CrossEncoder 权重质量已经验收。

## TESTS NOT RUN

- full pytest: NOT RUN
- Core Agent Evaluation: NOT RUN
- Extended Evaluation: NOT RUN
- full RAG Evaluation: NOT RUN
- real model smoke: NOT RUN
- 专项 SSE 回归：NOT RUN；SSE 源码未修改，并通过受保护路径 diff 检查。

## FINDINGS

- Legacy RAG Graph remains outside production path；legacy 文件保留。
- decompose_top_k remains legacy compatibility debt，未在本阶段移除。
- MCP outputs remain intentionally un-unified，普通操作结果仍兼容无引用回答。
- confidence remains 0.0，未恢复置信度计算。
- SSE remains validated final single delta，未新增 compare event 或逐 token 输出。
- 唯一失败为已在当前 HEAD 复现的 Docling 导入副作用；该项未通过，未计入通过数量。
- Evidence ID 校验确认引用确实存在，不等于语义蕴含验证；没有新增 Judge LLM。
- 历史 `agentic-rag-3-result.md` 的 LLM-backed compare 描述记录 Phase 3 当时状态，
  当前状态以本报告及 current-system 为准。

## GREP PROOF

```text
$ rg -n "get_compare_model|CompareDocumentsLLMOutput|_call_compare_documents_llm" app
(no matches)

$ rg -n "extract_reference_keywords" app/tools/compare_documents.py
(no matches)

$ rg -n "comparison|conflict_detected|summary" app/tools/compare_documents.py app/tools/schemas.py
app/tools/compare_documents.py:102:            "comparison_side": side,
app/tools/compare_documents.py:103:            "comparison_document": doc_name,

$ rg -n "compare_documents" app/agent/final_response.py
45:    "compare_documents",

$ rg -n "CompareAgent|CompareGraph|ComparisonEngine|ComparisonManager|ComparisonEvidencePool" app
(no matches)
```

`comparison` 的剩余匹配仅为用户要求的分组 metadata，不是 answer-like 字段。
对 `app tests scripts` 再扫描工厂、内部 LLM schema 和 compare gateway role，均无匹配。
`extract_reference_keywords` 只在 legacy `app/rag/citations.py` 自身保留。

## NEXT PHASE

AGENTIC-RAG-5 should remove the legacy RAG graph/decomposition/generation compatibility layer and retire decompose_top_k from the production Agent API/runtime after AGENTIC-RAG-4 is accepted.
