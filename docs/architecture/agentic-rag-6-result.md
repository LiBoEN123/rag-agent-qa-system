# AGENTIC-RAG-6 RESULT

## STATUS

**PASS_WITH_FINDINGS**。

Gate A 在删除旧运行时前通过 105 项 focused tests 和 `--validate-only`。
Gate B 已完成删除与调用链收敛。最终关联测试 **268 passed、1 deselected**；
被排除的导入副作用测试已先执行并失败，且在未修改的 Phase 5 HEAD 文件副本中复现同一失败。
这是已确认的既有问题，不能将本报告理解为所有测试均通过。

未执行真实模型、正式 Agent/RAG baseline 或全量 pytest。

## BASELINE

| 项目 | 事实 |
| --- | --- |
| branch | `refactor/tool-output-models` |
| head_before | `b9c9bd83c97093700369839cd6ed78d340c6e24d` |
| head_after | `3fc2f0b66b13de06564014127eaf9ef35bc272b9` |
| worktree_before | dirty，已包含 Phase 5 的生产合同及测试改动；生产路径不再依赖旧 Graph |
| worktree_after | dirty，本阶段修改、删除和新增文件保留待审查 |

任务恢复时 HEAD 已更新为 Phase 5 提交 `3fc2f0b`；该提交不是本助手执行的。
本助手未创建/切换分支，未 commit、push、reset、rebase 或 merge。
下文代码规模与文件清单以当前 Phase 5 HEAD 为对照，避免把已提交的 Phase 5 改动计入本阶段。

## GATE A — EVALUATION CUTOVER

原预测路径：`rag_predictions → retrieve → 独立 RAG workflow → 固定拆分/生成/引用`。

当前路径：

```text
RAG question
→ run_evaluation_chat_turn
→ ChatService.stream
→ Query Rewrite boundary → Router
→ stream_agent
→ 唯一 create_agent.astream
→ governed StructuredTool → Document evidence → ToolMessage
→ Agent 可修改 query/top_k 再检索
→ ToolStrategy(FinalAgentResponse)
→ grounding → ChatServiceOutcome
→ v3 prediction adapter
```

`run_evaluation_chat_turn` 是从现有 Agent Evaluation 提取到同一文件的函数，
复用 InMemorySessionAdapter、正式 ChatService dependencies 和评测身份；没有新建 runtime 类或第二个 Agent。
Agent Evaluation 的多轮会话沿用原有 session adapter；RAG 每题使用空 session。

RAG 请求固定为 `ToolMode.RAG_ONLY`、`permissions=("knowledge_base:read",)`。
当前 checkout 原先只允许单一搜索工具；为满足本阶段三类 KB evidence 工具要求，
可见性与执行策略现在共用三项白名单：`search_knowledge_base`、`compare_documents`、`get_chunk_text`。
权限检查继续执行。Web、MCP、`list_documents` 均不参与 RAG baseline。

捕获点仍是 `execute_project_tool`，只保存工具名、调用 ID、成功标志、字段名、evidence 标记和 KB chunk ID。
不保存 Document 正文、完整 ToolMessage、工具参数或外部来源内容。
调用 ID 与生产工具 trace 对齐，避免用线程完成顺序替代 ToolMessage 观察顺序；无法对齐或 ID 歧义时失败。

## PREDICTION SCHEMA V3

当前 generator 只生成以下字段：

| 字段 | 语义 |
| --- | --- |
| `prediction_schema_version` | 固定为 3 |
| `qid` | 原评测题 ID |
| `question` | 原评测问题 |
| `answer` | 正式 ChatServiceOutcome 的答案 |
| `status` | 正式生产状态，评估不重写 |
| `retrieved_chunk_ids` | 成功工具返回的 KB evidence，按观察顺序稳定去重后的 chunk ID |
| `reference_chunk_ids` | 最终通过 grounding 的 references 对应 chunk ID |
| `retrieved_chunk_count` | 全部去重观察证据数量 |
| `reference_chunk_count` | 最终引用数量 |
| `retrieval_call_count` | 成功返回 evidence 字段的 KB 工具调用数；包括成功返回空 evidence 的检索 |
| `tool_call_count` | 生产结果中实际工具 trace 数；不包含结构化响应的合成工具 |
| `llm_call_count` | 正式结果的模型调用次数 |

只采集 `metadata.source_type == "knowledge_base"` 的 ID。
KB evidence 缺失合法整数 chunk_id，或 ID 不属于当前运行的评测索引时，生成明确失败。
模型错误、工具错误和内部错误中止正式预测；不会写入伪正常答案。
`NO_RELEVANT_INFO`、`NEED_CLARIFICATION` 是合法预测，进入原有行为评分。

不生成旧 final/context IDs、context 字符数、请求级 top_k 或拆分参数。

## RETRIEVAL METRIC SEMANTICS

`retrieved_chunk_ids = ordered unique KB evidence observed across Agent tool calls`。

例如两轮 `[1, 2]`、`[2, 3]` 得到 `[1, 2, 3]`；不按最终分数全局重排。
最终仅声明 `[1, 3]` 时，reference IDs 为 `[1, 3]`。

`--retrieval-k` 是 **scoring cutoff only**：只对观察序列前 K 项计算 retrieval metrics，
不会进入 ChatServiceRequest 或任何工具参数。每次工具 `top_k` 由模型自主决定。
报告保留完整观察 ID/数量；cutoff 仅作用于评分。

Manifest 使用 `prediction_schema_version=3`、`retrieval_metric_k`、索引来源与计数、
embedding/rerank 模型、chunk size/overlap、RAG tool mode/permissions。
requested/effective 参数验证与旧 context/decompose 配置均已移除。

## MULTI-SUBQUESTION PROOF

确定性集成测试通过真实 ChatService、真实 create_agent、真实 KB handler 和治理边界执行：

```text
Model 1 → search("A 申请条件", top_k=3)
ToolMessage → chunks [1, 2]
Model 2 确认已收到 A evidence
        → search("B 申请材料 办理时间", top_k=7)
ToolMessage → chunks [2, 3]
Model 3 → FinalAgentResponse(evidence_ids=[A-id, B-id])
```

断言：retrieval/tool calls 为 2，LLM calls 为 3，top_k 序列 `[3, 7]`，
retrieved IDs `(1, 2, 3)`，grounded reference IDs `(1, 3)`。
测试还覆盖同一轮并行工具完成顺序反转，观察 ID 顺序保持 `(1, 3)`。

最终回归直接读取原题集的 hard `cross_doc` q042 与 hard `partial_answer` q052，
使用确定性 fake model/evidence 验证控制流，不修改题目或 gold，不声称真实答案质量达标。
另覆盖单次搜索、比较文档、chunk 原文、部分证据、零证据、澄清、非法 ID 和策略拒绝。
旧拆分模型及 workflow 已删除，多子问题能力仍由同一 Agent 的连续工具调用提供。

## GATE B — LEGACY RETIREMENT

Gate A 通过后重新扫描消费者，表格诊断先收敛到 retrieve/rerank/诊断选择，再删除：

- `app/rag/graph/` 的全部六个源码文件。
- 独立 decomposition、generation、citations 及 RAG schemas/results。
- Decompose model 工厂、缓存清理、exports、QueryDecompose schemas；旧 AnswerOutput 同步退役。
- Decompose settings、validators、示例环境变量及旧 context 配置。
- 旧生成/拆分 prompts 和 fallback constant。
- 旧 multi-query 检索 helper 及 pipeline wrapper；Graph 专用澄清 helpers。
- 一一对应的旧 Graph、生成、引用、context 与 v2 producer 测试。
- 已无目标模块的临时 Import Linter 隔离合同。

未留下 shim、转发模块或旧 Python workflow 兼容层。

## RETAINED RAG CORE

保留 indexing、store/chroma、parsing/ingest/structured chunking、BM25/Embedding、
0.4/0.6 hybrid、candidate_k、CrossEncoder、普通 retrieval/rerank/pipeline、Document identity/evidence、
生产 grounding、Query Rewrite、Agent 主模型、Evaluation Judge/cache。

保留 `should_refuse` 和 `vector_threshold`，因为诊断 `/search` 与表格诊断仍使用它们。
表格关系诊断现在明确表示关系 chunk 是否通过重排及最终诊断选择，不再推断 Agent context。
对保留的 retrieval、pipeline、policy 函数进行了 AST 对比，实现均与 Phase 5 HEAD 一致。
Agent/Judge prompt 常量正文一致，题集文件无差异。

## EVALUATION CHANGES

Prediction producer 从固定 workflow 切换到真实生产 Agent，schema 2 → 3。
v3 的 context metrics 为 `null` / unavailable / not applicable，不计为零；Reference metrics 保留。
历史 v1/v2 CSV 继续纯数据兼容，不需要旧 runtime。
Judge、judge cache、关键点和 forbidden 判断保持；Final Score 的行为、关键点、检索及 forbidden 权重未修改。

Prediction Schema v3 is a new baseline generation contract.
Raw scores are not strictly comparable to historical v2 workflow scores.

生成的当前 Markdown baseline report 也包含上述基线不可严格比较的说明。

## CODE REDUCTION

计数为相对 `3fc2f0b` 的物理 Python 行数，包含注释/空行；包含新增、尚未跟踪的 v3 测试。

| 范围 | 增加 | 删除 | 净变化 |
| --- | ---: | ---: | ---: |
| production Python (`app/`) | 15 | 1703 | **-1688** |
| evaluation Python (`scripts/evaluation/`) | 265 | 445 | **-180** |
| 全部 Python（含 tests/scripts） | 606 | 3965 | **-3359** |

完整删除的 11 个 legacy RAG 文件共 **1377 行**；其他旧 helper、模型、prompt/config 和测试删除另外计入上表。
没有新增 EvaluationGraph、Runtime/Manager 类、第二个 create_agent 或第二个推理模型。

## FILES CHANGED

以下路径相对仓库根目录。

| 文件 | 改动 |
| --- | --- |
| `.env.example` | 删除旧拆分/context 配置，注明 TOP_K 仅为诊断默认值 |
| `.importlinter` | 删除已无目标模块的临时 legacy 隔离合同 |
| `README.md` | 更新当前架构、目录、配置、评估 CLI 与 v3 语义 |
| `app/contracts/chat.py` | 定义三项 RAG evidence 工具白名单 |
| `app/agent/tool_visibility.py` | 模型可见性使用共享白名单及原权限检查 |
| `app/tools/policy.py` | 执行授权使用同一白名单，继续 fail closed |
| `app/config.py` | 删除 Decompose/context 字段及 validator 引用 |
| `app/llm/__init__.py` | 移除拆分工厂 export |
| `app/llm/factory.py` | 移除拆分客户端/缓存清理，修正 Agent 主模型说明 |
| `app/llm/schemas.py` | 删除旧回答及拆分 schemas，保留 QueryRewriteOutput |
| `app/prompts.py` | 删除旧 workflow 常量，保留 Agent/Judge 正文 |
| `app/rag/__init__.py` | 更新包职责说明 |
| `app/rag/pipeline.py` | 删除旧 multi-query wrapper 和零消费者 STOP_WORD |
| `app/rag/retrieval.py` | 删除零消费者 multi-query helper/export |
| `app/rag/policy.py` | 删除 Graph 澄清 helpers，保留诊断分数门 |
| `scripts/evaluation/agent.py` | 提取共用 ChatService 函数；扩展安全工具 metadata capture |
| `scripts/evaluation/rag_predictions.py` | 正式 ChatService v3 producer、ID 校验、顺序对齐和失败关闭 |
| `scripts/evaluation/runner.py` | cutoff CLI 与调度，删除旧参数传播及验证 |
| `scripts/evaluation/manifest.py` | v3/cutoff 元数据，删除旧配置和 legacy prompt hash |
| `scripts/evaluation/rag/common.py` | v3 按观察 ID 解析，保留历史纯数据兼容 |
| `scripts/evaluation/rag/evaluation.py` | v3 retrieval/reference 评分和 N/A context，保留公式 |
| `scripts/evaluation/report.py` | 当前报告使用 observed/grounded 语义及不可严格比较说明 |
| `scripts/evaluation/table_diagnostics.py` | 删除 context 依赖，关系指标指向最终诊断选择 |
| `scripts/evaluation/README.md` | 更新评估运行方式、schema 与证据口径 |
| `scripts/inspect_agent_runtime.py` | 退役已无模块的旧符号检查，保留实际检索链检查 |
| `scripts/smoke_test_chat_stream.py` | 删除已被 ChatRequest 拒绝的旧召回 CLI/请求字段 |
| `tests/test_evaluation_rag_predictions_v3.py` | 新增完整生产链 fake model、证据/状态/cutoff 回归 |
| `tests/test_evaluation_rag_evidence_funnel.py` | v3 cutoff、引用和 context N/A 回归 |
| `tests/test_evaluation_runner.py` | 新 CLI、capture metadata、当前报告说明与调用隔离断言 |
| `tests/test_evaluation_manifest.py` | v3/cutoff/权限及旧字段退役断言 |
| `tests/test_evaluation_smoke.py` | 离线预测与报告 fixture 使用新调用边界 |
| `tests/test_evaluation_table_diagnostics.py` | 最终诊断选择关系指标及无 context 字段验证 |
| `tests/test_dynamic_tool_visibility.py` | 三工具白名单、非 RAG 拦截，fake model 适配 ToolStrategy |
| `tests/test_tool_policy.py` | 三工具都继续接受 knowledge_base 权限检查 |
| `tests/test_agent_knowledge_tool_integration.py` | 移除指向已删除模块的禁止调用 monkeypatch，保留实际 Agent/KB 断言 |
| `tests/test_chat_service_agent_integration.py` | 移除旧模块 monkeypatch，更新 RAG_ONLY 三工具可见性 |
| `tests/test_web_search_agent_integration.py` | 移除已删除 Graph 的禁止调用 monkeypatch，保留 Web 流程测试 |
| `tests/test_config.py` | 移除已删除参数的 validator cases |
| `tests/test_config_import_safety.py` | 移除已删除 model cache 的检查项；既有导入问题未掩盖 |
| `tests/test_llm_factory.py` | 删除独立拆分工厂测试 |
| `tests/test_llm_gateway.py` | 删除旧 generation 接入测试；用测试本地 schema 保留通用 Gateway 回归 |
| `tests/test_rag_retrieval.py` | 删除拆分 workflow cases，保留检索和诊断策略测试 |
| `docs/architecture/current-system.md` | 当前唯一生产/评估路径与退役边界 |
| `docs/architecture/module-boundaries.md` | RAG 模块职责收敛到检索与索引 |
| `docs/agent-runtime-maintenance.md` | 当前工具输出、参数和维护入口 |
| `docs/api/chat-streaming.md` | 说明 RAG_ONLY 三项 KB evidence 工具范围 |
| `docs/evaluation/baseline-v2.md` | 当前 v3 schema、CLI 和指标；保留历史运行数值 |
| `docs/evaluation/table-aware-eval.md` | 只增加当前入口说明，保留历史报告原文 |
| `docs/ingestion/structured-documents.md` | 更新旧 context 限制的当前状态 |
| `docs/tools/mcp-client.md` | 更新唯一 Agent 的工具边界与 RAG evaluation 排除 MCP 说明 |
| `docs/architecture/agentic-rag-6-result.md` | 本验收报告 |

## FILES DELETED

| 文件 | 删除原因 |
| --- | --- |
| `app/rag/graph/__init__.py` | 旧 Graph exports |
| `app/rag/graph/builder.py` | 第二个回答 workflow 构建器 |
| `app/rag/graph/conditions.py` | 旧 workflow 分支 |
| `app/rag/graph/nodes.py` | 旧拆分、生成、context、引用节点 |
| `app/rag/graph/runtime.py` | 旧独立运行入口 |
| `app/rag/graph/state.py` | 旧 Graph State |
| `app/rag/decomposition.py` | 独立拆分模型 workflow |
| `app/rag/generation.py` | 旧 context/回答/置信度计算 |
| `app/rag/citations.py` | 旧引用 heuristic |
| `app/rag/results.py` | 旧 RAG result builders |
| `app/rag/schemas.py` | 旧 RAGResult schema |
| `tests/test_rag_graph_runtime.py` | 对应已退役 runtime |
| `tests/test_rag_graph_builder.py` | 对应已退役 Graph |
| `tests/test_rag_context_selection.py` | 对应旧 context 阶段 |
| `tests/test_rag_generation_confidence.py` | 对应旧生成/置信度算法 |
| `tests/test_citation_verifier.py` | 对应旧引用 heuristic |
| `tests/test_evaluation_rag_predictions_v2.py` | 旧 producer/参数合同，已由 v3 测试替代 |

## TESTS RUN

Gate A 在删除旧模块前：105 passed；随后 `python scripts/run_evaluation_baseline.py --validate-only` PASS。

最终通过 `.venv` 执行以下 focused 测试：

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest -q \
  tests/test_evaluation_rag_predictions_v3.py \
  tests/test_evaluation_rag_evidence_funnel.py \
  tests/test_evaluation_runner.py \
  tests/test_evaluation_manifest.py \
  tests/test_evaluation_agent.py \
  tests/test_evaluation_rag_behavior.py \
  tests/test_evaluate_rag_judge.py \
  tests/test_rag_eval_retrieval.py \
  tests/test_dynamic_tool_visibility.py \
  tests/test_tool_policy.py \
  tests/test_evaluation_smoke.py \
  tests/test_evaluation_table_diagnostics.py \
  tests/test_llm_factory.py \
  tests/test_llm_gateway.py \
  tests/test_config.py \
  tests/test_config_import_safety.py \
  tests/test_rag_retrieval.py \
  tests/test_agent_knowledge_tool_integration.py \
  tests/test_chat_service_agent_integration.py \
  tests/test_web_search_agent_integration.py \
  tests/test_scripts_layout.py \
  -k 'not test_primary_consumers_import_without_key_or_runtime_side_effects'
```

结果：**268 passed, 1 deselected, 1 warning**。
Warning 来自 jieba 对 `pkg_resources` 的弃用提示。

工程检查：

- `.venv/bin/python scripts/run_evaluation_baseline.py --validate-only`：PASS。
- `.venv/bin/lint-imports`：7 kept，0 broken。
- `.venv/bin/python scripts/inspect_agent_runtime.py --check`：PASS。
- `.venv/bin/ruff check` 本阶段 38 个仍存在的修改/新增 Python 文件：PASS。
- `git diff --check`：PASS。
- Python compileall：PASS。
- 数据集、Agent/Judge prompts 和保留检索函数实现对比：PASS。

## TESTS NOT RUN

- full pytest: NOT RUN。
- Core Agent Evaluation: NOT RUN。
- Extended Evaluation: NOT RUN。
- full real-model Agentic RAG Evaluation: NOT RUN。
- real model smoke: NOT RUN。
- frontend tests/build: NOT RUN（前端无改动）。

## FINDINGS

1. **历史 v2 原始分数不与 v3 严格直接比较。** 本阶段验证合同和控制流，不证明真实模型质量或规划效果。
2. **既有导入测试失败**：`test_primary_consumers_import_without_key_or_runtime_side_effects`
   全局替换 `BaseSettings.__init__`，导入链进入 `docling_core.utils.settings.CoreSettings()` 时被测试自身拒绝。
   在 `3fc2f0b` 的 `git archive` 文件副本中独立执行同一测试，仍为 **1 failed, 1 deselected**，
   错误同为 `AssertionError: external resource or settings created during import`。
   未更改生产导入方式或放宽该测试；最终 focused 命令明确排除它。
3. `router_llm_used` 保留为 `Literal[False]` 的兼容字段；没有 Router LLM。
4. retrieval/answer confidence 继续为 `0.0`。
5. MCP evidence/Bridge 合同未修改；RAG_ONLY 不发现 MCP，治理仍拒绝非 RAG 调用。
6. SSE 协议和生产 streaming 实现未修改，仍为校验后的一次完整 answer delta 与权威 final。
7. 诊断 `should_refuse` / `vector_threshold` 保留，因为 `/search` 与表格诊断仍有真实消费者。
8. RAG_ONLY 从原先单搜索工具扩展为三种 KB evidence 工具，是使正式比较与原文读取参与 baseline 的必要合同调整；可见性与执行授权同时覆盖。
9. 当前手动 SSE smoke 客户端仍残留已退役请求参数，因此同步删除其 CLI/请求字段；没有运行真实服务 smoke。

## GREP PROOF

以下五组扫描对 `app scripts` 均 **无匹配**（rg exit 1 表示无匹配）：

```bash
rg -n "run_rag_subgraph|get_compiled_rag_graph|RAGGraphState|RAGGraphError" app scripts
rg -n "get_decompose_model|QueryDecomposeItem|QueryDecomposeOutput" app scripts
rg -n "SYSTEM_PROMPT_B|DECOMPOSE_PROMPT|NO_ANSWER_FALLBACK" app scripts
rg -n "retrieve_with_sub_queries" app scripts
rg -n "build_context_with_selection|max_context_chars" app scripts
```

同一 Agent 链路的实际扫描：

```text
scripts/evaluation/rag_predictions.py:16: ... capture_tool_output_keys, run_evaluation_chat_turn
scripts/evaluation/rag_predictions.py:96: outcome = run_evaluation_chat_turn(
scripts/evaluation/rag_predictions.py:99: tool_mode=ToolMode.RAG_ONLY,
scripts/evaluation/rag_predictions.py:102: permissions=("knowledge_base:read",),
scripts/evaluation/agent.py:1293: def run_evaluation_chat_turn(
scripts/evaluation/agent.py:1326: async for event in stream_agent(**kwargs):
scripts/evaluation/agent.py:1342: async for event in service.stream(request):
scripts/evaluation/agent.py:1371: outcome = run_evaluation_chat_turn(
app/agent/streaming.py:257: agent_stream = invocation.agent.astream(
app/agent/framework_agent.py:185: agent = create_agent(
```

历史验收 Markdown 中保留当时的旧符号与结果；未改写历史报告。

## NEXT PHASE

AGENTIC-RAG-7 should run a controlled real-model Agentic RAG baseline on the existing evaluation set and use the results to tune Agent retrieval planning / tool guidance only where measured regressions justify it.
