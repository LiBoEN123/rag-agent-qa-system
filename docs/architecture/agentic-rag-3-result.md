# AGENTIC-RAG-3 RESULT

## STATUS

PASS_WITH_FINDINGS

结构化回答、确定性引用绑定、状态恢复和 SSE 验证通过。主要权衡是答案在校验完成后
作为单次完整 delta 发出；LangChain 1.3.14 的工具计数需要一层过滤合成调用的适配。

## BASELINE

```text
branch: refactor/tool-output-models
head_before: 8eb8086316b12baeb10c40d9cccb0fab16473912
head_after: 8eb8086316b12baeb10c40d9cccb0fab16473912
worktree_before: clean
worktree_after: 16 modified files + 2 new files, all uncommitted
commit / push / branch switch: NOT PERFORMED
```

编码前已读取指定生产模块、测试和引用扫描。使用当前 `.venv` 实际检查：
`langchain==1.3.14`、`langchain-core==1.4.9`；`create_agent` 支持 `response_format`，
`ToolStrategy` 支持 `handle_errors=True` 和原生 Pydantic validation retry。

## ARCHITECTURE BEFORE

```text
Agent → normal AIMessage → references=[]
```

## ARCHITECTURE AFTER

```text
唯一 create_agent
→ Retrieval / Fetch Tool → Document → ToolMessage
→ Agent observe / retry / retrieve again
→ LangChain structured_response
→ cited evidence IDs
→ deterministic evidence-ID validation
→ grounded references / no-evidence status / clarification
→ AgentRunResult → SSE final
```

没有新增 Parent Graph、第二个 Agent Loop、业务 SubmitAnswer Tool 或 Judge LLM。
终止治理优先于结构化输出；非法或缺失的结构化输出不会回退到普通 AIMessage。

## FRAMEWORK REUSE

- `create_agent(response_format=ToolStrategy(FinalAgentResponse, handle_errors=True))`。
- LangChain 原生 schema 工具绑定、Pydantic 校验、错误反馈、retry 和 structured_response state。
- LangChain `Document.model_validate` 与现有 `validate_evidence_documents`。
- 本次 ToolMessage history、现有 Agent middleware/runtime、原生调用上限。
- 现有 `make_snippet`、`safe_float` 和公开 Reference API。

新增第三方依赖：NO。依赖文件、Retrieval、Document ID 与 Evidence metadata 实现未修改。

## FINAL RESPONSE CONTRACT

| 字段 | 合同 |
| --- | --- |
| `outcome` | `answer` / `no_evidence` / `clarify` |
| `answer` | 默认空字符串；answer outcome 要求非空 |
| `evidence_ids` | 默认空列表；禁止重复；动态检查当前 Run 成员资格 |
| `clarification_question` | 仅 clarify 可非空，且 clarify 时必须非空 |

Schema 不强制 answer 必须携带 ID；是否已有 Document evidence 由 Result Builder 判断。
模型不能选择 Tool/Model/Internal Error 状态。

## GROUNDING RULES

- 仅从本次成功的 `search_knowledge_base`、`get_chunk_text`、`web_fetch`、
  `web_pdf_fetch` ToolMessage 的标准 project tool result 恢复 Document。
- `web_search` snippets、目录、比较、普通 MCP 输出和模型文本不自动成为 evidence。
- 所有声明 ID 必须存在于本次证据历史；未知 ID 返回 `MODEL_ERROR`，清空 references，
  不向用户暴露候选答案或 Document 正文错误详情。
- 已有 Document 且 outcome=answer 时，evidence_ids 必须非空。
- 未产生 Document 时，非 evidence 工具允许无引用 answer。
- `no_evidence` → `NO_RELEVANT_INFO`；引用为空，无澄清标志。空说明使用固定 fallback。
- `clarify` → `NEED_CLARIFICATION`；问题非空，引用为空，保留真实工具执行摘要。
- fatal/tool/model/budget stop 优先；同轮残留 structured_response 不能覆盖业务工具上限。

## REFERENCES

Retrieved evidence 数量 ≠ 最终 references 数量。真实集成测试取回两条 Document，只声明
第二条，最终只投影第二条。另一个测试两次检索分别只返回不同 Document，最终引用同时包含
两次 ToolMessage 的证据，证明恢复覆盖完整本次历史。

保持 evidence_ids 声明顺序；schema 拒绝重复 ID，重复收到同一 Document 不生成重复引用。
reference_id 使用真实 Document.id。KB 保留文件、来源、chunk、position 与检索分数；
缺少旧 chunk_id 时使用 Document.id 维持公开合同，整数 0 正常保留。
Web/PDF 使用标题或 URL、source URI、Document.id 和起始页；分数为 0.0。
snippet 使用现有 300 字符有限预览。公开 ChatReferences、API 和前端模型未修改。

## STREAMING

普通 AIMessage 文本、AIMessageChunk 文本及 schema args 均不会直接成为 answer_delta。
Result Builder 校验通过后，answer/no_evidence 在 result 前发送一次完整 answer_delta，
与公开 final answer 一致。澄清依照现有公开策略输出 clarification_question；入口 Router
短路不变。业务工具开始/完成和 progress 继续实时输出。

当前不是 token-level answer streaming；这是为避免无效引用答案提前泄漏而接受的权衡。
没有 JSON 增量解析器，也没有重复发送最终答案。

实际框架测试确认：合成响应由 LangChain 在模型节点解析，不进入业务 ToolNode，因而不进入
ProjectStructuredTool execution/governance、业务 tool audit 或 tool_call trace。模型 trace
中的业务可见工具与请求次数也排除合成调用；SSE 不出现其内部名称或原始 JSON。

适配前测试确实失败：1 次 list_documents + 1 次合成响应被原生 ToolCallLimit 计为 2。
`ProjectToolCallLimitMiddleware` 只对原生 after_model 的计数输入过滤合成调用，剩余计数、
拦截和工具错误消息仍由原生中间件处理；超限同步记录项目 stop_reason，防止残留 final 胜出。
适配后预算为 1 的检索/回答和 schema retry 均通过，不新增业务 audit。

llm_call_count 按实际 Agent Model 调用计数（含失败和真实重试）。Schema 校验本身不增加调用：
普通检索/回答为 2，检索后一次 schema retry 为 3。重试继续受原生 ModelCallLimit 约束。

## FILES CHANGED

| 文件 | 变更 |
| --- | --- |
| `app/agent/final_response.py` | 新增最终合同、ToolMessage Document 恢复与引用投影，97 行 |
| `app/agent/framework_agent.py` | 接入 ToolStrategy，模型业务工具指标排除合成调用 |
| `app/agent/framework_middleware.py` | 原生工具限流输入适配与结构化完成状态 |
| `app/agent/results.py` | structured_response 权威输出、动态校验、状态/引用映射；删除旧 final AI 回退 |
| `app/agent/streaming.py` | 移除普通模型文本输出，校验后发送一次完整答案 |
| `app/prompts.py` | 最终 outcome、实际引用 ID、检索重试与证据不足规则 |
| `tests/support/agent_runtime.py` | 最小扩展 Fake：绑定 kwargs、真实 schema/name、结构化响应和回调 |
| `tests/test_create_agent_builder.py` | 框架接入、合成调用计数、重试预算、同轮终止优先级 |
| `tests/test_agent_results.py` | 静态 schema、选择性引用、非法 ID/结果、状态、KB/Web/PDF API 合同 |
| `tests/test_agent_knowledge_tool_integration.py` | 真实 handler/ToolMessage 多轮检索、grounding、retry 与 audit |
| `tests/test_agent_streaming.py` | 原生 Agent 流、合成事件过滤、未通过校验的答案零泄漏 |
| `tests/test_chat_service_agent_integration.py` | 更新真实服务的结构化模型夹具、KB 无证据与 Web/PDF 引用 |
| `tests/test_chat_service_streaming.py` | 实际 stream_agent → ChatService → 公共响应、delta 与指标一致 |
| `tests/test_web_search_agent_integration.py` | 复用已有 Fake，验证 Web/PDF 实际引用和权限边界 |
| `docs/architecture/current-system.md` | 更新生产结构化回答、grounding 与计数边界 |
| `docs/api/chat-streaming.md` | 更新单次最终 delta 及安全投影合同 |
| `README.md` | 更新生产链路、引用、状态与流式行为 |
| `docs/architecture/agentic-rag-3-result.md` | 本验收报告 |

## TESTS RUN

```bash
.venv/bin/python -m pytest -q \
  tests/test_create_agent_builder.py \
  tests/test_agent_results.py \
  tests/test_agent_knowledge_tool_integration.py \
  tests/test_agent_streaming.py \
  tests/test_chat_service_agent_integration.py \
  tests/test_chat_service_streaming.py \
  tests/test_web_search_agent_integration.py
```

结果：**161 passed, 1 warning**。warning 为 jieba 的 pkg_resources 弃用提示。
验证使用真实 create_agent、middleware、工具执行链、解析器和服务；模型/检索底层/网络使用
确定性 fixture。不是在线模型或外部服务验收。

- Ruff：所有本阶段实际修改/新增 Python 文件 PASS。
- `.venv/bin/lint-imports`：7 kept, 0 broken。
- `.venv/bin/python scripts/inspect_agent_runtime.py --check`：PASS。
- `git diff --check`：PASS。
- `git diff --exit-code -- app/tools app/rag app/api app/contracts app/services frontend requirements.txt requirements.lock`：PASS（这些边界无修改）。

## TESTS NOT RUN

- full pytest: NOT RUN
- Core Agent Evaluation: NOT RUN
- Extended Evaluation: NOT RUN
- full RAG Evaluation: NOT RUN
- 真实 Answer Model / 在线 MCP / 真实 HTTP 服务 smoke: NOT RUN

## FINDINGS

- compare_documents remains LLM-backed specialized tool。
- MCP output 尚未统一为 Document evidence，操作回答继续兼容无引用路径。
- Legacy RAG Graph remains outside production path；未删除 legacy 文件，也未恢复其 citation gate。
- confidence remains 0.0；不同来源分数没有统一可比较尺度。
- semantic entailment Judge 未引入；ID 绑定验证不代表已证明答案与正文的语义蕴含。
- SSE 答案为 final single delta，非逐 token；实时进度和业务工具事件继续保留。
- LangChain 1.3.14 原生 ToolCallLimit 会计入合成响应，已通过先失败后通过的测试证明并作最小适配。
- 没有运行或声称通过全量测试、完整评测或真实模型兼容性验证。

## GREP PROOF

```text
$ rg -n "response_format|structured_response|ToolStrategy" app
app/agent/framework_middleware.py:329:            state.get("structured_response") is not None
app/agent/results.py:276:        final = framework_output.get("structured_response")
app/agent/framework_agent.py:14:from langchain.agents.structured_output import ToolStrategy
app/agent/framework_agent.py:189:        response_format=ToolStrategy(FinalAgentResponse, handle_errors=True),

$ rg -n "EvidencePool|EvidenceManager|FinalizerGraph|CitationGraph|submit_grounded_answer" app
(no matches)

$ rg -n "verify_answer_citations" app/agent app/services
(no matches)

$ rg -n "_last_final_ai" app/agent
(no matches)
```

## NEXT PHASE

AGENTIC-RAG-4 should address the remaining specialized/non-Document knowledge tools and legacy RAG/decomposition cleanup after structured grounding is accepted.
