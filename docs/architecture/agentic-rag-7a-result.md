# AGENTIC-RAG-7A RESULT

## STATUS

**PASS_WITH_FINDINGS**。q054 的 locator contract blocker 已修复：147 个 focused tests
通过，两次真实模型运行均为 `OK`，未复现 Document.id/chunk_id 混淆。
两次仍重复读取 Search 已返回的 chunk，作为效率 finding 保留，不继续调优。

本阶段只完成代码修复、focused tests、真实模型 blocker smoke。正式 Prediction v3
baseline **尚未建立**；没有运行 AGENTIC-RAG-7B。

## BASELINE

- branch: `refactor/tool-output-models`
- head_before: `cbc5f916dbaef6e8badeffe700588293c5837d4f`
- head_after: `cbc5f916dbaef6e8badeffe700588293c5837d4f`
- worktree_before: clean
- worktree_after: dirty；11 个已有文件修改，新增本报告，均未暂存或提交。
- Python: 项目 `.venv`，Pydantic `2.12.5`。
- smoke 时间：2026-09-18 15:26:37–15:27:40，Asia/Shanghai。
- smoke 目录：[agentic-rag-7a-smoke-cbc5f91](../../outputs/evaluation_baseline_v2/agentic-rag-7a-smoke-cbc5f91/)。
  这是 Git 忽略的 dirty-worktree 诊断产物，不是正式 baseline。

未 commit、push、切换/创建分支、reset、rebase 或 merge；未使用 `--allow-dirty`。
Phase 7 原 FAIL 报告和旧输出目录保持不变。

## ROOT CAUSE

`Document.id = knowledge_base:<sha256>` 是 stable evidence identity，用于最终
`evidence_ids`、grounding 和引用身份。`Document.metadata.chunk_id` 是当前内存索引的
runtime compatibility locator，用于 `get_chunk_text`、评测 chunk 匹配和诊断。

旧 schema 允许任意非空字符串，模型传入 Document.id 也能通过校验，但
`get_chunk_full_text` 实际仅按 `metadata.chunk_id` 查找。schema 允许的值域大于
handler 支持的值域，造成 Phase 7 q054 连续三次工具错误后触发
`max_tool_errors_reached / MODEL_ERROR`。本次只修复这一合同歧义。

## CONTRACT BEFORE

| 边界 | 旧合同 |
| --- | --- |
| GetChunkTextArgs.chunk_id | `int \| arbitrary nonempty str`，也接受 0 |
| Document.id | `knowledge_base:<sha256>` |
| handler | 按 `metadata.chunk_id` 查找，不按 Document.id 查找 |
| locator 不存在 | `execution_error` |

## CONTRACT AFTER

| 边界 | 新合同 |
| --- | --- |
| GetChunkTextArgs.chunk_id | `int`，`ge=1`；公开 JSON Schema 为 `type=integer, minimum=1` |
| locator 来源 | `Document.metadata.chunk_id` |
| 数字字符串 | 保留 Pydantic 标准解析：`"382" → 382`，handler 收到整数 |
| Document.id、非数值字符串、空字符串、0、负数 | schema 拒绝 |
| handler annotation | `get_chunk_text(chunk_id: int)` |

未添加 strict/custom validator、额外 evidence_id 参数或 Union locator。

## WHY NOT SUPPORT DOCUMENT.ID

两个 ID 职责不同；Search/Compare 已返回完整 `page_content`。增加双 locator resolver
会模糊合同，并引入遍历 chunks、重算 evidence identity 的兼容逻辑。
本次没有新增 resolver、Manager、Registry、Graph、Runtime 或其他定位子系统。
稳定 ID 算法和 runtime chunk 编号算法完全不变。

## TOOL ERROR SEMANTICS

- schema 非法值在 StructuredTool/Pydantic 层 fail fast，不进入 handler；直接 BaseTool
  调用抛 `ValidationError`，现有 Agent 治理路径负责标准 `INVALID_ARGUMENTS` 映射。
- 类型合法但不存在的 `chunk_id=999999`：handler 抛现有 `ToolExecutionError`，
  `code=ToolErrorCode.INVALID_ARGUMENTS`、`retryable=False`；治理层返回标准
  `invalid_arguments`，不再包装为 `execution_error`。
- 错误消息明确提示：`get_chunk_text.chunk_id 必须使用知识库 evidence 的
  metadata.chunk_id，不要使用 Document.id`。typed error 不被通用异常捕获吞掉。
- 未新增错误码或扩大 `DEPENDENCY_UNAVAILABLE` 含义；不存在 locator 不映射为依赖错误。

## PROMPT CHANGE

仅在原第 8 条 evidence ID 规则后新增：

```text
Document.id 只用于 evidence_ids / 引用身份；get_chunk_text 必须传 Document.metadata.chunk_id，
绝不能传 Document.id。search_knowledge_base / compare_documents 返回的 page_content 已是完整
chunk 正文，不要为重复读取同一 evidence 再调用 get_chunk_text。
```

无 q054 特例，无 planning framework 或 partial-answer 规则修改。

## TOOL VERSION

- before: `get_chunk_text 1.0.0`
- after: `get_chunk_text 1.1.0`

其他工具版本保持 `1.0.0`。description 同时明确 locator 来源、禁止 Document.id、
Search/Compare 已返回完整正文，以及用户明确给出 chunk_id 时直接读取的用途。

## DETERMINISTIC TESTS

实际执行：

```bash
.venv/bin/python -m pytest -q \
  tests/test_tool_schema.py \
  tests/test_tool_registry.py \
  tests/test_local_tool_catalog.py \
  tests/test_agent_knowledge_tool_integration.py \
  tests/test_tool_governance_integration.py \
  tests/test_compare_documents_tool.py \
  tests/test_iterative_agent_loop.py
```

**147 passed，1 warning，2.47s**。warning 为 jieba 引用 `pkg_resources` 的弃用提示。
[完整测试日志](../../outputs/evaluation_baseline_v2/agentic-rag-7a-smoke-cbc5f91/focused-tests.log)。

| 测试文件 | 本次相关覆盖 |
| --- | --- |
| test_tool_schema.py | 382、数字字符串解析；拒绝 Document.id、0、负数、空串和非数值字符串 |
| test_tool_registry.py | 唯一版本升级；非法 locator 不进入 lookup；真实数字 lookup、跨工具 identity/provenance；999999 返回不可重试参数错误 |
| test_local_tool_catalog.py | 模型可见 JSON Schema 为 integer/minimum=1；版本与 description |
| test_agent_knowledge_tool_integration.py | 真实 Search → ToolMessage → fake Model；正文可直接回答；ToolMessage 的 Document.id 校验失败而 metadata.chunk_id 成功 |
| test_tool_governance_integration.py | 现有工具治理、trace 与正文审计边界回归 |
| test_compare_documents_tool.py | 同一 chunk 的 Search/Get/Compare identity 保持一致；旧 0 fixture 改为 382 |
| test_iterative_agent_loop.py | 显式数字读取、顺序/并行调用、参数修正恢复、原预算行为；旧任意字符串 fixture 改为正整数 |

未编写要求 q054 必须调用 `get_chunk_text` 的测试。

## REAL-MODEL Q054

直接复用 `initialize_rag_for_predictions()` 与 `run_rag_once_with_result()`，
使用原评测题：`学业警示《通知书》或《告知书》的具体模板内容是什么？`
空 session，`rag_only`，权限为 `knowledge_base:read`。

真实配置：qwen-plus，temperature=0.1，max_tokens=1024；Embedding 为
Qwen/Qwen3-Embedding-0.6B，Reranker 为 BAAI/bge-reranker-v2-m3。
预算保持 iterations=10、tool_calls=10、tool_errors=3、total_cost_ms=120000。
Chroma 加载 16 documents / 493 chunks；force_rebuild=False，corpus unchanged。

| 项目 | Run 1 | Run 2 |
| --- | --- | --- |
| status | OK | OK |
| tool sequence | search → get_chunk_text → get_chunk_text → get_chunk_text；全部 success | 同左；全部 success |
| search args | query=`学业警示 通知书 模板`，top_k=5 | 同左 |
| get_chunk_text args | `{"chunk_id":382}`、`{"chunk_id":383}`、`{"chunk_id":384}` | 同左；全部 JSON integer |
| observed chunk IDs | 382, 239, 383, 384, 385 | 同左 |
| reference chunk IDs | 382, 383, 384 | 同左 |
| tool / LLM calls | 4 / 5 | 4 / 5 |
| elapsed | 18.617s | 15.818s |
| stop reason | 正常结构化结束；result.reason=`Agent model produced structured answer` | 同左 |
| tool errors / max_tool_errors_reached | 0 / 未出现 | 0 / 未出现 |

`AgentRunResult` 对外字段是 `reason`，不另造独立 stop_reason 观测值。
两次均说明可确认文书名称/流程，但检索证据没有展示具体模板；未执行 Judge 或进行答案质量评分。

**Document.id/chunk_id confusion reproduced after fix: NO（两次运行）。**
runtime blocker = **FIXED**；答案质量与重复原文读取留待正式 baseline 分析。

证据：[Run 1](../../outputs/evaluation_baseline_v2/agentic-rag-7a-smoke-cbc5f91/q054-run-1.json)、
[Run 2](../../outputs/evaluation_baseline_v2/agentic-rag-7a-smoke-cbc5f91/q054-run-2.json)、
[真实运行日志](../../outputs/evaluation_baseline_v2/agentic-rag-7a-smoke-cbc5f91/real-model-smoke.log)、
[安全配置与完整性快照](../../outputs/evaluation_baseline_v2/agentic-rag-7a-smoke-cbc5f91/smoke-context.json)。

索引 manifest SHA-256 前后均为
`148159a7077c5d5a340dcb35db5fa8b27f1575d788c57207923539c6ec0f3ead`；
smoke 期间记录的源码/数据集指纹也全部不变。

## Q052 OBSERVATION

额外执行 **1 次**：`OK`，两次 search 成功，tool/LLM=2/3，11.739s。
observed=`[375,377,376,378,379,445,466,456]`，references=`[375,376]`。
本次回答了学院制定考核形式/录取规则等可确认部分，并说明具体面试题和评分细则未统一提供。
没有调用 `get_chunk_text`，没有工具错误。

仅记录这一单次结果，不据此宣称 Phase 7 已观察到的 refusal weakness 被修复；
未修改 partial-answer policy，也未运行 Judge。
[q052 observation](../../outputs/evaluation_baseline_v2/agentic-rag-7a-smoke-cbc5f91/q052-run-1.json)。

## UNCHANGED

- retrieval algorithms
- Document identity（包括 build_evidence_id、chunk_to_evidence_document）
- runtime chunk assignment、ordering、index schema
- grounding
- ToolStrategy
- budgets
- compare_documents 生产逻辑、schema、文档匹配
- partial-answer policy、no_evidence、FinalAgentResponse
- SSE
- evaluation dataset

`get_chunk_text` 仍保留在 catalog 和 RAG_ONLY 三工具白名单中。
保护边界已与 HEAD 文件/函数核对，见
[unchanged-boundaries.json](../../outputs/evaluation_baseline_v2/agentic-rag-7a-smoke-cbc5f91/unchanged-boundaries.json)。

## FILES CHANGED

| 文件 | 变更 |
| --- | --- |
| app/tools/schemas.py | 正整数 locator 和参数说明；移除不再使用的 Annotated import |
| app/tools/knowledge_base.py | handler int 类型；不存在 locator 的标准参数错误与 typed error 透传 |
| app/tools/catalog.py | 仅 get_chunk_text 升级 1.1.0，消除 description 歧义 |
| app/prompts.py | evidence ID 规则附近新增三行 locator 说明 |
| tests/test_tool_schema.py | 更新参数值域测试 |
| tests/test_tool_registry.py | 版本、正整数解析/读取、无效参数拒绝、不存在 locator、共享 identity 回归 |
| tests/test_local_tool_catalog.py | 模型可见 schema、版本和 description 回归 |
| tests/test_agent_knowledge_tool_integration.py | Search ToolMessage 的两种 ID 确定性回归 |
| tests/test_compare_documents_tool.py | 跨工具 identity 测试改用合法正整数 fixture |
| tests/test_iterative_agent_loop.py | 合法数字 fixture；不存在 locator 的错误与恢复断言 |
| docs/architecture/current-system.md | 当前 locator 合同说明 |
| docs/architecture/agentic-rag-7a-result.md | 本报告 |

生产代码只修改上述 4 个 app 文件，新增 23 行、删除 11 行。
测试治理文件只运行，未修改；历史 Phase 1–7 报告及 docs/history 未修改。

## TESTS NOT RUN

- full pytest / Phase 6 的 268 tests
- 65-question RAG baseline / full RAG Evaluation
- Agent Core
- Judge
- Online
- MCP
- Combined

## ENGINEERING CHECKS

| 检查 | 结果 |
| --- | --- |
| `.venv/bin/lint-imports` | PASS，7 kept / 0 broken |
| `.venv/bin/python scripts/inspect_agent_runtime.py --check` | PASS |
| `.venv/bin/ruff check` 本次 10 个修改的 Python 文件 | PASS |
| `git diff --check` | PASS |
| schema grep | `GetChunkTextArgs.chunk_id: int`，`ge=1` |
| catalog / prompt grep | 明确 Document.id 与 metadata.chunk_id 的职责 |
| `rg -n "EvidenceLocator\|ChunkIdentityManager\|EvidenceIdRegistry" app` | 无匹配 |

## FINDINGS

- q052 refusal weakness remains pending complete baseline；本次单次 OK 不能关闭历史 finding。
- q049 variance remains unresolved；本阶段未复跑、未提高预算。
- compare_documents name matching errors remain measured efficiency findings；本阶段未修。
- 两次 q054 仍对刚检索到的 382/383/384 再读正文；定位正确、执行成功，重复读取效率留待后续分析。
- official v3 baseline still NOT established；本次 smoke 不替代 clean-commit 正式结果。

## NEXT PHASE

AGENTIC-RAG-7B should rerun the official clean-worktree Prediction v3 real-model baseline from a new output directory after this locator-contract fix is reviewed and committed, then complete Agent Core and selective Judge analysis before any quality tuning.
