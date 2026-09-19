# AGENTIC-RAG-8 RESULT

## STATUS

**PASS_WITH_FINDINGS**。已实现基于 7D 测量证据的最小 refusal-policy Prompt candidate，focused offline tests 通过。未进行真实模型验证，不宣称 q052、q054、q056、q061 已被实测修复。

## BASELINE

- branch: `refactor/tool-output-models`
- head: `4e5e95321bf744ce30ab4006c035a0dfa3ddf2c6`（本阶段前后不变）
- worktree_before: **clean**。
- worktree_after: 仅修改 `app/prompts.py`、`tests/test_create_agent_builder.py`，新增本报告。
- 未切换或新建分支，未 commit，未 push。

## MOTIVATION

[AGENTIC-RAG-7D accepted baseline](agentic-rag-7d-result.md)：65/65 predictions，runtime failures=0。以下事实来自已有 7D 报告，本阶段未重跑测量。

Measured dominant failure: **REFUSAL_POLICY**。

- q052、q054、q056：已经观察到可回答 evidence，仍整体 `NO_RELEVANT_INFO`，丢弃已确认内容。q054 同时存在局部检索缺口，不能据此忽略已有可回答部分。
- q061：适用对象未确认，仍把相近规定的具体金额、比例作为答案主体；附带适用性声明不足以使这些细节成为受支持答案。

## CHANGE

生产 Python 只修改 `app/prompts.py` 中的 `AGENT_SYSTEM_PROMPT`，保留原有十条规则及整体结构：

- 第 4 条明确先回答实质性已知部分，再限定具体未知部分；禁止因局部信息或适用范围不确定而整体拒答，禁止用模型内部知识补齐。
- 第 4 条补充适用对象、年份、学历层次、人员类别、政策版本及其他范围条件的一致性边界。
- 第 9 条将 `no_evidence` 限定为完全没有实质性支持，同时保留继续检索和禁止无依据作答的指导。
- 第 7 条的 `no_evidence` 描述、第 10 条的证据不足指导同步对齐上述边界，避免旧措辞导致冲突；第 8 条明确只引用支持已确认内容的 `Document.id`。

## PARTIAL-ANSWER CONTRACT

```text
partial evidence
→ outcome=answer
→ answer supported portion first
→ explicitly limit unsupported portion
→ evidence_ids contain only Document.id supporting confirmed content

no material evidence
→ outcome=no_evidence
→ evidence_ids=[]
```

未知部分必须明确说明无法根据当前 evidence 确认，不得使用模型内部知识补齐或编造引用。局部 scope 未确认，但其他实质性内容已确认时，仍使用 `outcome=answer` 回答已确认部分。

继续使用现有 `answer`、`no_evidence`、`clarify`，不新增 outcome 或结构化字段。

## SCOPE CONTRACT

```text
unconfirmed applicability
→ do not present specific numeric/rule details
  as facts for the requested population
```

无法由 evidence 确认规定的适用对象、年份、学历层次、人员类别、政策版本或其他范围条件与问题一致时，不得直接将金额、比例、期限、条件作为该问题的事实答案。可以说明找到相近规定，但必须明确当前 evidence 无法确认其适用于用户所问对象。

## UNCHANGED

- Retrieval、BM25、Embedding、Chroma、CrossEncoder、chunking、top_k
- Tool catalog、tool contracts、Document identity
- ToolStrategy
- FinalAgentResponse
- grounding（保留 fail-closed）
- budgets
- compare_documents 工具实现与合同
- get_chunk_text 工具实现与 `metadata.chunk_id` 定位合同
- Evaluation formulas、refusal classifier
- Evaluation dataset
- Query Rewrite
- ChatService、create_agent architecture
- Judge prompts

## TESTS

复用已有 Prompt 检查所在的 `tests/test_create_agent_builder.py`，新增 5 个确定性合同测试，覆盖 partial evidence、无实质性支持、未知部分、适用范围，以及旧引用/不可信正文约束。只检查关键语义片段，不锁定完整 Prompt snapshot；未新增 Fake Model 行为模拟。

实际运行：

```bash
.venv/bin/pytest -q tests/test_create_agent_builder.py tests/test_agent_results.py
```

结果：**106 passed, 1 warning**。warning 来自第三方 `jieba` 使用已弃用的 `pkg_resources`。已有离线框架与结果合同测试通过，不代表真实模型一定遵守 Prompt。

工程检查：

```bash
.venv/bin/ruff check app/prompts.py tests/test_create_agent_builder.py
git diff --check
```

均通过。

## TESTS NOT RUN

- real-model evaluation / real-model smoke
- 65-question baseline
- Agent Core
- LLM Judge / Selective Judge
- targeted reruns（含 q052、q054、q056、q061）
- full pytest / Evaluation suite
- Online
- MCP
- Mixed
- Combined
- lint-imports / runtime inspection（未修改 import architecture）

## FILES CHANGED

- `app/prompts.py`：仅加强 `AGENT_SYSTEM_PROMPT` 的 partial-answer / scope 边界。
- `tests/test_create_agent_builder.py`：在已有测试文件中新增 5 个 Prompt contract tests。
- `docs/architecture/agentic-rag-8-result.md`：本报告。

## FINDINGS

- 这是 prompt-policy candidate：implemented the smallest evidence-backed refusal-policy candidate。不宣称真实模型问题已被实测消除。
- 7D baseline 继续作为 accepted v3 baseline；本阶段没有新的 baseline、Judge 分数或效果提升结论。
- evaluator refusal classifier 不识别“我无法根据当前取得的证据确定答案。”的 `EVALUATION_CONTRACT` finding 保留，未修改 `REFUSAL_PATTERNS`、`STANDALONE_REFUSAL_PATTERNS`、`behavior_score` 或 `final_score`。

## PROJECT STATUS

Engineering implementation: **COMPLETE**。

Accepted baseline: **AGENTIC-RAG-7D**。

Optional future work: only if a concrete regression/quality need appears. 届时可单独开展 optional regression check，不是当前工程的未完成项。

当前 Agentic RAG 工程主线结束。

## NEXT

**NO REQUIRED NEXT PHASE.**

Do not start another phase automatically. **STOP**。
