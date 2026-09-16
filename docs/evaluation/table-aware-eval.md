# TABLE-AWARE-EVAL RESULT

本节保留修复前的评估记录；后续引用完整性修复及复验见文末
[RAG-CITATION-INTEGRITY-1 RESULT](#rag-citation-integrity-1-result)。

## STATUS

**PASS_WITH_FINDINGS** — 评估完成，真实模型与现有评估入口均可运行。
这表示评估结果可复核，不表示真实表格语料质量已经达标。

**结论：受控样本的检索、关系保持和上下文选择正常；两个英文 PDF 问题在回答后的引用校验阶段被误拒答。**
当前生产库没有 table chunks，因此不能把这些受控成绩解释为真实业务表格质量。

## BRANCH / HEAD / WORKTREE

- Branch：`refactor/tool-output-models`。
- HEAD：`84f53a1c3ab16cdc5ee9f825e96879d40f059c42`，包含 TABLE-AWARE-1D。
- 开始时 clean；结束时 dirty，仅新增本报告、独立题集、eval helper 和对应测试。
- 未切分支、reset、pull、rebase、commit 或 push。
- 日期：2026-09-15，Asia/Shanghai。

## EVALUATION SCOPE

正式执行独立题集 [rag_eval_table_v1.json](../../data/eval/rag_eval_table_v1.json)，共 **14 题：13 answer + 1 abstain**。

| 样本类型 | 题数 | 说明 |
| --- | ---: | --- |
| Real corpus case | 0 | 当前真实 Chroma 无 `element_type=table` |
| Controlled parser fixture | 9 | 真实 Docling 解析仓库既有 PDF、DOCX、HTML |
| Controlled structure fixture | 5 | 复用既有 TableBlock 测试数据；不计入解析成功证据 |
| Controlled 合计 | 14 | 单独隔离的 8 文档、17 chunks，无生产语料背景 |

来源覆盖：

| 来源 | 题数 | case |
| --- | ---: | --- |
| Local PDF | 4 | t001–t003、t013；`tests/fixtures/structured-table.pdf` |
| Local DOCX | 1 | t004；`build_table_docx(merged=False)` |
| Upload DOCX | 2 | t005–t006；`build_table_docx(merged=True)` |
| Firecrawl HTML adapter | 2 | t007–t008；原有普通 HTML、多层表头 HTML；没有网络抓取 |
| 直接持久化的结构 fixture | 5 | t009–t012、t014；隔离 upload namespace，不代表真实 Upload 文档 |

DOCX/HTML 的原始 fixture 见 [structured_document.py](../../tests/support/structured_document.py)
和 [test_structured_parsing.py](../../tests/test_structured_parsing.py)。
结构 fixture 见 [test_structured_chunking.py](../../tests/test_structured_chunking.py)。
所有题目在模型运行前固定；未根据回答改 gold，未合入正式 65 题 baseline。

### 当前真实库的实际情况

- `mini_rag_chunks_v2`：**473 chunks，1024 维**，manifest `pipeline_version=3`。
- Local：423 chunks，其中当前 PDF 为 10 个 text chunks；没有 Local table chunk。
- 已入库 Firecrawl：47 chunks，均无 `element_type=table`。
- 已入库 Upload DOCX：3 chunks，均无 `element_type=table`。
- SQLite：3 个 Firecrawl candidate（2 approved、1 rejected）、1 个 approved Upload candidate；**4 个均未持久化 structured_content**。
- 这是历史纯文本内容的覆盖限制，不能据此认定当前 Docling 解析失败；没有重新抓网页、重新解析历史候选或重建生产库。

证据：[real-corpus-audit.json](../../outputs/evaluation_baseline_v2/table-aware-84f53a1-diagnostics/real-corpus-audit.json)。

## ENVIRONMENT

| 参数 | 实际值 |
| --- | --- |
| Embedding | `Qwen/Qwen3-Embedding-0.6B`，真实本地缓存，CUDA |
| Reranker | `BAAI/bge-reranker-v2-m3`，真实 CrossEncoder，本地缓存，CUDA |
| Retrieval | 现有 Chroma + jieba/BM25 + 0.4 BM25 / 0.6 vector |
| top_k / decompose_top_k | **5 / 5**，按正式 baseline 口径；原 settings 的 decompose_top_k=10 未修改 |
| chunk_size / chunk_overlap | **600 / 120** |
| max_context_chars | **4000** |
| pipeline_version | **3** |
| Answer / Decompose LLM | **真实 `qwen-plus` / `qwen-plus`** |
| LLM Judge | **关闭** |
| 模型下载 / 更换 / 调参 | 无 |

正式 14 题中 5 题调用了 Decompose LLM。另仅对 t001、t003 各补采一次失败链路，
两次补采单独保存，没有替换正式 predictions。
Embedding、Reranker、Docling 保持离线缓存模式；Answer/Decompose 使用当前配置的模型服务。

证据：[run-environment.json](../../outputs/evaluation_baseline_v2/table-aware-84f53a1-diagnostics/run-environment.json)、
[正式 manifest](../../outputs/evaluation_baseline_v2/table-aware-84f53a1/manifest.json)。

## TABLE CHUNKS

- 当前受控索引：**9 个 table chunks，来自 8 张表**；另有 8 个 text chunks。
- Small / single-chunk table：**7 张**。
- Multi-chunk table：**1 张、2 chunks**，来自已有 oversized-row 结构 fixture。
- Rowspan / colspan：**2 张真实解析 fixture 表**（merged DOCX、多层 HTML）。
- `oversized-structure`：前块 row 1，755 字符；后块 row 2，64 字符。t011/t012 分别验证这两块。
- `large-structure` 的原 20 行数据在固定参数下只有 **554 字符、1 chunk**；t014 只验证末行，不冒充跨块测试。
- **真实 corpus 的多块大表、跨前中后三块、单行超过 4000 字符：未覆盖。** 未缩小 chunk/context 参数，也未增造更长数据补齐指标。

证据：[fixture-chunks.json](../../outputs/evaluation_baseline_v2/table-aware-84f53a1-diagnostics/fixture-chunks.json)。

## METRIC CONTRACT

复用现有 `resolve_gold_chunk_ids`、`score_retrieval`、`score_evidence_chunks` 与正式报告写入逻辑。
题集沿用正式 RAG 字段，gold 来自原表 cells/短事实。ID 仅保存当前快照；真实评分重新通过 evidence 对齐。

新增 [table_diagnostics.py](../../scripts/evaluation/table_diagnostics.py) 只负责：

1. 调用 `pipeline.retrieve()`、`pipeline.rerank()`、现有 evidence gate 和 `build_context_with_selection()`。
2. 为固定 K=1/3/5 调用现有 scorer；不实现第二套检索或 Recall 公式。
3. 检查同一 gold 来源的同一个 `table` chunk 是否同时包含全部 gold_key_points。
4. 保留每题 Top5、分数、来源、table_id、行范围、完整 gold chunk 与各阶段命中。

拒答题 t013 不进入 retrieval/relation/context/reference 的正例分母。
现有 scorer 对空 gold 有默认成功语义，因此原生 overall 的 14 题均值不能直接当表格正例指标；下表统一使用 **13 题**。
当前每题只有一个重新定位的 gold chunk，故 Recall@K 与 Hit@K 相同。现有 scorer 无 MRR，未另造指标。

## RETRIEVAL METRICS

这是调用 Answer/Decompose LLM **之前**的单查询诊断。

| 指标 | 命中 / 正例数 | 比例 |
| --- | ---: | ---: |
| Hit@1 | 5 / 13 | 38.46% |
| Hit@3 | 11 / 13 | 84.62% |
| Hit@5 | 13 / 13 | 100% |
| Recall@5 | 13 / 13 | 100% |

## RERANK METRICS

| 指标 | 命中 / 正例数 | 比例 |
| --- | ---: | ---: |
| Gold Top1 | 9 / 13 | 69.23% |
| Gold Top3 | 13 / 13 | 100% |
| Gold Top5 | 13 / 13 | 100% |

按 fixture 类型分组，避免把结构构造当解析证明：

| 分组（仅 answer） | 题数 | Retrieve @1 / @3 / @5 | Rerank @1 / @3 / @5 |
| --- | ---: | --- | --- |
| Parsed fixtures | 8 | 3 / 7 / 8 | 5 / 8 / 8 |
| Structure fixtures | 5 | 2 / 4 / 5 | 4 / 5 / 5 |

生产 retrieve 的候选上限为 `top_k * 5`；本次仅 17 chunks，实际候选为整个小库。
这些数值不能外推到大规模、有大量相似表格的真实语料。

完整 RAG 保留现有分解流程，因此 final Top1/3/5 是 **10/13、12/13、13/13**。
t006 的 gold 在完整 RAG final rank 4，仍进入 context；这与前面的单查询 Rerank 指标分开报告。

证据：[逐题 retrieval.jsonl](../../outputs/evaluation_baseline_v2/table-aware-84f53a1-diagnostics/retrieval.jsonl)、
[retrieval-summary.json](../../outputs/evaluation_baseline_v2/table-aware-84f53a1-diagnostics/retrieval-summary.json)。

## RELATION INTEGRITY

**13 / 13 passed**；失败 qid：无。
这里的通过含义是：同一正确来源、同一个 table chunk 同时承载全部关键事实，并进入 context。
不接受从不同 chunks 拼凑关键词；它不等同于证明模型最终回答正确。

## CONTEXT INCLUSION

- 单查询诊断：正确 evidence 被 retrieve 后，**13 / 13** 进入 context。
- 以 retrieve Top5 为分母也是 **13 / 13**。
- 完整 RAG：**13 / 13** 正确关系进入实际预测 trace 的 context。
- 因 `max_context_chars` 丢失 gold 的 case：**0**。
- 实际受控表格最长 755 字符，小于 4000；这不能排除已知超长行问题。

## ANSWER / REFERENCE

正式入口完成 **14 / 14 predictions**，没有 MODEL_ERROR、timeout 或 connection failure。

| 检查 | 结果 |
| --- | --- |
| 可回答题：最终答案包含所需事实并有依据（逐题人工检查） | **11 / 13** |
| 正确 evidence chunk 最终进入 references | **13 / 13** |
| Not-in-table abstain | **1 / 1**，t013 |
| 意外拒答 | **2 / 13**，t001、t003 |

t005/t006 将合并单元格的“身份证\n证明”压成“身份证证明”，两个词均保留，但分隔表达不够清楚。
t013 的费用在原 PDF 中确实不存在；本次拒答成功不能证明任意真实表格上的拒答能力。

Judge 关闭时，原生报告的 `key_point_recall`、`forbidden_rate` 和 `avg_final_score=0.5071`
不能当完整语义质量评分。本文没有据此设 PASS 门槛，答案结论来自人工核对。
Reference Support Rate 只证明证据进入引用；t001/t003 正好说明“有正确引用”仍可能被后续校验拒答。

证据：[正式 predictions.csv](../../outputs/evaluation_baseline_v2/table-aware-84f53a1/rag/predictions.csv)、
[case-review.jsonl](../../outputs/evaluation_baseline_v2/table-aware-84f53a1-diagnostics/case-review.jsonl)、
[full-rag-summary.json](../../outputs/evaluation_baseline_v2/table-aware-84f53a1-diagnostics/full-rag-summary.json)。

## FAILURE TAXONOMY

| 主要类别 | case 数 | qid |
| --- | ---: | --- |
| PARSE | 0 | — |
| CHUNK | 0 | — |
| RETRIEVE | 0 | — |
| RERANK | 0 | — |
| CONTEXT_BUDGET | 0 | — |
| GENERATION | **2** | **t001、t003** |
| EVAL_DATA | 0 | — |
| INFRA | 0 | — |

分类按最终失败的主要原因计数，Top1 未命中本身不是整题失败。
`GENERATION` 在本次 taxonomy 中包含答案生成后的引用验证；**不能把这两个 case 说成 LLM 不知道答案**。
真实 table corpus 缺失是整个评估的覆盖 Finding，不伪造对应失败 qid。

## REPRESENTATIVE FAILURES

### t001 — Registration 的材料

- Question：`structured-table.pdf 中，Registration 需要什么材料？`
- Classification：**GENERATION / citation verification**。
- Gold evidence：`Registration Identity card`。
- Retrieve Top5（chunk_id: hybrid）：`4:0.9742, 5:0.8924, 7:0.6641, 2:0.6222, 12:0.6122`。
- Rerank Top5（chunk_id: rerank）：`5:0.7338, 2:0.6295, 12:0.3823, 9:0.3297, 7:0.2072`。
- Gold：chunk 5；`source_type=local_file`、`file_name=structured-table.pdf`、`element_type=table`、`table_id=b1`、`row_start=1`、`row_end=2`、`page_number=1`。
- Gold text：`Item：Registration；Condition：Age 18+；Material：Identity card。`
- Context included：**yes**；Reference included：**yes**；正式最终答案为拒答。

### t003 — Registration 的条件和材料

- Question：`structured-table.pdf 中，Registration 的条件和材料分别是什么？`
- Classification：**GENERATION / citation verification**。
- Gold evidence：`Registration Age 18+ Identity card`。
- Retrieve Top5（chunk_id: hybrid）：`4:0.9413, 5:0.8594, 12:0.6482, 7:0.6267, 2:0.6031`。
- Rerank Top5（chunk_id: rerank）：`5:0.8908, 2:0.7846, 12:0.6104, 9:0.4729, 16:0.3650`。
- 正确 chunk、metadata、事实与 t001 相同；Context included：**yes**；Reference included：**yes**。
- 正式最终答案仍为拒答，不能归因于未检索到、跨块断裂或 context budget。

### Root cause 与独立补采证据

正式 prediction schema 不保存原始答案与 citation_verification reason。
因此只对上述两题各补采一次，在 eval-only 内观察并原样调用生产 verifier；不改变返回值。
**原正式结果与补采结果分开保存，补采不能冒充原运行的逐 token 记录。** 两次补采均再次得到相同拒答机制：

| qid | 补采模型原始答案中的正确事实 | 真实引用校验 overlap | supported |
| --- | --- | ---: | --- |
| t001 | `Identity card`，并正确给出 `Age 18+` | 0.0435 | false |
| t003 | `Age 18+`、`Identity card` | 0.0556 | false |

[make_snippet](../../app/rag/retrieval.py) 使用 `"".join(text.split())` 删除全部空白。
[format_reference](../../app/rag/generation.py) 将这种摘要用于引用。
于是 evidence 中的 `Age 18+` / `Identity card` 变成 `Age18+` / `Identitycard`；
[verify_answer_citations](../../app/rag/citations.py) 的英文分词结果与答案不一致，两题仅匹配到 `registration`。
生产 graph 使用 overlap 门槛 0.1，随后以 NO_RELEVANT_INFO 替换正确答案。

在纯确定性 eval-only 对照中，保持同一补采答案、同一表格事实、同一 verifier 和门槛，
只让引用摘要保留词间空格，t001/t003 的 overlap 分别变为 **0.2174 / 0.2778**，均 supported=true。
**这个对照只证明已定位机制，不计入正式成绩，也没有改生产实现。**

完整证据（含来源路径、Top5 metadata、gold 全文）：
[table-failures.jsonl](../../outputs/evaluation_baseline_v2/table-aware-84f53a1-diagnostics/table-failures.jsonl)、
[failure-replay.json](../../outputs/evaluation_baseline_v2/table-aware-84f53a1-diagnostics/failure-replay.json)。

## PRODUCTION CODE CHANGES / INDEX SAFETY

**production functional changes = 0**。
生产 app/、Prompt、模型配置、阈值和正式 65 题集均未改。
没有调用生产 `delete_collection()`、force rebuild、重新 embedding 或外部 fetch。

受控 Chroma、RecordManager、SQLite、data directory 均位于
`outputs/evaluation_baseline_v2/table-aware-84f53a1-diagnostics/isolated/`。
正式入口的 Local 同步仅作用于该隔离目录：**added=0、updated=0、skipped=6、deleted=0**。

物理文件审计存在一项需要明确的限制：79 个保护文件中 77 个 SHA-256 不变；
生产 `data/chroma/chroma.sqlite3` 与一个 HNSW `length.bin` 在读取审计后字节发生变化。
嵌入式 Chroma 的读取/初始化不能视为文件系统层面的绝对只读；本次没有保存可逐字节比较的原 DB 副本，
不能进一步断言内部具体改了什么，也没有回写或恢复这些文件。
随后用 SQLite `mode=ro` 核对：v2 仍为 **473 条**，旧 collection 仍为 423 条；
最新 vector operation 时间为 **2026-09-15 09:01:08 UTC**，早于本次评估。
RecordManager、生产 app.sqlite、manifest 和 65 题集的保护哈希均不变。

证据：[protected-after.json](../../outputs/evaluation_baseline_v2/table-aware-84f53a1-diagnostics/protected-after.json)、
[production-readonly-audit.json](../../outputs/evaluation_baseline_v2/table-aware-84f53a1-diagnostics/production-readonly-audit.json)。

## FILES CHANGED

| 新增文件 | 用途 |
| --- | --- |
| `data/eval/rag_eval_table_v1.json` | 独立 14 题；沿用现有 schema，标明 controlled 来源与 gold |
| `scripts/evaluation/table_diagnostics.py` | 薄诊断胶水与同 chunk 关系检查；复用现有评分 |
| `tests/test_evaluation_table_diagnostics.py` | 数据合同、跨块/跨来源误命中、ID 重定位、abstain 分母防护 |
| `docs/evaluation/table-aware-eval.md` | 本报告 |

本地 `outputs/` 按仓库既有规则忽略；保存正式产物、逐题诊断、隔离存储及一次性准备/补采脚本。
这些本地产物未提交，移动报告到其他机器时需一并归档相关输出。

## TESTS / COMMANDS

实际执行并通过：

- `.venv/bin/python scripts/run_evaluation_baseline.py --help`。
- `.venv/bin/python -m pytest -q tests/test_evaluation_table_diagnostics.py tests/test_rag_eval_retrieval.py`：**7 passed**，1 个既有 jieba/pkg_resources 弃用警告。
- `.venv/bin/lint-imports`：**7 kept, 0 broken**。
- `.venv/bin/ruff check scripts/evaluation/table_diagnostics.py tests/test_evaluation_table_diagnostics.py`：通过。
- `git diff --check`：通过。

现有 `--validate-only` 只验证 Agent dataset，不能验证独立 RAG file；未把它作为 table 合同证据。
本次使用现有 RAG loader、上述 focused test 和当前隔离 chunks 的 gold realignment 校验。

运行命令使用以下**进程级环境**，没有改 `.env`（`$PWD` 为仓库根目录）：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ANONYMIZED_TELEMETRY=False
export RAG_FILE_PATH="$PWD/outputs/evaluation_baseline_v2/table-aware-84f53a1-diagnostics/isolated/kb"
export INDEX_STORE_DIR="$PWD/outputs/evaluation_baseline_v2/table-aware-84f53a1-diagnostics/isolated/manifest"
export CHROMA_PERSIST_DIR="$PWD/outputs/evaluation_baseline_v2/table-aware-84f53a1-diagnostics/isolated/chroma"
export SQLITE_DB_PATH="$PWD/outputs/evaluation_baseline_v2/table-aware-84f53a1-diagnostics/isolated/app.sqlite"
```

本次实际以命令前环境赋值运行，等价于上述环境。一次性 preparation 调用现有 parser、admission、SQLite、LangChain indexing；
首次准备后又增量加入原有 oversized-row fixture，新增 2 chunks。
[prepare_fixtures.py](../../outputs/evaluation_baseline_v2/table-aware-84f53a1-diagnostics/prepare_fixtures.py)
保留了完整准备步骤，拒绝覆盖已存在的隔离 Chroma；重跑应使用全新输出目录。

```bash
PYTHONPATH=. .venv/bin/python -u outputs/evaluation_baseline_v2/table-aware-84f53a1-diagnostics/prepare_fixtures.py
PYTHONPATH=. .venv/bin/python -u outputs/evaluation_baseline_v2/table-aware-84f53a1-diagnostics/run_diagnostics.py

.venv/bin/python -u scripts/run_evaluation_baseline.py \
  --rag-only \
  --rag-eval-file data/eval/rag_eval_table_v1.json \
  --output-dir outputs/evaluation_baseline_v2/table-aware-84f53a1 \
  --top-k 5 --decompose-top-k 5 --allow-dirty

PYTHONPATH=. .venv/bin/python -u outputs/evaluation_baseline_v2/table-aware-84f53a1-diagnostics/inspect_failures.py
```

`--allow-dirty` 用于未提交的 focused evaluation；不是正式 65 题基线的新版本。
完整入口 exit=0；正式结果见 [baseline_report.md](../../outputs/evaluation_baseline_v2/table-aware-84f53a1/baseline_report.md)。
NOT_RUN_BY_DESIGN：全量 pytest、65 题 baseline、Core Agent、Combined、MCP、Online、Mixed、SQL、SSE、A/B、LLM Judge。

## FINDINGS

1. **真实 corpus 没有结构表格**：v3/可读并不代表已有 table-aware 内容；当前结论仅适用于受控样本。
2. **两题失败发生在回答后的引用验证**：正确事实已进入 context，补采模型答案也正确；英语词间空格在 snippet 中丢失，导致错误拒答。
3. **没有当前证据支持增加检索架构**：受控 Hit@5、Rerank Top3、Relation、Context 均为 13/13；小库和重复 fixture 限制了外推价值。
4. **4000 字符 context 风险未覆盖**：现有 oversized fixture 只超过 chunk_size，没有超过 max_context_chars；不能宣布该风险已解除。
5. **生产存储物理哈希变化已披露**：没有本次增删改向量操作证据，但不能称 Chroma 文件逐字节未变。以后若要求物理零写，先建立一致快照，再在副本上打开嵌入式客户端。

## DECISION

**E. NEXT_GENERATION**。

具体指已复现的**引用摘要与引用校验**边界，不是调 Prompt、替换 LLM 或扩充表格检索架构。
真实业务表格是否“已经足够好”仍缺乏样本，不能从本次控制集宣告结论 A。

## NEXT

下一阶段只针对这两条失败证据，最小化检查引用摘要保留英文词边界的方式，并保留现有引用验证的治理作用。
用 t001/t003 及必要的引用校验回归验证，不改 parser、chunker、retrieval、reranker、context 参数，也不降低验证门槛。
真实表格质量结论需另有已正常入库的结构化业务样本。本次到评估与 Finding 为止，**未实施修复**。

---

# RAG-CITATION-INTEGRITY-1 RESULT

## STATUS

**PASS** — 确定性回归、固定原始答案重放及 t001/t003 真实模型复跑均通过。
生产改动仅为 snippet 空白规范化和 graph node 内的验证证据来源。

## BRANCH / HEAD

- Branch：`refactor/tool-output-models`。
- HEAD：`d42f301abb0eb6585fed530972edddb27228e0cf`，修复保留为未提交工作区改动。
- 开始时工作区 clean；没有切分支、reset、rebase、commit 或 push。

## ROOT CAUSE

修复前链路：full chunk → `format_reference()` → `make_snippet()` 删除空白并截断到
300 字符 → verifier 使用展示 snippet → 正确答案可能被误拒答。
其中 `Identity card` 被变成 `Identitycard`，破坏了原有英文关键词匹配；
即使只修空白，300 字符之后的证据仍然不可见。

## FIX

1. `make_snippet()` 用 `" ".join((text or "").split())` 合并连续空白，随后执行原有
   `[:max_len]` 截断；`Identity card` 保留词边界，`身份证\n申请表` 展示为 `身份证 申请表`。
2. `bind_and_verify_citations_node()` 先调用原有 `format_reference()` 完成绑定，
   再按成功绑定的 `reference_id` 从 `final_chunks` 取完整 `text`，只将内部临时
   `{"text": ...}` 列表交给原 verifier。
3. 公开 references 仍只有原有八个字段，没有新增 `text`/`content`；临时证据不写入
   graph state 或公开 references。既有内部 RAG result 的 `final_chunks` 合同不变。

graph 门槛仍为 **0.1**，verifier 默认门槛仍为 **0.2**。
没有修改关键词算法、refusal/Evidence Gate 语义、graph topology/state schema、Prompt、模型配置、
parser、chunking、检索、reranker、pipeline version 或依赖。

## SECURITY / BINDING

- 只有 `Reference 999` 或空引用：非 refusal 答案仍失败，返回 `NO_RELEVANT_INFO`。
- 正确证据在未引用的 Reference 1，而模型仅引用无关的 Reference 2：仍失败。
- 无关 Reference 2 与无效 Reference 999 混合：仍失败；无效 ID 不贡献证据。
- 有效 Reference 1 与无效 Reference 999 混合：仅 Reference 1 绑定并提供证据，沿用原有过滤语义。
- 引用按 ID 绑定；测试以乱序、不连续的 Reference 2 / Reference 7 防止按列表位置误配。
- Refusal 仍跳过支撑验证，最终保留原有拒答状态和答案。

## FILES CHANGED

| 文件 | 修改 |
| --- | --- |
| `app/rag/retrieval.py` | snippet 合并空白、保留词边界 |
| `app/rag/graph/nodes.py` | 仅将已绑定 chunk 全文传给 verifier |
| `tests/test_citation_verifier.py` | 空白规范化与 max_len 回归 |
| `tests/test_rag_graph_runtime.py` | 全文证据、截断解耦、绑定边界、公开字段和 refusal 回归 |
| `docs/evaluation/table-aware-eval.md` | 保留历史报告并补充本次结果 |

`app/rag/generation.py`、`app/rag/citations.py` 和 eval gold 未修改。

## TESTS

实际执行：

- 修复前运行新增 focused cases：**7 failed、10 passed、36 deselected**，复现词边界和截断误拒答。
  命令为 `.venv/bin/python -m pytest tests/test_citation_verifier.py tests/test_rag_graph_runtime.py -q
  -k 'make_snippet or full_bound_chunk or successfully_bound_references or refusal_keeps_existing'`。
- `.venv/bin/python -m pytest tests/test_citation_verifier.py -q`：**13 passed**，包含既有中文用例。
- `.venv/bin/python -m pytest tests/test_rag_graph_runtime.py tests/test_rag_graph_builder.py tests/test_rag_generation_confidence.py -q`：**48 passed**。
- `.venv/bin/ruff check app/rag/retrieval.py app/rag/graph/nodes.py tests/test_citation_verifier.py tests/test_rag_graph_runtime.py`：通过。
- `.venv/bin/lint-imports`：**7 kept、0 broken**。
- `git diff --check`：通过。

新增测试中，正确英文证据放在 500 字符前缀之后；模型确实在 context 看到并引用该 chunk，
公开 snippet 不含 `Identity card`，实际 graph/verifier 仍返回 `supported=True` 和 `OK`。
这证明验证已与 display truncation 解耦，而不只是修复了英文空格。

未跑全量 pytest、完整 14 题重测、65 题 baseline、Core Agent、MCP、Online、Mixed、SQL、SSE 或 LLM Judge。
pytest 仅出现既有 jieba/pkg_resources 弃用警告。

## TABLE EVAL REGRESSION

固定题集仍为 `data/eval/rag_eval_table_v1.json`；仅选择 **t001、t003**。
before 使用上一阶段保存的独立补采记录，不冒充正式 predictions 的原始 token 记录。

| qid | before：supported / overlap / status | 固定原始答案重放 after | 真实模型复跑 after |
| --- | --- | --- | --- |
| t001 | false / 0.0435 / NO_RELEVANT_INFO | true / 0.2174 / OK | true / 0.2174 / OK |
| t003 | false / 0.0556 / NO_RELEVANT_INFO | true / 0.2778 / OK | true / 0.2778 / OK |

**确定性验证**：保留上一阶段同一原始答案、成功绑定 ID 和 final chunks，
直接重放生产绑定节点及成功结果节点，没有调用模型。两题都恢复正确答案。

**真实模型复跑**：复用原有只选择 t001/t003 的 `inspect_failures.py` 逻辑，
仅调整输出位置并记录不含凭据的运行参数；observer 原样调用生产 verifier，不改变返回值。
两题 raw answer 均包含 `Identity card`，t003 同时包含 `Age 18+`，且最终答案与 raw answer 一致。
Answer/Decompose 为 `qwen-plus`；Embedding 为 `Qwen/Qwen3-Embedding-0.6B`，
Reranker 为 `BAAI/bge-reranker-v2-m3`，使用本地 CUDA 缓存；top_k/decompose_top_k=5/5，
max_context_chars=4000，Judge 关闭。没有用两题结果推算完整 13/13 成绩。

产物单独保存在 [citation-integrity-d42f301](../../outputs/evaluation_baseline_v2/citation-integrity-d42f301/)：

- [deterministic-replay.json](../../outputs/evaluation_baseline_v2/citation-integrity-d42f301/deterministic-replay.json)：固定答案重放。
- [live-replay.json](../../outputs/evaluation_baseline_v2/citation-integrity-d42f301/live-replay.json)：本次真实模型原始答案、完整内部验证输入及结果。
- [run-environment.json](../../outputs/evaluation_baseline_v2/citation-integrity-d42f301/run-environment.json)：模型和运行参数。
- [protected-after.json](../../outputs/evaluation_baseline_v2/citation-integrity-d42f301/protected-after.json)：**91/91 保护文件 SHA-256 不变**。

只打开上一阶段 17 chunks 受控索引的文件副本，未触发 rebuild、重新解析或语料 embedding。
生产 Chroma、RecordManager、SQLite、manifest、`.env`、表格题集和正式 RAG 题集，
以及上一阶段隔离索引文件均在保护哈希内；没有覆盖旧补采结果。
这些输出按仓库既有规则忽略，不属于提交内容，跨机器复核需一并归档。

复现命令（从仓库根目录执行，`isolated/` 已由原评估目录复制）：

```bash
PYTHONPATH=. .venv/bin/python outputs/evaluation_baseline_v2/citation-integrity-d42f301/replay_fixed_answers.py

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ANONYMIZED_TELEMETRY=False \
RAG_FILE_PATH="$PWD/outputs/evaluation_baseline_v2/citation-integrity-d42f301/isolated/kb" \
INDEX_STORE_DIR="$PWD/outputs/evaluation_baseline_v2/citation-integrity-d42f301/isolated/manifest" \
CHROMA_PERSIST_DIR="$PWD/outputs/evaluation_baseline_v2/citation-integrity-d42f301/isolated/chroma" \
SQLITE_DB_PATH="$PWD/outputs/evaluation_baseline_v2/citation-integrity-d42f301/isolated/app.sqlite" \
PYTHONPATH=. .venv/bin/python -u outputs/evaluation_baseline_v2/citation-integrity-d42f301/inspect_citations.py
```

## PRODUCTION LOC

以 `git diff --numstat -- app/` 统计，包含注释/docstring：
**added=11、deleted=3、net=+8**。

## FINDINGS

本次已验证的 citation false-negative 消失。现有 verifier 仍是关键词重合启发式，
不代表逐项事实推理验证；本次没有改变该算法的能力边界。
真实模型结果只覆盖这两道受控题，原报告关于真实业务表格覆盖不足的限制仍成立。

## NEXT

**STOP_TABLE_AWARE_EXPANSION**。

当前没有证据支持继续修改 parser / chunking / retrieval / reranker。
本次到引用完整性修复及复验结束，不启动下一阶段。
