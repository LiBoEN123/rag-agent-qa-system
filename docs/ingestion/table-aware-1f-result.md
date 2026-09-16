# TABLE-AWARE-1F RESULT

## STATUS

PASS。先完成真实诊断再修改代码；真实页面的 direct chunks 与当前实际 Chroma 均已通过一致性检查。

## BRANCH

`refactor/tool-output-models`。初始工作区干净；未切分支、reset、rebase、commit 或 push。

## HEAD

`b91d02c558e46795493384ce884e73acd15a516c`，开始和结束相同。

## REAL URL

https://zsb.imut.edu.cn/info/1034/1741.htm

2026-09-16 两次实际调用 `ExternalKnowledgeSourceService.propose_firecrawl_page()`，经项目受治理 MCP、FirecrawlPageSourceAdapter、parse_html_document 和 Docling。候选在诊断进程内显式 approve，用于直接投影；没有写入生产候选审批。

## ROOT CAUSE CLASSIFICATION

**C. HEADER_FLAG_MISSING**，没有发现 secondary lifecycle 问题。

- `source_id`: `src_dc2c7f689a2a81654b6a2b93a2a1af35a879d839346168ae30451e468cf22743`
- `candidate_id`: `cand_97fafb5488003a54123704104ec49254a43790e8d032dd62220c82891de68e26`
- `source_type=firecrawl`，`structured_content is not None`。
- `structure_hash`: `fb6d1b6e13918c38dbe61ced516b2b14560b7f1f7fd0e914ef189fc22210dd2e`
- 两次实际抓取的 candidate_id、structure_hash 相同，保持既有 dedupe 语义。
- 真实表头签名在 b9 内重复 **35 次**，但所有表格 cell 的 `is_header=False`。
- 修改前直接切分与实际 Chroma 的 **45 个 chunks 完全一致**，两边都没有 section。故不属于 CASE D；结构和表格关系保留，亦不属于 A/B。

## FIRECRAWL OBSERVATION

两次观察一致；仅记录长度和状态：

| 字段 | 结果 |
| --- | --- |
| markdown | 存在，22,014 字符 |
| html | 存在，415,536 字符 |
| mime_type | text/html |
| truncated | false |
| status code | 200 |

未复制完整网页 HTML。

## DOCLING OBSERVATION

目标 `TableBlock.block_id=b9`，`row_count=610`，`column_count=5`，header cell 总数 **0**。行号为 0-based。

| 行 | 角色 | 文本 | column_index / column_span | row_span | is_header |
| --- | --- | --- | --- | --- | --- |
| 0 | global | 内蒙古工业大学2026年分省录取分数统计 | 0 / 5 | 1 | false |
| 1 | section A | 内蒙古本科提前B段艺术类 | 0 / 5 | 1 | false |
| 2 | first header | 专业名称 / 最高分 / 最低分 / 平均分 / 备注 | 0..4，每个 span=1 | 每个 1 | 全 false |
| 6 | section B | 内蒙古本科批物理类 | 0 / 5 | 1 | false |
| 7 | repeated header | 专业名称 / 最高分 / 最低分 / 平均分 / 备注 | 0..4，每个 span=1 | 每个 1 | 全 false |
| 8 | data | 安全工程 / 527 / 501 / 508.32 / 物理类 | 0..4，每个 span=1 | 每个 1 | 全 false |
| 9 | data | 材料成型及控制工程 / 549 / 509 / 517.98 / 物理类 | 0..4，每个 span=1 | 每个 1 | 全 false |
| 10 | data | 材料成型及控制工程 / 503 / 480 / 487.75 / 民族班 | 0..4，每个 span=1 | 每个 1 | 全 false |

新 fixture 冻结原表前 11 行（含 row 5 总计行），全部 anchor cell 与实抓结果逐字段相等；仅 block_id/order 归零。未用 `<th>` 人工补 flag。

<details>
<summary>实抓前 25 行：113 个 unique anchor cells</summary>

每项为 `[row_index, column_index, row_span, column_span, is_header, text]`；每一行 JSON 对应一个原始表格行。

```jsonl
[[0, 0, 1, 5, false, "内蒙古工业大学2026年分省录取分数统计"]]
[[1, 0, 1, 5, false, "内蒙古本科提前B段艺术类"]]
[[2, 0, 1, 1, false, "专业名称"], [2, 1, 1, 1, false, "最高分"], [2, 2, 1, 1, false, "最低分"], [2, 3, 1, 1, false, "平均分"], [2, 4, 1, 1, false, "备注"]]
[[3, 0, 1, 1, false, "服装与服饰设计"], [3, 1, 1, 1, false, "501"], [3, 2, 1, 1, false, "465"], [3, 3, 1, 1, false, "470"], [3, 4, 1, 1, false, "投档成绩"]]
[[4, 0, 1, 1, false, "环境设计"], [4, 1, 1, 1, false, "487"], [4, 2, 1, 1, false, "475"], [4, 3, 1, 1, false, "477.87"], [4, 4, 1, 1, false, "投档成绩"]]
[[5, 0, 1, 1, false, "总计"], [5, 1, 1, 1, false, "501"], [5, 2, 1, 1, false, "465"], [5, 3, 1, 1, false, "475.4"], [5, 4, 1, 1, false, "投档成绩"]]
[[6, 0, 1, 5, false, "内蒙古本科批物理类"]]
[[7, 0, 1, 1, false, "专业名称"], [7, 1, 1, 1, false, "最高分"], [7, 2, 1, 1, false, "最低分"], [7, 3, 1, 1, false, "平均分"], [7, 4, 1, 1, false, "备注"]]
[[8, 0, 1, 1, false, "安全工程"], [8, 1, 1, 1, false, "527"], [8, 2, 1, 1, false, "501"], [8, 3, 1, 1, false, "508.32"], [8, 4, 1, 1, false, "物理类"]]
[[9, 0, 1, 1, false, "材料成型及控制工程"], [9, 1, 1, 1, false, "549"], [9, 2, 1, 1, false, "509"], [9, 3, 1, 1, false, "517.98"], [9, 4, 1, 1, false, "物理类"]]
[[10, 0, 1, 1, false, "材料成型及控制工程"], [10, 1, 1, 1, false, "503"], [10, 2, 1, 1, false, "480"], [10, 3, 1, 1, false, "487.75"], [10, 4, 1, 1, false, "民族班"]]
[[11, 0, 1, 1, false, "材料物理"], [11, 1, 1, 1, false, "538"], [11, 2, 1, 1, false, "499"], [11, 3, 1, 1, false, "505.04"], [11, 4, 1, 1, false, "物理类"]]
[[12, 0, 1, 1, false, "财务管理"], [12, 1, 1, 1, false, "455"], [12, 2, 1, 1, false, "452"], [12, 3, 1, 1, false, "453.25"], [12, 4, 1, 1, false, "民族班"]]
[[13, 0, 1, 1, false, "采矿工程"], [13, 1, 1, 1, false, "533"], [13, 2, 1, 1, false, "466"], [13, 3, 1, 1, false, "482.76"], [13, 4, 1, 1, false, "物理类"]]
[[14, 0, 1, 1, false, "测控技术与仪器"], [14, 1, 1, 1, false, "565"], [14, 2, 1, 1, false, "532"], [14, 3, 1, 1, false, "536.67"], [14, 4, 1, 1, false, "物理类"]]
[[15, 0, 1, 1, false, "车辆工程"], [15, 1, 1, 1, false, "553"], [15, 2, 1, 1, false, "536"], [15, 3, 1, 1, false, "538.84"], [15, 4, 1, 1, false, "物理类"]]
[[16, 0, 1, 1, false, "城乡规划"], [16, 1, 1, 1, false, "516"], [16, 2, 1, 1, false, "516"], [16, 3, 1, 1, false, "516"], [16, 4, 1, 1, false, "物理类"]]
[[17, 0, 1, 1, false, "储能科学与工程"], [17, 1, 1, 1, false, "566"], [17, 2, 1, 1, false, "555"], [17, 3, 1, 1, false, "556.69"], [17, 4, 1, 1, false, "新能源学院"]]
[[18, 0, 1, 1, false, "储能科学与工程"], [18, 1, 1, 1, false, "537"], [18, 2, 1, 1, false, "529"], [18, 3, 1, 1, false, "533.4"], [18, 4, 1, 1, false, "兴边固边（新能源学院）"]]
[[19, 0, 1, 1, false, "储能科学与工程"], [19, 1, 1, 1, false, "561"], [19, 2, 1, 1, false, "544"], [19, 3, 1, 1, false, "546.38"], [19, 4, 1, 1, false, "物理类"]]
[[20, 0, 1, 1, false, "道路桥梁与渡河工程"], [20, 1, 1, 1, false, "516"], [20, 2, 1, 1, false, "470"], [20, 3, 1, 1, false, "478.92"], [20, 4, 1, 1, false, "物理类"]]
[[21, 0, 1, 1, false, "地质工程"], [21, 1, 1, 1, false, "506"], [21, 2, 1, 1, false, "464"], [21, 3, 1, 1, false, "475.34"], [21, 4, 1, 1, false, "物理类"]]
[[22, 0, 1, 1, false, "电气工程及其自动化"], [22, 1, 1, 1, false, "582"], [22, 2, 1, 1, false, "562"], [22, 3, 1, 1, false, "566.06"], [22, 4, 1, 1, false, "新能源学院"]]
[[23, 0, 1, 1, false, "电气工程及其自动化"], [23, 1, 1, 1, false, "605"], [23, 2, 1, 1, false, "573"], [23, 3, 1, 1, false, "580.34"], [23, 4, 1, 1, false, "物理类"]]
[[24, 0, 1, 1, false, "电气工程及其自动化"], [24, 1, 1, 1, false, "596"], [24, 2, 1, 1, false, "579"], [24, 3, 1, 1, false, "584.7"], [24, 4, 1, 1, false, "地方专项"]]
```

</details>

## DIRECT CHUNKING BEFORE FIX

45 chunks，其中表格 43 chunks；section_count=0。

首个表格 chunk 的 `row_start=2, row_end=14`，全文 SHA-256：
`6f3988435b37d628422b7bf843cc333b7c958a75377e380670290ac3eae14d42`。

```text
[表格]
表头：列1；列2；列3；列4；列5
说明：内蒙古工业大学2026年分省录取分数统计
说明：内蒙古本科提前B段艺术类
列1：专业名称；列2：最高分；列3：最低分；列4：平均分；列5：备注。
```

下一个表格 chunk `row_start=15, row_end=27` 仍以 `表头：列1；列2；列3；列4；列5` 开头，没有 global/section。因为表头 flags 全 False，旧逻辑的 `headers[row + 1]` 不成立，section detector 不启动。

## FIX

`_row_signature()` 对 grid 中的原始 anchor 去重，按列序生成 `(column_index, column_span, normalized_text)` 元组；包含空值 anchor，文本仅做 NFKC 和空白折叠。不使用业务关键词、模糊匹配或模型。

候选签名必须至少有两个非空 anchor，所有 cell 在当前行锚定且 row_span=1，同一 TableBlock 内完全重复至少两次。至少一次 occurrence 紧邻独立 full-width contextual row，后一行必须有正常 data 结构：至少两个非空 anchor、无 rowspan、非官方 header、不是重复候选签名。

将可靠签名对应的行临时加入 header mask；保留 Docling authoritative header 优先的 OR 路径，并复用原有 global/section state machine、header refresh 和 `_build_header_labels()`。不修改 ParsedDocument/TableCell，不改 parsing、Firecrawl、candidate identity、retrieval、reranker 或 generation。

## FALSE-POSITIVE PROTECTION

- 重复数据没有相邻 full-width context，不会仅因重复被识别为 header；同名专业分数不同也不匹配完整签名。
- 业务词无特殊地位；英文 neutral fixture 同样通过。
- 普通 full-width colspan 只有一个 anchor，不满足两个非空 cell；不同列位置/span 即使文本相同也不匹配。
- 排除 rowspan 及继承的 grid 投影，不跨 TableBlock 累计签名；纯连续重复行不能互相提供 data 证据。
- authoritative simple/multilevel/rowspan/colspan 原测试继续通过。

## DIRECT CHUNKING AFTER FIX

真实页面现在生成 **67 chunks（65 个表格 chunks）/35 个分区**，没有 `专业名称：专业名称`。以下仅展示每个 chunk 前 300 字符，hash 使用完整正文；物理类样本为截取，未人为添加结束标记。

`section_title=内蒙古本科提前B段艺术类`，`row_start=3`，`row_end=5`。

```text
[表格]
上下文：内蒙古工业大学2026年分省录取分数统计
分区：内蒙古本科提前B段艺术类
表头：专业名称；最高分；最低分；平均分；备注
专业名称：服装与服饰设计；最高分：501；最低分：465；平均分：470；备注：投档成绩。
专业名称：环境设计；最高分：487；最低分：475；平均分：477.87；备注：投档成绩。
专业名称：总计；最高分：501；最低分：465；平均分：475.4；备注：投档成绩。
[/表格]
```

全文 SHA-256：`730bb46195442f16deeadb85490dc26e63a358b34a136e16dca358384ecb41d1`。

`section_title=内蒙古本科批物理类`，`row_start=8`，`row_end=18`。

```text
[表格]
上下文：内蒙古工业大学2026年分省录取分数统计
分区：内蒙古本科批物理类
表头：专业名称；最高分；最低分；平均分；备注
专业名称：安全工程；最高分：527；最低分：501；平均分：508.32；备注：物理类。
专业名称：材料成型及控制工程；最高分：549；最低分：509；平均分：517.98；备注：物理类。
专业名称：材料成型及控制工程；最高分：503；最低分：480；平均分：487.75；备注：民族班。
专业名称：材料物理；最高分：538；最低分：499；平均分：505.04；备注：物理类。
专业名称：财务管理；最高分：455；最低分：452；平均分：453.25；备注：民族班。
```

全文 SHA-256：`3f2a56dcd0ebda2e37282e8c7a5f9e4dfef64f22ad0b553fc8f0848d2bb0e4b2`。

## CHROMA VERIFICATION

对上述 source_id 精确调用 `create_vector_store(...).get(where={"source_id": source_id}, include=["documents", "metadatas"])`。

| 层级 | chunk 数 | section 数 | direct/indexed 一致 |
| --- | --- | --- | --- |
| 修改前现有实库 | 45 | 0 | 是，旧投影与实库完全相同 |
| 修改后现有实库 | 67 | 35 | 是，全文和全部 metadata 相同 |
| 隔离真实 Chroma + SQLRecordManager | 67 | 35 | 是，且 runtime reload 一致 |

物理类样本全文 SHA-256：

- direct chunk hash: `3f2a56dcd0ebda2e37282e8c7a5f9e4dfef64f22ad0b553fc8f0848d2bb0e4b2`
- indexed chunk hash: `3f2a56dcd0ebda2e37282e8c7a5f9e4dfef64f22ad0b553fc8f0848d2bb0e4b2`
- `section_title=内蒙古本科批物理类`，`row_start=8`，`row_end=18`，一致。

67 条 `(page_content SHA-256, section_title, row_start, row_end)` 排序后集合摘要：

- direct: `7e76fa28b9f9a289c2ffd2f8ebfb7d4903634ed9e940e7cf40a143e15686f33a`
- indexed: `7e76fa28b9f9a289c2ffd2f8ebfb7d4903634ed9e940e7cf40a143e15686f33a`

隔离索引使用同一个实抓 candidate、原 namespace/collection 名称、临时 SQLite/Chroma 目录和 deterministic fake embeddings，复用正式 source service：

| 操作 | added | skipped | deleted |
| --- | --- | --- | --- |
| 导入真实旧 45 chunks | 45 | 0 | 0 |
| index_approved 新表示 | 66 | 1 | 44 |
| 再次 index_approved | 0 | 67 | 0 |
| remove_indexed | 0 | 0 | 67 |
| 同 candidate 再次 index_approved | 67 | 0 | 0 |

每步检查 source_id 查询；最终 RecordManager keys、Chroma ids、runtime 文本相符，candidate 内容不变。移除后空 runtime、重新索引后 ready。未发现 lifecycle bug。

## PIPELINE VERSION

before: **4**；after: **5**。chunk representation 改变，既有 persisted structured candidates 应经 `needs_rebuild()` 和现有全局 force rebuild 重新投影。测试覆盖旧版本 2/3/4，不改 collection、namespace，不新增 migration script。结束时实库 manifest 也已为 5。

## FILES CHANGED

- `app/rag/structured_chunking.py`：小型签名 helper、可靠签名集合和 header mask 合并。
- `app/rag/store.py`：pipeline_version 4→5。
- `tests/support/structured_document.py`：冻结实抓前 11 行的无 header flag fixture。
- `tests/test_structured_chunking.py`：真实回归、分区/全局传播、重复专业、结构保留及误判防护。
- `tests/test_rag_retrieval.py`：复用既有 Chroma/BM25 + fake embeddings smoke，加入实抓 fixture 三条查询。
- `tests/test_external_source_indexing.py`：4→5 持久结构重建、无重新抓取/解析、Chroma 全 metadata 一致性。
- `tests/test_rag_indexing.py`：版本判断断言及旧版本 4 覆盖。
- `docs/ingestion/table-aware-1f-result.md`：本报告及有限真实结构证据。

## TESTS

实际执行：

1. 修复前：`.venv/bin/pytest tests/test_structured_chunking.py -q -k real_header_signature`：2 个新增回归均以缺失 `section_title` 失败，确认复现。
2. 修复后：`.venv/bin/pytest tests/test_structured_chunking.py -q`：**55 passed**。
3. `.venv/bin/pytest tests/test_rag_retrieval.py tests/test_external_source_indexing.py -q -k 'section or admission or header or migration or pipeline or firecrawl_html_sqlite'`：**14 passed, 47 deselected**。
4. 加强 inherited-context 测试并补齐版本断言后：`.venv/bin/pytest tests/test_structured_chunking.py tests/test_rag_indexing.py -q -k 'fallback_rejects_rowspan or pipeline_version'`：**5 passed, 75 deselected**（2 个重复验证 + 3 个版本用例）。
5. Ruff：本次全部 7 个 Python 文件通过。
6. lint-imports：**7 kept, 0 broken**。
7. git diff --check：通过。
8. 两次实际 Firecrawl/source/Docling 诊断，以及上述实库只读对比和隔离 Chroma lifecycle 人工验证：通过。

共 **72 个不同的 focused tests** 在最终相关代码下通过。没有运行全量 pytest、完整 Agent/MCP/RAG Evaluation、14 题 LLM Evaluation、SQL/SSE/Online/Mixed，也没有 LLM 调用或 reranker 推理。

## LOC / COMPLEXITY

production added: **35**；production deleted: **3**；net: **+32**。

new dependency: **0**。没有新增 detector class、registry 或第二套 table chunker。

## FINDINGS

- fallback 有意局限于无 rowspan 的重复单行签名，需紧邻独立全宽 context 和正常 data；证据不足的表仍保留旧投影，不进行模糊推断。
- 实库在两次只读观察之间已从旧 45 chunks/manifest 4 更新至 67 chunks/manifest 5；诊断没有主动执行生产 reindex/服务重启，未追溯触发此次刷新的进程。最终实库内容已逐条核对一致。
- runtime reload 验证针对隔离进程；没有检查既有在线服务内存，也没有评估端到端答案质量。
- 隔离 source service 的原有 reload 会加载本地 reranker 权重；本次没有调用 rerank 推理。pytest 出现既有 pkg_resources/httpx 弃用警告，无失败。

## NEXT

**STOP_TABLE_AWARE_EXPANSION**。真实页面已形成 global context + section + repeated header + 正确 data rows，direct 与当前实际 Chroma 一致；不启动下一阶段。
