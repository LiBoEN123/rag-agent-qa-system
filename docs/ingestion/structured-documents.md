# TABLE-AWARE-1A/1B/1C：结构化解析、持久化与切分

```text
Local PDF/DOCX ─┐
               ├→ app.rag.parsing.parse_document()
Upload PDF/DOCX ┘       → Docling DocumentConverter
                       → app.ingestion.document_blocks.ParsedDocument
                       → KnowledgeDocument(content + structured_content + structure_hash)
                       → IngestionCandidate → SQLite candidate_json
                       → reload / load_latest_successfully_indexed()
                       → ordered TextBlock / TableBlock chunking → LangChain Documents
```

Docling `2.127.0` 的公开 `DocumentConverter`、`PdfPipelineOptions`、
`DoclingDocument.iterate_items()`、`TableItem.data.table_cells`、`caption_text()` 和
`prov` 负责解析。项目只映射字段，结构合同不依赖 Docling，也不保存框架对象。
使用方式参考 [Docling converter](https://docling-project.github.io/docling/reference/document_converter/)
及 [Docling document](https://docling-project.github.io/docling/reference/docling_document/)。

## 稳定结构合同

四种冻结、禁止额外字段的 Pydantic 值对象定义在 `app/ingestion/document_blocks.py`：

| 模型 | 字段 |
| --- | --- |
| `TextBlock` | `kind="text"`、`block_id`、`text`、`order`、`page_number` |
| `TableCell` | `text`、`row_index`、`column_index`、`row_span`、`column_span`、`is_header` |
| `TableBlock` | `kind="table"`、`block_id`、`caption`、`order`、`page_number`、`cells`、`row_count`、`column_count` |
| `ParsedDocument` | `structure_version=1`、`blocks`、`page_count` |

`blocks` 是单一 reading-order 序列，`order` 从 0 连续编号，`block_id` 是稳定的
`b0/b1/...`。行列从 0 编号，页码从 1 编号。无页信息时为 `None`，DOCX 不推算
排版页码；多页 provenance 的 block 记录 Docling 给出的起始页。Header 使用官方
`column_header or row_header`，caption 有关联时保存；缺失时不自行猜测。
Rich cell 正文不再重复输出为独立段落。更复杂的 nested table 层级暂未建模。
当前 Docling DOCX backend 把 1×1 单元格表格当作正文容器，不返回 TableItem；项目
遵循该框架输出，不另写识别算法恢复这种边界情况。

`content` 是规范化结构的确定性正文视图：保留段落、页标记和 `[表格]` 行列；
合并区域只在左上锚点输出文字，其他格为空；cell 内反斜线、竖线、换行分别转义。
Span 和 header 以结构字段为准，文本视图不承担完整表达这些关系的职责。

## 安全、identity 与兼容

- `prepare_structured_content()` 对每个 block text、cell text 和 caption 复用
  `prepare_content()`，执行同一套脱敏、NFC、换行和行尾空白规范化。任一字段脱敏
  都设置 `content_redacted`。模型只接受明确字段和 JSON 值，ID 不接收自由文本。
- 单结构最多 10,000 blocks、50,000 cells、100,000 个逻辑表格位置；单文本字段
  最多 1,000,000 字符，完整结构 JSON 最多 8,000,000 UTF-8 bytes。校验
  cell/span 边界、重叠和 reading order。超限拒绝，不截断或丢弃一部分表格。
  正文仍经过现有 ingestion policy 的可配置大小上限。
- Factory 有结构时从安全结构生成正文；直接构造或 reload `KnowledgeDocument`
  时核对正文视图和两个 hash，拒绝结构篡改或不一致的正文。
- `build_content_hash()`、`source_id`、`document_uid` 的定义不变。
  `structure_hash` 是规范化结构（含 `structure_version`）的 canonical JSON
  SHA-256：固定 key 排序、紧凑分隔符、UTF-8；cell 顺序规范为行列顺序。
  结构 candidate 使用 `ingestion-candidate:structured-v1` namespace，把
  `source_id + content_hash + structure_hash` 纳入 identity；相同结构幂等，span
  或 header 等结构变化会产生不同 candidate。
- 无结构时 `structured_content=None`、`structure_hash=None`，candidate ID 完全
  沿用旧 v1 算法。KnowledgeDocument/Candidate `schema_version` 保持 1，旧 JSON
  缺少新字段时用默认值加载。SQLite 沿用 `candidate_json`，不新增表或迁移 schema。

## Local / Upload 与 rebuild

TXT/MD 的原有读取和清洗行为保持；GitHub/Firecrawl 不需要创建结构。
Local 文件仍是其权威来源，Local rebuild 会重新解析磁盘文件，其 admission candidate
仍是原有内存流程。Upload 原始文件在临时目录解析后删除，结构随 candidate 持久化；
global force rebuild 读取最新成功 INDEX 对应的 candidate 原样恢复结构，不重新
上传、抓取或解析外部来源。Approve 与 Index 继续是两个独立动作。

Upload 保留 filename 校验、分块字节上限、PDF header 校验、DOCX ZIP 展开限制、
空文件/无正文检测、固定安全错误消息和临时目录清理。结构超限或不完整解析返回
`UPLOAD_PARSE_FAILED`。完整结构不会放入 metadata、chunk metadata 或摘要响应。

历史上传若仅保存了纯文本，已经丢失的 table structure 无法自动恢复，必须重新上传。
这也适用于 force rebuild：它能恢复已保存的结构，不能从旧纯文本重建原始单元格关系。

## TABLE-AWARE-1C：进入现有索引的 chunks

`candidate_to_langchain_documents()` 优先判断 `structured_content is not None`，
与文件扩展名和来源类型无关。有结构时交给 `app.rag.structured_chunking`：

- 连续同页的 TextBlock 用双换行组成文本 segment，再交给现有 LangChain
  `RecursiveCharacterTextSplitter`，保留 `chunk_size`、`chunk_overlap` 和
  `_TEXT_SEPARATORS`。遇到表格、页码变化（包括已知/未知切换）或文档结束时 flush。
- TableBlock 只根据持久化 cells/span/header 建立临时 grid，不经过字符 splitter，
  不读源文件、不调用 Docling/LLM、不重复 domain 的结构校验。
- 前导非空且全部标记为 header 的行构成列标题带；空行不提供标题。多级表头按
  `父标题 > 子标题` 连接，colspan 贡献给覆盖的列，rowspan 不重复同一标题。
  混合数据行内的 header cell 作为行值保留。合同只有合并后的 `is_header`，
  不恢复已经丢失的行头/列头类别；缺失列标题用“列N”，不猜第一行是表头。
- Body rowspan 的值出现在其覆盖的每个逻辑行；body colspan 用组合列标题描述
  一个值。原 cells、structure_hash、candidate identity 不变。

例如表头 `事项 | 条件 | 材料` 和一行 `企业注册 | 年满18岁 | 身份证、申请表`，
带 caption 的最终 `Document.page_content` 为：

```text
[表格]
标题：企业登记办理要求
表头：事项；条件；材料
事项：企业注册；条件：年满18岁；材料：身份证、申请表。
[/表格]
```

小表格完整保留在一块；大表格按连续 body rows 分组，预算包含标题、表头和标记，
复用 `chunk_size`（默认 600 字符）。每组重复 caption/headers，body rows 无 overlap。
一个完整行连同固定标题开销超过预算时独立成块，允许超过 soft target，不硬切、
截断或丢弃。全空行不生成证据；仅有 caption/header 的表仍保留，省略 body 行范围。

Metadata 保留原 provenance，并统一按 reading order 分配从 1 开始的连续 `position`。
只增加以下 scalar 字段，页码未知时省略：

| Chunk | 新增 metadata |
| --- | --- |
| Text | `element_type=text`、`block_start`、`block_end`、可选 `page_number` |
| Table | `element_type=table`、`table_id`、`block_order`、`table_chunk_index`、`table_chunk_count`、`row_start`、`row_end`、可选 `page_number` |

`block_start/end` 是文本 segment 的原始 block order 范围，拆出的 chunks 共用它。
`table_id` 是该 source 内的 `block_id`；`table_chunk_index` 从 1 开始。
`row_start/end` 是原 TableCell 行索引，**0-based、inclusive**，排除前导列标题带，
范围可能包含不生成文字的空行。Chroma metadata 不保存结构对象或完整表格副本。

没有结构的 TXT/MD 和历史 PDF/DOCX 仍使用原有正文/Markdown 切分；目前的
GitHub/Firecrawl Markdown/plain content 行为不变。`page_content` 是进入现有
embedding、BM25、CrossEncoder 和 evidence context 的同一份文本。

### 索引迁移与验证边界

`pipeline_version` 从 **2 → 3**。启动时现有 `needs_rebuild()` 发现旧 manifest，
触发 global force rebuild：Local 重新解析切分；SQLite 中最新成功 INDEX 对应的
external candidates 恢复后，有结构走新 chunker，无结构走旧路径。
迁移会重新 embedding 受管 corpus；只 approved、未成功 index 的 candidate 不会入库。
沿用同一个 `mini_rag_chunks_v2` collection 和全部原 RecordManager namespaces，
继续使用 LangChain Indexing API 的 `source_id_key="source_id"`、`key_encoder="sha256"`
及 full/incremental cleanup，不创建第二个表格索引或 migration script。

Focused retrieval smoke 使用真实 Chroma/SQLRecordManager/BM25 和确定性 fake embedding，
查询“企业注册需要哪些材料”，验证命中含“事项：企业注册；条件：年满18岁；材料：身份证、申请表。”
的 table chunk。它证明语义文本进入现有检索，不提供真实模型效果提升指标。
Retrieval、reranker、Evidence Gate、generation 和 `max_context_chars`（默认 2500）
均不变；极长行的 chunk 可能超过最终 context budget，现有 context builder 会整块排除。

## 运行依赖与阶段限制

PDF 启用官方布局和表格模型，**OCR、图片理解、远程模型服务和外部插件均关闭**。
DOCX 不需要下载布局/表格模型。PDF 首次转换会通过 Docling/Hugging Face 下载模型，
之后使用本机缓存；无网络部署应提前准备模型缓存，见
[Docling advanced options](https://docling-project.github.io/docling/usage/advanced_options/)。
真实 PDF focused tests 也需要这些模型，首次运行有下载成本。

本次验证环境为 Python 3.12、Docling 2.127.0、docling-core 2.96.0，既有
`torch==2.12.1+cu126` 对应 `torchvision==0.27.1+cu126`。已安装特定 CUDA wheel
的环境必须使用匹配的 torchvision 构建；从 PyPI 默认安装 torchvision 后曾出现
`torchvision::nms` 缺失，改用 PyTorch 官方 cu126 wheel 后真实解析测试通过。
测试 PDF 为 1,933 bytes 的专用 fixture；真实 DOCX 在测试中用 python-docx 动态生成。

本阶段未实现 Web/Firecrawl structured parsing、表格上下文扩展或新检索路由。
下一阶段建议 **TABLE-AWARE-1D — Web Structured Ingestion**：Firecrawl cleaned HTML
→ structured parser → ParsedDocument，直接复用本阶段 chunker。
