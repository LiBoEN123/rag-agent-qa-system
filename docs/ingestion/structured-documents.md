# TABLE-AWARE-1A/1B/1C/1D：结构化解析、持久化与切分

```text
Local PDF/DOCX ─┐
               ├→ app.ingestion.parsing.parse_document()
Upload PDF/DOCX ┘       → Docling DocumentConverter
Firecrawl cleaned HTML → 同一 converter.convert_string(..., InputFormat.HTML)
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
| `TextBlock` | `kind="text"`、`block_id`、`text`、`order`、`page_number`、`text_role`、`heading_level` |
| `TableCell` | `text`、`row_index`、`column_index`、`row_span`、`column_span`、`is_header` |
| `TableBlock` | `kind="table"`、`block_id`、`caption`、`order`、`page_number`、`cells`、`row_count`、`column_count` |
| `ParsedDocument` | `structure_version=1`、`blocks`、`page_count` |

`blocks` 是单一 reading-order 序列，`order` 从 0 连续编号，`block_id` 是稳定的
`b0/b1/...`。行列从 0 编号，页码从 1 编号。无页信息时为 `None`，DOCX 不推算
排版页码；多页 provenance 的 block 记录 Docling 给出的起始页。Header 使用官方
`column_header or row_header`，caption 有关联时保存；缺失时不自行猜测。
Rich cell 正文不再重复输出为独立段落。更复杂的 nested table 层级暂未建模。
`text_role` 为 `text`、`title` 或 `section_header`；只有 `section_header` 必须提供
`heading_level >= 1`，其余为 `None`。角色不改变 `kind="text"` 或 reading order。
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
  Hash canonicalization 仅从普通正文省略新增的默认 `text_role="text"` 和
  `heading_level=None`，保留所有旧字段；旧 structured JSON reload 后 hash、
  candidate ID 不变。Title/section header 的角色和级别完整参与 hash，不使用
  会一并删除旧默认字段的 `exclude_defaults`。
  结构 candidate 使用 `ingestion-candidate:structured-v1` namespace，把
  `source_id + content_hash + structure_hash` 纳入 identity；相同结构幂等，span
  或 header 等结构变化会产生不同 candidate。
- 无结构时 `structured_content=None`、`structure_hash=None`，candidate ID 完全
  沿用旧 v1 算法。KnowledgeDocument/Candidate `schema_version` 保持 1，旧 JSON
  缺少新字段时用默认值加载。SQLite 沿用 `candidate_json`，不新增表或迁移 schema。

## Local / Upload 与 rebuild

TXT/MD 的原有读取和清洗行为保持；GitHub 和没有 HTML 的 Firecrawl 继续走正文路径。
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
与文件扩展名和来源类型无关。有结构时交给 `app.indexing.chunking`：

- 连续同页的 TextBlock 用双换行组成文本 segment，再交给现有 LangChain
  `RecursiveCharacterTextSplitter`，保留 `chunk_size`、`chunk_overlap` 和
  `_TEXT_SEPARATORS`。遇到 heading、表格、页码变化（包括已知/未知切换）或文档结束时 flush。
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

当前两类 chunk 都增加 scalar `title_path` 和 `[标题路径]` 文本前缀，详见下方
WEB-HEADING-AWARE-INGESTION；本节无前缀示例展示的是 table projection 本身。

`block_start/end` 是文本 segment 的原始 block order 范围，拆出的 chunks 共用它。
`table_id` 是该 source 内的 `block_id`；`table_chunk_index` 从 1 开始。
`row_start/end` 是原 TableCell 行索引，**0-based、inclusive**，排除前导列标题带，
范围可能包含不生成文字的空行。Chroma metadata 不保存结构对象或完整表格副本。

没有结构的 TXT/MD 和历史 PDF/DOCX 仍使用原有正文/Markdown 切分；目前的
GitHub 和 legacy Firecrawl Markdown/plain content 行为不变。`page_content` 是进入现有
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
该阶段未修改 Retrieval 与 reranker。当前 AGENTIC-RAG-6 已退役独立生成及 context
选择阶段，Agent 直接观察 ToolMessage evidence；表格诊断只衡量 chunk 是否通过重排和诊断选择。

## TABLE-AWARE-1D：Firecrawl 网页结构

显式 `propose_firecrawl_page()` 请求 `formats=["markdown", "html"]`，保留
`onlyMainContent=True`、`removeBase64Images=True`。Firecrawl 负责网页获取和正文清理；
`html` 是 transient cleaned HTML，`content` 保持 Markdown/plain fallback。
不请求 `rawHtml`、JSON extraction 或截图；不从任意 MCP text block 猜 HTML。
正文、HTML、metadata 均取自同一个经过验证的 response payload。

`FirecrawlPageSourceAdapter` 在非空 HTML 且 MIME 为 `text/html` 或
`application/xhtml+xml` 时调用同一 `app.ingestion.parsing.parse_html_document()`，忽略
MIME 的 charset 参数并规范大小写。MIME 缺失时，受验证的非空 cleaned HTML 也启用
结构解析；HTML 缺失/空白或 MIME 明确为 PDF/octet-stream 等其他类型时保留旧正文路径。
明确失败或部分成功的 HTML conversion 会拒绝提案，返回固定的
`SOURCE_PAYLOAD_INVALID` 安全错误，不静默回退 Markdown。

Docling 2.127.0 使用官方 HTML backend 与 `convert_string(..., format=InputFormat.HTML)`，
只消费字符串，不传源 URL、不创建临时文件。显式关闭 `enable_remote_fetch`、
`enable_local_fetch`、`fetch_images`、`render_page`，并关闭 `infer_furniture`，避免
把 Firecrawl 已清理的首个 heading 前正文丢弃。PDF options 不变。共同 adapter 继续
映射 reading order、cell offsets、spans 和 headers，HTML 的 `page_count=None`。

实测普通 HTML 表格保留 TextBlock/TableBlock/TextBlock 顺序；多级表头的 rowspan、
colspan、is_header 保留。**框架限制：测试 fixture 的 `<caption>` 未由该版本 HTML
backend 关联到 TableItem，故项目得到 `caption=None`，不自研补齐。** 对应 table chunk：

```text
[表格]
表头：事项；条件；材料
事项：企业注册；条件：年满18岁；材料：身份证、申请表。
[/表格]
```

Factory 继续从规范化结构生成 `content/content_hash/structure_hash`，SQLite 只保存
现有 `candidate_json`。原 HTML 不进入 metadata 或新表；Approve 与 Index 仍分开。
重启与 global force rebuild 从最新成功索引的持久 candidate 恢复，绝不重新调用
Firecrawl 或 HTML parser。历史 Firecrawl candidate 没有已保存结构，force rebuild
**不会自动升级**；必须显式重新抓取 URL、生成新候选、审批并索引。
`pipeline_version` **3 → 3**：本阶段只增加结构来源，未改变 chunk/index algorithm。

普通 Agent MCP 结果不创建候选；结构化永久知识只通过显式来源入库入口产生。

双 format 共享既有大小预算，**没有提高上限**：显式 ingestion 的 JSON text blocks
合计超过 1,000,000 字符会截断并拒绝；Agent text 默认上限 12,000。
原生 `structuredContent` 还经过既有 structured sanitizer，默认总预算 20,000，
首轮单字符串最多 4,000 字符；focused test 中 4,100 字符字段即触发拒绝。
不会因为另有完整 JSON block 而绕过全局 truncated 标记。较大的网页可能安全失败，
不会接收截断结构。测试使用本地 HTML、fake MCP 和确定性 embedding，无真实网页依赖。

## WEB-HEADING-AWARE-INGESTION：保留网页标题与章节边界

原 adapter 将 `TitleItem`、`SectionHeaderItem` 都降为普通 TextBlock，chunker
因而把相邻章节合成一个 segment。现在 adapter 直接映射 Docling 的公开角色与
`SectionHeaderItem.level`，不重新解析 HTML、不推断标题。当前安装的 Docling
2.127.0 / docling-core 2.96.0 实测 `<h1>` 为 `TitleItem`，`<h2>` 至 `<h6>` 为
level 1 至 5 的 `SectionHeaderItem`；这些是 Docling level，不是 HTML 标签编号。
PDF/DOCX 共用 adapter 也保留已有角色，但不启用新的 heading hierarchy 推断或 OCR。

Chunker 使用 `(level, text)` stack：title 在栈内使用 level 0，section 使用原 level；
新 heading 先 flush 正文，再弹出同级及更深标题。遇表格或换页只 flush，不清空路径。
连续祖先 heading 作为后续正文/表格的上下文，避免单独重复索引；没有正文或表格的
同级、末尾空章节仍生成仅含标题路径的证据。

完整路径由 `KnowledgeDocument.title + heading stack` 组成，去首尾空白后仅消除
相邻相同标题。正文先用原 RecursiveCharacterTextSplitter 分割，再给每个子 chunk
添加相同路径；表格仍用原 table chunker，之后补入同一路径。前缀不计入原字符
分组预算，与 Markdown 路径一样可能使最终 `page_content` 超过 `chunk_size`。
Table 的 `table_id`、`block_order`、`table_chunk_index/count`、`row_start/end`、
`section_index/title` 等原 metadata 全部保留。完整结构仍只在 candidate 中持久化。

例如 `document.title="研究生国家奖学金管理办法"`，cleaned HTML 为：

```html
<h1>研究生国家奖学金管理办法</h1>
<h2>第二章 申请条件</h2>
<h3>第五条</h3>
<p>申请人应当符合以下条件。</p>
```

正文 chunk 的 `title_path` 为 `研究生国家奖学金管理办法 > 第二章 申请条件 > 第五条`，
`block_start=block_end=3`，送入现有 Embedding/BM25/CrossEncoder/evidence 的正文为：

```text
[标题路径] 研究生国家奖学金管理办法 > 第二章 申请条件 > 第五条

申请人应当符合以下条件。
```

**Pipeline version 5 → 6**：即使旧 ParsedDocument 没有 heading 字段，现在也会将
已持久化的 `document.title` 加入正文和表格 chunk，旧 chunk 输出确实改变。因此
沿用现有 manifest force rebuild 同步检索表示；Local 重新解析，external 读取
SQLite 最新成功索引的 candidate，既有版本迁移测试覆盖 5 → 6 的标题前缀更新。
不新增数据库迁移，不在本次代码修改中执行生产索引重建。

旧 structured candidate 能正常 reload，structure hash 和 candidate ID 保持兼容；
但 **历史 Firecrawl candidate 中已丢失的 heading 信息不能通过 rebuild 恢复**。
获取章节边界必须重新抓取 URL、创建并审批新 candidate、索引。没有 HTML 的
legacy Markdown/plain fallback 和原 MarkdownHeaderTextSplitter 路径保持不变。
Firecrawl 的双 format、正文提取、安全校验、truncation 和 redaction 规则不变。

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

本阶段未实现表格上下文扩展或新检索路由，也未增加依赖。Web 普通表格已接入现有
chunking/indexing；下一步建议小型 **TABLE-AWARE-EVAL — Real Table Retrieval Evaluation**，
用真实检索结果决定是否继续扩展，1D 不自动开始该阶段。
