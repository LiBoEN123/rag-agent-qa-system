# TABLE-AWARE-1A/1B：结构化解析与持久化

```text
Local PDF/DOCX ─┐
               ├→ app.rag.parsing.parse_document()
Upload PDF/DOCX ┘       → Docling DocumentConverter
                       → app.ingestion.document_blocks.ParsedDocument
                       → KnowledgeDocument(content + structured_content + structure_hash)
                       → IngestionCandidate → SQLite candidate_json
                       → reload / load_latest_successfully_indexed() → 原有正文 chunking
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

本阶段只保存结构。下一阶段为 **TABLE-AWARE-1C — Table-aware Chunking**；当前
`candidate_to_langchain_documents()` 仍按正文切分，尚未实现 table-aware chunking、
表格检索扩展、rerank 改造或 Agent 路由。
