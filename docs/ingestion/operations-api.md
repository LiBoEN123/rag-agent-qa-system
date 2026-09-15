# Persistent Ingestion Operations API

Phase 12 将 Phase 11B 的 GitHub / Firecrawl proposal 与 approved-only LangChain indexing
组装为持久运营接口。它是 operator/service 级临时管理边界，不是用户认证系统，也不是
Agent Tool；项目没有借此增加用户表、JWT、OAuth 或 RBAC。

## 启用与认证

接口默认不注册：

```dotenv
INGESTION_ADMIN_API_ENABLED=false
INGESTION_ADMIN_API_KEY=
```

显式启用时必须配置独立 key，否则 lifespan 启动校验失败。请求通过
`X-Ingestion-Admin-Key` 发送，服务使用 FastAPI `APIKeyHeader` 读取并用
`secrets.compare_digest()` 比较。Key 不进入响应、业务日志或错误信息。

## API

```text
POST /admin/ingestion/sources/github
POST /admin/ingestion/sources/firecrawl
POST /admin/ingestion/sources/upload
GET  /admin/ingestion/candidates
GET  /admin/ingestion/candidates/{candidate_id}
POST /admin/ingestion/candidates/{candidate_id}/approve
POST /admin/ingestion/candidates/{candidate_id}/reject
POST /admin/ingestion/candidates/{candidate_id}/index
POST /admin/ingestion/candidates/{candidate_id}/remove
GET  /admin/ingestion/index-runs
```

GitHub 与 Firecrawl 请求只描述 source，不接受正文、trust、permissions、source ID、
candidate ID 或 approval 状态。服务端分别只授予 `mcp:github` 或 `mcp:firecrawl`，并使用
固定的 `ingestion-admin-api` service requester。

列表支持 `status`、`source_type`、`limit`、`offset`，默认 `limit=50`，上限 100；列表
只返回 `IngestionService.summarize()` 生成的安全摘要。详情最多返回 2000 字正文预览和
已经过 ingestion contract 清理的 metadata，没有 full-content endpoint。

## 本地文件上传（LOCAL-KB-UPLOAD-V1）

在现有「知识入库 → 来源提案 → 本地文件」中，Element Plus `el-upload` 负责单文件
选择/拖拽；关闭自动上传，提交时由 `ingestionHttp` 使用 `FormData` 与原有管理员 header
发送 `POST /admin/ingestion/sources/upload`。Multipart 字段为必填 `file` 和可选
`reason`（最多 1000 字符），成功返回 HTTP 201 与现有 `IngestionCandidateSummary`。

后端只接受 `.pdf/.docx/.txt/.md`，扩展名不区分大小写。Local 与 Upload 的 PDF/DOCX
共用 `app.rag.parsing.parse_document()`，由 Docling 提供布局、reading order 和表格
cell/span 解析；TXT/MD 继续使用 `charset-normalizer` 编码识别。PDF 保留页边界，
统一 chunking 由 `candidate_to_langchain_documents()` 完成：有结构时文本按 segment
切分、表格按完整连续行分组；无结构的 legacy candidate 保持原正文切分，详见
[结构化文档切分](structured-documents.md#table-aware-1c进入现有索引的-chunks)。

`INGESTION_UPLOAD_MAX_BYTES` 默认 20 MiB，必须为正整数。框架接收 multipart 并管理
spooled 文件，source adapter 在线程池中按最多 64 KiB 分块读取，超限即停止；DOCX 在
loader 解压前还校验 ZIP 声明的展开总大小不超过同一限额。该配置限制单文件处理大小，
不是 ASGI 收包之前的整个 HTTP 请求限额。

用户文件名只用于安全的标题/metadata，不能决定服务器路径；标准 multipart parser
可能先将 Windows 客户端路径简化为 basename，adapter 拒绝其余路径及控制字符。
解析仅使用 `TemporaryDirectory` 下的固定文件名，成功、解析失败和超限都会清理。
原始文件不长期保存，loader 的临时 `source` metadata 不进入候选。

原始文件 SHA-256 生成 `upload://local/<sha256>`；`build_knowledge_document()` 负责
其余 source/document/content identity、规范化和脱敏。Metadata 保留 `file_name`、
`file_type`、`file_size`、`file_sha256`，PDF 另有 `page_count`；chunk `source_path`
沿用现有 fallback，取稳定 `source_uri`。改名重传同一文件返回首次保存的候选，
不会重置既有审批状态。

上传固定为 `UPLOAD / UNTRUSTED_EXTERNAL`，正常全文需审批后再手动索引。既有 policy
仍执行正文上限（默认 1,000,000 字符），超出时生成 rejected 候选，不截断正文。
上传和批准都不会写 Chroma，完整候选继续保存在现有 SQLite repository。
成功索引后通过 `load_index_cache()` 重建共享 runtime/BM25；global force rebuild
从 `load_latest_successfully_indexed()` 恢复最新成功操作为 INDEX 的上传全文，无需原始文件；
最新成功操作为 REMOVE 的来源不再恢复。

上传错误使用现有 `{ "error": { "code", "message" } }` 格式：

| HTTP | code | 条件 |
| --- | --- | --- |
| 413 | `UPLOAD_TOO_LARGE` | 文件或 DOCX 展开大小超限 |
| 415 | `UPLOAD_UNSUPPORTED_TYPE` | 不支持的扩展名 |
| 422 | `UPLOAD_INVALID_FILENAME` | 文件名包含路径/控制字符或无效 |
| 422 | `UPLOAD_EMPTY_FILE` | 零字节文件 |
| 422 | `UPLOAD_NO_TEXT` | 空白全文或无文本 PDF |
| 422 | `UPLOAD_PARSE_FAILED` | 文档损坏、伪造、加密或文本编码/内容无效 |

PDF/DOCX 依赖固定为 `docling==2.127.0`，移除不再使用的 `docx2txt` 顶层依赖；
`python-multipart==0.0.32` 保持原版本。Docling 的底层依赖由其自身管理，项目不实现
表格识别或引入第二套表格 parser。结构合同、模型缓存和限制见
[结构化文档解析与持久化](structured-documents.md)。

## Agent 工具结果提出候选（默认关闭）

在部署者的 MCP Server 配置中显式添加 `knowledge_promotion`，键为远端工具名，
值为已有 source 类型。例如在 `servers.github` 中添加：

```json
"knowledge_promotion": {"get_file_contents": "github"}
```

在 `servers.firecrawl` 中添加：

```json
"knowledge_promotion": {"firecrawl_scrape": "firecrawl"}
```

这些工具还必须位于该 Server 的 `allowed_tools` 中，并保持只读。省略该映射或使用
`{}` 就关闭 promotion。配置经 discovery 写入受控 `BaseTool.metadata`；远端 MCP
metadata、工具正文或模型参数不能开启 promotion，也不能设定 trust。

`ProjectToolGovernanceMiddleware.wrap_tool_call/awrap_tool_call` 共用的完成钩子确认
工具执行成功后，用现有 source 提取函数和 adapter 构建 `KnowledgeDocument`，再调用
`IngestionService(SQLiteCandidateRepository(settings.sqlite_db_path)).propose()`。
候选默认是 `pending / untrusted_external`，管理 API 的列表、审批和索引接口可直接操作。
创建候选不依赖管理 API 是否启用；人工管理仍须启用上面的 API 并提供管理 key。

GitHub 必须返回匹配 owner/repository/path、带合法 SHA 的完整文本资源。Firecrawl
promotion 接受 `formats=["markdown"]`（旧 plain 路径）及 `["markdown", "html"]`
（顺序可交换），返回完整 markdown、有效页面 URL 和 2xx status；拒绝 json、rawHtml、
screenshot、images 等其他组合。非空 cleaned HTML 按 HTML MIME 交给 Docling，MIME
缺失也允许；PDF 等明确非 HTML 类型保持旧路径。
工具错误、权限/治理拦截、审计失败、缺失来源、格式错误、空正文或任何截断均不入候选。
promotion 校验/持久化失败不改变成功的只读 ToolResult，治理事件仅附安全状态和候选 ID。

Agent 仍使用原来的长度上限（默认 12,000 字符）；净化器在本次 `ToolRuntime` 中暂存
安全 source 内容块，完成后移除，不把这些内容块或候选正文追加到模型消息。没有重抓取，
也不把摘要或截断文本转为永久知识。较长正文仍应通过已有显式 source API 提案。

显式 Firecrawl source API 现在请求 Markdown + cleaned HTML，并保持
`onlyMainContent=True`、`removeBase64Images=True`。链路为 Firecrawl → cleaned HTML
→ Docling `convert_string` → ParsedDocument → 现有 table-aware chunks；关闭二次网络/
本地资源抓取、图片读取与浏览器渲染。HTML conversion 失败返回安全错误，不静默回退。
仍保留既有 serialization/正文上限和 truncated 拒绝；双 format 较大的结果可能失败，
具体预算及 Docling caption 限制见[结构化解析文档](structured-documents.md)。

HTML 原文不持久化，结构和 hash 保存在现有 candidate_json 中。历史 Firecrawl 候选
继续按纯文本索引；force rebuild 只读持久候选，不重抓 URL 或重跑 HTML parser，
不会自动升级历史网页结构。获得结构需要重新提案该 URL、批准并索引。

同一 source/content hash 复用既有 candidate ID；重新读取已审批/拒绝的候选不会重置
审批状态。只有显式 `/approve` 后再调用 `/index` 才会执行增量索引和 RAG runtime reload。

## 持久化与双阶段操作

`SQLiteCandidateRepository` 使用现有 `SQLITE_DB_PATH` 的 `ingestion_candidates` 表，完整
candidate 以版本化 Pydantic JSON 保存；每次读取都调用
`IngestionCandidate.model_validate_json()`，损坏记录 fail closed。既有 session、message、
agent run 与 agent step 表保持不变。

审批与索引是两个显式请求：approve 只调用 `IngestionService.approve()`，不会自动索引；
只有 approved candidate 才能调用 `ExternalKnowledgeSourceService.index_approved()`。
Candidate 状态仍只有 pending、approved、rejected、superseded。

每次合法 index 请求先写 `ingestion_index_runs(running)`，再同步执行 LangChain indexing
和 RAG runtime reload。成功保存 added/updated/deleted/skipped，失败只保存短的稳定错误码；
不保存正文、MCP payload、traceback 或原始异常。重复 index 由 LangChain RecordManager
自然返回 skipped，Candidate approval state 不受 index 成败影响。

## 移出知识库（EXTERNAL-KNOWLEDGE-REMOVE-1）

管理员在候选详情点击「移出知识库」并确认，发送无请求体的
`POST /admin/ingestion/candidates/{candidate_id}/remove`，继续使用同一管理员 header。
入口是 candidate ID，实际按其 `document.source_id` 撤除当前入库版本的全部 chunks；
旧候选也能撤除同一来源的新版本，不要求入口候选仍为 approved。
仅支持既有 namespace mapping 中的 Upload、GitHub、Firecrawl，Local/Web 返回
`422 SOURCE_TYPE_UNSUPPORTED`。Local 仍由文件系统与 Local namespace sync 控制。

`ExternalKnowledgeSourceService.remove_indexed()` 先查询对应 namespace RecordManager 的
`group_ids=[source_id]` keys，再通过 Chroma 官方 metadata 过滤删除同 source 的 chunks，
随后删除这些 keys 并 reload runtime/BM25。`deleted` 是实际查到的 Chroma chunk 数；
其余计数为 0。重复撤库可成功返回 `deleted=0`，无需 embedding 或远端抓取。
移除最后一份资料时，严格验证 collection 为空后发布空 runtime，清除旧 chunks、
BM25、文档列表和向量矩阵，`RAG_READY=false`。

INDEX / REMOVE 共用 `ingestion_index_runs`，增加 `operation=index|remove`，继续使用
running/success/failed。现有 SQLite 在 `init_db()` 中幂等 ADD COLUMN，历史行默认 index，
不删表或重建表。响应及详情的 `latest_index_result` 包含 operation；记录列表支持
`operation` 筛选。详情仍按 candidate 展示最近操作，不将它当作整个 source 的成员状态。

global rebuild 按 run ID 倒序查看每个 source 最新成功操作：INDEX 恢复对应候选全文，
REMOVE 排除该 source，绝不回退到更老 INDEX。FAILED/RUNNING 不改变成员状态。
再次成功 INDEX 后，该来源重新参与重建。候选与审批历史均保留。

各持久存储和内存发布之间沿用现有无跨库事务边界：物理删除或 reload 失败会记录
FAILED REMOVE，可能已部分删除，但不会成为撤库成功事实；修复后可手动重试撤库，
或 force rebuild 按上一成功 INDEX 恢复。异常只对外暴露安全错误码，不自动重试。
