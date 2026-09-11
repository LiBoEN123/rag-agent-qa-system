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
GET  /admin/ingestion/candidates
GET  /admin/ingestion/candidates/{candidate_id}
POST /admin/ingestion/candidates/{candidate_id}/approve
POST /admin/ingestion/candidates/{candidate_id}/reject
POST /admin/ingestion/candidates/{candidate_id}/index
GET  /admin/ingestion/index-runs
```

GitHub 与 Firecrawl 请求只描述 source，不接受正文、trust、permissions、source ID、
candidate ID 或 approval 状态。服务端分别只授予 `mcp:github` 或 `mcp:firecrawl`，并使用
固定的 `ingestion-admin-api` service requester。

列表支持 `status`、`source_type`、`limit`、`offset`，默认 `limit=50`，上限 100；列表
只返回 `IngestionService.summarize()` 生成的安全摘要。详情最多返回 2000 字正文预览和
已经过 ingestion contract 清理的 metadata，没有 full-content endpoint。

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
必须请求 `formats=["markdown"]`，返回完整 markdown、有效页面 URL 和 2xx status。
工具错误、权限/治理拦截、审计失败、缺失来源、格式错误、空正文或任何截断均不入候选。
promotion 校验/持久化失败不改变成功的只读 ToolResult，治理事件仅附安全状态和候选 ID。

Agent 仍使用原来的长度上限（默认 12,000 字符）；净化器在本次 `ToolRuntime` 中暂存
安全 source 内容块，完成后移除，不把这些内容块或候选正文追加到模型消息。没有重抓取，
也不把摘要或截断文本转为永久知识。较长正文仍应通过已有显式 source API 提案。

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
