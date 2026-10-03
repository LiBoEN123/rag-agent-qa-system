# Governed GitHub / Firecrawl Source Adapters

Phase 11B 的永久知识流程是显式 operator/service orchestration，不是 Agent Tool：

```text
cached governed MCP tool
  → ingestion serialization
  → validated source payload
  → pure Source Adapter
  → KnowledgeDocument
  → IngestionService.propose() = pending

Explicit IngestionService.approve()
  → IndexingService.index_approved()
  → LangChain index(cleanup="incremental")
  → persistent index

Combined FastAPI composition root → optional query runtime refresh
```

`GitHubFileSourceAdapter` 只消费已验证的 `GitHubFilePayload`。GitHub fetch 固定使用
`github_get_file_contents`，正文只取匹配请求 owner/repository/path 的
`EmbeddedResource(TextResourceContents)`；描述性 TextContent 不会拼入正文。真实
`repo://.../sha/<sha>/contents/...` URI 中存在合法 SHA 时用作 version，否则 version 为
`None`。source identity 固定为 `github://owner/repository/path`。

`FirecrawlPageSourceAdapter` 只消费已验证的 `FirecrawlPagePayload`。fetch 固定使用
`firecrawl_firecrawl_scrape` 和 markdown + html formats；当前真实响应的 `markdown` 是权威正文，
metadata 只保留 final URL、status、MIME 和 language。用户显式选择的 canonical requested
URL 是 source identity，redirect URL 只作 metadata。Search 只可用于未来 discovery，
Crawl、scheduled sync 和 remote delete detection 均不在本阶段。

两个 adapter 都无 HTTP、MCP、approval、embedding 或 Chroma 依赖，只调用 Phase 9
`build_knowledge_document()`。外部 trust 固定从 `untrusted_external` 开始。正文先经 MCP
serializer 脱敏，再由 KnowledgeDocument 做权威 normalization/redaction/hash；永久索引只
消费 `KnowledgeDocument.content`，不消费 raw MCP payload。

MCP 仅供入库，文本上限为 1,000,000 字符，structured content 总预算为 2,000,000 字符；
经直接 bridge invocation 使用只读来源工具，保留完整 Markdown/HTML 和安全 metadata。任一截断、超限、block 歧义、
错误 trust marker 或无完整正文的结果都 fail closed，不创建 candidate。MCP audit 只保存
tool、字段、字符数、截断和状态摘要，不保存正文、base64、Header 或凭据。

GitHub 与 Firecrawl 分别使用 `rag-agent/github/v1` 和
`rag-agent/firecrawl/v1` SQLRecordManager namespace，共用 `mini_rag_chunks_v2`。external
sync 使用 incremental cleanup；Local 继续使用自己的 namespace 和 full cleanup。候选仍由
领域 repository 管理；正式 API 使用 `SQLiteCandidateRepository` 持久化。Source service
只做 fetch/validation/proposal，不持有 embedding、索引依赖或 QA runtime。
