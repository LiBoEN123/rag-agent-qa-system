# Knowledge Ingestion Contract

Phase 9 建立独立于 RAG、Agent、Tool Runtime 和 MCP 的永久知识进入合同。它回答
“内容来自哪里、逻辑身份是什么、实际内容是否变化、信任级别是什么、是否允许进入”；
它不切分、不嵌入、不写 Chroma，也不重建 BM25。

## 数据流与阶段边界

```text
External / Local Source
        ↓
Source Adapter（只转换已取得的 payload，不发起网络请求）
        ↓
KnowledgeDocument
        ↓
Ingestion Policy（纯函数）
        ↓
IngestionCandidate
     ↙       ↘
 APPROVED    PENDING / REJECTED
     ↓
[STOP IN PHASE 9]
```

后续阶段的边界是：

```text
PHASE 10: Approved Candidate → Incremental Indexing
PHASE 11: GitHub / Firecrawl → live Source Adapter
```

Phase 9 没有公开 API、Agent Tool 或 `add_to_knowledge_base(content)`，也没有让一次
MCP/Web 调用成功自动产生永久知识。`approve()` 和 `reject()` 只替换冻结的 candidate
domain state。

## KnowledgeDocument

`KnowledgeDocument` 是 `extra="forbid"`、`frozen=True` 的 Pydantic v2 model，
`schema_version=1`。核心字段包括：

- `source_type`：`local_file`、`upload`、`github`、`web`、`firecrawl`；
- `source_uri`、`source_id`、`document_uid`：canonical provenance 与稳定身份；
- `content`、`content_hash`、`content_normalization`：实际候选正文及版本 identity；
- `version`：可选 source-native version，例如 commit SHA、ETag；
- `fetched_at`、`title`、`mime_type`、`language`、受限 `metadata`；
- `trust_level`：`trusted_internal`、`trusted_external`、`untrusted_external`；
- `content_redacted`、`metadata_redacted`：明确记录进入合同前是否发生脱敏。

正文无权声明或覆盖 `source_type`、`source_uri`、`trust_level`、policy 或 approval。
这些值只能由 adapter/调用边界作为独立参数提供。Firecrawl 等 payload 的
`content_is_untrusted=true` 会强制 `untrusted_external`，不能被同时传入的更高信任
级别覆盖。

## Stable identity 与版本语义

所有 ID 使用带版本 namespace 的 SHA-256，不使用 Python `hash()`、mtime、文件大小
或枚举顺序：

```text
source_id    = src_<sha256("knowledge-source:v1", source_type, canonical_source_uri)>
document_uid = doc_<sha256("knowledge-document:v1", source_id)>
content_hash = sha256(normalized_safe_content UTF-8 bytes)
candidate_id = cand_<sha256("ingestion-candidate:v1", source_id, content_hash)>
```

`source_id` 表示逻辑来源，`document_uid` 使用独立 namespace 表示逻辑文档；Phase 9
采用一源一文档语义，但没有让两个字段相等或共享前缀。GitHub commit SHA、Web ETag
等放在 `version`/metadata：version 变化但正文不变时，`content_hash` 与 candidate ID
不变；正文变化时才产生新 candidate。

## Canonical source URI

- Local：绝对路径或 `file:` URI，canonical 结果为本地绝对 `file://` URI；拒绝
  query、fragment、远程 host 与凭据。
- GitHub：`github://owner/repository/path`；owner/repository 小写，文件 path 保留
  大小写；拒绝 query、fragment、凭据和相对路径段。
- Web/Firecrawl：仅 `http/https`；scheme/host 小写，移除 fragment 和默认端口，
  保留 path 与完整业务 query。包含 token、API key、签名参数或 userinfo 的 URI
  fail closed。
- Upload：`upload://namespace/id`；拒绝 query、fragment 与凭据。

## Content normalization 与 secret 边界

进入文档合同前按固定顺序处理完整正文：

1. 复用 `app.common.redaction` 的权威敏感信息规则；
2. Unicode NFC；
3. `CRLF/CR → LF`；
4. 删除每行尾部空格/Tab 和文档尾部空白；
5. 对结果的完整 UTF-8 文本计算 SHA-256。

这不是摘要或语义清洗：不会翻译、改写、删除 prompt injection 或重排段落。正文
若含 `Ignore previous instructions` 等文本，它仍是普通 `content`。发生 credential
脱敏时 `content_redacted=true`，因此 hash 明确标识的是安全存储版本，不是假装等同
于未脱敏原文。

Metadata 只接受 `str/int/float/bool/None/list/dict`，拒绝 bytes、SecretStr、client、
response、Message、Tool、文件句柄、NaN/Infinity 等对象。限制为：最大深度 6、总键数
100、单容器 100 项、key 128 字符、字符串 2000 字符、整体 32768 UTF-8 bytes。
通过验证后再使用同一权威规则脱敏；不会 silent truncate。

`repr(KnowledgeDocument)` 隐藏正文和 metadata；`IngestionCandidateSummary` 只包含
title、source、hash、content length、trust、policy、status 与 approval status。

## Policy

默认字符上限 `INGESTION_MAX_CONTENT_CHARS=1_000_000`，按规范化、脱敏后的 Python
字符数计算；超过上限直接 DENY，不截断。

| 条件 | 决策 | 初始 candidate / approval |
| --- | --- | --- |
| 空正文 | DENY | rejected / rejected |
| 超大正文 | DENY | rejected / rejected |
| deployment 不支持该 source | DENY | rejected / rejected |
| `local_file + trusted_internal` | ALLOW | approved / not_required |
| 外部 source + `trusted_external` | ALLOW | approved / not_required |
| 外部 source + `untrusted_external` | REQUIRE_APPROVAL | pending / pending |
| 外部 source 错标为 `trusted_internal` | DENY | rejected / rejected |

GitHub、Web、Firecrawl 和 Upload 的 factory 默认都是 `untrusted_external`。可信外部
来源必须由 adapter/deployment policy 明确标为 `trusted_external`；GitHub 仓库名称
不会在通用合同中被硬编码为可信。

Policy 输出 `IngestionPolicyDecision`，decision 为 `allow`、`require_approval` 或
`deny`，reason 使用固定 Enum code，不接收正文或自由文本。Policy 不写 repository、
不调用 LLM/MCP/Web、不访问网络。

## Candidate、审批与状态迁移

`IngestionCandidate` 同样是 `extra="forbid"`、`frozen=True` 的 Pydantic model，
`schema_version=1`。`requested_by` 是由 `principal_id`、`principal_type`、`source`
组成的结构化 `IngestionRequester`，不是任意聊天文本。

Phase 9 使用 `InMemoryCandidateRepository`：

- 相同 `source_id + content_hash` 重复 propose 返回首次保存的同一逻辑 candidate；
- 同一 source 内容变化会创建新 candidate；
- 不建立 SQLite migration、source registry 或 persistent audit subsystem。

合法显式迁移只有：

```text
pending → approved
pending → rejected
```

`approved`、`rejected` 等 terminal 状态不能直接反转或回到 pending；需要改变内容并
重新 propose。`superseded` 只作为 contract 扩展状态保留，Phase 9 Service 不产生
该状态。Knowledge ingestion approval 与 MCP tool execution approval 是两个独立
domain，不共享持久状态。

## 明确不做的事

Phase 9 不修改现有 `app/rag/ingest.py` 的读取、清洗、split、顺序 `doc_N` 或全局
`chunk_id`，不修改 manifest、Chroma record 格式、Retrieval、Reranker、Prompt、
Evidence Gate、Citation、Agent loop 或 Evaluation ground truth。

Phase 10 才设计 source registry、stable chunk UID、增量 Chroma/BM25、tombstone、
atomic swap、rollback 和 live reload。Phase 11 才连接 GitHub/Firecrawl MCP、recrawl、
ETag polling 与 branch sync。

Phase 8B 的 GitHub MCP serializer 问题仍为已知项：

```text
github_mcp_serializer_issue: BLOCKER_FOR_PHASE_11
github_mcp_serializer_issue: NOT_BLOCKER_FOR_PHASE_10
```

Phase 9 没有修改 `app/tools/mcp/serialization.py`、bridge 或任何 live adapter。
