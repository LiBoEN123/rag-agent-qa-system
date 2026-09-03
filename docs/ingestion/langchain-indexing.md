# LangChain-native Incremental Indexing

Phase 10 将 Local Knowledge 的持久写入迁移到 LangChain 官方 Indexing API。Phase 9
继续拥有 provenance、trust、policy 和 approval；RAG 侧只消费 approved candidate。

```text
Local loader / cleaner / splitter
        ↓
KnowledgeDocument → IngestionCandidate(approved)
        ↓
LangChain Document(source_id, position, provenance)
        ↓
index(cleanup="full", source_id_key="source_id", key_encoder="sha256")
        ├─ SQLRecordManager: hash、unchanged 检测、cleanup 记录
        └─ langchain-chroma: vector add/delete → mini_rag_chunks_v2
        ↓
从最终 v2 corpus 重建 BM25 与 runtime compatibility IDs
        ↓
发布现有 Hybrid Retrieval runtime
```

职责边界：LangChain 拥有 document hash、重复/变化检测和 cleanup；官方
`SQLRecordManager` 使用 `data/manifest/record_manager.sqlite3`，Local namespace 为
`rag-agent/local/v1`；`langchain-chroma` 拥有向量写入。项目只保留 Phase 9 admission、
现有 splitter、embedding/reranker 配置、BM25 全量内存重建和现有读取/检索接口。

Local loader 每轮返回完整 authoritative corpus，因此使用 `cleanup="full"`。Local
namespace 不与未来 GitHub/Firecrawl namespace 共享，完整 cleanup 不会误删外部知识。
BM25 不做增量实现；`index()` 成功后从最终 Chroma corpus 全量重建，未变化文档不会
重新 embedding。

首次迁移写入 `mini_rag_chunks_v2`，允许一次全量 embedding；旧
`mini_rag_chunks` 不删除，作为 comparison/rollback artifact。正常启动始终执行
incremental sync：第二次相同 corpus 由 RecordManager 全部 skip。embedding 模型、
chunk 参数、namespace/collection 或索引版本变化，以及 RecordManager/collection 丢失时
才 force rebuild。只改 reranker 不触发重新 embedding。

整数 `doc_id` / `chunk_id` 只用于 runtime/debug/evaluation。它们在读取最终 corpus 后按
`source_path + position` 重新分配，不写入 LangChain hash；长期 provenance 使用
`source_id`、`document_uid`、`source_uri` 和 `position`。

恢复保持简单：普通 sync 失败时报告错误且不发布新内存 runtime；下一次启动重试。
显式 force rebuild 只清理 v2 collection 和 Local RecordManager namespace，再由框架
全量 index。没有跨 SQLite/Chroma 事务、rollback journal 或自定义 registry。

Phase 10 不新增 Agent Tool/Public API。

Phase 11B 在该边界上增加两个独立 external namespace：

- GitHub file：`rag-agent/github/v1`；
- Firecrawl page：`rag-agent/firecrawl/v1`。

二者继续写入统一 `mini_rag_chunks_v2` collection，但只对显式 approved candidate 调用
`index(cleanup="incremental")`。因此更新一个 source 只清理相同 `source_id` 的旧 chunks，
不会删除 Local 或其他 external source；Local authoritative sync 仍默认
`cleanup="full"`，且只操作 Local namespace 已记录的 keys。成功后直接复用现有
`load_index_cache()` 从最终 Chroma corpus 重建 BM25/runtime，不新增 registry、手写
Vector CRUD 或增量 BM25。
