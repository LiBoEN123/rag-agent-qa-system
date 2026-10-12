# LangChain-native Incremental Indexing

`app.indexing` 是独立写侧，`app.rag.runtime` 是只读查询侧。共享持久 Chroma 配置与
embedding factory 位于 `app.storage.vector`，两侧不导入对方。

```text
Local files / approved external candidate
→ ingestion parsing / admission
→ LangChain Document（source_id、position、provenance）
→ IndexingService → LangChain index → SQLRecordManager + Chroma

QA startup → load existing Chroma → chunks / BM25 / query embedding / CrossEncoder
```

准备好本地 embedding/rerank 模型缓存后，同步知识索引，再启动 QA：

```bash
python -m app.rebuild_kb
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

该既有 CLI 默认增量同步；需要强制重建时加 `--force-rebuild`。CLI 只调用 production
`IndexingService`，不初始化 Chat/Agent/RAG runtime。独立 CLI 更新后重启 QA 加载新快照。
QA 启动不检查 manifest、不扫描 `data/`、不写 Chroma 或 RecordManager。

`app.indexing.langchain` 复用官方 `langchain_core.indexing.index`、
`SQLRecordManager`、`langchain_chroma.Chroma` 和现有 splitters。Local 返回完整 corpus，
使用 `cleanup="full"`；外部 approved candidate 使用 `cleanup="incremental"`。
`source_id_key="source_id"`、`key_encoder="sha256"` 保持不变。未变化文档由框架 skip。

共用 `mini_rag_chunks_v2`，RecordManager 位于 `data/manifest/record_manager.sqlite3`：

| Source | Namespace |
| --- | --- |
| Local | `rag-agent/local/v1` |
| GitHub | `rag-agent/github/v1` |
| Firecrawl | `rag-agent/firecrawl/v1` |
| Upload | `rag-agent/upload/v1` |

普通 Local 同步只清理自己的 namespace。`app.indexing.manifest` 拥有文件快照、
manifest 和重建判断；embedding 模型、chunk 参数、namespace/collection、索引版本
变化或存储缺失触发重建。Reranker 配置不参与写侧。旧 `mini_rag_chunks` collection 保留。

Force rebuild 先读取 Local 与 SQLite 中各 source 最新成功 INDEX 对应候选，并验证、
切分完整语料，再清空共享 v2 collection 和各 namespace keys。仅批准、未成功索引的
候选不会恢复；最新成功 REMOVE 排除该来源，FAILED/RUNNING 不改变 authoritative state。
恢复使用已持久化全文与结构，不重抓 GitHub/Firecrawl、不重新解析上传文件。

INDEX/REMOVE 只返回 added/updated/deleted/skipped。组合 API 在记录持久操作 SUCCESS
后，由 `app.api.application` 回调选择刷新查询缓存；刷新失败不会撤销该成功历史。
移除最后一份资料后，查询侧严格验证 collection 为空，再发布空 runtime。

没有跨 SQLite/Chroma 事务或自动回滚；写入中途失败可能留下部分数据，可显式重试或重建。
manifest 只在整个同步成功后保存。查询 runtime 不受独立写操作直接修改。

现有 Hybrid Retrieval、CrossEncoder、证据 identity 和结构化表格切分算法保持不变。
整数 `doc_id` / `chunk_id` 在读取后按 `source_path + position` 分配，不写入 LangChain hash；
稳定 provenance 使用 `source_id`、`document_uid`、`source_uri` 和 `position`。
