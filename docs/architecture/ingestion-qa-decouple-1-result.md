# INGESTION-QA-DECOUPLE-1

## STATUS

PASS_WITH_FINDINGS。要求的模块、运行生命周期和持久存储边界已落实；已执行测试中的失败项已全部定向修正并复跑通过。未扩展到真实模型评测或跨库事务框架。

## BRANCH / HEAD

- branch: `refactor/tool-output-models`
- HEAD: `a64b4add60927753aca4091346242315d1ea1498`
- 起始工作区干净。未切换 main、未创建分支、未 commit、未 push、未 rebase/reset/修改历史。
- 另对被 Git 忽略的本地配置进行了精确清理：`.env` 删除已失效的启动重建开关/注释；MCP local JSON 只删除 Firecrawl 的旧 promotion 字段，其余内容不变，配置 schema 校验通过。未记录或输出凭据。

`git status --short`：

```text
 M .env.example
 M .importlinter
 M DEPLOY.md
 M README.md
 M app/agent/final_response.py
 M app/agent/framework_middleware.py
 M app/agent/tool_events.py
 M app/api/application.py
 M app/api/ingestion.py
 M app/api/lifecycle.py
 M app/api/responses.py
 M app/config.py
 M app/rag/chroma.py
 D app/rag/indexing.py
 D app/rag/ingest.py
 D app/rag/parsing.py
 D app/rag/pipeline.py
 M app/rag/policy.py
 D app/rag/scoring.py
 D app/rag/store.py
 D app/rag/structured_chunking.py
 M app/rebuild_kb.py
 M app/sources/__init__.py
 M app/sources/firecrawl.py
 M app/sources/service.py
 M app/sources/upload.py
 M app/storage/database.py
 M app/storage/ingestion.py
 M app/storage/memory.py
 M app/tools/audit.py
 M app/tools/document_search.py
 M app/tools/knowledge_base.py
 M app/tools/mcp/__init__.py
 M app/tools/mcp/bridge.py
 M app/tools/mcp/config.py
 M app/tools/mcp/discovery.py
 M app/tools/mcp/serialization.py
 M docs/architecture/current-system.md
 M docs/architecture/module-boundaries.md
 M docs/ingestion/governed-source-adapters.md
 M docs/ingestion/langchain-indexing.md
 M docs/ingestion/operations-api.md
 M docs/ingestion/structured-documents.md
 M docs/tools/mcp-client.md
 M scripts/evaluation/agent.py
 M scripts/evaluation/rag_predictions.py
 M scripts/evaluation/runner.py
 M scripts/evaluation/table_diagnostics.py
 M tests/test_agent_knowledge_tool_integration.py
 M tests/test_agent_streaming.py
 M tests/test_api_application.py
 M tests/test_application_storage.py
 M tests/test_chat_service_agent_integration.py
 M tests/test_chat_service_streaming.py
 M tests/test_chunk_context_tool.py
 M tests/test_config.py
 M tests/test_config_import_safety.py
 M tests/test_create_agent_builder.py
 M tests/test_document_search_tools.py
 M tests/test_evaluation_rag_predictions_v3.py
 M tests/test_evaluation_runner.py
 M tests/test_evaluation_smoke.py
 M tests/test_evaluation_table_diagnostics.py
 M tests/test_external_source_indexing.py
 M tests/test_ingestion_api.py
 M tests/test_ingestion_service.py
 M tests/test_ingestion_storage.py
 M tests/test_iterative_agent_loop.py
 M tests/test_mcp_agent_integration.py
 M tests/test_mcp_bridge.py
 M tests/test_mcp_discovery.py
 M tests/test_module_boundaries.py
 M tests/test_rag_indexing.py
 D tests/test_rag_ingest.py
 M tests/test_rag_retrieval.py
 D tests/test_rag_store.py
 M tests/test_source_service.py
 M tests/test_sqlite_store.py
 M tests/test_structured_chunking.py
 M tests/test_structured_ingestion.py
 M tests/test_structured_parsing.py
 M tests/test_tool_governance_integration.py
 D tests/test_tool_knowledge_promotion.py
 M tests/test_tool_registry.py
?? app/common/numeric.py
?? app/indexing/
?? app/ingestion/files.py
?? app/ingestion/parsing.py
?? app/rag/runtime.py
?? app/storage/vector.py
?? docs/architecture/ingestion-qa-decouple-1-result.md
?? tests/test_ingestion_files.py
?? tests/test_rag_runtime.py
?? tests/test_read_write_independence.py
```

## COUPLING FOUND

| Coupling | Before | After |
| --- | --- | --- |
| rag pipeline write/read | 同时同步、恢复候选和发布查询状态 | IndexingService 写持久索引；rag.runtime 读取 |
| source service → rag runtime | 读取 QA embedding global；写后 reload | Source 只抓取、校验、proposal |
| agent → ingestion promotion | MCP 成功后创建候选 | Tool Result 直接回到 Agent |
| chat storage → ingestion schema | Chat CRUD 调用全量 init_db/migration | 独立 init_session_storage / init_ingestion_storage |
| QA startup → index sync | 扫描、解析、写索引、重建 runtime | 只读取已有索引 |
| rag chroma → indexing factory | 查询侧从写侧导入 Chroma factory | 共用 storage.vector |
| Source → query 间接导入 | audit 数值 helper、MCP 包级导出牵入 QA catalog | helper 下沉 common；删除无消费者的 MCP re-export |

## IMPLEMENTED

- `app.indexing` 使用既有 LangChain index、SQLRecordManager、Chroma 和 splitters；显式注入 settings/candidate repository/embeddings，默认 embedding factory 也由写侧独立加载。
- `IndexingService.index_approved/remove_indexed/sync_local_corpus` 不引用或刷新 QA；approved-only、namespace cleanup、最新成功 INDEX/REMOVE 恢复规则保持。
- API 在记录持久操作 SUCCESS 后调用 composition root 注入的可选刷新回调；刷新失败日志化，不把成功 REMOVE 改记失败。
- 复用 `python -m app.rebuild_kb`：默认增量同步，`--force-rebuild` 显式全量重建，不新增第二个 CLI。Evaluation 初始化默认读取，已有显式 force 参数调用写侧；未改 scoring。
- 查询 runtime 合并旧 pipeline/store 的读取和状态职责；不保留兼容壳。
- SQLite 文件不拆分；候选 schema/ORM/migration 归 storage.ingestion，Chat schema 归 storage.memory，共用连接工厂。
- Firecrawl 原有只读 scrape 的参数限制与 promotion 脱钩，继续支持显式来源抓取；权限、审批、序列化、截断、审计保持。
- README、DEPLOY、当前架构、边界、索引、来源、结构化文档和 MCP 文档同步。

## DELETED

- 旧文件路径：`app/rag/{pipeline,store,indexing,ingest,parsing,structured_chunking,scoring}.py`。
- 死函数/生命周期：`_sync_and_build_runtime`、`rebuild_and_persist_index`、`initialize_rag`、`StartupDependencies`、`read_pdf_text/read_docx_text/read_file_text`、`ExternalKnowledgeSourceService.approve`、`promote_tool_result`、`index_cache_exists/get_index_cache_source`、混合 `init_db`。
- 配置/metadata：自动 promotion 开关及配置校验、promotion trace/status、仅为 promotion 保留正文的 serializer 参数、`AUTO_REBUILD_ON_STARTUP`。
- 导出：Sources 的索引 namespace re-export、MCP 包级无消费者的 runtime/provider re-export；没有增加兼容 wrapper。
- 测试：删除 `test_tool_knowledge_promotion.py` 和死 reader/wrapper/旧启动行为测试。`test_rag_ingest.py` 剩余 live-reader 测试迁至 `test_ingestion_files.py`；`test_rag_store.py` 迁至 `test_rag_runtime.py`。

## MOVED RESPONSIBILITIES

| Responsibility | Before | After |
| --- | --- | --- |
| 索引写入、Local admission、RecordManager | rag.indexing / pipeline | indexing.langchain / service |
| approved index / remove | sources.service | indexing.service |
| manifest / rebuild decision | rag.store | indexing.manifest |
| 文件读取/清洗 | rag.ingest | ingestion.files |
| Docling parsing | rag.parsing | ingestion.parsing（依赖惰性导入） |
| 结构化 chunking | rag.structured_chunking | indexing.chunking |
| Chroma / embedding factory | rag.indexing / store | storage.vector |
| 查询加载、BM25、runtime state | rag.pipeline / store | rag.runtime |
| ingestion ORM / migration | storage.database | storage.ingestion |
| sessions/messages schema | storage.database | storage.memory |
| 安全浮点转换 | rag.scoring | common.numeric |

## ARCHITECTURE BEFORE / AFTER

```text
Before:
QA startup → rag.pipeline → local admission + persistent writes + query globals
Sources → rag.pipeline.model / reload
Agent MCP → candidate creation
Chat CRUD → ingestion schema migration

After:
FastAPI / CLI composition root
  ├─ Sources → Ingestion proposal/approval → IndexingService → persistent index
  └─ ChatService → Agent → KB tools → rag.runtime → persistent index

Combined API: successful persistent INDEX/REMOVE → optional query refresh
```

## IMPORT BOUNDARIES

`app.indexing` 加入层次；基础层、ingestion domain、HTTP/settings 门禁纳入新包。
新增 `rag-read-side-independence`、`indexing-write-side-independence`、
`sources-do-not-index`、`agent-no-ingestion-side-effects`、`chat-storage-is-independent`。
`lint-imports --no-cache`：12 kept，0 broken，未增加 ignore 或放宽间接依赖规则。
另外以新进程 import blocker 验证真实 package 初始化，不仅依赖静态图。

## DEAD CODE CLEANUP

reader 包装没有 production 消费者，PDF/DOCX 直接走结构 parser；删除旧 reader 及其自测。
正式审批只走 IngestionService，删除 SourceService 重复入口。
删除自动 promotion 的完整调用/配置/metadata 链；保留显式 INGESTION serializer profile。
删除无消费者的缓存来源 wrapper、旧 startup sync wrapper、Sources namespace 导出。
MCP 包级导出无人使用却间接加载 QA catalog，删除该 re-export；真实调用继续导入具体模块。
职责搬迁本身不计为 dead-code reduction。

## LOC DELTA

Python physical LOC（含空行/注释；基线为 HEAD，after 包含新文件）：

| Scope | Before | After | Delta |
| --- | ---: | ---: | ---: |
| app | 16687 | 16331 | -356 |
| app/rag | 1891 | 891 | -1000 |
| app/indexing | 0 | 726 | +726 |
| app/ingestion | 1519 | 1674 | +155 |
| app/sources | 836 | 650 | -186 |
| app/storage | 584 | 632 | +48 |
| tests | 28698 | 28192 | -506 |

## TESTS

采用 `.venv`，临时 Chroma/SQLite、确定性 embeddings/回答模型和本地 Docling；没有改生产持久索引。
以下是实际执行记录，各批有重叠，不能相加为唯一测试数量：

| Batch | Targeted tests | Result |
| --- | --- | --- |
| 写侧首批 | test_rag_indexing、test_external_source_indexing、test_structured_ingestion、test_structured_parsing、test_structured_chunking、test_source_service、test_source_adapters、test_upload_source、test_ingestion_files | 193 passed / 5 failed；失败项后续修正并通过 |
| 读侧/存储首批 | test_rag_chroma、test_rag_retrieval、test_rag_runtime、test_document_search_tools、test_chunk_context_tool、test_application_storage、test_session_store、test_api_application、test_ingestion_api、test_ingestion_storage、test_sqlite_store、test_module_boundaries | 188 passed / 38 failed；旧依赖/旧假服务与基线断言随后修正 |
| MCP/配置首批 | test_mcp_agent_integration、test_mcp_bridge、test_mcp_config、test_mcp_discovery、test_config、test_config_import_safety | 161 passed / 2 failed；启动路径新合同、旧 import-safety 断言随后修正 |
| 迁移回归 | test_rag_indexing、test_external_source_indexing、test_rag_retrieval、test_rag_runtime、test_application_storage、test_ingestion_api、test_module_boundaries | 141 passed / 1 failed；剩余 PDF 基线断言后续通过 |
| 独立性首批 | test_read_write_independence、test_config、test_config_import_safety、test_application_storage、test_mcp_agent_integration、test_agent_knowledge_tool_integration、test_ingestion_service | 118 passed / 2 failed；探针配置及旧 runtime 断言随后修正 |
| 精确失败项复跑 | 独立性、import safety、结构恢复、PDF 上传、组合刷新失败两种情况 | 5 passed / 2 failed；包级隐式 import、旧 row metadata 断言随后修正 |
| 最终边界批次 | test_read_write_independence、test_mcp_discovery、test_mcp_bridge、test_source_service、test_structured_ingestion、test_config_import_safety | **80 passed** |
| 最终集成批次 | 写侧 embedding 失败安全错误、test_mcp_agent_integration、test_api_application、test_chat_service_agent_integration、test_chat_stream_api、test_evaluation_rag_predictions_v3 | **79 passed** |
| 最终存储/解析批次 | test_application_storage、test_session_store、test_sqlite_store、test_structured_parsing、test_module_boundaries | **43 passed** |

原始 HEAD 的隔离 `git archive` 快照上执行两个疑似旧断言，均复现失败：合并说明行的
旧 chunk 文本/row metadata，以及 PDF 页标记后的旧换行预期。只修测试预期；
结构切分算法与 HEAD 逐字节相同，parser 只调整位置和惰性 import。

验收覆盖：

- 写侧新进程禁止导入 rag/agent/services，完成上传 proposal、approve、index、remove、Local sync/unchanged skip。
- 读侧新进程禁止导入 ingestion/indexing/sources/ingestion storage/RecordManager writer，读取上述索引、构建 BM25，通过正式 KB Tool + create_agent 回答。
- Combined API index/remove 正常刷新；刷新返回失败或抛异常都不改写已成功 REMOVE 的 authoritative history。
- 普通 MCP Agent 调用禁止创建 IngestionService candidate 或 SQLiteCandidateRepository，工具结果仍成功回 Agent。
- Chat 面对缺失/损坏 ingestion 历史仍可读写且不迁移其 schema；并发首次 Chat 写入只创建 session 表。

静态检查：changed Python files 的 Ruff、Import Linter、`compileall -q app`、`git diff --check` 均通过。
`python -m app.rebuild_kb --help` 通过。

Full pytest: NOT RUN。
Full Evaluation / Core Evaluation / RAG Evaluation: NOT RUN。
Reason: architecture/indexing boundary refactor; retrieval scoring and Agent answer semantics unchanged;
only implicit candidate side effects were intentionally removed.
真实 LLM/embedding Evaluation、真实 MCP/外网 smoke、frontend build：NOT RUN。

## BEHAVIOR CHECK

| Contract | Evidence |
| --- | --- |
| Hybrid retrieval unchanged | retrieval.py 与 HEAD 字节相同；真实临时 Chroma/BM25 回归通过 |
| CrossEncoder rerank unchanged | 查询加载/调用保持；重排与 KB Agent 集成通过 |
| Evidence IDs unchanged | tools/evidence.py 未改；chunk locator/evidence 合同回归通过 |
| Chat API unchanged | /chat/stream schema 未改；流式 HTTP/ChatService 集成通过 |
| Ingestion API unchanged | endpoint/request/response schema 保持；持久操作成功独立于可选刷新 |
| MCP normal tool behavior / approval unchanged | 普通工具、审批、权限、审计、截断测试通过；Firecrawl 只读参数约束保留 |
| Web SSRF unchanged | web_* 下载/安全实现未改 |
| Ingestion policy / structured chunking unchanged | domain contracts/service 与 chunking 算法对比 HEAD 字节相同 |

## REMAINING FINDINGS

- 原有 Chroma/SQLRecordManager/application SQLite 之间没有跨库事务。部分写入失败仍需显式重试或重建；本次不增加事务框架。
- CLI 更新不主动通知 QA；需要重启读侧加载新快照。组合应用只刷新当前进程；刷新失败时可能保留旧快照，已有日志提示，持久成功历史不撤销。
- 本次没有验证真实远程 MCP 或真实 LLM 效果；定向测试使用替身，不能替代线上验证。

## NEXT RECOMMENDATION

部署时按 README/DEPLOY 顺序显式同步索引，再启动/重启 QA。若未来需要多个 QA 进程自动同步快照，再单独设计刷新协调；本次未自动进入下一阶段。
