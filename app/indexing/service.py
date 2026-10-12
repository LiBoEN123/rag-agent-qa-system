"""知识索引写侧；只操作持久存储，不初始化或刷新 QA runtime。"""

from langchain_core.embeddings import Embeddings

from app.config import AppSettings, get_settings
from app.indexing.langchain import (
    EXTERNAL_RECORD_NAMESPACE_BY_SOURCE_TYPE,
    LOCAL_RECORD_NAMESPACE,
    candidate_to_langchain_documents,
    create_record_manager,
    load_local_langchain_documents,
    sync_documents,
)
from app.indexing.manifest import build_current_manifest, needs_rebuild, save_manifest
from app.ingestion.repository import CandidateRepository
from app.storage.ingestion import SQLiteCandidateRepository
from app.storage.vector import create_vector_store, load_embedding_model


class IndexingError(RuntimeError):
    """保持管理 API 的安全错误码，不包含来源正文。"""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class IndexingService:
    """复用 LangChain Indexing API、Chroma 与既有 candidate 合同。"""

    def __init__(
        self, *, settings: AppSettings | None = None,
        candidates: CandidateRepository | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._candidates = candidates
        self._embeddings = embeddings

    def _embedding_model(self) -> Embeddings:
        if self._embeddings is None:
            try:
                self._embeddings = load_embedding_model(self._settings)
            except Exception:
                raise IndexingError("SOURCE_EMBEDDINGS_UNAVAILABLE", "embedding model 无法加载") from None
        return self._embeddings

    def _candidate_and_namespace(self, candidate_id: str):
        repository = self._candidates
        if repository is None:
            repository = SQLiteCandidateRepository(self._settings.sqlite_db_path)
        candidate = repository.get(candidate_id)
        if candidate is None:
            raise IndexingError("SOURCE_CANDIDATE_NOT_FOUND", "candidate 不存在")
        namespace = EXTERNAL_RECORD_NAMESPACE_BY_SOURCE_TYPE.get(candidate.document.source_type)
        if namespace is None:
            raise IndexingError("SOURCE_TYPE_UNSUPPORTED", "不支持该 external source")
        return candidate, namespace

    def index_approved(self, candidate_id: str) -> dict[str, int]:
        """只将 approved external candidate 增量写入对应 namespace。"""
        candidate, namespace = self._candidate_and_namespace(candidate_id)
        documents = candidate_to_langchain_documents(
            candidate, chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        return sync_documents(
            documents, self._embedding_model(), settings=self._settings,
            namespace=namespace, cleanup="incremental",
        )

    def remove_indexed(self, candidate_id: str) -> dict[str, int]:
        """按 logical source 撤库；同时修复向量与 RecordManager 的部分缺失。"""
        candidate, namespace = self._candidate_and_namespace(candidate_id)
        source_id = candidate.document.source_id
        manager = create_record_manager(self._settings, namespace=namespace)
        keys = manager.list_keys(group_ids=[source_id])
        store = create_vector_store(None, self._settings)
        where = {"source_id": source_id}
        ids = store.get(where=where, include=[])["ids"]
        if ids:
            store.delete(where=where)
        if keys:
            manager.delete_keys(keys)
        return {
            "num_added": 0, "num_updated": 0,
            "num_deleted": len(ids), "num_skipped": 0,
        }

    def sync_local_corpus(self, *, force_rebuild: bool = False) -> dict[str, int]:
        """显式同步 Local；配置变化或 force 时恢复完整 authoritative corpus。"""
        settings = self._settings
        data_dir = str(settings.rag_file_path)
        rebuild_required, _reason = needs_rebuild(data_dir, settings=settings)
        force_rebuild = force_rebuild or rebuild_required
        documents_by_namespace = {LOCAL_RECORD_NAMESPACE: load_local_langchain_documents(
            data_dir, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
        )}
        if force_rebuild:
            documents_by_namespace.update(
                (namespace, []) for namespace in EXTERNAL_RECORD_NAMESPACE_BY_SOURCE_TYPE.values()
            )
            candidates = SQLiteCandidateRepository(
                settings.sqlite_db_path
            ).load_latest_successfully_indexed()
            for candidate in candidates:
                namespace = EXTERNAL_RECORD_NAMESPACE_BY_SOURCE_TYPE.get(candidate.document.source_type)
                if namespace is None:
                    raise ValueError("成功 index candidate 的 external source type 不支持恢复")
                documents_by_namespace[namespace].extend(candidate_to_langchain_documents(
                    candidate, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
                ))

        embeddings = self._embedding_model()
        if force_rebuild:
            # 全部候选已验证、切分后才能清空共享 collection。
            create_vector_store(embeddings, settings).delete_collection()
            for namespace in documents_by_namespace:
                manager = create_record_manager(settings, namespace=namespace)
                keys = manager.list_keys()
                if keys:
                    manager.delete_keys(keys)
        summary: dict[str, int] = {}
        for namespace, documents in documents_by_namespace.items():
            result = sync_documents(documents, embeddings, settings=settings, namespace=namespace)
            for key, count in result.items():
                summary[key] = summary.get(key, 0) + count
        save_manifest(build_current_manifest(data_dir, settings=settings), settings)
        return summary
