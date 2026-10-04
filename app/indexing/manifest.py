"""Local corpus 快照与显式重建判断。"""

import json

from app.config import AppSettings, get_settings
from app.ingestion.files import collect_supported_files
from app.indexing.langchain import LOCAL_RECORD_NAMESPACE, record_manager_path
from app.storage.vector import create_vector_store


def _manifest_path(settings: AppSettings):
    """返回当前配置对应的索引 manifest 路径。"""
    return settings.index_store_dir / "index_manifest.json"


def build_file_snapshot(data_dir: str):
    """记录知识库文件的路径、大小和修改时间快照。"""
    files = collect_supported_files(data_dir)
    snapshot = []

    for file_path in files:
        stat = file_path.stat()
        snapshot.append({
            "path": str(file_path.resolve()),
            "size": stat.st_size,
            "time": stat.st_mtime
        })

    snapshot.sort(key=lambda x: x["path"])
    return snapshot


def build_current_manifest(
    data_dir: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    *, settings: AppSettings | None = None,
):
    """生成当前索引构建配置和文件快照的 manifest。"""
    settings = settings or get_settings()
    chunk_size = settings.chunk_size if chunk_size is None else chunk_size
    chunk_overlap = (
        settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    )
    return {
        "pipeline_version": 6,
        "embedding_model": settings.embedding_model_name,
        "record_manager_namespace": LOCAL_RECORD_NAMESPACE,
        "chroma_collection": settings.chroma_index_collection_name,
        "build_config": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        "files": build_file_snapshot(data_dir),
    }


def load_manifest(settings: AppSettings | None = None):
    """从磁盘读取历史索引 manifest。"""
    path = _manifest_path(settings or get_settings())
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_manifest(manifest: dict, settings: AppSettings | None = None):
    """将当前 manifest 写入磁盘。"""
    path = _manifest_path(settings or get_settings())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def needs_rebuild(
    data_dir: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    *, settings: AppSettings | None = None,
) -> tuple[bool, str]:
    """判断是否需要清空 v2 索引；普通文件变化只做增量 sync。"""
    settings = settings or get_settings()
    old_manifest = load_manifest(settings)
    current_manifest = build_current_manifest(data_dir, chunk_size, chunk_overlap, settings=settings)

    if old_manifest is None:
        return True, "没有找到 v2 manifest，需要首次 bootstrap"

    if old_manifest.get("pipeline_version") != current_manifest.get("pipeline_version"):
        return True, "pipeline_version 变化，需要 force rebuild"

    if old_manifest.get("embedding_model") != current_manifest.get("embedding_model"):
        return True, "embedding 模型变化，需要 force rebuild"

    if (
        old_manifest.get("record_manager_namespace")
        != current_manifest.get("record_manager_namespace")
        or old_manifest.get("chroma_collection")
        != current_manifest.get("chroma_collection")
    ):
        return True, "index namespace 或 collection 变化，需要 force rebuild"

    if old_manifest.get("build_config") != current_manifest.get("build_config"):
        return True, "chunk 参数变化，需要 force rebuild"

    if not record_manager_path(settings).is_file():
        return True, "SQLRecordManager 不存在，需要 force rebuild"

    try:
        create_vector_store(None, settings, create_collection_if_not_exists=False)
    except Exception:
        return True, "v2 Chroma collection 不可用，需要 force rebuild"

    if old_manifest.get("files") != current_manifest.get("files"):
        return False, "Local corpus 变化，将执行 LangChain 增量同步"

    return False, "配置与 corpus 未变化，将验证 unchanged skip"


