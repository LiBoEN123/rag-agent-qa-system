"""知识库重建命令入口。

读取当前配置并触发文档索引的完整重建与持久化流程。
"""

from app.config import get_settings, validate_rag_environment
from app.rag import rebuild_and_persist_index


def main():
    """命令行重建知识库索引并在失败时退出非零状态。"""
    settings = get_settings()
    validate_rag_environment(settings)
    ok, msg = rebuild_and_persist_index(
        data_dir=str(settings.rag_file_path),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    print(msg)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
