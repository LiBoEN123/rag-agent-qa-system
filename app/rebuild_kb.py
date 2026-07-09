"""知识库重建命令入口。

读取当前配置并触发文档索引的完整重建与持久化流程。
"""

from app.config import Settings
from app.rag import rebuild_and_persist_index


def main():
    """命令行重建知识库索引并在失败时退出非零状态。"""
    Settings.validate()
    ok, msg = rebuild_and_persist_index(
        data_dir=str(Settings.RAG_FILE_PATH),
        chunk_size=Settings.CHUNK_SIZE,
        chunk_overlap=Settings.CHUNK_OVERLAP,
    )
    print(msg)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
