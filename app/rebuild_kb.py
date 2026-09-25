"""显式同步知识索引；不导入或初始化 QA runtime。"""

import argparse
import json

from app.config import get_settings, validate_rag_environment
from app.indexing.service import IndexingService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-rebuild", action="store_true", help="从持久候选历史重建所有来源")
    args = parser.parse_args()
    settings = get_settings()
    validate_rag_environment(settings)
    try:
        result = IndexingService(settings=settings).sync_local_corpus(force_rebuild=args.force_rebuild)
    except Exception as exc:
        parser.exit(1, f"索引同步失败：{exc}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
