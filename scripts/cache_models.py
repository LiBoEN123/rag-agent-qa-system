import sys
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import ensure_model_cache_dir, get_settings


def cache_model(model_name: str, cache_dir: Path) -> None:
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        print(f"[model-cache] skip local path: {model_path}", flush=True)
        return

    print(f"[model-cache] downloading {model_name} -> {cache_dir}", flush=True)
    cached_path = snapshot_download(
        repo_id=model_name,
        cache_dir=str(cache_dir),
        local_files_only=False,
    )
    print(f"[model-cache] cached {model_name}: {cached_path}", flush=True)


def main() -> None:
    """读取应用配置并依次缓存嵌入与重排模型。"""
    settings = get_settings()
    cache_dir = ensure_model_cache_dir(settings)
    print(f"[model-cache] cache_dir={cache_dir}", flush=True)
    for role, model_name in (
        ("embedding", settings.embedding_model_name),
        ("rerank", settings.rerank_model_name),
    ):
        try:
            cache_model(model_name, cache_dir)
        except Exception as exc:
            print(f"[model-cache] failed {role} model={model_name}: {exc}", flush=True)
            raise
    print("[model-cache] done", flush=True)


if __name__ == "__main__":
    main()
