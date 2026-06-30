import sys
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings


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
    cache_dir = Settings.ensure_model_cache_dir()
    print(f"[model-cache] cache_dir={cache_dir}", flush=True)
    for role, model_name in (
        ("embedding", Settings.EMBEDDING_MODEL_NAME),
        ("rerank", Settings.RERANK_MODEL_NAME),
    ):
        try:
            cache_model(model_name, cache_dir)
        except Exception as exc:
            print(f"[model-cache] failed {role} model={model_name}: {exc}", flush=True)
            raise
    print("[model-cache] done", flush=True)


if __name__ == "__main__":
    main()
