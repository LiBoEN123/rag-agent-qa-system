"""FastAPI 服务启动入口。"""

from app.api.application import create_app


app = create_app()

__all__ = ["app"]
