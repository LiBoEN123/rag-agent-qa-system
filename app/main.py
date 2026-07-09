"""服务启动入口。

向 Uvicorn 暴露 FastAPI 应用，并保持启动命令 ``uvicorn app.main:app`` 稳定。
"""

import sys

from app.api import chat as _application

sys.modules[__name__] = _application
