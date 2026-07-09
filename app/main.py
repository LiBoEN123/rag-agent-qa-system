"""Stable ASGI entry point for ``uvicorn app.main:app``."""

import sys

from app.api import chat as _application

sys.modules[__name__] = _application
