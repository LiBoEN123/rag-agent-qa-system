"""轻量进程存活检查 Router。"""

from fastapi import APIRouter


def create_health_router() -> APIRouter:
    """创建不访问数据库、模型、RAG 或工具的健康检查路由。"""
    router = APIRouter()

    # 返回服务存活状态的统一响应。
    @router.get("/health")
    def health() -> dict:
        return {
            "code": 0,
            "message": "success",
            "data": {"status": "ok"},
        }

    return router


__all__ = ["create_health_router"]
