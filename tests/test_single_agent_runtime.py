"""LangGraph 是唯一 Agent Runtime 的静态回归测试。"""

from pathlib import Path

from app.api.schemas import ChatRequest
import app.services as services
import app.services.chat_service as chat_service


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_uses_single_langgraph_runtime():
    """请求模型、服务依赖与导出均只暴露 LangGraph Runtime。"""
    selector_path = PROJECT_ROOT / "app" / "services" / ("runtime_" + "selection.py")
    selector_export = "select_agent_" + "runtime"
    assert not selector_path.exists()
    assert not hasattr(services, selector_export)
    fields = chat_service.ChatServiceDependencies.__dataclass_fields__
    runtime_fields = {name for name in fields if name.startswith("run_agent_")}
    assert runtime_fields == {"run_agent_langgraph"}
    assert "runtime" not in ChatRequest.model_fields
