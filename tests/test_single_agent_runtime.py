"""LangGraph 是唯一 Agent Runtime 的静态回归测试。"""

from pathlib import Path

from app.api.schemas import ChatRequest
from app.config import Settings
from app.services import chat_service
import app.services as services


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_configuration_switch_is_absent():
    removed_name = "AGENT_" + "RUNTIME"
    assert not hasattr(Settings, removed_name)
    assert removed_name not in (PROJECT_ROOT / ".env.example").read_text(
        encoding="utf-8"
    )


def test_runtime_selector_module_and_export_are_absent():
    selector_path = PROJECT_ROOT / "app" / "services" / ("runtime_" + "selection.py")
    selector_export = "select_agent_" + "runtime"
    assert not selector_path.exists()
    assert not hasattr(services, selector_export)


def test_chat_service_has_only_langgraph_runtime_dependency():
    fields = chat_service.ChatServiceDependencies.__dataclass_fields__
    runtime_fields = {name for name in fields if name.startswith("run_agent_")}
    assert runtime_fields == {"run_agent_langgraph"}


def test_chat_api_has_no_runtime_request_parameter():
    assert "runtime" not in ChatRequest.model_fields
