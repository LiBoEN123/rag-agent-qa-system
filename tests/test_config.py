"""应用配置与 LangGraph Checkpointer 配置测试。"""

from app.config import Settings


def test_agent_runtime_has_no_configuration_switch():
    """Agent Runtime 固定后，Settings 不再暴露选择项。"""
    removed_name = "AGENT_" + "RUNTIME"
    assert not hasattr(Settings, removed_name)


def test_llm_transport_has_no_configuration_switch():
    """应用内部固定使用统一 LangChain transport。"""
    removed_name = "LLM_CLIENT" + "_MODE"
    assert not hasattr(Settings, removed_name)


def test_settings_validate_keeps_llm_rag_and_checkpointer_configuration(
    monkeypatch,
    tmp_path,
):
    """配置校验不访问模型、工具、检索或其他外部服务。"""
    monkeypatch.setattr(Settings, "LANGGRAPH_CHECKPOINTER", "disabled")
    monkeypatch.setattr(Settings, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(Settings, "RAG_FILE_PATH", tmp_path)

    Settings.validate()

    assert Settings.LANGGRAPH_CHECKPOINTER == "disabled"
    assert Settings.LLM_API_KEY == "test-key"
    assert Settings.RAG_FILE_PATH == tmp_path
