"""只把工具名称交给 DeepEval；本地确定性评分，不创建模型客户端或云端 run。"""

import os

# Must precede DeepEval imports, including its config and telemetry modules.
os.environ.update(
    DEEPEVAL_TELEMETRY_OPT_OUT="YES",
    DEEPEVAL_DISABLE_DOTENV="1",
    DEEPEVAL_UPDATE_WARNING_OPT_IN="0",
    DEEPEVAL_FILE_SYSTEM="READ_ONLY",
    CONFIDENT_TRACING_ENABLED="NO",
)

from deepeval.metrics import ToolCorrectnessMetric  # noqa: E402
from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase, ToolCall  # noqa: E402


class _DisabledJudge(DeepEvalBaseLLM):
    """3.9.6 即使不用 judge 也要求 model；拒绝生成以保证 Core 零模型调用。"""

    def load_model(self):
        return None

    def generate(self, *args, **kwargs):
        raise RuntimeError("LLM judge is disabled for deterministic tool scoring")

    async def a_generate(self, *args, **kwargs):
        raise RuntimeError("LLM judge is disabled for deterministic tool scoring")

    def get_model_name(self):
        return "disabled"


def tool_correctness(
    expected: list[str], actual: list[str], *, exact: bool = False,
) -> bool:
    """未声明期望时不约束；contains 按名称集合，exact 保留次数及完整顺序。"""
    if not expected:
        return True
    test_case = LLMTestCase(
        input="tool selection",
        actual_output="",
        tools_called=[ToolCall(name=name) for name in actual],
        expected_tools=[
            ToolCall(name=name)
            for name in (expected if exact else dict.fromkeys(expected))
        ],
    )
    metric = ToolCorrectnessMetric(
        model=_DisabledJudge(),
        available_tools=None,
        strict_mode=True,
        async_mode=False,
        should_exact_match=exact,
    )
    return metric.measure(
        test_case, _show_indicator=False, _log_metric_to_confident=False,
    ) == 1.0
