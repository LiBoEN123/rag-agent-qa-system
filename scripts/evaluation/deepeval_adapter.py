"""DeepEval 3.9.6：确定性工具评分与可选 GEval 答案评分，不创建云端 run。"""

import asyncio
import os

# Must precede DeepEval imports, including its config and telemetry modules.
os.environ.update(
    DEEPEVAL_TELEMETRY_OPT_OUT="YES",
    DEEPEVAL_DISABLE_DOTENV="1",
    DEEPEVAL_UPDATE_WARNING_OPT_IN="0",
    DEEPEVAL_FILE_SYSTEM="READ_ONLY",
    CONFIDENT_TRACING_ENABLED="NO",
)

from deepeval.metrics import GEval, ToolCorrectnessMetric  # noqa: E402
from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall  # noqa: E402

from app.llm.factory import get_judge_model  # noqa: E402
from app.llm.gateway import ModelGateway  # noqa: E402
from scripts.evaluation.rag.common import point_texts  # noqa: E402


class ProjectJudge(DeepEvalBaseLLM):
    """只转发文本；配置、模型调用及错误由项目 Factory/Gateway 管理。"""

    def __init__(self, **model_options):
        self.model_options = model_options
        super().__init__()

    def load_model(self):
        return ModelGateway(role="judge", langchain_model=get_judge_model(**self.model_options))

    def generate(self, prompt: str) -> str:
        return self.model.invoke_text([{"role": "user", "content": prompt}])

    async def a_generate(self, prompt: str) -> str:
        return await asyncio.to_thread(self.generate, prompt)

    def get_model_name(self):
        return getattr(self.model.langchain_model, "model_name", "judge")


def answer_correctness(question: dict, answer: str, *, model: DeepEvalBaseLLM) -> tuple[float, str]:
    """Gold 构成评价规范；GEval 独立负责提示词、响应解析与分数计算。"""
    expected = "\n\n".join([
        f"Expected behavior:\n{question.get('expected_behavior', 'answer')}",
        f"Reference answer:\n{question.get('gold_answer', '')}",
        "Required key points:\n" + "\n".join(f"- {p}" for p in point_texts(question.get("gold_key_points", []))),
        "Forbidden claims:\n" + "\n".join(f"- {p}" for p in point_texts(question.get("forbidden_points", []))),
    ])
    metric = GEval(
        name="answer_correctness",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        evaluation_steps=[
            "Read the question and expected output as the evaluation specification; assess semantic correctness, not wording or style.",
            "Check compliance with expected behavior (answer, partial_answer, abstain). For partial_answer, answer the known part and clearly state limits on the unknown part. For abstain, a justified refusal is correct and must not be penalized for lacking an answer.",
            "Compare the actual output with the reference answer and required key points. Penalize important factual omissions and contradictions; accept semantically equivalent wording.",
            "Penalize forbidden claims asserted as facts. Explicitly denying a forbidden claim or stating it is unknown is not an assertion of that claim.",
            "Give one correctness score considering these requirements together, and explain the material errors or why the answer is correct.",
        ],
        model=model,
        async_mode=False,
    )
    metric.measure(
        LLMTestCase(input=question["question"], actual_output=answer, expected_output=expected),
        _show_indicator=False, _log_metric_to_confident=False,
    )
    if metric.error or metric.score is None:
        raise ValueError(metric.error or "GEval returned no score")
    return metric.score, metric.reason or ""


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
