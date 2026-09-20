"""Evaluation Baseline v2 的版本化数据合同与覆盖率校验。

用途：用严格 Pydantic 模型约束 Agent/RAG 输入，并检查类别、工具和 case 覆盖率。
输入：版本化评测 JSON、当前注册工具名，以及覆盖率要求。
输出：不可变的评测对象和可序列化的覆盖率报告；无效输入直接给出明确错误。
边界：这里只验证合同，不执行 Agent、模型、RAG、网络请求或输出写入。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.contracts.chat import AgentResultStatus


SuiteName = Literal["core", "online", "mcp", "mixed"]
SourceMatch = Literal["any", "all", "none"]
ToolCallStatus = Literal["success", "error"]
ToolOrigin = Literal["local", "web", "mcp"]


_SAFE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class ExpectedOutcome(BaseModel):
    """单轮输出的确定性期望。"""

    model_config = ConfigDict(extra="forbid")

    decision: str = ""
    action: str = ""
    status: AgentResultStatus | None = None
    first_tool: str = ""
    first_tool_any_of: list[str] = Field(default_factory=list)
    tools_contains: list[str] = Field(default_factory=list)
    tool_sequence: list[str] = Field(default_factory=list)
    tool_status_sequence: list[ToolCallStatus] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    min_keyword_hits: int = Field(default=0, ge=0)
    forbidden_keywords: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    source_match: SourceMatch = "any"
    tool_output_keys: list[str] = Field(default_factory=list)
    rewrite_keywords: list[str] = Field(default_factory=list)
    min_rewrite_keyword_hits: int = Field(default=0, ge=0)
    rewritten: bool | None = None
    rewrite_llm_used: bool | None = None
    tool_origin: ToolOrigin | None = None
    mcp_server: str = ""
    approval_required: bool | None = None
    content_is_untrusted: bool | None = None
    external_output_nonempty: bool | None = None

    @model_validator(mode="after")
    def validate_hit_bounds(self) -> ExpectedOutcome:
        """关键词最低命中数不能超过对应关键词数量。"""
        if self.min_keyword_hits > len(self.keywords):
            raise ValueError("min_keyword_hits 不能超过 keywords 数量")
        if self.min_rewrite_keyword_hits > len(self.rewrite_keywords):
            raise ValueError(
                "min_rewrite_keyword_hits 不能超过 rewrite_keywords 数量"
            )
        if self.tool_status_sequence and (
            len(self.tool_status_sequence) != len(self.tool_sequence)
        ):
            raise ValueError(
                "tool_status_sequence 长度必须与 tool_sequence 一致"
            )
        if self.first_tool and self.first_tool_any_of:
            raise ValueError(
                "first_tool 与 first_tool_any_of 只能使用一个"
            )
        tool_names = [
            self.first_tool,
            *self.first_tool_any_of,
            *self.tools_contains,
            *self.tool_sequence,
        ]
        if any(
            name and not _SAFE_NAME_PATTERN.fullmatch(name)
            for name in tool_names
        ):
            raise ValueError("expected tool name 必须使用安全名称格式")
        if self.mcp_server and not _SAFE_NAME_PATTERN.fullmatch(
            self.mcp_server
        ):
            raise ValueError("mcp_server 必须使用安全 Server alias 格式")
        if self.mcp_server and self.tool_origin != "mcp":
            raise ValueError("mcp_server 只能与 tool_origin=mcp 一起使用")
        if self.tool_origin == "mcp" and not self.mcp_server:
            raise ValueError("tool_origin=mcp 时必须声明 mcp_server")
        return self

    def expected_tools(self) -> set[str]:
        """返回本条期望引用的全部工具名。"""
        return {
            name
            for name in (
                [self.first_tool]
                + self.first_tool_any_of
                + self.tools_contains
                + self.tool_sequence
            )
            if name
        }


class EvaluationTurn(BaseModel):
    """多轮 case 中的一轮。"""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    expected: ExpectedOutcome = Field(default_factory=ExpectedOutcome)


class AgentEvaluationCase(BaseModel):
    """单轮或多轮 Agent 评测 case。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    version: Literal[2]
    suite: SuiteName
    category: str = Field(min_length=1)
    question: str | None = None
    expected: ExpectedOutcome | None = None
    session_id: str = ""
    turns: list[EvaluationTurn] | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    legacy_id: str = ""
    disabled: bool = False
    disabled_reason: str = ""
    mcp_servers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_case_shape(self) -> AgentEvaluationCase:
        """单轮 question 与多轮 turns 必须且只能存在一种。"""
        has_question = self.question is not None
        has_turns = self.turns is not None
        if has_question == has_turns:
            raise ValueError("question 与 turns 必须且只能提供一个")
        if has_question:
            if not self.question or not self.question.strip():
                raise ValueError("question 不能为空")
            if self.expected is None:
                raise ValueError("单轮 case 必须提供 expected")
        if has_turns and len(self.turns or []) < 2:
            raise ValueError("多轮 case 的 turns 至少包含两轮")
        if self.disabled and not self.disabled_reason.strip():
            raise ValueError("disabled case 必须提供 disabled_reason")
        if len(self.mcp_servers) != len(set(self.mcp_servers)):
            raise ValueError("mcp_servers 不允许重复")
        if any(
            not _SAFE_NAME_PATTERN.fullmatch(alias)
            for alias in self.mcp_servers
        ):
            raise ValueError("mcp_servers 必须使用安全 Server alias 格式")
        if self.suite in {"core", "online"} and self.mcp_servers:
            raise ValueError("core/online case 的 mcp_servers 必须为空")
        if self.suite == "mcp" and not self.mcp_servers:
            raise ValueError("mcp case 的 mcp_servers 必须非空")
        for expected in self.iter_expected():
            if (
                expected.mcp_server
                and expected.mcp_server not in self.mcp_servers
            ):
                raise ValueError(
                    "expected.mcp_server 必须包含在 case.mcp_servers 中"
                )
        return self

    def iter_expected(self) -> list[ExpectedOutcome]:
        """按执行顺序返回每轮期望。"""
        if self.turns is not None:
            return [turn.expected for turn in self.turns]
        return [self.expected or ExpectedOutcome()]

    def iter_questions(self) -> list[str]:
        """按执行顺序返回问题。"""
        if self.turns is not None:
            return [turn.question for turn in self.turns]
        return [self.question or ""]


class AgentEvaluationDataset(BaseModel):
    """完整 v2 Agent 数据集。"""

    model_config = ConfigDict(extra="forbid")

    dataset_version: Literal[2] = 2
    name: str = Field(min_length=1)
    cases: list[AgentEvaluationCase]

    @model_validator(mode="after")
    def validate_unique_ids(self) -> AgentEvaluationDataset:
        """拒绝重复 case ID。"""
        counts = Counter(case.id for case in self.cases)
        duplicates = sorted(case_id for case_id, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"重复 case ID: {', '.join(duplicates)}")
        return self


def load_agent_evaluation_dataset(path: str | Path) -> AgentEvaluationDataset:
    """读取并校验 v2 数据集；兼容最外层 list。"""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"找不到 Agent 评测集：{resolved}")
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        raw = {
            "dataset_version": 2,
            "name": resolved.stem,
            "cases": raw,
        }
    dataset = AgentEvaluationDataset.model_validate(raw)
    json.dumps(
        dataset.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return dataset


def validate_tool_names(
    dataset: AgentEvaluationDataset,
    registered_tool_names: list[str] | tuple[str, ...] | set[str],
) -> None:
    """确认每个期望工具都来自当前本地 ToolSpec catalog。"""
    known = set(registered_tool_names)
    unknown: dict[str, list[str]] = {}
    for case in dataset.cases:
        names = set().union(
            *(expected.expected_tools() for expected in case.iter_expected())
        )
        runtime_names = set().union(
            *(
                expected.expected_tools()
                if expected.tool_origin == "mcp"
                else set()
                for expected in case.iter_expected()
            )
        )
        missing = sorted(names - runtime_names - known)
        if missing:
            unknown[case.id] = missing
    if unknown:
        detail = "; ".join(
            f"{case_id}={','.join(names)}"
            for case_id, names in sorted(unknown.items())
        )
        raise ValueError(f"数据集包含未知工具：{detail}")


def _counter(values: list[str]) -> dict[str, int]:
    """返回键稳定排序的计数。"""
    return dict(sorted(Counter(values).items()))


def build_coverage_report(
    dataset: AgentEvaluationDataset,
    registered_tool_names: list[str] | tuple[str, ...] | set[str],
) -> dict[str, Any]:
    """汇总数据集覆盖矩阵并显式列出稳定能力 gap。"""
    active = [case for case in dataset.cases if not case.disabled]
    expected_items = [
        (case, expected)
        for case in active
        for expected in case.iter_expected()
    ]
    all_tools = [
        tool
        for _, expected in expected_items
        for tool in sorted(expected.expected_tools())
    ]
    first_tools = [
        expected.first_tool
        or "|".join(expected.first_tool_any_of)
        or "<none>"
        for _, expected in expected_items
    ]
    statuses = [
        expected.status.value if expected.status is not None else "<none>"
        for _, expected in expected_items
    ]
    decisions = [
        expected.decision or "<none>"
        for _, expected in expected_items
    ]

    capabilities = {
        "search_knowledge_base": "search_knowledge_base" in all_tools,
        "search_document": "search_document" in all_tools,
        "find_documents": "find_documents" in all_tools,
        "get_chunk_context": "get_chunk_context" in all_tools,
        "direct_answer": "direct_answer" in decisions,
        "clarify": "clarify" in decisions,
        "NO_RELEVANT_INFO": "NO_RELEVANT_INFO" in statuses,
        "query_rewrite": any(case.turns is not None for case in active),
        "multi_tool": any(
            len(expected.tool_sequence) > 1
            for _, expected in expected_items
        ),
        "tool_failure_or_correction": any(
            {"tool_recovery", "argument_correction"} & set(case.tags)
            for case in active
        ),
        "web_search": "web_search" in all_tools,
        "web_fetch": "web_fetch" in all_tools,
        "web_pdf_fetch": "web_pdf_fetch" in all_tools,
    }
    return {
        "dataset_version": dataset.dataset_version,
        "total_cases": len(dataset.cases),
        "active_cases": len(active),
        "disabled_cases": len(dataset.cases) - len(active),
        "suite": _counter([case.suite for case in active]),
        "category": _counter([case.category for case in active]),
        "status": _counter(statuses),
        "decision": _counter(decisions),
        "first_tool": _counter(first_tools),
        "expected_tools": _counter(all_tools),
        "mcp_servers": _counter(
            [alias for case in active for alias in case.mcp_servers]
        ),
        "case_shape": {
            "single_turn": sum(case.turns is None for case in active),
            "multi_turn": sum(case.turns is not None for case in active),
        },
        "checks": {
            "sources": sum(
                bool(expected.sources) for _, expected in expected_items
            ),
            "keywords": sum(
                bool(expected.keywords) for _, expected in expected_items
            ),
            "forbidden_keywords": sum(
                bool(expected.forbidden_keywords)
                for _, expected in expected_items
            ),
            "rewrite": sum(
                bool(expected.rewrite_keywords)
                or expected.rewritten is not None
                or expected.rewrite_llm_used is not None
                for _, expected in expected_items
            ),
        },
        "capabilities": capabilities,
        "gaps": sorted(
            name for name, covered in capabilities.items() if not covered
        ),
        "registered_tools": sorted(registered_tool_names),
        "disabled": [
            {
                "id": case.id,
                "reason": case.disabled_reason,
            }
            for case in dataset.cases
            if case.disabled
        ],
    }


__all__ = [
    "AgentEvaluationCase",
    "AgentEvaluationDataset",
    "EvaluationTurn",
    "ExpectedOutcome",
    "build_coverage_report",
    "load_agent_evaluation_dataset",
    "validate_tool_names",
]
