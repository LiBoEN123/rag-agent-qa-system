# ADR 0001：开源 Agent 技术栈

- 状态：Accepted
- 日期：2026-07-19
- 决策者：Project Maintainer

> 2026-08-21 Current implementation note：开源、可替换模型和不重写 RAG/ToolRuntime
> 的原则继续有效；“Agent 主图”、SQLite Checkpointer 和“全部第三方工具进入本地
> ToolRuntime”的实现描述已分别由 ADR 0006、0011 及当前架构修订。当前只有
> `create_agent` 内部循环与独立 RAG Graph，MCP 保持独立 Bridge/Interceptor 治理。

## 背景

项目当前以 FastAPI 提供 API，以 LangGraph 编排 Agent 主图和 RAG 子图，以 LangChain Core 适配结构化工具，并保留了既有的混合检索、重排、引用校验、工具治理和 SQLite 状态存储能力。后续需要引入迭代式 Agent Loop、更多工具类型和更完整的可观测与评测能力，因此需要先确定技术选型原则，避免演进过程中形成模型厂商锁定、重复治理链路或不必要的整体重写。

## 当前问题

- 当前 Agent 仍以 Router 决策和固定 Planner 计划驱动，后续需要迭代式模型—工具循环，但这不构成推翻现有 RAG 与工具基础设施的理由。
- 项目已经具备 `ToolRegistry`、`ToolRuntime`、Pydantic 参数校验、权限策略、审计、超时和错误归一化；直接采用另一套工具执行通道会造成治理能力重复和边界不一致。
- 后续候选组件很多，若缺少统一原则，容易为了框架本身增加依赖、部署成本和调试复杂度。
- 模型、搜索、数据库和轨迹平台都存在托管与自托管实现，需要明确可替换性和部署控制权。

## 决策

项目后续 Agent 能力优先采用开源框架和可自行部署的组件，但不为了使用框架而使用框架。每次引入新组件前，必须先验证它是否解决当前明确问题，并评估对现有 API、RAG、工具治理、部署和测试基线的影响。

项目不绑定单一模型厂商。模型通过 OpenAI-compatible 接口或 LangChain 模型抽象接入，并保持模型实现可替换。

引入新框架不得推翻现有 RAG，也不得重复实现现有 `ToolRuntime` 的治理能力。所有第三方工具都必须适配到项目现有工具契约中，经本地注册、策略、审计和运行时执行。

## 采用的方案

| 能力 | 优先方案 | 定位 |
| --- | --- | --- |
| Agent 编排 | LangGraph | 保留现有主图和 RAG 子图，后续承载迭代循环 |
| 模型和工具抽象 | LangChain Core | 使用可替换模型抽象与 `StructuredTool` |
| 工具定义 | Pydantic Schema + `StructuredTool` | Pydantic 定义参数边界，适配 LangChain/LangGraph |
| 工具执行治理 | 现有 `ToolRegistry` + `ToolRuntime` | 作为权限、审批信号、审计、超时、重试和错误处理的唯一治理入口 |
| API 服务 | FastAPI | 保留现有 API 服务和协议 |
| 状态持久化 | 当前 SQLite Checkpointer | 继续使用现有可配置的 LangGraph SQLite Checkpointer；当前默认仍可保持禁用 |
| 知识库 | ChromaDB + BM25 + Embedding + Rerank | 保留当前混合检索与二阶段重排链路 |
| 工具标准协议 | 后续考虑 MCP 官方 Python SDK | 作为外部工具接入协议，仍须适配本地工具契约 |
| 网络搜索 | 后续优先考虑自托管 SearXNG | 降低服务绑定并保留部署控制权 |
| 网页读取 | 后续优先考虑 HTTPX + Trafilatura | 用于静态网页获取和正文抽取 |
| 动态网页 | 后续按需考虑 Playwright | 仅在静态读取不能满足明确需求时引入 |
| 数据库访问 | 后续优先考虑 SQLAlchemy Core + SQLGlot | 分离执行和 SQL 解析/校验，并优先建设只读路径 |
| 可观测性 | 后续优先考虑 OpenTelemetry | 提供厂商中立的 traces、metrics 和上下文传播 |
| LLM 轨迹平台 | 后续可选自托管 Langfuse | 作为可选增强，不替代项目内审计 |
| Agent 评测 | 后续考虑 DeepEval | 补充通用 Agent 轨迹和任务完成度评测 |
| 安全红队 | 后续考虑 Promptfoo | 补充提示注入和工具滥用回归测试 |
| 代码沙箱 | 后续考虑 Docker，必要时升级 gVisor | 仅在明确需要代码执行时引入隔离边界 |

上述组件均为优先候选而非无条件依赖。引入时仍需通过独立 ADR 或 PR 说明当前问题、最小接入面、退出方案和验证结果。

## 暂不采用的方案

- CrewAI。
- AutoGen。
- 多个角色型 Agent。
- 完全重写为 `create_agent`。
- 云端强绑定的 Agent 平台。
- 直接把 Shell 权限交给模型。

当前主要问题是缺少迭代式工具调用循环，而不是缺少多 Agent 框架。现有 `ToolRuntime`、RAG 子图、审计能力和 API 协议必须保留；CrewAI、AutoGen 或多个角色型 Agent 会增加模型调用次数、延迟、状态同步和调试复杂度，却不会自动解决当前单 Agent 的循环问题。

完全重写为 `create_agent` 会扩大迁移面并弱化现有节点与治理边界的可验证性。云端强绑定平台与模型可替换、自托管优先原则不一致。Shell 属于高风险执行能力，在具备明确需求、权限最小化、审批、审计和沙箱前不得直接暴露给模型。

## 影响

- 现有 FastAPI、LangGraph、RAG、工具契约和治理链路继续作为演进基础，不发生一次性重写。
- 新组件需要提供本地适配层和可测试边界，不能绕过 `ToolRegistry` 与 `ToolRuntime`。
- 开源与可自托管优先会增加部分本地运维责任，但换取可审计、可替换和可控的数据边界。
- 模型抽象保持厂商中立；特定模型能力只能作为可选优化，不能成为无降级路径的核心依赖。

## 后续行动

1. 在 PR1 以现有 LangGraph 主图为基础实现单 Agent 的迭代式工具调用循环。
2. 为每个新增组件提交最小可验证方案，说明解决的问题、部署方式、权限和退出策略。
3. 所有新增工具先定义本地 `ToolSpec` 与 Pydantic 参数模型，再提供 `StructuredTool` 或其他协议适配。
4. 在引入 MCP、搜索、数据库、浏览器或代码执行能力前分别补充安全与治理 ADR。
