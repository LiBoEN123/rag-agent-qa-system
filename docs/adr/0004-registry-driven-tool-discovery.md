# ADR 0004：基于注册表的工具发现

- 状态：Accepted
- 日期：2026-07-20
- 决策者：Project Maintainer

## 背景

PR1 已经建立 `agent_model ↔ governed_tools` 的迭代式 Agent Loop，但模型绑定的工具仍由适配层直接遍历全局 Registry，Router 和 Planner 仍按具体工具名选择、校验和规划。新增普通工具时需要修改多个编排模块，工具定义的事实来源并不唯一。

## 当前问题

- Router 同时承担入口判断、具体工具选择和工具参数校验。
- Planner 为每个具体工具维护分支，新增工具会修改核心编排代码。
- 测试难以为单次请求注入独立 Registry，容易污染生产全局工具集合。
- 后续增加其他工具来源时缺少只负责发现 ToolSpec 的边界。
- 具体工具必须继续经过既有权限、审批判断、超时、重试、审计、脱敏和错误码治理，不能为了动态发现而直接绑定 handler。

## 决策

`ToolRegistry` 是本地生产工具定义的事实来源。`ToolProvider` 只按请求上下文提供稳定有序的 `ToolSpec`，LangChain Adapter 自动把这些定义转换为请求级 `StructuredTool`，Agent Model 根据 description 和 Pydantic `args_schema` 动态选择具体工具。

合法链路为：

```text
LocalToolProvider
→ ToolRegistry 中已注册的 ToolSpec
→ 请求级 StructuredTool
→ ToolRegistry.dispatch
→ ToolRuntime.execute
→ ToolSpec.handler
```

Router 只允许 `agent`、`rag_fast_path`、`direct_answer` 和 `clarify` 四种粗粒度 action，不接收普通工具 Schema，也不校验普通工具参数。由于静态 Router 无法判断动态 Provider 中是否存在刚注册的适用工具，正常自然语言请求采用 Agent-first：除安全短路外默认进入 `agent`。Planner/Executor 不参与正常 `agent` 路径的具体工具选择；它们保留显式 RAG Fast Path、澄清、直接回答和旧 decision 的薄兼容能力。

## 采用的方案

- `ToolProvider` 使用 Protocol，只定义按 `ToolInvocationContext` 返回 `ToolSpec` 序列的发现职责。
- `LocalToolProvider` 绑定一个 `ToolRegistry`，按注册顺序返回当前 Context 满足全部 `required_permissions` 的 ToolSpec，不执行 handler、完整 Policy 或 Runtime，也不缓存请求 Context。
- `collect_tool_specs()` 按 Provider 顺序合并定义，保持各 Provider 内部顺序，并在重复名称出现时明确报错。
- LangChain Adapter 校验每个 ToolSpec 确实属于用于分发的 Registry，然后从 `name`、`description` 和 `args_model` 构造 StructuredTool。
- StructuredTool 闭包调用所属 Registry 的 `dispatch()`，不直接调用 `ToolSpec.handler`，因此 Runtime 仍是唯一执行和治理入口。
- `get_agent_tool_provider()` 是可替换的薄工厂。测试可以注入独立 Registry/Provider，而不修改全局生产 Registry，也不把 Provider 放入可持久化 Graph State。
- Provider 复用 `has_permission()` 做最小权限可见性过滤；模型伪造未绑定的隐藏名称按未知工具处理。模型可见性不替代授权，Runtime 始终执行最终精确权限与执行期策略校验。
- 显式或旧协议 RAG Fast Path 继续通过已注册的 `search_knowledge_base` 和原 RAG 子图保留问题拆分、重排、证据门控、拒答及引用绑定；正常 Agent 选择该检索工具后也进入同一 RAG 子图。

## 暂不采用的方案

- 不把任意 callable 直接作为 Provider 输出或绑定给模型。
- 不在 Provider 内执行工具、做审批决定、复制参数校验或记录 Runtime 审计。
- 不把请求级 Context 缓存在全局 Provider 或 StructuredTool 集合中。
- 不为新增普通工具增加 Router、Planner、Executor 或 Graph 分支。
- 本阶段不实现 MCP、Web、SQL、GitHub、浏览器、Shell、Python 沙箱、外部工具加载、多 Agent 或人工审批 interrupt。
- 当前不增加 Provider 配置环境变量，生产环境固定使用本地 Provider。

## 影响

- 注册合法 ToolSpec 后，普通工具可自动进入模型 `bind_tools()`，不同 Pydantic 参数结构无需编排特判。
- 测试 Registry 可以请求级注入，`echo_text` 和 `join_texts` 不会进入生产 Registry，也不会跨请求泄漏。
- 未注册或当前 Context 不可见的工具只会形成结构化 `unknown_tool` ToolMessage，不会执行任意函数或泄漏隐藏契约；模型可以继续决策或由预算受控停止。
- 多 Provider 重名不会静默覆盖，但当前生产环境尚未组合多个 Provider。
- Router 的显式 RAG Fast Path 和 RAG 子图仍保留少量知识库工具名，这是专用兼容边界，不是普通工具选择机制。
- Agent-first 会比直接 RAG Fast Path 多一次 Agent Model 决策，但避免了静态 Router 对新工具和陌生口吻的不可恢复误分流。
- API Schema、SQLite 表结构、依赖版本和 RAG 算法保持不变。

## 后续行动

1. 在引入第二种生产 Provider 前定义其本地 Registry 适配、生命周期和失败隔离策略。
2. 外部 Provider 必须定义自己的请求级可见性规则，同时保留本地 Runtime 最终授权边界。
3. 后续独立 PR 再评估删除旧具体工具 decision 和 legacy Planner/Executor 入口。
4. 在引入任何写工具前完成持久化幂等、人工审批 interrupt 与恢复语义。

## 2026-08-16 修订说明

Phase 4 保留本 ADR 关于 Registry 事实来源、Provider 请求级可见性、测试隔离和 Runtime
最终授权的决策，但收口了其后逐步形成的纯转发 plumbing。当前本地执行链修订为：

```text
LocalToolProvider
→ ToolRegistry 中已注册的 ToolSpec
→ 请求级 StructuredTool
→ execute_registered_tool
→ ToolRuntime.execute
→ ToolSpec.handler
```

`ToolRegistry.dispatch()` 与旧 `dispatch_tool()` 门面已经删除。Registry 只负责注册、
重名拒绝、查找、稳定顺序和契约导出；`execute_registered_tool()` 是唯一公共本地执行
入口，继续按 Registry 复用 ToolRuntime。Provider 不执行工具，MCP 仍使用独立的 adapter
interceptor 安全边界。本修订同时取代 ADR 0003 和 ADR 0011 中对旧 dispatch 链路的当前
实现描述，不改变那些 ADR 记录的历史背景。

## 2026-08-20 修订说明

Phase 5 将本地 ToolSpec 的参数契约收敛为 Pydantic `args_model` 单一事实来源。
`ToolRuntime` 参数校验、LangChain StructuredTool 的模型可见 Schema 与
`ToolRegistry.contract_specs()` 导出均由该模型驱动，ToolSpec 构造器不再接收手写
`args_schema`。输出侧不强求形式对称：存在 `output_model` 时由它校验并派生 Schema；
没有输出模型时保留显式 `output_schema` 作为 handler payload 的展示 fallback，且禁止
两种来源同时声明。本修订不改变 Runtime result envelope、Provider、MCP 或 RAG
边界。
