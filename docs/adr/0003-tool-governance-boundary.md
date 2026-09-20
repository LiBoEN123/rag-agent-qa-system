# ADR 0003：工具治理边界

> 历史设计记录。本地工具当前实现以 [current-system](../architecture/current-system.md) 为准：TOOL-SLIM-1/2 已移除动态 runtime schema、自定义工具子类、私有线程池与假设性的 Local write 治理；MCP 写入审批保留。

- 状态：Accepted
- 日期：2026-07-19
- 决策者：Project Maintainer

> 2026-08-16 修订：旧 `ToolRegistry.dispatch` / `dispatch_tool` plumbing 已按
> [ADR 0004 修订说明](0004-registry-driven-tool-discovery.md#2026-08-16-修订说明)
> 收口为 `execute_registered_tool → ToolRuntime.execute`；下文保留当时决策原文。
>
> 2026-09-08 修订：本地定义改由不可变 ToolSpec catalog 提供，LangChain 原生
> `StructuredTool` 直接进入 `execute_project_tool → ToolSpec.handler`；权限可见性
> 由模型调用前的 `ProjectToolVisibilityMiddleware` 处理。下文 Registry/Provider 描述
> 仅记录历史实现，以[当前架构](../architecture/current-system.md)为准。

## 背景

项目已经围绕 `ToolSpec`、`ToolRegistry` 和 `ToolRuntime` 建立统一工具契约与执行治理。当前 `LocalToolProvider` 提供已注册 ToolSpec，LangChain Adapter 构造请求级 `StructuredTool`，其适配函数再调用定义所属的 Registry 和 Runtime。后续引入 Tool Calling、LangGraph `ToolNode`、MCP、搜索、数据库或浏览器工具时，必须保留这条边界，防止模型或第三方框架绕过权限、审计和运行时限制。

## 当前问题

- 通用 Tool Calling 框架通常可以直接绑定 Python callable；若直接绑定业务函数，现有治理能力将失效。
- 第三方工具的参数、风险、权限、超时和审计语义不一定与项目一致，需要本地适配。
- 后续写工具和外部系统工具的风险高于当前只读工具，必须在 handler 执行前做确定性决策。
- MCP 或框架级 ToolNode 是协议/编排适配层，不应成为第二套不受本地策略约束的执行运行时。

## 决策

工具治理边界遵循三条不可绕过的规则：

```text
LLM 不直接调用业务函数。
本地工具必须注册到 ToolRegistry 并通过 ToolRuntime 执行。
Adapter-native 外部工具必须经过项目权限、审计、超时和输出安全 bridge。
```

本地工具的合法执行链为：

```text
Agent Model
→ LangChain StructuredTool / LangGraph ToolNode
→ ToolRegistry.dispatch
→ ToolRuntime.execute
→ ToolSpec.handler
```

项目的 `dispatch_tool` 可以继续作为日志与兼容门面，但只能转发到上述 Registry/Runtime 链路，不能自行调用 handler。

官方 adapter 已经提供的外部 `StructuredTool` 不反向转换成 `ToolSpec`。例如只读 MCP
Client 工具保留 adapter Schema/coroutine，经本地 allowlist、per-server 权限、审计、
超时和不可信输出归一化后，与本地工具一起进入同一个 `create_agent` 和
`ProjectToolGovernanceMiddleware`。这是一条受治理的协议适配边界，不是直接绑定远端
handler，也不能绕过调用数、错误数、重复调用、总耗时或 ToolMessage pairing。

明确禁止：

```text
Agent Model
→ 直接调用 handler

ToolNode
→ 绕过 ToolRuntime 调用业务函数

MCP Tool
→ 绕过本地权限和审计直接暴露给模型
```

## 采用的方案

### 当前已经具备的治理能力

- `ToolSpec` 工具契约，包含名称、SemVer 版本、描述、输入/输出 Schema 和 handler。
- Pydantic 参数模型与执行前参数校验，以及可选的 Pydantic 输出模型校验。
- 四级风险枚举、精确权限要求、审批要求、幂等性、超时和有限重试元数据。
- 非幂等工具禁止配置自动重试；运行时对可重试错误实施有限重试、指数退避和抖动。当前五个内置工具的 `max_retries` 均为 0。
- 绑定工具名、版本和规范化参数的稳定 SHA-256 参数指纹。
- `requested`、`denied`、`approval_required`、`started`、`retrying`、`succeeded`、`failed`、`timed_out` 结构化审计事件，以及可注入审计 sink。
- 对敏感参数和输出字段的递归脱敏、循环引用防护、深度/项目数/字符串/序列化体量限制。
- 按工具配置的输出审计模式：不记录正文、有限摘要或脱敏输出；`get_chunk_text` 当前不把正文写入审计摘要。
- 统一工具错误码和兼容结果结构。
- 进程内复用的受限共享线程池、每工具超时、运行次数和耗时元数据，以及生命周期关闭能力。
- 请求级、不可变的 `ToolInvocationContext`，携带 trace、session、主体、精确权限、来源和幂等键。
- Registry 在注册阶段校验契约，并拒绝工具重名被静默覆盖。
- Policy Engine 在线程池提交 handler 前检查主体、权限、写操作幂等键、风险和审批要求。
- 写工具的 `requested` 审计事件写入失败时 fail-closed，handler 不执行；只读工具可 fail-open，但结果会标记审计写入失败。

审批要求目前表现为执行前的 `approval_required` 策略结果；项目尚未实现 LangGraph interrupt、人工批准和恢复执行流程。

### 风险分类

只使用代码中已有的 `ToolRiskLevel`：

| 枚举名称 | 序列化值 | 含义 |
| --- | --- | --- |
| `READ_ONLY` | `read_only` | 读取或计算，不应产生持久副作用 |
| `INTERNAL_WRITE` | `internal_write` | 修改项目控制范围内的状态 |
| `EXTERNAL_WRITE` | `external_write` | 修改外部系统状态 |
| `DESTRUCTIVE` | `destructive` | 删除、覆盖或其他难以恢复的操作 |

### 第三方工具接入原则

需要由项目直接拥有执行实现的第三方工具、GitHub、数据库、搜索引擎和浏览器工具必须：

1. 转换为项目本地 `ToolSpec`。
2. 使用 Pydantic 参数模型。
3. 设置风险等级。
4. 设置所需权限。
5. 设置合理超时。
6. 明确是否幂等。
7. 明确是否需要审批。
8. 通过统一审计。
9. 对敏感输入和输出进行脱敏，并选择安全的输出摘要模式。
10. 不允许工具重名被静默覆盖。

如果正式协议 adapter 已经负责 Schema、coroutine 和协议调用（当前只读 MCP Client
属于此类），不得为满足上述本地链路而重复实现 JSON Schema 到 Pydantic、同步 handler
或协议 Client；应改用薄 bridge 达成等价的本地权限、审计、超时、脱敏和 Agent
Guardrail 约束。

### 写操作原则

- 写工具默认需要比只读工具更严格的主体信息和权限。
- 外部写入和破坏性操作原则上需要人工审批；现有契约也要求这两类风险必须声明 `requires_approval=True`。
- 审批决策必须发生在 handler 执行前。
- 执行前审计失败时，高风险写操作不得继续执行。
- 恢复执行时必须考虑幂等性；写工具需要幂等键，真正恢复前还需建设持久化去重或等价的一致性机制。

## 暂不采用的方案

- 不直接把任意 Python callable、Shell、数据库连接或 HTTP 客户端绑定给模型。
- 不使用绕过 Registry/Runtime 的 ToolNode 快捷执行路径。
- 不把远端 MCP Provider 的权限声明视为本地授权结果。
- 不允许第三方工具继承宽泛默认权限，也不允许以记录完整输入输出代替结构化审计。
- 在审批恢复和持久化幂等机制完成前，不启用需要恢复执行的高风险写工具。

## 影响

- 所有工具接入都有一层本地适配成本，但权限、审计、错误和运行时行为保持一致。
- LangChain、LangGraph 和 MCP 仅负责模型可见 Schema、协议与编排；项目自身继续拥有最终执行决策权。
- 高风险工具可能因为权限、幂等键、审批或审计不可用而在执行前被拒绝，这是预期的安全行为。
- 工具测试必须同时覆盖契约校验、策略拒绝、审计安全、超时/重试和 handler 是否未被越权调用。

## 后续行动

1. PR1 的 `governed_tools` 节点与 PR2 的 ToolProvider/Adapter 复用现有 Registry/Runtime，不直接使用业务 handler。
2. 增加模型 `tool_call_id` 与 Runtime `tool_call_id`、`ToolMessage` 和审计事件之间的关联设计。
3. 在首个写工具前实现 LangGraph interrupt 审批恢复与持久化幂等去重。
4. 为 MCP、数据库、搜索和浏览器分别提供最小权限的本地 Adapter 与安全测试。
