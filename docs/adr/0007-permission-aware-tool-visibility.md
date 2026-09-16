# ADR 0007：权限感知的工具可见性

- 状态：Accepted
- 日期：2026-07-22
- 决策者：Project Maintainer

> 2026-09-08 修订：双层安全语义保持不变；模型可见性现由
> `ProjectToolVisibilityMiddleware` 在 `wrap_model_call/awrap_model_call` 中过滤本地
> `ProjectStructuredTool`，执行授权仍由 `ToolRuntime` 决定。下文 Provider/Registry
> 描述仅作历史记录，现状见[当前架构](../architecture/current-system.md)。

## 背景

`LocalToolProvider` 原先忽略 `ToolInvocationContext`，把 Registry 中所有 ToolSpec 的名称、description 和参数 Schema 绑定给模型。无权主体虽然最终会被 `ToolRuntime` 拒绝，但仍能看到隐藏能力，并可能反复选择注定失败的工具。

## 决策

建立两个独立安全层：

```text
模型可见性：ToolProvider 按请求权限过滤 ToolSpec
执行授权：ToolRuntime 对每次真实调用做最终治理
```

Provider 可见性判定固定为：

```python
all(
    has_permission(context.permissions, permission)
    for permission in spec.required_permissions
)
```

因此空 `required_permissions` 始终可见，多个权限采用 AND 语义，匹配继续精确且区分大小写，不支持通配权限。Provider 保持 Registry 顺序，不修改 Context、ToolSpec 或 Registry，也不执行 handler、Runtime 或审计。

## 为什么只检查 required_permissions

完整 `ToolPolicyEngine` 还会判断主体、来源、风险、审批和幂等键，这些是执行期条件。需要审批、需要幂等键或高风险的工具，只要权限满足仍应让模型知道其存在，以便未来接入 Human-in-the-loop；Provider 不复制 Policy Engine。

## 隐藏名称的调用

`RegistryToolsMiddleware` 继续使用 Provider 结果覆盖模型工具和 ToolNode 工具。模型伪造或复用隐藏名称时，`request.tool is None`，框架按未知工具返回稳定 `unknown_tool`。该消息不包含缺失权限、description、Schema、风险或审批元数据，也不为区分“未注册”和“无权隐藏”反查 Registry。

## Runtime 仍是最终边界

可见性过滤不是授权。已绑定工具仍严格经过：

```text
StructuredTool
→ ToolRegistry.dispatch
→ ToolRuntime.execute
→ ToolSpec.handler
```

直接调用 Registry 或发生上下文异常时，Runtime 仍重新执行 Policy；缺权结果为 `permission_denied`，handler 不运行。审批、幂等、审计、超时、有限重试和输出校验也继续只在 Runtime 执行。

## 后果

- 模型只接收当前请求所需的最小工具 Schema，减少信息暴露和无效选择。
- 同一 Provider 可服务不同权限请求，不缓存请求 Context 或过滤结果。
- 默认 `/chat` 服务 Context 通过 `TOOL_DEFAULT_PERMISSIONS` 保持现有生产工具可见性。
- 未新增认证、JWT、角色系统、API 字段、第三方依赖、Agent 节点或自定义 ToolNode。
