# FRAMEWORK-SLIM-3 — Local Tool declaration 审计

审计基线：`refactor/tool-output-models`，HEAD `2acdeaa50801303bead362b6ff12679b382bcb15`，工作区干净。
当前安装：LangChain 1.3.14、langchain-core 1.4.9、LangGraph 1.2.9、Pydantic 2.12.5。

## 修改前的执行链与消费者

```text
catalog / SQL factory: ToolSpec
  → get_local_tool_specs(settings)
  → build_local_tool(spec)
  → runtime_schema = create_model(__base__=spec.args_model, runtime=ToolRuntime)
  → ProjectStructuredTool.from_function(...)
  → create_agent / LangChain ToolNode
  → Pydantic input validation + ToolRuntime injection
  → execute_project_tool
  → ToolSpec.handler
```

`ProjectToolVisibilityMiddleware` 读取 `project_spec.required_permissions`；
Agent builder 读取 `project_spec.idempotent/max_retries` 生成原生 retry middleware 分组。
`ProjectToolExecutionMiddleware` 在参数校验尚未进入 handler 时读取 spec、settings、audit sink，
生成项目 `invalid_arguments` 结果并审计。Governance middleware 消费结果和事件。
`project_policy_engine` 没有读取者，实际执行使用 construction closure 中的 policy。

全部 11 个生产 Local Tool 都声明 `output_model`。手写 `output_schema` 回退仅测试使用。
`args_schema` property、`resolved_output_schema()` 只有 `to_contract_dict()` 消费；
该 exporter 的唯一生产消费者为 `local_tool_contract()`，其他直接消费者在 tests。
manifest 和 runtime inspection 已消费原生工具 exporter。

## 字段归属

| 字段 | LangChain 原生能力 | 项目保留理由与本阶段处理 |
| --- | --- | --- |
| name | BaseTool.name | catalog 单处声明，构建时传入；导出取实际 tool.name；保留项目命名校验与审计身份 |
| description | BaseTool.description | catalog 单处声明，构建时传入；导出取实际 tool.description |
| args_model / args_schema | StructuredTool.args_schema，Pydantic 校验 | 业务 Pydantic 类是唯一 schema 定义；保留 args_model 供默认值/JSON 规范化与兼容 manifest 别名，删除独立 schema property |
| output_model / output_schema | Runnable.get_output_schema 不会自动校验 handler 的业务 output | 保留必填 output_model 和项目输出校验；删除手写 schema 回退，导出直接从模型派生 |
| handler | StructuredTool.func / coroutine | func 是治理入口，spec.handler 是通过治理后才执行的业务函数；不可直接绑定业务 handler 绕过治理 |
| runtime | ToolRuntime 注入；tool_call_schema 隐藏注入参数 | KEEP_RUNTIME_SCHEMA，原因见下；上下文仍由框架逐请求注入 |
| version | metadata 可携带声明 | 保留审计、参数指纹、runtime 结果中的版本语义 |
| risk_level | metadata 可携带声明，无项目策略语义 | 保留风险、审批、写工具审计 fail-closed |
| required_permissions | metadata 可携带声明，无授权机制 | 保留模型可见性和执行处权限检查 |
| timeout | 当前 StructuredTool 无等价同步 handler 截止机制 | 保留现有有界执行器、timeout 错误与不自动重试语义 |
| max_retries | ToolRetryMiddleware | 保留重试资格声明，框架拥有 retry loop |
| idempotent | 无项目幂等治理 | 保留重试资格和策略约束 |
| requires_approval | 无项目审批语义 | 保留审批判断 |
| sensitive_arg_fields | 无项目脱敏语义 | 保留输入审计及公开 trace 脱敏 |
| sensitive_output_fields | 无项目脱敏语义 | 保留输出摘要及审计脱敏 |
| audit_output_mode | 无项目审计语义 | 保留 none / summary / sanitized 模式 |

## 原生 runtime 注入实验与 KEEP 决策

未修改依赖。在普通 `StructuredTool.from_function(func, args_schema=Args)` 上直接声明
`runtime: ToolRuntime[ToolInvocationContext]`，同步 invoke 与异步 ainvoke 的结果一致；随后在真实 StateGraph/ToolNode 上以
context_schema 注入上下文，invoke、ainvoke、astream 共 9 组实验也复现同样结果：

| 业务模型 | 结果 |
| --- | --- |
| 非空、默认 extra=ignore | 成功；value 从字符串转为 int；context 和 tool_call_id 正确；LLM schema 无 runtime |
| 非空、extra=forbid | ValidationError：runtime 被当成多余字段拒绝 |
| 空业务参数模型 | TypeError：函数缺少 runtime |

安装源码证据：`BaseTool._parse_input()` 先调用业务模型 `model_validate(tool_input)`，之后才补回
signature 注入参数，因此 extra=forbid 会先失败。`BaseTool._to_args_and_kwargs()` 对空模型
直接返回 `(), {}`，跳过 runtime 转发。SQL 工具使用 extra=forbid，list_documents 使用空参数。

**KEEP_RUNTIME_SCHEMA**：保留统一的模型继承适配，不放宽业务参数规则，不给空参数工具加入假参数，
不通过私有方法 patch、额外 runtime 或分支适配器规避框架限制。

**KEEP_PROJECT_STRUCTURED_TOOL（缩减）**：删除无读取者的 policy instance field；
保留 spec/settings/audit sink 的排除序列化字段，确保框架校验失败仍使用该工具自己的治理依赖。
将重依赖放入 BaseTool.metadata 会使其进入 callback metadata，改成侧表又需要额外 lookup。
metadata 继续只包含可复制的 tool_version 声明。

**KEEP_TOOL_SPEC（缩减）**：仍是经 Pydantic 冻结校验的构建声明和治理输入。
拆成独立 policy + native tool 会增加跨模块连接代码，且无法消除本版本的 runtime schema 适配。
工具名、描述、业务输入模型仍在 catalog/SQL factory 中各声明一次；构建的框架字段是派生值。

## 本阶段删减边界

删除 ToolSpec 的重复 exporter、args schema property、输出 schema resolver、手写输出 schema
及其拷贝/JSON 校验，合并到唯一 `local_tool_contract()`。
该函数直接读取实际 LangChain name/description，业务模型引用用于兼容原始 schema title、alias、
extra 设置和 manifest 中的模型名称，取消 `get_input_schema().__bases__[0]` 反射。
输出 schema 始终来自 output_model；execute_project_tool 对每次成功输出做 Pydantic 校验。
测试用 Pydantic 模型表达 fixture 输出，不再要求生产保留手写 schema 回退。

不修改检索算法、RAG/Agent graph topology、MCP、SQL 安全策略、审批状态机、API/SQLite schema 或评测数据。


# FRAMEWORK-SLIM-3 RESULT

## STATUS

**PASS_WITH_FINDINGS**。安全删减已完成；runtime schema 和自定义子类均有实测保留理由。

## BASELINE

- branch: `refactor/tool-output-models`
- head_before: `2acdeaa50801303bead362b6ff12679b382bcb15`
- worktree_clean_before: `true`

## FINDINGS

- runtime_schema: **KEEP_RUNTIME_SCHEMA**，当前版本的严格模型及空模型限制见上。
- ProjectStructuredTool: 保留并从 4 个项目字段减至 3 个；policy 仅留在构建闭包。
- ToolSpec: 缩减，保留冻结构建声明和治理输入；未强行改成只有 policy 的新抽象。
- local_tool_contract: 唯一导出入口，取消先导出再覆盖及基类反射。
- LangChain name/description 直接用于导出；业务输入及输出 JSON schema 均由原始 Pydantic 模型派生。
- Pydantic 输入校验、ToolRetryMiddleware、ToolErrorMiddleware、ToolRuntime 注入及 ToolMessage
  原本已归框架管理，本阶段继续复用，没有新建 loop 或冒充本阶段新增迁移。
- execute_project_tool、权限/策略、审批、审计与 fail-closed、幂等治理、安全错误映射、
  输出校验、敏感输出处理和 ToolErrorCode 保留。

## CHANGES

| 文件 | 关键修改 |
| --- | --- |
| app/tools/contracts.py | output_model 改为必填；移除手写输出 schema 和重复导出 helper |
| app/agent/framework_tools.py | 删除 policy instance field；统一原生工具契约导出；注明 runtime schema 的保留原因 |
| app/tools/catalog.py | 删除仅转发同签名参数的 _handle_search_knowledge_base，直接引用原业务函数 |
| app/tools/execution.py | 每个成功输出均按必填 output_model 校验，删除不可达的无模型分支 |
| tests/test_tool_contracts.py | 改测唯一原生工具 exporter、输出模型必填及拒绝手写 schema |
| tests/test_tool_runtime.py | typed 输出 fixture；增加严格/空模型注入、sync/async/stream、真实 list_documents Agent Result、校验失败审计验证 |
| tests/test_local_tool_catalog.py | 验证导出与模型/工具顺序一致，保留副本隔离断言 |
| tests/test_tool_registry.py | 使用原生 exporter 与 Pydantic 输出 fixture，移除 fallback 分支断言 |
| tests/test_registry_driven_agent.py | 使用原生 exporter 与 Pydantic 输出 fixture |
| tests/test_tool_policy.py | 策略 fixture 声明输出模型 |
| tests/test_dynamic_tool_visibility.py | 可见性与伪造调用 fixture 使用 Pydantic 输出模型 |
| tests/test_tool_provider_visibility.py | 可见性 fixture 使用 Pydantic 输出模型 |
| tests/test_agent_knowledge_tool_integration.py | 知识工具 fixture 使用 Pydantic 输出模型 |
| tests/test_agent_error_recovery.py | 错误恢复 fixture 使用 Pydantic 输出模型 |
| tests/test_agent_runtime_inspection.py | inspection fixture 使用 Pydantic 输出模型 |
| tests/test_agent_budgets.py | 仅迁移输出 fixture；未运行该文件 pytest |
| tests/test_agent_router.py | 仅迁移输出 fixture；未运行该文件 pytest |
| tests/test_chat_service_agent_integration.py | 仅迁移输出 fixture；未运行该文件 pytest |
| tests/test_iterative_agent_loop.py | 仅迁移输出 fixture；未运行该文件 pytest |
| docs/tools/native-tool-declaration.md | 逐字段审计、框架限制、删减决策及验证记录 |

测试中不关注业务输出约束的 fixture 使用 Pydantic 原生 `RootModel[dict]`，
具体输出校验测试仍使用相应业务模型；没有新增生产 fixture facade。

## DELETED / SIMPLIFIED

- 删除 `ToolSpec.args_schema` property、`resolved_output_schema()`、`to_contract_dict()`、`copy_schema()`。
- 删除手写 `ToolSpec.output_schema`、对应 JSON/deepcopy 逻辑和 output_model 可空分支。
- 删除 `ProjectStructuredTool.project_policy_engine`。
- 删除 `_handle_search_knowledge_base()` 同签名转发 facade。
- 删除 `get_input_schema().__bases__[0]` 反射及契约二次覆盖。
- 未删除 class；未新增生产 class、registry、runtime 或 schema wrapper。

## LOC

计数口径：Git 跟踪的 `app/**/*.py` 物理行，包含空行/注释；tests/scripts/docs 不计入生产 LOC。

```text
production_python_loc_before: 20429
production_python_loc_after: 20377
net_python_loc_change: -52

tool_contract_loc_before: 167
tool_contract_loc_after: 113
```

测试 Python 净增 116 行（主要为 runtime/Agent 回归），全部修改的 Python 合计净增 64 行。
本阶段的净减少指标使用上面明确的生产代码口径。

## CORE VERIFICATION

修改前：runtime/catalog/SQL 三文件基线 21 passed。
修改后：15 个核心文件/测试组 199 passed；随后同一 runtime 文件补充真实 list_documents
sync/stream smoke 2 passed，累计 **201 个不同测试通过**。未扩大全量测试范围。

| 测试文件/选择器 | 结果 | 通过数 |
| --- | --- | --- |
| tests/test_tool_contracts.py | PASS | 39 |
| tests/test_tool_runtime.py | PASS | 27 |
| tests/test_local_tool_catalog.py | PASS | 6 |
| tests/test_tool_registry.py | PASS | 24 |
| tests/test_registry_driven_agent.py | PASS | 7 |
| tests/test_dynamic_tool_visibility.py | PASS | 11 |
| tests/test_tool_provider_visibility.py | PASS | 12 |
| tests/test_agent_knowledge_tool_integration.py | PASS | 3 |
| tests/test_sql_readonly_tools.py | PASS | 6 |
| tests/test_agent_streaming.py::test_stream_projects_native_command_tool_completion | PASS | 1 |
| tests/test_agent_runtime_inspection.py | PASS | 10 |
| tests/test_tool_policy.py | PASS | 4 |
| tests/test_tool_redaction.py | PASS | 3 |
| tests/test_tool_governance_integration.py | PASS | 37 |
| tests/test_agent_error_recovery.py | PASS | 11 |

- Ruff：全部 19 个实际修改 Python 文件通过，新增 smoke 后单独重查 runtime 测试文件通过。
- Compile：全部实际修改 Python 文件通过；4 个修改的生产模块 import smoke 通过。
- `git diff --check`：通过。
- 全量 contract + LLM schema 快照：SQL 关闭的 8 个工具、开启的 11 个工具，修改前后均完全相等。
- 原生 ToolNode 独立实验：普通模型 3 路径成功；严格/空模型在 3 路径分别复现 extra/missing-runtime 限制。
- MCP 未修改；现有 dynamic visibility 中的 MCP/local 共存 smoke 通过，未额外执行在线 discovery。

本次本机证据位于 `/tmp/framework-slim-3-audit/`：`before.json`、`after.json`、
`runtime_probe.py/json`、`core.xml`、`list_documents.xml`；该目录未加入仓库。

## BEHAVIOR COMPATIBILITY

| 边界 | 结论与证据 |
| --- | --- |
| tool names / descriptions | 8/11 工具快照完全一致，含三个 SQL 工具 |
| input schemas | manifest 和 LLM schema 完全一致，runtime 不可见，严格模型拒绝额外业务参数 |
| output schemas | 11 个生产工具完全一致；删除仅测试使用的手写 fallback |
| permissions | 声明一致，sync/async 伪造隐藏调用仍拒绝且 handler 不执行 |
| approval | 仍在 handler 前返回 approval_required |
| audit | REQUESTED/DENIED 配对调用 ID 正常；写操作 fail-closed、读操作审计失败标记保留 |
| retry | 声明一致；框架仍仅重试符合条件的失败，非幂等/不可重试错误执行一次 |
| timeout | 返回 timeout，原有截止及不自动重试行为通过 |
| Agent Result / tool_trace | calculator streaming、list_documents sync/stream、knowledge 工具与 trace 核心测试通过 |
| MCP coexistence | 现有 github/firecrawl 命名 fixture 在 sync/async 下与 local 共存通过；collision 生产代码未修改 |

## FINDINGS / DEFERRED

- 当前锁定框架需保留 runtime schema。升级依赖后的删除评估属于后续任务，本阶段未改版本。
- ToolSpec 仍包含构建字段，不是纯治理模型；继续拆分会增加连接代码，当前未引入双重人工维护源。
- 移除的内部 ToolSpec 构造/导出 API 已迁移仓库内调用者；生产 tool 外部 schema 与行为保持兼容。
- 唯一测试 warning 为既有 jieba/pkg_resources 弃用提示，无测试失败。
- 未运行整个 pytest suite、完整 streaming 文件、上述 4 个仅迁移 fixture 的测试文件、
  完整评测、在线模型/网络工具、真实 MCP discovery 或真实知识索引重建。

## WORKTREE

- worktree_clean_after: `false`，保留 4 个生产 Python 文件、15 个测试文件和本审计文档的变更。
- 分支与 HEAD 未变；未 commit、push、切换分支。

## FINAL DECISION

1. 真正减少了自研工具抽象：重复 exporter、schema fallback、无用字段及转发 facade 已删除，生产净减 52 行。
2. LangChain 的实际 name/description 成为导出直接来源；输入模型仍是唯一人工定义，框架负责可见 tool schema 与调用校验。
3. 未引入新的双重 Runtime、Registry 或 schema；保留的 runtime 继承模型是原有框架兼容边界。
4. 可以进入下一阶段；无需为进一步减少本阶段 LOC 增加 registry、私有反射或放宽安全约束。
