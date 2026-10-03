# MCP 知识入库来源

MCP 只用于知识入库阶段访问 GitHub / Firecrawl。问答 Agent 只注册本地
StructuredTool（知识库与按权限启用的 Web 工具），不读取 MCP Provider，不依赖
MCP 初始化，也不接收 MCP ToolMessage。

```text
ExternalKnowledgeSourceService
→ McpToolProvider（启动期只读 discovery cache）
→ invoke_mcp_tool → 官方 adapter BaseTool.ainvoke
→ McpToolCallInterceptor（来源授权、allowlist、超时、摘要审计）
→ GitHub / Firecrawl
→ ingestion serialization（脱敏、完整性和体积限制）
→ source payload / KnowledgeDocument → pending candidate
```

候选的人工审批和索引流程保持独立；fetch 不直接批准或写入索引。

## 配置与生命周期

`.env` 仅保存全局开关、静态配置路径与调用限制：

```dotenv
MCP_TOOLS_ENABLED=false
MCP_SERVERS_CONFIG_PATH=config/mcp_servers.local.json
MCP_DISCOVERY_TIMEOUT_SECONDS=5
MCP_TOOL_TIMEOUT_SECONDS=15
MCP_MAX_TOOLS_PER_SERVER=20
```

HTTP 应用仅在 `INGESTION_ADMIN_API_ENABLED=true` 时初始化来源 MCP；同时须设置
`MCP_TOOLS_ENABLED=true` 才会读取配置并连接远端。来源 smoke CLI 可独立初始化。
QA Agent 的构图与问答不依赖 MCP，关闭入库管理 API 时不执行 MCP discovery。
开启入库服务后的配置、认证引用或 discovery 失败仍会阻止启动。
关闭时清空 Provider；没有新增 lazy runtime、永久 session 或后台线程。

复制 [安全示例](../../config/mcp_servers.example.json) 到 Git 忽略的
`config/mcp_servers.local.json`，配置真实的来源端点和凭据，再启用所需 Server。
修改配置后需要重启。GitHub 入库精确 allowlist 是 `get_file_contents`；
Firecrawl 入库精确 allowlist 是 `firecrawl_scrape`。当前 Source Service 没有 crawl
调用、轮询或定时同步流程。

```json
{
  "version": 1,
  "servers": {
    "github": {
      "enabled": true,
      "transport": "http",
      "url": "https://api.githubcopilot.com/mcp/",
      "allowed_tools": ["get_file_contents"],
      "headers_env": {
        "Authorization": "MCP_GITHUB_AUTHORIZATION"
      }
    }
  }
}
```

`headers` 的直接值使用 `SecretStr`；`headers_env` 引用部署进程环境中的凭据。
同一个 Header（忽略大小写）不得同时声明两种来源。配置对象、日志、结果和审计
不回显凭据；只有建立官方 client connection 时解开 Secret。

配置只允许 Streamable HTTP；远端必须为 HTTPS，本机可用 HTTP。URL 禁止
userinfo、query、fragment；Header 禁止控制字符、空值以及 Host、Content-Length、
Transfer-Encoding 等传输控制字段。allowlist 非空、精确且无重复，不支持通配符。
停用的 Server 不解析认证环境变量、不建立连接、不 discovery。

旧部署迁移时需删除 Server 的 `enabled_for_default_principal` 和
`approval_required_tools` 字段，将 allowlist 收窄到实际入库工具。
旧 `MCP_MAX_TEXT_CHARS` / `MCP_MAX_STRUCTURED_CHARS` 环境变量已移除。
不保留旧配置兼容层；含有未知字段的 Server 配置会安全拒绝。

## 只读 discovery 与来源授权

官方 `MultiServerMCPClient` 保留 `tool_name_prefix=True`，工具命名为
`github_get_file_contents` / `firecrawl_firecrawl_scrape`。只缓存 allowlist 工具，
缺失名称、重复名称、跨 Server 前缀冲突、超时和连接失败均 fail closed。
不再与 Agent 本地 catalog 检查名称冲突。

`destructiveHint=true` 始终拒绝；`readOnlyHint=false` 也拒绝，只有现有
Firecrawl scrape 例外：interceptor 强制只接受
`url/formats/onlyMainContent/removeBase64Images`，禁止浏览器 actions 等参数。
部署者的精确 allowlist、只读凭据和可信 Server 仍是准入边界。
不存在外部写工具执行或工具审批恢复能力。

来源调用仍在 Provider 和执行边界校验 Server 权限。管理员 API 在验证独立管理密钥
后，仅为对应请求授予 `mcp:github` 或 `mcp:firecrawl`；它们是来源服务内部权限，
不加入 Agent 的 `TOOL_DEFAULT_PERMISSIONS`。默认聊天主体没有 MCP 特权。
执行边界还检查远端名称仍在当前 Server allowlist 中。

缓存工具保留官方 schema/coroutine，metadata 仅包含 `mcp_server` 和
`remote_tool_name`。直接来源调用的局部上下文承接 interceptor 的安全结果，
不通过文本标记、Agent ToolMessage 或序列化 profile 传递项目数据。

## 结果完整性与审计

所有来源结果带 `content_is_untrusted=true`。文本预算为 1,000,000 字符，
structured content 的总 JSON 预算为 2,000,000 字符，保留完整 Markdown、HTML 和
嵌套 metadata。最终 KnowledgeDocument 正文仍受入库领域的 1,000,000 字符限制。

- TextContent 与嵌入文本资源保留脱敏后的文本、类型、原始字符数和安全资源 URI。
- URI 移除 userinfo、query、fragment；资源链接不自动 fetch。
- Blob、image、audio 仅保留类型、MIME、体积摘要，不保留二进制内容。
- 不支持的内容类型安全拒绝，错误不含 payload repr。
- 真正截断、结构超限或无法安全净化的结果标记不完整，Source Service 拒绝创建候选。
- HTML 不完整时不得降级为 Markdown 入库；GitHub 正文必须唯一匹配请求的资源路径。

远端 `isError=true` 保留失败语义，错误文本脱敏且最多 2,000 字符。
超时和 transport 异常使用安全错误码，不回显底层连接细节。项目层不重试 MCP。
审计复用 ToolAuditEvent，保留主体、trace、工具名、Server、参数指纹、尝试次数、
耗时和状态，只记录参数字段名与输出长度等摘要。只读调用在审计写入失败时继续，
并标记 `audit.write_failed`。

## 来源 smoke

这个 CLI 只支持当前两个入库来源工具，不产生 candidate 或索引：

```bash
.venv/bin/python scripts/smoke_test_mcp_tools.py list
.venv/bin/python scripts/smoke_test_mcp_tools.py invoke \
  --tool github_get_file_contents \
  --args-json '{"owner":"example","repo":"docs","path":"README.md"}'
.venv/bin/python scripts/smoke_test_mcp_tools.py invoke \
  --tool firecrawl_firecrawl_scrape \
  --args-json '{"url":"https://example.com","formats":["markdown","html"],"onlyMainContent":true,"removeBase64Images":true}'
```

输出经过脱敏和预览限长，不打印原始参数、Header 或 client。preview 的截断与来源
完整性标记是不同概念。自动测试使用 fake client/官方 adapter 和临时路径；真实远端
连通性需单独 smoke 验证。
