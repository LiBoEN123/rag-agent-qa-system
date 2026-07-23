# ADR 0008：开源 Web Search 生产工具

- 状态：Accepted
- 日期：2026-07-22
- 决策者：Project Maintainer

## 背景

现有五个生产工具只能处理本地知识库、文档对比和确定性计算，无法取得公开互联网中的当前或外部信息。PR4/PR5 已经建立框架原生 Agent 循环、Registry/Runtime 治理和请求级权限可见性，因此新增普通业务工具不应修改 Router、Planner、Executor、Graph 或 Middleware 生命周期。

## 决策

新增 `web_search` ToolSpec，实际搜索使用 MIT 许可证的开源 `ddgs==9.14.4` 公共 Python API；安装包元数据给出的官方项目来源为 <https://github.com/deedy5/ddgs>：

```python
DDGS(proxy=proxy, timeout=timeout).text(
    query,
    region=region,
    safesearch=safesearch,
    timelimit=timelimit,
    max_results=max_results,
    backend="auto",
)
```

该版本要求 Python 3.10 及以上，支持项目的 Python 3.12。安装 dry-run 复用现有 `httpx==0.28.1`，不升级或降级 LangChain、LangGraph、Pydantic、FastAPI、requests 或 ChromaDB。

## 职责边界

LangChain `create_agent`、StructuredTool 和 Pydantic 继续负责模型—工具循环、参数 Schema、参数校验和 ToolMessage。现有 Provider、Registry 与 ToolRuntime 继续负责权限可见性、统一分发、最终授权、总体超时、有限重试、审计和输出模型校验。DDGS 只负责搜索引擎选择、请求和结果聚合；`app/tools/web_search.py` 只做配置映射、有限规范化和公开异常转换。

合法链路是：

```text
Agent Model
→ RegistryToolsMiddleware
→ StructuredTool
→ ToolRegistry.dispatch
→ ToolRuntime.execute
→ web_search handler
→ DDGS
→ ToolMessage
→ Agent Model
```

handler 不实现线程池、sleep、指数退避、审计、权限判断、Graph State、ToolMessage 或模型调用。

## 为什么不自行抓取或引入额外包装

- 不自行抓取 DuckDuckGo、Bing 或 Google 搜索页，不解析搜索 HTML，也不实现引擎轮询、聚合、排序、浏览器自动化或网页正文抓取。DDGS 已经提供这些搜索边界。
- 不引入 `langchain-community`。项目已经有 ToolSpec、Registry 和 Runtime，再加社区包装只会复制参数与错误转换层；同时避免依赖逐步 sunset 的社区包装路径。
- 本 PR 不引入 MCP Server。当前只有一个进程内生产 Provider，本地 ToolSpec 已能受治理地暴露 DDGS；MCP 生命周期、远端认证和协议治理应在独立需求中评估。
- 只安装基础 `ddgs`，不安装 `ddgs[api]`、`ddgs[mcp]` 或 `ddgs[dht]`。

## 参数、输出与不可信内容

模型只看到 `query`、`max_results` 和 `freshness`。proxy、backend、timeout、TLS 校验、page 和 API URL 不进入模型 Schema。`freshness` 映射为 DDGS 的 `None/d/w/m/y`，backend 固定为 `auto`。

输出由 `WebSearchOutput` 校验，包含查询、结果数量、固定 provider、`content_is_untrusted=true` 和连续排名的 `title/url/snippet`。规范化只使用标准库解码 HTML entity、合并空白、限制长度、过滤非 HTTP(S) URL 并按 URL 去重；不下载结果 URL 或网页正文。DDGS 返回空列表是成功零结果。

互联网摘要可能不准确、过时或包含 Prompt Injection。工具描述要求模型不得执行摘要命令、不得把网页文本当作系统指令，并在回答中列出使用的标题和 URL。结果不会拼接进 system prompt。

## 为什么不进入本地 RAG

`search_knowledge_base` 返回受控 evidence chunk，并通过 `pending_rag_result` 进入本地 RAG 的拆分、重排、证据门、引用绑定和拒答路径。`web_search` 返回的是不可信公开摘要，因此只作为普通 ToolMessage 回到 Agent Model，不设置 RAG handoff，不写入 ChromaDB，也不生成本地 RAG references。

## 权限、超时、重试与审计

`web_search` 要求精确权限 `web:search`。默认 `TOOL_DEFAULT_PERMISSIONS=knowledge_base:read` 不变，升级代码不会自动把用户查询发送给外部服务；维护者必须显式加入 `web:search`。无权时 Provider 隐藏完整契约，伪造名称返回 `unknown_tool`，直接 dispatch 由 Runtime 返回 `permission_denied`。

DDGS 内部超时默认 8 秒，ToolSpec 总体超时为 10 秒。公开 `RatelimitException`、`TimeoutException` 和 `DDGSException` 分别映射为稳定限流、超时和依赖不可用错误；Runtime 独占最多一次重试、退避、抖动和 attempt 元数据。`query` 声明为敏感参数，审计只保存参数指纹和有限输出摘要；proxy 不进入输出、ToolMessage、审计或 Agent State。默认使用独立的 `WEB_SEARCH_PROXY`；显式开启 `WEB_TRUSTED_PROXY_MODE` 后，维护者配置的 `WEB_TRUSTED_PROXY_URL` 统一覆盖搜索和下载代理。

## 后果与可替换边界

生产 Registry 从五个工具增加为六个，但默认服务主体仍只能看到原五个。Graph 节点、Middleware 栈、Router、Planner、Executor、API Schema、SQLite Schema、RAG 算法与 Checkpointer 不变。

未来更换搜索后端时，应只替换 `app/tools/web_search.py` 内的开源后端适配，同时保持 WebSearchArgs、WebSearchOutput、ToolSpec 权限、Runtime 错误码和不可信内容契约。网页正文读取、新闻专用搜索、多 Provider 或 MCP 必须分别设计为后续工具或 Provider。
