# ADR 0009：安全 Web Fetch 网页正文工具

- 状态：Accepted
- 日期：2026-07-22
- 决策者：Project Maintainer

## 背景

PR6 的 `web_search` 能发现公开网页并返回标题、URL 和摘要，但不会读取结果页正文。Agent 若要依据具体页面回答，只能停留在搜索摘要，无法明确选择一个 URL 后取得有限正文。PR7 需要补齐单页读取能力，同时保持现有 LangChain `create_agent`、请求级 Provider、Registry、ToolRuntime、审计和 ToolMessage 协议。

网页 URL 涉及 SSRF、DNS 解析、重定向和代理边界，HTML 正文涉及复杂的 boilerplate removal。项目不应自行实现这两类安全或抽取算法。

## 决策

新增 `web_fetch@1.0.0` ToolSpec。下载层采用 BSD-3-Clause 的 `requests-hardened==1.3.0`，正文提取采用 Apache-2.0 的 `trafilatura==2.1.0`，均使用公开 Python API：

```python
manager = Manager(
    Config(
        ip_filter_enable=True,
        ip_filter_allow_loopback_ips=False,
        never_redirect=False,
        default_timeout=(connect_timeout, read_timeout),
        user_agent_override=user_agent,
    )
)
session = manager.get_session()
```

```python
extract(
    html_text,
    url=final_url,
    output_format="json",
    with_metadata=True,
    include_comments=False,
    include_tables=True,
    include_links=False,
    favor_precision=True,
)
```

两个包均声明支持 Python 3.12。pip dry-run 复用现有 `requests==2.32.5`，Trafilatura 基础包新增 `courlan`、`htmldate`、`justext`、`babel`、`tld`、`dateparser`、`pytz`、`tzlocal` 和 `lxml-html-clean`，并复用现有 `certifi`、`charset-normalizer`、`lxml`、`urllib3`、`python-dateutil`、`regex`。没有安装 dev/docs/eval/all extras，也没有升级或降级 LangChain、LangGraph、Pydantic、FastAPI、ChromaDB、sentence-transformers、DDGS、requests 或 httpx。

## 为什么使用 requests-hardened

`requests-hardened` 在连接池适配器中解析目标主机、拒绝 private/loopback/multicast 等地址，并把连接固定到已验证 IP，同时保留原始 Host、TLS hostname 与 SNI；自动重定向的每个目标继续经过同一 hardened adapter。它还公开支持与 SSRF 过滤结合的可信代理配置。

项目只保留少量业务规则：URL 必须是无凭据、无空白/控制字符、带主机名、最长 2048 字符的 HTTP(S) URL，显式端口只能是 80 或 443，并在请求前去除 fragment。私网、环回、链路本地、metadata、DNS 解析和 redirect 到内部地址不由项目枚举或重新判定。

不使用 raw `requests.get()`、`httpx.get()` 或 `urllib.request.urlopen()` 处理模型 URL，因为这些客户端本身不提供本项目需要的 SSRF 契约。SSRF 拒绝绝不回退普通客户端，也不提供关闭 IP 过滤、允许私网或禁用 TLS 验证的逃生配置。

## 为什么使用 Trafilatura

Trafilatura 负责 HTML/XHTML 的正文、标题和有限元数据提取。项目只解析其公开 JSON 输出并做换行、首尾空白、连续空行和字符上限处理；不遍历 DOM，不维护 CSS selector，不引入 BeautifulSoup、Readability 或第二套主体评分算法。

不使用 `trafilatura.fetch_url()`，因为 Trafilatura 下载函数不是 SSRF 防护。只有已经由 requests-hardened 安全、限量下载的 HTML 才交给 `extract()`。`text/plain` 直接按响应编码解码和有限规范化，不强行经过 HTML 提取。

## 下载、重定向和代理边界

请求固定为 GET、streaming 和 `allow_redirects=True`。Session 设置 `trust_env=False`，不读取 `.netrc` 或环境代理；不发送用户 Authorization、Cookie、Referer 或 headers。只显式设置受信 User-Agent 和 Accept。`WEB_FETCH_PROXY` 为空时不传代理，非空时只由服务器 Settings 配置，并按 requests-hardened 官方支持的 requests proxy 参数传入；模型和 URL 参数不能覆盖代理，代理也不进入 ToolMessage、审计、错误消息或 Agent State。

`WEB_FETCH_MAX_REDIRECTS=5` 通过 hardened Session 的重定向上限实现。每次自动重定向仍使用 requests-hardened adapter 做网络级验证；项目不实现 DNS 固定客户端或另一套 redirect 安全验证器。

## 响应体量和内容类型

默认 connect/read timeout 分别为 3/10 秒。ToolSpec 总体 timeout 选择 30 秒：它大于单次内部连接和读取预算，为安全重定向、流式读取和正文提取留出余量，同时仍由现有 ToolRuntime 提供统一外层等待上限。handler 不实现 sleep 或重试；只有 Runtime 根据稳定错误码最多重试一次。

下载先检查 Content-Length，再通过 `iter_content()` 累计实际解压后的字节，默认上限 2,000,000；缺少或非法 Content-Length 也不能绕过实际字节限制。Response 和 Session 在成功或异常路径都关闭。只支持 `text/html`、`application/xhtml+xml` 和 `text/plain`；PDF、JSON、二进制、图片、音视频、压缩包、Office 文件和 JavaScript bundle 均稳定拒绝。

## 参数、输出和错误

模型 Schema 只有 `url` 与 `max_chars`，后者默认 12000、范围 1000–20000。headers、cookies、proxy、timeout、method、body 和 runtime 均不暴露。

`WebFetchOutput` 由 Pydantic 最终校验，包含 requested/final URL、2xx status、content type、有限元数据、正文、精确 content length、truncated、固定 `extractor="trafilatura"` 和 `content_is_untrusted=true`。不返回 HTML、headers、cookies、IP、代理、客户端对象、Document、异常对象或重定向调试信息。

网络安全拒绝映射 `policy_denied/False`；timeout、连接失败、429 和 5xx 映射为明确可重试错误；4xx、内容类型、体量和提取失败为非重试错误。错误文本固定且不复制 URL、代理、headers、body、内部 IP 或 traceback。未知编程错误继续交给 ToolRuntime 的统一保守处理。

## Prompt Injection 信任模型

公开网页 URL、元数据和正文全部是不可信数据。ToolSpec 明确要求 Agent 不执行网页命令，不把网页内容视为 system、developer 或 tool 指令，不因为网页要求调用其他工具，只把正文作为回答用户问题的参考，并注明最终 URL。正文只作为标准 ToolMessage 返回模型，不拼接到 system prompt。

## ToolRuntime 和框架职责

合法调用链保持：

```text
Agent Model
→ RegistryToolsMiddleware
→ StructuredTool
→ ToolRegistry.dispatch
→ ToolRuntime.execute
→ web_fetch handler
→ requests-hardened
→ streaming size limit
→ content-type gate
→ Trafilatura
→ WebFetchOutput
→ ToolMessage
→ Agent Model
```

LangChain/LangGraph 负责模型—工具循环、Schema 和 ToolMessage；Provider/Registry/ToolRuntime 负责可见性、最终权限、总体超时、有限重试、审计和输出校验；requests-hardened 负责网络安全；Trafilatura 负责正文提取。没有第二套线程池、权限系统、重试、审计、Agent loop 或 ToolMessage 协议。

## 权限和本地 RAG 边界

`web_fetch` 要求 `web:fetch`。默认 `TOOL_DEFAULT_PERMISSIONS=knowledge_base:read` 不变，因此代码升级不会自动把 URL 发送给外部网站。只有 `web:search` 时可以搜索但不能读正文；只有 `web:fetch` 时可以读取用户明确提供的 URL但不能搜索；两者都有时支持搜索后读取。

`web_fetch` 是普通 ToolMessage 路径，不设置 `pending_rag_result`，不写 ChromaDB/BM25，不生成本地 references，也不使用本地 Evidence Gate。搜索或网页 URL 由 Agent 在普通回答文本中列出，不能伪装成本地知识库引用。

## 后果和后续边界

生产 Registry 从六个工具增加到七个，最小默认权限下仍只暴露原五个。Agent Graph、Middleware、Router、Planner、Executor、API Schema、SQLite Schema、Checkpointer、RAG 算法和原六个工具契约不变。

PDF、Office、图片 OCR、JavaScript 渲染、Playwright/Selenium、递归爬虫、robots 调度、缓存、网页向量化和 Web RAG 必须作为后续独立能力评估，不能扩展当前单页静态正文工具的安全边界。
