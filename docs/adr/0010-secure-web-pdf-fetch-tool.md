# ADR 0010：安全 Web PDF 页范围读取工具

- 状态：Accepted
- 日期：2026-07-23
- 决策者：Project Maintainer

## 背景

`web_search` 可以发现公开 PDF URL，`web_fetch` 则有意只接受 HTML、XHTML 和纯文本。把 PDF 解析塞进 `web_fetch` 会让一个工具同时维护网页正文和分页文档两套输出协议，也会模糊模型选错工具后的可恢复错误。因此 PR8 新增独立 `web_pdf_fetch@1.0.0`，同时保持现有 `create_agent` 模型—工具循环和治理边界。

## 决策

PDF 下载与 HTML 共用薄适配层 `app/tools/web_download.py`。该层从原 `web_fetch.py` 移入 URL 业务校验、`requests-hardened.Manager` 构造、只读 GET 参数、HTTP 状态映射、Content-Type 白名单、streaming 字节限制以及 Response/Session 关闭。它不是通用 HTTP 框架，也不引入第二套 Provider、重试、权限或审计系统。

PDF 解析固定使用项目已经锁定的 `pypdf==6.14.2` 公共 API：

```python
reader = PdfReader(BytesIO(body), strict=False)
text = reader.pages[index].extract_text()
```

不引入 `langchain-community` 的 `PyPDFLoader`、pdfplumber、PyMuPDF、Tika、Unstructured 或 OCR 栈。现有 pypdf 已能完成本阶段数字文本型 PDF 的页树、有限元数据和正文提取；额外 Loader 只会增加依赖与新的行为边界。

## 下载与资源边界

requests-hardened 继续负责 DNS/IP、私网与环回拒绝、连接固定和每次安全重定向。Session 继续设置 `trust_env=False`，只发 GET，不读取 `.netrc`，不发送 Cookie、Authorization 或用户 headers；代理仍只有维护者配置的 `WEB_FETCH_PROXY`，没有 PDF 专用代理。

HTML 和 PDF 共用连接超时、读取超时、最大重定向、User-Agent 和代理配置。PDF 使用独立的 `WEB_PDF_MAX_RESPONSE_BYTES=10000000`，因为 PDF 通常比单页 HTML 大；默认 10 MB 仍能在内存解析前提供明确上限。`WEB_PDF_MAX_DOCUMENT_PAGES=1000` 在正文提取前限制页树规模。ToolSpec 总体超时默认 45 秒，大于默认 3 秒连接加 10 秒读取预算，为有限下载和 pypdf 解析保留时间；handler 不自行重试，只有 ToolRuntime 最多重试一次明确可重试的网络失败。

允许 `application/pdf`、`application/x-pdf`，以及用于兼容错误 MIME 的 `application/octet-stream`。所有类型都必须满足 `body.startswith(b"%PDF-")`；文件扩展名不构成信任依据。PDF 类型无 magic、octet-stream 非 PDF 或 HTML 错误页均返回非重试 `invalid_result`。

## 页码、文本、元数据与截断

模型参数只有 `url`、1-based `start_page`、1–20 的 `max_pages` 和 1000–30000 的 `max_chars`。headers、cookies、proxy、timeout、password、strict、解析模式和下载字节上限都不暴露。

选定页按真实 1-based 页码输出：

```text
[Page 3]
page text

[Page 4]
next page text
```

空白页不进入正文或 `page_numbers`；页面间使用两个换行。实现只统一 CRLF、连续水平空白和过多空行，不摘要、翻译或调用 LLM 清洗。完整页范围先组合，再按 `max_chars` 截断；`truncated=true` 表示仍有组合内容未返回。

元数据只从公开 `reader.metadata` 映射读取 title、author 和 subject，非字符串变为空字符串，并做单行空白规范化与字段长度限制。不返回完整 metadata、producer、创建工具、pypdf 对象或原始字节。

## 加密、损坏、扫描件和超限行为

`reader.is_encrypted` 为真时直接返回非重试 `invalid_result`，不接收或尝试密码。pypdf 公共异常映射为不含 xref、object id、URL、正文或 traceback 的稳定 `invalid_result`。`start_page` 超出总页数返回 `invalid_arguments`；总页数超过服务端上限返回 `invalid_result`。

选定范围完全没有可提取文字时返回非重试 `invalid_result`，并说明可能是扫描件或图片型 PDF；本阶段不自动 OCR。工具不提取图片、表格结构、附件、表单、JavaScript 或外部链接。

## Prompt Injection 与审计

PDF 正文和元数据都是不可信外部数据，输出固定 `content_is_untrusted=true`。ToolSpec 明确禁止把 PDF 当作 system、developer 或 tool 指令、执行其中命令，或因为 PDF 要求调用其他工具；回答只把它作为参考，并注明 `final_url` 和相关页码。PDF 内容只通过普通 ToolMessage 返回，不拼入 system prompt。

URL 是敏感参数。SUMMARY 审计只记录 status code、content type、total/start page、pages read、content length、truncated 和不可信标记；不记录 requested/final URL、正文、元数据原值、PDF bytes、代理、headers、页对象或异常堆栈。

## ToolRuntime、Agent 和 RAG 边界

唯一执行链保持：

```text
Agent Model
→ RegistryToolsMiddleware
→ StructuredTool
→ ToolRegistry.dispatch
→ ToolRuntime.execute
→ web_pdf_fetch handler
→ shared requests-hardened download
→ pypdf
→ WebPdfFetchOutput
→ ToolMessage
→ Agent Model
```

`web_pdf_fetch` 是 READ_ONLY、幂等、无需审批，要求与 `web_fetch` 相同的 `web:fetch` 权限。Provider 只控制模型可见性，ToolRuntime 仍做最终授权、总体超时、有限重试、审计和输出校验。没有 PDF Graph 节点、Router/Planner/Executor 特判、自定义 ToolNode、线程池、ToolMessage 或权限引擎。

合法多工具链为 `web_search → web_pdf_fetch → final answer`。模型误用 `web_fetch` 时会收到 unsupported content type 的 ToolMessage，并可在同一 `create_agent` 循环中改用 `web_pdf_fetch`。

该工具不设置 `pending_rag_result`，不进入 RAG 子图，不写 ChromaDB、BM25、manifest 或知识库目录，不生成本地 references，也不永久缓存 PDF。Web PDF RAG、OCR、文件上传和持久化是后续独立能力。

## 后果

生产 Registry 从七个工具增加到八个；默认 `TOOL_DEFAULT_PERMISSIONS=knowledge_base:read` 下仍只暴露原五个本地工具。显式授予 `web:fetch` 时同时暴露 `web_fetch` 和 `web_pdf_fetch`。Agent Graph、Middleware、Router、Planner、Executor、API Schema、SQLite Schema、Checkpointer、RAG 算法和原七个工具契约保持不变。
