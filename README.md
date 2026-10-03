# Agentic RAG 知识库问答系统

一个面向中文制度类文档的 Agentic RAG 问答系统，支持文档解析、混合检索、重排、引用生成、Agent 路由、多轮追问改写、可观测性和离线评测。

当前实现的职责、调用链和持久化边界见
[当前系统架构](docs/architecture/current-system.md)。
架构演进见 [2026 年 Agent 演进归档](docs/history/agent-evolution-2026.md)，已接受的
Model Governance 边界见 [ADR 0006](docs/adr/0006-framework-native-agent-runtime.md)。
外部知识入库来源接入见
[MCP 知识入库来源](docs/tools/mcp-client.md)。

本项目基于 FastAPI 构建，围绕中文知识库问答场景实现了一条完整的 RAG 工程链路：读取本地 `.txt` / `.md` / `.pdf` / `.docx` 文档，按文档结构切分 chunk，构建 BM25 + SentenceTransformer Embedding 混合检索索引，使用 CrossEncoder reranker 重排候选证据，再通过 OpenAI-compatible Chat Completions 模型生成带引用的答案。聊天通过唯一公开入口 `/chat/stream` 执行 LangChain `create_agent` 模型—工具循环；传入 `session_id` 时还会结合 SQLite 中最近的会话历史对短追问做 Query Rewrite。

## 项目亮点

- 中文知识库问答：面向中文制度、校务、规范类文档，支持本地文档解析、检索、回答和引用溯源。
- 混合检索：BM25 负责关键词精确匹配，SentenceTransformer Embedding 负责语义召回，提升中文问答召回稳定性。
- CrossEncoder rerank：对候选 chunk 做二阶段重排，缓解初召回排序不准的问题。
- ChromaDB 持久化：chunk 文本、metadata 和 embedding 持久化保存，服务重启可复用索引缓存。
- 普通 Python 入口 Router：由 Agent Runtime 在每次调用时执行一次，只处理空问题、少量稳定直答和明确澄清；其他自然语言请求统一进入 `agent`，具体工具由 Agent Model 根据当前请求可见的动态 Tool Schema 选择。
- 框架原生 Agent Runtime：直接执行 LangChain `create_agent` 返回的 CompiledStateGraph，由框架管理唯一的模型—工具循环、参数校验、ToolMessage 和调用上限。
- 权限感知的工具可见性：Agent 仅使用本地 StructuredTool，Middleware 按 `tool_mode` 与 `required_permissions` 过滤知识库和 Web 工具。
- Agent 循环防护：工具错误总数、连续相同调用、连续相同错误和模型/工具累计耗时均有独立预算；终止错误对同批后续调用 fail-closed，并补齐关联 ToolMessage。
- 多轮追问改写：同一 `session_id` 下通过 SQLite 保留最近会话历史，短追问和指代问题会先改写成独立检索问题。
- 工具调用：Agent 支持知识库检索、文档发现、公开网页搜索和正文提取；MCP 仅服务 GitHub / Firecrawl 知识入库。
- 引用绑定和拒答机制：模型声明的 evidence ID 经 grounding 校验后投影为真实引用；未知 ID 失败关闭。
- 流式聊天：唯一公开入口 `POST /chat/stream`，响应 `Content-Type: text/event-stream`，提供 progress、tool、安全答案增量和权威 final SSE 事件。
- 可观测性：每次聊天请求记录 `trace_id`、阶段耗时、状态、工具调用、引用数量和 LLM 调用次数。
- bad case 日志：对模型格式异常、引用异常、工具异常、索引未就绪等失败样例做结构化记录。
- 离线评测闭环：支持批量生成预测结果，并从答案质量、检索命中、行为约束和可选 LLM Judge 语义判断等维度评测。

## 系统架构（已冻结）

Phase 7 起架构状态为 `frozen`，当前唯一主链是：

```text
FastAPI / Chat Router
→ ChatService / Query Rewrite
→ Agent Runtime（普通 Python Router）
→ LangChain create_agent（唯一 Agent Loop）
→ Project Governance Middleware
→ 缓存的 Local StructuredTool + 请求级可见性 Middleware
→ execute_project_tool
→ Tool Handler → Document evidence → ToolMessage → 下一轮 Agent
→ Framework State → AgentRunResult
→ ChatStatusPolicy → HTTP JSON / SSE
```

本地工具固定经过 `StructuredTool → execute_project_tool → ToolSpec.handler`。MCP
仅供知识入库获取 GitHub / Firecrawl 来源，永不进入 Agent Loop。LangGraph 只存在于
`create_agent` 内部循环；RAG Evaluation 复用同一个 ChatService，不存在第二个回答工作流。

Phase 6B 的 Model private marker 是已接受设计，不是待迁移 TODO。详细当前事实、稳定边界
与禁止无证据重构的规则见[当前系统架构](docs/architecture/current-system.md)；历史原因只在
[演进归档](docs/history/agent-evolution-2026.md)和 [ADR](docs/adr/) 中维护。

## 核心能力

| 模块 | 已实现能力 |
| --- | --- |
| 文档读取 | 支持 `.txt`、`.md`、`.pdf`、`.docx`；文本类文件自动识别编码，Local/Upload PDF/DOCX 共用 Docling，保留正文及表格结构 |
| 文档切分 | 普通文本按段落和标点递归切分；Markdown 按标题层级切分并保留标题路径 |
| 混合检索 | jieba + BM25 关键词召回，SentenceTransformer Embedding 语义召回 |
| 重排 | CrossEncoder reranker 对候选 chunk 重新排序 |
| 多轮检索 | Agent 阅读 ToolMessage 后决定下一轮 query/top_k，生产与评估共用同一 Agent |
| 多轮追问 | `session_id` 通过 SQLite 维持最近会话历史，对短追问、指代问题做 Query Rewrite |
| 答案生成 | 调用 OpenAI-compatible Chat Completions，要求模型返回 JSON |
| 引用绑定 | 将实际声明的 Document.id 校验并映射回文件、`chunk_id`、`snippet` 和分数 |
| Agent Runtime | 直接执行唯一的 LangChain `create_agent` CompiledStateGraph；框架负责模型—工具循环、参数校验、ToolMessage 和模型/工具调用上限 |
| Agent 路由 | `app.agent.runtime` 是唯一 Router 调用者；正常请求输出 `agent`，`direct_answer` / `clarify` 在普通 Python 中安全短路 |
| 工具发现 | 不可变本地 ToolSpec/StructuredTool catalog 按配置缓存；LangChain Model Middleware 按请求权限过滤模型可见工具 |
| Agent Guardrail | 工具错误总数、连续相同调用/错误、累计模型与工具耗时预算、同批 fail-closed、无悬空 ToolMessage |
| 工具调用 | 默认 catalog 包含七个生产工具。模型可见集合由请求 tool_mode 与权限共同决定 |
| 可观测性 | `trace_id`、请求耗时、阶段耗时、工具调用日志、metrics 汇总接口 |
| 失败分析 | `logs/bad_cases.jsonl` 记录值得分析的失败样例 |
| 评测 | 支持规则评测、Chroma/chunks 索引取证和可选 LLM Judge 语义评估 |
| 索引缓存 | ChromaDB 持久化 chunk、metadata、embedding，manifest 记录索引构建状态 |

## RAG 流程说明

1. 文档加载：从 `RAG_FILE_PATH` 读取 `.txt` / `.md` / `.pdf` / `.docx` 文档。
2. 文本清洗：统一清理不可见字符、空白和文档噪声。
3. 文档切分：Markdown 先按标题层级切分并保留标题路径，普通文本按段落和标点递归切分。
4. 建立索引：构建 BM25 关键词索引，并使用 SentenceTransformer 生成 embedding；向量、chunk 和 metadata 持久化到 ChromaDB。
5. 多轮改写：`/chat/stream` 如果传入 `session_id`，会读取最近历史；短追问或指代问题会先改写成独立问题。
6. Agent 执行：ChatService 将改写后的问题交给普通 Python Router，再进入唯一的 LangChain `create_agent`；`direct_answer` 和入口 `clarify` 不调用 Agent Model。
7. 检索与观察：`search_knowledge_base` 复用 BM25、Chroma/Embedding 和 CrossEncoder 重排，返回带稳定 ID 的 LangChain Document；Agent 阅读 ToolMessage 后可修改 query 再检索。
8. 结构化回答：`ToolStrategy(FinalAgentResponse)` 原生校验并重试，最终输出 `answer`、`no_evidence` 或 `clarify` 及实际使用的 `evidence_ids`。
9. 引用绑定：代码检查 ID 是否来自本次成功的 KB、原文读取、Web/PDF 工具消息，只按声明顺序投影真实引用。已有 Document 的 answer 不允许漏引用；未知 ID 返回 `MODEL_ERROR`。
10. 状态与展示：证据不足映射 `NO_RELEVANT_INFO`，澄清映射 `NEED_CLARIFICATION`；两项 confidence 保持 `0.0`。校验通过的答案在 SSE 中作为一次完整 delta 发送，随后返回权威 final。

旧 Graph、独立拆分模型、context builder 与引用 heuristic 已删除。Agent 根据观察动态检索多个子问题；诊断检索分数门独立保留。

## 历史 RAG 单项评测结果

下表是 Evaluation Baseline v2 引入前记录的 65 题 RAG 单项结果，仅用于历史对照；
它不包含 Agent Core、Online tools、多轮会话或统一 manifest，也不代表当前代码与
当前模型配置下的最新成绩。当前统一评测入口、复现约束和结果比较方式见
[Evaluation Baseline v2](docs/evaluation/baseline-v2.md)。

| Metric              |  Score |
| ------------------- | -----: |
| Evaluation Cases    |     65 |
| Final Score         | 0.9605 |
| Key Point Recall    | 0.9353 |
| Gold Chunk Hit@K    | 0.9846 |
| Gold Chunk Recall@K | 0.9756 |
| Gold Document Hit@K | 0.9846 |
| Behavior Score      | 0.9846 |
| Forbidden Hit       |      0 |

该历史评测集覆盖 `single_doc`、`cross_doc`、`partial_answer`、`not_in_doc`、
`trap` 五类问题；当时的主要优化方向是 `partial_answer` 场景和新旧政策冲突类
`trap` 问题。

## 工程化设计

- 配置通过 `.env` 管理，`.env.example` 提供默认模板，回答/Agent 模型、追问改写模型、拆分模型和评测 Judge 均使用 OpenAI-compatible Chat Completions 配置。
- 应用内部模型调用统一由 LangChain `ChatOpenAI(base_url=...)` 访问 OpenAI-compatible 服务，各角色原有模型环境变量仍然有效。
- 生产 Agent 使用 `ToolStrategy(FinalAgentResponse)` 与 evidence ID 校验；Query Rewrite 和 Evaluation Judge 继续使用 `ModelGateway`。比较文档时分别调用 `search_document`，由 Agent 阅读两组证据并生成结论。
- `requirements.txt` 保留顶层依赖声明，`requirements.lock` 固定 Python 3.12 验证过的版本；Docker 和 CI 均使用 lock 文件安装依赖。
- `.importlinter` 使用 Import Linter 2.13 固化应用包的单向依赖、顶层无环、基础层、HTTP 框架和 Settings 框架边界；CI 在 Ruff 后执行全量 `lint-imports`。
- 支持 embedding / rerank 模型本地缓存，默认缓存目录为 `./.cache/huggingface/hub`，也可通过 `MODEL_CACHE_DIR`、`HF_HOME` 等环境变量调整。
- 支持 ChromaDB 持久化，默认将向量、chunk 文本、metadata 和文档注册信息保存到 `data/chroma/`。
- 显式知识同步由写侧基于 manifest 判断重建；QA 启动只加载索引。文件变化增量处理，chunk/embedding/索引版本变化触发重建。
- 多轮会话历史保存在 `SQLITE_DB_PATH` 指定的 SQLite 数据库中，每个 `session_id` 默认读取最近 6 条 user/assistant 消息，服务重启后仍可继续使用。
- Agent trace 通过 `AgentRunResult.tool_trace` 返回，供 API、Evaluation 和运行时诊断使用。旧数据库可能保留历史 `agent_runs`、`agent_steps` 表；应用不再读写或删除这些表，新数据库不再创建它们。
- `/chat/stream` 请求支持继承或生成 `trace_id`，响应头返回 `X-Trace-Id`，真实总耗时位于 `final.data.total_cost_ms`。
- metrics 记录 Query Rewrite、路由、Agent、检索、问题拆分、重排、上下文构造、LLM 调用、结构化结果规范化等阶段耗时和 LLM 调用次数。
- bad case 日志用于定位失败样例，包括模型结构化输出校验失败、引用异常、工具异常、内部异常和索引未就绪。

模型客户端、唯一 `create_agent` Agent Loop 和 KB 检索工具彼此分层；RAG Evaluation 复用生产链，检索、Rerank、ChromaDB 和 grounding 合同保持。

## MCP 知识入库来源

Agent 只注册本地知识库工具与按权限启用的 Web 工具。MCP 仅供
`ExternalKnowledgeSourceService` 抓取 GitHub 文件和 Firecrawl 页面，不参与问答循环。

MCP 默认关闭。复制 `config/mcp_servers.example.json` 到 Git 忽略的
`config/mcp_servers.local.json`，配置来源端点、凭据及精确只读 allowlist。
HTTP 应用同时启用 `INGESTION_ADMIN_API_ENABLED` 与 `MCP_TOOLS_ENABLED` 才会执行
来源 discovery；来源 smoke CLI 可独立初始化。QA Agent 不依赖 MCP 初始化。

官方 client、discovery cache、来源权限、超时、脱敏和摘要审计继续保留。
GitHub 使用 `get_file_contents`，Firecrawl 使用受只读参数限制的 `firecrawl_scrape`。
完整内容经 Source Service 校验后产生 pending candidate，再经显式审批和索引。
配置迁移、来源完整性限制和 smoke 命令见 [MCP 知识入库来源](docs/tools/mcp-client.md)。

## Web Search

`web_search` 使用开源 `ddgs` 的公共 Python API 搜索公开互联网中的当前或外部信息，只返回有限的 `title`、`url` 和 `snippet`。它与 `search_knowledge_base` 的边界不同：后者读取受控知识库并返回 Document evidence；二者都通过 ToolMessage 回到唯一 Agent。

Web Search 默认关闭可见性。要让默认 `/chat/stream` 服务主体看到它，必须显式配置：

```env
TOOL_DEFAULT_PERMISSIONS=knowledge_base:read,web:search
```

搜索 query 会发送到外部搜索服务，可能受搜索服务限流、地区设置和网络条件影响。搜索摘要可能不准确、过时或含恶意指令，输出始终带有 `content_is_untrusted=true`；模型不得执行摘要中的命令，使用搜索信息回答时应列出实际使用的标题和 URL。代理配置不会写入工具输出或审计。

真实网络验证只通过维护者手动运行，不进入 Pytest/CI：

```bash
python scripts/smoke_test_online_tools.py search \
  --query "LangGraph latest release" \
  --max-results 3
```

脚本默认经过 `StructuredTool → execute_project_tool → DDGS` 生产链路，不调用 Agent 模型、不读取或修改知识库；缺少 `web:search` 权限时会直接给出启用提示。

## Web Fetch

`web_fetch` 用于已知具体公开 URL 后读取单个网页正文；通常由 Agent 在 `web_search` 返回候选标题、URL 和摘要后选择调用，也可读取用户明确提供的 URL。下载层固定使用 `requests-hardened==1.3.0` 做 SSRF IP 过滤和逐跳安全重定向，正文提取固定使用 `trafilatura==2.1.0`，而权限、总体超时、一次有限重试、审计和输出校验仍由现有 `ToolRuntime` 负责。`search_knowledge_base` 则只读取本地受控知识库并进入 RAG 引用路径。

模型只能提供 `url` 和 `max_chars`，不能提供 method、headers、Cookie、Authorization、proxy、timeout 或 body。应用层只接受 HTTP(S)、无 credentials、无控制字符且端口为 80/443 的 URL。默认 `WEB_TRUSTED_PROXY_MODE=false`，私网、环回、链路本地、metadata 地址、DNS 解析和 redirect 目标继续由 `requests-hardened` 校验。响应以 streaming 方式读取解压后内容并受 `WEB_FETCH_MAX_RESPONSE_BYTES` 限制，只接受 HTML、XHTML 和纯文本，不支持 PDF、二进制文件或 JavaScript 渲染。

受控本地开发环境可显式设置 `WEB_TRUSTED_PROXY_MODE=true` 和 `WEB_TRUSTED_PROXY_URL=http://127.0.0.1:<port>`。开启后统一代理覆盖 `WEB_SEARCH_PROXY`、`WEB_FETCH_PROXY`，搜索、HTML 和 PDF 工具共用该代理；下载层仍禁用环境代理并保留 URL、超时、响应体量、Content-Type 和资源关闭约束，但把 DNS/IP、私网及重定向目标的网络策略委托给受信代理。该代理必须自行拒绝局域网、环回、metadata 地址和危险重定向，不能把此模式当作生产安全默认值；关闭开关即恢复原有 hardened SSRF 校验。

URL、页面元数据和正文全部是不可信输入。输出固定包含 `content_is_untrusted=true`；模型不得执行页面中的命令、不得把正文视为 system/developer/tool 指令，也不得因为页面要求而调用其他工具，只能将其作为回答用户问题的参考，并在回答中注明实际 `final_url`。网页不会写入 ChromaDB/BM25，不进入本地 RAG，其 Document 在被最终结构化回答实际引用后，按现有公开字段投影为 Web references。

默认权限仍只有 `knowledge_base:read`，因此三个 Web 工具都默认不可见。两项 Web 权限彼此独立：只有 `web:search` 时只能搜索；只有 `web:fetch` 时可用 `web_fetch` 和 `web_pdf_fetch` 读取明确 URL；两者都有时才支持搜索后读取：

```env
TOOL_DEFAULT_PERMISSIONS=knowledge_base:read,web:search,web:fetch
```

真实网络验证仅由维护者手动执行，不进入 Pytest/CI：

```bash
python scripts/smoke_test_online_tools.py fetch \
  --url "https://example.com/" \
  --max-chars 4000
```

脚本默认经过 `StructuredTool → execute_project_tool → requests-hardened → Trafilatura`，只显示有限正文预览和安全元数据，不显示代理、headers、cookies 或完整正文。

## Web PDF Fetch

`web_pdf_fetch` 用于读取已知公开 PDF 的指定连续页范围。它与 `web_fetch` 保持明确分工：`web_fetch` 只处理 HTML、XHTML 和纯文本并继续拒绝 PDF；`web_pdf_fetch` 只处理 PDF，并使用项目已经锁定的 `pypdf==6.14.2` 提取数字文本。两者共用 `app/tools/web_download.py` 中基于 `requests-hardened==1.3.0` 的下载、逐跳重定向、可信代理、streaming 和资源关闭边界；默认模式实施本地 SSRF 校验，显式受信代理模式则按上一节所述委托网络策略。

模型只能提供 `url`、1-based `start_page`、1–20 的 `max_pages` 和 1000–30000 的 `max_chars`。服务端另以 `WEB_PDF_MAX_RESPONSE_BYTES`、`WEB_PDF_MAX_DOCUMENT_PAGES` 和 `WEB_PDF_TOOL_TIMEOUT_SECONDS` 控制下载字节、总页数和总体 ToolRuntime 时间。工具接受 `application/pdf`、`application/x-pdf`，并仅在内容具有 `%PDF-` 签名时兼容 `application/octet-stream`；所有允许类型最终都检查 PDF magic。

正文按实际页码返回 `[Page N]` 边界，空白页不列入 `page_numbers`，title、author 和 subject 只做有限单行规范化。加密、损坏、页数超限和无可提取文本的 PDF 会返回稳定非重试错误；扫描件/图片型 PDF 不自动 OCR，也不接收密码、提取图片/附件/脚本或调用 LLM 清洗。

PDF 文本和元数据固定标记 `content_is_untrusted=true`，只能作为回答参考；模型不得执行 PDF 中的命令或把内容视为 system/developer/tool 指令，回答时应注明 `final_url` 和相关页码。`web_pdf_fetch` 要求 `web:fetch` 权限，经过普通 `StructuredTool → execute_project_tool → handler → ToolMessage` 链路；它不进入本地 RAG、不写 ChromaDB/BM25，其 Document 在被最终结构化回答实际引用后，按现有公开字段投影为 PDF references。

真实网络验证仅由维护者手动执行，不进入 Pytest/CI：

```bash
python scripts/smoke_test_online_tools.py pdf \
  --url "https://example.com/document.pdf" \
  --start-page 1 \
  --max-pages 3 \
  --max-chars 8000
```

脚本默认经过 `StructuredTool → execute_project_tool → requests-hardened → pypdf`，只显示安全有限 URL、页码、长度和正文预览，不显示代理、headers、PDF bytes 或完整正文。

## Agent Runtime 检查

以下命令通过 LangGraph/LangChain 公开 introspection 与 Schema API 查看当前 Runtime，不调用模型、工具或网络，也不会创建 Checkpoint 数据库：

```bash
python scripts/inspect_agent_runtime.py
python scripts/inspect_agent_runtime.py --format json
python scripts/inspect_agent_runtime.py --format mermaid
python scripts/inspect_agent_runtime.py --check
lint-imports
```

图结构、维护路径、文件职责和常见问题定位见 [Agent Runtime 维护指南](docs/agent-runtime-maintenance.md)。Runtime inspection 只确认 `.importlinter` 及关键合同声明存在；真实 import graph 由独立 `lint-imports` 验证。CI 会同时运行二者，阻止模块反向依赖、旧自定义循环节点回归、关键框架节点消失、工具契约失效或核心文件超过架构预算。

真实模型 Tool Calling、MCP、Online 与 SQL 冒烟命令见维护指南及对应工具文档；这些
环境相关检查不会进入默认 Pytest/CI。

## 文档入口

- [当前系统架构](docs/architecture/current-system.md)
- [Agent Runtime 维护指南](docs/agent-runtime-maintenance.md)
- [模块依赖边界](docs/architecture/module-boundaries.md)
- [Evaluation Baseline v2](docs/evaluation/baseline-v2.md)
- [2026 年 Agent 演进归档](docs/history/agent-evolution-2026.md)
- [ADR 0001：开源 Agent 技术栈](docs/adr/0001-open-source-agent-stack.md)
- [ADR 0002：迭代式 Agent Loop](docs/adr/0002-iterative-agent-loop.md)
- [ADR 0003：工具治理边界](docs/adr/0003-tool-governance-boundary.md)
- [ADR 0004：基于注册表的工具发现](docs/adr/0004-registry-driven-tool-discovery.md)
- [ADR 0005：Agent 错误恢复、预算与循环防护](docs/adr/0005-agent-error-recovery-and-budgets.md)
- [ADR 0006：框架原生 Agent Runtime](docs/adr/0006-framework-native-agent-runtime.md)
- [ADR 0007：权限感知的工具可见性](docs/adr/0007-permission-aware-tool-visibility.md)
- [ADR 0008：开源 Web Search 生产工具](docs/adr/0008-open-source-web-search-tool.md)
- [ADR 0009：安全 Web Fetch 网页正文工具](docs/adr/0009-secure-web-fetch-tool.md)
- [ADR 0010：安全 Web PDF 页范围读取工具](docs/adr/0010-secure-web-pdf-fetch-tool.md)
- [ADR 0011：单一框架 Agent 执行链](docs/adr/0011-single-framework-agent-execution-path.md)
- [ADR 0012：Graph 独占确定性 Router](docs/adr/0012-graph-owned-deterministic-routing.md)
- [ADR 0013：单一平铺 Agent Runtime State](docs/adr/0013-single-flat-agent-runtime-state.md)
- [ADR 0014：单一 Agent Result 契约](docs/adr/0014-single-agent-result-contract.md)
- [ADR 0015：pydantic-settings 配置](docs/adr/0015-pydantic-settings-configuration.md)
- [ADR 0016：可执行模块依赖边界](docs/adr/0016-enforced-module-dependency-boundaries.md)

## 快速启动

建议使用 Python 3.12，与 Docker 和 CI 环境保持一致。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
cp .env.example .env
```

Windows PowerShell 可使用：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.lock
copy .env.example .env
```

锁文件已经包含 `ddgs==9.14.4`、`requests-hardened==1.3.0` 和
`trafilatura==2.1.0` 基础包；未安装这些依赖的 dev、docs、eval、all、MCP
等 extras，也不需要 `langchain-community` 包装。

将知识库文档放到 `data/kb/`，然后编辑 `.env` 中的模型配置。先缓存模型并同步索引，再启动服务：

```bash
python scripts/cache_models.py
python -m app.rebuild_kb
uvicorn app.main:app --reload
```

启动后访问：

```text
Swagger: http://127.0.0.1:8000/docs
Health:  http://127.0.0.1:8000/health
```

QA 启动只读取已有 Chroma 索引和本地模型缓存，不扫描或解析知识文件。修改本地知识后再次执行同步命令，并重启 QA 加载新快照。

## Vue 前端

Vue 3 + TypeScript 是主要 Demo / 管理界面，提供智能问答、检索调试、知识库、知识入库和系统诊断。需要 Node.js 22.12+（已用 22.21.1 验证）。先启动 FastAPI 后端：

```bash
uvicorn app.main:app --reload
```

另一个终端启动前端：

```bash
cd frontend
npm install
npm run dev
```

Vue 地址为 `http://localhost:5173`，FastAPI 为 `http://127.0.0.1:8000`。
API 使用同源相对路径，经 Vite proxy 访问后端；需要修改目标时复制
`frontend/.env.example` 为 `frontend/.env`，设置 `VITE_DEV_PROXY_TARGET` 后重启 Vite。
页面使用 hash 路由（例如 `/#/ingestion`），避免页面路径与 `/chat/stream`、`/search` API 冲突。

知识入库管理需在**后端**配置 `INGESTION_ADMIN_API_ENABLED=true` 和
`INGESTION_ADMIN_API_KEY=<你的管理员密钥>`，然后重启后端。密钥在入库页的密码框中输入，
仅保留在浏览器内存，刷新页面即清除；不要放入 `VITE_*`、URL 或仓库。
在「待审批知识」刷新 Agent 已生成的候选，查看安全预览后批准，再单独点击「写入知识库」。
「索引记录」可检查结果；GitHub / Firecrawl 手动提案及本地文件上传位于「来源提案」。
Upload、GitHub、Firecrawl 可在候选详情确认「移出知识库」，撤除同一来源当前入库的全部内容。
候选和审批历史保留；撤库成功后全量重建不会恢复该来源，再次手动写入后才会恢复。
「索引记录」区分 index/remove，并可按操作筛选。
选择「本地文件」后可选择或拖拽单个 PDF、DOCX、TXT、MD 文件；上传只生成待审批候选，
仍需批准后手动写入知识库。默认上限 20 MiB，通过 `INGESTION_UPLOAD_MAX_BYTES` 配置；
扫描 PDF 暂不支持 OCR，原始文件解析后删除，候选持久化全文与表格结构；
历史纯文本 PDF/DOCX 候选继续按正文切分，重新上传才能获得结构。有 `structured_content`
时，连续文本复用 LangChain splitter，表格保留标题/表头并按完整连续行分组；
`pipeline_version` 从 2 升至 3，旧索引在显式同步时通过写侧 force rebuild 迁移。
Firecrawl 显式提案请求 Markdown + cleaned HTML，经同一 Docling parser 得到
`ParsedDocument`，直接复用现有 table-aware chunks；禁用二次抓取、图片读取与渲染。
HTML 原文不保存，结构随 candidate_json 持久化。历史 Firecrawl 候选保持纯文本，
force rebuild 不重抓 URL，也不会自动升级结构；需重新抓取、生成候选、审批并索引。
本次 `pipeline_version` 保持 3。详见
[结构化解析、持久化与切分](docs/ingestion/structured-documents.md)。

聊天只使用 `/chat/stream`，`answer_delta` 用于增量展示，`final.data.response` 更新本轮权威结果，
`done` 表示流结束。中断不会自动重试。session ID 保存在 sessionStorage，
聊天记录保留在内存；「新建对话」生成新 ID 并清空记录。

前端验证：`npm run typecheck`、`npm run test -- --run`、`npm run build`。
`npm run preview` 可在 `http://localhost:4173` 本地预览构建结果，仍使用相同代理；
`dist/` 是静态资源，本阶段不包含生产托管部署。

## 环境变量

常用配置如下，完整示例见 `.env.example`。

应用配置由唯一的冻结 `AppSettings` 模型管理。生产代码通过惰性缓存入口
`get_settings()` 获取一个进程级快照；导入 `app.config` 不读取 `.env`，也不检查
API key、路径或创建任何目录。配置来源优先级为：

```text
显式 AppSettings(...) 参数 > 环境变量 > 项目根目录 .env > 字段默认值
```

测试和一次性脚本应显式构造
`AppSettings(_env_file=None, ...)`，并把实例传给现有依赖边界，以免维护者本地
`.env`、代理或权限污染结果。配置模型只负责解析和纯校验；真实
API key 及可写持久化路径等外部前提由 FastAPI lifespan 在启动时通过
`validate_startup_environment()` 检查。

`get_settings()` 的结果不会随环境变量或 `.env` 动态变化。受控刷新必须先调用
`clear_settings_cache()`，并同步清理持有旧快照的 Model Factory、Tool Registry、
Tool Runtime 与 Agent Runtime 缓存；生产服务通常直接重启进程。不要把 `.env`
当作动态热更新配置。

| 配置项 | 说明 |
| --- | --- |
| `RAG_FILE_PATH` | 知识库目录，示例：`./data/kb` |
| `INDEX_STORE_DIR` | 索引构建 manifest 目录，示例：`./data/manifest` |
| `CHROMA_PERSIST_DIR` | ChromaDB 本地持久化目录，示例：`./data/chroma` |
| `CHROMA_COLLECTION_NAME` | 保留的旧 ChromaDB collection，默认 `mini_rag_chunks` |
| `CHROMA_INDEX_COLLECTION_NAME` | LangChain 增量索引 collection，默认 `mini_rag_chunks_v2` |
| `TOP_K` | `/search` 默认召回条数；不控制 Agent Chat 的 Tool Call |
| `CHUNK_SIZE` | chunk 最大长度 |
| `CHUNK_OVERLAP` | 相邻 chunk 重叠长度 |
| `EMBEDDING_MODEL_NAME` | SentenceTransformer embedding 模型 |
| `RERANK_MODEL_NAME` | CrossEncoder rerank 模型 |
| `MODEL_CACHE_DIR` | embedding/rerank 模型权重本地缓存目录 |
| `LLM_API_KEY` | 回答模型 API Key；兼容旧变量 `DEEPSEEK_API_KEY` |
| `MODEL_NAME` | 回答模型名称 |
| `LLM_API_URL` | OpenAI-compatible Chat Completions 地址 |
| `TEMPERATURE` | 回答模型温度 |
| `LLM_MAX_TOKENS` | 回答模型最大输出 token |
| `LLM_TIMEOUT` | 单次回答模型请求超时秒数 |
| `LLM_MAX_RETRIES` | 回答模型失败重试次数 |
| `WEB_SEARCH_REGION` | DDGS 地区设置，默认 `us-en` |
| `WEB_SEARCH_SAFESEARCH` | 安全搜索级别：`on`、`moderate` 或 `off`，默认 `moderate` |
| `WEB_SEARCH_TIMEOUT` | DDGS 内部 HTTP 超时秒数，默认 `8`，不得超过工具总体超时 |
| `WEB_SEARCH_PROXY` | 可选 DDGS 代理；留空时传入 `None`，只在确实需要时配置 |
| `WEB_TRUSTED_PROXY_MODE` | 是否启用统一受信代理模式，默认 `false`；开启时委托 DNS/IP 网络策略，只适合受控环境 |
| `WEB_TRUSTED_PROXY_URL` | 统一受信 HTTP(S) 代理 URL；模式开启时必填并覆盖两个 Web 工具独立代理 |
| `WEB_FETCH_CONNECT_TIMEOUT` | Web Fetch 连接阶段超时秒数，默认 `3` |
| `WEB_FETCH_READ_TIMEOUT` | Web Fetch 单次响应读取超时秒数，默认 `10` |
| `WEB_FETCH_MAX_RESPONSE_BYTES` | Web Fetch 解压后响应正文上限，默认 `2000000` bytes |
| `WEB_FETCH_MAX_REDIRECTS` | Web Fetch 最大安全重定向次数，默认 `5` |
| `WEB_FETCH_USER_AGENT` | 维护者配置的非空 User-Agent，不接受模型输入 |
| `WEB_FETCH_PROXY` | 可选受信任代理；留空时禁用，禁止使用未经信任的用户输入 |
| `WEB_PDF_MAX_RESPONSE_BYTES` | Web PDF 解压后响应上限，默认 `10000000` bytes |
| `WEB_PDF_MAX_DOCUMENT_PAGES` | Web PDF 允许的文档总页数上限，默认 `1000` |
| `WEB_PDF_TOOL_TIMEOUT_SECONDS` | Web PDF ToolRuntime 总体超时，默认 `45` 秒，必须大于连接与读取超时之和 |
| `MCP_TOOLS_ENABLED` | 是否启用入库 MCP，默认 `false`；HTTP 生命周期仅在开启入库管理 API 时初始化 |
| `MCP_SERVERS_CONFIG_PATH` | MCP Server 静态 JSON 路径；默认是 Git 忽略的 `config/mcp_servers.local.json`，可用项目内相对路径或绝对路径覆盖 |
| `MCP_DISCOVERY_TIMEOUT_SECONDS` / `MCP_TOOL_TIMEOUT_SECONDS` | 启动 discovery 与单次 MCP Tool 超时，默认 `5` / `15` 秒 |
| `MCP_MAX_TOOLS_PER_SERVER` | 每 Server 精确 allowlist 上限，默认 `20` |
| `INGESTION_ADMIN_API_ENABLED` | 是否注册永久知识运营 API，默认 `false`；它不是 Agent Tool |
| `INGESTION_ADMIN_API_KEY` | `SecretStr` operator/service 管理密钥；启用运营 API 时必填，通过 `X-Ingestion-Admin-Key` 发送 |
| `INGESTION_UPLOAD_MAX_BYTES` | 单文件上传及 DOCX 展开大小上限，默认 `20971520`（20 MiB），必须为正整数 |
| `TOOL_DEFAULT_PERMISSIONS` | 默认 Agent 主体权限；默认仅 `knowledge_base:read`，Web 需显式 `web:search` / `web:fetch` |
| `USE_LLM_JUDGE` | 评测脚本是否启用 LLM Judge，默认 `false` |
| `JUDGE_API_KEY` | 评测 Judge 模型 API Key；为空时可复用回答模型 Key |
| `JUDGE_MODEL` | 评测 Judge 模型名称 |
| `JUDGE_API_URL` | 评测 Judge 模型接口地址 |
| `JUDGE_TIMEOUT` | Judge 模型请求超时秒数 |
| `JUDGE_MAX_RETRIES` | Judge 模型失败重试次数 |
| `JUDGE_MAX_TOKENS` | Judge 模型最大输出 token |
| `AGENT_MAX_ITERATIONS` | 迭代式 Agent 最大模型轮次，默认 `6` |
| `AGENT_MAX_TOOL_CALLS` | 迭代式 Agent 最大工具调用总数，默认 `10` |
| `AGENT_MAX_TOOL_ERRORS` | 累计工具失败上限，第三次失败后停止，默认 `3` |
| `AGENT_MAX_CONSECUTIVE_SAME_TOOL_CALLS` | 连续完全相同调用最多执行次数，默认 `2` |
| `AGENT_MAX_CONSECUTIVE_SAME_TOOL_ERRORS` | 连续相同调用与错误码停止阈值，默认 `2` |
| `AGENT_MAX_TOTAL_COST_MS` | Agent 模型与工具累计协作式耗时预算（毫秒），默认 `120000` |
| `SQLITE_DB_PATH` | 会话历史、ingestion candidate/index run 共用的 SQLite 文件路径，默认 `./data/app.sqlite` |
| `ENABLE_QUERY_REWRITE` | 是否启用多轮追问改写，默认 `true` |
| `QUERY_REWRITE_API_KEY` | 追问改写模型 API Key；为空时复用回答模型 Key |
| `QUERY_REWRITE_MODEL_NAME` | 追问改写模型名称 |
| `QUERY_REWRITE_API_URL` | 追问改写模型接口地址 |
| `QUERY_REWRITE_TIMEOUT` | 追问改写模型请求超时秒数 |
| `QUERY_REWRITE_MAX_TOKENS` | 追问改写模型最大输出 token |
| `QUERY_REWRITE_TEMPERATURE` | 追问改写模型温度 |
| `VECTOR_THRESHOLD` | 召回证据不足时拒答的向量分数阈值 |

### DeepSeek 示例

```env
RAG_FILE_PATH=./data/kb
INDEX_STORE_DIR=./data/manifest
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION_NAME=mini_rag_chunks
CHROMA_INDEX_COLLECTION_NAME=mini_rag_chunks_v2

TOP_K=5
CHUNK_SIZE=600
CHUNK_OVERLAP=120

EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding-0.6B
RERANK_MODEL_NAME=BAAI/bge-reranker-v2-m3
MODEL_CACHE_DIR=./.cache/huggingface/hub

LLM_API_KEY=your-api-key
MODEL_NAME=deepseek-chat
LLM_API_URL=https://api.deepseek.com/v1/chat/completions
TEMPERATURE=0.1
LLM_MAX_TOKENS=1024
LLM_TIMEOUT=60
LLM_MAX_RETRIES=2

AGENT_MAX_ITERATIONS=6
AGENT_MAX_TOOL_CALLS=10
AGENT_MAX_TOOL_ERRORS=3
AGENT_MAX_CONSECUTIVE_SAME_TOOL_CALLS=2
AGENT_MAX_CONSECUTIVE_SAME_TOOL_ERRORS=2
AGENT_MAX_TOTAL_COST_MS=120000
SQLITE_DB_PATH=./data/app.sqlite

ENABLE_QUERY_REWRITE=true
QUERY_REWRITE_API_KEY=
QUERY_REWRITE_MODEL_NAME=deepseek-chat
QUERY_REWRITE_API_URL=https://api.deepseek.com/v1/chat/completions
QUERY_REWRITE_MAX_TOKENS=256


USE_LLM_JUDGE=false
VECTOR_THRESHOLD=0.1
```

### Ollama 示例

```env
LLM_API_KEY=ollama
MODEL_NAME=qwen2.5:7b
LLM_API_URL=http://127.0.0.1:11434/v1/chat/completions

QUERY_REWRITE_API_KEY=ollama
QUERY_REWRITE_MODEL_NAME=qwen2.5:7b
QUERY_REWRITE_API_URL=http://127.0.0.1:11434/v1/chat/completions

```

## API 示例

| 接口 | 方法 | 作用 |
| --- | --- | --- |
| `/health` | GET | 健康检查 |
| `/docs-preview` | GET | 查看已加载文档列表 |
| `/chunks-preview` | GET | 查看 chunk 预览，支持 `max_len`、`limit`、`file_name` |
| `/chat/stream` | POST | 唯一聊天 HTTP 入口；SSE `final.data.response` 是权威结果 |
| `/search` | POST | 只执行检索和 rerank，不调用回答模型 |
| `/decide` | POST | 只返回 Agent 路由决策，不执行工具或 RAG |
| `/kb_summary` | POST | 返回知识库 ready 状态、文档数、chunk 数和 `index_source` |
| `/metrics-summary` | GET | 汇总最近 N 条聊天请求指标 |

### `/chat/stream` SSE 示例

```bash
curl --no-buffer -X POST "http://127.0.0.1:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"question":"缓考应该如何办理？"}'
```

通过 Schema 校验的问题返回 `HTTP 200 text/event-stream`，正常事件边界为
`start → progress/tool/answer_delta... → final → done`。空问题也建流，事件为
`start → final → done(completed)`，其中 `final.data.http_status=400`、
`final.data.response.code=EMPTY_QUESTION`。请求 Schema 错误仍在建流前返回 HTTP 422 JSON。
`final.data.response` 是权威 ChatResponse，`done` 表示流结束。Agent 的 answer/no_evidence 经校验后，
以一次完整 `answer_delta` 发出；当前不提供逐 token 答案流。Router 短路、澄清或错误可能没有
`answer_delta`，最终结果始终在 `final`。事件字段、安全过滤、断开语义
和 Python 冒烟命令见 [聊天流式 API 文档](docs/api/chat-streaming.md)。

核心响应字段：

| 字段 | 说明 |
| --- | --- |
| `trace_id` | 本次请求追踪 ID，也会写入响应头 `X-Trace-Id` |
| `data.question` | 原始用户问题；如果触发 Query Rewrite，改写后的问题可在 `tool_trace` 中查看 |
| `data.answer` | 最终回答 |
| `data.references` | 引用来源列表，包含 `file_name`、`source_path`、`chunk_id`、`snippet`、`score`、`rerank_score` |
| `data.retrieval_confidence` | 0 到 1 的检索证据置信度估计 |
| `data.answer_confidence` | 0 到 1 的最终回答置信度估计 |
| `data.status` | `ok`、`no_evidence`、`empty_index`、`error`、`clarify`、`out_of_scope` |
| `data.need_clarification` | 是否需要用户补充问题 |
| `data.clarification_question` | 澄清追问 |
| `data.reason` | 当前状态原因 |
| `data.action` | Agent 选择的动作 |
| `data.tool_result_summary` | 工具或 RAG 路径摘要 |
| `data.tool_trace` | Query Rewrite、路由、工具调用、检索命中、最终状态等过程信息 |
| `data.router_llm_used` | 兼容字段；确定性 Router 不调用模型，因此固定为 `false` |
| `data.answer_llm_used` | 本次请求是否调用最终回答 LLM |
| `data.llm_call_count` | 本次请求累计 LLM 调用次数 |

### 多轮追问与 Query Rewrite

`/chat/stream` 只接受 `question`、`session_id`、`tool_mode`；未知字段由 Pydantic `extra="forbid"` 拒绝并返回 HTTP 422。旧 `top_k`、`decompose_top_k` 已退役。Agent 每轮通过工具 schema 自主选择 query/top_k，`search_knowledge_base.top_k` 和 `search_document.top_k` 范围仍为 1–10。`TOP_K` 只用于诊断搜索；独立 Decompose 配置已删除。

`/chat/stream` 的 `session_id` 是可选字段。为空时系统按单轮问题处理；非空时系统会读取该会话最近 6 条 user/assistant 消息，并在问题像短追问或包含“这个、那个、它、上面、刚才”等指代词时尝试改写成独立问题。

改写逻辑的优先级是：先用规则判断是否需要改写；如果 `ENABLE_QUERY_REWRITE=true` 且改写模型配置可用，就调用改写模型；如果改写模型不可用或返回异常，会回退为“基于上一轮问题的追问”格式，保证主链路仍可继续。

### `/search`

`/search` 适合调试召回质量。它只执行检索和 rerank，不调用回答模型。

```json
{
  "question": "普通本科四年制专业学生现在最长学习年限是不是 8 年？",
  "top_k": 5
}
```

返回结果包含 `reference_id`、`file_name`、`source_path`、`chunk_id`、`position`、`snippet`、`score`、`bm25_score`、`vector_score`、`rerank_score`。

### `/decide`

`/decide` 用于观察 `app.agent.runtime` 使用的确定性 Router 对问题产生的结果；它复用同一 Router 函数，但不执行后续 Agent、工具或 RAG。

```json
{
  "question": "1 + 2 * 3 等于多少？"
}
```

只会返回 `agent`、`clarify` 或 `direct_answer`。具体工具由后续 Agent Model 根据当前可见 Schema 选择。

`web_search` 不作为 Router 关键词 action。普通问题先进入 `agent`，再由 Agent Model 根据动态 Tool Schema 和当前请求权限决定是否调用 Web Search。

`web_fetch` 同样不作为 Router action，也不在 Query Rewrite 或 Runtime 中增加 URL 特判；它由 Agent Model 在普通工具循环中按权限和 Schema 选择。

`web_pdf_fetch` 也不增加 PDF Router action 或 Runtime 特判；模型可根据 `web_fetch` 的类型错误 ToolMessage 在同一框架循环中改用 PDF 工具。

### `/kb_summary`

返回当前知识库是否 ready、文档数量、chunk 数量、文件列表和 `index_source`。其中 `index_source=chroma` 表示当前索引来自 ChromaDB 缓存，`memory` 表示刚重建完成，`none` 表示没有可用索引。

### `/metrics-summary`

```text
GET /metrics-summary?limit=20
```

用于查看最近请求的状态分布、成功数、拒答数、错误数、平均延迟、平均 LLM 调用次数，以及 Router / Answer LLM 的使用次数。

## Agent 路由与工具

正常 `/chat/stream` 的确定性 Router action：

| action | 路径 |
| --- | --- |
| `agent` | 普通问题统一进入 LangChain `create_agent` |
| `direct_answer` | 由 Answer Composer 生成系统能力说明，不把 Router 的内部 reason 直接暴露为用户回答 |
| `clarify` | 明确指代不明时追问；完全空问题由 ChatService 在进入 Runtime 前生成 final，业务 http_status 为 400 |

如果 `/chat/stream` 传入 `session_id`，Query Rewrite 会在 Agent Router 之前执行；Router 看到的是改写后的有效问题，响应中的 `data.question` 仍保留原始输入，改写细节记录在 `data.tool_trace` 的 `query_rewrite` step 中。

Router 不调用独立模型，也不解析具体工具 action。普通问题进入 `create_agent` 后，Agent Model 根据可见性 Middleware 绑定的 StructuredTool/Pydantic Schema 选择本地 catalog 工具，执行统一经过 `execute_project_tool`。

## 索引构建与缓存

知识索引写入与 RAG 查询独立运行，持久 Chroma 是共享边界：

```text
python -m app.rebuild_kb [--force-rebuild]
→ IndexingService → LangChain Indexing API → Chroma + SQLRecordManager

启动 QA → 读取已有 Chroma → BM25 / embedding query / CrossEncoder → Agent KB tools
```

Local/GitHub/Firecrawl/Upload 共用 `mini_rag_chunks_v2`，各有独立 RecordManager namespace。
Local 文件变化通过显式同步增量处理；chunk 参数、embedding 或索引版本变化触发重建。
`--force-rebuild` 恢复最新成功 INDEX 对应候选，排除最新成功 REMOVE 来源；不重新抓取外部内容。
Reranker、Prompt 和回答模型变更不要求重建向量索引。独立 CLI 同步后需重启 QA。
组合应用 admin INDEX/REMOVE 成功后由 API composition root 刷新查询 runtime。
MCP 只供知识入库获取来源，永久入库必须经过显式 proposal、approval 和 index。
详见 [LangChain-native Incremental Indexing](docs/ingestion/langchain-indexing.md)。

## 日志与可观测性

`/chat/stream` 会为每次请求生成或继承 `trace_id`：

- 请求头可传入 `X-Trace-Id`。
- 响应头返回 `X-Trace-Id`；完整业务耗时位于 `final.data.total_cost_ms`。
- 每个 SSE 信封及 `final.data.response` 都包含 `trace_id`。

日志文件：

| 文件 | 说明 |
| --- | --- |
| `logs/metrics.jsonl` | 每次聊天的耗时、状态、action、引用数、工具信息、Query Rewrite、LLM 与流事件统计 |
| `logs/tool_calls.jsonl` | 每次工具调用的输入、输出、耗时和是否成功 |
| `logs/bad_cases.jsonl` | 模型结构化输出校验失败、引用异常、工具异常、内部异常、索引未就绪等失败样例 |

日志汇总脚本：

```bash
python scripts/summarize_logs.py
```

默认读取 `logs/metrics.jsonl` 和 `logs/bad_cases.jsonl`，并将汇总结果写到 `outputs/week9_summary.md`。

## 测试与评测

执行与 CI 一致的静态检查和测试：

```bash
python -m compileall -q app scripts tests
ruff check app tests scripts
lint-imports
python scripts/inspect_agent_runtime.py --check
python -m pytest -q
```

Evaluation Baseline v2 支持 Core Agent、RAG 和可选 Online Web 评测。离线校验同时检查
Agent 数据合同与当前 65 题 RAG 数据，不调用模型、索引或网络：

```bash
python scripts/run_evaluation_baseline.py --validate-only
```

Agent Evaluation 检查最终 action/status、必要工具能力、答案关键信息、引用来源和本轮
真实 evidence grounding；仅 multi-tool/recovery case 约束工具顺序，多轮按最终结果评测。

默认真实运行 Core Agent 和本地 RAG；Online 工具与 LLM Judge 必须显式启用：

```bash
python scripts/run_evaluation_baseline.py \
  --agent-cases data/eval/agent_eval_cases_v2.json \
  --rag-eval-file data/eval/rag_eval_questions_v2_semantic_fixed.json \
  --output-dir outputs/evaluation_baseline_v2/initial-core \
  --retrieval-k 5
```

`retrieval-k` 只控制 RAG 指标的评分 cutoff，不传入 Agent Tool Call。

每次运行会生成 commit/数据/索引/模型/Prompt/工具合同 manifest、Prediction v3 与评分 cutoff、Agent/RAG 指标、覆盖矩阵、多原因失败报告和 Markdown 摘要。RAG 报告
headline 指标为 prediction coverage、retrieval Gold Recall@K、reference Gold Recall
及可选 DeepEval GEval 的 Answer Correctness；Judge coverage 单独表示判分覆盖率。工作区不干净时默认拒绝正式
运行；实验运行必须显式使用 `--allow-dirty` 并会被标记。两轮结果可比较：

```bash
python scripts/compare_evaluation_runs.py \
  --baseline outputs/evaluation_baseline_v2/run-a \
  --candidate outputs/evaluation_baseline_v2/run-b
```

完整说明见 [Evaluation Baseline v2](docs/evaluation/baseline-v2.md)。RAG prediction、
Judge、检索打分和报告实现都属于 `scripts/evaluation/` 内部包，不提供独立兼容
CLI。需要语义级答案正确性评分时，由统一 Runner 显式启用 LLM Judge：

```bash
python scripts/run_evaluation_baseline.py \
  --output-dir outputs/evaluation_baseline_v2/rag-llm-judge \
  --rag-only \
  --use-llm-judge \
  --judge-model deepseek-chat
```

整次运行的唯一机器可读汇总为 `baseline_summary.json`；Agent 明细保留 `agent/per_case.jsonl`。
RAG 输出保留 `predictions.csv`、`prediction_failures.jsonl` 和 `per_question_report.csv`
（亦为 Judge 缓存来源），不再生成重复 summary。比较先检查 manifest 中的运行控制条件，
缺失或不兼容默认失败；Git commit、Prompt 和工具合同变化允许比较。
显式 `--allow-incompatible` 可强制比较，JSON 和 Markdown 会标记不兼容及原因。
正式 Prediction 只接受 Schema v3，保存观察/引用 chunk IDs 与安全计数；旧 v1/v2 输入
不再读取。Runtime 失败写入 sidecar，缺失预测降低 coverage、召回计零，Answer Correctness 保持 `None`。
专项表格题集和 `table_diagnostics.py` 属于 diagnostics，不参与主 Baseline。

Core 不评测真实联网能力；Judge 关闭时不具备完整语义答案质量判断；Online 结果受
外部服务和内容变化影响，不适合作为稳定 CI 门禁。

HTTP 性能测试是独立入口；`concurrency=1` 即串行基线，问题只从 cases 文件读取：

```bash
python scripts/benchmark_chat_api.py \
  --base-url http://127.0.0.1:8000 \
  --cases data/eval/performance_cases.json \
  --concurrency 1,2,3,5 \
  --repeats 5 \
  --warmup 1 \
  --timeout 120 \
  --output-dir outputs/performance/run-001
```

它只生成 `requests.jsonl`、`summary.json` 和 `summary.md`。真实外部工具连通性使用
`smoke_test_online_tools.py` 的 `search`、`fetch`、`pdf` 子命令；模型供应商 Tool
Calling 协议单独使用 `smoke_test_agent_tool_calling.py`。两者用途不同，且都不进入
Pytest/CI。架构合同由 `inspect_agent_runtime.py --check` 检查，运行日志由
`summarize_logs.py` 汇总。没有可用模型、网络或 API Key 时只运行离线验证与 mock
回归测试，不把环境失败或空结果当作正式质量数据。

## 适合面试讲解的设计取舍

- 为什么不用纯向量检索，而是 BM25 + Embedding：中文制度文档里经常出现政策名、条款号、年份、专业名称等精确关键词，BM25 对这些字面匹配更敏感；Embedding 能补足语义相近但表述不同的问题，两者混合比单一路径更稳。
- 为什么引入 reranker：初召回阶段更关注“找全”，排序可能受关键词密度或向量相似度误差影响；CrossEncoder 可以基于 query-chunk 交互重新判断相关性，把更适合作答的证据排到前面。
- 为什么需要引用绑定：RAG 问答不能只给自然语言答案，还需要能追溯到文件、chunk 和片段，便于用户核验，也便于排查模型是否引用了错误证据。
- 为什么证据不足时拒答：制度类问答对准确性要求高，低相关度证据强行生成容易造成误导；系统通过阈值、引用绑定和模型输出检查，在证据不足时返回 `no_evidence`。
- 为什么普通请求默认进 Agent：Router 看不到 Provider 的动态工具 Schema，若它先按自然语言猜成 RAG，新工具或换一种说法就可能永远到不了 Agent。Agent-first 让具体工具选择只发生在动态 Schema 已绑定之后。
- 为什么 Query Rewrite 只对追问触发：完整问题直接进入检索可以减少额外 LLM 成本；短追问和指代问题才结合历史改写，能在成本和多轮体验之间取得平衡。
- 为什么要做离线评测：RAG 优化不能只看单个 demo，需要用覆盖多场景的评测集持续观察召回、答案关键点、拒答行为和 forbidden 命中情况，形成可复现的优化闭环。

## 项目结构

```text
.
├── app/
│   ├── main.py                  # 正式 ASGI 入口：app.main:app
│   ├── api/
│   │   ├── application.py       # FastAPI 组合根、lifespan 和 Router 注册
│   │   ├── chat.py              # /chat/stream HTTP Router
│   │   ├── streaming.py         # SSE 信封、白名单投影与编码
│   │   ├── health.py            # 轻量进程存活检查
│   │   ├── middleware.py        # HTTP trace 和端到端耗时
│   │   ├── http_logging.py      # 显式 Uvicorn access-log 适配
│   │   ├── schemas.py           # 请求、响应和引用数据模型
│   │   ├── diagnostics.py       # 检索、知识库和指标诊断接口
│   │   ├── responses.py         # ChatOutcome 到 HTTP Schema 的单一路径
│   │   └── lifecycle.py         # RAG/Agent 初始化和 Runtime 关闭
│   ├── contracts/
│   │   └── chat.py              # 唯一 Chat Result/HTTP/API/metrics 状态策略
│   ├── agent/
│   │   ├── __init__.py          # Agent 包入口，不聚合业务函数
│   │   ├── router.py            # 确定性入口路由
│   │   ├── rules.py             # Router 问题规范化规则
│   │   ├── query_rewrite.py     # 多轮追问识别和 Query Rewrite
│   │   ├── state.py             # Observation 与内部 Agent 状态
│   │   ├── results.py           # Framework 输出到 AgentRunResult 的唯一构造/校验边界
│   │   ├── answer.py            # 最终答案合成
│   │   ├── framework_agent.py   # 唯一 create_agent Agent Loop 构建
│   │   ├── framework_middleware.py # 项目特有 Agent 治理 Middleware
│   │   ├── framework_tools.py   # 动态框架工具与 Context 适配
│   │   ├── tool_events.py       # Framework 事件到 observation/trace/metrics 的投影
│   │   ├── guardrails.py        # 治理规则、预算、错误分类、Stop Reason 与指纹
│   │   ├── runtime.py           # 请求准备、Router 与 Agent 缓存
│   │   └── streaming.py         # 直接 astream 与安全 Agent 事件投影
│   ├── indexing/               # 独立索引写侧：service/langchain/chunking/manifest
│   ├── rag/
│   │   ├── __init__.py          # RAG 包说明
│   │   ├── runtime.py           # 只读索引加载、查询状态与检索 API
│   │   ├── retrieval.py         # 混合召回与重排
│   │   ├── policy.py            # 诊断检索分数门
│   │   ├── chroma.py            # ChromaDB 持久化
│   │   └── debug.py             # 文档和 chunk 调试查询
│   ├── tools/
│   │   ├── __init__.py          # 包说明；业务从具体职责模块导入
│   │   ├── result.py            # 统一成功/失败返回协议
│   │   ├── knowledge_base.py    # 文档、chunk 和检索工具
│   │   ├── context.py           # 请求级工具调用上下文
│   │   ├── contracts.py         # ToolSpec、风险和审计输出契约
│   │   ├── catalog.py           # 按配置缓存的不可变本地 ToolSpec catalog
│   │   ├── execution.py         # 只读权限、审计与输出校验；原生异步调度和重试
│   │   ├── web_search.py        # DDGS 调用、结果规范化和公开异常映射
│   │   ├── web_download.py      # HTML/PDF 共用的 hardened 有限下载适配
│   │   ├── web_fetch.py         # HTML/XHTML/纯文本的 Trafilatura 提取
│   │   ├── web_pdf_fetch.py     # PDF magic、pypdf 页范围与元数据提取
│   │   ├── mcp/                 # 入库 MCP 配置、discovery、来源调用与安全序列化
│   │   └── schemas.py           # 工具参数模型
│   ├── services/
│   │   ├── chat_models.py       # 框架无关请求、结果和内部上下文
│   │   ├── chat_stream_models.py # 框架无关流事件合同
│   │   ├── chat_service.py      # 流式 Query Rewrite、Runtime、状态策略和会话编排
│   │   └── chat_telemetry.py    # metrics、bad case、stream accounting 和调用链收尾
│   ├── llm/                     # 统一模型工厂、调用网关、Schema 和错误类型
│   ├── storage/                 # SQLite 会话历史和 ingestion 存储
│   ├── observability/
│   │   ├── __init__.py          # trace_id 生成
│   │   └── metrics.py           # 指标、bad case 和 trace 汇总
│   ├── common/logging.py        # 框架无关的公共日志构造
│   ├── prompts.py               # RAG、问题拆分和评测 Prompt
│   └── config.py                # 环境变量配置
├── data/
│   ├── kb/                      # 默认知识库文档目录
│   ├── eval/                    # v1/v2 Agent 与 RAG 评测数据、示例阈值
│   ├── chroma/                  # ChromaDB 持久化目录
│   └── manifest/                # 索引构建 manifest
├── scripts/
│   ├── run_evaluation_baseline.py # Evaluation Baseline v2 统一入口
│   ├── compare_evaluation_runs.py # 两轮 v2 结果比较
│   ├── benchmark_chat_api.py    # /chat/stream 串行与并发性能测试
│   ├── smoke_test_online_tools.py # Online Tool 真实连通性子命令
│   ├── smoke_test_agent_tool_calling.py # 模型 Tool Calling 协议
│   ├── smoke_test_chat_stream.py # /chat/stream 安全摘要冒烟客户端
│   ├── smoke_test_mcp_tools.py  # GitHub / Firecrawl 来源 smoke
│   ├── inspect_agent_runtime.py # CI 架构合同检查
│   ├── summarize_logs.py        # metrics/bad_cases 日志汇总
│   ├── cache_models.py          # 预下载 embedding/rerank 模型
│   └── evaluation/              # Runner、Agent/RAG evaluator、Judge 与报告内部包
├── frontend/                    # Vue 3 + TypeScript Demo / 管理界面
│   ├── src/                     # 页面、composable 与薄 API client
│   └── vite.config.ts           # 开发 / 本地预览代理与 Vitest 配置
├── docs/
│   ├── architecture/
│   │   ├── current-system.md    # 当前实现的主要架构入口
│   │   └── module-boundaries.md # 可执行模块依赖方向
│   ├── evaluation/
│   │   └── baseline-v2.md       # v2 数据、运行、输出与比较合同
│   ├── api/
│   │   └── chat-streaming.md     # SSE HTTP、事件、安全与断开合同
│   └── history/
│       └── agent-evolution-2026.md # PR0～PR16 历史演进归档
├── tests/                       # 单元测试和接口测试
├── logs/                        # metrics、bad_cases、tool_calls
├── outputs/                     # 评测和日志汇总输出
├── .github/workflows/tests.yml  # 静态检查、评测数据校验和 pytest CI
├── .importlinter                # 可执行模块依赖合同
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── requirements.lock            # Python 3.12 已验证的固定依赖
├── requirements.txt
├── ruff.toml
└── README.md
```

> 说明：旧的根目录转发模块已经删除，项目统一使用上面的领域包路径。

## Docker

构建镜像：

```bash
docker build -t agent-rag-demo:latest .
```

推荐运行方式：

```bash
docker compose up -d
```

如果不用 compose，至少挂载知识库、索引和模型缓存目录：

```bash
docker run --rm -p 8000:8000 --env-file .env \
  -e RAG_FILE_PATH=/app/data/kb \
  -e INDEX_STORE_DIR=/app/data/manifest \
  -e CHROMA_PERSIST_DIR=/app/data/chroma \
  -e HF_HOME=/cache/huggingface \
  -e MODEL_CACHE_DIR=/cache/huggingface/hub \
  -e SENTENCE_TRANSFORMERS_HOME=/cache/sentence-transformers \
  -v "$(pwd)/data/kb:/app/data/kb:ro" \
  -v "$(pwd)/data/chroma:/app/data/chroma" \
  -v "$(pwd)/data/manifest:/app/data/manifest" \
  -v "$(pwd)/logs:/app/logs" \
  -v agent-rag-model-cache:/cache \
  --add-host=host.docker.internal:host-gateway \
  agent-rag-demo:latest
```

## 后续优化方向

- 优化 `partial_answer` 场景的 prompt，让模型在只能回答部分问题时更清晰地说明已回答和未覆盖部分。
- 强化新旧政策冲突判断，提升 `trap` 类问题中对时间、版本和政策优先级的识别能力。
- 继续优化 Query Rewrite 对跨轮限定、复杂指代和历史冲突的处理能力。
- 在现有编译、Ruff、Import Linter、Runtime inspection、评测数据校验和 pytest
  基础上增加 Docker 构建检查；真实基线经过人工审查后再决定是否加入回退阈值门禁。

## 已知限制

- 当前支持 `.txt`、`.md`、`.pdf`、`.docx`；PDF 依赖可提取文本，暂不内置 OCR，旧版 `.doc` 和图片文件暂不支持。
- 回答/Agent 模型、追问改写模型、拆分模型和 Judge 模型都按 OpenAI-compatible Chat Completions 格式调用。
- embedding 和 rerank 模型首次加载较慢，可能需要下载权重。
- 多轮历史使用本地 SQLite 持久化，适合单实例运行；多实例部署时需要额外规划共享存储和并发策略。
- Agent 已通过唯一的 `create_agent` 路径支持 Registry-driven 多轮工具观察、同轮多个工具调用和参数修正，并具备错误次数、重复调用/错误、总耗时和同批 fail-closed 防护。
- 当前尚未实现 RAG JSON token streaming、Interrupt/Resume、SQL 写工具或多 Agent；入库 MCP 不支持 Server、stdio、OAuth、Resources 或 Prompts。
- retrieval_confidence 和 answer_confidence 是启发式估计，不等同于严格概率。

### Agentic RAG 评估合同（v3）

RAG 预测经 `ChatService.stream → stream_agent → create_agent` 执行，使用 `RAG_ONLY` 与
`knowledge_base:read`，通过 `search_knowledge_base`、`search_document`、`get_chunk_context` 获取 KB 证据。每题空 session；
Agent 自主决定每轮 query/top_k。`retrieved_chunk_ids` 是按观察顺序去重的全部 KB 证据，
`reference_chunk_ids` 是最终通过 grounding 的引用。`--retrieval-k` 仅是评分 cutoff。
正式评分只保留 v3 观察/引用 Recall；Judge 与评分权重保留。
Prediction Schema v3 is a new baseline generation contract.
Raw scores are not strictly comparable to historical v2 workflow scores.
