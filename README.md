# Mini RAG Agent

一个基于 FastAPI 的中文 RAG 知识库问答项目，并在 RAG 外层加了一层最小 Agent 路由。

系统可以读取本地 `.txt` / `.md` 文档，构建 BM25 + 向量检索索引，通过 reranker 重排候选 chunk，再调用 OpenAI-compatible 的大模型生成带引用的答案。`/chat` 接口会先判断用户问题应该走知识库问答、工具调用、直接回答，还是澄清追问。

## 当前能力

- 文档读取：支持 `.txt`、`.md`，自动识别文本编码。
- 文档切分：普通文本按段落和标点递归切分，Markdown 会先按标题层级切分并保留标题路径。
- 混合检索：使用 jieba + BM25 做关键词召回，使用 SentenceTransformer embedding 做语义召回。
- 重排：使用 CrossEncoder reranker 对候选 chunk 重新排序。
- 可选问题拆分：复杂问题可由拆分模型生成子问题，多路召回后合并证据。
- 答案生成：调用 OpenAI-compatible Chat Completions 接口，要求模型返回 JSON。
- 引用绑定：模型返回的 `Reference N` 会映射回真实文件、chunk_id、snippet 和分数。
- 最小 Agent：支持规则路由，也可以开启 LLM 路由；失败时会回退到规则路由。
- 工具调用：支持简单四则运算、列出知识库文档、查看指定 chunk 原文。
- 可观测性：`trace_id`、请求耗时、工具调用日志、失败案例日志和 metrics 汇总接口。
- 索引缓存：构建后的 chunk、文本、embedding、文档注册表和 manifest 会保存到本地。

## 项目结构

```text
mini-rag/
├── app/
│   ├── main.py                  # FastAPI 入口、接口模型、指标记录
│   ├── agent.py                 # Agent 路由、LLM router、工具/RAG 调度
│   ├── tools.py                 # simple_calculator/list_documents/get_chunk_text
│   ├── rag.py                   # RAG 总控、索引生命周期、问答流程
│   ├── rag_ingest.py            # 文档读取、清洗、切分
│   ├── rag_retrieval.py         # BM25 + 向量召回、rerank、问题拆分
│   ├── rag_generation.py        # Prompt、模型调用、JSON 解析、引用绑定
│   ├── rag_store.py             # 索引缓存、manifest、模型加载
│   ├── rag_debug.py             # 文档/chunk 预览与 chunk 原文查询
│   ├── observability.py         # trace_id 生成
│   ├── prompts.py               # RAG、问题拆分、路由 Prompt
│   ├── config.py                # 环境变量配置
│   └── untils.py                # 日志工具
├── data/
│   ├── kb/                      # 默认知识库文档目录
│   ├── eval/                    # 评测数据
│   └── manifest/                # 索引缓存和 manifest
├── scripts/
│   ├── eval.py                  # RAG 离线评测
│   ├── agent_eval.py            # Agent 路由评测
│   ├── summarize_logs.py        # metrics/bad_cases 日志汇总
│   ├── perf_test.py             # 简单串行压测
│   ├── concurrent_perf_test.py  # 简单并发压测
│   ├── decision.py              # 路由决策调试
│   └── test_local_llm.py        # 本地模型接口测试
├── tests/
│   └── test_api.py              # FastAPI 接口测试
├── logs/                        # metrics、bad_cases、tool_calls
├── outputs/                     # 评测和日志汇总输出
├── .env.example
├── Dockerfile
├── requirements.txt
└── README.md
```

> 说明：当前代码里工具文件名是 `untils.py`，README 也按当前实现写这个名字。

## 快速开始

建议使用 Python 3.10+。

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

把知识库文档放到 `data/kb/`，然后编辑 `.env` 里的模型配置。启动服务：

```bash
uvicorn app.main:app --reload
```

启动后访问：

```text
Swagger: http://127.0.0.1:8000/docs
Health:  http://127.0.0.1:8000/health
```

## 环境变量

`.env.example` 已包含一份可用模板。常用配置如下：

| 配置项 | 说明                                    |
| --- |---------------------------------------|
| `RAG_FILE_PATH` | 知识库目录，示例：`./data/kb`                  |
| `INDEX_STORE_DIR` | 索引缓存目录，示例：`./data/manifest`           |
| `CHROMA_PERSIST_DIR` | ChromaDB 本地持久化目录，示例：`./data/chroma` |
| `CHROMA_COLLECTION_NAME` | ChromaDB collection 名称，默认 `mini_rag_chunks` |
| `ENABLE_CHROMA` | 是否启用 ChromaDB 向量存储；默认 `true`，关闭后无法保存或加载向量索引 |
| `TOP_K` | `/chat` 和 `/search` 默认召回条数            |
| `DECOMPOSE_TOP_K` | 多子问题召回合并后的候选上限                        |
| `CHUNK_SIZE` | chunk 最大长度                            |
| `CHUNK_OVERLAP` | 相邻 chunk 重叠长度                         |
| `EMBEDDING_MODEL_NAME` | SentenceTransformer embedding 模型      |
| `RERANK_MODEL_NAME` | CrossEncoder rerank 模型                |
| `MAX_CONTEXT_CHARS` | 拼给回答模型的最大上下文长度                        |
| `DEEPSEEK_API_KEY` | 回答模型 API Key；本地 Ollama 可填占位值          |
| `MODEL_NAME` | 回答模型名称                                |
| `LLM_API_URL` | OpenAI-compatible Chat Completions 地址 |
| `TEMPERATURE` | 回答模型温度                                |
| `LLM_MAX_TOKENS` | 回答模型最大输出 token                        |
| `LLM_TIMEOUT` | 单次回答模型请求超时秒数                          |
| `LLM_MAX_RETRIES` | 回答模型失败重试次数                            |
| `LLM_RETRY_SLEEP` | 每次重试前等待秒数                             |
| `ENABLE_LLM_ROUTER` | 是否启用 LLM Agent 路由；默认 `true`           |
| `ROUTER_API_KEY` | 路由模型 API Key；为空时复用回答模型 Key            |
| `ROUTER_MODEL_NAME` | 路由模型名称                                |
| `ROUTER_API_URL` | 路由模型接口地址                              |
| `ROUTER_TIMEOUT` | 路由模型请求超时秒数                            |
| `ROUTER_MAX_RETRIES` | 路由模型失败重试次数                            |
| `ROUTER_MAX_TOKENS` | 路由模型最大输出 token                        |
| `ROUTER_TEMPERATURE` | 路由模型温度                                |
| `ENABLE_QUERY_DECOMPOSE` | 是否启用问题拆分；默认 `true`                    |
| `DECOMPOSE_API_KEY` | 问题拆分模型 API Key                        |
| `DECOMPOSE_MODEL_NAME` | 问题拆分模型名称                              |
| `DECOMPOSE_API_URL` | 问题拆分模型接口地址                            |
| `DECOMPOSE_TIMEOUT` | 问题拆分模型请求超时秒数                          |
| `VECTOR_THRESHOLD` | 召回证据不足时拒答的向量分数阈值                      |
| `MARGIN_THRESHOLD` | 预留的分数差距阈值配置                           |
| `AUTO_REBUILD_ON_STARTUP` | 是否启动时强制重建索引                           |

### DeepSeek 示例

```env
RAG_FILE_PATH=./data/kb
INDEX_STORE_DIR=./data/manifest
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION_NAME=mini_rag_chunks
ENABLE_CHROMA=true

TOP_K=5
DECOMPOSE_TOP_K=10
CHUNK_SIZE=600
CHUNK_OVERLAP=120

EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding-0.6B
RERANK_MODEL_NAME=BAAI/bge-reranker-v2-m3
MAX_CONTEXT_CHARS=4000

DEEPSEEK_API_KEY=your-api-key
MODEL_NAME=deepseek-chat
LLM_API_URL=https://api.deepseek.com/v1/chat/completions
TEMPERATURE=0.1
LLM_MAX_TOKENS=1024

ENABLE_LLM_ROUTER=false
ENABLE_QUERY_DECOMPOSE=true
DECOMPOSE_API_KEY=your-api-key
DECOMPOSE_MODEL_NAME=qwen-plus
DECOMPOSE_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions

VECTOR_THRESHOLD=0.1
```

### Ollama 示例

```env
DEEPSEEK_API_KEY=ollama
MODEL_NAME=qwen2.5:7b
LLM_API_URL=http://127.0.0.1:11434/v1/chat/completions

DECOMPOSE_API_KEY=ollama
DECOMPOSE_MODEL_NAME=qwen2.5:7b
DECOMPOSE_API_URL=http://127.0.0.1:11434/v1/chat/completions
```

## 索引构建与缓存

服务启动时会检查当前知识库目录和构建参数：

```text
启动服务
  -> Settings.validate()
  -> 检查 manifest 和缓存
  -> 文档或配置变化：重建索引并保存缓存
  -> 无变化：从本地缓存加载索引
```

缓存文件分为两部分：ChromaDB 向量库默认在 `data/chroma/`，`data/manifest/` 只保留索引构建账本和文档注册表：

- `index_manifest.json`
- `docs_registry.json`

chunk 文本、chunk metadata 和 embedding 向量都存入 ChromaDB，不再写入 `chunks.json`、`texts.json` 和 `doc_embeddings.npy`。

以下变化会触发重建：新增/删除/修改知识库文件、修改 `CHUNK_SIZE`、修改 `CHUNK_OVERLAP`、修改 embedding 模型、修改 rerank 模型、pipeline 版本变化。修改 `TOP_K`、Prompt、回答模型名称、温度等通常不需要重建索引。

`/kb_summary` 会返回 `index_source` 字段：`chroma` 表示当前从 ChromaDB collection 加载，`memory` 表示刚重建完成但尚未重新加载持久化缓存，`none` 表示没有可用索引缓存。

## API

| 接口 | 方法 | 作用 |
| --- | --- | --- |
| `/health` | GET | 健康检查 |
| `/docs-preview` | GET | 查看已加载文档列表 |
| `/chunks-preview` | GET | 查看 chunk 预览，支持 `max_len`、`limit`、`file_name` |
| `/search` | POST | 只执行检索和 rerank，不调用回答模型 |
| `/chat` | POST | 完整 Agent + RAG 问答入口 |
| `/kb_summary` | POST | 返回知识库 ready 状态、文档数和 chunk 数 |
| `/decide` | POST | 只返回 Agent 路由决策，不执行工具或 RAG |
| `/metrics-summary` | GET | 汇总最近 N 条请求指标 |

### `/chat`

请求：

```json
{
  "question": "每个 Vue 应用都是通过什么函数创建应用实例的？",
  "top_k": 5,
  "decompose_top_k": 10
}
```

核心响应字段：

| 字段 | 说明 |
| --- | --- |
| `trace_id` | 本次请求的追踪 ID，也会写入响应头 `X-Trace-Id` |
| `data.answer` | 最终回答 |
| `data.references` | 引用来源列表 |
| `data.confidence` | 0 到 1 的置信度估计 |
| `data.status` | `ok`、`no_evidence`、`empty_index`、`error`、`clarify`、`out_of_scope` |
| `data.need_clarification` | 是否需要用户补充问题 |
| `data.clarification_question` | 澄清追问 |
| `data.reason` | 当前状态原因 |
| `data.action` | Agent 选择的动作 |
| `data.tool_result_summary` | 工具或 RAG 路径摘要 |
| `data.tool_trace` | 路由、工具调用、检索命中、最终状态等过程信息 |

### `/search`

`/search` 适合调试召回质量。它只检索，不调用回答模型。

```json
{
  "question": "每个 Vue 应用都是通过什么函数创建应用实例的？",
  "top_k": 5
}
```

返回结果会包含：

- `reference_id`
- `file_name`
- `source_path`
- `chunk_id`
- `position`
- `snippet`
- `score`
- `bm25_score`
- `vector_score`
- `rerank_score`

### `/decide`

`/decide` 用来观察 Agent 路由结果。

```json
{
  "question": "1 + 2 * 3 等于多少？",
  "top_k": 5
}
```

可能返回：

```json
{
  "decision": "use_tool",
  "action": "simple_calculator",
  "tool_name": "simple_calculator",
  "tool_args": {
    "expression": "1 + 2 * 3"
  },
  "router_source": "rule_based"
}
```

## Agent 路由

当前支持的 action：

| action | 路径 |
| --- | --- |
| `search_knowledge_base` | 走 RAG 检索、重排、生成、引用绑定 |
| `simple_calculator` | 调用安全四则运算工具 |
| `list_documents` | 返回当前知识库文档数量和列表 |
| `get_chunk_text` | 根据 chunk_id 返回完整 chunk 文本 |
| `direct_answer` | 直接回答系统能力类问题 |
| `clarify` | 问题过短、为空或指代不明时追问 |

默认使用规则路由。设置 `ENABLE_LLM_ROUTER=true` 后，系统会先调用路由模型生成 JSON 决策；如果模型调用失败、输出不是 JSON、action 非法或工具参数不合法，会自动回退到规则路由。

## RAG 流程

```text
用户问题
  -> Agent 路由
  -> 知识库路径 search_knowledge_base
  -> BM25 + embedding 混合召回
  -> 可选问题拆分
  -> CrossEncoder rerank
  -> 低相关度拒答判断
  -> 构造 evidence context
  -> 调用回答模型
  -> 解析模型 JSON
  -> 绑定 references
  -> 估算 confidence
  -> 返回统一 ChatResponse
```

回答模型应返回 JSON，例如：

```json
{
  "answer": "每个 Vue 应用都是通过 createApp 函数创建应用实例的。",
  "references": ["Reference 1"]
}
```

如果模型没有按 JSON 返回，接口会返回 `MODEL_BAD_OUTPUT`，并把失败案例写入日志。

## 日志与观测

`/chat` 会为每次请求生成或继承 `trace_id`：

- 请求头可传入 `X-Trace-Id`。
- 响应头会返回 `X-Trace-Id` 和 `X-Total-Cost-Ms`。
- 响应体也包含 `trace_id`。

日志文件：

| 文件 | 说明 |
| --- | --- |
| `logs/metrics.jsonl` | 每次 `/chat` 的耗时、状态、action、引用数、工具信息 |
| `logs/tool_calls.jsonl` | 每次工具调用的输入、输出、耗时和是否成功 |
| `logs/bad_cases.jsonl` | 真正值得分析的失败案例，如模型 JSON 解析失败、引用异常、工具异常 |

`/metrics-summary?limit=20` 可以查看最近请求的状态分布和平均耗时。

日志汇总脚本：

```bash
python scripts/summarize_logs.py
```

默认会读取 `logs/metrics.jsonl` 和 `logs/bad_cases.jsonl`，并把汇总结果写到 `outputs/week9_summary.md`。

## 测试和评测

运行接口测试：

```bash
pytest
```

运行 RAG 或 Agent 评测脚本：

```bash
python scripts/eval.py
python scripts/agent_eval.py
```

压测和调试脚本：

```bash
python scripts/perf_test.py
python scripts/concurrent_perf_test.py
python scripts/decision.py
python scripts/test_local_llm.py
```

## Docker

构建镜像：

```bash
docker build -t mini-rag .
```

运行：

```bash
docker run --env-file .env -p 8000:8000 mini-rag
```

注意：首次启动可能会下载 embedding / rerank 模型权重，耗时取决于网络和模型大小。生产环境建议把模型缓存和知识库目录挂载为 volume。

## 常见问题

### 1. `/chat` 返回 `EMPTY_INDEX`

通常是知识库没有加载成功。检查：

- `RAG_FILE_PATH` 是否指向存在的目录。
- 目录下是否有 `.txt` 或 `.md` 文件。
- 首次启动是否完成索引构建。

### 2. 模型调用失败

检查：

- `DEEPSEEK_API_KEY` 是否正确。
- `MODEL_NAME` 是否存在且账号有权限。
- `LLM_API_URL` 是否是 OpenAI-compatible Chat Completions 地址。
- 本地 Ollama 是否已经启动并拉取对应模型。

### 3. 检索到了内容但回答拒答

可能原因：

- `VECTOR_THRESHOLD` 过高。
- chunk 切分过碎或过长。
- 问题和文档表述差异太大。
- 模型返回的 references 没有正确引用检索结果。

可以先用 `/search` 查看召回和分数，再调整文档、阈值或 chunk 参数。

### 4. LLM router 输出不稳定

保持 `ENABLE_LLM_ROUTER=false` 会使用规则路由，最稳定。开启 LLM router 后，系统已经做了校验和回退，但路由质量仍取决于路由模型输出。

## 已知限制

- 当前只支持 `.txt` 和 `.md`，暂不支持 PDF、DOCX、图片等复杂格式。
- 回答模型、路由模型和拆分模型都按 OpenAI-compatible Chat Completions 格式调用。
- embedding 和 rerank 模型首次加载较慢，可能需要下载权重。
- Agent 仍是单轮最小实现，没有长期记忆和复杂任务规划。
- confidence 是基于召回分数、引用数量等启发式估计，不等同于严格概率。
