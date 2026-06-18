# Mini RAG Agent：中文知识库问答与最小 Agent 系统

本项目是一个基于 **FastAPI** 的中文 RAG 知识库问答系统，并在此基础上升级为一个**最小 Agent**。

系统不仅可以根据本地知识库回答问题，还可以先判断用户问题类型，再选择不同执行路径：

- 知识库问题：走 RAG 检索问答；
- 简单计算问题：调用 `simple_calculator`；
- 文档列表问题：调用 `list_documents`；
- chunk 查看问题：调用 `get_chunk_text`；
- 指代不清的问题：返回澄清提示；
- 明显超出知识库范围的问题：返回拒答结果。

项目目标是实现一个**可运行、可调试、可演示、可评测**的 RAG + Agent 入门系统。

---

## 1. 项目背景

普通大模型无法直接访问本地知识库内容。  
如果直接提问，大模型容易出现以下问题：

- 不知道本地文档中的具体内容；
- 回答可能脱离文档，产生幻觉；
- 答案缺少引用来源，难以验证；
- 所有问题都交给大模型处理，缺少任务分流能力。

本项目使用 RAG 解决“根据本地文档回答问题”的问题，并进一步加入最小 Agent，使系统具备基础的任务判断和工具调用能力。

---

## 2. 项目核心能力

### 2.1 RAG 问答能力

系统支持：

- 读取本地 `.txt` 和 `.md` 文档；
- 对文档进行清洗和 chunk 切分；
- 使用 BM25 做关键词检索；
- 使用 Embedding 做语义检索；
- 融合 BM25 分数和向量相似度进行混合排序；
- 判断问题是否为多子问题，然后觉得是否将问题拆分；
- 若非多子问题直接对原问题进行rerank重排序，若为多子问题则采用：子问题独立召回—证据合并；
- 将召回片段拼接为上下文；
- 调用大模型生成答案；
- 返回答案、引用来源、置信度和状态信息。

### 2.2 最小 Agent 能力

系统在 `/chat` 接口中加入 Agent 决策流程。

用户提问后，系统会先调用 `route_agent_action()` 判断问题类型，再由 `run_agent_once()` 执行对应路径。

当前支持的 action 包括：

| action | 说明 |
|---|---|
| `search_knowledge_base` | 默认知识库问答路径，走 RAG 检索与生成 |
| `simple_calculator` | 处理简单四则运算 |
| `list_documents` | 查看当前知识库文档数量和列表 |
| `get_chunk_text` | 根据 chunk_id 查看指定 chunk 内容 |
| `direct_answer` | 回答系统能力类问题 |
| `clarify` | 问题不完整或指代不明时要求用户补充 |
| `refuse` | 明显超出知识库范围时拒答 |

---

## 3. 普通 RAG 与最小 Agent 的区别

普通 RAG 的流程比较固定：

```text
用户提问
  ↓
检索知识库
  ↓
拼接上下文
  ↓
调用大模型
  ↓
返回答案和引用
```

本项目升级后的最小 Agent 流程是：

```text
用户提问
  ↓
Agent 决策
  ↓
判断问题类型
  ↓
选择执行路径
  ↓
RAG 问答 / 工具调用 / 澄清 / 拒答
  ↓
返回统一结构结果
```

也就是说，本项目第 8 周的重点不是“多写几个工具”，而是让系统具备初步的**任务分流能力**。

---

## 4. 系统架构

### 4.1 整体架构

```text
                    用户问题
                       ↓
                 FastAPI /chat
                       ↓
              route_agent_action()
                       ↓
              判断应该执行的 action
                       ↓
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
   RAG 问答路径      工具调用路径      澄清
        ↓              ↓              ↓
 search_knowledge   calculator     clarify
        ↓          list_documents
  检索相关 chunk    get_chunk_text
        ↓              ↓
  拼接上下文          格式化工具结果
        ↓              ↓
  调用大模型          直接返回结果
        ↓              ↓
        └────── 统一 ChatResponse ──────┘
```

### 4.2 RAG 问答路径

```text
用户问题
  ↓
BM25 + 向量混合检索
  ↓
llm判断问题是否为多子问题
  ↓                     ↓
非多子问题rerank       多子问题：子问题独立召回—证据合并
  ↓                     ↓
相关性阈值判断          同左
  ↓
构建上下文 context
  ↓
调用大模型
  ↓
解析模型 JSON 输出
  ↓
绑定引用 references
  ↓
估算 confidence
  ↓
返回答案
```

### 4.3 工具调用路径

```text
用户问题
  ↓
Agent 判断为工具类问题
  ↓
提取工具参数
  ↓
dispatch_tool()
  ↓
工具执行
  ↓
format_tool_answer()
  ↓
返回工具结果
```

---

## 5. 项目目录结构

```text
mini-rag/
├── app/
│   ├── main.py                    # FastAPI 入口，定义接口和响应结构
│   ├── agent.py                   # Agent 决策与执行逻辑
│   ├── rag.py                     # RAG 总控逻辑
│   ├── rag_ingest.py              # 文档读取、清洗和切分
│   ├── rag_retrieval.py           # BM25 + 向量检索与拒答判断
│   ├── rag_generation.py          # Prompt 构建、模型调用、结果解析
│   ├── rag_store.py               # 索引构建、缓存保存、manifest 判断
│   ├── rag_debug.py               # 文档和 chunk 预览工具
│   ├── extract_refund_policy.py   # 退款类信息抽取接口逻辑
│   ├── tools.py                   # 工具函数注册与分发
│   ├── config.py                  # 环境变量和配置项
│   ├── prompts.py                 # Prompt 模板
│   └── untils.py                  # 日志等通用工具
├── data/
│   ├── kb/                        # 知识库文档
│   └── eval/                      # 评测数据
├── scripts/
│   ├── eval.py                    # RAG 离线评测脚本
│   └── agent_eval.py              # Agent 路由评测脚本
│   └── concurrent_perf_test.py    # 简单并发压测
│   └── decision.py                # 工具类路由决策测试
│   └── perf_test.py               # 简单串发压测
│   └── summarize_logs.py          # 日志汇总脚本
│   └── test_local_llm.py          # 本地大模型测试
├── logs/                          # 各类日志存放点
├── outputs/                       # 评测输出结果
├── tests/                         # 接口测试
├── .env.example                   # 环境变量示例
├── requirements.txt
└── README.md
```

> 说明：当前代码中实际导入的是 `app.untils`，所以 README 里应写 `untils.py`，不要写成 `utils.py`，除非后续把文件名统一改掉。

---

## 6. 环境准备

建议使用：

```text
Python 3.10+
```

安装依赖：

```bash
pip install -r requirements.txt
```

创建 `.env` 文件：

Windows：

```bash
copy .env.example .env
```

macOS / Linux：

```bash
cp .env.example .env
```

---

## 7. 环境变量配置

`.env` 中需要配置以下内容：

| 配置项                                | 是否必填 | 说明                       |
| ---------------------------------- | ---- | ------------------------ |
| `RAG_FILE_PATH`                    | 是    | 知识库目录，例如 `./data/kb`     |
| `TOP_K`                            | 否    | 默认召回 chunk 数量            |
| `CHUNK_SIZE`                       | 否    | chunk 最大长度               |
| `CHUNK_OVERLAP`                    | 否    | 相邻 chunk 重叠长度            |
| `EMBEDDING_MODEL_NAME`             | 是    | Embedding 模型名称           |
| `LLM_API_KEY` 或 `DEEPSEEK_API_KEY` | 是    | 大模型 API Key，按你的代码配置保持一致  |
| `MODEL_NAME`                       | 否    | 大模型名称，例如 `deepseek-chat` |
| `LLM_API_URL`                      | 否    | 大模型接口地址                  |
| `VECTOR_THRESHOLD`                 | 否    | 向量分数过低时拒答                |
| `MARGIN_THRESHOLD`                 | 否    | Top1 与 Top2 分数差距阈值       |
| `MAX_CONTEXT_CHARS`                | 否    | 拼接给大模型的最大上下文长度           |
| `AUTO_REBUILD_ON_STARTUP`          | 否    | 是否启动时强制重建索引              |
| `DECOMPOSE_API_KEY`                | 是    | 问题拆分大模型 API Key          |
| `DECOMPOSE_MODEL_NAME`             | 否    | 问题拆分大模型名称                |
| `DECOMPOSE_API_URL`                | 否    | 问题拆分大模型接口地址              |

---

## 8. 启动项目

在项目根目录运行：

```bash
uvicorn app.main:app --reload
```

启动后访问 Swagger：

```text
http://127.0.0.1:8000/docs
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

---

## 9. 索引构建与缓存机制

项目启动时会检查知识库是否需要重建索引：

```text
启动服务
  ↓
Settings.validate()
  ↓
needs_rebuild()
  ↓
判断是否需要重建
  ↓
需要重建：rebuild_and_persist_index()
不需要重建：load_index_cache()
```

需要重建索引的情况：

- 新增知识库文档；
- 删除知识库文档；
- 修改知识库文档内容；
- 修改 `CHUNK_SIZE`；
- 修改 `CHUNK_OVERLAP`；
- 修改 Embedding 模型。
- 修改`CrossEncoder`模型

通常不需要重建索引的情况：

- 修改 `TOP_K`；
- 修改 Prompt；
- 修改 `MODEL_NAME`；
- 修改 `TEMPERATURE`；
- 修改拒答提示文案。

---

## 10. API 接口说明

| 接口 | 方法 | 作用 |
|---|---|---|
| `/health` | GET | 健康检查 |
| `/docs-preview` | GET | 查看已加载文档 |
| `/chunks-preview` | GET | 查看切分后的 chunk |
| `/search` | POST | 只检索，不调用大模型 |
| `/chat` | POST | 完整 Agent + RAG 问答入口 |
| `/extract_refund_policy` | POST | 退款政策结构化抽取示例 |
| `/kb_summary` | POST | 查看知识库统计信息 |
| `/decide` | POST | 只查看 Agent 决策结果，不执行最终问答 |

---

## 11. `/chat` 接口说明

`/chat` 是本项目最重要的接口。

请求示例：

```json
{
  "question": "每个 Vue 应用都是通过什么函数创建应用实例的？",
  "top_k": 5
}
```

返回结构核心字段：

| 字段 | 说明 |
|---|---|
| `answer` | 最终回答 |
| `references` | 引用来源 |
| `confidence` | 置信度 |
| `status` | 本次回答状态 |
| `reason` | 状态原因 |
| `action` | Agent 本次选择的动作 |
| `tool_result_summary` | 工具或 RAG 路径结果摘要 |
| `tool_trace` | Agent 决策和执行过程追踪 |


---

## 12. Agent 决策规则

当前 Agent 使用规则路由，不是完整的模型自主 function calling。

决策逻辑如下：

```text
1. 空问题
   → clarify

2. 用户询问系统能力
   → direct_answer

3. 问题中包含 chunk_id 或“查看第几个 chunk”
   → get_chunk_text

4. 问题询问文档数量、文档列表、有哪些文档
   → list_documents

5. 问题是简单四则运算表达式
   → simple_calculator

6. 问题指代不明
   → clarify

7. 其他情况
   → search_knowledge_base
```

---

## 13. 演示样例

### 13.1 知识库问答路径

请求：

```json
{
  "question": "每个 Vue 应用都是通过什么函数创建应用实例的？",
  "top_k": 5
}
```

预期路径：

```text
Agent → search_knowledge_base → 检索 chunk → 调用大模型 → 返回答案和引用
```

重点观察：

```json
{
  "action": "search_knowledge_base",
  "tool_result_summary": "知识库检索完成，召回 5 个 chunk"
}
```

---

### 13.2 简单计算工具路径

请求：

```json
{
  "question": "1 + 2 * 3 等于多少？",
  "top_k": 5
}
```

预期路径：

```text
Agent → simple_calculator → 返回计算结果
```

预期回答：

```text
计算结果是：7
```

重点观察：

```json
{
  "action": "simple_calculator",
  "tool_result_summary": "simple_calculator 执行成功，结果为 7"
}
```

---

### 13.3 文档列表工具路径

请求：

```json
{
  "question": "当前知识库有哪些文档？",
  "top_k": 5
}
```

预期路径：

```text
Agent → list_documents → 返回文档数量和文档名
```

重点观察：

```json
{
  "action": "list_documents"
}
```

---

### 13.4 chunk 查看工具路径

请求：

```json
{
  "question": "查看 chunk 1 的内容",
  "top_k": 5
}
```

预期路径：

```text
Agent → get_chunk_text → 返回指定 chunk 原文
```

重点观察：

```json
{
  "action": "get_chunk_text"
}
```

---

### 13.5 指代不明问题

请求：

```json
{
  "question": "这个怎么弄？",
  "top_k": 5
}
```

预期路径：

```text
Agent → clarify → 要求用户补充说明
```

预期状态：

```json
{
  "status": "clarify",
  "need_clarification": true
}
```


---

## 14. `/decide` 接口演示

`/decide` 只展示 Agent 的决策结果，不执行工具或 RAG。

请求：

```json
{
  "question": "1 + 2 * 3 等于多少？",
  "top_k": 5
}
```

预期返回：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "question": "1 + 2 * 3 等于多少？",
    "decision": {
      "decision": "use_tool",
      "action": "simple_calculator",
      "tool_name": "simple_calculator",
      "tool_args": {
        "expression": "1 + 2 * 3"
      },
      "reason": "用户问题是简单四则运算"
    }
  }
}
```

这个接口适合用来演示 Agent 的 decision 流程。

---

## 15. `/search` 接口演示

`/search` 只执行检索，不调用大模型。

适合用来观察：

- 哪些 chunk 被召回；
- 每个 chunk 的 BM25 分数；
- 每个 chunk 的 vector 分数；
- 每个 chunk 的 rerank 分数；
- hybrid score 排序是否合理。

请求：

```json
{
  "question": "每个 Vue 应用都是通过什么函数创建应用实例的？",
  "top_k": 5
}
```

返回中重点看：

```json
{
  "reference_id": "Reference 1",
  "file_name": "application.md",
  "chunk_id": 1,
  "snippet": "...",
  "score": 1,
  "bm25_score": 34.05,
  "vector_score": 0.74
}
```

---

## 16. Agent 基础评测

评测重点是验证 Agent 是否选对路径。

评测样例可以包含：

```json
[
  {
    "id": "case_001",
    "question": "1 + 2 * 3 等于多少？",
    "expected_action": "simple_calculator"
  },
  {
    "id": "case_002",
    "question": "当前知识库有哪些文档？",
    "expected_action": "list_documents"
  },
  {
    "id": "case_003",
    "question": "查看 chunk 1 的内容",
    "expected_action": "get_chunk_text"
  },
  {
    "id": "case_004",
    "question": "每个 Vue 应用都是通过什么函数创建应用实例的？",
    "expected_action": "search_knowledge_base"
  },
  {
    "id": "case_005",
    "question": "这个怎么弄？",
    "expected_action": "clarify"
  }
]
```

评测指标：

```text
Action Accuracy = 预测正确的样例数 / 总样例数
```

示例输出：

```text
总样例数：20
通过数：18
失败数：2
Action Accuracy：90.00%
```

---

## 17. 手工测试清单

### 17.1 RAG 问答是否正常

测试问题：

```json
{
  "question": "每个 Vue 应用都是通过什么函数创建应用实例的？",
  "top_k": 5
}
```

检查点：

- `action` 是否为 `search_knowledge_base`；
- 是否返回 `references`；
- `snippet` 是否能支撑答案；
- `confidence` 是否在 0 到 1 之间。

---

### 17.2 工具调用是否正常

测试问题：

```json
{
  "question": "1 + 2 * 3 等于多少？",
  "top_k": 5
}
```

检查点：

- `action` 是否为 `simple_calculator`；
- 是否没有走 RAG；
- 结果是否为 7；
- `tool_trace` 中是否出现 `call_tool`。

---

### 17.3 澄清是否正常

测试问题：

```json
{
  "question": "这个怎么弄？",
  "top_k": 5
}
```

检查点：

- `action` 是否为 `clarify`；
- `need_clarification` 是否为 `true`；
- 是否返回 `clarification_question`。

---

## 18. 当前项目特点

本项目的重点不是追求复杂 Agent 框架，而是完成一个清晰、可解释的最小闭环：

```text
问题输入
  ↓
Agent 决策
  ↓
选择路径
  ↓
执行工具或 RAG
  ↓
返回统一结果
  ↓
通过样例评测验证
```

当前实现的优点：

- 结构清晰，适合理解 RAG 与 Agent；
- 接口返回包含 action 和 trace，便于展示路径差异；
- RAG 与工具调用共用 `/chat` 入口；
- 提供 `/decide` 接口，可以单独观察 Agent 决策；
- 支持索引缓存，避免每次启动都完整重建；
- 返回引用来源，方便验证答案依据。

---

## 19. 已知限制

当前版本仍然是最小实现，存在以下限制：

- Agent 决策主要依赖规则判断，不是大模型自主规划；
- 工具数量较少，只覆盖计算、文档列表和 chunk 查看；
- `simple_calculator` 只适合简单四则运算；
- RAG 的回答质量依赖检索结果和大模型输出稳定性；
- 暂不支持 PDF、DOCX、图片等复杂文档格式。

---

## 20. 后续优化方向

后续可以继续优化：

- 增加更多工具，例如知识库统计、关键词搜索、文档摘要；
- 改进 Agent 路由策略，减少硬编码关键词；
- 将 Agent 决策结果与评测脚本更好地结合；
- 增加多轮对话能力；
- 支持更多文档格式；
- 优化 confidence 计算方式；
- 增加 pytest 自动化测试；
- 完善 Docker Compose 部署方式。

---

## 21.可观测性与监控

本小节将对`trace_id、日志字段、失败案例集、日志汇总脚本`进行说明

### 21.1 trace_id
`trace_id`置于`\chat`接口中，用于追踪一次请求流程，每次请求都会生成不同的`trace_id`以区分

一次请求的主要流程包括
1. 接收用户问题
2. Agent根据问题判断路径
3. 工具调用或RAG检索
4. 构造上下文
5. 调用模型生成回答
6. 解析模型输出
7. 校验回答可靠性
8. 返回最终结果

### 21.2 日志字段

| 文件                    | 作用              |     |
| --------------------- | --------------- | --- |
| logs/metrics.jsonl    | 记录系统运行日志        |     |
| logs/tool_calls.jsonl | 记录 Agent 工具调用情况 |     |
| logs/bad_cases.jsonl  | 记录值得分析的失败案例     |     |
  
bad_cases.jsonl 不记录正常拒答、空问题、模糊问题等预期行为。  
它主要记录系统“本应答好但没有答好”的情况，例如检索失败、引用不可靠、模型输出格式异常等。

### 21.3 失败案例集

会被记录为失败案例的情况

1. 模型输出格式异常
2. 工具调用异常
3. 模型输出正常回答，但引用为空
4. 模型输出正常回答，但引用不支撑
5. 检索结果不够，导致本能回答的问题回答不出来

不会记录为失败案例的情况

1. 空问题
2. 用户问题模糊，指代不清
3. 检索结果不足以回答问题，系统正常拒答
4. 用户问题超出知识库范围

### 21.4 日志汇总脚本

日志汇总脚本位置为`mini-rag\scripts\summarize_logs.py`，其用于接收位于``mini-rag\logs`中的系统运行日志`metrics.jsonl`和失败案例日志`bad_cases.jsonl`，输出日志汇总记录`mini-rag\outputs/week9_summary.md`。

其主要功能可归为以下几点
1. 总览：针对请求次数，成功次数，失败次数等等这些数据进行统计
2. status分布：针对最终状态进行统计
3. action分布：针对Agent所选择的路由进行统计
4. 工具调用分布：针对所调用的工具进行统计
5. error_type 失败类型排行：针对失败类型进行统计并按次数排序
6. 最慢请求 Top 5：针对请求耗时最长的五次进行排序，并展示相关信息
7. 失败案例：挑选部分失败案例展示其相关信息
8. 日志格式问题: 统计有格式问题的日志
9. 统计口径说明：对脚本的运行流程进行说明

## 22. 模型来源配置说明

### 22.1 使用 DeepSeek 云端 API
`.env`示例
```
TOP_K=5  
CHUNK_OVERLAP=120  
CHUNK_SIZE=600  
  
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  
  
MAX_CONTEXT_CHARS=3000  
  
# =========================  
# 请求控制  
# =========================  
  
# 单次模型请求最多等 60 秒  
LLM_TIMEOUT=60  
# 失败后最多重试两次  
LLM_MAX_RETRIES=2  
# 每次重试前等待1秒  
LLM_RETRY_SLEEP=1  
  
# =========================  
# 模型相关参数  
# =========================  
DEEPSEEK_API_KEY=your_api_key
MODEL_NAME=deepseek-chat  
LLM_API_URL=https://api.deepseek.com/v1/chat/completions  
  
TEMPERATURE=0.1  
LLM_MAX_TOKENS=1024  
  
# =========================  
# 召回chunk的拒绝阈值  
# =========================  
VECTOR_THRESHOLD=0.35
```
### 22.2 使用本地 Ollama 模型
`.env`示例
```
TOP_K=5  
CHUNK_OVERLAP=120  
CHUNK_SIZE=600  
  
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  
  
MAX_CONTEXT_CHARS=3000  
  
# =========================  
# 请求控制  
# =========================  
  
# 单次模型请求最多等 60 秒  
LLM_TIMEOUT=60  
# 失败后最多重试两次  
LLM_MAX_RETRIES=2  
# 每次重试前等待1秒  
LLM_RETRY_SLEEP=1  
  
# =========================  
# 模型相关参数  
# =========================  
DEEPSEEK_API_KEY=ollama
MODEL_NAME=qwen2:7b  
LLM_API_URL=http://127.0.0.1:11434/v1/chat/completions  
  
TEMPERATURE=0.1  
LLM_MAX_TOKENS=1024  
  
# =========================  
# 召回chunk的拒绝阈值  
# =========================  
VECTOR_THRESHOLD=0.35
```

## 23. 部署到 Kubernetes，组件如何拆分

当前项目可以按照 Kubernetes 的对象职责拆成以下几类：  
  
### 23.1. FastAPI 后端服务  
  
FastAPI 是主要的业务入口，负责接收用户问题、调用检索逻辑、访问向量数据库、调用模型服务并返回结果。  
  
在 Kubernetes 中，FastAPI 适合使用 Deployment + Service 部署：  
  
- Deployment：负责运行 FastAPI Pod，维护副本数量，并支持滚动更新。  
- Service：为 FastAPI Pod 提供稳定访问入口，避免直接依赖 Pod IP。  
- ConfigMap：保存普通配置，例如模型名称、向量库地址、top_k、chunk_size 等。  
- Secret：保存敏感信息，例如 LLM API Key、数据库密码、JWT Secret 等。  
  
### 23.2. 模型服务  
  
如果项目调用的是外部大模型 API，例如 OpenAI、Claude、DeepSeek 或通义千问，则模型服务本身不需要部署在 Kubernetes 内部。项目只需要通过 ConfigMap 配置模型名称、API 地址，通过 Secret 保存 API Key。  
  
如果后续改为自托管模型服务，例如 embedding 服务、reranker 服务或本地 LLM 服务，则可以使用 Deployment + Service 部署。但这类服务需要额外考虑 GPU、模型权重、显存、冷启动时间和资源限制。  
  
### 23.3. 数据库  
  
数据库属于有状态组件，不适合像普通无状态 API 一样简单用 Deployment 管理。  
  
生产环境中，PostgreSQL、Redis、向量数据库等更适合使用外部托管服务。FastAPI 通过 ConfigMap 和 Secret 获取数据库连接信息。  
  
如果必须部署到 Kubernetes 内部，则应考虑 StatefulSet、PersistentVolume、PersistentVolumeClaim、备份恢复和数据一致性。  
  
### 23.4. 向量数据库  
  
向量数据库保存文档 chunk、embedding 向量、metadata 和索引，属于有状态组件。  
  
如果使用外部向量数据库，FastAPI 只需要通过配置连接它。如果在 Kubernetes 内部自建向量数据库，则需要考虑持久化存储、备份、恢复和服务稳定性，不建议简单地按普通 Deployment 处理。  
  
### 23.5. 配置与密钥  
  
普通配置放入 ConfigMap，例如：  
  
- MODEL_NAME  
- EMBEDDING_MODEL  
- VECTOR_DB_HOST  
- VECTOR_DB_PORT  
- TOP_K  
- CHUNK_SIZE  
- LOG_LEVEL  
  
敏感配置放入 Secret，例如：  
  
- LLM_API_KEY  
- DATABASE_PASSWORD  
- JWT_SECRET  
- S3_ACCESS_KEY  
- S3_SECRET_KEY  
  
总体上，项目的 Kubernetes 拆分思路是：  
  
用户请求先进入 FastAPI Service，再转发到 FastAPI Deployment 管理的多个 Pod。FastAPI Pod 从 ConfigMap 读取普通配置，从 Secret 读取敏感配置，并访问外部数据库、向量数据库和模型服务。
