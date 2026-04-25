# Mini RAG：中文知识库问答系统

一个基于 **FastAPI** 的最小可运行中文 RAG 问答系统。  
项目支持读取本地 `.txt` 和 `.md` 文档，完成文档加载、文本切块、向量化、BM25 + 向量混合检索、答案生成、拒答判断和引用返回，适合用于学习、演示和简历项目展示。

---

## 1. 项目简介

传统大模型无法直接理解用户本地或私有文档内容，直接提问时容易出现回答不准确、无法追溯来源、甚至编造答案的问题。

本项目通过 RAG（Retrieval-Augmented Generation，检索增强生成）流程解决这一问题：

1. 先读取本地知识库文档；
2. 将文档切分成可检索的 chunk；
3. 使用 Embedding 模型生成文本向量；
4. 同时构建 BM25 关键词检索与向量语义检索；
5. 用户提问；
6. 根据用户问题类型选择走工具路径还是问答路径；
7. 再将问题与上下文一起交给大模型生成答案；
8. 最终返回答案和引用来源。

项目目标不是做一个复杂系统，而是实现一个**可运行、可调试、可演示、可说明**的最小 RAG 系统。

---

## 2. 项目目标

本项目主要用于解决以下问题：

- 用户面对大量文档时，难以快速定位答案；
- 普通关键词搜索无法理解自然语言问题；
- 直接问大模型时，模型无法知道本地知识库内容；
- 大模型可能脱离文档编造答案；
- 问答结果缺少引用来源，难以验证可信度。

本项目希望实现：

- 支持用户使用自然语言提问；
- 支持从本地知识库中召回相关片段；
- 支持基于检索结果生成回答；
- 支持返回答案引用来源；
- 支持低相关问题拒答，减少幻觉；
- 支持通过接口演示完整 RAG 链路。

---

## 3. 当前已实现功能

### 3.1 文档加载

- 自动读取本地知识库目录中的 `.txt` 和 `.md` 文件；
- 暂不支持 PDF、DOCX、图片等文件类型；
- 当知识库为空时，系统会返回 `NO_DOCUMENTS` 状态，避免后续初始化异常。

### 3.2 文档切块

- `.txt` 文档使用递归切分策略，按照段落、换行、标点等规则切块；
- `.md` 文档先按照标题层级切分，再进行二次递归切块；
- 每个 chunk 会保留元数据，例如：
  - `source_file`
  - `chunk_id`
  - `chunk_index`

### 3.3 混合检索

系统同时使用两种检索方式：

| 检索方式 | 作用 |
|---|---|
| BM25 关键词检索 | 适合匹配明确关键词 |
| Embedding 向量检索 | 适合理解语义相近的问题 |

当前混合打分策略：

```text
hybrid_score = 0.4 * bm25_score + 0.76* vector_score
```

系统根据混合分数排序后返回 Top-K 相关片段。

### 3.4 问答生成

- 将用户问题和检索得到的上下文片段拼接为 Prompt；
- 调用大模型生成答案；
- 当前默认使用 DeepSeek 接口；
- 要求模型只能基于参考资料回答；
- 返回结构化 JSON，包括答案、引用、置信度等字段。

### 3.5 拒答机制

当系统判断文档不足以回答问题时，会拒绝回答。

触发拒答的常见情况：

- 知识库中没有文档；
- 检索结果为空；
- Top1 与 Top2 的 hybrid 分数差距过低；
- Top1 的 vector 分数过低；
- 问题明显超出知识库范围。

默认拒答回答：

```text
我无法根据提供的文档确定答案
```

### 3.6 调试与演示接口

项目提供多个接口，方便观察 RAG 的中间过程：

| 接口                       | 作用           |
| ------------------------ | ------------ |
| `/health`                | 健康检查         |
| `/docs-preview`          | 查看已加载文档      |
| `/chunks-preview`        | 预览切分后的 chunk |
| `/search`                | 只检索，不生成答案    |
| `/chat`                  | 完整问答链路       |
| `/extract_refund_policy` | 退款政策结构化抽取示例  |
| `/kb_summary`            | 统计知识库文档情况    |

### 3.7 基础工具调用

当前项目支持简单的规则路由工具调用：

| 工具 | 作用 |
|---|---|
| `list_documents` | 查看当前加载的文档列表与数量 |
| `get_chunk_text` | 根据 `chunk_id` 查看原文 |

当前工具调用属于基础版规则路由，还不是完整的模型自主 function calling。

---

## 4. 系统架构

### 4.1 整体流程

```text
本地文档
  ↓
文档加载
  ↓
文档切块
  ↓
Embedding 向量化
  ↓
构建 BM25 索引与向量索引
  ↓
用户提问
  ↓
根据用户问题类型选择工具和问答
  ↓
问答                              工具                 
  ↓                                ↓
BM25 + 向量混合检索               调用大模型生成答案
  ↓
相关性阈值判断
  ↓
拼接 Prompt
  ↓
调用大模型生成答案
  ↓
解析结构化 JSON
  ↓
返回答案与引用来源
```

### 4.2 分层说明

| 层级 | 说明 |
|---|---|
| 数据处理层 | 读取本地 `.txt` 和 `.md` 文档 |
| 分块层 | 将长文档切分为适合检索的 chunk |
| 检索层 | 使用 BM25 和向量相似度召回相关片段 |
| 生成层 | 将问题和上下文传给大模型生成答案 |
| 输出层 | 返回统一 JSON，包括答案、引用和状态信息 |

---

## 5. 项目目录结构

```text
mini-rag/
├── app/
│   ├── main.py          # FastAPI 入口，暴露接口
│   ├── rag.py           # RAG 逻辑主控
│   ├── rag_debug.py     # RAG 调试和预览
│   ├── rag_generation.py# RAG 答案生成与结果解析
│   ├── rag_ingest.py    # RAG 文档摄取与切分
│   ├── rag_store.py     # RAG 索引构建与缓存持久化
│   ├── rag_retrieval.py # RAG 检索与检索前后判断
│   ├── config.py        # 配置读取与校验
│   ├── prompts.py       # Prompt 管理
│   ├── rebuild_kb.py    # 手动重建知识库索引
│   └── utils.py         # 日志等工具函数
├── data/
│   ├── eval/            # 评测问题
│   └── kb/              # 知识库文档
├── outputs/             # 评测结果输出目录
├── scripts/
│   └── eval.py          # 离线评测脚本
├── tests/
│   └── test_api.py      # 接口测试
├── .env.example         # 环境变量模板
├── .gitignore
├── requirements.txt
└── README.md
```

> 注意：如果你的实际文件名仍是 `untils.py` 或目录名仍是 `test/`，请保持 README 与代码仓库一致。建议后续统一改为 `utils.py` 和 `tests/`。

---

## 6. 环境依赖

建议使用：

```text
Python 3.10+
```

核心依赖：

- fastapi
- uvicorn
- pydantic
- python-dotenv
- charset-normalizer
- langchain-text-splitters
- numpy
- jieba
- rank-bm25
- sentence-transformers
- requests

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 7. 环境配置

本项目使用 `.env` 管理运行时配置。

### 7.1 创建配置文件

macOS / Linux：

```bash
cp .env.example .env
```

Windows：

```bash
copy .env.example .env
```

### 7.2 配置项说明

| 配置项                    | 是否必填 | 默认值                                            | 说明                        |
| ---------------------- | ---- | ---------------------------------------------- | ------------------------- |
| `RAG_FILE_PATH`        | 是    | 无                                              | 知识库文档路径或目录，例如 `./data/kb` |
| `TOP_K`                | 否    | `3`                                            | 默认召回条数                    |
| `CHUNK_SIZE`           | 否    | `800`                                          | 每个文本块最大长度                 |
| `CHUNK_OVERLAP`        | 否    | `120`                                          | 相邻文本块重叠长度                 |
| `EMBEDDING_MODEL_NAME` | 是    | 无                                              | Embedding 模型名称            |
| `LLM_API_KEY`          | 是    | 无                                              | 大模型接口密钥                   |
| `MODEL_NAME`           | 否    | `deepseek-chat`                                | 大模型名称                     |
| `LLM_API_URL`          | 否    | `https://api.deepseek.com/v1/chat/completions` | 大模型接口地址                   |
| `MARGIN_THRESHOLD`     | 否    | `0.01`                                         | Top1与Top2 hybrid 分数差距拒答阈值 |
| `VECTOR_THRESHOLD`     | 否    | `0.35`                                         | vector 分数拒答阈值             |
| `MAX_CONTEXT_CHARS`    | 否    | 视代码配置而定                                        | 拼接给大模型的最大上下文长度            |
| `TEMPERATURE`          | 否    | 视代码配置而定                                        | 大模型生成随机性                  |

---

## 8. 何时需要重建索引

### 8.1 需要重建索引的情况

以下情况需要重建索引：

- 新增知识库文档；
- 删除知识库文档；
- 修改知识库文档内容；
- 修改 `CHUNK_SIZE`；
- 修改 `CHUNK_OVERLAP`；
- 修改 `EMBEDDING_MODEL_NAME`。

### 8.2 通常不需要重建索引的情况

以下情况通常不需要重建索引：

- 修改 `TOP_K`；
- 修改 `MAX_CONTEXT_CHARS`；
- 修改 `MODEL_NAME`；
- 修改 `TEMPERATURE`；
- 修改 Prompt；
- 修改拒答提示文案。

### 8.3 手动重建索引

macOS / Linux：

```bash
python app/rebuild_kb.py
```

Windows：

```bash
python app\rebuild_kb.py
```

---

## 9. 启动项目

### 9.1 本地启动

1. 创建并激活虚拟环境；
2. 安装依赖；
3. 将 `.txt` 或 `.md` 文档放入知识库目录；
4. 配置 `.env`；
5. 在项目根目录运行服务：

```bash
uvicorn app.main:app --reload
```

启动后访问：

```text
http://127.0.0.1:8000/docs
```

---

## 10. Docker 启动

### 10.1 构建镜像

```bash
docker build -t rag-demo:latest .
```

### 10.2 运行容器

```bash
docker run --rm -p 8000:8000 rag-demo:latest
```

服务启动后访问：

```text
http://localhost:8000/docs
```

---

## 11. API 接口说明

### 11.1 健康检查

```http
GET /health
```

示例：

```bash
curl http://127.0.0.1:8000/health
```

返回示例：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "status": "ok"
  }
}
```

---

### 11.2 文档预览

```http
GET /docs-preview
```

示例：

```bash
curl http://127.0.0.1:8000/docs-preview
```

作用：查看当前加载了哪些文档。

---

### 11.3 Chunk 预览

```http
GET /chunks-preview
```

示例：

```bash
curl "http://127.0.0.1:8000/chunks-preview?limit=10"
```

作用：查看文档切分后的 chunk 内容，方便调试切块效果。

---

### 11.4 检索接口

```http
POST /search
```

作用：只执行检索，不调用大模型生成答案，适合演示“先检索，再生成”的过程。

请求示例：

```bash
curl -X POST "http://127.0.0.1:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "每个 Vue 应用都是通过什么函数创建应用实例的？",
    "top_k": 5
  }'
```

返回示例：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "question": "每个 Vue 应用都是通过什么函数创建应用实例的？",
    "status": "ok",
    "reason": "检索成功",
    "results": [
      {
        "reference_id": "Reference 1",
        "source_path": "D:\\python-learn\\mini-rag\\data\\kb\\application.md",
        "file_name": "application.md",
        "chunk_id": 1,
        "position": 1,
        "snippet": "[标题路径]创建一个Vue应用{#creating-a-vue-application}>应用实例{#the-application-instance}每个Vue应用都是通过[`createApp`](/api/application#createapp)函数创建一个新的**应用实例**：```jsimport{createApp}from'vue'constapp=createApp({/*根组件选项*/})```",
        "score": 1,
        "bm25_score": 31.69338498402554,
        "vector_score": 0.741374135017395
      },
      {
        "reference_id": "Reference 2",
        "source_path": "D:\\python-learn\\mini-rag\\data\\kb\\application.md",
        "file_name": "application.md",
        "chunk_id": 6,
        "position": 6,
        "snippet": "[标题路径]创建一个Vue应用{#creating-a-vue-application}>多个应用实例{#multiple-application-instances}应用实例并不只限于一个。`createApp`API允许你在同一个页面中创建多个共存的Vue应用，而且每个应用都拥有自己的用于配置和全局资源的作用域。```jsconstapp1=createApp({/*...*/})app1.mount('#container-1')constapp2=createApp({/*...*/})app2.mount('#container-2')```如果你正在使用Vue来增强服务端渲染HTM",
        "score": 0.8979523181915283,
        "bm25_score": 27.480818881081085,
        "vector_score": 0.6945750117301941
      },
      {
        "reference_id": "Reference 3",
        "source_path": "D:\\python-learn\\mini-rag\\data\\kb\\application.md",
        "file_name": "application.md",
        "chunk_id": 5,
        "position": 5,
        "snippet": "[标题路径]创建一个Vue应用{#creating-a-vue-application}>应用配置{#app-configurations}应用实例会暴露一个`.config`对象允许我们配置一些应用级的选项，例如定义一个应用级的错误处理器，用来捕获所有子组件上的错误：```jsapp.config.errorHandler=(err)=>{/*处理错误*/}```应用实例还提供了一些方法来注册应用范围内可用的资源，例如注册一个组件：```jsapp.component('TodoDeleteButton',TodoDeleteButton)```这使得`TodoDeleteButton`在应",
        "score": 0.7884707450866699,
        "bm25_score": 23.78919390226851,
        "vector_score": 0.6310862302780151
      },
      {
        "reference_id": "Reference 4",
        "source_path": "D:\\python-learn\\mini-rag\\data\\kb\\application.md",
        "file_name": "application.md",
        "chunk_id": 2,
        "position": 2,
        "snippet": "[标题路径]创建一个Vue应用{#creating-a-vue-application}>根组件{#the-root-component}我们传入`createApp`的对象实际上是一个组件，每个应用都需要一个“根组件”，其他组件将作为其子组件。如果你使用的是单文件组件，我们可以直接从另一个文件中导入根组件。```jsimport{createApp}from'vue'//从一个单文件组件中导入根组件importAppfrom'./App.vue'constapp=createApp(App)```虽然本指南中的许多示例只需要一个组件，但大多数真实的应用都是由一棵嵌套的、可重用的组件树组成的。",
        "score": 0.7358264923095703,
        "bm25_score": 24.326581091026174,
        "vector_score": 0.5634583234786987
      },
      {
        "reference_id": "Reference 5",
        "source_path": "D:\\python-learn\\mini-rag\\data\\kb\\introduction.md",
        "file_name": "introduction.md",
        "chunk_id": 116,
        "position": 16,
        "snippet": "[标题路径]简介{#introduction}>API风格{#api-styles}>该选哪一个？{#which-to-choose}两种API风格都能够覆盖大部分的应用场景。它们只是同一个底层系统所提供的两套不同的接口。实际上，选项式API是在组合式API的基础上实现的！关于Vue的基础概念和知识在它们之间都是通用的。选项式API以“组件实例”的概念为中心(即上述例子中的`this`)，对于有面向对象语言背景的用户来说，这通常与基于类的心智模型更为一致。同时，它将响应性相关的细节抽象出来，并强制按照选项来组织代码，从而对初学者而言更为友好。组合式API的核心思想是直接在函数作用域内定义响应式",
        "score": 0.7060389518737793,
        "bm25_score": 19.44158894043517,
        "vector_score": 0.6084402203559875
      }
    ]
  }
}
```

---

### 11.5 问答接口

```http
POST /chat
```

作用：执行完整 RAG 问答链路。

请求示例：

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "每个 Vue 应用都是通过什么函数创建应用实例的？",
    "top_k": 5
  }'
```

返回示例：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "question": "每个 Vue 应用都是通过什么函数创建应用实例的？",
    "answer": "每个 Vue 应用都是通过 `createApp` 函数创建应用实例的。",
    "references": [
      {
        "reference_id": "Reference 1",
        "file_name": "application.md",
        "source_path": "D:\\python-learn\\mini-rag\\data\\kb\\application.md",
        "chunk_id": 1,
        "position": 1,
        "snippet": "[标题路径]创建一个Vue应用{#creating-a-vue-application}>应用实例{#the-application-instance}每个Vue应用都是通过[`createApp`](/api/application#createapp)函数创建一个新的**应用实例**：```jsimport{createApp}from'vue'constapp=createApp({/*根组件选项*/})```",
        "score": 1
      }
    ],
    "confidence": 0.521,
    "need_clarification": false,
    "status": "ok",
    "clarification_question": "",
    "reason": "检索命中有效证据，且答案成功绑定引用"
  }
}
```

---

## 12. 知识库范围示例：Vue 3 入门问答

当前项目可作为 Vue 3 入门知识库问答系统使用。

### 12.1 当前覆盖内容

- Vue 是什么，以及适用场景；
- Vue 项目的快速开始方式；
- Vue 应用的创建与挂载；
- 模板语法与常见指令；
- 响应式基础；
- 计算属性；
- 条件渲染；
- 列表渲染；
- 事件处理；
- 基础组件概念与使用方法。

### 12.2 适合回答的问题

- Vue 是什么？
- Vue 应用如何创建？
- 模板语法怎么写？
- 什么是响应式？
- 计算属性和方法有什么区别？
- `v-if` 和 `v-show` 有什么区别？
- `v-for` 为什么建议配合 `key`？
- Vue 事件怎么绑定？
- Vue 组件是什么，怎么使用？

### 12.3 不适合回答的问题

- Vue 组件高级通信机制的完整实现细节；
- 插槽、依赖注入等进阶专题；
- Vue 工程化、路由、状态管理等扩展主题；
- 与 Vue 无关的通用前端问题。

---

## 13. 手工测试清单

### 13.1 能回答的问题

请求示例：

```json
{
  "question": "每个 Vue 应用都是通过什么函数创建应用实例的？",
  "top_k": 5
}
```

检查点：

- 回答内容是否和文档一致；
- 是否返回引用来源；
- 引用片段是否能支撑答案。

---

### 13.2 不能回答的问题

请求示例：

```json
{
  "question": "这份文档中提到的 Vue 当前最新稳定版本号是多少？",
  "top_k": 5
}
```

期望结果：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "answer": "我无法根据提供的文档确定答案",
    "references": []
  }
}
```

---

### 13.3 空问题

请求示例：

```json
{
  "question": "",
  "top_k": 5
}
```

期望结果：

```json
{
  "success": false,
  "code": "EMPTY_QUESTION",
  "message": "问题不能为空，请输入你想查询的内容"
}
```

---

### 13.4 重复提问

请求示例：

```json
{
  "question": "这份文档中提到的 Vue 当前最新稳定版本号是多少？",
  "top_k": 5
}
```

检查点：

- 多次回答的核心内容是否一致；
- 多次返回的引用来源是否大体一致；
- 不应出现明显随机编造。

---

## 14. 已踩过的坑

### 14.1 数据目录为空导致初始化异常

问题：如果数据目录为空，还继续构建 Embedding 和 BM25，后续链路会不稳定。  
解决：没有文档时直接返回 `NO_DOCUMENTS` 状态，不继续初始化。

### 14.2 单一检索方式不够稳定

问题：只用关键词检索容易漏召回，只用向量检索有时排序不稳定。  
解决：使用 BM25 + 向量检索的混合检索方式，提高召回稳定性。

### 14.3 模型输出格式不稳定

问题：大模型有时会输出 Markdown 代码块或额外解释，导致 JSON 解析失败。  
解决：增加 JSON 解析和兜底逻辑，解析失败时返回安全默认答案。

### 14.4 模型可能脱离文档回答

问题：RAG 不能完全避免幻觉。  
解决：

- 检索分数过低时直接拒答；
- Prompt 中要求模型只能基于参考资料回答；
- 返回引用来源，方便用户核验。

---

## 15. 已知问题

### 15.1 `/chat` 用户输出与日志状态不一致

现象：当问题无法根据文档回答时，接口会正常返回“我无法根据提供的文档确定答案”，但日志中仍显示流程成功结束。

影响：日志只能说明接口执行成功，不能直观看出本次回答是“正常回答”还是“拒答”。

当前处理：暂时接受。

后续优化：

- 在返回结果中明确区分 `answered`、`rejected`、`need_clarification` 等状态；
- 在日志中记录拒答原因，例如 `LOW_SCORE`、`NO_DOCUMENTS`、`OUT_OF_SCOPE`。

---

## 16. 下一步优化方向

后续可以从以下方向继续完善：

- 支持 PDF、DOCX 文档解析；
- 支持扫描版 PDF 的 OCR；
- 增加 rerank，提高 Top-K 排序质量；
- 优化 chunk 切分策略；
- 增加多轮对话能力；
- 增加前端页面；
- 增加自动化评测集；
- 增加更完整的 function calling 能力；
- 增加日志中的回答状态与拒答原因；
- 支持多知识库切换。

---

## 17. 项目展示讲稿

可以用下面这段话快速介绍项目：

> 我做的是一个基于 FastAPI 的最小中文 RAG 知识库问答系统，主要解决大模型无法直接回答本地文档内容、容易编造答案、缺少引用来源的问题。  
> 系统会先读取本地 `.txt` 和 `.md` 文档，对文档进行切块和向量化，同时构建 BM25 关键词索引和向量索引。用户提问时，系统会通过混合检索召回相关片段，再把问题和上下文一起交给大模型生成答案。  
> 为了减少幻觉，我加入了相关性阈值判断和拒答机制；为了方便验证，我让系统返回答案对应的引用来源。目前项目已经支持健康检查、文档预览、chunk 预览、检索接口和完整问答接口，适合作为一个可演示、可调试、可写进简历的 RAG 入门项目。

---

## 18. 简历描述参考

可以将项目压缩成下面几条写进简历：

- 基于 FastAPI 实现中文 RAG 知识库问答系统，支持 `.txt` / `.md` 文档加载、递归切块、Embedding 向量化、BM25 + 向量混合检索和大模型答案生成。
- 设计检索增强问答链路，支持返回引用来源、结构化 JSON 输出和低相关问题拒答，提升回答可解释性并降低模型幻觉风险。
- 提供 `/search`、`/chat`、`/docs-preview`、`/chunks-preview` 等接口，支持完整 RAG 流程调试、演示和手工测试。
- 封装 Prompt 管理、基础工具调用和评测脚本，使项目具备可维护、可验证、可展示的 LLM 应用工程雏形。
