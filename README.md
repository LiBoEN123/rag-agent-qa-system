# Mini RAG
一个基于FastApi的最小可运行的中文问答RAG系统
它目前支持读取```.txt和.md```本地文件，完成文档切块，文本向量化，混合检索，答案生成和引用返回，适合用来学习和展示RAG的完整流程

---

## 1.项目目标
此项目的目标是实现一个可运行，可调式，可展示的最小RAG中文问答系统，来解决用户面对大量文档时难以准确快速找到信息的难题

相较于传统的关键词检索，本项目希望做到：

- 用户直接用自然语言提问
- 系统先从知识库里检索相关片段
- 然后再基于检索结果生成答案
- 最后返回答案引用，提升可解释性

## 2.当前已实现功能

### 2.1 文档加载
- 自动读取目标数据目录中的`.txt`和`.md`文档
- 暂不支持其他类型文档

### 2.2 文档切块
- `.txt`文档使用递归切分器按段落，句号，标点等规则切块
- `.md`文档先按标题层级切分，然后按`.txt`文档切分方法进行二次切分
- 为每个 chunk 保留 `source_file`、`chunk_id`、`chunk_index` 等元信息

### 2.3 检索
- 使用`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 生成向量
- 使用`jieba`为问题分词+`BM25`进行关键词检索
- 最终采用混合检索：
  - `0.4 * bm25_score`
  - `0.6 * vector_score`

### 2.4 问答生成
- 把用户问题和召回的上下文一起放入大模型
- 当前默认使用 DeepSeek 接口
- 要求模型根据`system_prompt`中的守则进行回答，避免脱离文档编造答案
- 输出结构化 JSON，而且返回实际引用的`Reference`

### 2.5 拒答机制
- 如果召回结果为空，或top1 的 hybrid / vector 分数过低，则拒绝回答
- 默认返回：`我无法根据提供的文档确定答案`

### 2.6 调试接口
- `docs-preview`：查看加载了哪些文档
- `chunks-preview`：预览切分结果chunk
- `search`：根据问题，查看召回结果，不走生成
- `chat`：完整回答链路

---

## 3.系统架构

整体流程如下

文档加载
-> 文档解析
-> 文档切块
-> embedding向量化
-> 构建BM25与向量索引
-> 用户提问
-> 混合检索
-> 阈值判断
-> 调用 LLM 与生成答案
-> 返回答案与引用

### 3.1 数据处理层
负责读取本地数据目录中的`.txt`和`.md`文件，并将其转化为适合切块的内容

### 3.2 分块层
- `.txt`文件按段落，换行，标点等层级进行递归切块
- `.md`文件先按标题层级切分，然后再递归切块
- 输出chunk及其元数据

### 3.3 检索层
- 关键词检索：BM25
- 语义检索：embedding + 向量相似度
- 融合策略：混合打分后取top_k

### 3.4 生成层
- 将问题与检索上下文拼接为prompt
- 让模型只基于检索资料进行作答
- 输出answer + reference

### 3.5 输出层
对外返回统一 JSON，包括：
- answer
- reference
- chunk 来源信息

---

## 4.项目目录结构

```text
mini-rag/
├── app/
│   └── main.py      # FastAPI 入口，暴露接口
│   └── rag.py       # RAG 核心逻辑：加载、切块、检索、生成
│   └── config.py    # 配置读取与校验
│   └── untils.py    # 日志文件配置(logger配置)
├── data/            
│   └── eval/        # 评测问题
│   └── kb/          # 知识库文档
├── outputs/         # 测评结果存放目录
├── scripts/         
│   └── eval.py      # 离线评测脚本
├── test           
│   └── test_api.py  # 接口功能运行测试
├── .env.example     # 用户环境变量
├── .gitignore       # 传入仓库时忽略的文档
└── requirements.txt # 依赖
```

## 5.环境依赖
建议 Python 3.10+

核心依赖包括：
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

## 6.环境配置
本项目使用 `.env` 管理运行时配置，请先根据模板创建本地配置文件。

### 6.1. 复制模板文件

macOS / Linux:

```bash
cp .env.example .env
```

Windows:
```bash
copy .env.example .env
```

### 6.2. 配置项说明
项目根目录的`.env`文件保存了以下环境变量
- RAG_FILE_PATH：必填，要参考的文档的路径，例如：./data/白鹿原.txt
- TOP_K：选填，默认3，表示默认召回条数
- CHUNK_OVERLAP：选填，默认120，表示相邻文本块之间的重叠长度（按字符数）
- CHUNK_SIZE：选填，默认800，表示每个文本块的最大长度（按字符数）
- EMBEDDING_MODEL_NAME：必填，表示嵌入模型名
- LLM_API_KEY：必填，表示模型接口密钥
- MODEL_NAME：选填，默认deepseek-chat，表示大模型名称
- LLM_API_URL：选填，默认https://api.deepseek.com/v1/chat/completions，表示模型接口地址
- HYBRID_THRESHOLD：选填，默认0.40，表示混合分数拒答阈值，如果结果分数低于该值，则触发拒答。
- VECTOR_THRESHOLD：选填，默认0.35，表示向量分数拒答阈值，如果结果分数低于该值，则触发拒答。

## 7.启动方式
```text
1.创建并激活虚拟环境
2.安装依赖：pip install -r requirements.txt
3.把需要检索的`.txt`和`.md`文档放入`data/`目录中
4.在`.env`中把相应的配置项配置好
5.在项目根目录运行 uvicorn app.main:app --reload
6.打开 /docs 测试接口
```

## 8.接口说明与示例

### 8.1 健康检查
```
在浏览器网址输入 http://127.0.0.1:8000/health
返回
{"code":0,"message":"success","data":{"status":"ok"}}
```
### 8.2 问答接口
```
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "DevHub 会不会收集我的个人数据？",
    "top_k": 3
  }'

返回示例：
{
  "code": 0,
  "message": "success",
  "data": {
    "answer": "DevHub 是一个离线本地应用程序，不会主动收集、存储或传输任何个人身份识别信息或敏感信息。",
    "references": [
      {
        "reference_id": "Reference 1",
        "source_file": "privacy-policy.zh.md",
        "chunk_id": "privacy-policy.zh_chunk_1",
        "text": "<p align=\"right\">\n<a href=\"./privacy-policy.md\">English</a>\n</p>\n<!--rehype:style=float: right; bottom: -36px; position: relative;-->  \n隐私政策\n===  \nKenny Wang 开发的 DevHub 应用是一个免费应用。此服务由 Kenny Wang 免费提供，仅供使用。  \n欢迎使用我们的应用！我们尊重并保护您的隐私。本隐私政策旨在解释我们如何收集、使用和保护您的个人信息。请仔细阅读以下内容，以了解我们的隐私做法。  \n**信息收集**  \n我们的应用是一个离线本地应用程序，不会主动收集、存储或传输任何个人身份识别信息或敏感信息。  \n**本地数据缓存**  \n我们可能会在您的设备上本地缓存一些数据，以增强用户体验。这些数据仅存储在您的设备上，不会传输到我们的服务器。  \n**使用权限**  \n我们的应用可能会请求某些设备权限，如相机、相册或位置信息。这些权限仅用于提供特定的应用功能，我们不会滥用它们来收集额外的信息。  \n**第三方服务**  \n我们的应用可能会利用第三方服务，但这些服务仅限于提供应用所需的功能，并不用于收集个人身份信息。  \n**儿童隐私**  \n我们的应用不适用于13岁以下的儿童，我们也不会有意收集与儿童有关的信息。"
      }
    ]
  }
}
```

### 8.3 检索接口`/search`
```
curl -X POST "http://127.0.0.1:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "DevHub 会不会收集我的个人数据？",
    "top_k": 3
  }'
```
此接口适合演示“先检索，再生成”的过程

### 8.4 文档预览`/docs-preview`
```
curl http://127.0.0.1:8000/docs-preview
```

### 8.5 chunk预览`/chunks-preview`
```
curl "http://127.0.0.1:8000/chunks-preview?limit=10"
```

## 9.踩过的坑
### 坑1：没有文档时，初始化会出问题
如果数据目录为空，还继续构建 `embedding 和 BM25`，会导致后续链路不稳定。
现在的做法是：没有文档时直接返回 `NO_DOCUMENTS` 状态，不继续初始化。
### 坑2：只做单一检索时结果不够稳
只用关键词检索容易漏召回，只用向量检索会排序不稳定
所以当前版本使用了 BM25 + 向量混合检索，让召回结果更稳。
### 坑3：模型输出格式不稳定
大模型有时会输出 markdown 代码块或额外解释。
所以我增加了 JSON 解析和兜底逻辑，解析失败时返回安全默认答案。
### 坑4：模型可能脱离文档乱答
RAG 不等于一定不会幻觉。
所以我在两个地方做了控制：
- 检索分数过低直接拒答
- prompt 中要求模型只能基于参考资料回答

## 10.手工测试清单
本节内容是为了补充自动化测试，主要检查接口可用性，问答体验和明显回归问题

### 10.1.能回答的问题
- 输入示例
```
{
  "question": "DevHub 的最低操作系统要求是什么？",
  "top_k": 5
}
```
- 输出示例
```
{
  "code": 0,
  "message": "success",
  "data": {
    "answer": 回答内容与文档大致一致,
    "references":来源主要关注引用文本内容，对比回答是否合理非捏造
  }
}
```

### 10.2.不能答
- 输入示例
```
{
  "question": "DevHub 在 App Store 上的平均评分和评论数是多少？",
  "top_k": 5
}
```
- 输出示例
```
{
  "code": 0,
  "message": "success",
  "data": {
    "answer": "我无法根据提供的文档确定答案"类似输出内容,
    "references": []
  }
}
```

### 10.3.空问题
- 输入示例
```
{
  "question": "",
  "top_k": 5
}
```
- 输出示例
```
{
  "success": false,
  "code": "EMPTY_QUESTION",
  "message": "问题不能为空，请输入你想查询的内容"
}
```

### 10.4.重复提问
- 输入示例
```
{
  "question": "DevHub 的最低操作系统要求是什么？", # 此问题多次重复输入
  "top_k": 5
}
```
- 输出示例
```
{
  "code": 0,
  "message": "success",
  "data": {
    "answer": 多次回答内容核心一致,
    "references":多次引用数量和来源大体一致
  }
}
```

## 11.bug list

### 2026-3-27

#### 11.1. /chat 用户界面输出与日志流程冲突
- 现象：有些根据文档回答不了的问题会以正常输出的方式提示“我无法根据提供的文档确定答案”，但日志这边显示流程一切正常
- 影响：日志如果只是显示“流程成功结束”，对排查价值不够。
- 临时方案：今天接受。
- 后续建议：在main或rag里面增加一个逻辑函数来将此问题显示到我的日志流程当中。

## 12.下一步优化
- 增加 `PDF/DOCX` 文档解析
- 增加 rerank 提升top_k排序质量

## 13.Docker Build

在项目根目录执行：

```bash
docker build -t rag-demo:latest .
```

## 14.Docker Run

构建完成后，运行容器：

```bash
docker run --rm -p 8000:8000 rag-demo:latest
```

服务启动后可访问：

```bash
http://localhost:8000/docs
```

## 15.API使用示例

### 15.1.健康检查（/health）
- 输入示例
```
在浏览器网址输入
http://127.0.0.1:8000/health
```

- 输出示例
```
{"code":0,"message":"success","data":{"status":"ok"}}
```

### 15.2.检索示例(/search)
- 输入示例
```
在浏览器网址输入
http://127.0.0.1:8000/docs
进入search接口
示例输入
{
  "question": "购买七天后还能退款吗",
  "top_k": 5
}
```

- 输出示例
```
{
  "code": 0,
  "data": {
    "question": "购买七天后还能退款吗",
    "references": [
      {
        "bm25_score": 7.186296820637358,
        "chunk_id": "terms-of-service.zh_chunk_5",
        "reference_id": "Reference 1",
        "score": 1,
        "source_file": "terms-of-service.zh.md",
        "text": "[标题路径] 4. 退款服务\n\n如果您在订阅赞助过程中遇到任何问题，都可以通过底部的邮件联系我们获得帮助。如果您仍然感到无法满足您的要求，可以选择申请退款。请注意，退款不由开发者操作，您需联系苹果的支持团队来处理退款事宜。  \n[申请退款](https://support.apple.com/118223)",
        "vector_score": 0.3061803877353668
      },
      {
        "bm25_score": 4.213482890118754,
        "chunk_id": "terms-of-service.zh_chunk_4",
        "reference_id": "Reference 2",
        "score": 0.7092396020889282,
        "source_file": "terms-of-service.zh.md",
        "text": "[标题路径] 3. 订阅赞助\n\n如果您决定订阅赞助，那么您对我的付出表示感谢，就像偶尔给我买杯咖啡一样。捐赠后将不再弹出赞助提示，并解锁部分实用功能。感谢您的慷慨支持，我将信心十足地继续维护 DevHub 并确保其健康运行。",
        "vector_score": 0.21269017457962036
      },
      {
        "bm25_score": 0,
        "chunk_id": "terms-of-service.zh_chunk_6",
        "reference_id": "Reference 3",
        "score": 0.3499600887298584,
        "source_file": "terms-of-service.zh.md",
        "text": "[标题路径] 5. 使用条款的更改\n\n当有新的使用条款更新时，我将在此页面上进行更新，并立即生效。",
        "vector_score": 0.119601771235466
      },
      {
        "bm25_score": 0,
        "chunk_id": "CHANGELOG.zh_chunk_11",
        "reference_id": "Reference 4",
        "score": 0.3488190770149231,
        "source_file": "CHANGELOG.zh.md",
        "text": "[标题路径] [v1.42.1](https://github.com/jaywcjlove/DevHub/releases/tag/v1.42.1)\n\n1. pref: 优化首页侧边栏展示性能\n2. ui: 更新首页\n3. ui: 更新关于页面\n4. ui: 优化生命倒计时布局",
        "vector_score": 0.11875034123659134
      },
      {
        "bm25_score": 0,
        "chunk_id": "terms-of-service.zh_chunk_1",
        "reference_id": "Reference 5",
        "score": 0.32936036586761475,
        "source_file": "terms-of-service.zh.md",
        "text": "<p align=\"right\">\n<a href=\"./terms-of-service.md\">English</a>\n</p>\n<!--rehype:style=float: right; bottom: -36px; position: relative;-->  \n服务条款\n===  \n在订阅捐赠之前，您可以阅读应用的使用协议，以获取更多信息。",
        "vector_score": 0.1042303591966629
      }
    ]
  },
  "message": "success"
}
```

### 15.3.问答示例(/chat)
- 输入示例
```
在浏览器网址输入
http://127.0.0.1:8000/docs
进入chat接口
示例输入
{
  "question": "购买七天后还能退款吗",
  "top_k": 5
}
```

- 输出示例
```
{
  "code": 0,
  "data": {
    "answer": "可以申请退款，但退款不由开发者操作，需联系苹果的支持团队处理。",
    "references": [
      {
        "chunk_id": "terms-of-service.zh_chunk_5",
        "reference_id": "Reference 1",
        "source_file": "terms-of-service.zh.md",
        "text": "[标题路径] 4. 退款服务\n\n如果您在订阅赞助过程中遇到任何问题，都可以通过底部的邮件联系我们获得帮助。如果您仍然感到无法满足您的要求，可以选择申请退款。请注意，退款不由开发者操作，您需联系苹果的支持团队来处理退款事宜。  \n[申请退款](https://support.apple.com/118223)"
      }
    ]
  },
  "message": "success"
}
```
