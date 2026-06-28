# DEPLOY

## 1.项目简介
这是一个基于Docker运行的RAG/Agent项目，提供中文问答服务

## 2.环境要求
- Docker已安装
- 需要有可用的大模型API-key
- 准备好要查询的知识库文档目录

## 3.构建镜像
先在终端尝试命令
```bash
docker build -t agent-rag-demo:latest .
```
若出现错误则先执行以下命令，再执行上述命令
```bash
docker pull python:3.14-slim
```

## 4.环境变量
```
RAG_FILE_PATH：必填，要参考的文档的路径，例如：./data/白鹿原.txt
TOP_K：选填，默认3，表示默认召回条数
CHUNK_OVERLAP：选填，默认120，表示相邻文本块之间的重叠长度（按字符数）
CHUNK_SIZE：选填，默认800，表示每个文本块的最大长度（按字符数）
EMBEDDING_MODEL_NAME：必填，表示嵌入模型名
DEEPSEEK_API_KEY：必填，表示模型接口密钥
MODEL_NAME：选填，默认deepseek-chat，表示大模型名称
LLM_API_URL：选填，默认https://api.deepseek.com/v1/chat/completions，表示模型接口地址
HYBRID_THRESHOLD：选填，默认0.40，表示混合分数拒答阈值，如果结果分数低于该值，则触发拒答。
VECTOR_THRESHOLD：选填，默认0.35，表示向量分数拒答阈值，如果结果分数低于该值，则触发拒答。
```
这里要说明一下两种情况
### 4.1 项目用本地 Ollama
那`.env`中模型的配置参数为
```text
MODEL_NAME=qwen2:7b
LLM_API_URL=http://host.docker.internal:11434/v1/chat/completions
LLM_API_KEY=ollama
```

### 4.2 DeepSeek 或其他云 API
那`.env`中模型的配置参数为
```text
MODEL_NAME=deepseek-chat
LLM_API_URL=https://api.deepseek.com/v1/chat/completions
LLM_API_KEY=你的真实 key
```

## 5.启动服务
由于配置文件.env未传入镜像，故在常规启动代码上添加`--env-file .env`
```bash
docker run --rm -p 8000:8000 --env-file .env agent-rag-demo:latest
```

## 6.验证服务
在浏览器输入以下网址并打开
```
http://localhost:8000/docs
```
然后先检查接口`/health`结果是否为
```
{"code":0,"message":"success","data":{"status":"ok"}}
```
然后在`/search`及`/chat`接口输入问题及召回条数
```
{
  "question": "每个 Vue 应用都是通过什么函数创建应用实例的？",
  "top_k": 5
}
```
查看结果是否能正常返回

## 7.常见问题
- API Key不可用
- 直接运行命令`docker run --rm -p 8000:8000 --env-file .env rag-demo:latest`可能会出现网络不兼容问题导致失败























