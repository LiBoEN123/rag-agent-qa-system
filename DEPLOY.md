# DEPLOY

## 1.项目简介
这是一个基于Docker运行的RAG/Agent项目，提供中文问答服务

## 2.环境要求
- Docker已安装
- 如果要使用 GPU：宿主机已安装 NVIDIA 驱动和 NVIDIA Container Toolkit，并且 `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` 能正常输出显卡信息
- 需要有可用的大模型API-key
- 准备好要查询的知识库文档目录

## 3.构建镜像
代码或依赖变化时才需要重新构建镜像：
```bash
docker build -t agent-rag-demo:latest .
```

默认会安装 CUDA 12.1 版 PyTorch，供 embedding/rerank 在容器内使用 GPU。若宿主机驱动不支持对应 CUDA 运行时，可在构建时指定其他 PyTorch CUDA wheel 源：
```bash
docker build \
  --build-arg TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu121 \
  -t agent-rag-demo:latest .
```

若基础镜像拉取失败，可先执行以下命令，再重新 build：
```bash
docker pull python:3.11-slim
```

## 4.环境变量
```
RAG_FILE_PATH：必填，要参考的文档的路径，例如：./data/白鹿原.txt
INDEX_STORE_DIR：选填，默认 ./data/manifest，保存索引 manifest
CHROMA_PERSIST_DIR：选填，默认 ./data/chroma，保存 ChromaDB 向量索引
ENABLE_CHROMA：选填，默认 true，是否启用 ChromaDB 持久化索引
TOP_K：选填，默认3，表示默认召回条数
CHUNK_OVERLAP：选填，默认120，表示相邻文本块之间的重叠长度（按字符数）
CHUNK_SIZE：选填，默认800，表示每个文本块的最大长度（按字符数）
EMBEDDING_MODEL_NAME：必填，表示嵌入模型名
LLM_API_KEY：必填，表示模型接口密钥；也兼容旧变量 DEEPSEEK_API_KEY
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
推荐使用 docker compose 启动。`docker-compose.yml` 已经把知识库、ChromaDB 索引、manifest、日志和模型缓存挂载到了宿主机，重启容器不会丢缓存。

第一次启动建议前台运行，方便直接看到下载模型、构建索引和启动完成日志：
```bash
docker compose up
```

看到以下日志后，表示服务已经可以访问：
```text
Application startup complete.
```

如果希望后台运行，则使用：
```bash
docker compose up -d
```

后台运行不会在当前终端显示启动过程。查看状态：
```bash
docker compose ps
```

查看实时日志：
```bash
docker compose logs -f rag-api
```

停止服务：
```bash
docker compose down
```

如果不用 compose，至少要把运行时数据挂载出来：
```bash
docker run --rm --gpus all -p 8000:8000 --env-file .env \
  -e RAG_FILE_PATH=/app/data/kb \
  -e INDEX_STORE_DIR=/app/data/manifest \
  -e CHROMA_PERSIST_DIR=/app/data/chroma \
  -e HF_HOME=/cache/huggingface \
  -e MODEL_CACHE_DIR=/cache/huggingface/hub \
  -e SENTENCE_TRANSFORMERS_HOME=/cache/sentence-transformers \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v "$(pwd)/data/kb:/app/data/kb:ro" \
  -v "$(pwd)/data/chroma:/app/data/chroma" \
  -v "$(pwd)/data/manifest:/app/data/manifest" \
  -v "$(pwd)/logs:/app/logs" \
  -v agent-rag-model-cache:/cache \
  --add-host=host.docker.internal:host-gateway \
  agent-rag-demo:latest
```

注意：`--rm` 会删除容器本身，但不会删除镜像；只要上面的 volume 还在，索引和模型缓存仍会保留。第一次启动或知识库文件变化时会重建索引，这是正常的。

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

如果启用了 GPU，可以再验证容器内 PyTorch 是否能看到显卡：
```bash
docker compose exec rag-api python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```
第一行应输出 `True`，第二行应输出显卡名称。若仍是 `False`，优先检查宿主机 NVIDIA 驱动、NVIDIA Container Toolkit 和 Docker Compose 版本。

## 7.常见问题
- API Key不可用
- 每次启动都重建索引：检查是否挂载了 `./data/chroma:/app/data/chroma` 和 `./data/manifest:/app/data/manifest`。
- 每次启动都重新下载 embedding/rerank 模型：检查是否挂载了 `agent-rag-model-cache:/cache`，并确认 `MODEL_CACHE_DIR=/cache/huggingface/hub`，或使用 `docker compose up -d`。
- 容器内 `torch.cuda.is_available()` 是 `False`：确认启动命令包含 GPU 配置，宿主机能运行 `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`，并重新构建镜像确保安装的是 CUDA 版 PyTorch。
- 修改知识库 `.txt`/`.md`、`CHUNK_SIZE`、`CHUNK_OVERLAP`、`EMBEDDING_MODEL_NAME`、`RERANK_MODEL_NAME` 后会触发重建索引，这是预期行为。
- 本地 Ollama 无法访问时，确认启动命令包含 `--add-host=host.docker.internal:host-gateway`，或使用 compose 配置。


















