# ADR 0015：使用官方 pydantic-settings 管理应用配置

- 状态：已接受
- 日期：2026-07-28

## 背景

项目原配置由一个可变静态 `Settings` 类承载。导入 `app.config` 会立即调用
`load_dotenv()`，再通过 `os.getenv()`、`env_int()`、`env_float()`、
`env_bool()` 和 `env_csv()` 构造类属性。其结果是导入顺序决定配置快照，本地
`.env` 可污染测试，测试依赖 monkeypatch 类属性，并且 Registry、Runtime、模型和
Graph 很难证明使用同一组配置。

## 决策

使用官方 `pydantic-settings==2.14.2`。`app.config.AppSettings` 是唯一完整应用
配置模型，继承公开的 `BaseSettings`，只依赖标准库、Pydantic 和
pydantic-settings。项目删除静态 `Settings` 类、直接 python-dotenv 调用和手写
环境解析框架。

`AppSettings` 使用 lower snake case 字段；pydantic-settings 继续从现有 uppercase
环境变量读取值。配置来源优先级为：

```text
显式初始化参数 > 环境变量 > 项目根目录 .env > 字段默认值
```

测试和 inspection 显式使用 `_env_file=None`。生产代码通过
`get_settings()` 的惰性 `lru_cache` 获取快照，不在模块导入时实例化配置。

## 不可变与注入

模型配置为 `frozen=True`。冻结快照防止请求期间修改字段，也让缓存键、显式注入和
两个配置实例的隔离语义清晰。FastAPI lifespan、ChatService、LLM Factory、Tool
Registry/Runtime、Graph/Checkpointer、RAG、Storage 和 Observability 在现有边界
读取或接收实例，不增加第二条授权或工具执行路径。

配置模型构造只做解析、规范化和纯校验，不创建目录、文件、模型客户端、Registry、
线程池、Chroma client、SQLite 连接或 Checkpointer。`validate_startup_environment`
在 lifespan 中单独检查真实 API key、RAG 目录和持久化父目录；因此导入、inspection
及不需要模型的单元测试不受外部前提影响。

## 路径、密钥与角色 fallback

相对路径统一相对项目根解析，绝对路径保持不变，`~` 会展开。配置构造不创建路径；
模型缓存目录只在模型加载边界由 `ensure_model_cache_dir()` 创建。缓存优先级是
`MODEL_CACHE_DIR`、`HF_HUB_CACHE`、`HF_HOME/hub`、
`SENTENCE_TRANSFORMERS_HOME`、项目默认目录。

密钥字段使用 `SecretStr`。主回答模型保留 `LLM_API_KEY` 优先、
`DEEPSEEK_API_KEY` fallback。Query Rewrite 的空 key/model/URL 分别复用回答模型；
Judge 的空 key/URL/timeout 复用回答配置；Decompose 继续使用独立 key，不隐式复用
回答 key。

## Permissions 与 Web proxy

`TOOL_DEFAULT_PERMISSIONS` 使用公开 `NoDecode` 标记阻止 settings source 把 CSV
当 JSON 解码，再由字段 validator 转为保持顺序的 `tuple[str, ...]`。每个权限必须
是精确的 `resource:action`，不接受 wildcard，且集合不能为空。

`WEB_SEARCH_PROXY` 和 `WEB_FETCH_PROXY` 保留独立值。默认安全模式直接使用各自
代理并保留 hardened DNS/IP 约束；只有显式开启 `WEB_TRUSTED_PROXY_MODE` 且
`WEB_TRUSTED_PROXY_URL` 通过 HTTP(S) URL 校验时，computed property 才为搜索、
HTML 和 PDF 下载返回统一代理。模型和工具结果看不到代理值。

## 缓存边界

缓存按资源职责分开，避免 `app.config` 反向依赖高层模块：

- Settings：`get_settings()` / `clear_settings_cache()`；
- Model Factory：角色 getter / `clear_model_factory_cache()`；
- Local Tool Catalog：`get_local_tool_specs()` 与 `get_local_tools()` 按冻结 Settings
  缓存，受控测试通过各函数的 `cache_clear()` 失效；
- Agent Runtime：`_get_cached_framework_agent(settings)` / `reset_agent_runtime()`；
- Graph/Checkpointer：按冻结 Settings 构建 / `reset_agent_graph_runtime()` 或
  `close_agent_graph_runtime()`。

更改环境变量不会自动改变已经运行的快照。测试必须按依赖方向清理缓存；生产服务
优先通过重启应用配置变更。

## 后果

收益是配置来源、类型、范围、fallback、导入安全和测试隔离可以由单一 Schema
验证；生产模块不再读取全局可变类属性。代价是需要显式传递快照，并在受控配置刷新
时清理相关资源缓存。

本 ADR 不引入动态热更新、远程配置中心、多个配置子模型或完整依赖倒置。Import
Linter 与正式模块依赖合同留给后续 PR15；本阶段只用 Runtime inspection 和搜索
检查守住 `app.config` 的向下依赖边界。
