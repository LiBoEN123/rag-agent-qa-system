# FRAMEWORK-SLIM-4 — DeepEval Tool Correctness

此文是 FRAMEWORK-SLIM-4 的历史记录，保留该阶段的实现边界和版本选择依据。
后续 EVALUATION-SLIM-2 已迁移 RAG Judge，SLIM-3 已移除首工具与路径等过程合同；当前能力与运行方式以
[Baseline 合同](baseline-v2.md)为准。

该阶段 `deepeval_adapter.py` 只转换实际/期望工具名称，使用 `ToolCorrectnessMetric`
判断包含关系与完整有序序列。`first_tool_any_of`、正文证据、来源、状态与关键词
仍由项目检查。既有可选 RAG Judge 未迁移到 GEval。

## 框架兼容性与迁移边界

官方 [PyPI 4.2.2](https://pypi.org/project/deepeval/4.2.2/) 是审计时最新稳定版，
但其 Python 3.12 依赖 PostHog >=7，与当前 Chroma 1.3.7 要求的 PostHog <6
冲突。安装探针的 `pip check` 确认冲突后，已恢复 PostHog 5.4.0。
检查 3.9.1–3.9.9、4.0.10、4.1.10、4.2.1 的 PyPI 依赖元数据，选择
[DeepEval 3.9.6](https://pypi.org/project/deepeval/3.9.6/) 固定到两个 requirements
文件；3.9.7 起的 PostHog 要求已不兼容。3.9.6 支持 Python 3.12，要求
Pydantic >=2.11.7,<3，当前环境最终 `pip check` 通过。
[Tool Correctness](https://deepeval.com/docs/metrics-tool-correctness) 可独立 measure；
下载 [3.9.6 源码](https://github.com/confident-ai/deepeval/blob/v3.9.6/deepeval/metrics/tool_correctness/tool_correctness.py)
确认：不提供 available_tools 时不进行 LLM 评分，但构造器仍会初始化
model。须使用拒绝生成的最小模型占位，避免默认 OpenAI client/API key 要求。

已实测：空期望代表“未声明约束”；first_tool_any_of 是 OR；contains
允许额外调用且重复期望不增加次数要求；sequence 是含重复项的完整有序列表。
只向框架传工具名称，不传完整 trace 或正文。

[Faithfulness](https://deepeval.com/docs/metrics-faithfulness) 衡量答案相对 context
的事实支持；[Contextual Recall](https://deepeval.com/docs/metrics-contextual-recall)
衡量检索 context 对 expected output 的支持，均不等于本项目逐 gold key point
覆盖合同。Answer Relevancy、Contextual Precision/Relevancy 也不是当前 gold-ID
指标的等价替代。Argument Correctness 和这些新增 LLM 指标本轮均不启用。

只调用原生 metric.measure，并显式传 `_log_metric_to_confident=False`，
不用 evaluate/assert_test/Cloud/observe。
在导入前关闭 telemetry、tracing、dotenv autoload、版本检查和框架文件写入；
禁用 DeepEval 自动 pytest 插件，避免它早于项目配置导入并接管普通测试。
无第二套 LLM 配置；已有 RAG judge 继续使用项目 factory + ModelGateway。

## 该阶段边界

- `ToolCorrectnessMetric` 不接收工具正文、问题或答案；不调用真实 Judge。
- Core Agent、RAG Prediction v3 与可选 Online Web 继续使用统一项目 Runner。
- 评测 Settings 禁用生产写入与 MCP；manifest 保留数据、索引、Prompt、工具合同和模型信息。
- RAG headline 为 coverage、retrieval Gold Recall@K、reference Gold Recall 和可选关键点覆盖。
- behavior/final score 与现有 Judge 保留，后续可单独审计；本阶段没有升级依赖。

离线验证见 `tests/test_evaluation_deepeval.py`；语义 Judge 的 fake/cache 验证见
`tests/test_evaluate_rag_judge.py`。
