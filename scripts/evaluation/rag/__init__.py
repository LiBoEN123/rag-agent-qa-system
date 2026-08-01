"""Evaluation Baseline v2 的 RAG 评测内部模块集合。

用途：分离公共解析、检索对齐、语义判分、LLM judge 和报告写入职责。
输入：由上层 ``rag_evaluator`` 传入的评测题、预测记录、索引与 judge 配置。
输出：逐题评测结构、汇总指标和可序列化报告行。
边界：这里不提供独立 CLI；正式入口仍由 ``run_evaluation_baseline.py`` 统一管理。
"""
