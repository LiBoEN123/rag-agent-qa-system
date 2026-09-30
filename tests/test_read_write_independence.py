"""在全新进程中禁止导入另一子系统，并经过真实持久索引执行。"""

import os
import subprocess
import sys
from pathlib import Path


def _run(source, tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", source, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


_SETUP = '''
import sys
import importlib.abc
from pathlib import Path

class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
            raise AssertionError("forbidden dependency: " + fullname)

sys.meta_path.insert(0, BlockImports())
from app import config
from langchain_core.embeddings import Embeddings
root = Path(sys.argv[1])
settings = config.AppSettings(
    _env_file=None, llm_api_key="fake", mcp_tools_enabled=False, tool_audit_enabled=False,
    rag_file_path=root / "corpus", index_store_dir=root / "manifest",
    chroma_persist_dir=root / "chroma", sqlite_db_path=root / "app.sqlite3",
)
config.get_settings = lambda: settings
class Embedding(Embeddings):
    def embed_documents(self, texts): return [[1.0, 0.0] for _ in texts]
    def embed_query(self, text): return [1.0, 0.0]
'''


def test_writer_then_reader_and_agent_run_in_separate_processes(tmp_path):
    _run('forbidden = ["app.rag", "app.agent", "app.services"]\n' + _SETUP + '''
from app.indexing.service import IndexingService
from app.ingestion import IngestionService, IngestionRequester, build_knowledge_document
from app.sources.service import ExternalKnowledgeSourceService
from app.storage.ingestion import SQLiteCandidateRepository

requester = IngestionRequester(principal_id="operator", principal_type="service", source="test")
domain = IngestionService(SQLiteCandidateRepository(settings.sqlite_db_path))
source = ExternalKnowledgeSourceService(requester=requester, ingestion_service=domain, settings=settings)
from io import BytesIO
pending = source.propose_upload_file(file=BytesIO(b"temporary external content"), filename="note.txt")
approved = domain.approve(pending.candidate_id, decided_by=requester)
writer = IndexingService(settings=settings, candidates=domain.repository, embeddings=Embedding())
assert writer.index_approved(approved.candidate_id)["num_added"] == 1
assert writer.remove_indexed(approved.candidate_id)["num_deleted"] == 1
settings.rag_file_path.mkdir()
(settings.rag_file_path / "guide.txt").write_text("申请条件是年满十八岁。", encoding="utf-8")
assert writer.sync_local_corpus()["num_added"] == 1
assert writer.sync_local_corpus()["num_skipped"] == 1
''', tmp_path)

    _run('forbidden = ["app.ingestion", "app.indexing", "app.sources", "app.storage.ingestion", "langchain_classic.indexes"]\n' + _SETUP + '''
import asyncio
from langchain_core.messages import AIMessage, ToolMessage
from app.rag import runtime
from app.agent import framework_agent
from app.tools.context import ToolInvocationContext
from tests.support.agent_runtime import FakeToolCallingModel, build_agent_harness

class Reranker:
    def predict(self, pairs): return [1.0 for _ in pairs]
runtime.load_embedding_model = lambda: Embedding()
runtime._load_rerank_model = lambda: Reranker()
assert runtime.load_index_cache()[0]
assert runtime.bm25.corpus_size == 1

def answer(model, messages):
    message = next(m for m in reversed(messages) if isinstance(m, ToolMessage))
    evidence = message.artifact["output"]["evidence"]
    assert evidence[0]["page_content"] == "申请条件是年满十八岁。"
    return model.final_response(outcome="answer", answer="申请条件是年满十八岁。", evidence_ids=[evidence[0]["id"]])

model = FakeToolCallingModel([
    AIMessage(content="", tool_calls=[{"name": "search_knowledge_base", "args": {"query": "申请条件", "top_k": 1}, "id": "search", "type": "tool_call"}]),
    answer,
])
framework_agent.get_agent_model = lambda: model
context = ToolInvocationContext(principal_id="qa", permissions=("knowledge_base:read",), trace_id="independent", source="test")
async def run():
    states = [s async for s in build_agent_harness(settings).astream({
        "original_question": "申请条件是什么？", "effective_question": "申请条件是什么？",
        "trace_id": "independent",
    }, context=context)]
    assert states[-1]["result"]["status"] == "OK", states[-1]["result"]
    assert states[-1]["result"]["answer"] == "申请条件是年满十八岁。"
asyncio.run(run())
''', tmp_path)
