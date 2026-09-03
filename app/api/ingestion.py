"""受 API key 保护的永久知识 ingestion operations HTTP adapter。"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Annotated, Awaitable, Callable

from fastapi import APIRouter, Depends, Query, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field, JsonValue, SecretStr, StrictStr, field_validator

from app.config import secret_value
from app.ingestion import (
    CandidateNotFoundError,
    IngestionApproval,
    IngestionCandidate,
    IngestionCandidateStatus,
    IngestionCandidateSummary,
    IngestionRequester,
    IngestionService,
    InvalidCandidateTransitionError,
    KnowledgeSourceType,
    TrustLevel,
)
from app.sources import ExternalKnowledgeSourceService, SourceIngestionError
from app.storage.ingestion import (
    CandidatePersistenceError,
    IndexRunStatus,
    IngestionIndexRun,
    SQLiteIndexRunRepository,
)
from app.tools.context import PrincipalType, ToolInvocationContext


INGESTION_ADMIN_PREFIX = "/admin/ingestion"
INGESTION_ADMIN_HEADER = "X-Ingestion-Admin-Key"
CONTENT_PREVIEW_LIMIT = 2000


class _StrictAPIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GitHubProposalRequest(_StrictAPIModel):
    owner: StrictStr = Field(min_length=1, max_length=128)
    repository: StrictStr = Field(min_length=1, max_length=128)
    path: StrictStr = Field(min_length=1, max_length=1024)
    ref: StrictStr | None = Field(default=None, max_length=256)
    reason: StrictStr | None = Field(default=None, max_length=1000)


class FirecrawlProposalRequest(_StrictAPIModel):
    url: StrictStr = Field(min_length=1, max_length=2048)
    reason: StrictStr | None = Field(default=None, max_length=1000)


class ApproveRequest(_StrictAPIModel):
    reason: StrictStr | None = Field(default=None, max_length=1000)


class RejectRequest(_StrictAPIModel):
    reason: StrictStr = Field(min_length=1, max_length=1000)

    @field_validator("reason")
    @classmethod
    def _reason_required(cls, value: str) -> str:
        if not (value := value.strip()):
            raise ValueError("reject reason 不能为空")
        return value


class CandidateDetailResponse(_StrictAPIModel):
    summary: IngestionCandidateSummary
    requester: IngestionRequester
    approval: IngestionApproval
    safe_metadata: dict[str, JsonValue]
    content_preview: StrictStr
    content_length: int = Field(ge=0)
    preview_truncated: bool
    content_is_untrusted: bool
    latest_index_result: IngestionIndexRun | None = None


@dataclass(frozen=True)
class IngestionOperationsDependencies:
    ingestion_service: IngestionService
    source_service: ExternalKnowledgeSourceService
    index_runs: SQLiteIndexRunRepository
    requester: IngestionRequester
    admin_api_key: SecretStr


class IngestionAPIError(Exception):
    """稳定且不携带内部异常信息的 HTTP error。"""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(code)


async def handle_ingestion_api_error(
    _request: Request,
    exc: IngestionAPIError,
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


def _error(status: int, code: str, message: str) -> IngestionAPIError:
    return IngestionAPIError(status, code, message)


def _context(permission: str) -> ToolInvocationContext:
    return ToolInvocationContext(
        principal_id="ingestion-admin-api",
        principal_type=PrincipalType.SERVICE,
        permissions=(permission,),
        source="ingestion_admin_api",
    )


def _source_error(exc: SourceIngestionError) -> IngestionAPIError:
    code = exc.code
    mappings = {
        "SOURCE_CANDIDATE_NOT_FOUND": (404, "CANDIDATE_NOT_FOUND", "candidate 不存在"),
        "SOURCE_TOOL_UNAVAILABLE": (503, code, "source runtime 当前不可用"),
        "SOURCE_TOOL_NOT_READ_ONLY": (503, code, "source runtime 当前不可用"),
        "SOURCE_EMBEDDINGS_UNAVAILABLE": (503, code, "source runtime 当前不可用"),
        "SOURCE_CONTENT_TOO_LARGE": (422, code, "source 不符合 ingestion 要求"),
        "SOURCE_TYPE_UNSUPPORTED": (422, code, "source 不符合 ingestion 要求"),
        "SOURCE_FETCH_FAILED": (502, code, "external source operation 失败"),
        "SOURCE_PAYLOAD_INVALID": (502, code, "external source operation 失败"),
        "SOURCE_CONTENT_TRUNCATED": (502, code, "external source operation 失败"),
        "SOURCE_RUNTIME_RELOAD_FAILED": (502, code, "external source operation 失败"),
    }
    return _error(*mappings.get(
        code,
        (502, "SOURCE_OPERATION_FAILED", "external source operation 失败"),
    ))


def _candidate(service: IngestionService, candidate_id: str) -> IngestionCandidate:
    try:
        candidate = service.repository.get(candidate_id)
    except CandidatePersistenceError:
        raise _error(
            500, "CANDIDATE_STORAGE_INVALID",
            "candidate persistence validation 失败",
        ) from None
    if candidate is None:
        raise _error(404, "CANDIDATE_NOT_FOUND", "candidate 不存在")
    return candidate


async def _proposal(
    operation: Awaitable[IngestionCandidate],
    service: IngestionService,
    label: str,
) -> IngestionCandidateSummary:
    try:
        return service.summarize(await operation)
    except SourceIngestionError as exc:
        raise _source_error(exc) from None
    except (TypeError, ValueError):
        raise _error(
            422, "SOURCE_VALIDATION_ERROR", f"{label} source request 无效"
        ) from None


def _review(
    operation: Callable[[], IngestionCandidate],
    service: IngestionService,
) -> IngestionCandidateSummary:
    try:
        return service.summarize(operation())
    except InvalidCandidateTransitionError:
        raise _error(
            409, "INVALID_CANDIDATE_TRANSITION",
            "candidate 当前状态不允许该操作",
        ) from None
    except CandidateNotFoundError:
        raise _error(404, "CANDIDATE_NOT_FOUND", "candidate 不存在") from None


def _counts(result: dict[str, int]) -> dict[str, int]:
    counts = {
        name: result.get(f"num_{name}")
        for name in ("added", "updated", "deleted", "skipped")
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts.values()
    ):
        raise ValueError("index result count 无效")
    return counts


def create_ingestion_router(deps: IngestionOperationsDependencies) -> APIRouter:
    """创建只依赖显式 composition object 的 ingestion operations router。"""
    header = APIKeyHeader(
        name=INGESTION_ADMIN_HEADER,
        scheme_name="IngestionAdminKey",
        auto_error=False,
    )

    def require_admin(provided: str | None = Security(header)) -> None:
        configured = secret_value(deps.admin_api_key)
        if (
            provided is None
            or configured is None
            or not secrets.compare_digest(provided, configured)
        ):
            raise _error(
                401, "INGESTION_ADMIN_UNAUTHORIZED",
                "ingestion admin authentication failed",
            )

    router = APIRouter(
        prefix=INGESTION_ADMIN_PREFIX,
        tags=["ingestion-admin"],
        dependencies=[Depends(require_admin)],
    )

    @router.post("/sources/github", response_model=IngestionCandidateSummary, status_code=201)
    async def propose_github(req: GitHubProposalRequest):
        return await _proposal(
            deps.source_service.propose_github_file(
                owner=req.owner,
                repository=req.repository,
                path=req.path,
                ref=req.ref,
                reason=req.reason,
                context=_context("mcp:github"),
            ),
            deps.ingestion_service,
            "GitHub",
        )

    @router.post("/sources/firecrawl", response_model=IngestionCandidateSummary, status_code=201)
    async def propose_firecrawl(req: FirecrawlProposalRequest):
        return await _proposal(
            deps.source_service.propose_firecrawl_page(
                url=req.url,
                reason=req.reason,
                context=_context("mcp:firecrawl"),
            ),
            deps.ingestion_service,
            "Firecrawl",
        )

    @router.get("/candidates", response_model=list[IngestionCandidateSummary])
    def list_candidates(
        status: Annotated[IngestionCandidateStatus | None, Query()] = None,
        source_type: Annotated[KnowledgeSourceType | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ):
        try:
            candidates = deps.ingestion_service.repository.list(
                status=status, source_type=source_type,
                limit=limit, offset=offset,
            )
            return [deps.ingestion_service.summarize(item) for item in candidates]
        except CandidatePersistenceError:
            raise _error(
                500, "CANDIDATE_STORAGE_INVALID",
                "candidate persistence validation 失败",
            ) from None

    @router.get("/candidates/{candidate_id}", response_model=CandidateDetailResponse)
    def candidate_detail(candidate_id: str):
        candidate = _candidate(deps.ingestion_service, candidate_id)
        content = candidate.document.content
        return CandidateDetailResponse(
            summary=deps.ingestion_service.summarize(candidate),
            requester=candidate.requested_by,
            approval=candidate.approval,
            safe_metadata=candidate.document.metadata,
            content_preview=content[:CONTENT_PREVIEW_LIMIT],
            content_length=len(content),
            preview_truncated=len(content) > CONTENT_PREVIEW_LIMIT,
            content_is_untrusted=(
                candidate.document.trust_level is TrustLevel.UNTRUSTED_EXTERNAL
            ),
            latest_index_result=deps.index_runs.latest(candidate_id),
        )

    @router.post("/candidates/{candidate_id}/approve", response_model=IngestionCandidateSummary)
    def approve_candidate(candidate_id: str, req: ApproveRequest):
        return _review(
            lambda: deps.ingestion_service.approve(
                candidate_id, decided_by=deps.requester, reason=req.reason
            ),
            deps.ingestion_service,
        )

    @router.post("/candidates/{candidate_id}/reject", response_model=IngestionCandidateSummary)
    def reject_candidate(candidate_id: str, req: RejectRequest):
        return _review(
            lambda: deps.ingestion_service.reject(
                candidate_id, decided_by=deps.requester, reason=req.reason
            ),
            deps.ingestion_service,
        )

    @router.post("/candidates/{candidate_id}/index", response_model=IngestionIndexRun)
    def index_candidate(candidate_id: str):
        candidate = _candidate(deps.ingestion_service, candidate_id)
        if candidate.status is not IngestionCandidateStatus.APPROVED:
            raise _error(
                409, "CANDIDATE_NOT_APPROVED",
                "只有 approved candidate 可以 index",
            )
        run = deps.index_runs.create_running(candidate_id)
        try:
            return deps.index_runs.succeed(
                run.id,
                **_counts(deps.source_service.index_approved(candidate_id)),
            )
        except SourceIngestionError as exc:
            error = _source_error(exc)
        except Exception:
            error = _error(
                500, "INDEX_OPERATION_FAILED", "index operation 失败"
            )
        try:
            deps.index_runs.fail(
                run.id,
                error_code=error.code,
            )
        except Exception:
            pass
        raise error from None

    @router.get("/index-runs", response_model=list[IngestionIndexRun])
    def list_index_runs(
        candidate_id: Annotated[str | None, Query()] = None,
        status: Annotated[IndexRunStatus | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ):
        return deps.index_runs.list(
            candidate_id=candidate_id,
            status=status,
            limit=limit,
            offset=offset,
        )

    return router


__all__ = ["CONTENT_PREVIEW_LIMIT", "INGESTION_ADMIN_HEADER",
           "INGESTION_ADMIN_PREFIX", "IngestionAPIError",
           "IngestionOperationsDependencies", "create_ingestion_router",
           "handle_ingestion_api_error"]
