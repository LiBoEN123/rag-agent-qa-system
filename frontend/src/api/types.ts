// Minimal public HTTP contracts, verified against app/api/schemas.py,
// app/api/ingestion.py, app/ingestion/contracts.py and app/storage/ingestion.py.
// Diagnostics endpoints without Pydantic response models live in diagnostics.ts.
export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue }
export type PublicStatus = 'ok' | 'no_evidence' | 'empty_index' | 'error' | 'clarify' | 'out_of_scope'

export interface ChatRequest {
  question: string
  top_k?: number | null
  decompose_top_k?: number | null
  session_id?: string | null
}
export interface ChatReference {
  reference_id: string
  file_name: string
  source_path: string
  chunk_id: string | number
  position?: number | null
  snippet: string
  score: number
  rerank_score: number
}
export interface ChatData {
  question: string
  answer: string
  references: ChatReference[]
  retrieval_confidence: number
  answer_confidence: number
  need_clarification: boolean
  status: PublicStatus
  clarification_question: string
  reason: string
  action: string
  tool_result_summary: string
  tool_trace: Record<string, unknown>[]
  router_llm_used: boolean
  decompose_llm_used: boolean
  answer_llm_used: boolean
  llm_call_count: number
}
export interface ChatResponse {
  code: number | string
  message: string
  trace_id: string
  data: ChatData
}

export type IngestionCandidateStatus = 'pending' | 'approved' | 'rejected' | 'superseded'
export type KnowledgeSourceType = 'local_file' | 'upload' | 'github' | 'web' | 'firecrawl'
export type ApprovalStatus = 'not_required' | 'pending' | 'approved' | 'rejected'
export interface IngestionCandidateSummary {
  candidate_id: string
  title: string
  source_type: KnowledgeSourceType
  source_uri: string
  content_hash: string
  content_length: number
  trust_level: 'trusted_internal' | 'trusted_external' | 'untrusted_external'
  policy_decision: 'allow' | 'require_approval' | 'deny'
  policy_reasons: string[]
  status: IngestionCandidateStatus
  approval_status: ApprovalStatus
  requested_at: string
}
export interface IngestionRequester {
  principal_id: string
  principal_type: 'user' | 'service' | 'system'
  source: string
}
export interface IngestionApproval {
  status: ApprovalStatus
  decided_at: string | null
  decided_by: IngestionRequester | null
  reason: string | null
}
export interface IngestionIndexRun {
  id: number
  candidate_id: string
  operation: 'index' | 'remove'
  status: 'running' | 'success' | 'failed'
  started_at: string
  completed_at: string | null
  added: number | null
  updated: number | null
  deleted: number | null
  skipped: number | null
  error_code: string | null
  error_message: string | null
}
export interface CandidateDetailResponse {
  summary: IngestionCandidateSummary
  requester: IngestionRequester
  approval: IngestionApproval
  safe_metadata: Record<string, JsonValue>
  content_preview: string
  content_length: number
  preview_truncated: boolean
  content_is_untrusted: boolean
  latest_index_result: IngestionIndexRun | null
}
export interface GitHubProposalRequest {
  owner: string
  repository: string
  path: string
  ref?: string | null
  reason?: string | null
}
export interface FirecrawlProposalRequest {
  url: string
  reason?: string | null
}
export interface UploadProposalRequest {
  file: File
  reason?: string | null
}
