import { ingestionHttp } from './http'
import type {
  CandidateDetailResponse, FirecrawlProposalRequest, GitHubProposalRequest,
  IngestionCandidateStatus, IngestionCandidateSummary, IngestionIndexRun, KnowledgeSourceType,
  UploadProposalRequest,
} from './types'

const base = '/admin/ingestion'
const candidatePath = (id: string) => `${base}/candidates/${encodeURIComponent(id)}`

export interface CandidateQuery {
  status?: IngestionCandidateStatus
  source_type?: KnowledgeSourceType
  limit: number
  offset: number
}

export interface IndexRunQuery {
  candidate_id?: string
  status?: IngestionIndexRun['status']
  operation?: IngestionIndexRun['operation']
  limit: number
  offset: number
}

export async function listCandidates(params: CandidateQuery) {
  return (await ingestionHttp.get<IngestionCandidateSummary[]>(`${base}/candidates`, { params })).data
}

export async function getCandidate(id: string) {
  return (await ingestionHttp.get<CandidateDetailResponse>(candidatePath(id))).data
}

export async function approveCandidate(id: string, reason: string | null) {
  return (await ingestionHttp.post<IngestionCandidateSummary>(`${candidatePath(id)}/approve`, { reason })).data
}

export async function rejectCandidate(id: string, reason: string) {
  return (await ingestionHttp.post<IngestionCandidateSummary>(`${candidatePath(id)}/reject`, { reason })).data
}

export async function indexCandidate(id: string) {
  return (await ingestionHttp.post<IngestionIndexRun>(`${candidatePath(id)}/index`)).data
}

export async function removeCandidate(id: string) {
  return (await ingestionHttp.post<IngestionIndexRun>(`${candidatePath(id)}/remove`)).data
}

export async function deleteCandidate(id: string) {
  await ingestionHttp.delete(candidatePath(id))
}

export async function listIndexRuns(params: IndexRunQuery) {
  return (await ingestionHttp.get<IngestionIndexRun[]>(`${base}/index-runs`, { params })).data
}

export async function proposeGitHub(request: GitHubProposalRequest) {
  return (await ingestionHttp.post<IngestionCandidateSummary>(`${base}/sources/github`, request)).data
}

export async function proposeFirecrawl(request: FirecrawlProposalRequest) {
  return (await ingestionHttp.post<IngestionCandidateSummary>(`${base}/sources/firecrawl`, request)).data
}

export async function proposeUpload(request: UploadProposalRequest) {
  const form = new FormData()
  form.append('file', request.file)
  if (request.reason) form.append('reason', request.reason)
  return (await ingestionHttp.post<IngestionCandidateSummary>(`${base}/sources/upload`, form)).data
}
