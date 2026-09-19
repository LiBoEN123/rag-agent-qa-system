import { isAxiosError } from 'axios'
import { http } from './http'
import type { ChatReference, PublicStatus } from './types'

interface ResponseEnvelope<T> {
  code: number | string
  message: string
  data: T
}

export interface HealthStatus {
  status: string
}

export interface DocumentPreview {
  doc_id: string | number
  file_name: string
  file_type: string
  chunk_count: number
  source_path: string
}

export interface ChunkPreview {
  file_name: string | null
  chunk_id: string | number | null
  position: number | null
  text_length: number
  text_preview: string
  source_path: string | null
}

export interface PreviewResponse<T> {
  count: number
  data: T[]
}

export interface KnowledgeSummary {
  ready: boolean
  message: string
  doc_count: number
  chunk_count: number
  index_source: string
  file_names: string[]
  documents: DocumentPreview[]
}

export interface SearchResult extends ChatReference {
  bm25_score: number
  vector_score: number
}

export type SearchResponse = ResponseEnvelope<{
  question: string
  status: PublicStatus
  reason: string
  results: SearchResult[]
}>

export interface RouteDecision {
  decision: 'use_agent' | 'direct_answer' | 'clarify'
  action: 'agent' | 'direct_answer' | 'clarify'
  reason: string
  clarification_question?: string
  router_source: 'rule_based' | 'rule_based_guardrail'
  router_confidence: number
}

export interface DecisionResult {
  question: string
  decision: RouteDecision | 'clarify'
  reason?: string
}

export interface MetricsSummary {
  limit: number
  total: number
  status_count: Record<string, number>
  success_count: number
  no_relevant_info_count: number
  tool_error_count: number
  model_error_count: number
  clarify_count: number
  internal_error_count: number
  avg_latency_ms: number
  avg_llm_call_count: number
  router_llm_used_count: number
  answer_llm_used_count: number
}

export async function getHealth(): Promise<HealthStatus> {
  return (await http.get<ResponseEnvelope<HealthStatus>>('/health', { timeout: 5000 })).data.data
}

export async function getKnowledgeSummary(): Promise<KnowledgeSummary> {
  return (await http.post<ResponseEnvelope<KnowledgeSummary>>('/kb_summary', {}, { timeout: 20000 })).data.data
}

export async function searchKnowledge(question: string, top_k: number): Promise<SearchResponse> {
  try {
    return (await http.post<SearchResponse>('/search', { question, top_k })).data
  } catch (error) {
    // Empty indexes are a structured diagnostic result even though HTTP is 503.
    if (isAxiosError<SearchResponse>(error) && error.response?.status === 503) {
      const body = error.response.data
      if (body?.code === 'EMPTY_INDEX' && body.data?.status === 'empty_index' && Array.isArray(body.data.results)) return body
    }
    throw error
  }
}

export async function getDocuments(): Promise<PreviewResponse<DocumentPreview>> {
  return (await http.get<PreviewResponse<DocumentPreview>>('/docs-preview')).data
}

export async function getChunks(params: {
  max_len: number
  limit: number
  file_name?: string
}): Promise<PreviewResponse<ChunkPreview>> {
  return (await http.get<PreviewResponse<ChunkPreview>>('/chunks-preview', { params })).data
}

export async function decideRoute(question: string): Promise<DecisionResult> {
  return (await http.post<ResponseEnvelope<DecisionResult>>('/decide', { question })).data.data
}

export async function getMetrics(limit: number): Promise<MetricsSummary> {
  return (await http.get<ResponseEnvelope<MetricsSummary>>('/metrics-summary', { params: { limit } })).data.data
}
