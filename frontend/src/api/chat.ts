import { fetchEventSource } from '@microsoft/fetch-event-source'
import { http, apiResponseError } from './http'
import type { ChatRequest, ChatResponse } from './types'

interface StreamData {
  start: { session_id_present: boolean }
  progress: { stage: 'query_rewrite' | 'routing' | 'agent' | 'rag' | 'finalizing'; status: 'started' | 'completed' | 'skipped' | 'failed' }
  tool: { tool_name: string; phase: 'started' | 'completed'; status: string; cost_ms?: number }
  answer_delta: { text: string }
  final: { http_status: number; response: ChatResponse; total_cost_ms: number }
  error: { code: string; message: string }
  done: { reason: 'completed' | 'error' }
}

export type ChatStreamEvent = {
  [K in keyof StreamData]: { event: K; trace_id: string; sequence: number; data: StreamData[K] }
}[keyof StreamData]

function isChatResponse(value: unknown): value is ChatResponse {
  if (!value || typeof value !== 'object') return false
  const response = value as Partial<ChatResponse>
  return (typeof response.code === 'string' || typeof response.code === 'number') &&
    typeof response.message === 'string' && typeof response.trace_id === 'string' && !!response.data &&
    typeof response.data.answer === 'string' && typeof response.data.status === 'string' &&
    Array.isArray(response.data.references) && Array.isArray(response.data.tool_trace)
}

export async function sendChat(request: ChatRequest, signal: AbortSignal): Promise<ChatResponse> {
  // A business failure may still carry the authoritative ChatResponse, including trace details.
  const response = await http.post<unknown>('/chat', request, { signal, validateStatus: () => true })
  if (isChatResponse(response.data)) return response.data
  throw apiResponseError(response.status, response.data)
}

export async function streamChat(
  request: ChatRequest,
  signal: AbortSignal,
  onEvent: (event: ChatStreamEvent) => void,
): Promise<void> {
  let sequence = 0
  let traceId = ''
  let terminal: 'final' | 'error' | undefined
  let done = false
  const events = new Set(['start', 'progress', 'tool', 'answer_delta', 'final', 'error', 'done'])

  await fetchEventSource('/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(request),
    signal,
    // Visibility-based reconnects would repeat a POST and can execute tools twice.
    openWhenHidden: true,
    async onopen(response) {
      if (!response.ok) throw apiResponseError(response.status, await response.json().catch(() => undefined))
      if (!response.headers.get('content-type')?.startsWith('text/event-stream')) {
        throw new Error('后端没有返回 text/event-stream。')
      }
    },
    onmessage(message) {
      // Empty messages are keep-alives emitted by the SSE library, not application events.
      if (!message.data && !message.event) return
      if (!events.has(message.event) || done ||
          // The route can fail before the service emits start and send error → done.
          (sequence === 0 && message.event !== 'start' && message.event !== 'error') ||
          (sequence > 0 && message.event === 'start') ||
          (terminal && message.event !== 'done' && !(terminal === 'final' && message.event === 'error'))) {
        throw new Error('聊天流事件顺序无效，请重新发送。')
      }
      let envelope: { schema_version: number; trace_id: string; sequence: number; timestamp: string; data: Record<string, unknown> }
      try {
        envelope = JSON.parse(message.data) as typeof envelope
      } catch {
        throw new Error('聊天流返回了无效的 JSON。')
      }
      if (!envelope || envelope.schema_version !== 1 || envelope.sequence !== sequence + 1 ||
          message.id !== String(envelope.sequence) || typeof envelope.trace_id !== 'string' || !envelope.trace_id ||
          (traceId && traceId !== envelope.trace_id) || typeof envelope.timestamp !== 'string' || !envelope.timestamp ||
          !envelope.data || typeof envelope.data !== 'object' || Array.isArray(envelope.data)) {
        throw new Error('聊天流信封无效或事件缺失，请重新发送。')
      }
      if (message.event === 'answer_delta' && typeof envelope.data.text !== 'string') {
        throw new Error('聊天流增量文本无效。')
      }
      if (message.event === 'final') {
        if (!isChatResponse(envelope.data.response) || !Number.isInteger(envelope.data.http_status)) {
          throw new Error('聊天流缺少完整的最终响应。')
        }
        terminal = 'final'
      } else if (message.event === 'error') {
        if (typeof envelope.data.code !== 'string' || typeof envelope.data.message !== 'string') {
          throw new Error('聊天流错误响应无效。')
        }
        terminal = 'error'
      } else if (message.event === 'done') {
        if (!((envelope.data.reason === 'completed' && terminal === 'final') ||
              (envelope.data.reason === 'error' && terminal === 'error'))) {
          throw new Error('聊天流结束前缺少 final 或 error。')
        }
        done = true
      }
      sequence = envelope.sequence
      traceId = envelope.trace_id
      onEvent({ event: message.event, trace_id: traceId, sequence, data: envelope.data } as ChatStreamEvent)
    },
    onclose() {
      if (!done) throw new Error('聊天流意外中断，未收到 done。已收到的内容仅供参考，请重新发送。')
    },
    onerror(error: unknown) {
      // Throwing disables fetch-event-source's automatic retry for this side-effecting POST.
      throw error
    },
  })
}
