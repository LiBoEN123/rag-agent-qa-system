import { onScopeDispose, reactive, readonly, ref, watch } from 'vue'
import { streamChat, type ChatStreamEvent } from '../api/chat'
import { apiErrorMessage } from '../api/http'
import type { ChatResponse, ToolMode } from '../api/types'
import { useSession } from './useSession'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  state: 'pending' | 'completed' | 'error' | 'cancelled'
  steps: string[]
  response?: ChatResponse
  traceId?: string
  error?: string
  httpStatus?: number
  totalCostMs?: number
}

const stages = { query_rewrite: '整理多轮问题', routing: '分析问题处理路径', agent: '执行 Agent', rag: '整理知识库证据', finalizing: '生成最终响应' }
const statuses = { started: '进行中', completed: '完成', skipped: '跳过', failed: '失败' }
const { sessionId } = useSession()
// Keep this small conversation buffer across navigation; only the session ID is persisted.
const messages = ref<ChatMessage[]>([])
const sending = ref(false)
let active: { controller: AbortController; message: ChatMessage } | undefined

function cancel(): void {
  if (!active) return
  active.controller.abort()
  if (active.message.state === 'pending') {
    active.message.state = 'cancelled'
    active.message.error = '已停止生成；当前内容可能不完整。'
    active.message.steps.push('已停止')
  }
  active = undefined
  sending.value = false
}

watch(sessionId, () => { cancel(); messages.value = [] }, { flush: 'sync' })

function applyFinal(message: ChatMessage, response: ChatResponse): void {
  message.response = response
  message.traceId = response.trace_id
  message.content = response.data.answer || response.data.clarification_question || response.message || response.data.reason
}

function applyEvent(message: ChatMessage, event: ChatStreamEvent): void {
  message.traceId = event.trace_id
  switch (event.event) {
    case 'start': message.steps.push('请求已开始'); break
    case 'progress': message.steps.push(`${stages[event.data.stage]}：${statuses[event.data.status]}`); break
    case 'tool':
      message.steps.push(`调用 ${event.data.tool_name} · phase=${event.data.phase} · status=${event.data.status || '—'} · cost_ms=${event.data.cost_ms ?? '—'}`)
      break
    case 'answer_delta': message.content += event.data.text; break
    case 'final':
      applyFinal(message, event.data.response)
      message.httpStatus = event.data.http_status
      message.totalCostMs = event.data.total_cost_ms
      message.steps.push('已收到最终响应')
      break
    case 'error':
      message.state = 'error'
      message.error = `${event.data.message}（${event.data.code}）`
      message.steps.push('流式处理失败')
      break
    case 'done':
      if (event.data.reason === 'completed') {
        message.state = 'completed'
        message.steps.push('完成')
      }
      break
  }
}

export function useChat() {
  onScopeDispose(cancel)

  async function send(question: string, topK: number, decomposeTopK: number, toolMode: ToolMode = 'rag_only'): Promise<void> {
    if (!question.trim() || sending.value) return
    const controller = new AbortController()
    const message = reactive<ChatMessage>({ id: crypto.randomUUID(), role: 'assistant', content: '', state: 'pending', steps: [] })
    messages.value.push({ id: crypto.randomUUID(), role: 'user', content: question.trim(), state: 'completed', steps: [] }, message)
    active = { controller, message }
    sending.value = true
    const request = { question: question.trim(), top_k: topK, decompose_top_k: decomposeTopK, session_id: sessionId.value, tool_mode: toolMode }
    try {
      await streamChat(request, controller.signal, (event) => {
        if (!controller.signal.aborted) applyEvent(message, event)
      })
    } catch (error: unknown) {
      if (!controller.signal.aborted) {
        message.state = 'error'
        message.error ||= apiErrorMessage(error)
      }
    } finally {
      if (active?.controller === controller) {
        active = undefined
        sending.value = false
      }
    }
  }

  return { messages: readonly(messages), sending: readonly(sending), send, cancel }
}
