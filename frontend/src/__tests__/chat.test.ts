import { defineComponent, nextTick } from 'vue'
import { flushPromises, mount } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { streamChat, type ChatStreamEvent } from '../api/chat'
import type { ChatResponse } from '../api/types'
import { useChat } from '../composables/useChat'
import { useSession } from '../composables/useSession'
import ChatView from '../views/ChatView.vue'

vi.mock('../api/chat', () => ({ streamChat: vi.fn() }))

const finalResponse: ChatResponse = {
  code: 'NEED_CLARIFICATION', message: '请补充问题', trace_id: 'trace-1',
  data: {
    question: '测试问题', answer: '', clarification_question: '请问哪一年？', status: 'clarify',
    references: [{ reference_id: 'ref-1', file_name: '说明.md', source_path: '/docs/说明.md', chunk_id: 7, snippet: '公开引用', score: 0.8, rerank_score: 0.7 }],
    tool_trace: [{ step: 'tool_call', tool_name: 'github_get_file_contents', tool_status: 'success', cost_ms: 12 }],
    retrieval_confidence: 0.8, answer_confidence: 0.4, action: 'clarify', reason: '需要年份',
    need_clarification: true, tool_result_summary: '已读取来源', llm_call_count: 2,
    router_llm_used: true, answer_llm_used: true,
  },
}

const Harness = defineComponent({
  setup: useChat,
  template: '<div><article v-for="message in messages" :key="message.id" :data-state="message.state">{{ message.content }}<span>{{ message.error }}</span></article></div>',
})

let callback: (event: ChatStreamEvent) => void
let finish: () => void
let rejectStream: (error: Error) => void
let signal: AbortSignal
let sequence: number
const wrappers: ReturnType<typeof mount>[] = []

function event<E extends ChatStreamEvent['event']>(name: E, data: Extract<ChatStreamEvent, { event: E }>['data']): void {
  callback({ event: name, data, trace_id: 'trace-1', sequence: ++sequence } as ChatStreamEvent)
}

beforeEach(() => {
  useSession().newSession()
  sequence = 0
  vi.mocked(streamChat).mockImplementation((_request, abortSignal, onEvent) => {
    callback = onEvent
    signal = abortSignal
    return new Promise<void>((resolve, reject) => { finish = resolve; rejectStream = reject })
  })
})
afterEach(() => {
  wrappers.splice(0).forEach((wrapper) => wrapper.unmount())
  vi.clearAllMocks()
})

function mountChat() {
  const wrapper = mount(Harness)
  wrappers.push(wrapper)
  return wrapper
}

describe('chat conversation', () => {
  it('renders deltas immediately, then takes every final field from the authoritative response', async () => {
    const wrapper = mountChat()
    const sending = wrapper.vm.send(' 测试问题 ')
    event('start', { session_id_present: true })
    event('progress', { stage: 'routing', status: 'completed' })
    event('tool', { tool_name: 'github_get_file_contents', phase: 'completed', status: 'success', cost_ms: 12 })
    event('answer_delta', { text: '临时' })
    await nextTick()
    expect(wrapper.findAll('article')[1]?.text()).toContain('临时')
    event('answer_delta', { text: '答案' })
    await nextTick()
    expect(wrapper.findAll('article')[1]?.text()).toContain('临时答案')
    event('final', { http_status: 200, response: finalResponse, total_cost_ms: 321 })
    await nextTick()
    expect(wrapper.findAll('article')[1]?.text()).toContain('请问哪一年？')
    expect(wrapper.text()).not.toContain('临时答案')
    expect(wrapper.vm.messages[1]?.response).toEqual(finalResponse)
    expect(wrapper.vm.messages[1]?.totalCostMs).toBe(321)
    expect(wrapper.vm.messages[1]?.steps.join(' ')).toContain('cost_ms=12')
    event('done', { reason: 'completed' })
    finish()
    await sending
    expect(wrapper.vm.sending).toBe(false)
    expect(wrapper.vm.messages[1]?.state).toBe('completed')
    expect(streamChat).toHaveBeenCalledWith({ question: '测试问题', session_id: useSession().sessionId.value, tool_mode: 'rag_only' }, expect.any(AbortSignal), expect.any(Function))
  })

  it('defaults the view to rag_only, disables the switch while sending, and sends full when enabled', async () => {
    const wrapper = mount(ChatView, { global: { plugins: [ElementPlus] } })
    wrappers.push(wrapper)
    const toggle = wrapper.get('input[role="switch"]')
    expect(toggle.attributes('aria-checked')).toBe('false')
    expect(wrapper.text()).toContain('扩展工具')
    await wrapper.get('textarea').setValue('默认知识库问题')
    await wrapper.get('form.composer').trigger('submit')
    expect(vi.mocked(streamChat).mock.calls[0]?.[0].tool_mode).toBe('rag_only')
    expect(toggle.attributes('disabled')).toBeDefined()
    finish()
    await flushPromises()
    expect(toggle.attributes('disabled')).toBeUndefined()
    await toggle.setValue(true)
    expect(toggle.attributes('aria-checked')).toBe('true')
    await wrapper.get('textarea').setValue('使用扩展工具')
    await wrapper.get('form.composer').trigger('submit')
    expect(vi.mocked(streamChat).mock.calls[1]?.[0].tool_mode).toBe('full')
    expect(toggle.attributes('disabled')).toBeDefined()
    finish()
    await flushPromises()
  })

  it('shows public stream errors and does not treat done(error) as success', async () => {
    const wrapper = mountChat()
    const sending = wrapper.vm.send('测试问题')
    event('start', { session_id_present: true })
    event('answer_delta', { text: '未完成的文本' })
    event('error', { code: 'STREAM_ERROR', message: '流式处理失败' })
    event('done', { reason: 'error' })
    finish()
    await sending
    expect(wrapper.vm.messages[1]?.state).toBe('error')
    expect(wrapper.text()).toContain('STREAM_ERROR')
    expect(wrapper.vm.messages[1]?.response).toBeUndefined()
  })

  it('preserves received text with an interruption warning, without manufacturing a final response', async () => {
    const wrapper = mountChat()
    const sending = wrapper.vm.send('测试问题')
    event('answer_delta', { text: '部分回答' })
    rejectStream(new Error('聊天流意外中断，未收到 done。'))
    await sending
    expect(wrapper.text()).toContain('部分回答')
    expect(wrapper.text()).toContain('未收到 done')
    expect(wrapper.vm.messages[1]?.state).toBe('error')
    expect(wrapper.vm.sending).toBe(false)
    expect(streamChat).toHaveBeenCalledOnce()
  })

  it('keeps the authoritative final details when the stream closes before done', async () => {
    const wrapper = mountChat()
    const sending = wrapper.vm.send('测试问题')
    event('final', { http_status: 200, response: finalResponse, total_cost_ms: 321 })
    rejectStream(new Error('聊天流意外中断，未收到 done。'))
    await sending
    expect(wrapper.vm.messages[1]?.response).toEqual(finalResponse)
    expect(wrapper.vm.messages[1]?.content).toBe('请问哪一年？')
    expect(wrapper.vm.messages[1]?.state).toBe('error')
    expect(wrapper.text()).toContain('未收到 done')
  })

  it('aborts on cancel, ignores late callbacks, and preserves an explicitly incomplete answer', async () => {
    const wrapper = mountChat()
    const sending = wrapper.vm.send('测试问题')
    event('answer_delta', { text: '部分回答' })
    wrapper.vm.cancel()
    expect(signal.aborted).toBe(true)
    event('final', { http_status: 200, response: finalResponse, total_cost_ms: 1 })
    finish()
    await sending
    expect(wrapper.vm.messages[1]?.content).toBe('部分回答')
    expect(wrapper.vm.messages[1]?.state).toBe('cancelled')
    expect(wrapper.vm.messages[1]?.response).toBeUndefined()
    expect(wrapper.vm.sending).toBe(false)
  })

  it('creates a new stored session and clears messages while aborting the previous stream', async () => {
    const wrapper = mountChat()
    const oldSession = useSession().sessionId.value
    const sending = wrapper.vm.send('测试问题')
    useSession().newSession()
    expect(signal.aborted).toBe(true)
    expect(wrapper.vm.messages).toHaveLength(0)
    expect(useSession().sessionId.value).toMatch(/^web_[0-9a-f-]{36}$/)
    expect(useSession().sessionId.value).not.toBe(oldSession)
    expect(sessionStorage.getItem('rag-agent-session-id')).toBe(useSession().sessionId.value)
    event('answer_delta', { text: '旧对话迟到事件' })
    finish()
    await sending
    expect(wrapper.vm.messages).toHaveLength(0)
  })

  it('aborts safely on navigation and keeps the conversation when the chat view is mounted again', async () => {
    const wrapper = mountChat()
    const sending = wrapper.vm.send('测试问题')
    event('answer_delta', { text: '部分回答' })
    wrapper.unmount()
    expect(signal.aborted).toBe(true)
    finish()
    await sending
    const next = mountChat()
    expect(next.text()).toContain('部分回答')
    expect(next.vm.messages[1]?.state).toBe('cancelled')
  })

  it('ignores empty questions without opening a stream', async () => {
    const wrapper = mountChat()
    await wrapper.vm.send('   ')
    expect(streamChat).not.toHaveBeenCalled()
    expect(wrapper.vm.messages).toHaveLength(0)
  })
})
