import { afterEach, describe, expect, it, vi } from 'vitest'
import { streamChat, type ChatStreamEvent } from '../api/chat'
import type { ChatResponse } from '../api/types'

const start = { event: 'start', data: { session_id_present: true } }
const chatResponse: ChatResponse = {
  code: 0, message: 'success', trace_id: 'trace-api',
  data: {
    question: '测试问题', answer: '最终结果', status: 'ok', references: [], tool_trace: [],
    clarification_question: '', reason: '回答完成', action: 'direct_answer', need_clarification: false,
    retrieval_confidence: 0, answer_confidence: 1, tool_result_summary: '', llm_call_count: 0,
    router_llm_used: false, decompose_llm_used: false, answer_llm_used: false,
  },
}
const final = {
  event: 'final', data: {
    http_status: 200, total_cost_ms: 3, response: chatResponse,
  },
}

function response(events: { event: string; data: object }[]): Response {
  const body = events.map((event, index) => `id: ${index + 1}\nevent: ${event.event}\ndata: ${JSON.stringify({ schema_version: 1, trace_id: 'trace-api', sequence: index + 1, timestamp: '2026-09-11T12:00:00', data: event.data })}\n\n`).join('')
  return new Response(new TextEncoder().encode(body), { headers: { 'Content-Type': 'text/event-stream' } })
}

afterEach(() => vi.unstubAllGlobals())

describe('POST chat SSE transport', () => {
  it('uses the SSE library for all seven public events and sends no admin credential', async () => {
    const received: ChatStreamEvent[] = []
    const fetch = vi.fn().mockResolvedValue(response([
      start, { event: 'progress', data: { stage: 'agent', status: 'started' } },
      { event: 'tool', data: { tool_name: 'web_fetch', phase: 'completed', status: 'success', cost_ms: 5 } },
      { event: 'answer_delta', data: { text: '临时结果' } }, final, { event: 'done', data: { reason: 'completed' } },
    ]))
    vi.stubGlobal('fetch', fetch)
    await streamChat({ question: '测试问题', session_id: 'web_test' }, new AbortController().signal, (event) => received.push(event))
    expect(received.map((event) => event.event)).toEqual(['start', 'progress', 'tool', 'answer_delta', 'final', 'done'])
    expect(fetch).toHaveBeenCalledOnce()
    expect(fetch.mock.calls[0]?.[0]).toBe('/chat/stream')
    const options = fetch.mock.calls[0]?.[1] as RequestInit
    expect(options.method).toBe('POST')
    expect(new Headers(options.headers).get('X-Ingestion-Admin-Key')).toBeNull()
    expect(JSON.parse(String(options.body))).toEqual({ question: '测试问题', session_id: 'web_test' })

    fetch.mockResolvedValue(response([start, { event: 'error', data: { code: 'STREAM_ERROR', message: '流式处理失败' } }, { event: 'done', data: { reason: 'error' } }]))
    await streamChat({ question: '失败' }, new AbortController().signal, (event) => received.push(event))
    expect(received.at(-2)?.event).toBe('error')
  })

  it('rejects a prematurely closed stream without retrying the POST', async () => {
    const fetch = vi.fn().mockResolvedValue(response([start, { event: 'answer_delta', data: { text: '部分回答' } }]))
    vi.stubGlobal('fetch', fetch)
    await expect(streamChat({ question: '测试问题' }, new AbortController().signal, vi.fn())).rejects.toThrow('未收到 done')
    expect(fetch).toHaveBeenCalledOnce()
  })

  it('preserves the public error when the backend fails before start or after final', async () => {
    const error = { event: 'error', data: { code: 'STREAM_ERROR', message: '流式处理失败' } }
    const done = { event: 'done', data: { reason: 'error' } }
    const received: ChatStreamEvent[] = []
    const fetch = vi.fn().mockResolvedValueOnce(response([error, done]))
      .mockResolvedValueOnce(response([start, final, error, done]))
    vi.stubGlobal('fetch', fetch)
    await streamChat({ question: '测试问题' }, new AbortController().signal, (event) => received.push(event))
    expect(received.map((event) => event.event)).toEqual(['error', 'done'])
    received.length = 0
    await streamChat({ question: '测试问题' }, new AbortController().signal, (event) => received.push(event))
    expect(received.map((event) => event.event)).toEqual(['start', 'final', 'error', 'done'])
    expect(fetch).toHaveBeenCalledTimes(2)
  })

  it('rejects a malformed final instead of passing unusable detail arrays into the view', async () => {
    const fetch = vi.fn().mockResolvedValue(response([start, {
      event: 'final', data: { ...final.data, response: { ...chatResponse, data: { answer: '不完整响应', status: 'ok' } } },
    }]))
    vi.stubGlobal('fetch', fetch)
    await expect(streamChat({ question: '测试问题' }, new AbortController().signal, vi.fn())).rejects.toThrow('完整的最终响应')
    expect(fetch).toHaveBeenCalledOnce()
  })

  it('requires a final response before completed done', async () => {
    const fetch = vi.fn().mockResolvedValue(response([start, { event: 'done', data: { reason: 'completed' } }]))
    vi.stubGlobal('fetch', fetch)
    await expect(streamChat({ question: '测试问题' }, new AbortController().signal, vi.fn())).rejects.toThrow('缺少 final 或 error')
    expect(fetch).toHaveBeenCalledOnce()
  })

  it('keeps public HTTP errors and disables network retries', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ code: 'REQUEST_VALIDATION_ERROR', message: '请求格式无效' }), { status: 422, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetch)
    await expect(streamChat({ question: '' }, new AbortController().signal, vi.fn())).rejects.toThrow('请求格式无效')
    expect(fetch).toHaveBeenCalledOnce()
    fetch.mockRejectedValue(new TypeError('Failed to fetch'))
    await expect(streamChat({ question: '测试问题' }, new AbortController().signal, vi.fn())).rejects.toThrow('Failed to fetch')
    expect(fetch).toHaveBeenCalledTimes(2)
  })

  it('consumes an empty question as HTTP 200 with final business status 400', async () => {
    const emptyFinal = { event: 'final', data: {
      ...final.data, http_status: 400,
      response: { ...chatResponse, code: 'EMPTY_QUESTION', message: '问题不能为空',
        data: { ...chatResponse.data, answer: '', status: 'clarify', need_clarification: true, clarification_question: '请输入问题' } },
    } }
    const fetch = vi.fn().mockResolvedValue(response([start, emptyFinal, { event: 'done', data: { reason: 'completed' } }]))
    vi.stubGlobal('fetch', fetch)
    const received: ChatStreamEvent[] = []
    await streamChat({ question: '   ' }, new AbortController().signal, (event) => received.push(event))
    expect(received.map((event) => event.event)).toEqual(['start', 'final', 'done'])
    expect(received[1]?.data).toEqual(emptyFinal.data)
    expect(fetch).toHaveBeenCalledOnce()
  })

  it.each([
    ['"sequence":2', '"sequence":3'],
    ['"trace_id":"trace-api","sequence":2', '"trace_id":"other-trace","sequence":2'],
  ])('rejects missing sequence or changed trace without retrying (%s)', async (before, after) => {
    const body = await response([start, final, { event: 'done', data: { reason: 'completed' } }]).text()
    const fetch = vi.fn().mockResolvedValue(new Response(body.replace(before, after), { headers: { 'Content-Type': 'text/event-stream' } }))
    vi.stubGlobal('fetch', fetch)
    await expect(streamChat({ question: '测试问题' }, new AbortController().signal, vi.fn())).rejects.toThrow('信封无效或事件缺失')
    expect(fetch).toHaveBeenCalledOnce()
  })

})
