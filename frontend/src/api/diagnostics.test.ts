import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import ElementPlus, { ElInputNumber, ElPagination } from 'element-plus'
import { http } from './http'
import * as diagnostics from './diagnostics'
import { useBackendStatus } from '../composables/useBackendStatus'
import SearchView from '../views/SearchView.vue'
import KnowledgeView from '../views/KnowledgeView.vue'
import DiagnosticsView from '../views/DiagnosticsView.vue'

describe('diagnostic response contracts', () => {
  it('keeps the authoritative empty-index diagnostic returned with HTTP 503', async () => {
    const response: diagnostics.SearchResponse = {
      code: 'EMPTY_INDEX', message: '当前知识库为空',
      data: { question: '示例问题', status: 'empty_index', reason: 'RAG 尚未初始化', results: [] },
    }
    vi.spyOn(http, 'post').mockRejectedValue({ isAxiosError: true, response: { status: 503, data: response } })

    expect(await diagnostics.searchKnowledge('示例问题', 5)).toEqual(response)
  })

  it('does not treat an unrelated proxy 503 as a successful diagnostic', async () => {
    const error = { isAxiosError: true, response: { status: 503, data: 'Service Unavailable' } }
    vi.spyOn(http, 'post').mockRejectedValue(error)

    await expect(diagnostics.searchKnowledge('示例问题', 5)).rejects.toBe(error)
  })
})

describe('shared backend status', () => {
  const summary: diagnostics.KnowledgeSummary = {
    ready: true, message: 'OK', doc_count: 2, chunk_count: 12, index_source: 'cache',
    file_names: ['guide.md', 'readme.md'], documents: [],
  }

  it('keeps health online and clears stale KB data when only the KB refresh fails', async () => {
    vi.spyOn(diagnostics, 'getHealth').mockResolvedValue({ status: 'ok' })
    const kbRequest = vi.spyOn(diagnostics, 'getKnowledgeSummary').mockResolvedValueOnce(summary)
    const status = useBackendStatus()
    await status.refreshStatus()
    expect(status.kb.value?.doc_count).toBe(2)

    kbRequest.mockRejectedValueOnce(new Error('KB 暂不可用'))
    await status.refreshStatus()
    expect(status.health.value?.status).toBe('ok')
    expect(status.kb.value).toBeNull()
    expect(status.error.value).toContain('KB 暂不可用')
    expect(status.loading.value).toBe(false)
  })

  it('shares explicit refreshes across pages without issuing duplicate in-flight calls', async () => {
    let finishHealth!: (health: diagnostics.HealthStatus) => void
    const healthRequest = vi.spyOn(diagnostics, 'getHealth').mockReturnValue(new Promise((resolve) => { finishHealth = resolve }))
    const kbRequest = vi.spyOn(diagnostics, 'getKnowledgeSummary').mockResolvedValue(summary)
    const sidebar = useBackendStatus()
    const ingestion = useBackendStatus()
    expect(healthRequest).not.toHaveBeenCalled()

    const sidebarRefresh = sidebar.refreshStatus()
    const ingestionRefresh = ingestion.refreshStatus()
    expect(ingestionRefresh).toBe(sidebarRefresh)
    expect(sidebar.loading.value).toBe(true)
    finishHealth({ status: 'ok' })
    await ingestionRefresh
    expect(healthRequest).toHaveBeenCalledTimes(1)
    expect(kbRequest).toHaveBeenCalledTimes(1)
    expect(ingestion.kb.value?.chunk_count).toBe(12)
  })

  it('refreshes after a pending read when indexing requests fresh state', async () => {
    let finishOldSummary!: (summary: diagnostics.KnowledgeSummary) => void
    const healthRequest = vi.spyOn(diagnostics, 'getHealth').mockResolvedValue({ status: 'ok' })
    const kbRequest = vi.spyOn(diagnostics, 'getKnowledgeSummary')
      .mockReturnValueOnce(new Promise((resolve) => { finishOldSummary = resolve }))
      .mockResolvedValue({ ...summary, doc_count: 3, chunk_count: 20 })
    const status = useBackendStatus()

    const initialRead = status.refreshStatus()
    const afterIndex = status.refreshStatus(true)
    const anotherForcedRefresh = status.refreshStatus(true)
    expect(kbRequest).toHaveBeenCalledTimes(1)
    finishOldSummary(summary)
    await Promise.all([initialRead, afterIndex, anotherForcedRefresh])

    expect(healthRequest).toHaveBeenCalledTimes(2)
    expect(kbRequest).toHaveBeenCalledTimes(2)
    expect(status.kb.value?.doc_count).toBe(3)
    expect(status.kb.value?.chunk_count).toBe(20)
    expect(status.loading.value).toBe(false)
  })
})

describe('diagnostic forms and preview paging', () => {
  let wrapper: VueWrapper | undefined

  beforeEach(() => {
    vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} })
  })
  afterEach(() => {
    wrapper?.unmount()
    wrapper = undefined
    vi.unstubAllGlobals()
  })

  it('rejects a cleared top_k instead of silently requesting the backend default', async () => {
    const request = vi.spyOn(diagnostics, 'searchKnowledge')
    wrapper = mount(SearchView, { global: { plugins: [ElementPlus] } })
    await wrapper.get('textarea').setValue('有效的问题')
    wrapper.getComponent(ElInputNumber).vm.$emit('update:modelValue', undefined)
    await wrapper.get('form').trigger('submit')

    expect(request).not.toHaveBeenCalled()
    expect(wrapper.text()).toContain('召回条数 top_k 必须是 1–10 的整数')
  })

  it('keeps chunk pagination limited to the returned preview and validates cleared bounds', async () => {
    vi.spyOn(diagnostics, 'getDocuments').mockResolvedValue({ count: 0, data: [] })
    const request = vi.spyOn(diagnostics, 'getChunks').mockResolvedValue({
      count: 70,
      data: Array.from({ length: 70 }, (_, index) => ({
        file_name: 'guide.md', chunk_id: index, source_path: 'docs/guide.md',
        position: index, text_length: 600, text_preview: '片段预览',
      })),
    })
    wrapper = mount(KnowledgeView, { global: { plugins: [ElementPlus] } })
    await wrapper.get('#tab-chunks').trigger('click')
    await flushPromises()

    expect(request).toHaveBeenCalledWith({ max_len: 300, limit: 70, file_name: undefined })
    expect(wrapper.text()).toContain('分页仅浏览本次返回内容')
    expect(wrapper.text()).toContain('接口未提供匹配总数')
    expect(wrapper.getComponent(ElPagination).props('total')).toBe(70)
    wrapper.findAllComponents(ElInputNumber)[0]!.vm.$emit('update:modelValue', undefined)
    await wrapper.get('form').trigger('submit')
    expect(request).toHaveBeenCalledTimes(1)
    expect(wrapper.text()).toContain('预览长度 max_len 和返回上限 limit 必须是 1–1000 的整数')
  })

  it('rejects a cleared metrics limit instead of dropping the query parameter', async () => {
    const request = vi.spyOn(diagnostics, 'getMetrics').mockRejectedValue(new Error('暂无指标'))
    wrapper = mount(DiagnosticsView, { global: { plugins: [ElementPlus] } })
    await wrapper.get('#tab-metrics').trigger('click')
    await flushPromises()
    expect(request).toHaveBeenCalledWith(20)
    wrapper.getComponent(ElInputNumber).vm.$emit('update:modelValue', undefined)
    await wrapper.get('.metrics-controls').trigger('submit')

    expect(request).toHaveBeenCalledTimes(1)
    expect(wrapper.text()).toContain('统计请求数 limit 必须是 1–200 的整数')
  })
})
