import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'
import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import ElementPlus, { ElSelect } from 'element-plus'
import * as api from '../api/ingestion'
import type { IngestionCandidateSummary } from '../api/types'
import IngestionCandidates from '../views/ingestion/IngestionCandidates.vue'
import IngestionIndexRuns from '../views/ingestion/IngestionIndexRuns.vue'
import { candidateDetail, indexRun } from './ingestion-fixtures'

vi.mock('../api/ingestion', () => ({
  listCandidates: vi.fn(), getCandidate: vi.fn(), approveCandidate: vi.fn(), rejectCandidate: vi.fn(),
  indexCandidate: vi.fn(), removeCandidate: vi.fn(), listIndexRuns: vi.fn(),
}))

let wrapper: VueWrapper | undefined
beforeEach(() => {
  vi.resetAllMocks()
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} })
  vi.mocked(api.listCandidates).mockImplementation(async ({ limit }) => Array.from({ length: limit }, () => candidateDetail().summary))
  vi.mocked(api.listIndexRuns).mockImplementation(async ({ limit }) => Array.from({ length: limit }, () => indexRun()))
})
afterEach(() => { wrapper?.unmount(); vi.unstubAllGlobals() })

describe('入库列表筛选与分页', () => {
  it('来源和状态从第一页重新查询，加载期间固定页大小且忽略被新刷新取代的结果', async () => {
    const page = mount(IngestionCandidates, { global: { plugins: [ElementPlus] } })
    wrapper = page
    await page.vm.refresh(true)
    await flushPromises()
    const next = () => page.findAll('.paging button')[1]!
    await next().trigger('click')
    await flushPromises()
    expect(api.listCandidates).toHaveBeenLastCalledWith({ status: 'pending', source_type: undefined, limit: 20, offset: 20 })
    const [status, source, size] = page.findAllComponents(ElSelect)
    source!.vm.$emit('update:modelValue', 'github')
    await nextTick()
    expect(page.get('.paging').text()).toContain('第 1 页')
    expect(next().attributes('disabled')).toBeDefined()
    source!.vm.$emit('change', 'github')
    await flushPromises()
    await next().trigger('click')
    await flushPromises()
    status!.vm.$emit('update:modelValue', 'approved')
    status!.vm.$emit('change', 'approved')
    await flushPromises()
    expect(api.listCandidates).toHaveBeenLastCalledWith({ status: 'approved', source_type: 'github', limit: 20, offset: 0 })

    let completeOld!: (value: IngestionCandidateSummary[]) => void
    vi.mocked(api.listCandidates).mockReturnValueOnce(new Promise((resolve) => { completeOld = resolve }))
    await next().trigger('click')
    expect(size!.props('disabled')).toBe(true)
    expect(page.get('.paging').text()).toContain('第 2 页')
    size!.vm.$emit('update:modelValue', 50)
    size!.vm.$emit('change', 50)
    await flushPromises()
    expect(page.get('.paging').text()).toContain('第 1 页 · 本页 50 条')
    expect(api.listCandidates).toHaveBeenLastCalledWith({ status: 'approved', source_type: 'github', limit: 50, offset: 0 })
    completeOld([])
    await flushPromises()
    expect(page.get('.paging').text()).toContain('本页 50 条')
  })

  it('索引记录候选 ID 和状态改变即重置页码，查询仅携带真实筛选字段', async () => {
    const page = mount(IngestionIndexRuns, { global: { plugins: [ElementPlus] } })
    wrapper = page
    await page.vm.refresh(true)
    await flushPromises()
    const next = () => page.findAll('.paging button')[1]!
    await next().trigger('click')
    await flushPromises()
    const candidateInput = page.get('input[aria-label="按候选 ID 筛选索引记录"]')
    ;(candidateInput.element as HTMLInputElement).value = 'cand_other'
    await candidateInput.trigger('input')
    expect(page.get('.paging').text()).toContain('第 1 页')
    expect(next().attributes('disabled')).toBeDefined()
    await candidateInput.trigger('change')
    await flushPromises()
    expect(api.listIndexRuns).toHaveBeenLastCalledWith({ candidate_id: 'cand_other', status: undefined, operation: undefined, limit: 20, offset: 0 })
    await next().trigger('click')
    await flushPromises()
    const status = page.findAllComponents(ElSelect)[0]!
    status.vm.$emit('update:modelValue', 'failed')
    status.vm.$emit('change', 'failed')
    await flushPromises()
    expect(api.listIndexRuns).toHaveBeenLastCalledWith({ candidate_id: 'cand_other', status: 'failed', operation: undefined, limit: 20, offset: 0 })
    await next().trigger('click')
    await flushPromises()
    const operation = page.findAllComponents(ElSelect)[2]!
    operation.vm.$emit('update:modelValue', 'remove')
    operation.vm.$emit('change', 'remove')
    await flushPromises()
    expect(api.listIndexRuns).toHaveBeenLastCalledWith({ candidate_id: 'cand_other', status: 'failed', operation: 'remove', limit: 20, offset: 0 })
    expect(page.get('.paging').text()).toContain('第 1 页')
  })
})
