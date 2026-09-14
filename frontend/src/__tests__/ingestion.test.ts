import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { defineComponent } from 'vue'
import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import ElementPlus, { ElMessageBox } from 'element-plus'
import { AxiosError, AxiosHeaders } from 'axios'
import * as api from '../api/ingestion'
import { setIngestionAdminKey } from '../api/http'
import type { CandidateDetailResponse } from '../api/types'
import IngestionView from '../views/IngestionView.vue'
import { candidateDetail, indexRun } from './ingestion-fixtures'

const { refreshStatus } = vi.hoisted(() => ({ refreshStatus: vi.fn().mockResolvedValue(undefined) }))
vi.mock('../composables/useBackendStatus', () => ({ useBackendStatus: () => ({ refreshStatus }) }))
vi.mock('../api/ingestion', () => ({
  listCandidates: vi.fn(), getCandidate: vi.fn(), approveCandidate: vi.fn(), rejectCandidate: vi.fn(),
  indexCandidate: vi.fn(), listIndexRuns: vi.fn(), proposeGitHub: vi.fn(), proposeFirecrawl: vi.fn(), proposeUpload: vi.fn(),
}))

const drawer = defineComponent({ props: ['modelValue'], template: '<section v-if="modelValue"><slot /></section>' })
let wrapper: VueWrapper | undefined
let current: CandidateDetailResponse

function backendError(status: number, code: string, message: string) {
  return new AxiosError('request failed', undefined, undefined, undefined, {
    status, statusText: '', data: { error: { code, message } },
    headers: {}, config: { headers: new AxiosHeaders() },
  })
}

async function openCandidate() {
  wrapper = mount(IngestionView, { global: { plugins: [ElementPlus], stubs: { ElDrawer: drawer } } })
  await flushPromises()
  const button = wrapper.findAll('button').find((item) => item.text() === '查看详情')
  expect(button).toBeDefined()
  await button!.trigger('click')
  await flushPromises()
  return wrapper
}

beforeEach(() => {
  vi.resetAllMocks()
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} })
  setIngestionAdminKey('test-memory-key')
  current = candidateDetail()
  vi.mocked(api.listCandidates).mockImplementation(async () => [structuredClone(current.summary)])
  vi.mocked(api.getCandidate).mockImplementation(async () => structuredClone(current))
  vi.mocked(api.listIndexRuns).mockResolvedValue([])
  vi.mocked(api.approveCandidate).mockImplementation(async () => {
    current = candidateDetail('approved')
    return structuredClone(current.summary)
  })
  // confirm resolves an action string; Element Plus declares a prompt-data intersection.
  vi.spyOn(ElMessageBox, 'confirm').mockResolvedValue('confirm' as Awaited<ReturnType<typeof ElMessageBox.confirm>>)
})

afterEach(() => {
  wrapper?.unmount()
  wrapper = undefined
  setIngestionAdminKey('')
  vi.unstubAllGlobals()
})

describe('知识候选审核', () => {
  it('pending 可审批但不能索引，正文始终作为文本展示', async () => {
    const page = await openCandidate()
    expect(page.get('[data-testid="approve-candidate"]').attributes('disabled')).toBeUndefined()
    expect(page.get('[data-testid="reject-candidate"]').attributes('disabled')).toBeUndefined()
    expect(page.get('[data-testid="index-candidate"]').attributes('disabled')).toBeDefined()
    expect(page.text()).toContain('外部未信任内容')
    expect(page.text()).toContain('原文长度为 3000')
    expect(page.find('.preview script').exists()).toBe(false)
    expect(page.get('.preview').text()).toContain('<script>untrusted()</script>')
    expect(page.get('input[aria-label="管理员 API Key"]').attributes('type')).toBe('password')
  })

  it('批准先确认，刷新详情与列表，但不会自动索引', async () => {
    const page = await openCandidate()
    await page.get('[data-testid="approve-candidate"]').trigger('click')
    await flushPromises()
    expect(ElMessageBox.confirm).toHaveBeenCalledOnce()
    expect(api.approveCandidate).toHaveBeenCalledWith('cand_example', null)
    expect(api.indexCandidate).not.toHaveBeenCalled()
    expect(api.getCandidate).toHaveBeenCalledTimes(2)
    expect(api.listCandidates).toHaveBeenCalledTimes(2)
    expect(page.get('[data-testid="approve-candidate"]').attributes('disabled')).toBeDefined()
    expect(page.get('[data-testid="reject-candidate"]').attributes('disabled')).toBeDefined()
    expect(page.get('[data-testid="index-candidate"]').attributes('disabled')).toBeUndefined()
    expect(page.text()).toContain('尚未执行索引')
  })

  it('取消批准不发起写请求，拒绝原因必须非空', async () => {
    const page = await openCandidate()
    vi.mocked(ElMessageBox.confirm).mockRejectedValueOnce('cancel')
    await page.get('[data-testid="approve-candidate"]').trigger('click')
    await flushPromises()
    expect(api.approveCandidate).not.toHaveBeenCalled()
    await page.get('textarea[aria-label="审批原因"]').setValue('   ')
    await page.get('[data-testid="reject-candidate"]').trigger('click')
    expect(api.rejectCandidate).not.toHaveBeenCalled()
    expect(page.text()).toContain('拒绝原因不能为空')
    vi.mocked(api.rejectCandidate).mockResolvedValue({ ...current.summary, status: 'rejected', approval_status: 'rejected' })
    await page.get('textarea[aria-label="审批原因"]').setValue('  来源不合适  ')
    await page.get('[data-testid="reject-candidate"]').trigger('click')
    await flushPromises()
    expect(api.rejectCandidate).toHaveBeenCalledWith('cand_example', '来源不合适')
  })

  it('approved 只启用索引，成功显示四项计数并刷新记录和 KB 状态', async () => {
    current = candidateDetail('approved')
    vi.mocked(api.indexCandidate).mockImplementation(async () => {
      current.latest_index_result = indexRun()
      return indexRun()
    })
    const page = await openCandidate()
    expect(page.get('[data-testid="approve-candidate"]').attributes('disabled')).toBeDefined()
    expect(page.get('[data-testid="reject-candidate"]').attributes('disabled')).toBeDefined()
    expect(page.get('[data-testid="index-candidate"]').attributes('disabled')).toBeUndefined()
    await page.get('[data-testid="index-candidate"]').trigger('click')
    await flushPromises()
    expect(api.indexCandidate).toHaveBeenCalledWith('cand_example')
    const counts = page.get('[data-testid="index-counts"]').text()
    for (const value of ['added：2', 'updated：3', 'deleted：4', 'skipped：5']) expect(counts).toContain(value)
    expect(api.getCandidate).toHaveBeenCalledTimes(2)
    expect(api.listIndexRuns).toHaveBeenCalledTimes(2)
    expect(refreshStatus).toHaveBeenCalledOnce()
    expect(refreshStatus).toHaveBeenCalledWith(true)
  })

  it('401 显示管理员认证提示且不回显密钥', async () => {
    vi.mocked(api.listCandidates).mockRejectedValue(backendError(401, 'INGESTION_ADMIN_UNAUTHORIZED', 'ingestion admin authentication failed'))
    wrapper = mount(IngestionView, { global: { plugins: [ElementPlus] } })
    await flushPromises()
    expect(wrapper.text()).toContain('管理员认证失败，请检查 Ingestion Admin Key')
    expect(wrapper.text()).not.toContain('test-memory-key')
    expect(api.listCandidates).toHaveBeenCalledOnce()
  })

  it('409 保留后端业务 code/message，失败不会自动重试', async () => {
    current = candidateDetail('approved')
    vi.mocked(api.indexCandidate).mockRejectedValue(backendError(409, 'CANDIDATE_NOT_APPROVED', '只有 approved candidate 可以 index'))
    const page = await openCandidate()
    await page.get('[data-testid="index-candidate"]').trigger('click')
    await flushPromises()
    expect(page.get('[data-testid="candidate-error"]').text()).toContain('CANDIDATE_NOT_APPROVED')
    expect(page.get('[data-testid="candidate-error"]').text()).toContain('只有 approved candidate 可以 index')
    expect(api.indexCandidate).toHaveBeenCalledOnce()
    expect(refreshStatus).not.toHaveBeenCalled()
    expect(api.listIndexRuns).toHaveBeenCalledTimes(2)
  })

  it('404 候选缺失显示安全错误且不保留可操作的旧详情', async () => {
    vi.mocked(api.getCandidate).mockRejectedValue(backendError(404, 'CANDIDATE_NOT_FOUND', 'candidate 不存在'))
    const page = await openCandidate()
    expect(page.get('[data-testid="candidate-error"]').text()).toContain('CANDIDATE_NOT_FOUND · candidate 不存在')
    expect(page.find('[data-testid="approve-candidate"]').exists()).toBe(false)
    expect(page.find('[data-testid="index-candidate"]').exists()).toBe(false)
    expect(api.getCandidate).toHaveBeenCalledOnce()
  })

  it('索引失败保留 approved 状态并刷新安全失败记录，允许手动重试', async () => {
    current = candidateDetail('approved')
    vi.mocked(api.indexCandidate).mockImplementation(async () => {
      current.latest_index_result = {
        ...indexRun(), status: 'failed', added: null, updated: null, deleted: null, skipped: null,
        error_code: 'INDEX_OPERATION_FAILED', error_message: 'index operation 失败',
      }
      throw backendError(500, 'INDEX_OPERATION_FAILED', 'index operation 失败')
    })
    const page = await openCandidate()
    await page.get('[data-testid="index-candidate"]').trigger('click')
    await flushPromises()
    expect(page.get('[data-testid="candidate-error"]').text()).toContain('INDEX_OPERATION_FAILED')
    expect(page.text()).toContain('failed')
    expect(page.find('[data-testid="index-counts"]').exists()).toBe(false)
    expect(page.get('[data-testid="index-candidate"]').attributes('disabled')).toBeUndefined()
    expect(page.get('[data-testid="approve-candidate"]').attributes('disabled')).toBeDefined()
    expect(api.indexCandidate).toHaveBeenCalledOnce()
    expect(api.getCandidate).toHaveBeenCalledTimes(2)
    expect(api.listIndexRuns).toHaveBeenCalledTimes(2)
    expect(refreshStatus).not.toHaveBeenCalled()
  })
})
