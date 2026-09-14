import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import * as api from '../api/ingestion'
import IngestionProposals from '../views/ingestion/IngestionProposals.vue'
import { candidateDetail } from './ingestion-fixtures'

vi.mock('../api/ingestion', () => ({
  proposeGitHub: vi.fn(), proposeFirecrawl: vi.fn(), proposeUpload: vi.fn(),
  approveCandidate: vi.fn(), indexCandidate: vi.fn(),
}))

let wrapper: VueWrapper
beforeEach(async () => {
  vi.resetAllMocks()
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} })
  wrapper = mount(IngestionProposals, { global: { plugins: [ElementPlus] } })
  await wrapper.get('input[value="upload"]').setValue(true)
  vi.mocked(api.proposeUpload).mockResolvedValue({
    ...candidateDetail().summary, source_type: 'upload', source_uri: 'upload://local/digest',
  })
})
afterEach(() => { wrapper.unmount(); vi.unstubAllGlobals() })

async function chooseFile(file: File) {
  const input = wrapper.get('input[type="file"]')
  Object.defineProperty(input.element, 'files', { value: [file], configurable: true })
  await input.trigger('change')
  await flushPromises()
}

describe('本地文件来源提案', () => {
  it.each(['pdf', 'docx', 'txt', 'md'])('选择 %s 文件仅生成候选并可查看详情', async (extension) => {
    const file = new File(['source text'], `guide.${extension}`)
    expect(wrapper.get('input[type="file"]').attributes('accept')).toBe('.pdf,.docx,.txt,.md')
    await chooseFile(file)
    expect(api.proposeUpload).not.toHaveBeenCalled()
    await wrapper.get('textarea').setValue('  上传审核  ')
    await wrapper.get('form').trigger('submit')
    await flushPromises()
    expect(api.proposeUpload).toHaveBeenCalledWith({ file, reason: '上传审核' })
    expect(wrapper.text()).toContain('来源提案已生成知识候选')
    expect(wrapper.text()).toContain('pending')
    expect(wrapper.text()).toContain('仍需批准后手动写入知识库')
    expect(wrapper.emitted('proposed')).toHaveLength(1)
    await wrapper.findAll('button').find(button => button.text() === '查看知识候选')!.trigger('click')
    expect(wrapper.emitted('inspect')).toEqual([['cand_example']])
    expect(api.approveCandidate).not.toHaveBeenCalled()
    expect(api.indexCandidate).not.toHaveBeenCalled()
  })

  it('Element Plus 拖拽接收单文件，移除后禁止提交', async () => {
    const file = new File(['dragged text'], 'dragged.txt')
    await wrapper.get('.el-upload-dragger').trigger('drop', { dataTransfer: { files: [file] } })
    await flushPromises()
    expect(wrapper.text()).toContain('dragged.txt')
    await wrapper.get('.el-upload-list__item .el-icon--close').trigger('click')
    await wrapper.get('form').trigger('submit')
    expect(wrapper.text()).toContain('请选择一个文件')
    expect(api.proposeUpload).not.toHaveBeenCalled()
  })

  it('拒绝不支持格式、空文件和第二个文件', async () => {
    await chooseFile(new File(['data'], 'bad.exe'))
    expect(wrapper.text()).toContain('仅支持 PDF、DOCX、TXT、MD 文件')
    await chooseFile(new File([], 'empty.txt'))
    expect(wrapper.text()).toContain('不能上传空文件')
    const first = new File(['data'], 'first.txt')
    await chooseFile(first)
    await chooseFile(new File(['data'], 'second.txt'))
    expect(wrapper.text()).toContain('每次只能上传一个文件')
    await wrapper.get('form').trigger('submit')
    await flushPromises()
    expect(api.proposeUpload).toHaveBeenCalledWith({ file: first, reason: null })
  })

  it('上传失败显示统一错误且不自动重试', async () => {
    vi.mocked(api.proposeUpload).mockRejectedValue(new Error('文件超过上传大小限制'))
    await chooseFile(new File(['body'], 'guide.txt'))
    await wrapper.get('form').trigger('submit')
    await flushPromises()
    expect(wrapper.text()).toContain('文件超过上传大小限制')
    expect(api.proposeUpload).toHaveBeenCalledOnce()
    expect(wrapper.emitted('proposed')).toBeUndefined()
  })
})
