import { afterEach, describe, expect, it } from 'vitest'
import type { AxiosAdapter, InternalAxiosRequestConfig } from 'axios'
import { http, ingestionHttp, setIngestionAdminKey } from '../api/http'
import { approveCandidate, getCandidate, listCandidates, listIndexRuns, proposeFirecrawl, proposeGitHub } from '../api/ingestion'

const originalHttpAdapter = http.defaults.adapter
const originalIngestionAdapter = ingestionHttp.defaults.adapter
afterEach(() => {
  http.defaults.adapter = originalHttpAdapter
  ingestionHttp.defaults.adapter = originalIngestionAdapter
  setIngestionAdminKey('')
})

describe('入库 API 边界', () => {
  it('管理员密钥仅附加到 ingestion 请求，且不会持久化到浏览器存储', async () => {
    const requests: InternalAxiosRequestConfig[] = []
    const adapter: AxiosAdapter = async (config) => {
      requests.push(config)
      return { config, status: 200, statusText: 'OK', headers: {}, data: [] }
    }
    http.defaults.adapter = adapter
    ingestionHttp.defaults.adapter = adapter
    setIngestionAdminKey('test-only-admin-key')
    await http.post('/chat', { question: '你好' })
    await listCandidates({ status: 'pending', source_type: 'github', limit: 20, offset: 40 })
    expect(requests[0]!.headers.has('X-Ingestion-Admin-Key')).toBe(false)
    expect(requests[1]!.headers.get('X-Ingestion-Admin-Key')).toBe('test-only-admin-key')
    expect(requests[1]!.params).toEqual({ status: 'pending', source_type: 'github', limit: 20, offset: 40 })
    expect(JSON.stringify(localStorage)).not.toContain('test-only-admin-key')
    expect(JSON.stringify(sessionStorage)).not.toContain('test-only-admin-key')
    setIngestionAdminKey('')
    await listCandidates({ limit: 20, offset: 0 })
    expect(requests[2]!.headers.has('X-Ingestion-Admin-Key')).toBe(false)
    await expect(ingestionHttp.get('https://example.com/admin/ingestion/candidates')).rejects.toThrow('管理员请求地址无效')
    expect(requests).toHaveLength(3)
  })

  it('使用真实 query/body 合同、编码候选路径，来源请求仅发送给 FastAPI', async () => {
    const requests: InternalAxiosRequestConfig[] = []
    ingestionHttp.defaults.adapter = async (config) => {
      requests.push(config)
      return { config, status: 200, statusText: 'OK', headers: {}, data: {} }
    }
    await getCandidate('candidate/a?b')
    await approveCandidate('candidate_a', null)
    await listIndexRuns({ candidate_id: 'candidate_a', status: 'failed', limit: 10, offset: 10 })
    await proposeGitHub({ owner: 'example', repository: 'docs', path: 'guide.md', ref: 'main', reason: null })
    await proposeFirecrawl({ url: 'https://example.com/guide', reason: '审核' })
    expect(requests[0]!.url).toBe('/admin/ingestion/candidates/candidate%2Fa%3Fb')
    expect(JSON.parse(requests[1]!.data as string)).toEqual({ reason: null })
    expect(requests[2]!.params).toEqual({ candidate_id: 'candidate_a', status: 'failed', limit: 10, offset: 10 })
    expect(requests[3]!.url).toBe('/admin/ingestion/sources/github')
    expect(JSON.parse(requests[3]!.data as string)).toEqual({ owner: 'example', repository: 'docs', path: 'guide.md', ref: 'main', reason: null })
    expect(requests[4]!.url).toBe('/admin/ingestion/sources/firecrawl')
  })
})
