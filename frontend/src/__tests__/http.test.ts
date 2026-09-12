import { afterEach, describe, expect, it } from 'vitest'
import { AxiosError, AxiosHeaders, type AxiosAdapter, type InternalAxiosRequestConfig } from 'axios'
import { http, ingestionHttp, ingestionAdminKey, setIngestionAdminKey, apiErrorMessage, apiResponseError } from '../api/http'

afterEach(() => setIngestionAdminKey(''))

describe('管理员凭据与公开错误', () => {
  it('管理员 key 只发送给 ingestion，公开诊断请求不带 key', async () => {
    const sent: InternalAxiosRequestConfig[] = []
    const adapter: AxiosAdapter = async (config) => {
      sent.push(config)
      return { data: [], status: 200, statusText: 'OK', headers: {}, config }
    }
    setIngestionAdminKey('test-only-admin-key')
    await ingestionHttp.get('/admin/ingestion/candidates', { adapter })
    await http.post('/search', { question: '测试' }, { adapter })
    await http.get('/health', { adapter })
    expect(sent.map((config) => config.headers.get('X-Ingestion-Admin-Key'))).toEqual(['test-only-admin-key', undefined, undefined])
    expect(localStorage.length).toBe(0)
    expect(sessionStorage.getItem('test-only-admin-key')).toBeNull()
    expect(ingestionAdminKey.value).toBe('test-only-admin-key')
  })

  it('管理员 client 拒绝普通端点和外站地址', async () => {
    setIngestionAdminKey('test-only-admin-key')
    await expect(ingestionHttp.get('/search')).rejects.toThrow('管理员请求地址无效')
    await expect(ingestionHttp.get('https://example.com/admin/ingestion/candidates')).rejects.toThrow('管理员请求地址无效')
    await expect(ingestionHttp.get('/admin/ingestion/candidates', { baseURL: 'https://example.com' })).rejects.toThrow('管理员请求地址无效')
  })

  it('401 显示中文认证错误，409 保留后端 code/message', () => {
    expect(apiErrorMessage(apiResponseError(401, { error: { code: 'INGESTION_ADMIN_UNAUTHORIZED', message: 'ingestion admin authentication failed' } })))
      .toContain('管理员认证失败，请检查 Ingestion Admin Key')
    expect(apiErrorMessage(apiResponseError(409, { error: { code: 'CANDIDATE_NOT_APPROVED', message: '只有 approved candidate 可以 index' } })))
      .toBe('CANDIDATE_NOT_APPROVED · 只有 approved candidate 可以 index')
  })

  it('不展示 Axios config、校验输入或 key，即使错误中包含 key', () => {
    setIngestionAdminKey('test-only-admin-key')
    const config = { headers: new AxiosHeaders({ 'X-Ingestion-Admin-Key': ingestionAdminKey.value }) } as InternalAxiosRequestConfig
    const failure = new AxiosError('test-only-admin-key', 'ERR_BAD_REQUEST', config, undefined, {
      config, data: { error: { code: 'BUSINESS_ERROR', message: 'invalid test-only-admin-key' } }, status: 409, statusText: 'Conflict', headers: {},
    })
    expect(apiErrorMessage(failure)).toBe('BUSINESS_ERROR · invalid [REDACTED]')
    expect(apiErrorMessage(apiResponseError(422, { detail: [{ input: 'test-only-admin-key' }] }))).not.toContain('test-only-admin-key')
  })
})
