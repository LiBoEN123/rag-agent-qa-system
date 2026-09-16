import axios from 'axios'
import { ref } from 'vue'

export const ingestionAdminKey = ref('')
export function setIngestionAdminKey(value: string) { ingestionAdminKey.value = value.trim() }

export const http = axios.create({ timeout: 120_000 })
export const ingestionHttp = axios.create({ timeout: 180_000 })
ingestionHttp.interceptors.request.use((config) => {
  // Only attach credentials to this same-origin administrative namespace.
  if (!config.url?.startsWith('/admin/ingestion/') || config.baseURL) {
    throw new Error('管理员请求地址无效')
  }
  if (ingestionAdminKey.value) config.headers.set('X-Ingestion-Admin-Key', ingestionAdminKey.value)
  return config
})

class ApiError extends Error {
  constructor(public status: number, public code: string | number, message: string) {
    super(message)
  }
}
function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' ? value as Record<string, unknown> : {}
}
function redact(message: string): string {
  return ingestionAdminKey.value ? message.split(ingestionAdminKey.value).join('[REDACTED]') : message
}
export function apiResponseError(status: number, body: unknown): Error {
  const payload = record(body)
  const detail = record(payload.error)
  const code = detail.code ?? payload.code ?? `HTTP_${status}`
  let message = detail.message ?? payload.message
  if (status === 401) {
    message = '管理员认证失败，请检查 Ingestion Admin Key'
  }
  if (typeof message !== 'string') {
    message = status === 404 ? '接口或候选不存在；请确认 Ingestion 管理接口已启用'
      : status === 422 ? '请求参数无效，请检查必填项和取值范围'
      : status === 502 || status === 503 ? '后端或来源服务暂不可用，请检查服务状态'
      : `请求失败（HTTP ${status}）`
  }
  return new ApiError(status, typeof code === 'string' || typeof code === 'number' ? code : `HTTP_${status}`, redact(message as string))
}
export function apiErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    if (error.response) return apiErrorMessage(apiResponseError(error.response.status, error.response.data))
    return error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT'
      ? '请求超时；后端可能仍在处理，请刷新状态后再决定是否重试'
      : '无法连接后端，请确认 FastAPI 已启动且开发代理地址正确'
  }
  if (error instanceof ApiError) return redact(`${error.code} · ${error.message}`)
  // Never stringify an Axios config, request, or arbitrary exception object.
  return error instanceof Error ? redact(error.message) : '请求失败，请刷新后重试'
}
