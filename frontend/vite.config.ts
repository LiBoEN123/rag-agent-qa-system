import { defineConfig } from 'vitest/config'
import { loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), 'VITE_DEV_')
  const target = env.VITE_DEV_PROXY_TARGET || 'http://127.0.0.1:8000'
  // Hash routing keeps browser navigation distinct from POST /chat and /search.
  const endpoints = [
    '/health', '/chat', '/search', '/kb_summary', '/docs-preview',
    '/chunks-preview', '/decide', '/metrics-summary', '/admin/ingestion',
  ]
  const proxy = Object.fromEntries(endpoints.map((path) => [
    `^${path}(?:/|\\?|$)`, { target, changeOrigin: true },
  ]))
  return {
    plugins: [vue()],
    server: { host: 'localhost', port: 5173, strictPort: true, proxy },
    preview: { host: 'localhost', port: 4173, strictPort: true, proxy },
    test: { environment: 'jsdom', clearMocks: true, restoreMocks: true },
  }
})
