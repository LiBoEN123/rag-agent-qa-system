import { ref } from 'vue'
import { getHealth, getKnowledgeSummary, type HealthStatus, type KnowledgeSummary } from '../api/diagnostics'
import { apiErrorMessage } from '../api/http'

const health = ref<HealthStatus | null>(null)
const kb = ref<KnowledgeSummary | null>(null)
const loading = ref(false)
const error = ref('')
let pendingRefresh: Promise<void> | null = null

function refreshStatus(force = false): Promise<void> {
  if (pendingRefresh) {
    // A refresh after indexing must read state after that operation, even if a
    // sidebar request that began before indexing is still in flight.
    return force === true ? pendingRefresh.then(() => refreshStatus()) : pendingRefresh
  }
  loading.value = true
  error.value = ''
  pendingRefresh = Promise.allSettled([getHealth(), getKnowledgeSummary()])
    .then(([healthResult, kbResult]) => {
      health.value = healthResult.status === 'fulfilled' ? healthResult.value : null
      kb.value = kbResult.status === 'fulfilled' ? kbResult.value : null
      const messages: string[] = []
      if (healthResult.status === 'rejected') messages.push(`后端：${apiErrorMessage(healthResult.reason)}`)
      if (kbResult.status === 'rejected') messages.push(`知识库：${apiErrorMessage(kbResult.reason)}`)
      error.value = messages.join('；')
    })
    .finally(() => {
      loading.value = false
      pendingRefresh = null
    })
  return pendingRefresh
}

export function useBackendStatus() {
  return { health, kb, loading, error, refreshStatus }
}
