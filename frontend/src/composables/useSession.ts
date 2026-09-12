import { readonly, ref } from 'vue'

const storageKey = 'rag-agent-session-id'

function createSessionId(): string {
  return `web_${crypto.randomUUID()}`
}

function readSessionId(): string {
  try {
    const saved = sessionStorage.getItem(storageKey)
    if (saved && /^web_[0-9a-f-]{36}$/i.test(saved)) return saved
  } catch {
    // Private browsing or storage restrictions still allow an in-memory conversation.
  }
  return createSessionId()
}

const sessionId = ref(readSessionId())

function saveSessionId(): void {
  try { sessionStorage.setItem(storageKey, sessionId.value) } catch { /* memory-only fallback */ }
}
saveSessionId()

export function useSession() {
  function newSession(): void {
    sessionId.value = createSessionId()
    saveSessionId()
  }
  return { sessionId: readonly(sessionId), newSession }
}
