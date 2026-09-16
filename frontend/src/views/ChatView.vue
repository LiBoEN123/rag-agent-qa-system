<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'
import MarkdownContent from '../components/MarkdownContent.vue'
import { useChat } from '../composables/useChat'

const { messages, sending, send, cancel } = useChat()
const question = ref('')
const topK = ref(5)
const decomposeTopK = ref(5)
const conversation = ref<HTMLElement>()

async function submit(): Promise<void> {
  if (!question.value.trim() || sending.value) return
  const text = question.value
  question.value = ''
  await send(text, topK.value, decomposeTopK.value)
}

watch(() => messages.value.at(-1)?.content, async () => {
  const element = conversation.value
  if (!element || element.scrollHeight - element.scrollTop - element.clientHeight > 180) return
  await nextTick()
  element.scrollTo?.({ top: element.scrollHeight })
})

// Public diagnostics emitted by tool_events.py, results.py and metrics.py.
// Tool output bodies are intentionally not copied into this diagnostic view.
const traceFields = [
  'summary', 'error', 'error_code', 'error_type', 'success', 'step_id', 'tool_call_id',
  'sanitized_tool_args', 'policy_decision', 'risk_level', 'approval_required', 'args_fingerprint',
  'attempt_count', 'max_attempts', 'timed_out', 'handler_executed', 'failure_kind',
  'will_return_to_agent_model', 'stop_reason', 'call_fingerprint',
  'final_chunk_count', 'top_chunk_ids', 'top_scores', 'hits',
  'iteration_index', 'model_has_tool_calls', 'model_cost_ms', 'requested_tool_call_count', 'available_tool_names',
  'route_decision', 'decision_type', 'router_source', 'router_confidence', 'router_fallback_reason', 'route_cost_ms',
  'router_llm_used', 'decompose_llm_used', 'answer_llm_used', 'llm_call_count',
  'original_question', 'rewritten_question', 'history_used', 'rewritten', 'need_check', 'need_rewrite', 'rewrite_llm_used',
  'mode', 'answer_len', 'reference_count', 'retrieval_confidence', 'answer_confidence', 'timings', 'rag_llm_call_count',
]
function text(value: unknown): string {
  if (value === undefined || value === null || value === '') return '—'
  return typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)
}
</script>

<template>
  <section class="chat-page">
    <h1 class="page-heading">智能问答</h1>
    <p class="page-subtitle">结合知识库与受治理工具回答问题，查看引用和执行过程。</p>
    <el-form inline class="chat-options" :disabled="sending" label-position="left">
      <el-form-item label="召回条数"><el-input-number v-model="topK" :min="1" :max="10" size="small" aria-label="召回条数" /></el-form-item>
      <el-form-item label="拆分后召回条数"><el-input-number v-model="decomposeTopK" :min="1" :max="10" size="small" aria-label="拆分后召回条数" /></el-form-item>
    </el-form>

    <div ref="conversation" class="conversation" aria-live="polite" aria-relevant="additions text">
      <el-empty v-if="!messages.length" description="开始一段对话。同一 session 会保留后端对话上下文。" />
      <article v-for="message in messages" :key="message.id" class="message" :class="message.role" :data-state="message.state">
        <div class="message-heading">
          <strong>{{ message.role === 'user' ? '你' : 'Assistant' }}</strong>
          <el-tag v-if="message.response" :type="message.response.data.status === 'error' ? 'danger' : message.response.data.status === 'ok' ? 'success' : 'warning'" size="small">{{ message.response.data.status }}</el-tag>
          <el-tag v-else-if="message.role === 'assistant' && message.state === 'pending'" size="small">正在处理</el-tag>
        </div>
        <p v-if="message.role === 'user'" class="user-text">{{ message.content }}</p>
        <MarkdownContent v-else-if="message.content" :content="message.content" />
        <p v-else-if="message.state === 'pending'" class="muted">正在等待回答…</p>
        <p v-if="message.state === 'pending' && message.steps.length" class="muted">{{ message.steps.at(-1) }}</p>
        <el-alert v-if="message.error" :title="message.error" :type="message.state === 'cancelled' ? 'warning' : 'error'" :closable="false" show-icon />
        <el-alert v-if="message.response?.data.status === 'error'" :title="`${message.response.message}（${message.response.code}）`" type="error" :closable="false" show-icon />

        <el-collapse v-if="message.role === 'assistant'" class="message-details">
          <el-collapse-item v-if="message.steps.length" title="执行过程" name="steps">
            <ol class="execution-steps"><li v-for="(step, index) in message.steps" :key="index">{{ step }}</li></ol>
          </el-collapse-item>
          <el-collapse-item title="查看详情" name="details">
            <el-descriptions :column="2" border size="small">
              <el-descriptions-item label="trace_id">{{ message.response?.trace_id || message.traceId || '—' }}</el-descriptions-item>
              <el-descriptions-item label="请求状态">{{ message.state }}</el-descriptions-item>
              <template v-if="message.response">
                <el-descriptions-item label="status">{{ message.response.data.status }}</el-descriptions-item>
                <el-descriptions-item label="code">{{ message.response.code }}</el-descriptions-item>
                <el-descriptions-item label="action">{{ text(message.response.data.action) }}</el-descriptions-item>
                <el-descriptions-item label="retrieval_confidence">{{ message.response.data.retrieval_confidence }}</el-descriptions-item>
                <el-descriptions-item label="answer_confidence">{{ message.response.data.answer_confidence }}</el-descriptions-item>
                <el-descriptions-item label="llm_call_count">{{ message.response.data.llm_call_count }}</el-descriptions-item>
                <el-descriptions-item label="total_cost_ms">{{ message.totalCostMs ?? '—' }}</el-descriptions-item>
                <el-descriptions-item label="reason" :span="2">{{ text(message.response.data.reason) }}</el-descriptions-item>
                <el-descriptions-item v-if="message.response.data.clarification_question" label="clarification_question" :span="2">{{ message.response.data.clarification_question }}</el-descriptions-item>
              </template>
            </el-descriptions>
            <template v-if="message.response">
              <h3>References</h3>
              <el-table :data="[...message.response.data.references]" empty-text="本轮没有引用" size="small" border>
                <el-table-column type="expand"><template #default="{ row }"><p class="snippet">{{ row.snippet }}</p></template></el-table-column>
                <el-table-column prop="reference_id" label="reference_id" min-width="115" />
                <el-table-column prop="file_name" label="file_name" min-width="130" show-overflow-tooltip />
                <el-table-column prop="chunk_id" label="chunk_id" min-width="110" show-overflow-tooltip />
                <el-table-column prop="score" label="score" width="90" />
                <el-table-column prop="rerank_score" label="rerank_score" width="115" />
                <el-table-column prop="source_path" label="source_path" min-width="170" show-overflow-tooltip />
              </el-table>
              <h3>Tool Trace</h3>
              <el-table :data="[...message.response.data.tool_trace]" empty-text="本轮没有工具链路" size="small" border>
                <el-table-column type="expand">
                  <template #default="{ row }">
                    <el-descriptions :column="2" border size="small" class="trace-details">
                      <template v-for="field in traceFields" :key="field">
                        <el-descriptions-item v-if="row[field] !== undefined" :label="field"><span class="trace-value">{{ text(row[field]) }}</span></el-descriptions-item>
                      </template>
                    </el-descriptions>
                  </template>
                </el-table-column>
                <el-table-column prop="step" label="step" min-width="145" />
                <el-table-column prop="action" label="action" min-width="100" />
                <el-table-column prop="tool_name" label="tool_name" min-width="180" />
                <el-table-column label="status" min-width="100"><template #default="{ row }">{{ row.tool_status || row.status || row.final_status || '—' }}</template></el-table-column>
                <el-table-column prop="cost_ms" label="cost_ms" width="95" />
                <el-table-column prop="reason" label="reason" min-width="180" show-overflow-tooltip />
              </el-table>
            </template>
            <p v-else class="muted">{{ message.state === 'pending' ? '等待后端返回最终响应。' : '本轮未收到最终响应。' }}</p>
          </el-collapse-item>
        </el-collapse>
      </article>
    </div>

    <form class="composer" @submit.prevent="submit">
      <el-input v-model="question" type="textarea" :autosize="{ minRows: 2, maxRows: 6 }" placeholder="输入问题，Ctrl / Cmd + Enter 发送" :disabled="sending" aria-label="问题" @keydown.ctrl.enter.prevent="submit" @keydown.meta.enter.prevent="submit" />
      <div class="composer-actions">
        <span class="muted">回答会展示可用引用；外部内容请结合来源核对。</span>
        <el-button v-if="sending" type="danger" plain @click="cancel">停止生成</el-button>
        <el-button v-else type="primary" native-type="submit" :disabled="!question.trim()">发送</el-button>
      </div>
    </form>
  </section>
</template>

<style scoped>
.chat-page { max-width: 1200px; margin: 0 auto; }
.chat-options { margin-bottom: 8px; }
.conversation { min-height: 270px; max-height: calc(100vh - 370px); overflow-y: auto; padding: 4px 6px 16px; }
.message { padding: 18px; border: 1px solid var(--el-border-color-light); border-radius: 10px; margin-bottom: 16px; background: var(--el-bg-color); }
.message.user { margin-left: 12%; background: var(--el-color-primary-light-9); }
.message-heading { display: flex; gap: 12px; align-items: center; margin-bottom: 12px; }
.user-text, .snippet, .trace-value { white-space: pre-wrap; overflow-wrap: anywhere; }
.user-text { margin: 0; }
.snippet, .trace-details { margin: 12px; }
.message-details { margin-top: 14px; }
.execution-steps { padding-left: 24px; line-height: 1.9; overflow-wrap: anywhere; }
.composer { margin-top: 12px; }
.composer-actions { display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-top: 10px; }
.muted { color: var(--el-text-color-secondary); font-size: 13px; }
.message :deep(.el-alert) { margin-top: 12px; }
@media (max-width: 760px) { .conversation { max-height: none; } .message { padding: 12px; } .message.user { margin-left: 5%; } .composer-actions > span { display: none; } .composer-actions { justify-content: flex-end; } }
</style>
