<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { decideRoute, getMetrics, type DecisionResult, type MetricsSummary } from '../api/diagnostics'
import { apiErrorMessage } from '../api/http'

const tab = ref('decision')
const question = ref('')
const decision = ref<DecisionResult | null>(null)
const decisionLoading = ref(false)
const decisionError = ref('')
const route = computed(() => {
  const value = decision.value?.decision
  return value && typeof value === 'object' ? value : null
})

const limit = ref<number | undefined>(20)
const metrics = ref<MetricsSummary | null>(null)
const metricsLoading = ref(false)
const metricsError = ref('')
const statusRows = computed(() => Object.entries(metrics.value?.status_count ?? {}).map(([status, count]) => ({ status, count })))
const cards: { key: keyof Omit<MetricsSummary, 'status_count'>; label: string; precision?: number }[] = [
  { key: 'total', label: '请求数' },
  { key: 'success_count', label: '成功数' },
  { key: 'avg_latency_ms', label: '平均耗时 (ms)', precision: 2 },
  { key: 'avg_llm_call_count', label: '平均 LLM 调用', precision: 2 },
  { key: 'tool_error_count', label: '工具错误' },
  { key: 'model_error_count', label: '模型错误' },
  { key: 'clarify_count', label: '待澄清' },
  { key: 'internal_error_count', label: '内部错误' },
  { key: 'no_relevant_info_count', label: '无相关信息' },
  { key: 'router_llm_used_count', label: 'Router 使用 LLM' },
  { key: 'answer_llm_used_count', label: 'Answer 使用 LLM' },
]

watch(tab, (value) => {
  if (value === 'metrics' && !metrics.value && !metricsError.value) void loadMetrics()
})

async function analyze() {
  if (decisionLoading.value) return
  if (!question.value.trim()) {
    decisionError.value = '请输入待决策问题'
    return
  }
  decisionLoading.value = true
  decisionError.value = ''
  decision.value = null
  try {
    decision.value = await decideRoute(question.value.trim())
  } catch (cause) {
    decisionError.value = apiErrorMessage(cause)
  } finally {
    decisionLoading.value = false
  }
}

async function loadMetrics() {
  if (metricsLoading.value) return
  const requestedLimit = limit.value
  if (typeof requestedLimit !== 'number' || !Number.isInteger(requestedLimit) || requestedLimit < 1 || requestedLimit > 200) {
    metricsError.value = '统计请求数 limit 必须是 1–200 的整数'
    return
  }
  metricsLoading.value = true
  metricsError.value = ''
  try {
    metrics.value = await getMetrics(requestedLimit)
  } catch (cause) {
    metrics.value = null
    metricsError.value = apiErrorMessage(cause)
  } finally {
    metricsLoading.value = false
  }
}
</script>

<template>
  <section>
    <div>
      <h1 class="page-heading">系统诊断</h1>
      <p class="page-subtitle">检查入口路由决策与近期请求指标。</p>
    </div>
    <el-card shadow="never">
      <el-tabs v-model="tab">
        <el-tab-pane label="路由决策" name="decision">
          <p class="hint">查看当前确定性入口规则如何选择 Agent、直接回答或澄清。</p>
          <el-form label-position="top" @submit.prevent="analyze">
            <el-form-item label="待决策问题">
              <el-input v-model="question" type="textarea" :rows="3" placeholder="输入要分析的问题" />
            </el-form-item>
            <el-button type="primary" native-type="submit" :loading="decisionLoading">分析路由</el-button>
          </el-form>
          <el-alert v-if="decisionError" :title="decisionError" type="error" show-icon :closable="false" class="feedback" />
          <div v-loading="decisionLoading" class="diagnostic-result" aria-live="polite">
            <template v-if="decision">
              <p>问题：{{ decision.question }}</p>
              <el-descriptions v-if="route" :column="1" border>
                <el-descriptions-item label="decision"><el-tag>{{ route.decision }}</el-tag></el-descriptions-item>
                <el-descriptions-item label="action">{{ route.action }}</el-descriptions-item>
                <el-descriptions-item label="reason">{{ route.reason }}</el-descriptions-item>
                <el-descriptions-item label="router_source">{{ route.router_source }}</el-descriptions-item>
                <el-descriptions-item label="router_confidence">{{ route.router_confidence }}</el-descriptions-item>
                <el-descriptions-item v-if="route.clarification_question" label="澄清问题">{{ route.clarification_question }}</el-descriptions-item>
              </el-descriptions>
              <template v-else>
                <el-tag>{{ decision.decision }}</el-tag>
                <p v-if="decision.reason">原因：{{ decision.reason }}</p>
              </template>
            </template>
            <el-empty v-else-if="!decisionLoading && !decisionError" description="输入问题，查看路由决策" />
          </div>
        </el-tab-pane>
        <el-tab-pane label="运行指标" name="metrics" lazy>
          <el-form label-position="top" class="metrics-controls" @submit.prevent="loadMetrics">
            <el-form-item label="统计最近请求数 limit">
              <el-input-number v-model="limit" :min="1" :max="200" :precision="0" />
            </el-form-item>
            <el-button type="primary" native-type="submit" :loading="metricsLoading">刷新指标</el-button>
          </el-form>
          <el-alert v-if="metricsError" :title="metricsError" type="error" show-icon :closable="false" />
          <div v-loading="metricsLoading" class="diagnostic-result" aria-live="polite">
            <template v-if="metrics">
              <p class="hint">统计范围：最近最多 {{ metrics.limit }} 条请求，实际读取 {{ metrics.total }} 条。</p>
              <el-empty v-if="!metrics.total" description="暂无请求指标" />
              <template v-else>
                <div class="metrics-grid">
                  <el-card v-for="card in cards" :key="card.key" shadow="never">
                    <el-statistic :title="card.label" :value="metrics[card.key]" :precision="card.precision ?? 0" />
                    <span class="metric-key">{{ card.key }}</span>
                  </el-card>
                </div>
                <h2>状态分布</h2>
                <el-table :data="statusRows" stripe>
                  <el-table-column prop="status" label="Status" />
                  <el-table-column prop="count" label="请求数" />
                </el-table>
              </template>
            </template>
          </div>
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </section>
</template>

<style scoped>
.hint { color: var(--el-text-color-secondary); line-height: 1.6; }
.feedback { margin-top: 20px; }
.diagnostic-result { min-height: 160px; margin-top: 24px; overflow-wrap: anywhere; }
.metrics-controls { display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }
.metrics-controls :deep(.el-form-item) { margin-bottom: 0; }
.metrics-controls > .el-button { margin-top: 22px; }
.metrics-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
.metric-key { display: block; margin-top: 8px; font-size: 11px; color: var(--el-text-color-secondary); overflow-wrap: anywhere; }
h2 { font-size: 18px; margin-top: 28px; }
@media (max-width: 1200px) { .metrics-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 800px) { .metrics-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
</style>
