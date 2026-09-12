<script setup lang="ts">
import { ref, watch } from 'vue'
import { apiErrorMessage } from '../../api/http'
import { listIndexRuns } from '../../api/ingestion'
import type { IngestionIndexRun } from '../../api/types'
import IndexRunTable from '../../components/ingestion/IndexRunTable.vue'

const runs = ref<IngestionIndexRun[]>([])
const candidateId = ref('')
const status = ref<IngestionIndexRun['status'] | ''>('')
const limit = ref(20)
const offset = ref(0)
const loading = ref(false)
const loaded = ref(false)
const error = ref('')
let listRequest = 0

watch([candidateId, status, limit], () => {
  offset.value = 0
  loaded.value = false
}, { flush: 'sync' })

async function refresh(reset = false) {
  if (reset) offset.value = 0
  const request = ++listRequest
  loading.value = true
  error.value = ''
  try {
    const result = await listIndexRuns({
      candidate_id: candidateId.value.trim() || undefined,
      status: status.value || undefined, limit: limit.value, offset: offset.value,
    })
    if (request !== listRequest) return
    runs.value = result
    loaded.value = true
  } catch (cause) {
    if (request !== listRequest) return
    runs.value = []
    loaded.value = false
    error.value = apiErrorMessage(cause)
  } finally { if (request === listRequest) loading.value = false }
}

function changePage(direction: number) {
  if (loading.value || !loaded.value) return
  offset.value = Math.max(0, offset.value + direction * limit.value)
  void refresh()
}

defineExpose({ refresh })
</script>

<template>
  <div class="filters">
    <el-input v-model="candidateId" :disabled="loading" aria-label="按候选 ID 筛选索引记录" placeholder="候选 ID（可选）" clearable @change="refresh(true)" />
    <el-select v-model="status" :disabled="loading" aria-label="索引状态筛选" placeholder="全部状态" clearable @change="refresh(true)">
      <el-option v-for="item in ['running', 'success', 'failed']" :key="item" :label="item" :value="item" />
    </el-select>
    <el-select v-model="limit" :disabled="loading" aria-label="索引每页条数" @change="refresh(true)">
      <el-option v-for="size in [10, 20, 50, 100]" :key="size" :label="`每页 ${size} 条`" :value="size" />
    </el-select>
    <el-button :loading="loading" @click="refresh(true)">刷新索引记录</el-button>
  </div>
  <el-alert v-if="error" :title="error" type="error" :closable="false" show-icon />
  <IndexRunTable v-loading="loading" :runs="runs" />
  <div class="paging">
    <el-button :disabled="loading || offset === 0" @click="changePage(-1)">上一页</el-button>
    <span>第 {{ offset / limit + 1 }} 页 · 本页 {{ runs.length }} 条</span>
    <el-button :disabled="loading || !loaded || runs.length < limit" @click="changePage(1)">下一页</el-button>
  </div>
</template>

<style scoped>
.filters, .paging { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin: 16px 0; }
.filters .el-input { width: 260px; }
.filters .el-select { width: 160px; }
.paging { justify-content: flex-end; }
</style>
