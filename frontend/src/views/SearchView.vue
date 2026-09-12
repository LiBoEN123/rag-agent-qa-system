<script setup lang="ts">
import { ref } from 'vue'
import { searchKnowledge, type SearchResponse } from '../api/diagnostics'
import { apiErrorMessage } from '../api/http'

const question = ref('')
const topK = ref<number | undefined>(5)
const loading = ref(false)
const error = ref('')
const result = ref<SearchResponse | null>(null)

async function search() {
  if (loading.value) return
  if (!question.value.trim()) {
    error.value = '请输入检索问题'
    return
  }
  const requestedTopK = topK.value
  if (typeof requestedTopK !== 'number' || !Number.isInteger(requestedTopK) || requestedTopK < 1 || requestedTopK > 10) {
    error.value = '召回条数 top_k 必须是 1–10 的整数'
    return
  }
  loading.value = true
  error.value = ''
  result.value = null
  try {
    result.value = await searchKnowledge(question.value.trim(), requestedTopK)
  } catch (cause) {
    error.value = apiErrorMessage(cause)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <section>
    <div>
      <h1 class="page-heading">检索调试</h1>
      <p class="page-subtitle">查看知识库召回、重排与证据判断结果。</p>
    </div>
    <el-card shadow="never">
      <el-form label-position="top" @submit.prevent="search">
        <el-form-item label="检索问题">
          <el-input v-model="question" type="textarea" :rows="3" placeholder="输入要检索的问题" />
        </el-form-item>
        <div class="search-controls">
          <el-form-item label="召回条数 top_k">
            <el-input-number v-model="topK" :min="1" :max="10" :precision="0" />
          </el-form-item>
          <el-button type="primary" native-type="submit" :loading="loading">执行检索</el-button>
        </div>
      </el-form>
    </el-card>
    <el-alert v-if="error" :title="error" type="error" show-icon :closable="false" class="feedback" />
    <div v-loading="loading" class="search-results" aria-live="polite">
      <template v-if="result">
        <div class="result-summary">
          <el-tag :type="result.data.status === 'ok' ? 'success' : 'warning'">{{ result.data.status }}</el-tag>
          <span>结果数：{{ result.data.results.length }}</span>
          <span v-if="result.code !== 0">{{ result.code }}：{{ result.message }}</span>
        </div>
        <p>问题：{{ result.data.question }}</p>
        <p>原因：{{ result.data.reason || '—' }}</p>
        <el-empty v-if="!result.data.results.length" description="本次检索没有返回可用片段" />
        <template v-else>
          <el-table :data="result.data.results" stripe row-key="reference_id">
            <el-table-column prop="reference_id" label="Reference" min-width="120" />
            <el-table-column prop="file_name" label="文件名" min-width="180" show-overflow-tooltip />
            <el-table-column prop="chunk_id" label="Chunk ID" min-width="140" show-overflow-tooltip />
            <el-table-column prop="score" label="Score" min-width="95" />
            <el-table-column prop="bm25_score" label="BM25" min-width="95" />
            <el-table-column prop="vector_score" label="Vector" min-width="95" />
            <el-table-column prop="rerank_score" label="Rerank" min-width="95" />
            <el-table-column prop="source_path" label="来源路径" min-width="200" show-overflow-tooltip />
          </el-table>
          <el-collapse class="feedback">
            <el-collapse-item v-for="item in result.data.results" :key="item.reference_id" :name="item.reference_id" :title="`${item.reference_id} · ${item.file_name}`">
              <p class="snippet">{{ item.snippet }}</p>
              <p class="source">source_path：{{ item.source_path }}</p>
            </el-collapse-item>
          </el-collapse>
        </template>
      </template>
      <el-empty v-else-if="!loading && !error" description="输入问题，开始一次检索" />
    </div>
  </section>
</template>

<style scoped>
.search-controls { display: flex; align-items: center; gap: 24px; flex-wrap: wrap; }
.search-controls :deep(.el-form-item) { margin-bottom: 0; }
.search-controls > .el-button { margin-top: 22px; }
.feedback { margin-top: 20px; }
.search-results { min-height: 160px; margin-top: 24px; }
.result-summary { display: flex; gap: 18px; flex-wrap: wrap; align-items: center; }
.snippet { white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.7; }
.source { overflow-wrap: anywhere; color: var(--el-text-color-secondary); }
</style>
