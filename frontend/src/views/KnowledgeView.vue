<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import {
  getChunks, getDocuments,
  type ChunkPreview, type DocumentPreview, type PreviewResponse,
} from '../api/diagnostics'
import { apiErrorMessage } from '../api/http'

const tab = ref('documents')
const docs = ref<PreviewResponse<DocumentPreview> | null>(null)
const docsLoading = ref(false)
const docsError = ref('')
const docsFilter = ref('')
const docsPage = ref(1)
const docsPageSize = ref(10)
const filteredDocs = computed(() => {
  const needle = docsFilter.value.trim().toLocaleLowerCase()
  return (docs.value?.data ?? []).filter((doc) => doc.file_name.toLocaleLowerCase().includes(needle))
})
const pagedDocs = computed(() => filteredDocs.value.slice(
  (docsPage.value - 1) * docsPageSize.value, docsPage.value * docsPageSize.value,
))

const chunks = ref<PreviewResponse<ChunkPreview> | null>(null)
const chunksLoading = ref(false)
const chunksError = ref('')
const fileName = ref('')
const maxLen = ref<number | undefined>(300)
const limit = ref<number | undefined>(70)
const fetchedLimit = ref(70)
const fetchedFileName = ref('')
const chunksPage = ref(1)
const chunksPageSize = ref(10)
const pagedChunks = computed(() => (chunks.value?.data ?? []).slice(
  (chunksPage.value - 1) * chunksPageSize.value, chunksPage.value * chunksPageSize.value,
))

watch([docsFilter, docsPageSize], () => { docsPage.value = 1 })
watch(chunksPageSize, () => { chunksPage.value = 1 })
watch(tab, (value) => {
  if (value === 'chunks' && !chunks.value && !chunksError.value) void loadChunks()
})

async function loadDocuments() {
  if (docsLoading.value) return
  docsLoading.value = true
  docsError.value = ''
  try {
    docs.value = await getDocuments()
    docsPage.value = 1
  } catch (cause) {
    docs.value = null
    docsError.value = apiErrorMessage(cause)
  } finally {
    docsLoading.value = false
  }
}

async function loadChunks() {
  if (chunksLoading.value) return
  const requestedMaxLen = maxLen.value
  const requestedLimit = limit.value
  if (typeof requestedMaxLen !== 'number' || !Number.isInteger(requestedMaxLen) || requestedMaxLen < 1 || requestedMaxLen > 1000
    || typeof requestedLimit !== 'number' || !Number.isInteger(requestedLimit) || requestedLimit < 1 || requestedLimit > 1000) {
    chunksError.value = '预览长度 max_len 和返回上限 limit 必须是 1–1000 的整数'
    return
  }
  chunksLoading.value = true
  chunksError.value = ''
  const params = { max_len: requestedMaxLen, limit: requestedLimit, file_name: fileName.value.trim() || undefined }
  try {
    chunks.value = await getChunks(params)
    fetchedLimit.value = params.limit
    fetchedFileName.value = params.file_name ?? ''
    chunksPage.value = 1
  } catch (cause) {
    chunks.value = null
    chunksError.value = apiErrorMessage(cause)
  } finally {
    chunksLoading.value = false
  }
}

onMounted(() => { void loadDocuments() })
</script>

<template>
  <section>
    <div>
      <h1 class="page-heading">知识库</h1>
      <p class="page-subtitle">浏览当前 RAG 索引中的文档与 Chunk 预览。</p>
    </div>
    <el-card shadow="never">
      <el-tabs v-model="tab">
        <el-tab-pane label="文档" name="documents">
          <div class="toolbar">
            <el-input v-model="docsFilter" aria-label="按文件名筛选文档" placeholder="按文件名筛选已加载文档" clearable class="file-filter" />
            <el-button :loading="docsLoading" @click="loadDocuments">刷新文档</el-button>
          </div>
          <el-alert v-if="docsError" :title="docsError" type="error" show-icon :closable="false" />
          <div v-loading="docsLoading" class="preview-results" aria-live="polite">
            <template v-if="docs">
              <p class="result-caption">后端返回 {{ docs.count }} 份文档；筛选匹配 {{ filteredDocs.length }} 份。</p>
              <el-table v-if="filteredDocs.length" :data="pagedDocs" stripe>
                <el-table-column prop="doc_id" label="Doc ID" min-width="160" show-overflow-tooltip />
                <el-table-column prop="file_name" label="文件名" min-width="180" show-overflow-tooltip />
                <el-table-column prop="file_type" label="文件类型" width="100" />
                <el-table-column prop="chunk_count" label="Chunk 数" width="105" />
                <el-table-column prop="source_path" label="来源路径" min-width="260" show-overflow-tooltip />
              </el-table>
              <el-empty v-else :description="docs.data.length ? '没有匹配文件名的文档' : '当前知识库暂无文档'" />
              <el-pagination
                v-if="filteredDocs.length"
                v-model:current-page="docsPage" v-model:page-size="docsPageSize"
                :total="filteredDocs.length" :page-sizes="[10, 20, 50]"
                layout="sizes, prev, pager, next" class="pagination"
              />
            </template>
          </div>
        </el-tab-pane>
        <el-tab-pane label="Chunks" name="chunks" lazy>
          <el-form label-position="top" @submit.prevent="loadChunks">
            <div class="chunk-controls">
              <el-form-item label="文件名（可选，完整匹配）" class="file-control">
                <el-input v-model="fileName" placeholder="输入完整文件名" clearable />
              </el-form-item>
              <el-form-item label="预览长度 max_len">
                <el-input-number v-model="maxLen" :min="1" :max="1000" :precision="0" />
              </el-form-item>
              <el-form-item label="返回上限 limit">
                <el-input-number v-model="limit" :min="1" :max="1000" :precision="0" />
              </el-form-item>
              <el-form-item label="加载预览">
                <el-button type="primary" native-type="submit" :loading="chunksLoading">加载 Chunks</el-button>
              </el-form-item>
            </div>
          </el-form>
          <el-alert v-if="chunksError" :title="chunksError" type="error" show-icon :closable="false" />
          <div v-loading="chunksLoading" class="preview-results" aria-live="polite">
            <template v-if="chunks">
              <p class="result-caption">
                本次返回 {{ chunks.count }} 条预览，请求上限 {{ fetchedLimit }} 条；分页仅浏览本次返回内容。
                <span v-if="fetchedFileName">文件名：{{ fetchedFileName }}。</span>
              </p>
              <el-alert
                v-if="chunks.count >= fetchedLimit" title="已达到本次返回上限，可调整 limit 或按文件名筛选。接口未提供匹配总数。"
                type="info" :closable="false" class="limit-note"
              />
              <el-table v-if="chunks.data.length" :data="pagedChunks" stripe>
                <el-table-column type="expand">
                  <template #default="{ row }">
                    <div class="chunk-detail">
                      <p class="snippet">{{ row.text_preview }}</p>
                      <p>source_path：{{ row.source_path || '—' }}</p>
                    </div>
                  </template>
                </el-table-column>
                <el-table-column prop="file_name" label="文件名" min-width="160" show-overflow-tooltip />
                <el-table-column prop="chunk_id" label="Chunk ID" min-width="150" show-overflow-tooltip />
                <el-table-column prop="position" label="位置" width="75" />
                <el-table-column prop="text_length" label="原文长度" width="100" />
                <el-table-column prop="text_preview" label="内容预览" min-width="240" show-overflow-tooltip />
                <el-table-column prop="source_path" label="来源路径" min-width="200" show-overflow-tooltip />
              </el-table>
              <el-empty v-else description="没有符合条件的 Chunk" />
              <el-pagination
                v-if="chunks.data.length"
                v-model:current-page="chunksPage" v-model:page-size="chunksPageSize"
                :total="chunks.data.length" :page-sizes="[10, 20, 50]"
                layout="sizes, prev, pager, next" class="pagination"
              />
            </template>
            <el-empty v-else-if="!chunksLoading && !chunksError" description="加载 Chunk 预览以浏览内容" />
          </div>
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </section>
</template>

<style scoped>
.toolbar { display: flex; gap: 12px; margin: 8px 0 20px; flex-wrap: wrap; }
.file-filter { max-width: 360px; }
.chunk-controls { display: flex; gap: 16px; flex-wrap: wrap; }
.file-control { flex: 1; min-width: 220px; }
.preview-results { min-height: 180px; }
.result-caption { color: var(--el-text-color-secondary); line-height: 1.6; }
.pagination { margin-top: 20px; max-width: 100%; overflow-x: auto; }
.chunk-detail { padding: 8px 24px; overflow-wrap: anywhere; }
.snippet { white-space: pre-wrap; line-height: 1.7; }
.limit-note { margin-bottom: 12px; }
</style>
