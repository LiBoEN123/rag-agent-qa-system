<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { ElMessageBox } from 'element-plus'
import { apiErrorMessage } from '../../api/http'
import { approveCandidate, deleteCandidate, getCandidate, indexCandidate, listCandidates, rejectCandidate, removeCandidate } from '../../api/ingestion'
import type { CandidateDetailResponse, IngestionCandidateStatus, IngestionCandidateSummary, IngestionIndexRun, KnowledgeSourceType } from '../../api/types'
import IndexRunTable from '../../components/ingestion/IndexRunTable.vue'

const emit = defineEmits<{ indexed: []; indexAttempted: [] }>()
const statuses: IngestionCandidateStatus[] = ['pending', 'approved', 'rejected', 'superseded']
const sources: KnowledgeSourceType[] = ['local_file', 'upload', 'github', 'web', 'firecrawl']
const candidates = ref<IngestionCandidateSummary[]>([])
const status = ref<IngestionCandidateStatus | ''>('pending')
const sourceType = ref<KnowledgeSourceType | ''>('')
const limit = ref(20)
const offset = ref(0)
const loading = ref(false)
const loaded = ref(false)
const error = ref('')
const drawerOpen = ref(false)
const detail = ref<CandidateDetailResponse | null>(null)
const detailLoading = ref(false)
const detailError = ref('')
const acting = ref(false)
const reason = ref('')
const success = ref('')
const indexResult = ref<IngestionIndexRun | null>(null)
// 候选的最近操作不能代表同 source 其他版本的状态；撤库不受审批状态限制。
const canRemove = computed(() => detail.value && ['upload', 'github', 'firecrawl'].includes(detail.value.summary.source_type))
let listRequest = 0

watch([status, sourceType, limit], () => {
  offset.value = 0
  loaded.value = false
}, { flush: 'sync' })

async function refresh(reset = false) {
  if (reset) offset.value = 0
  const request = ++listRequest
  loading.value = true
  error.value = ''
  try {
    const result = await listCandidates({
      status: status.value || undefined, source_type: sourceType.value || undefined,
      limit: limit.value, offset: offset.value,
    })
    if (request !== listRequest) return
    candidates.value = result
    loaded.value = true
  } catch (cause) {
    if (request !== listRequest) return
    candidates.value = []
    loaded.value = false
    error.value = apiErrorMessage(cause)
  } finally { if (request === listRequest) loading.value = false }
}

function changePage(direction: number) {
  if (loading.value || !loaded.value) return
  offset.value = Math.max(0, offset.value + direction * limit.value)
  void refresh()
}

function pendingOnly() {
  status.value = 'pending'
  void refresh(true)
}

async function refreshDetail(id: string) {
  detailLoading.value = true
  try { detail.value = await getCandidate(id) }
  catch (cause) {
    detail.value = null
    detailError.value ||= apiErrorMessage(cause)
  }
  finally { detailLoading.value = false }
}

async function openDetail(id: string) {
  if (acting.value || detailLoading.value) return
  detail.value = null
  detailError.value = ''
  reason.value = ''
  success.value = ''
  indexResult.value = null
  drawerOpen.value = true
  await refreshDetail(id)
}

async function review(action: 'approve' | 'reject') {
  if (!detail.value || detail.value.summary.status !== 'pending' || acting.value) return
  detailError.value = ''
  success.value = ''
  if (action === 'reject' && !reason.value.trim()) {
    detailError.value = '拒绝原因不能为空'
    return
  }
  const id = detail.value.summary.candidate_id
  acting.value = true
  try {
    await ElMessageBox.confirm(
      action === 'approve' ? '确认批准这条知识候选？批准后仍需手动写入知识库。' : '确认拒绝这条知识候选？',
      action === 'approve' ? '批准知识候选' : '拒绝知识候选',
      { confirmButtonText: '确认', cancelButtonText: '取消', type: 'warning' },
    )
    const summary = action === 'approve'
      ? await approveCandidate(id, reason.value.trim() || null)
      : await rejectCandidate(id, reason.value.trim())
    detail.value.summary = summary
    success.value = action === 'approve' ? '已批准，尚未执行索引。请手动点击「写入知识库」。' : '已拒绝该知识候选。'
    await Promise.allSettled([refreshDetail(id), refresh()])
  } catch (cause) {
    if (cause !== 'cancel' && cause !== 'close') detailError.value = apiErrorMessage(cause)
  } finally { acting.value = false }
}

async function index() {
  if (!detail.value || detail.value.summary.status !== 'approved' || acting.value) return
  const id = detail.value.summary.candidate_id
  acting.value = true
  detailError.value = ''
  success.value = ''
  indexResult.value = null
  try {
    indexResult.value = await indexCandidate(id)
    success.value = '索引完成，知识库状态已请求刷新。'
    emit('indexed')
  } catch (cause) { detailError.value = apiErrorMessage(cause) }
  finally {
    emit('indexAttempted')
    await refreshDetail(id)
    acting.value = false
  }
}

async function remove() {
  if (!detail.value || !canRemove.value || acting.value) return
  const id = detail.value.summary.candidate_id
  acting.value = true
  detailError.value = ''
  success.value = ''
  indexResult.value = null
  let attempted = false
  try {
    await ElMessageBox.confirm(
      '确认移出知识库？这会移除该来源当前入库版本的全部内容，保留候选和审批历史。后续全量重建不会恢复它；再次写入知识库后才会恢复。',
      '移出知识库',
      { confirmButtonText: '移出知识库', cancelButtonText: '取消', type: 'warning' },
    )
    attempted = true
    indexResult.value = await removeCandidate(id)
    success.value = '该来源已移出知识库，知识库状态已请求刷新。'
    emit('indexed')
  } catch (cause) {
    if (cause !== 'cancel' && cause !== 'close') detailError.value = apiErrorMessage(cause)
  } finally {
    if (attempted) {
      emit('indexAttempted')
      await refreshDetail(id)
    }
    acting.value = false
  }
}

async function deleteRecord() {
  if (!detail.value?.can_delete || acting.value || detailLoading.value) return
  const id = detail.value.summary.candidate_id
  acting.value = true
  detailError.value = ''
  success.value = ''
  try {
    await ElMessageBox.confirm(
      '确认永久删除这条知识候选记录？该来源已经移出知识库，本操作只删除候选及审批记录，不会删除已有 INDEX / REMOVE 操作历史。删除后，相同来源内容未来仍可以重新提交。',
      '删除记录',
      { confirmButtonText: '删除记录', cancelButtonText: '取消', type: 'warning' },
    )
    await deleteCandidate(id)
    drawerOpen.value = false
    detail.value = null
    indexResult.value = null
    await refresh()
  } catch (cause) {
    if (cause !== 'cancel' && cause !== 'close') detailError.value = apiErrorMessage(cause)
  } finally { acting.value = false }
}

defineExpose({ refresh, openDetail })
</script>

<template>
  <el-alert title="待审批知识" description="Agent 提出的合规外部资料、管理员来源提案和本地文件上传候选会出现在这里。刷新列表即可审核；批准后需要单独写入知识库。" type="info" :closable="false" show-icon />
  <div class="filters">
    <el-select v-model="status" :disabled="loading" aria-label="候选状态筛选" placeholder="全部状态" clearable @change="refresh(true)">
      <el-option v-for="item in statuses" :key="item" :label="item" :value="item" />
    </el-select>
    <el-select v-model="sourceType" :disabled="loading" aria-label="候选来源筛选" placeholder="全部来源" clearable @change="refresh(true)">
      <el-option v-for="item in sources" :key="item" :label="item" :value="item" />
    </el-select>
    <el-select v-model="limit" :disabled="loading" aria-label="候选每页条数" @change="refresh(true)">
      <el-option v-for="size in [10, 20, 50, 100]" :key="size" :label="`每页 ${size} 条`" :value="size" />
    </el-select>
    <el-button :loading="loading" data-testid="refresh-candidates" @click="refresh(true)">刷新候选</el-button>
    <el-button :disabled="loading" @click="pendingOnly">只看 Pending</el-button>
  </div>
  <el-alert v-if="error" :title="error" type="error" :closable="false" show-icon />
  <el-table v-loading="loading" :data="candidates" empty-text="当前筛选下暂无知识候选" style="width: 100%">
    <el-table-column label="状态" width="130">
      <template #default="{ row }"><el-tag :type="row.status === 'approved' ? 'success' : row.status === 'rejected' ? 'danger' : 'info'">{{ row.status }}</el-tag></template>
    </el-table-column>
    <el-table-column prop="title" label="标题" min-width="200" show-overflow-tooltip />
    <el-table-column prop="source_type" label="来源" width="110" />
    <el-table-column prop="trust_level" label="信任级别" min-width="175" />
    <el-table-column prop="content_length" label="内容长度" width="105" />
    <el-table-column prop="requested_at" label="申请时间" min-width="205" />
    <el-table-column label="操作" width="115" fixed="right">
      <template #default="{ row }"><el-button link type="primary" :disabled="acting || detailLoading" @click="openDetail(row.candidate_id)">查看详情</el-button></template>
    </el-table-column>
  </el-table>
  <div class="paging">
    <el-button :disabled="loading || offset === 0" @click="changePage(-1)">上一页</el-button>
    <span>第 {{ offset / limit + 1 }} 页 · 本页 {{ candidates.length }} 条</span>
    <el-button :disabled="loading || !loaded || candidates.length < limit" @click="changePage(1)">下一页</el-button>
  </div>

  <el-drawer v-model="drawerOpen" title="知识候选详情" size="min(850px, 96vw)" :close-on-click-modal="!acting" :close-on-press-escape="!acting" :show-close="!acting" destroy-on-close>
    <div v-loading="detailLoading" class="candidate-detail">
      <el-alert v-if="detailError" :title="detailError" type="error" :closable="false" show-icon data-testid="candidate-error" />
      <el-alert v-if="success" :title="success" type="success" :closable="false" show-icon />
      <template v-if="detail">
        <el-alert v-if="detail.content_is_untrusted" title="外部未信任内容" description="以下是待审核原始资料，请核对来源和内容后决定是否批准。" type="warning" :closable="false" show-icon />
        <el-descriptions :column="1" border>
          <el-descriptions-item label="标题">{{ detail.summary.title }}</el-descriptions-item>
          <el-descriptions-item label="候选 ID">{{ detail.summary.candidate_id }}</el-descriptions-item>
          <el-descriptions-item label="状态"><el-tag>{{ detail.summary.status }}</el-tag></el-descriptions-item>
          <el-descriptions-item label="来源">{{ detail.summary.source_type }}</el-descriptions-item>
          <el-descriptions-item label="来源 URI">{{ detail.summary.source_uri }}</el-descriptions-item>
          <el-descriptions-item label="内容 Hash">{{ detail.summary.content_hash }}</el-descriptions-item>
          <el-descriptions-item label="内容长度">{{ detail.content_length }}</el-descriptions-item>
          <el-descriptions-item label="信任级别">{{ detail.summary.trust_level }}</el-descriptions-item>
          <el-descriptions-item label="策略决定">{{ detail.summary.policy_decision }}</el-descriptions-item>
          <el-descriptions-item label="策略原因">{{ detail.summary.policy_reasons.join('、') }}</el-descriptions-item>
          <el-descriptions-item label="审批状态">{{ detail.summary.approval_status }}</el-descriptions-item>
          <el-descriptions-item label="申请时间">{{ detail.summary.requested_at }}</el-descriptions-item>
        </el-descriptions>
        <el-collapse>
          <el-collapse-item title="申请人 / 审批信息 / 安全元数据" name="metadata">
            <h3>申请人</h3><pre>{{ JSON.stringify(detail.requester, null, 2) }}</pre>
            <h3>审批信息</h3><pre>{{ JSON.stringify(detail.approval, null, 2) }}</pre>
            <h3>安全元数据</h3><pre>{{ JSON.stringify(detail.safe_metadata, null, 2) }}</pre>
          </el-collapse-item>
        </el-collapse>
        <h3>安全正文预览</h3>
        <p v-if="detail.preview_truncated">这里只展示后端提供的安全预览，原文长度为 {{ detail.content_length }}。</p>
        <pre class="preview">{{ detail.content_preview }}</pre>
        <el-form label-position="top" @submit.prevent>
          <el-form-item label="审批原因（拒绝时必填）">
            <el-input v-model="reason" type="textarea" :rows="2" :maxlength="1000" :disabled="acting || detail.summary.status !== 'pending'" aria-label="审批原因" />
          </el-form-item>
          <div class="actions">
            <el-button type="success" data-testid="approve-candidate" :disabled="acting || detailLoading || detail.summary.status !== 'pending'" @click="review('approve')">批准</el-button>
            <el-button type="danger" data-testid="reject-candidate" :disabled="acting || detailLoading || detail.summary.status !== 'pending'" @click="review('reject')">拒绝</el-button>
            <el-button type="primary" data-testid="index-candidate" :disabled="acting || detailLoading || detail.summary.status !== 'approved'" @click="index">写入知识库</el-button>
            <el-button type="danger" plain data-testid="remove-candidate" :disabled="acting || detailLoading || !canRemove" @click="remove">移出知识库</el-button>
            <el-button type="danger" plain data-testid="delete-candidate" :disabled="acting || detailLoading || !detail.can_delete" @click="deleteRecord">删除记录</el-button>
          </div>
        </el-form>
        <p v-if="!detail.can_delete">需要先将该来源移出知识库，才能删除候选记录。</p>
        <p v-if="acting" role="status">正在处理，请稍候…</p>
        <div v-if="indexResult" class="index-counts" data-testid="index-counts">
          <span>新增 added：{{ indexResult.added }}</span><span>更新 updated：{{ indexResult.updated }}</span>
          <span>删除 deleted：{{ indexResult.deleted }}</span><span>跳过 skipped：{{ indexResult.skipped }}</span>
        </div>
        <h3>本候选最近知识库操作</h3>
        <p>操作记录按候选展示；移出知识库会移除同一来源当前入库的全部内容。</p>
        <IndexRunTable :runs="detail.latest_index_result ? [detail.latest_index_result] : []" />
      </template>
    </div>
  </el-drawer>
</template>

<style scoped>
.filters, .paging, .actions, .index-counts { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin: 16px 0; }
.filters .el-select { width: 155px; }
.paging { justify-content: flex-end; }
.candidate-detail { min-height: 120px; overflow-wrap: anywhere; }
.candidate-detail .el-alert { margin-bottom: 16px; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.65; }
.preview { padding: 16px; background: var(--el-fill-color-light); border-radius: 6px; }
.index-counts { padding: 12px; background: var(--el-color-success-light-9); }
</style>
