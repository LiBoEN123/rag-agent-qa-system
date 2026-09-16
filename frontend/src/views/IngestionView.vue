<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from 'vue'
import { ingestionAdminKey, setIngestionAdminKey } from '../api/http'
import { useBackendStatus } from '../composables/useBackendStatus'
import IngestionCandidates from './ingestion/IngestionCandidates.vue'
import IngestionIndexRuns from './ingestion/IngestionIndexRuns.vue'
import IngestionProposals from './ingestion/IngestionProposals.vue'

const { refreshStatus } = useBackendStatus()
const adminKey = computed({ get: () => ingestionAdminKey.value, set: setIngestionAdminKey })
const tab = ref('candidates')
const candidates = ref<InstanceType<typeof IngestionCandidates> | null>(null)
const runs = ref<InstanceType<typeof IngestionIndexRuns> | null>(null)
const refreshing = ref(false)

async function refresh() {
  refreshing.value = true
  await Promise.allSettled([candidates.value?.refresh(true), runs.value?.refresh(true)])
  refreshing.value = false
}

async function inspect(id: string) {
  tab.value = 'candidates'
  await nextTick()
  await candidates.value?.openDetail(id)
}

function tabChanged(name: string | number) {
  if (name === 'runs' && adminKey.value) void runs.value?.refresh()
}

onMounted(() => { if (adminKey.value) void refresh() })
</script>

<template>
  <section>
    <h1 class="page-heading">知识入库</h1>
    <p class="page-subtitle">审核知识候选，批准后手动写入知识库，并查看索引结果。</p>
    <el-card shadow="never" class="admin-card">
      <el-form label-position="top" @submit.prevent="refresh">
        <el-form-item label="管理员 API Key">
          <el-input v-model="adminKey" type="password" autocomplete="off" aria-label="管理员 API Key" placeholder="输入 Ingestion Admin Key" clearable />
        </el-form-item>
        <el-button type="primary" native-type="submit" :loading="refreshing">连接并刷新</el-button>
        <span class="key-note">密钥仅在当前页面内存中保存，刷新浏览器后清除。</span>
      </el-form>
    </el-card>
    <el-tabs v-model="tab" @tab-change="tabChanged">
      <el-tab-pane label="待审批知识" name="candidates">
        <IngestionCandidates ref="candidates" @indexed="refreshStatus(true)" @index-attempted="runs?.refresh()" />
      </el-tab-pane>
      <el-tab-pane label="索引记录" name="runs"><IngestionIndexRuns ref="runs" /></el-tab-pane>
      <el-tab-pane label="来源提案" name="proposals">
        <IngestionProposals @proposed="candidates?.refresh(true)" @inspect="inspect" />
      </el-tab-pane>
    </el-tabs>
  </section>
</template>

<style scoped>
.admin-card { margin: 20px 0; }
.admin-card .el-input { max-width: 480px; }
.key-note { display: inline-block; margin: 8px 12px; color: var(--el-text-color-secondary); font-size: 13px; }
</style>
