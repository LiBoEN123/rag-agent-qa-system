<script setup lang="ts">
import type { IngestionIndexRun } from '../../api/types'

defineProps<{ runs: IngestionIndexRun[] }>()
</script>

<template>
  <el-table :data="runs" empty-text="暂无索引记录" size="small" style="width: 100%">
    <el-table-column type="expand">
      <template #default="{ row }">
        <el-descriptions :column="1" border class="run-details">
          <el-descriptions-item label="候选 ID">{{ row.candidate_id }}</el-descriptions-item>
          <el-descriptions-item label="错误码">{{ row.error_code ?? '—' }}</el-descriptions-item>
          <el-descriptions-item label="错误消息">{{ row.error_message ?? '—' }}</el-descriptions-item>
        </el-descriptions>
      </template>
    </el-table-column>
    <el-table-column prop="id" label="记录 ID" width="85" />
    <el-table-column prop="candidate_id" label="候选 ID" min-width="145" show-overflow-tooltip />
    <el-table-column label="状态" width="115">
      <template #default="{ row }"><el-tag :type="row.status === 'failed' ? 'danger' : row.status === 'success' ? 'success' : 'info'">{{ row.status }}</el-tag></template>
    </el-table-column>
    <el-table-column prop="started_at" label="开始时间" min-width="190" />
    <el-table-column prop="completed_at" label="结束时间" min-width="190" />
    <el-table-column prop="added" label="新增 added" width="110" />
    <el-table-column prop="updated" label="更新 updated" width="120" />
    <el-table-column prop="deleted" label="删除 deleted" width="120" />
    <el-table-column prop="skipped" label="跳过 skipped" width="120" />
    <el-table-column prop="error_code" label="错误码" min-width="180" />
  </el-table>
</template>

<style scoped>
.run-details { margin: 12px; overflow-wrap: anywhere; }
</style>
