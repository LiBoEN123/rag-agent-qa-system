<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useSession } from '../composables/useSession'
import { useBackendStatus } from '../composables/useBackendStatus'

const route = useRoute()
const router = useRouter()
const collapsed = ref(false)
const { sessionId, newSession } = useSession()
const { health, kb, loading, error, refreshStatus } = useBackendStatus()
const online = computed(() => health.value?.status === 'ok')
const navigation = [
  { path: '/chat', label: '智能问答', icon: '问' },
  { path: '/search', label: '检索调试', icon: '检' },
  { path: '/knowledge', label: '知识库', icon: '库' },
  { path: '/ingestion', label: '知识入库', icon: '审' },
  { path: '/diagnostics', label: '系统诊断', icon: '诊' },
]
function startConversation() {
  newSession()
  void router.push('/chat')
}
onMounted(refreshStatus)
</script>

<template>
  <el-container class="app-shell">
    <el-aside :width="collapsed ? '72px' : '244px'" class="sidebar">
      <div class="brand"><span class="brand-mark">R</span><div v-if="!collapsed"><strong>RAG Agent</strong><small>知识工作台</small></div></div>
      <div class="new-conversation"><el-button type="primary" :title="collapsed ? '新建对话' : undefined" @click="startConversation">{{ collapsed ? '+' : '＋ 新建对话' }}</el-button></div>
      <el-menu router :default-active="route.path" :collapse="collapsed" :collapse-transition="false">
        <el-menu-item v-for="item in navigation" :key="item.path" :index="item.path">
          <el-icon><span class="nav-glyph">{{ item.icon }}</span></el-icon>
          <template #title>{{ item.label }}</template>
        </el-menu-item>
      </el-menu>
      <div v-if="!collapsed" class="sidebar-status">
        <div class="status-title">运行状态 <el-tag :type="online ? 'success' : 'info'" size="small">{{ loading ? '检查中' : online ? '在线' : '离线' }}</el-tag></div>
        <dl>
          <div><dt>RAG ready</dt><dd>{{ kb ? (kb.ready ? '是' : '否') : '—' }}</dd></div>
          <div><dt>文档数</dt><dd>{{ kb?.doc_count ?? '—' }}</dd></div>
          <div><dt>Chunk 数</dt><dd>{{ kb?.chunk_count ?? '—' }}</dd></div>
          <div><dt>index_source</dt><dd>{{ kb?.index_source ?? '—' }}</dd></div>
        </dl>
        <p v-if="error" class="status-error" role="alert">{{ error }}</p>
        <el-button :loading="loading" size="small" @click="refreshStatus()">刷新状态</el-button>
        <div class="session-label">当前 session_id</div>
        <code class="session-id">{{ sessionId }}</code>
      </div>
      <el-button class="collapse-button" text :aria-label="collapsed ? '展开侧栏' : '折叠侧栏'" @click="collapsed = !collapsed">{{ collapsed ? '»' : '« 折叠侧栏' }}</el-button>
    </el-aside>
    <el-container class="workspace">
      <el-header class="workspace-header"><span>工作空间 <span class="header-divider">/</span> <strong>{{ route.meta.title }}</strong></span><span class="environment-label"><i :class="{ online }" />{{ online ? '后端已连接' : '后端未连接' }}</span></el-header>
      <el-main class="workspace-main"><router-view /></el-main>
    </el-container>
  </el-container>
</template>
