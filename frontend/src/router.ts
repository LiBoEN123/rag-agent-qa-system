import { createRouter, createWebHashHistory } from 'vue-router'

export default createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/', redirect: '/chat' },
    { path: '/chat', component: () => import('./views/ChatView.vue'), meta: { title: '智能问答', description: '基于知识库与受控工具，回答问题并保留证据。' } },
    { path: '/search', component: () => import('./views/SearchView.vue'), meta: { title: '检索调试', description: '查看召回片段、来源与各阶段评分。' } },
    { path: '/knowledge', component: () => import('./views/KnowledgeView.vue'), meta: { title: '知识库', description: '浏览当前知识库文档和已索引的文本片段。' } },
    { path: '/ingestion', component: () => import('./views/IngestionView.vue'), meta: { title: '知识入库', description: '审核待审批资料，独立执行索引并检查运行结果。' } },
    { path: '/diagnostics', component: () => import('./views/DiagnosticsView.vue'), meta: { title: '系统诊断', description: '检查路由决策与近期请求指标。' } },
    { path: '/:pathMatch(.*)*', redirect: '/chat' },
  ],
})
