<script setup lang="ts">
import { computed } from 'vue'
import MarkdownIt from 'markdown-it'
import DOMPurify from 'dompurify'

const props = defineProps<{ content: string }>()
const markdown = new MarkdownIt({ html: false, linkify: true, breaks: true })
const safeHtml = computed(() => DOMPurify.sanitize(markdown.render(props.content), {
  USE_PROFILES: { html: true },
  FORBID_TAGS: ['img'],
}))
</script>

<template>
  <div class="markdown-content" v-html="safeHtml" />
</template>

<style scoped>
.markdown-content { overflow-wrap: anywhere; line-height: 1.7; }
.markdown-content :deep(pre) { overflow: auto; padding: 12px; background: var(--el-fill-color-light); border-radius: 6px; }
.markdown-content :deep(table) { display: block; overflow-x: auto; border-collapse: collapse; }
.markdown-content :deep(td), .markdown-content :deep(th) { border: 1px solid var(--el-border-color); padding: 6px 10px; }
.markdown-content :deep(p:first-child) { margin-top: 0; }
.markdown-content :deep(p:last-child) { margin-bottom: 0; }
</style>
