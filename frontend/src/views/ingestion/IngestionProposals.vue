<script setup lang="ts">
import { reactive, ref } from 'vue'
import type { UploadFile, UploadUserFile } from 'element-plus'
import { apiErrorMessage } from '../../api/http'
import { proposeFirecrawl, proposeGitHub, proposeUpload } from '../../api/ingestion'
import type { IngestionCandidateSummary } from '../../api/types'

const emit = defineEmits<{ proposed: []; inspect: [id: string] }>()
const source = ref<'github' | 'firecrawl' | 'upload'>('github')
const files = ref<UploadUserFile[]>([])
const uploadFile = ref<File | null>(null)
const github = reactive({ owner: '', repository: '', path: '', ref: '' })
const url = ref('')
const reason = ref('')
const loading = ref(false)
const error = ref('')
const result = ref<IngestionCandidateSummary | null>(null)

function selectFile(file: UploadFile) {
  error.value = ''
  result.value = null
  uploadFile.value = null
  if (!/\.(pdf|docx|txt|md)$/i.test(file.name)) {
    files.value = []
    error.value = '仅支持 PDF、DOCX、TXT、MD 文件'
  } else if (!file.raw?.size) {
    files.value = []
    error.value = '不能上传空文件'
  } else {
    uploadFile.value = file.raw
  }
}

async function submit() {
  if (loading.value) return
  error.value = ''
  result.value = null
  if (source.value === 'github' && (!github.owner.trim() || !github.repository.trim() || !github.path.trim())) {
    error.value = '请填写 GitHub 所有者、仓库和文件路径'
    return
  }
  if (source.value === 'firecrawl' && !url.value.trim()) {
    error.value = '请填写网页 URL'
    return
  }
  if (source.value === 'upload' && !uploadFile.value) {
    error.value = '请选择一个文件'
    return
  }
  loading.value = true
  try {
    result.value = source.value === 'upload'
      ? await proposeUpload({ file: uploadFile.value!, reason: reason.value.trim() || null })
      : source.value === 'github'
      ? await proposeGitHub({ owner: github.owner.trim(), repository: github.repository.trim(), path: github.path.trim(), ref: github.ref.trim() || null, reason: reason.value.trim() || null })
      : await proposeFirecrawl({ url: url.value.trim(), reason: reason.value.trim() || null })
    emit('proposed')
  } catch (cause) { error.value = apiErrorMessage(cause) }
  finally { loading.value = false }
}
</script>

<template>
  <p>提交 GitHub、网页或本地文件来源。上传成功仅生成知识候选，仍需批准后手动写入知识库。</p>
  <el-form label-position="top" :disabled="loading" class="proposal-form" @submit.prevent="submit">
    <el-form-item label="来源类型">
      <el-radio-group v-model="source"><el-radio-button value="github">GitHub 文件</el-radio-button><el-radio-button value="firecrawl">Firecrawl 网页</el-radio-button><el-radio-button value="upload">本地文件</el-radio-button></el-radio-group>
    </el-form-item>
    <template v-if="source === 'github'">
      <el-form-item label="所有者 owner" required><el-input v-model="github.owner" :maxlength="128" aria-label="GitHub 所有者" /></el-form-item>
      <el-form-item label="仓库 repository" required><el-input v-model="github.repository" :maxlength="128" aria-label="GitHub 仓库" /></el-form-item>
      <el-form-item label="文件路径 path" required><el-input v-model="github.path" :maxlength="1024" placeholder="例如 docs/guide.md" aria-label="GitHub 文件路径" /></el-form-item>
      <el-form-item label="版本 ref（可选）"><el-input v-model="github.ref" :maxlength="256" aria-label="GitHub 版本" /></el-form-item>
    </template>
    <el-form-item v-else-if="source === 'firecrawl'" label="网页 URL" required><el-input v-model="url" :maxlength="2048" placeholder="https://example.com/article" aria-label="网页 URL" /></el-form-item>
    <el-form-item v-else label="本地文件" required>
      <el-upload v-model:file-list="files" drag accept=".pdf,.docx,.txt,.md" :limit="1" :auto-upload="false" :disabled="loading"
        :on-change="selectFile" :on-remove="() => { uploadFile = null; result = null }"
        :on-exceed="() => { error = '每次只能上传一个文件，请先移除已选文件' }">
        <div>拖拽文件到这里，或点击选择文件</div>
        <template #tip><div class="el-upload__tip">PDF / DOCX / TXT / MD；单文件，大小上限以服务器配置为准。扫描 PDF 暂不支持 OCR。</div></template>
      </el-upload>
    </el-form-item>
    <el-form-item label="申请原因（可选）"><el-input v-model="reason" type="textarea" :maxlength="1000" :rows="2" aria-label="来源提案申请原因" /></el-form-item>
    <el-button type="primary" native-type="submit" :loading="loading">提交来源提案</el-button>
  </el-form>
  <el-alert v-if="error" :title="error" type="error" :closable="false" show-icon />
  <template v-if="result">
    <el-alert title="来源提案已生成知识候选" type="success" :closable="false" show-icon />
    <el-descriptions :column="1" border>
      <el-descriptions-item label="标题">{{ result.title }}</el-descriptions-item>
      <el-descriptions-item label="候选 ID">{{ result.candidate_id }}</el-descriptions-item>
      <el-descriptions-item label="状态"><el-tag>{{ result.status }}</el-tag></el-descriptions-item>
      <el-descriptions-item label="来源">{{ result.source_type }} · {{ result.source_uri }}</el-descriptions-item>
      <el-descriptions-item label="内容长度">{{ result.content_length }}</el-descriptions-item>
      <el-descriptions-item label="信任级别">{{ result.trust_level }}</el-descriptions-item>
    </el-descriptions>
    <el-collapse><el-collapse-item title="完整候选摘要"><pre>{{ JSON.stringify(result, null, 2) }}</pre></el-collapse-item></el-collapse>
    <el-button @click="emit('inspect', result.candidate_id)">查看知识候选</el-button>
  </template>
</template>

<style scoped>
.proposal-form { max-width: 620px; margin: 20px 0; }
.el-alert { margin: 16px 0; }
.el-descriptions, pre { overflow-wrap: anywhere; }
pre { white-space: pre-wrap; }
</style>
