<template>
  <div class="admin-wrap">
    <div class="panel">
      <div class="toolbar">
        <div class="title">后台处理进度</div>
        <div class="actions">
          <button class="btn primary" :disabled="polling || !pipelineJobId" @click="startPolling">刷新进度</button>
          <button class="btn" :disabled="!polling" @click="stopPolling">停止</button>
        </div>
      </div>
      <div class="content">
        <div class="hint" v-if="!pipelineJobId">未检测到 Job ID，请在上方输入或从上传页启动管道。</div>
        
        <div class="hint error" v-if="uploadError">{{ uploadError }}</div>
        <div v-if="stepStatuses && stepStatuses.length" class="progress-wrap">
          <div class="progress-header">
            <span>处理进度</span>
            <button class="btn" @click="toggleDetails">{{ showDetails ? '隐藏详情' : '显示详情' }}</button>
          </div>
          <ul class="progress-list">
            <li v-for="(s, i) in stepStatuses" :key="i">
              <span class="step-name">{{ stepNames[i] }}</span>
              <span class="step-state" :class="stateClass(s)">{{ displayState(s) }}</span>
            </li>
          </ul>
          <div v-if="showDetails && pipelineLogs" class="guide logs">{{ pipelineLogs }}</div>
        </div>

        <div class="uploads-section">
          <div class="uploads-header">
            <span class="title">已上传的 JSON 数据</span>
            <div class="actions">
              <button class="btn" @click="refreshUploadsList">刷新</button>
            </div>
          </div>
          <div v-if="uploads && uploads.length" class="uploads-list" role="list">
            <div v-for="f in pagedUploads" :key="f.path" class="upload-item" role="listitem">
              <div class="upload-main">
                <span class="upload-name">{{ f.name }}</span>
                <span class="upload-size">（{{ humanSize(f.size) }}）</span>
              </div>
              <div class="upload-actions">
                <button class="btn danger" @click="onDeleteUpload(f.path)">删除</button>
              </div>
            </div>
          </div>
          <div v-if="uploads && uploads.length" class="pager">
            <button class="pager-chip" :class="{ disabled: page===1 }" @click="toFirst" aria-disabled="page===1">首页</button>
            <button class="pager-chip" :class="{ disabled: page===1 }" @click="toPrev" aria-disabled="page===1">上一页</button>
            <span class="pager-chip" :class="{ active: true }">{{ page }}</span>
            <span class="pager-chip" v-if="totalPages>1">{{ Math.min(page+1, totalPages) }}</span>
            <button class="pager-chip" :class="{ disabled: page===totalPages }" @click="toNext" aria-disabled="page===totalPages">下一页</button>
            <button class="pager-chip" :class="{ disabled: page===totalPages }" @click="toLast" aria-disabled="page===totalPages">末页</button>
          </div>
          <div v-else class="uploads-empty">暂无上传数据</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, computed } from 'vue'

export default {
  name: '后台管理组件',
  setup() {
    const pipelineJobId = ref('')
    const pipelineStatus = ref('')
    const pipelineLogs = ref('')
    const stepStatuses = ref([])
    const stepNames = ref(['数据预处理','因果发现','多方法参数学习','贝叶斯中介分析','三角测量验证','知识图谱构建'])
    const showDetails = ref(false)
    const polling = ref(false)
    const uploadMessage = ref('')
    const uploadError = ref('')

    const uploads = ref([])
    const humanSize = (bytes) => {
      const kb = 1024, mb = kb * 1024
      if (bytes >= mb) return (bytes / mb).toFixed(2) + ' MB'
      if (bytes >= kb) return (bytes / kb).toFixed(2) + ' KB'
      return (bytes ?? 0) + ' B'
    }
    const refreshUploadsList = async () => {
      try {
        const res = await fetch('/api/datasource/list')
        const json = await res.json()
        const files = json?.data || []
        uploads.value = files.filter(f => String(f.path || '').includes('/07分离/uploads/'))
      } catch (_) {
        uploads.value = []
      }
    }
    const onDeleteUpload = async (path) => {
      try {
        const resp = await fetch('/api/datasource/delete', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path })
        })
        const j = await resp.json().catch(()=>({}))
        if (j && j.success === false) {
          uploadError.value = j.error || '删除失败'
        } else {
          uploadMessage.value = '已删除上传数据'
          await refreshUploadsList()
        }
      } catch (e) {
        uploadError.value = '删除失败'
      }
    }

    // 分页状态（每页5条）
    const page = ref(1)
    const pageSize = ref(5)
    const totalPages = computed(() => Math.max(1, Math.ceil((uploads.value?.length || 0) / pageSize.value)))
    const pagedUploads = computed(() => {
      const start = (page.value - 1) * pageSize.value
      return (uploads.value || []).slice(start, start + pageSize.value)
    })
    const clampPage = () => { page.value = Math.min(totalPages.value, Math.max(1, page.value)) }
    const toFirst = () => { page.value = 1 }
    const toPrev = () => { page.value = Math.max(1, page.value - 1) }
    const toNext = () => { page.value = Math.min(totalPages.value, page.value + 1) }
    const toLast = () => { page.value = totalPages.value }

    const toggleDetails = () => { showDetails.value = !showDetails.value }
    const stateClass = (s) => ({ waiting: 'waiting', running: 'running', succeeded: 'succeeded', failed: 'failed' }[String(s)] || 'waiting')
    const displayState = (s) => ({ waiting: '等待', running: '执行中', succeeded: '成功', failed: '失败' }[String(s)] || '等待')

    const loadJobId = () => {
      try {
        const fromSession = sessionStorage.getItem('pipeline_job_id') || ''
        pipelineJobId.value = fromSession
      } catch (_) {}
    }
    const saveJobId = () => {}

    const stopPolling = () => { polling.value = false }

    const startPolling = async () => {
      if (!pipelineJobId.value) return
      uploadMessage.value = ''
      uploadError.value = ''
      polling.value = true
      try {
        for (;;) {
          if (!polling.value) break
          const res = await fetch('/api/pipeline/status?job_id=' + encodeURIComponent(pipelineJobId.value))
          const j = await res.json()
          if (!j?.success) { uploadError.value = j?.error || '状态获取失败'; polling.value = false; break }
          const d = j.data || {}
          pipelineStatus.value = d.status || ''
          stepStatuses.value = Array.isArray(d.step_statuses) ? d.step_statuses : []
          const rl = await fetch('/api/pipeline/logs?job_id=' + encodeURIComponent(pipelineJobId.value))
          const jl = await rl.json()
          if (jl?.success) pipelineLogs.value = jl.data || ''
          if (pipelineStatus.value === 'succeeded') { uploadMessage.value = '处理完成'; polling.value = false; break }
          if (pipelineStatus.value === 'failed') { uploadError.value = '处理失败'; polling.value = false; break }
          await new Promise(r => setTimeout(r, 2000))
        }
      } catch (e) {
        uploadError.value = '轮询失败'
        polling.value = false
      }
    }

    onMounted(() => {
      loadJobId()
      if (pipelineJobId.value) startPolling()
      refreshUploadsList()
      clampPage()
    })

    return {
      pipelineJobId,
      pipelineStatus,
      pipelineLogs,
      stepStatuses,
      stepNames,
      showDetails,
      polling,
      uploadMessage,
      uploadError,
      uploads,
      humanSize,
      page,
      pageSize,
      totalPages,
      pagedUploads,
      toFirst,
      toPrev,
      toNext,
      toLast,
      refreshUploadsList,
      onDeleteUpload,
      toggleDetails,
      stateClass,
      displayState,
      startPolling,
      stopPolling,
      saveJobId
    }
  }
}
</script>

<style scoped>
.admin-wrap { display: flex; height: 100%; }
.panel { display: flex; flex-direction: column; width: 100%; background: #ffffff; border: 1px solid #e9ecef; border-radius: 12px; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06); }
.toolbar { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 16px 20px; border-bottom: 1px solid #f1f3f5; background: linear-gradient(180deg, #ffffff, #fbfbfb); }
.title { font-size: 16px; font-weight: 600; color: #2c3e50; }
.actions { display: flex; gap: 8px; align-items: center; }
.btn { padding: 8px 14px; border: 1px solid #dee2e6; border-radius: 8px; background: #fff; color: #34495e; font-size: 13px; cursor: pointer; }
.btn.primary { border-color: #3b82f6; color: #fff; background: #3b82f6; }
.content { padding: 14px; }
.hint { margin-top: 8px; font-size: 12px; color: #6b7280; }
.hint.error { color: #ef4444; }
.hint.ok { color: #10b981; }
.progress-wrap { margin-top: 10px; border: 1px solid #e5e7eb; border-radius: 10px; padding: 10px; }
.progress-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.progress-list { list-style: none; padding: 0; margin: 0; display: grid; grid-template-columns: 1fr; gap: 6px; }
.progress-list li { display: flex; justify-content: space-between; align-items: center; padding: 6px 8px; border-radius: 8px; background: #f8fafc; }
.step-name { color: #334155; }
.step-state { font-weight: 600; }
.step-state.waiting { color: #64748b; }
.step-state.running { color: #2563eb; }
.step-state.succeeded { color: #10b981; }
.step-state.failed { color: #ef4444; }
.guide.logs { max-height: 260px; overflow: auto; white-space: pre-wrap; font-family: ui-monospace, Menlo, Monaco, Consolas, 'Liberation Mono', monospace; margin-top: 8px; }

.uploads-section { margin-top: 14px; border: 1px solid #e5e7eb; border-radius: 12px; background: #fff; box-shadow: 0 8px 20px rgba(0,0,0,0.06); }
.uploads-header { display: flex; align-items: center; justify-content: space-between; padding: 10px 12px; border-bottom: 1px solid #f1f5f9; }
.uploads-header .title { font-size: 14px; font-weight: 600; color: #1f2937; }
.uploads-list { display: grid; grid-template-columns: 1fr; }
.upload-item { display: flex; align-items: center; justify-content: space-between; padding: 10px 12px; border-top: 1px solid #f1f5f9; }
.upload-item:hover { background: #f8fafc; }
.upload-main { display: flex; align-items: center; gap: 8px; color: #334155; }
.upload-name { font-weight: 600; }
.upload-size { font-size: 12px; color: #64748b; }
.upload-actions .btn.danger { border-color: #ef4444; background: #ef4444; color: #fff; }
.uploads-empty { padding: 12px; color: #64748b; font-size: 13px; }

/* 简易分页样式 */
.pager { display: flex; align-items: center; gap: 8px; padding: 8px 12px; border-top: 1px solid #f1f5f9; }
.pager-chip { padding: 6px 10px; border: 1px solid #e5e7eb; border-radius: 8px; background: #fff; color: #334155; font-size: 12px; }
.pager-chip.active { background: #f8fafc; font-weight: 600; }
.pager-chip.disabled { opacity: .6; pointer-events: none; }
</style>
