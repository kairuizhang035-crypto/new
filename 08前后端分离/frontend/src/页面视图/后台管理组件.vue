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

        <div class="datasource-panel">
          <div class="ds-header">
            <div class="ds-title">数据源</div>
            <div class="sidebar-actions">
              <button class="ds-btn sm" @click="refreshDatasourceList">⟲ 刷新</button>
            </div>
          </div>
          <div class="ds-list">
            <div id="ds-menu" class="ds-dropdown-menu" role="menu">
              <div v-for="f in datasourceFilesDedup" :key="f.path" class="ds-dropdown-item">
                <div class="ds-item-select">
                  <span class="name">{{ f.name }}</span>
                  <span v-if="isCurrent(f.path)" class="current-badge">✓ 当前</span>
                  <span class="size">（{{ fmtSize(f.size) }}）</span>
                </div>
                <div class="ds-item-actions">
                  <button class="ds-item-apply" :class="{ disabled: isCurrent(f.path) }" :disabled="isCurrent(f.path)" @click="onApplyFromDropdown(f.path)">应用</button>
                  <button class="ds-item-delete" v-if="isUploadPath(f.path)" @click="onRequestDeleteDatasource(f)">删除</button>
                </div>
              </div>
              <div v-if="!datasourceFilesDedup || !datasourceFilesDedup.length" class="ds-empty">暂无数据源</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { useKnowledgeGraphStore } from '../状态管理/知识图谱状态'
import { storeToRefs } from 'pinia'

export default {
  name: '后台管理组件',
  setup() {
    const store = useKnowledgeGraphStore()
    const pipelineJobId = ref('')
    const pipelineStatus = ref('')
    const pipelineLogs = ref('')
    const stepStatuses = ref([])
    const stepNames = ref(['数据预处理','因果发现','多方法参数学习','贝叶斯中介分析','三角测量验证','知识图谱构建'])
    const showDetails = ref(false)
    const polling = ref(false)
    const uploadMessage = ref('')
    const uploadError = ref('')

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

    const sRefs = storeToRefs(store)
    const datasourceFiles = sRefs.datasourceFiles ?? ref([])
    const currentDatasource = sRefs.currentDatasource ?? ref({})
    const fmtSize = (s) => {
      if (!s && s !== 0) return '未知'
      const kb = s / 1024
      if (kb < 1024) return `${kb.toFixed(1)} KB`
      return `${(kb/1024).toFixed(1)} MB`
    }
    const isUploadPath = (p) => String(p || '').includes('/07分离/uploads/')
    const datasourceFilesDedup = computed(() => {
      const seen = new Set()
      const out = []
      for (const f of (datasourceFiles?.value || [])) {
        if (!f) continue
        const key = `${f.name}|${f.size}`
        if (seen.has(key)) continue
        seen.add(key)
        out.push(f)
      }
      return out
    })
    const selectedDatasourcePath = ref('')
    const isCurrent = (p) => String(p || '') === String(((currentDatasource?.value && currentDatasource.value.path) || ''))
    const onSelectDatasource = (path) => { selectedDatasourcePath.value = path }
    const onApplyFromDropdown = async (path) => {
      selectedDatasourcePath.value = path
      if (!selectedDatasourcePath.value) return
      await store.selectDatasource(selectedDatasourcePath.value)
      await loadCurrentDatasource()
      await refreshDatasourceList()
    }
    const onRequestDeleteDatasource = (f) => { onDeleteDatasource(f?.path) }
    const onDeleteDatasource = async (path) => {
      try {
        await fetch('/api/datasource/delete', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path }) })
        await refreshDatasourceList()
        await loadCurrentDatasource()
        if (selectedDatasourcePath.value === path) selectedDatasourcePath.value = ''
      } catch (e) {}
    }

    const refreshDatasourceList = async () => {
      try {
        const files = await store.listDatasources()
        datasourceFiles.value = files || []
      } catch (e) {}
    }
    const loadCurrentDatasource = async () => {
      try {
        const cur = await store.getCurrentDatasource()
        currentDatasource.value = cur || {}
      } catch (e) {}
    }

    onMounted(async () => {
      loadJobId()
      if (pipelineJobId.value) startPolling()
      await loadCurrentDatasource()
      await refreshDatasourceList()
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
      toggleDetails,
      stateClass,
      displayState,
      startPolling,
      stopPolling,
      saveJobId,
      datasourceFiles,
      currentDatasource,
      datasourceFilesDedup,
      fmtSize,
      isUploadPath,
      onSelectDatasource,
      onApplyFromDropdown,
      onRequestDeleteDatasource,
      refreshDatasourceList,
      loadCurrentDatasource,
      isCurrent
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
.hint { margin-top: 8px; font-size: 12px; color: #6b7280; padding: 10px 12px; background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 10px; }
.hint.error { color: #ef4444; }
.hint.ok { color: #10b981; }
.progress-wrap { margin-top: 12px; border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; background: #ffffff; box-shadow: 0 6px 18px rgba(0,0,0,0.04); }
.progress-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.progress-list { list-style: none; padding: 0; margin: 0; display: grid; grid-template-columns: 1fr; gap: 6px; }
.progress-list li { display: flex; justify-content: space-between; align-items: center; padding: 8px 10px; border-radius: 10px; background: #F8FAFC; border: 1px solid #EEF2F7; }
.step-name { color: #334155; }
.step-state { font-weight: 600; }
.step-state.waiting { color: #64748b; }
.step-state.running { color: #2563eb; }
.step-state.succeeded { color: #10b981; }
.step-state.failed { color: #ef4444; }
.guide.logs { max-height: 260px; overflow: auto; white-space: pre-wrap; font-family: ui-monospace, Menlo, Monaco, Consolas, 'Liberation Mono', monospace; margin-top: 8px; background: #0B1020; color: #E6EDF3; padding: 10px 12px; border-radius: 10px; }
</style>
