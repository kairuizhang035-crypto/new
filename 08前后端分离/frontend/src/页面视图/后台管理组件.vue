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
        <div class="hint" v-if="!pipelineJobId">未检测到 Job ID，请在上传页启动管道。</div>
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
          <div class="ds-row">
            <div class="ds-dropdown">
              <button 
                class="ds-dropdown-toggle" 
                :class="{ open: dsOpen }" 
                @click.prevent="dsOpen=!dsOpen"
                :aria-expanded="dsOpen ? 'true' : 'false'"
                aria-haspopup="menu"
                aria-controls="ds-menu"
                aria-label="选择数据源"
              >
                <span class="ds-name">{{ selectedLabel }}</span>
                <span class="ds-caret">▾</span>
              </button>
              <div v-if="dsOpen" id="ds-menu" class="ds-dropdown-menu" role="menu">
                <div v-for="f in datasourceFilesDedup" :key="f.path" class="ds-dropdown-item">
                  <button class="ds-item-select" role="menuitem" @click="onSelectDatasource(f.path)">
                    <span class="name">{{ f.name }}</span>
                    <span v-if="isCurrent(f.path)" class="current-badge">✓ 当前</span>
                    <span class="size">（{{ fmtSize(f.size) }}）</span>
                  </button>
                  <div class="ds-item-actions">
                    <button class="ds-item-apply" role="menuitem" :aria-disabled="isCurrent(f.path) ? 'true' : 'false'" :class="{ disabled: isCurrent(f.path) }" :disabled="isCurrent(f.path)" @click.stop="onApplyFromDropdown(f.path)">应用</button>
                    <button class="ds-item-delete" v-if="isUploadPath(f.path)" @click.stop="onRequestDeleteDatasource(f)">删除</button>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div v-if="dsConfirmVisible" class="ds-confirm">
            <div class="ds-confirm-text">确定删除该数据源文件吗？此操作不可恢复。</div>
            <div class="ds-confirm-name">{{ dsConfirmName }}</div>
            <div class="ds-confirm-actions">
              <button class="ds-btn sm" @click="onCancelDeleteDatasource">取消</button>
              <button class="ds-btn sm" @click="onConfirmDeleteDatasource">确认</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted, watch } from 'vue'
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
          if (pipelineStatus.value === 'succeeded') { polling.value = false; break }
          if (pipelineStatus.value === 'failed') { uploadError.value = '处理失败'; polling.value = false; break }
          await new Promise(r => setTimeout(r, 2000))
        }
      } catch (e) {
        uploadError.value = '轮询失败'
        polling.value = false
      }
    }

    // 数据源管理（共享 Pinia 状态）
    const { datasourceFiles, currentDatasource } = storeToRefs(store)
    const datasourceFilesDedup = computed(() => {
      const seen = new Set()
      const out = []
      for (const f of datasourceFiles.value || []) {
        if (!f) continue
        const key = `${f.name}|${f.size}`
        if (seen.has(key)) continue
        seen.add(key)
        out.push(f)
      }
      return out
    })
    const selectedDatasourcePath = ref('')

    const fmtSize = (s) => {
      if (!s && s !== 0) return '未知'
      const kb = s / 1024
      if (kb < 1024) return `${kb.toFixed(1)} KB`
      return `${(kb/1024).toFixed(1)} MB`
    }

    const refreshDatasourceList = async () => {
      try {
        await store.refreshDatasources()
        if (!selectedDatasourcePath.value && (datasourceFiles.value || []).length) {
          selectedDatasourcePath.value = datasourceFiles.value[0].path
        }
      } catch (e) {}
    }

    const loadCurrentDatasource = async () => {
      try {
        await store.refreshDatasources()
      } catch (e) {}
    }

    const applySelectedDatasource = async () => {
      if (!selectedDatasourcePath.value) return
      await store.selectDatasource(selectedDatasourcePath.value)
      await loadCurrentDatasource()
    }

    const onSelectDatasource = (path) => {
      selectedDatasourcePath.value = path
      dsOpen.value = false
    }

    const onApplyFromDropdown = async (path) => {
      selectedDatasourcePath.value = path
      await applySelectedDatasource()
      dsOpen.value = false
    }

    const isUploadPath = (p) => String(p || '').includes('/07分离/uploads/')
    const onDeleteDatasource = async (path) => {
      try {
        await fetch('/api/datasource/delete', {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path })
        })
        await store.refreshDatasources()
        if (selectedDatasourcePath.value === path) {
          selectedDatasourcePath.value = ''
        }
      } catch (e) {}
    }

    const dsOpen = ref(false)
    const selectedLabel = computed(() => {
      const curPath = (currentDatasource.value && currentDatasource.value.path) ? currentDatasource.value.path : ''
      if (curPath) {
        const f = (datasourceFilesDedup.value || []).find(x => x.path === curPath)
        if (f) return `${f.name}（${fmtSize(f.size)}）`
        const name = curPath.split('/').pop()
        return name || curPath
      }
      const first = (datasourceFilesDedup.value || [])[0]
      if (first) return `${first.name}（${fmtSize(first.size)}）`
      return '暂无数据源'
    })
    const isCurrent = (p) => String(p || '') === String((currentDatasource.value && currentDatasource.value.path) || '')

    const dsConfirmVisible = ref(false)
    const dsConfirmPath = ref('')
    const dsConfirmName = ref('')
    const onRequestDeleteDatasource = (f) => {
      dsConfirmPath.value = f?.path || ''
      dsConfirmName.value = f?.name || ''
      dsConfirmVisible.value = true
      dsOpen.value = false
    }
    const onCancelDeleteDatasource = () => { dsConfirmVisible.value = false }
    const onConfirmDeleteDatasource = async () => {
      const p = dsConfirmPath.value
      dsConfirmVisible.value = false
      if (!p) return
      await onDeleteDatasource(p)
      dsOpen.value = false
    }

    const pageSizeDS = ref(5)
    const currentPageDS = ref(1)
    const totalPagesDS = computed(() => Math.max(1, Math.ceil((datasourceFilesDedup.value || []).length / pageSizeDS.value)))
    const pagedDatasources = computed(() => {
      const start = (currentPageDS.value - 1) * pageSizeDS.value
      return (datasourceFilesDedup.value || []).slice(start, start + pageSizeDS.value)
    })
    const pageNumbersDS = computed(() => {
      const total = totalPagesDS.value
      const current = currentPageDS.value
      const WINDOW = 5
      let start = current - Math.floor(WINDOW / 2)
      if (start < 1) start = 1
      let end = start + WINDOW - 1
      if (end > total) { end = total; start = Math.max(1, end - WINDOW + 1) }
      const res = []
      for (let p = start; p <= end; p++) res.push(p)
      return res
    })
    const goToPageDS = (p) => {
      const n = Number(p)
      if (!Number.isFinite(n)) return
      if (n < 1 || n > totalPagesDS.value) return
      currentPageDS.value = n
    }
    const prevPageDS = () => { if (currentPageDS.value > 1) currentPageDS.value -= 1 }
    const nextPageDS = () => { if (currentPageDS.value < totalPagesDS.value) currentPageDS.value += 1 }

    onMounted(async () => {
      loadJobId()
      if (pipelineJobId.value) startPolling()
      await store.refreshDatasources()
      if (!selectedDatasourcePath.value && currentDatasource.value && currentDatasource.value.path) {
        selectedDatasourcePath.value = currentDatasource.value.path
      }
    })

    watch(datasourceFilesDedup, () => {
      const max = totalPagesDS.value
      if (currentPageDS.value > max) currentPageDS.value = max
      if (currentPageDS.value < 1) currentPageDS.value = 1
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
      datasourceFilesDedup,
      selectedDatasourcePath,
      currentDatasource,
      fmtSize,
      refreshDatasourceList,
      applySelectedDatasource,
      onSelectDatasource,
      onDeleteDatasource,
      isUploadPath,
      dsOpen,
      selectedLabel,
      dsConfirmVisible,
      dsConfirmName,
      onRequestDeleteDatasource,
      onCancelDeleteDatasource,
      onConfirmDeleteDatasource,
      onApplyFromDropdown,
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

.datasource-panel { margin-top: 16px; padding: 12px; border: 1px solid #e5e7eb; border-radius: 12px; background: #ffffff; }
.ds-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.ds-title { font-size: 13px; font-weight: 600; color: #374151; }
.sidebar-actions { display: flex; gap: 8px; }
.ds-row { display: flex; align-items: center; gap: 8px; }
.ds-dropdown { position: relative; flex: 1; }
.ds-dropdown-toggle { width: 100%; min-height: 40px; padding: 10px 12px; border: 1px solid #dee2e6; background: #fff; color: #374151; border-radius: 10px; text-align: left; display: flex; align-items: center; justify-content: space-between; transition: box-shadow .2s ease, border-color .2s ease; }
.ds-dropdown-toggle:hover { box-shadow: 0 6px 18px rgba(0,0,0,0.12); border-color: rgba(0,0,0,0.18); }
.ds-dropdown-toggle.open { box-shadow: 0 10px 24px rgba(0,0,0,0.18); }
.ds-caret { opacity: 0.9; transition: transform .16s ease; }
.ds-dropdown-toggle.open .ds-caret { transform: rotate(180deg); }
.ds-dropdown-menu { position: absolute; top: calc(100% + 6px); left: 0; right: 0; background: #ffffff; color: #111827; border-radius: 12px; box-shadow: 0 14px 28px rgba(0,0,0,0.14); padding: 8px; z-index: 10; max-height: 280px; overflow: auto; border: 1px solid #e5e7eb; }
.ds-dropdown-item { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 10px 12px; border-radius: 10px; }
.ds-dropdown-item:hover { background: #f5f7fb; }
.ds-btn { padding: 6px 10px; border: 1px solid #dee2e6; background: #fff; color: #34495e; border-radius: 8px; cursor: pointer; }
.ds-btn.sm { padding: 4px 8px; font-size: 12px; }
.ds-actions { display: flex; gap: 8px; }
.current-badge { margin-left: 8px; color: #16a34a; font-weight: 600; padding: 2px 8px; border-radius: 999px; border: 1px solid #86efac; background: #ecfdf5; font-size: 12px; }
.ds-item-apply { padding: 6px 10px; border: 1px solid rgba(59,130,246,0.8); background: rgba(59,130,246,0.08); color: #1d4ed8; border-radius: 8px; cursor: pointer; }
.ds-item-apply.disabled { opacity: 0.6; cursor: not-allowed; }
.ds-item-delete { padding: 6px 10px; border: 1px solid rgba(255, 80, 80, 0.8); background: rgba(255, 80, 80, 0.12); color: #b91c1c; border-radius: 8px; cursor: pointer; }
.ds-empty { padding: 10px; color: #6b7280; font-size: 13px; text-align: center; border: 1px dashed #e5e7eb; border-radius: 10px; }
.ds-confirm { margin-top: 10px; padding: 10px 12px; border-radius: 12px; background: #f8fafc; border: 1px solid #e5e7eb; }
.ds-confirm-text { font-size: 13px; color: #374151; margin-bottom: 6px; }
.ds-confirm-name { font-size: 12px; color: #6b7280; margin-bottom: 8px; }
.ds-confirm-actions { display: flex; gap: 8px; }
</style>
