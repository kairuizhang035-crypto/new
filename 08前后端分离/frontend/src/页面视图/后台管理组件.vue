<template>
  <div class="admin-wrap">
    <div class="panel">
      <div class="toolbar">
        <div class="title">后台处理进度</div>
        <div class="actions">
          <input class="job-input" v-model="pipelineJobId" placeholder="输入 Job ID" />
          <button class="btn" @click="saveJobId">保存</button>
          <button class="btn primary" :disabled="polling" @click="startPolling">开始轮询</button>
          <button class="btn" :disabled="!polling" @click="stopPolling">停止</button>
        </div>
      </div>
      <div class="content">
        <div class="hint" v-if="!pipelineJobId">未检测到 Job ID，请在上方输入或从上传页启动管道。</div>
        <div class="hint ok" v-if="uploadMessage">{{ uploadMessage }}</div>
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
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'

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

    const toggleDetails = () => { showDetails.value = !showDetails.value }
    const stateClass = (s) => ({ waiting: 'waiting', running: 'running', succeeded: 'succeeded', failed: 'failed' }[String(s)] || 'waiting')
    const displayState = (s) => ({ waiting: '等待', running: '执行中', succeeded: '成功', failed: '失败' }[String(s)] || '等待')

    const loadJobId = () => {
      try { pipelineJobId.value = sessionStorage.getItem('pipeline_job_id') || '' } catch (_) {}
    }
    const saveJobId = () => {
      try { sessionStorage.setItem('pipeline_job_id', pipelineJobId.value || '') } catch (_) {}
    }

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

    onMounted(() => { loadJobId() })

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
.job-input { width: 220px; padding: 8px 10px; border: 1px solid #dee2e6; border-radius: 8px; font-size: 13px; }
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
</style>
