<template>
  <div class="landing">
    <!-- 右上角用户信息面板 -->
    <div class="user-panel-wrap" :class="{ open: userPanelOpen }">
      <div class="user-trigger" @click="toggleUserPanel" :aria-expanded="userPanelOpen ? 'true' : 'false'" role="button">
        <div class="avatar">👤</div>
        <div class="user-basic">
          <div class="name">{{ userName }}</div>
          <div class="meta">{{ userEmail }}</div>
        </div>
        <span class="logout-inline" @click.stop="onLogout" aria-label="退出登录">退出</span>
      </div>
      <div class="user-panel" role="region" aria-label="用户信息面板">
        <div class="up-row">
          <div class="avatar lg">👤</div>
          <div class="up-basic">
            <div class="up-name">{{ userName }}</div>
            <div class="up-meta">{{ userEmail }}</div>
          </div>
        </div>
        <div class="up-actions">
          <button class="btn danger" @click="onLogout">退出登录</button>
        </div>
      </div>
    </div>
    <div class="layout">
      <!-- 左侧：上传与预览 -->
      <div class="card upload-card">
        <div class="card-header">
          <h1 class="title">数据上传与轻量预览</h1>
          <p class="desc">支持拖拽或选择文件，快速预览 CSV/JSON 前几行，随后进入增强知识图谱可视化</p>
        </div>

        <div
          class="drop-zone"
          :class="{ active: dropActive }"
          @dragover.prevent="onDragOver"
          @dragleave.prevent="onDragLeave"
          @drop.prevent="onDrop"
        >
          <input
            ref="fileInput"
            class="file-input"
            type="file"
            accept=".csv,application/json"
            @change="onFileChange"
          />
          <div class="dz-content">
            <div class="dz-icon">📄</div>
            <div class="dz-text">
              <div class="dz-title">拖拽文件到此区域，或</div>
              <button class="btn primary" @click="chooseFile">选择文件</button>
            </div>
            <div class="dz-hint">支持格式：CSV、JSON（本地解析，不上传服务器）</div>
          </div>
        </div>

        <!-- 文件摘要与预览 -->
        <div v-if="hasPreview" class="preview">
          <div class="preview-header">
            <div class="file-meta">
              <span class="meta-item"><strong>文件名:</strong> {{ preview.name }}</span>
              <span class="meta-item"><strong>类型:</strong> {{ preview.type }}</span>
              <span class="meta-item"><strong>大小:</strong> {{ humanSize(preview.size) }}</span>
            </div>
          <div class="preview-actions">
              <button class="btn" @click="clear">清空</button>
              <button class="btn success" :disabled="uploading" @click="goToGraph">上传数据并进入增强知识图谱</button>
          </div>
          <div v-if="uploadError" class="hint error">{{ uploadError }}</div>
          </div>

          <div class="preview-illustration" aria-label="解析摘要">
            <div class="illus-icon">{{ fileIcon }}</div>
            <div class="illus-content">
              <div class="illus-title">文件已解析</div>
              <div class="illus-desc">可直接上传进入增强知识图谱</div>
              <div class="illus-badges">
                <span class="badge">{{ preview.type }}</span>
                <span class="badge">{{ humanSize(preview.size) }}</span>
              </div>
              <div class="illus-chips">
                <span class="chip">本地解析</span>
                <span class="chip">安全</span>
                <span class="chip">快速</span>
              </div>
            </div>
          </div>

          <div class="preview-tips">
            <div class="tip">
              <div class="tip-icon">⬆️</div>
              <div class="tip-text">拖拽上传或点击按钮</div>
            </div>
            <div class="tip">
              <div class="tip-icon">🔒</div>
              <div class="tip-text">本地解析，数据不上传</div>
            </div>
            <div class="tip">
              <div class="tip-icon">🚀</div>
              <div class="tip-text">一键进入增强知识图谱</div>
            </div>
            <div class="tip">
              <div class="tip-icon">🧠</div>
              <div class="tip-text">智能摘要与因果分析</div>
            </div>
          </div>

        </div>
      </div>

      <!-- 右侧：说明与亮点 -->
      <div class="card info-card">
        <h2 class="sub-title">流程步骤</h2>
        <ul class="steps">
          <li><span class="emoji">🧹</span> 数据预处理（缺失值、标准化、编码）</li>
          <li><span class="emoji">🔍</span> 因果结构发现（PC/爬山/贪婪等价/TAN/专家在循环）</li>
          <li><span class="emoji">📈</span> 参数学习（MLE / Bayesian / EM / SEM）</li>
          <li><span class="emoji">📊</span> 条件概率表（CPT）预览与一致性分析</li>
          <li><span class="emoji">🔺</span> 证据三角验证与分层网络</li>
          <li><span class="emoji">🌐</span> 增强知识图谱可视化（V, E_core, R, W, Θ, Φ）</li>
        </ul>

        <h2 class="sub-title">功能亮点</h2>
        <div class="chips">
          <span class="chip">拖拽上传</span>
          <span class="chip">本地解析</span>
          <span class="chip">CSV/JSON</span>
          <span class="chip">轻量预览</span>
          <span class="chip">会话续航</span>
          <span class="chip">一键进入图谱</span>
        </div>

        <div class="guide">
          <p>上传后会在本地解析数据，生成预览与摘要。点击“进入增强知识图谱”即可继续探索，摘要将临时保存到浏览器会话中。</p>
          <div class="btn-row">
            <button class="btn primary" @click="goToGraph">立即进入图谱</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'

export default {
  name: '上传入口页面',
  setup() {
    const fileInput = ref(null)
    const dropActive = ref(false)
    const selectedFile = ref(null)
    const preview = ref({
      name: '',
      type: '',
      size: 0,
      headers: [],
      rows: [],
      rowsCount: 0,
      columns: 0
    })
    const uploading = ref(false)
    const uploadError = ref('')
    const uploadMessage = ref('')
    const fileIcon = computed(() => (String(preview.value.type).toUpperCase() === 'CSV' ? '📑' : '🧩'))

    const hasPreview = computed(() => !!(preview.value && preview.value.name))

    // 用户信息面板状态与数据（集成到脚本内）
    const userPanelOpen = ref(false)
    const user = ref({ name: '', email: '', avatar_url: '' })
    const userName = computed(() => user.value.name || user.value.username || '已登录用户')
    const userEmail = computed(() => user.value.email || '')
    const userAvatar = computed(() => user.value.avatar_url || 'https://api.dicebear.com/7.x/initials/svg?seed=' + encodeURIComponent(userName.value))
    const toggleUserPanel = () => { userPanelOpen.value = !userPanelOpen.value }

    const fetchMe = async () => {
      try {
        const res = await fetch('/api/auth/me')
        const j = await res.json()
        const candidate = j?.user || j || {}
        const fromSession = (() => {
          try { return JSON.parse(sessionStorage.getItem('current_user') || '{}') } catch (_) { return {} }
        })()
        user.value = {
          name: candidate.name || candidate.username || fromSession.name || fromSession.username || '',
          email: candidate.email || fromSession.email || '',
          avatar_url: candidate.avatar_url || fromSession.avatar_url || ''
        }
      } catch (e) {
        try {
          const u = JSON.parse(sessionStorage.getItem('current_user') || '{}')
          user.value = {
            name: u.name || u.username || '',
            email: u.email || '',
            avatar_url: u.avatar_url || ''
          }
        } catch (_) {}
      }
    }

    const clearClientSession = () => {
      try { localStorage.clear() } catch (e) {}
      try { sessionStorage.clear() } catch (e) {}
      try { if ('caches' in window) { caches.keys().then(keys => keys.forEach(k => caches.delete(k))).catch(()=>{}) } } catch (e) {}
    }
    const onLogout = async () => {
      clearClientSession()
      try { await fetch('/api/auth/logout', { method: 'POST' }) } catch (e) {}
      userPanelOpen.value = false
      const origin = encodeURIComponent(window.location.href)
      if (window?.$vueRouter) {
        window.$vueRouter.replace({ path: '/login', query: { redirect: '/upload', origin } })
      } else {
        window.location.assign('/login?redirect=/upload&origin=' + origin)
      }
    }

    const chooseFile = () => {
      try { fileInput.value && fileInput.value.click() } catch (e) {}
    }
    const onFileChange = async (e) => {
      const file = e?.target?.files?.[0]
      if (file) {
        selectedFile.value = file
        parseFile(file)
      }
    }
    const onDragOver = () => { dropActive.value = true }
    const onDragLeave = () => { dropActive.value = false }
    const onDrop = async (e) => {
      dropActive.value = false
      const file = e?.dataTransfer?.files?.[0]
      if (file) {
        selectedFile.value = file
        parseFile(file)
      }
    }

    const humanSize = (bytes) => {
      const kb = 1024, mb = kb * 1024
      if (bytes >= mb) return (bytes / mb).toFixed(2) + ' MB'
      if (bytes >= kb) return (bytes / kb).toFixed(2) + ' KB'
      return bytes + ' B'
    }

    const parseFile = (file) => {
      const type = (file.type || '').toLowerCase()
      const name = file.name || '未命名'
      const reader = new FileReader()
      reader.onload = () => {
        const text = reader.result
        try {
          if (name.endsWith('.csv') || type.includes('csv') || /text\/plain/.test(type)) {
            const { headers, rows } = parseCSV(String(text || ''))
            applyPreview(name, 'CSV', file.size, headers, rows)
          } else {
            const obj = JSON.parse(String(text || '{}'))
            const { headers, rows } = normalizeJSON(obj)
            applyPreview(name, 'JSON', file.size, headers, rows)
          }
        } catch (err) {
          console.warn('解析失败', err)
          preview.value = { name, type: type || '未知', size: file.size, headers: [], rows: [], rowsCount: 0, columns: 0 }
          uploadMessage.value = ''
          uploadError.value = '解析失败，请检查文件格式或内容'
        }
      }
      reader.readAsText(file)
    }

    const normalizeJSON = (obj) => {
      let rows = []
      let headers = []
      if (Array.isArray(obj)) {
        if (obj.length && typeof obj[0] === 'object' && !Array.isArray(obj[0])) {
          const sample = obj.slice(0, 50)
          const set = new Set()
          sample.forEach(r => Object.keys(r || {}).forEach(k => set.add(k)))
          headers = Array.from(set)
          rows = obj.slice(0, 12)
        } else if (obj.length && Array.isArray(obj[0])) {
          headers = obj[0].map((_, i) => `列${i+1}`)
          rows = obj.slice(1, 13).map(arr => {
            const o = {}
            headers.forEach((h, i) => { o[h] = arr[i] })
            return o
          })
        }
      } else if (typeof obj === 'object') {
        const arr = Array.isArray(obj.data) ? obj.data : []
        if (arr.length && typeof arr[0] === 'object') {
          const set = new Set()
          arr.slice(0, 50).forEach(r => Object.keys(r || {}).forEach(k => set.add(k)))
          headers = Array.from(set)
          rows = arr.slice(0, 12)
        } else {
          headers = Object.keys(obj || {})
          rows = [obj]
        }
      }
      return { headers, rows }
    }

    const parseCSV = (text) => {
      const lines = String(text || '').replace(/\r/g, '').split(/\n/).filter(l => l.trim().length)
      if (!lines.length) return { headers: [], rows: [] }
      const parseLine = (line) => {
        const out = []
        let cur = ''
        let q = false
        for (let i = 0; i < line.length; i++) {
          const ch = line[i]
          if (ch === '"') {
            if (q && line[i+1] === '"') { cur += '"'; i++ } else { q = !q }
          } else if (ch === ',' && !q) {
            out.push(cur)
            cur = ''
          } else {
            cur += ch
          }
        }
        out.push(cur)
        return out.map(s => s.trim())
      }
      const headerArr = parseLine(lines[0])
      const headers = headerArr.map(h => h || '列')
      const rows = []
      for (let i = 1; i < Math.min(lines.length, 13); i++) {
        const vals = parseLine(lines[i])
        const row = {}
        headers.forEach((h, idx) => { row[h] = vals[idx] })
        rows.push(row)
      }
      return { headers, rows }
    }

    const applyPreview = (name, type, size, headers, rows) => {
      preview.value = {
        name,
        type,
        size,
        headers: headers || [],
        rows: rows || [],
        rowsCount: Math.max(0, (rows || []).length),
        columns: Math.max(0, (headers || []).length)
      }
      try {
        const summary = {
          name, type, size,
          headers: preview.value.headers,
          rowsCount: preview.value.rowsCount,
          columns: preview.value.columns
        }
        sessionStorage.setItem('upload_summary', JSON.stringify(summary))
      } catch (e) {}
      uploadError.value = ''
      uploadMessage.value = ''
    }

    const formatCell = (val) => {
      if (val === null || val === undefined || val === '') return '—'
      if (typeof val === 'number') return Number.isFinite(val) ? val.toString() : String(val)
      return String(val)
    }

    const clear = () => {
      preview.value = { name: '', type: '', size: 0, headers: [], rows: [], rowsCount: 0, columns: 0 }
      try { if (fileInput.value) fileInput.value.value = '' } catch (e) {}
      try { sessionStorage.removeItem('upload_summary') } catch (e) {}
      uploadError.value = ''
      uploadMessage.value = ''
    }

    const listDatasources = async () => {
      try {
        const res = await fetch('/api/datasource/list')
        const json = await res.json()
        return json?.data || []
      } catch (_) { return [] }
    }
    const selectDatasource = async (path) => {
      try {
        await fetch('/api/datasource/select', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path })
        })
      } catch (_) {}
    }
    const uploadDatasource = async (file, autoSelect) => {
      try {
        const form = new FormData()
        form.append('file', file)
        form.append('select', autoSelect ? 'true' : 'false')
        const res = await fetch('/api/datasource/upload', { method: 'POST', body: form })
        const json = await res.json()
        return json || { success: false, error: '网络错误' }
      } catch (_) { return { success: false, error: '网络错误' } }
    }

    const goToGraph = async () => {
      // 数据上传页必须拥有有效JWT
      try {
        const resAuth = await fetch('/api/auth/me')
        const j = await resAuth.json()
        if (!j?.authenticated) {
          const origin = encodeURIComponent(window.location.href)
          if (window?.$vueRouter) {
            return window.$vueRouter.replace({ path: '/login', query: { redirect: '/upload', origin } })
          }
          return (window.location.href = '/login?redirect=/upload&origin=' + origin)
        }
      } catch (e) {
        const origin = encodeURIComponent(window.location.href)
        if (window?.$vueRouter) {
          return window.$vueRouter.replace({ path: '/login', query: { redirect: '/upload', origin } })
        }
        return (window.location.href = '/login?redirect=/upload&origin=' + origin)
      }
      uploadError.value = ''
      uploadMessage.value = ''
      try {
        if (selectedFile.value) {
          uploading.value = true
          const name = (selectedFile.value.name || '').toLowerCase()
          const isCSV = name.endsWith('.csv')
          const isJSON = name.endsWith('.json')
          const resp = await uploadDatasource(selectedFile.value, isJSON)
          if (!resp?.success) {
            uploading.value = false
            uploadError.value = resp?.error || '上传失败'
            return
          }
          const data = resp?.data || {}
          if (isJSON) {
            const saved = data?.selected?.path || data?.saved_path
            if (saved) {
              await selectDatasource(saved)
            }
          } else if (isCSV) {
            uploadMessage.value = 'CSV已保存至 07分离/原始数据，JSON文件可作为图谱数据源'
          } else {
            uploadError.value = '仅支持 .csv 或 .json 文件'
            uploading.value = false
            return
          }
          uploading.value = false
        } else {
          const files = await listDatasources()
          if (files && files.length) {
            const latest = files.sort((a,b) => (b.size||0)-(a.size||0))[0]
            if (latest?.path) await selectDatasource(latest.path)
          }
        }
      } catch (_) {
        uploading.value = false
        uploadError.value = '上传过程发生错误'
        return
      }
      // 跳转到增强知识图谱主界面
      if (window?.$vueRouter) {
        window.$vueRouter.push('/graph')
      } else {
        window.location.href = '#/graph'
      }
    }

    onMounted(() => { fetchMe() })

    return {
      fileInput,
      dropActive,
      preview,
      hasPreview,
      fileIcon,
      chooseFile,
      onFileChange,
      onDragOver,
      onDragLeave,
      onDrop,
      humanSize,
      formatCell,
      clear,
      goToGraph,
      uploading,
      uploadError,
      uploadMessage,
      selectedFile,
      listDatasources,
      selectDatasource,
      uploadDatasource,
      // 用户面板相关
      userPanelOpen,
      userName,
      userEmail,
      userAvatar,
      toggleUserPanel,
      onLogout
    }
  }
}
</script>

<style scoped>
.user-panel-wrap {
  position: fixed;
  top: 16px;
  right: 16px;
  z-index: 20;
  display: flex;
  align-items: center;
  gap: 8px;
}
.user-trigger {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: #ffffffee;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.08);
  cursor: pointer;
  transition: background-color .18s ease, box-shadow .18s ease, transform .1s ease, border-color .18s ease;
}
.user-trigger:hover { background: #f8fafc; box-shadow: 0 8px 20px rgba(0,0,0,0.10); transform: translateY(-1px); }
.user-trigger:active { transform: translateY(0); }
.avatar { width: 28px; height: 28px; border-radius: 999px; border: 1px solid #e5e7eb; display: flex; align-items: center; justify-content: center; background: #fff; color: #374151; font-size: 16px; }
.avatar.lg { width: 44px; height: 44px; font-size: 22px; }
.user-basic { display: grid; line-height: 1.1; }
.user-basic .name { font-size: 13px; color: #111827; font-weight: 600; }
.user-basic .meta { font-size: 12px; color: #6b7280; }
.chevron { margin-left: 4px; font-size: 12px; color: #6b7280; transition: transform .16s ease; }
.chevron.up { transform: rotate(180deg); display: none; }

.logout-inline {
  margin-left: 8px;
  padding: 4px 8px;
  border-radius: 10px;
  border: 1px solid #ef4444;
  background: #ef4444;
  color: #fff;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(239,68,68,.18);
  transition: background-color .18s ease, box-shadow .18s ease, transform .1s ease;
}
.logout-inline:hover { background: #dc2626; transform: translateY(-1px); box-shadow: 0 6px 14px rgba(220,38,38,.22); }
.logout-inline:active { transform: translateY(0); }

.logout-quick {
  padding: 8px 12px;
  border-radius: 12px;
  border: 1px solid #ef4444;
  background: #ef4444;
  color: #fff;
  font-weight: 600;
  cursor: pointer;
  box-shadow: 0 6px 18px rgba(239,68,68,.18);
  transition: background-color .18s ease, box-shadow .18s ease, transform .1s ease;
}
.logout-quick:hover { background: #dc2626; transform: translateY(-1px); box-shadow: 0 8px 20px rgba(220,38,38,.22); }
.logout-quick:active { transform: translateY(0); }

.user-panel {
  position: absolute;
  right: 0;
  margin-top: 8px;
  width: min(86vw, 300px);
  padding: 12px;
  border-radius: 14px;
  background: #fff;
  border: 1px solid #e5e7eb;
  box-shadow: 0 10px 24px rgba(0,0,0,0.12);
  transform-origin: top right;
  transform: scale(.96) translateY(-6px);
  opacity: 0;
  pointer-events: none;
  transition: transform .18s ease, opacity .18s ease;
}
.user-panel-wrap.open .user-panel { transform: scale(1) translateY(0); opacity: 1; pointer-events: auto; }
.up-row { display: grid; grid-template-columns: 44px 1fr; gap: 10px; align-items: center; }
.up-name { font-size: 14px; color: #111827; font-weight: 600; }
.up-meta { font-size: 12px; color: #6b7280; }
.up-actions { margin-top: 12px; display: flex; justify-content: flex-end; }
.btn.danger { border-color: #ef4444; background: #ef4444; color: #fff; }
.btn.danger:hover { background: #dc2626; }
.landing {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  background: linear-gradient(135deg, #eef2f3 0%, #e2eafc 100%);
}
.layout {
  display: grid;
  grid-template-columns: minmax(520px, 1fr) minmax(380px, 480px);
  gap: 18px;
  width: 100%;
  max-width: 1200px;
}
.card {
  padding: 20px;
  border-radius: 12px;
  background: #fff;
  box-shadow: 0 8px 24px rgba(0,0,0,0.06);
}
.upload-card { position: relative; min-height: clamp(420px, 52vh, 560px); }
.upload-card .title { font-size: 22px; margin: 0; color: #1f2937; }
.upload-card .desc { margin: 6px 0 14px; color: #6b7280; }
.file-input { display: none; }

.drop-zone {
  border: 2px dashed #93c5fd;
  border-radius: 12px;
  padding: 22px;
  text-align: center;
  background: linear-gradient(180deg, #f8fafc, #f1f5f9);
  transition: all .2s ease;
}
.drop-zone.active { background: #e0f2fe; border-color: #3b82f6; }
.dz-content { gap: 14px; }
.preview-header { margin-top: 12px; }
.dz-content { display: flex; align-items: center; gap: 14px; justify-content: center; }
.dz-icon { font-size: 26px; }
.dz-title { color: #334155; margin-bottom: 8px; }
.dz-hint { margin-top: 8px; color: #64748b; font-size: 12px; }

.btn {
  padding: 8px 14px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #fff;
  color: #334155;
  font-size: 13px;
  cursor: pointer;
}
.btn:hover { background: #f8fafc; }
.btn.primary { border-color: #3b82f6; color: #fff; background: #3b82f6; }
.btn.primary:hover { background: #2563eb; }
.btn.success { border-color: #10b981; color: #fff; background: #10b981; }
.btn.success:hover { background: #059669; }

.preview { margin-top: 16px; }
.preview-header { display: flex; align-items: center; justify-content: space-between; gap: 14px; }
.preview-header + .preview-illustration { margin-top: 10px; }
.file-meta { display: flex; flex-wrap: wrap; gap: 10px; color: #475569; }
.meta-item { background: #f1f5f9; padding: 6px 10px; border-radius: 8px; }
.preview-actions { display: flex; gap: 8px; }
.table-wrap, .preview-table, .table-hint { display: none; }
.hint { margin-top: 8px; font-size: 12px; }
.hint.error { color: #ef4444; }
.hint.ok { color: #10b981; }
.preview-illustration { display: grid; grid-template-columns: 90px 1fr; align-items: center; gap: 16px; padding: 18px; border: 1px solid #dae1e7; border-radius: 16px; background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 50%, #e0f2fe 100%); box-shadow: 0 12px 30px rgba(0,0,0,0.08); margin-top: 14px; }
.illus-icon { width: 90px; height: 90px; border-radius: 16px; display: flex; align-items: center; justify-content: center; font-size: 40px; background: #ffffff; border: 1px solid #e5e7eb; box-shadow: 0 12px 20px rgba(0,0,0,0.10); }
.illus-title { font-size: 14px; color: #1f2937; font-weight: 600; }
.illus-desc { margin-top: 6px; font-size: 12px; color: #6b7280; }
.illus-badges { margin-top: 6px; display: flex; gap: 8px; }
.badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 999px; background: #e0f2fe; color: #0f172a; font-size: 12px; border: 1px solid #e5e7eb; }
.illus-chips { display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }
.preview-tips { margin-top: 12px; display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }
.tip { display: flex; align-items: center; gap: 8px; padding: 10px; border: 1px dashed #e2e8f0; border-radius: 12px; background: #f8fafc; }
.tip-icon { width: 28px; height: 28px; border-radius: 8px; display: flex; align-items: center; justify-content: center; background: #fff; border: 1px solid #e5e7eb; box-shadow: 0 4px 10px rgba(0,0,0,0.06); }
.tip-text { font-size: 12px; color: #475569; }

@media (max-width: 640px) {
  .preview-illustration { margin-top: 12px; }
  .upload-card { min-height: auto; }
  .illus-icon { width: 64px; height: 64px; font-size: 30px; }
}

.info-card .sub-title { margin: 0 0 10px; font-size: 18px; color: #1f2937; }
.steps { list-style: none; padding: 0; margin: 0 0 12px; display: grid; gap: 8px; }
.steps li { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px; color: #334155; }
.emoji { margin-right: 6px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
.chip { background: linear-gradient(135deg, #eef2ff, #e0e7ff); color: #374151; padding: 6px 10px; border-radius: 999px; font-size: 12px; }
.guide { background: #f8fafc; border: 1px solid #e9ecef; border-radius: 12px; padding: 12px; color: #475569; }
.btn-row { margin-top: 8px; }
</style>
    // 用户信息面板状态与数据
    const userPanelOpen = ref(false)
    const user = ref({ name: '', email: '', avatar_url: '' })
    const userName = computed(() => user.value.name || user.value.username || '已登录用户')
    const userEmail = computed(() => user.value.email || '')
    const userAvatar = computed(() => user.value.avatar_url || 'https://api.dicebear.com/7.x/initials/svg?seed=' + encodeURIComponent(userName.value))
    const toggleUserPanel = () => { userPanelOpen.value = !userPanelOpen.value }

    const fetchMe = async () => {
      try {
        const res = await fetch('/api/auth/me')
        const j = await res.json()
        const candidate = j?.user || j || {}
        const fromSession = (() => {
          try { return JSON.parse(sessionStorage.getItem('current_user') || '{}') } catch (_) { return {} }
        })()
        user.value = {
          name: candidate.name || candidate.username || fromSession.name || fromSession.username || '',
          email: candidate.email || fromSession.email || '',
          avatar_url: candidate.avatar_url || fromSession.avatar_url || ''
        }
      } catch (e) {
        try {
          const u = JSON.parse(sessionStorage.getItem('current_user') || '{}')
          user.value = {
            name: u.name || u.username || '',
            email: u.email || '',
            avatar_url: u.avatar_url || ''
          }
        } catch (_) {}
      }
    }

    const clearClientSession = () => {
      try { localStorage.clear() } catch (e) {}
      try { sessionStorage.clear() } catch (e) {}
      try { if ('caches' in window) { caches.keys().then(keys => keys.forEach(k => caches.delete(k))).catch(()=>{}) } } catch (e) {}
    }
    const onLogout = async () => {
      clearClientSession()
      try { await fetch('/api/auth/logout', { method: 'POST' }) } catch (e) {}
      userPanelOpen.value = false
      const origin = encodeURIComponent(window.location.href)
      if (window?.$vueRouter) {
        window.$vueRouter.replace({ path: '/login', query: { redirect: '/upload', origin } })
      } else {
        window.location.assign('/login?redirect=/upload&origin=' + origin)
      }
    }

    onMounted(() => { fetchMe() })
      // 用户面板相关
      userPanelOpen,
      userName,
      userEmail,
      userAvatar,
      toggleUserPanel,
      onLogout,
