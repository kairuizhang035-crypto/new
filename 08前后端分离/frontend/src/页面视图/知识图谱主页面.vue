<template>
  <div class="container" ref="containerEl">
    <!-- 顶部固定用户条移除，改为侧栏内联显示 -->
    <!-- 侧边栏 -->
  <div class="sidebar" :style="{ width: sidebarWidth + 'px' }">
      <!-- 头部 -->
      <div class="header">
        <h1>增强知识图谱可视化</h1>
        <p>基于 (V, E_core, R, W, Θ, Φ)</p>
      </div>
      <div class="user-inline-bar" role="region" aria-label="当前用户信息">
        <div class="avatar">👤</div>
        <div class="user-name">{{ userName || '未登录' }}</div>
        <button class="ds-btn sm" @click="onLogout">退出登录</button>
        <button 
          class="ds-btn sm back" 
          @click="goToUpload" 
          @keydown.enter.prevent="goToUpload" 
          aria-label="返回上传界面"
          title="返回上传界面"
        >返回上传</button>
      </div>

      <!-- 数据源信息 -->
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

      <!-- 导航菜单 -->
      <div class="nav-menu">
        <div 
          class="nav-item" 
          :class="{ active: activePanel === 'overview' }"
          @click="setActivePanel('overview')"
        >
          <div>
            <span class="icon">📊</span>
            概览统计
          </div>
          <span class="count">总览</span>
        </div>

        <div 
          class="nav-item" 
          :class="{ active: activePanel === 'nodes' }"
          @click="setActivePanel('nodes')"
        >
          <div>
            <span class="icon">●</span>
            节点集合 (V)
          </div>
          <span class="count">{{ statistics.nodes || 0 }}</span>
        </div>

        <div 
          class="nav-item" 
          :class="{ active: activePanel === 'edges' }"
          @click="setActivePanel('edges')"
        >
          <div>
            <span class="icon">→</span>
            边集合 (E_core)
          </div>
          <span class="count">{{ statistics.edges || 0 }}</span>
        </div>

        <div 
          class="nav-item" 
          :class="{ active: activePanel === 'relations' }"
          @click="setActivePanel('relations')"
        >
          <div>
            <span class="icon">🔗</span>
            关系类型 (R)
          </div>
          <span class="count">{{ statistics.relations || 0 }}</span>
        </div>

        <div 
          class="nav-item" 
          :class="{ active: activePanel === 'weights' }"
          @click="setActivePanel('weights')"
        >
          <div>
            <span class="icon">⚖️</span>
            权重系统 (W)
          </div>
          <span class="count">{{ statistics.weights || 0 }}</span>
        </div>

        <div 
          class="nav-item" 
          :class="{ active: activePanel === 'parameters' }"
          @click="setActivePanel('parameters')"
        >
          <div>
            <span class="icon">📈</span>
            参数学习 (Θ)
          </div>
          <span class="count">{{ statistics.parameters || 0 }}</span>
        </div>

        <div 
          class="nav-item" 
          :class="{ active: activePanel === 'pathways' }"
          @click="setActivePanel('pathways')"
        >
          <div>
            <span class="icon">🛤️</span>
            路径分析 (Φ)
          </div>
          <span class="count">{{ statistics.pathways || 0 }}</span>
        </div>

        <div 
          class="nav-item" 
          :class="{ active: activePanel === 'graph' }"
          @click="setActivePanel('graph')"
        >
          <div>
            <span class="icon">🌐</span>
            网络图谱
          </div>
          <span class="count">可视化</span>
        </div>

        <div 
          class="nav-item" 
          :class="{ active: activePanel === 'admin' }"
          @click="setActivePanel('admin')"
        >
          <div>
            <span class="icon">🛠️</span>
            后台管理
          </div>
          <span class="count">进度</span>
        </div>
      </div>
    </div>

    <!-- 主内容区 -->
    <div class="splitter" @mousedown="startDrag" @touchstart.prevent="startTouchDrag"></div>
    <div class="main-content">
      <div class="content-panel">
        <!-- 概览统计组件（使用动态渲染以避免被当作文本显示） -->
        <div v-if="activePanel === 'overview'" class="panel active">
          <component 
            :is="OverviewComponent"
            :statistics="statistics"
            :relation-type-stats="relationTypeStats"
            :hierarchy-stats="hierarchyStats"
            :edges="edges"
            @navigate="setActivePanel"
          />
        </div>

        <!-- 节点管理组件（使用动态渲染以避免被当作文本显示） -->
        <div v-if="activePanel === 'nodes'" class="panel active">
          <component 
            :is="NodeComponent"
            :nodes="nodes"
            :node-type-stats="nodeTypeStats"
            @show-node-details="showNodeDetails"
          />
        </div>

        <!-- 边管理组件（使用动态渲染以避免被当作文本显示） -->
        <div v-if="activePanel === 'edges'" class="panel active">
          <component
            :is="EdgeComponent"
            :edges="edges"
            :available-relation-types="availableRelationTypes"
            :available-hierarchies="availableHierarchies"
            @show-edge-details="showEdgeDetails"
            @refresh-edges="onRefreshEdges"
          />
        </div>

        <!-- 关系类型组件（使用动态渲染以避免被当作文本显示） -->
        <div v-if="activePanel === 'relations'" class="panel active">
          <component
            :is="RelationComponent"
            :relations="relations"
            :relation-stats="relationTypeStatsDetailed"
            @show-relation-details="showRelationDetails"
            @show-edge-details="showEdgeDetails"
            @refresh-relations="onRefreshRelations"
          />
        </div>

        <!-- 权重系统组件（使用动态渲染以避免被当作文本显示） -->
        <div v-if="activePanel === 'weights'" class="panel active">
          <component 
            :is="WeightComponent"
            :weights="weights"
            @show-weight-details="showWeightDetails"
            @refresh-weights="onRefreshWeights"
          />
        </div>

        <!-- 参数学习组件（使用动态渲染以避免被当作文本显示） -->
        <div v-if="activePanel === 'parameters'" class="panel active">
          <component 
            :is="ParameterComponent"
            :parameters="parameters"
            @show-parameter-details="showParameterDetails"
            @search-parameters="onSearchParameters"
            @refresh-parameters="onRefreshParameters"
          />
        </div>

        <!-- 路径分析组件（使用动态渲染以避免被当作文本显示） -->
        <div v-if="activePanel === 'pathways'" class="panel active">
          <component 
            :is="PathwayComponent"
            :pathways="pathways"
            @show-pathway-details="showPathwayDetails"
            @refresh-pathways="onRefreshPathways"
          />
        </div>

        <!-- 网络图谱组件（使用动态渲染，避免中文标签被当作文本） -->
        <div v-if="activePanel === 'graph'" class="panel active">
          <component
            :is="GraphComponent"
            :nodes="nodes"
            :edges="edges"
            @show-node-details="onGraphNodeClick"
            @show-edge-details="onGraphEdgeClick"
          />
        </div>

        <div v-if="activePanel === 'admin'" class="panel active">
          <component :is="AdminComponent" />
        </div>
      </div>
    </div>

    <!-- 模态框 -->
    <div v-if="showModal" class="modal-overlay" @click="closeModal">
      <div class="modal-content" @click.stop>
        <div class="modal-header">
          <div class="title-with-badge">
            <h3 v-if="edgeHeaderSource && edgeHeaderTarget">
              边详情:
              <span class="link-node" @click="showNodeDetails(edgeHeaderSource)">{{ edgeHeaderSource }}</span>
              <span class="arrow"> → </span>
              <span class="link-node" @click="showNodeDetails(edgeHeaderTarget)">{{ edgeHeaderTarget }}</span>
            </h3>
            <h3 v-else>{{ modalTitle }}</h3>
            <span v-if="currentNodeId" :class="['type-badge', 'type-' + getNodeType(currentNodeId)]">{{ getNodeType(currentNodeId) }}</span>
          </div>
          <div class="modal-actions">
            <button class="back-btn" @click.stop="goBack" :disabled="!canGoBack">返回</button>
            <button class="close-btn" @click="closeModal">×</button>
          </div>
        </div>
        <div class="modal-body" ref="modalBodyEl" v-html="modalContent"></div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted, nextTick } from 'vue'
import { storeToRefs } from 'pinia'
import { useKnowledgeGraphStore } from '../状态管理/知识图谱状态'
import 概览统计组件 from './概览统计组件.vue'
import 节点管理组件 from './节点管理组件.vue'
import 边管理组件 from './边管理组件.vue'
import 关系类型组件 from './关系类型组件.vue'
import 权重系统组件 from './权重系统组件.vue'
import 参数学习组件 from './参数学习组件.vue'
import 路径分析组件 from './路径分析组件.vue'
import 网络图谱组件 from './网络图谱组件.vue'
import 后台管理组件 from './后台管理组件.vue'

export default {
  name: '知识图谱主页面',
  components: {
    概览统计组件,
    节点管理组件,
    边管理组件,
    关系类型组件,
    权重系统组件,
    参数学习组件,
    路径分析组件,
    网络图谱组件
  },
  setup() {
    const store = useKnowledgeGraphStore()
    // 动态组件别名，暴露到模板中使用
    const NodeComponent = 节点管理组件
    const EdgeComponent = 边管理组件
    const RelationComponent = 关系类型组件
    const WeightComponent = 权重系统组件
    const ParameterComponent = 参数学习组件
    const PathwayComponent = 路径分析组件
    const OverviewComponent = 概览统计组件
    // 网络图谱使用动态组件，避免在少数环境下中文标签渲染为文本
    const GraphComponent = 网络图谱组件
    const AdminComponent = 后台管理组件
    // 分界线与拖拽逻辑（在 setup 内定义）
    const containerEl = ref(null)
    const sidebarWidth = ref(300)
    const minWidth = 100
    const dragging = ref(false)
    let startX = 0
    let startWidth = 0

    const applyWidth = (w) => {
      const container = containerEl.value
      if (!container) return
      const total = container.clientWidth
      const splitter = 8
      const mainMin = 100
      let nw = Math.max(minWidth, Math.min(w, total - splitter - mainMin))
      sidebarWidth.value = nw
    }

    const onMove = (e) => {
      if (!dragging.value) return
      const clientX = e.clientX ?? (e.touches && e.touches[0]?.clientX)
      if (clientX == null) return
      const delta = clientX - startX
      applyWidth(startWidth + delta)
    }

    const stopDrag = () => {
      dragging.value = false
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', stopDrag)
      window.removeEventListener('touchmove', onMove)
      window.removeEventListener('touchend', stopDrag)
    }

    const startDrag = (e) => {
      dragging.value = true
      startX = e.clientX
      startWidth = sidebarWidth.value
      window.addEventListener('mousemove', onMove)
      window.addEventListener('mouseup', stopDrag)
    }

    const startTouchDrag = (e) => {
      dragging.value = true
      startX = e.touches[0].clientX
      startWidth = sidebarWidth.value
      window.addEventListener('touchmove', onMove, { passive: false })
      window.addEventListener('touchend', stopDrag)
    }

    const onResize = () => { applyWidth(sidebarWidth.value) }
    onMounted(() => { window.addEventListener('resize', onResize) })
    
    // 响应式数据
    const activePanel = ref('overview')
    const showModal = ref(false)
    const modalTitle = ref('')
    const modalContent = ref('')
    // 关系详情分页状态
    const currentRelationType = ref('')
    const relationExamplesMaster = ref([])
    const relationExamplesPageSize = ref(10)
    const relationExamplesPage = ref(1)
    const relationExamplesFilter = ref('all')

    // 通用模态历史栈：可跨 节点/边/关系/权重/参数/中介 详情进行返回
    const modalHistory = ref([])

    // 当前内容标识（用于判断返回与记录历史）
    const currentParameterKey = ref('')
    const currentPathwayKey = ref('')

    const canGoBack = computed(() => modalHistory.value.length > 0)

    // 权重详情上下文与缓存
    const currentWeightKey = ref('')
    const weightDetailsCache = ref(null)

    // 通用格式化与小工具（中文化缺失值）
    const fmtNum = (v) => {
      if (v === null || v === undefined || v === '') return '无'
      const n = Number(v)
      if (!Number.isFinite(n)) return String(v)
      return n.toFixed(4)
    }
    const renderKvGrid = (obj) => {
      if (!obj || typeof obj !== 'object') return '<p class="empty">暂无数据</p>'
      const entries = Object.entries(obj)
      if (!entries.length) return '<p class="empty">暂无数据</p>'
      const rows = entries.map(([k, v]) => {
        let display = v
        if (v === null || v === undefined || v === '') display = '无'
        else if (Array.isArray(v)) display = v.join('，')
        else if (typeof v === 'number') display = fmtNum(v)
        else if (typeof v === 'string') display = mapStrCn(v)
        return `<li class="kv-item"><span class="kv-label">${k}</span><span class="kv-value">${display}</span></li>`
      }).join('')
      return `<ul class="kv-grid" role="list">${rows}</ul>`
    }
    const mapQualityCn = (q) => {
      const s = String(q || '').toLowerCase()
      if (!s) return '未知'
      if (s.includes('high')) return '高'
      if (s.includes('medium')) return '中'
      if (s.includes('low')) return '低'
      if (s.includes('unknown')) return '未知'
      return q || '未知'
    }
    const mapAlgListCn = (list) => {
      const dict = {
        'MLE': '极大似然',
        'Bayesian': '贝叶斯',
        'EM': 'EM',
        'SEM': '结构方程',
        'Pearson': '皮尔逊',
        'Spearman': '斯皮尔曼'
      }
      if (!Array.isArray(list)) return list
      return list.map(x => {
        const k = String(x)
        return dict[k] || x
      })
    }
    const mapStrCn = (s) => {
      if (s === null || s === undefined || s === '') return '无'
      const dict = {
        'High': '高', 'high': '高',
        'Medium': '中', 'medium': '中',
        'Low': '低', 'low': '低',
        'Unknown': '未知', 'unknown': '未知',
        'True': '是', 'true': '是',
        'False': '否', 'false': '否',
        'complete': '完整', 'incomplete': '不完整'
      }
      return dict[s] || s
    }

    // 数据源选择
    const { datasourceFiles, currentDatasource } = storeToRefs(store)
    const datasourceFilesDedup = computed(() => {
      const seen = new Set()
      const out = []
      for (const f of datasourceFiles.value || []) {
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
        const files = datasourceFiles.value || []
        if (!selectedDatasourcePath.value && files.length) {
          selectedDatasourcePath.value = files[0].path
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

    const onUploadChange = async (evt) => {
      const file = evt.target.files?.[0]
      if (!file) return
      await store.uploadDatasource(file, true)
      await refreshDatasourceList()
      await loadCurrentDatasource()
      evt.target.value = ''
    }
    const isUploadPath = (p) => String(p || '').includes('/07分离/uploads/')
    const onDeleteDatasource = async (path) => {
      try {
        await fetch('/api/datasource/delete', {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path })
        })
        await refreshDatasourceList()
        await loadCurrentDatasource()
        if (selectedDatasourcePath.value === path) {
          selectedDatasourcePath.value = ''
        }
      } catch (e) {}
    }

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
    const onApplyFromDropdown = async (path) => {
      selectedDatasourcePath.value = path
      await applySelectedDatasource()
      dsOpen.value = false
    }
    const isCurrent = (p) => String(p || '') === String((currentDatasource.value && currentDatasource.value.path) || '')

    const onLogout = async () => {
      try {
        await fetch('/api/auth/logout', { method: 'POST' })
      } catch (e) {}
      if (window?.$vueRouter) {
        window.$vueRouter.replace({ path: '/login', query: { redirect: '/', origin: window.location.href } })
      } else {
        window.location.href = '/login?redirect=/&origin=' + encodeURIComponent(window.location.href)
      }
    }

    const goToUpload = () => {
      try {
        window.location.assign('http://localhost:8080/upload')
      } catch (e) {
        window.location.href = 'http://localhost:8080/upload'
      }
    }

    // 自定义下拉状态与选择
    const dsOpen = ref(false)
    const selectedLabel = computed(() => {
      const curPath = (currentDatasource.value && currentDatasource.value.path) ? currentDatasource.value.path : ''
      if (curPath) {
        const f = (datasourceFilesDedup.value || []).find(x => x.path === curPath)
        if (f) return `${f.name}（${fmtSize(f.size)}）`
        const name = curPath.split('/').pop()
        return name || curPath
      }
      return '请选择数据源…'
    })
    const onSelectDatasource = (path) => {
      selectedDatasourcePath.value = path
      dsOpen.value = false
    }

    // 从store获取响应式数据
    const { 
      nodes, 
      edges, 
      relations, 
      weights, 
      parameters, 
      pathways,
      nodeTypeStats,
      statistics,
      relationTypeStats,
      relationTypeStatsDetailed,
      hierarchyStats
    } = storeToRefs(store)

    // 计算属性
    const availableRelationTypes = computed(() => {
      return [...new Set(edges.value.map(edge => edge.relation_type))]
    })

    const availableHierarchies = computed(() => {
      return [...new Set(edges.value.map(edge => edge.edge_hierarchy))]
    })

    // 方法
    const setActivePanel = (panel) => {
      activePanel.value = panel
    }

    // 弹窗点击事件委托（用于 v-html 中的邻居 chip 点击）
    const modalBodyEl = ref(null)
    let isModalBodyListenerBound = false
    const handleModalClick = (event) => {
      // 三角验证边 chip 点击（格式：A → B），打开边详情
      const edgeChip = event.target.closest('.edge-chip')
      if (edgeChip) {
        const text = (edgeChip.textContent || '').trim()
        const parts = text.split('→')
        if (parts.length === 2) {
          const source = parts[0].trim()
          const target = parts[1].trim()
          const found = edges.value.find(e => e.source === source && e.target === target) || { source, target }
          showEdgeDetails(found)
        }
        return
      }

      // 节点 chip 点击（用于基本信息区中的源/目标节点）
      const nodeChip = event.target.closest('.chip.node')
      if (nodeChip) {
        const id = nodeChip.getAttribute('data-node-id') || nodeChip.textContent.trim()
        if (id) showNodeDetails(id)
        return
      }

      // 顶部操作按钮（权重/参数/中介）：跳转到对应面板
      const actionBtn = event.target.closest('.action-btn')
      if (actionBtn) {
        if (actionBtn.classList.contains('weight')) {
          const key = currentWeightKey.value
          // 记录当前边详情到历史栈
          if (edgeHeaderSource.value && edgeHeaderTarget.value) {
            modalHistory.value.push({ kind: 'edge', source: edgeHeaderSource.value, target: edgeHeaderTarget.value })
          }
          // 直接在当前弹窗内展示权重详情，不切换到权重系统面板
          if (key) nextTick(() => { showWeightDetails(key) })
          return
        }
        if (actionBtn.classList.contains('parameter')) {
          const source = edgeHeaderSource.value
          const target = edgeHeaderTarget.value
          const key = (source && target) ? `${source}→${target}` : (target || '')
          // 记录当前边详情到历史栈
          if (edgeHeaderSource.value && edgeHeaderTarget.value) {
            modalHistory.value.push({ kind: 'edge', source: edgeHeaderSource.value, target: edgeHeaderTarget.value })
          }
          // 直接在当前弹窗内展示参数详情，不切换到参数学习面板
          if (key) nextTick(() => { showParameterDetails(key) })
          return
        }
        if (actionBtn.classList.contains('mediation')) {
          const source = edgeHeaderSource.value
          const target = edgeHeaderTarget.value
          const key = (source && target) ? `${source}→${target}` : (target || source || '')
          // 记录当前边详情到历史栈
          if (edgeHeaderSource.value && edgeHeaderTarget.value) {
            modalHistory.value.push({ kind: 'edge', source: edgeHeaderSource.value, target: edgeHeaderTarget.value })
          }
          // 直接在当前弹窗内展示中介路径详情，不切换到路径分析面板
          if (key) nextTick(() => { showPathwayDetails(key) })
          return
        }
      }

      // 关系详情统计徽章点击筛选（全部/三角验证/非三角）
      const relBadge = event.target.closest('.stat-badge')
      if (relBadge && currentRelationType.value) {
        if (relBadge.classList.contains('all')) {
          relationExamplesFilter.value = 'all'
        } else if (relBadge.classList.contains('tri')) {
          relationExamplesFilter.value = 'tri'
        } else if (relBadge.classList.contains('non')) {
          relationExamplesFilter.value = 'non'
        }
        relationExamplesPage.value = 1
        modalContent.value = buildRelationDetailsContent(currentRelationType.value)
        return
      }
      // 邻居 chip 点击
      const chip = event.target.closest('.neighbor-chip')
      if (chip) {
        const id = chip.getAttribute('data-node-id') || chip.textContent.trim()
        if (id) {
          showNodeDetails(id)
        }
        return
      }

      // 三角验证边徽章点击，展示三角边明细
      const triBadge = event.target.closest('.stat-badge.tri')
      if (triBadge) {
        // 仅当当前为节点详情上下文时才响应
        if (!currentNodeId.value || currentRelationType.value) return
        modalContent.value = buildNodeDetailsContent(currentNodeId.value, true)
        return
      }

      // 已移除页大小选择（固定为 10 条/页）

      // 翻页
      const pagerChip = event.target.closest('.pager-chip')
      if (pagerChip) {
        const action = pagerChip.getAttribute('data-action')
        const pageStr = pagerChip.getAttribute('data-page')
        const totalPages = Math.max(1, Math.ceil(relationExamplesMaster.value.length / relationExamplesPageSize.value))
        if (action === 'first') {
          relationExamplesPage.value = 1
        } else if (action === 'prev') {
          relationExamplesPage.value = Math.max(1, relationExamplesPage.value - 1)
        } else if (action === 'next') {
          relationExamplesPage.value = Math.min(totalPages, relationExamplesPage.value + 1)
        } else if (action === 'last') {
          relationExamplesPage.value = totalPages
        } else if (pageStr) {
          const to = parseInt(pageStr, 10)
          if (!Number.isNaN(to)) {
            relationExamplesPage.value = Math.min(totalPages, Math.max(1, to))
          }
        }
        if (currentRelationType.value) {
          modalContent.value = buildRelationDetailsContent(currentRelationType.value)
        } else if (currentWeightKey.value) {
          modalContent.value = buildWeightDetailsContent(currentWeightKey.value)
        }
        return
      }
    }

    // 节点历史栈与当前节点
    const nodeHistory = ref([])
    const currentNodeId = ref('')
    const edgeHeaderSource = ref('')
    const edgeHeaderTarget = ref('')

    const buildNodeDetailsContent = (nodeId, includeTriList = false) => {
      // 维护历史栈：当弹窗已打开且不是返回操作时，记录当前节点
      // 注意：历史维护在 showNodeDetails 中进行
      // 计算入/出度与邻居列表
      const incomingEdges = edges.value.filter(e => e.target === nodeId)
      const outgoingEdges = edges.value.filter(e => e.source === nodeId)
      const inNeighbors = Array.from(new Set(incomingEdges.map(e => e.source)))
      const outNeighbors = Array.from(new Set(outgoingEdges.map(e => e.target)))
      const triangulatedIncidentEdges = edges.value.filter(
        e => (e.source === nodeId || e.target === nodeId) && e.edge_hierarchy === 'triangulated_verified'
      )

      const inDegree = incomingEdges.length
      const outDegree = outgoingEdges.length
      const triCount = triangulatedIncidentEdges.length

      // 组装邻居chips
      const inChips = inNeighbors.length
        ? inNeighbors.map(n => `<li><button class="neighbor-chip in" type="button" data-node-id="${n}">${n}</button></li>`).join('')
        : '<li class="empty">无</li>'
      const outChips = outNeighbors.length
        ? outNeighbors.map(n => `<li><button class="neighbor-chip out" type="button" data-node-id="${n}">${n}</button></li>`).join('')
        : '<li class="empty">无</li>'

      // 构建详情内容
      let content = `
        <div class="detail-section">
          <h4>基本信息</h4>
          <p><strong>节点名称:</strong> ${nodeId}</p>
          <p><strong>节点类型:</strong> ${getNodeType(nodeId)}</p>
        </div>
        <div class="detail-section">
          <h4>连接统计</h4>
          <div class="stats-row" role="group" aria-label="连接统计">
            <button class="stat-badge in" type="button"><span class="label">入度：</span><span class="value">${inDegree}</span></button>
            <button class="stat-badge out" type="button"><span class="label">出度：</span><span class="value">${outDegree}</span></button>
            <button class="stat-badge tri" type="button"><span class="label">三角验证边：</span><span class="value">${triCount}</span></button>
          </div>
        </div>
        <div class="detail-section">
          <h4>相邻节点</h4>
          <div class="neighbor-group">
            <div class="group-title"><strong>入邻居 (${inNeighbors.length})</strong></div>
            <ul class="neighbor-list" role="list">${inChips}</ul>
          </div>
          <div class="neighbor-group">
            <div class="group-title"><strong>出邻居 (${outNeighbors.length})</strong></div>
            <ul class="neighbor-list" role="list">${outChips}</ul>
          </div>
        </div>
      `

      if (includeTriList) {
        const triChips = triangulatedIncidentEdges.length
          ? triangulatedIncidentEdges
              .map(e => `<li><button class="edge-chip tri" type="button">${e.source} → ${e.target}</button></li>`)
              .join('')
          : '<li class="empty">无</li>'

        content += `
          <div class="detail-section">
            <h4>三角验证边明细</h4>
            <ul class="edge-list" role="list">${triChips}</ul>
          </div>
        `
      }

      return content
    }

    const showNodeDetails = (nodeId, fromBack = false) => {
      // 在切换前，记录当前上下文到通用历史栈
      if (showModal.value && !fromBack) {
        if (edgeHeaderSource.value && edgeHeaderTarget.value) {
          modalHistory.value.push({ kind: 'edge', source: edgeHeaderSource.value, target: edgeHeaderTarget.value })
        } else if (currentNodeId.value) {
          modalHistory.value.push({ kind: 'node', nodeId: currentNodeId.value })
        } else if (currentRelationType.value) {
          modalHistory.value.push({ kind: 'relation', type: currentRelationType.value })
        } else if (currentWeightKey.value) {
          modalHistory.value.push({ kind: 'weight', key: currentWeightKey.value })
        } else if (currentParameterKey.value) {
          modalHistory.value.push({ kind: 'parameter', key: currentParameterKey.value })
        } else if (currentPathwayKey.value) {
          modalHistory.value.push({ kind: 'pathway', key: currentPathwayKey.value })
        }
      }
      // 清理边标题的来源/目标，使标题回到节点模式
      edgeHeaderSource.value = ''
      edgeHeaderTarget.value = ''
      // 维护历史栈：当弹窗已打开且不是返回操作时，记录当前节点
      if (showModal.value && currentNodeId.value && currentNodeId.value !== nodeId && !fromBack) {
        nodeHistory.value.push(currentNodeId.value)
      }
      currentNodeId.value = nodeId

      modalTitle.value = `节点详情: ${nodeId}`
      // 先加载基本信息与邻居，再异步追加CPT区块
      const baseHtml = buildNodeDetailsContent(nodeId, false)
      modalContent.value = baseHtml
      // 异步拉取节点详情/参数详情，以展示CPT（优先节点接口，回退参数接口）
      nextTick(async () => {
        try {
          let data = {}
          let methods = {}
          let methodEstimates = {}
          let stability = {}

          // 1) 优先尝试节点详情接口（/api/nodes/{id}/details）
          try {
            const nodeResp = await store.getNodeDetails(nodeId)
            const nodeData = nodeResp?.data ?? nodeResp ?? {}
            data = nodeData
            methods = nodeData.methods || nodeData.parameter_methods || {}
            methodEstimates = nodeData.method_estimates || {}
            stability = nodeData.parameter_stability || {}
          } catch (e) {
            // 节点详情可能未提供参数方法，忽略错误，尝试参数详情
          }

          // 2) 若未取得方法或为空，则回退到参数详情接口（/api/parameters/{id}/details）
          const hasCPTInMethods = (mm) => {
            if (!mm || typeof mm !== 'object') return false
            return ['MLE','Bayesian','EM','SEM'].some(k => {
              const md = mm[k] || {}
              const c = md.cpt_data || md.cpt
              if (!c) return false
              if (Array.isArray(c)) return c.length > 0
              if (typeof c === 'object') {
                if (Array.isArray(c.table)) return c.table.length > 0
                return Object.keys(c).length > 0
              }
              return false
            })
          }
          if (!methods || Object.keys(methods).length === 0 || !hasCPTInMethods(methods)) {
            const incoming = edges.value.filter(e => e.target === nodeId)
            const candidates = Array.from(new Set(incoming.map(e => `${e.source}->${nodeId}`)))
            let chosen = null
            let pResp = null
            let pData = null
            for (const k of candidates) {
              try {
                pResp = await store.getParameterDetails(k)
                pData = pResp?.data ?? pResp ?? {}
                const m2 = pData.methods || {}
                if (hasCPTInMethods(m2)) { chosen = k; data = pData; methods = m2; methodEstimates = pData.method_estimates || {}; stability = pData.parameter_stability || {}; break }
              } catch (_) {}
            }
            if (!chosen) {
              try {
                const directResp = await store.getParameterDetails(nodeId)
                const directData = directResp?.data ?? directResp ?? {}
                const m3 = directData.methods || {}
                if (Object.keys(m3).length) { data = directData; methods = m3; methodEstimates = directData.method_estimates || {}; stability = directData.parameter_stability || {} }
              } catch (_) {}
            }
            console.log('CPT调试: 回退参数详情', { nodeId, candidates, chosen })
          }

          const fmt4 = (v) => {
            if (v === null || v === undefined) return 'N/A'
            const n = Number(v)
            if (!Number.isFinite(n)) return String(v)
            try { return n.toPrecision(4) } catch { return String(v) }
          }

          // 入邻居（父节点）集合（作为缺省父列表）
          const parentFallback = Array.from(new Set(
            edges.value.filter(e => e.target === nodeId).map(e => e.source)
          ))
          console.log('CPT调试: 父节点缺省列表', { nodeId, parentFallback })

          const renderCPTTable = (cpt, m) => {
            if (!cpt) return '<p class="empty">无CPT数据</p>'
            let rows = []
            let headers = []
            if (Array.isArray(cpt)) {
              rows = cpt
              headers = Array.from(rows.reduce((set, r) => { Object.keys(r||{}).forEach(k=>set.add(k)); return set }, new Set()))
              headers = headers.filter(h => String(h).toLowerCase() !== 'intercept' && String(h) !== '截距')
            } else if (typeof cpt === 'object') {
              if (Array.isArray(cpt.table)) {
                rows = cpt.table
                headers = Array.from(rows.reduce((set, r) => { Object.keys(r||{}).forEach(k=>set.add(k)); return set }, new Set()))
                headers = headers.filter(h => String(h).toLowerCase() !== 'intercept' && String(h) !== '截距')
              } else {
                const parentNames = Array.isArray(methods[m]?.parents) && methods[m].parents?.length
                  ? methods[m].parents
                  : parentFallback
                const entries = Object.entries(cpt)
                const parseMaybeArray = (v) => {
                  if (Array.isArray(v)) return v
                  if (typeof v === 'string') { const s = v.trim(); if (s.startsWith('[') && s.endsWith(']')) { try { const arr = JSON.parse(s); return Array.isArray(arr) ? arr : null } catch {} } }
                  return null
                }
                const firstArr = entries.map(([,v]) => parseMaybeArray(v)).find(a => Array.isArray(a))
                if (Array.isArray(firstArr)) {
                  const targetName = nodeId
                  const makeHeader = () => {
                    if (firstArr.length === 2) return `<thead><tr><th>父节点条件</th><th>${targetName} 不发生</th><th>${targetName} 发生</th></tr></thead>`
                    const cols = firstArr.map((_, idx) => `<th>${targetName} 状态${idx}</th>`).join('')
                    return `<thead><tr><th>父节点条件</th>${cols}</tr></thead>`
                  }
                  const labelState = (v) => { const s = String(v); if (s === '0' || s === '不发生') return '0'; if (s === '1' || s === '发生') return '1'; return s }
                  const formatDecimal4 = (num) => { const n = Number(num); return Number.isFinite(n) ? n.toFixed(4) : String(num) }
                  const esc = (s) => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')
                  const buildCond = (combo) => {
                    const parts = String(combo).split(/\s*,\s*/)
                    const states = parts.map(p => labelState(p))
                    const short = states.join(', ')
                    let full = short
                    if (parentNames.length === parts.length && parentNames.length > 0) {
                      full = parentNames.map((name, i) => `${name}: ${labelState(parts[i])}`).join('， ')
                    }
                    return { short, full }
                  }
                  const body = entries.map(([combo, dist]) => {
                    const cond = buildCond(combo)
                    const arr = parseMaybeArray(dist)
                    if (Array.isArray(arr)) {
                      const cols = arr.map(p => `<td>${formatDecimal4(p)}</td>`).join('')
                      return `<tr><td title="${esc(cond.full)}">${cond.short}</td>${cols}</tr>`
                    }
                    return `<tr><td title="${esc(cond.full)}">${cond.short}</td><td colspan="${firstArr.length}">${typeof dist === 'object' ? JSON.stringify(dist) : String(dist)}</td></tr>`
                  }).join('')
                  return `<table class="cpt-table">${makeHeader()}<tbody>${body}</tbody></table>`
                }
                headers = ['键', '值']
                rows = entries.map(([k, v]) => ({ 键: k, 值: typeof v === 'object' ? JSON.stringify(v) : v }))
              }
            }
            if (!rows.length) return '<p class="empty">无CPT数据</p>'
            const MAX_ROWS = 20
            const shown = rows.slice(0, MAX_ROWS)
            const shortenHeader = (h) => { const s = String(h); const parts = s.split(/->|→|—>|=>|➝|⟶/); return parts.length > 1 ? parts[parts.length-1].trim() : s }
            const headerHtml = headers.map(h => `<th>${shortenHeader(h)}</th>`).join('')
            const formatDecimal4 = (num) => { const n = Number(num); return Number.isFinite(n) ? n.toFixed(4) : String(num) }
            const formatCell = (val) => {
              if (val === null || val === undefined) return 'N/A'
              if (Array.isArray(val)) return `[${val.map(x => formatDecimal4(x)).join(', ')}]`
              if (typeof val === 'string') { const s = val.trim(); if ((s.startsWith('[')&&s.endsWith(']')) || (s.startsWith('{')&&s.endsWith('}'))) { try { const parsed = JSON.parse(s); if (Array.isArray(parsed)) return `[${parsed.map(x=>formatDecimal4(x)).join(', ')}]` } catch {} } const n = Number(s); if (Number.isFinite(n)) return n.toFixed(4); return s }
              if (typeof val === 'number') return val.toFixed(4)
              return String(val)
            }
            const bodyHtml = shown.map(r => `<tr>${headers.map(h => `<td>${formatCell(r[h])}</td>`).join('')}</tr>`).join('')
            return `<table class="cpt-table"><thead><tr>${headerHtml}</tr></thead><tbody>${bodyHtml}</tbody></table>${rows.length>MAX_ROWS?`<div class="table-hint">仅展示前 ${MAX_ROWS} 行</div>`:''}`
          }

          // 仅展示 MLE / Bayesian / EM，隐藏 SEM
          const methodsOrder = ['MLE','Bayesian','EM']
          let cptHtml = ''
          methodsOrder.forEach(m => {
            const mdata = methods[m] || {}
            const cpt = mdata.cpt_data || mdata.cpt || null
            const parents = Array.isArray(mdata.parents) && mdata.parents.length ? mdata.parents : parentFallback
            const statusHtml = (() => {
              const items = []
              const ntype = mdata.node_type
              if (ntype !== undefined && ntype !== null && ntype !== '') items.push(`<li class="kv-item"><span class="kv-label">节点类型</span><span class="kv-value">${ntype}</span></li>`)
              const pVal = parents && parents.length ? parents.join(', ') : ''
              if (pVal) items.push(`<li class="kv-item"><span class="kv-label">父节点</span><span class="kv-value">${pVal}</span></li>`)
              return items.length ? `<ul class="kv-grid" role="list">${items.join('')}</ul>` : ''
            })()
            const cptRowCount = (() => {
              if (!cpt) return 0
              if (Array.isArray(cpt)) return cpt.length
              if (typeof cpt === 'object') {
                if (Array.isArray(cpt.table)) return cpt.table.length
                return Object.keys(cpt).length
              }
              return 0
            })()
            console.log('CPT调试: 方法CPT状态', { nodeId, method: m, hasCPT: !!cpt, rows: cptRowCount })
            cptHtml += `
              <div class="detail-section">
                <h4>${m} 条件概率表（CPT）</h4>
                ${statusHtml}
                ${renderCPTTable(cpt, m)}
              </div>
            `
          })
          // 追加到现有内容之后
          modalContent.value = baseHtml + cptHtml
        } catch (err) {
          // 不影响基本信息展示
          console.warn('加载节点CPT失败', err)
        }
      })
      showModal.value = true

      // 绑定一次事件委托，确保点击 chip 可打开详情
      nextTick(() => {
        if (modalBodyEl.value && !isModalBodyListenerBound) {
          modalBodyEl.value.addEventListener('click', handleModalClick)
          isModalBodyListenerBound = true
        }
      })
    }

    const restoreModal = (state) => {
      if (!state || typeof state !== 'object') return
      if (state.kind === 'node') {
        showNodeDetails(state.nodeId, true)
      } else if (state.kind === 'edge') {
        showEdgeDetails({ source: state.source, target: state.target }, true)
      } else if (state.kind === 'weight') {
        showWeightDetails(state.key, true)
      } else if (state.kind === 'parameter') {
        showParameterDetails(state.key, undefined, true)
      } else if (state.kind === 'pathway') {
        showPathwayDetails(state.key, undefined, true)
      } else if (state.kind === 'relation') {
        showRelationDetails(state.type, true)
      }
    }

    const goBack = () => {
      if (modalHistory.value.length === 0) return
      const prev = modalHistory.value.pop()
      // 返回时淡出当前内容，然后淡入目标内容
      if (modalBodyEl.value) {
        modalBodyEl.value.classList.add('fade-out')
        setTimeout(() => {
          restoreModal(prev)
          nextTick(() => {
            if (modalBodyEl.value) {
              modalBodyEl.value.classList.remove('fade-out')
              modalBodyEl.value.classList.add('fade-in')
              setTimeout(() => {
                if (modalBodyEl.value) modalBodyEl.value.classList.remove('fade-in')
              }, 200)
            }
          })
        }, 200)
      } else {
        restoreModal(prev)
      }
    }

    const showEdgeDetails = async (edge, fromBack = false) => {
      try {
        // 在切换前，记录当前上下文到通用历史栈
        if (showModal.value && !fromBack) {
          if (currentNodeId.value) {
            modalHistory.value.push({ kind: 'node', nodeId: currentNodeId.value })
          } else if (currentRelationType.value) {
            modalHistory.value.push({ kind: 'relation', type: currentRelationType.value })
          } else if (currentWeightKey.value) {
            modalHistory.value.push({ kind: 'weight', key: currentWeightKey.value })
          } else if (currentParameterKey.value) {
            modalHistory.value.push({ kind: 'parameter', key: currentParameterKey.value })
          } else if (currentPathwayKey.value) {
            modalHistory.value.push({ kind: 'pathway', key: currentPathwayKey.value })
          } else if (edgeHeaderSource.value && edgeHeaderTarget.value) {
            modalHistory.value.push({ kind: 'edge', source: edgeHeaderSource.value, target: edgeHeaderTarget.value })
          }
        }

        // 切换到边详情，重置当前标识
        currentNodeId.value = ''
        currentRelationType.value = ''
        currentParameterKey.value = ''
        currentPathwayKey.value = ''

        const resp = await store.getEdgeDetails(edge.source, edge.target)
        const data = resp || {}
        const base = data.base || edge
        const weight = data.weight || {}
        const relationMeta = data.relation || {}

        modalTitle.value = `边详情: ${base.source} → ${base.target}`
        // 供标题可点击使用
        edgeHeaderSource.value = base.source
        edgeHeaderTarget.value = base.target
        // 保存权重引用，用于跳转到权重页面时打开具体详情
        currentWeightKey.value = base.weight_ref || ''
        modalContent.value = `
          <div class="detail-section">
            <h4>基本信息</h4>
            <p><strong>源节点:</strong> <span class="chip node" data-node-id="${base.source}">${base.source}</span></p>
            <p><strong>目标节点:</strong> <span class="chip node" data-node-id="${base.target}">${base.target}</span></p>
            <p><strong>关系类型:</strong> ${base.relation_type}</p>
            <p><strong>边层次:</strong> ${base.edge_hierarchy}</p>
            <p><strong>是否直接:</strong> ${base.is_direct ? '是' : '否'}</p>
            <p><strong>权重引用:</strong> ${base.weight_ref || 'N/A'}</p>
          </div>
          <div class="detail-section">
            <div class="button-group">
              <button class="action-btn weight">权重</button>
              <button class="action-btn parameter">参数</button>
              <button class="action-btn mediation">中介</button>
            </div>
          </div>
        `
        showModal.value = true
        // 绑定一次事件委托，确保点击按钮与 chip 可打开对应界面/详情
        nextTick(() => {
          if (modalBodyEl.value && !isModalBodyListenerBound) {
            modalBodyEl.value.addEventListener('click', handleModalClick)
            isModalBodyListenerBound = true
          }
        })
      } catch (e) {
        console.error('获取边详情失败', e)
        modalTitle.value = '边详情加载失败'
        modalContent.value = `<p>无法获取边详情，请稍后重试。</p>`
        showModal.value = true
      }
    }

    const onRefreshEdges = async () => {
      try {
        await store.reloadEdges()
      } catch (e) {
        console.error('刷新边集合失败', e)
      }
    }

    const onRefreshRelations = async () => {
      try {
        await store.fetchRelationTypeStats()
      } catch (e) {
        console.error('刷新关系类型统计失败', e)
      }
    }

    const onRefreshPathways = async () => {
      try {
        await store.reloadPathways()
      } catch (e) {
        console.error('刷新路径集合失败', e)
      }
    }

    const buildRelationDetailsContent = (type) => {
      const meta = relations.value?.[type] || {}
      const stats = relationTypeStatsDetailed.value?.[type] || {}
      // 完整示例集合（从 edges 过滤，避免后端限制影响）
      relationExamplesMaster.value = edges.value
        .filter(e => e.relation_type === type)
        .map(e => ({ source: e.source, target: e.target, edge_hierarchy: e.edge_hierarchy }))

      // 根据筛选计算列表
      const pageSize = relationExamplesPageSize.value
      const current = relationExamplesPage.value
      const filter = relationExamplesFilter.value
      let filteredList = relationExamplesMaster.value
      if (filter === 'tri') {
        filteredList = filteredList.filter(e => e.edge_hierarchy === 'triangulated_verified')
      } else if (filter === 'non') {
        filteredList = filteredList.filter(e => e.edge_hierarchy === 'non_triangulated')
      }
      const totalFiltered = filteredList.length
      const totalPages = Math.max(1, Math.ceil(totalFiltered / pageSize))
      const sliceStart = (current - 1) * pageSize
      const slice = filteredList.slice(sliceStart, sliceStart + pageSize)
      const examplesHtml = slice.length
        ? slice.map(ex => `<div class="edge-chip">${ex.source} → ${ex.target}</div>`).join('')
        : '<div class="empty">暂无示例</div>'

      // 5页滑窗
      const windowSize = 5
      let startPage = current - Math.floor(windowSize / 2)
      if (startPage < 1) startPage = 1
      let endPage = startPage + windowSize - 1
      if (endPage > totalPages) {
        endPage = totalPages
        startPage = Math.max(1, endPage - windowSize + 1)
      }
      const pages = []
      for (let p = startPage; p <= endPage; p++) pages.push(p)
      const pagerHtml = `
        <button class="pager-chip" type="button" data-action="first">首页</button>
        <button class="pager-chip" type="button" data-action="prev">上一页</button>
        ${pages.map(p => `<button class="pager-chip ${p===current?'active':''}" type="button" data-page="${p}">${p}</button>`).join('')}
        <button class="pager-chip" type="button" data-action="next">下一页</button>
        <button class="pager-chip" type="button" data-action="last">末页</button>
      `

      return `
        <div class="detail-section">
          <h4>基本信息</h4>
          <p><strong>关系类型:</strong> ${type}</p>
          <p><strong>名称:</strong> ${meta.name || type}</p>
          <p><strong>描述:</strong> ${meta.description || 'N/A'}</p>
          <p><strong>语义:</strong> ${meta.semantic || 'N/A'}</p>
        </div>
        <div class="detail-section">
          <h4>统计信息</h4>
          <div class="stats-row">
            <span class="stat-badge all ${filter==='all'?'active':''}"><span class="label">总边数：</span><span class="value">${stats.total || relationExamplesMaster.value.length}</span></span>
            <span class="stat-badge tri ${filter==='tri'?'active':''}"><span class="label">三角验证：</span><span class="value">${stats.triangulated_verified || 0}</span></span>
            <span class="stat-badge non ${filter==='non'?'active':''}"><span class="label">非三角：</span><span class="value">${stats.non_triangulated || 0}</span></span>
          </div>
        </div>
        <div class="detail-section">
          <h4>相关边列表</h4>
          <div class="edge-list" role="list">${examplesHtml}</div>
        </div>
      `
    }

    const showRelationDetails = (type, fromBack = false) => {
      // 在切换前，记录当前上下文到通用历史栈
      if (showModal.value && !fromBack) {
        if (edgeHeaderSource.value && edgeHeaderTarget.value) {
          modalHistory.value.push({ kind: 'edge', source: edgeHeaderSource.value, target: edgeHeaderTarget.value })
        } else if (currentNodeId.value) {
          modalHistory.value.push({ kind: 'node', nodeId: currentNodeId.value })
        } else if (currentWeightKey.value) {
          modalHistory.value.push({ kind: 'weight', key: currentWeightKey.value })
        } else if (currentParameterKey.value) {
          modalHistory.value.push({ kind: 'parameter', key: currentParameterKey.value })
        } else if (currentPathwayKey.value) {
          modalHistory.value.push({ kind: 'pathway', key: currentPathwayKey.value })
        }
      }

      currentRelationType.value = type
      relationExamplesPage.value = 1
      relationExamplesFilter.value = 'all'
      modalTitle.value = `关系类型详情: ${type}`
      modalContent.value = buildRelationDetailsContent(type)
      showModal.value = true
      nextTick(() => {
        if (modalBodyEl.value && !isModalBodyListenerBound) {
          modalBodyEl.value.addEventListener('click', handleModalClick)
          isModalBodyListenerBound = true
        }
      })
    }

    const buildWeightDetailsContent = (key) => {
      const details = weightDetailsCache.value || {}
      const base = details.base_weight || {}
      const cand = details.candidate_details || {}
      const hier = details.hierarchy_weight || {}
      const tri = details.triangulation_weights || {}
      const params = details.weight_params || {}
      const related = Array.isArray(details.related_edges) ? details.related_edges : []

      // 相关边分页（复用 relationExamples*）
      relationExamplesMaster.value = related.map(e => ({
        source: e.source || e[0] || e.source_node || e.src || '—',
        target: e.target || e[1] || e.target_node || e.dst || '—',
        edge_hierarchy: e.edge_hierarchy || e.hierarchy || '—'
      }))
      const pageSize = relationExamplesPageSize.value
      const current = relationExamplesPage.value
      const totalFiltered = relationExamplesMaster.value.length
      const totalPages = Math.max(1, Math.ceil(totalFiltered / pageSize))
      const sliceStart = (current - 1) * pageSize
      const slice = relationExamplesMaster.value.slice(sliceStart, sliceStart + pageSize)
      const examplesHtml = slice.length
        ? slice.map(ex => `<div class="edge-chip">${ex.source} → ${ex.target}</div>`).join('')
        : '<div class="empty">暂无相关边</div>'
      // 5页滑窗
      const windowSize = 5
      let startPage = current - Math.floor(windowSize / 2)
      if (startPage < 1) startPage = 1
      let endPage = startPage + windowSize - 1
      if (endPage > totalPages) {
        endPage = totalPages
        startPage = Math.max(1, endPage - windowSize + 1)
      }
      const pages = []
      for (let p = startPage; p <= endPage; p++) pages.push(p)
      const pagerHtml = `
        <button class="pager-chip" type="button" data-action="first">首页</button>
        <button class="pager-chip" type="button" data-action="prev">上一页</button>
        ${pages.map(p => `<button class="pager-chip ${p===current?'active':''}" type="button" data-page="${p}">${p}</button>`).join('')}
        <button class="pager-chip" type="button" data-action="next">下一页</button>
        <button class="pager-chip" type="button" data-action="last">末页</button>
      `

      const baseHtml = `
        <div class="detail-section">
          <h4>基础权重</h4>
          ${renderKvGrid({
            质量等级: mapQualityCn(base.quality || base.quality_level),
            综合评分: base.integrated_score
          })}
        </div>
      `

      // 中文映射：候选详情
      const candCn = {
        频次评分: cand.frequency_score,
        多样性评分: cand.diversity_score,
        综合评分: cand.comprehensive_score,
        算法一致性: cand.algorithm_consistency,
        网络拓扑: cand.network_topology,
        统计显著性: cand.statistical_significance,
        支持算法: Array.isArray(cand.support_algorithms) 
          ? mapAlgListCn(cand.support_algorithms).join('，') 
          : (cand.support_algorithms ? mapAlgListCn([cand.support_algorithms]).join('，') : '无')
      }
      const candHtml = `
        <div class="detail-section">
          <h4>候选详情</h4>
          ${renderKvGrid(candCn)}
        </div>
      `

      // 中文映射：层次权重
      const hierCn = {
        基础评分: hier.base_score,
        质量权重: hier.quality_weight,
        三角验证奖励: hier.triangulation_bonus,
        算法权重: hier.algorithm_weight,
        最终权重: hier.final_weight
      }
      const hierHtml = `
        <div class="detail-section">
          <h4>层次权重</h4>
          ${renderKvGrid(hierCn)}
        </div>
      `

      // 中文映射：三角验证权重（含四维评分）
      const four = tri.four_dimension_scores || {}
      const triCn = {
        联合置信度: tri.joint_confidence,
        质量调整置信度: tri.quality_adjusted_confidence,
        结构一致性: four.structural_consistency,
        参数拟合: four.parameter_fitting,
        中介支持: four.mediation_support,
        专家定向: four.expert_direction
      }
      const triHtml = `
        <div class="detail-section">
          <h4>三角验证权重</h4>
          ${renderKvGrid(triCn)}
        </div>
      `

      // 中文映射：权重参数
      const paramsCn = {
        基础权重: params.base_weight,
        候选权重: params.candidate_weight,
        三角权重: params.triangulation_weight
      }
      const paramsHtml = ''
      const relatedHtml = `
          <div class="edge-list" role="list">${examplesHtml}</div>
          <div class="pager" role="navigation">${pagerHtml}</div>
      `
      return baseHtml + candHtml + hierHtml + triHtml + relatedHtml
    }

    const showWeightDetails = async (key, fromBack = false) => {
      try {
        // 在切换前，记录当前上下文到通用历史栈
        if (showModal.value && !fromBack) {
          if (edgeHeaderSource.value && edgeHeaderTarget.value) {
            modalHistory.value.push({ kind: 'edge', source: edgeHeaderSource.value, target: edgeHeaderTarget.value })
          } else if (currentNodeId.value) {
            modalHistory.value.push({ kind: 'node', nodeId: currentNodeId.value })
          } else if (currentRelationType.value) {
            modalHistory.value.push({ kind: 'relation', type: currentRelationType.value })
          } else if (currentParameterKey.value) {
            modalHistory.value.push({ kind: 'parameter', key: currentParameterKey.value })
          } else if (currentPathwayKey.value) {
            modalHistory.value.push({ kind: 'pathway', key: currentPathwayKey.value })
          }
        }

        const resp = await store.getWeightDetails(key)
        const data = resp?.data ?? resp ?? {}
        weightDetailsCache.value = data
        currentWeightKey.value = key
        relationExamplesPage.value = 1
        modalTitle.value = `权重详情: ${key}`
        modalContent.value = buildWeightDetailsContent(key)
        showModal.value = true
        nextTick(() => {
          if (modalBodyEl.value && !isModalBodyListenerBound) {
            modalBodyEl.value.addEventListener('click', handleModalClick)
            isModalBodyListenerBound = true
          }
        })
      } catch (e) {
        console.error('获取权重详情失败', e)
        modalTitle.value = '权重详情加载失败'
        modalContent.value = `<p>无法获取权重详情，请稍后重试。</p>`
        showModal.value = true
      }
    }

    const showParameterDetails = async (key, summaryParam, fromBack = false) => {
      // 统一调用后端详情接口（加入键归一化与回退）
      try {
        // 在切换前，记录当前上下文到通用历史栈
        if (showModal.value && !fromBack) {
          if (edgeHeaderSource.value && edgeHeaderTarget.value) {
            modalHistory.value.push({ kind: 'edge', source: edgeHeaderSource.value, target: edgeHeaderTarget.value })
          } else if (currentNodeId.value) {
            modalHistory.value.push({ kind: 'node', nodeId: currentNodeId.value })
          } else if (currentRelationType.value) {
            modalHistory.value.push({ kind: 'relation', type: currentRelationType.value })
          } else if (currentWeightKey.value) {
            modalHistory.value.push({ kind: 'weight', key: currentWeightKey.value })
          } else if (currentPathwayKey.value) {
            modalHistory.value.push({ kind: 'pathway', key: currentPathwayKey.value })
          }
        }

        const normalizeParamKey = (k) => {
          if (k == null) return ''
          const s = String(k).trim()
          return s.replace(/\s*(?:→|—>|->|=>|➝|⟶)\s*/g, '->')
        }
        const uniq = (arr) => Array.from(new Set(arr.filter(Boolean)))
        const rawKey = key
        const primary = normalizeParamKey(rawKey)
        const candidates = uniq([
          primary,
          String(rawKey || primary).replace(/→/g, '->').trim(),
          String(rawKey || primary).replace(/->/g, '→').trim(),
        ])

        let useKey = null
        let resp = null
        let lastErr = null
        for (const k of candidates) {
          try {
            resp = await store.getParameterDetails(k)
            useKey = k
            break
          } catch (e) {
            lastErr = e
          }
        }
        if (!resp) throw lastErr || new Error('无法获取参数详情')
        const data = resp?.data ?? resp ?? {}
        console.log('CPT调试: 参数详情键解析', { rawKey, candidates, useKey })
        const sourceTag = data.source || 'Theta'
        const avail = data.available_methods || {}
        const summaries = data.method_summaries || {}
        const methods = data.methods || {}
        const methodEstimates = data.method_estimates || {}
        const stability = data.parameter_stability || {}
        const edgeCond = data.edge_conditional_prob || {}

        const badge = (ok) => ok ? '<span class="method-badge ok">可用</span>' : '<span class="method-badge no">不可用</span>'
        // 四位有效数字格式化（避免科学计数法，尽量保持直观）
        const fmt4 = (v) => {
          if (v === null || v === undefined) return 'N/A'
          const n = Number(v)
          if (!Number.isFinite(n)) return String(v)
          try {
            const s = n.toPrecision(4)
            // 保留尾随零，提升一致性显示
            return s
          } catch (e) {
            return String(v)
          }
        }
        const fmtCount = (v) => {
          if (v === null || v === undefined) return 'N/A'
          const n = Number(v)
          return Number.isFinite(n) ? String(Math.round(n)) : String(v)
        }
        const renderSumm = (m) => {
          const s = summaries[m] || {}
          const lines = []
          if (s.entries) lines.push(`键数: ${s.entries}`)
          if (s.preview_keys && s.preview_keys.length) lines.push(`预览: ${s.preview_keys.join(', ')}`)
          if (s.rows) lines.push(`行数: ${s.rows}`)
          if (s.type) lines.push(`类型: ${s.type}`)
          return lines.length ? `<div class="method-summary">${lines.join(' ｜ ')}</div>` : ''
        }
        const renderEstimatorInfo = (m) => {
          const mdata = methods[m] || {}
          const mest = methodEstimates[m] || {}
          const fields = [
            ['coefficient','系数'],
            ['coefficient_std_error','系数标准误'],
            ['intercept','截距'],
            ['r_squared','R²'],
            ['adjusted_r_squared','调整后R²'],
            ['mse','MSE'],
            ['rmse','RMSE'],
            ['t_statistic','t统计量'],
            ['data_quality','数据质量']
          ]
          const rows = []
          fields.forEach(([k, label]) => {
            const v = mdata?.[k]
            if (v !== undefined && v !== null) {
              let display = v
              if (Array.isArray(v)) display = v.join(', ')
              rows.push(`<li class="kv-item"><span class="kv-label">${label}</span><span class="kv-value">${fmt4(display)}</span></li>`)
            }
          })
          // 聚合方法级估计值（MLE/Bayesian/EM/SEM）
          const aggFields = [
            ['likelihood_gain','边际似然增益'],
            ['S_param','S参数分数']
          ]
          aggFields.forEach(([k,label]) => {
            const v = mest?.[k]
            if (v !== undefined && v !== null) {
              rows.push(`<li class="kv-item"><span class="kv-label">${label}</span><span class="kv-value">${fmt4(v)}</span></li>`)
            }
          })
          return rows.length
            ? `<ul class="kv-grid" role="list">${rows.join('')}</ul>`
            : '<p class="empty">暂无估计器信息</p>'
        }
        const renderCPTTable = (cpt, m) => {
          if (!cpt) return '<p class="empty">无CPT数据</p>'
          // 标准化为行列表
          let rows = []
          let headers = []
          if (Array.isArray(cpt)) {
            rows = cpt
            headers = Array.from(rows.reduce((set, r) => {
              Object.keys(r || {}).forEach(k => set.add(k))
              return set
            }, new Set()))
            // 过滤掉 intercept/截距 列
            headers = headers.filter(h => String(h).toLowerCase() !== 'intercept' && String(h) !== '截距')
          } else if (typeof cpt === 'object') {
            // dict：可能是 { state1: {..}, state2: {..} } 或 { parents: [...], table: [...] }
            if (Array.isArray(cpt.table)) {
              rows = cpt.table
              headers = Array.from(rows.reduce((set, r) => {
                Object.keys(r || {}).forEach(k => set.add(k))
                return set
              }, new Set()))
              // 过滤掉 intercept/截距 列
              headers = headers.filter(h => String(h).toLowerCase() !== 'intercept' && String(h) !== '截距')
            } else {
              // 更清晰的条件概率展示：父节点条件 + 目标状态概率
              const parentNames = Array.isArray(methods[m]?.parents) ? methods[m].parents : []
              // 仅显示目标节点名（解析 key 的右侧作为目标名），若方法数据有显式 target 则优先
              const targetName = (() => {
                const mdata = methods[m] || {}
                if (mdata && (mdata.target || mdata.target_node)) {
                  return mdata.target || mdata.target_node
                }
                const s = String(useKey || key || '')
                if (s) {
                  const parts = s.split(/->|→|—>|=>|➝|⟶/)
                  if (parts.length > 1) return parts[parts.length - 1].trim()
                }
                return (useKey || key || '目标')
              })()
              const entries = Object.entries(cpt)
              // 找到首个数组型分布（支持字符串数组如 "[0.1, 0.9]"）
              const parseMaybeArray = (v) => {
                if (Array.isArray(v)) return v
                if (typeof v === 'string') {
                  const s = v.trim()
                  if (s.startsWith('[') && s.endsWith(']')) {
                    try {
                      const arr = JSON.parse(s)
                      return Array.isArray(arr) ? arr : null
                    } catch (_) { /* ignore */ }
                  }
                }
                return null
              }
              const firstArr = entries.map(([,v]) => parseMaybeArray(v)).find(a => Array.isArray(a))
              if (Array.isArray(firstArr)) {
                // 构建表头
                const makeHeader = () => {
                  if (firstArr.length === 2) {
                    return `<thead><tr><th>父节点条件</th><th>${targetName} 不发生</th><th>${targetName} 发生</th></tr></thead>`
                  }
                  const cols = firstArr.map((_, idx) => `<th>${targetName} 状态${idx}</th>`).join('')
                  return `<thead><tr><th>父节点条件</th>${cols}</tr></thead>`
                }
                const labelState = (v) => {
                  const s = String(v)
                  // 条件列改为数值 0/1 显示，保留其他值原样
                  if (s === '0' || s === '不发生') return '0'
                  if (s === '1' || s === '发生') return '1'
                  return s
                }
                const formatDecimal4 = (num) => {
                  const n = Number(num)
                  if (!Number.isFinite(n)) return String(num)
                  return n.toFixed(4)
                }
                // 父条件显示：在单元格中仅展示状态序列（0/1），完整带名称的内容放在 title 提示
                const esc = (s) => String(s)
                  .replace(/&/g, '&amp;')
                  .replace(/</g, '&lt;')
                  .replace(/>/g, '&gt;')
                  .replace(/"/g, '&quot;')
                const buildCond = (combo) => {
                  const parts = String(combo).split(/\s*,\s*/)
                  const states = parts.map(p => labelState(p))
                  const short = states.join(', ')
                  let full = short
                  if (parentNames.length === parts.length && parentNames.length > 0) {
                    full = parentNames.map((name, i) => `${name}: ${labelState(parts[i])}`).join('， ')
                  } else if (!parts.length) {
                    full = `组合: ${combo}`
                  }
                  return { short, full }
                }
                const body = entries.map(([combo, dist]) => {
                  const cond = buildCond(combo)
                  const arr = parseMaybeArray(dist)
                  if (Array.isArray(arr)) {
                    const cols = arr.map((p) => `<td>${formatDecimal4(p)}</td>`).join('')
                    return `<tr><td title="${esc(cond.full)}">${cond.short}</td>${cols}</tr>`
                  }
                  // 非数组分布，退化为单值展示
                  return `<tr><td title="${esc(cond.full)}">${cond.short}</td><td colspan="${firstArr.length}">${typeof dist === 'object' ? JSON.stringify(dist) : String(dist)}</td></tr>`
                }).join('')
                return `
                  <table class="cpt-table">
                    ${makeHeader()}
                    <tbody>${body}</tbody>
                  </table>
                `
              }
              // 无法解析为数组分布时，退回通用键值表
              headers = ['键', '值']
              rows = entries.map(([k, v]) => ({ 键: k, 值: typeof v === 'object' ? JSON.stringify(v) : v }))
            }
          }
          if (!rows.length) return '<p class="empty">无CPT数据</p>'
          // 限制展示行数，避免过长
          const MAX_ROWS = 20
          const shown = rows.slice(0, MAX_ROWS)
          // 表头精简：若列名形如 "源->目标 不发生/发生"，仅显示右侧目标名
          const shortenHeader = (h) => {
            const s = String(h)
            const parts = s.split(/->|→|—>|=>|➝|⟶/)
            if (parts.length > 1) return parts[parts.length - 1].trim()
            return s
          }
          const headerHtml = headers.map(h => `<th>${shortenHeader(h)}</th>`).join('')
          const formatDecimal4 = (num) => {
            const n = Number(num)
            if (!Number.isFinite(n)) return String(num)
            return n.toFixed(4)
          }
          const formatCell = (val) => {
            if (val === null || val === undefined) return 'N/A'
            // 数组：逐个元素按四位小数
            if (Array.isArray(val)) {
              return `[${val.map(x => formatDecimal4(x)).join(', ')}]`
            }
            // 字符串：尝试解析成数组或数值
            if (typeof val === 'string') {
              const s = val.trim()
              if ((s.startsWith('[') && s.endsWith(']')) || (s.startsWith('{') && s.endsWith('}'))) {
                try {
                  const parsed = JSON.parse(s)
                  if (Array.isArray(parsed)) {
                    return `[${parsed.map(x => formatDecimal4(x)).join(', ')}]`
                  }
                } catch (e) {
                  // 非JSON字符串，继续按数值尝试
                }
              }
              const n = Number(s)
              if (Number.isFinite(n)) return n.toFixed(4)
              return s
            }
            // 数值：四位小数
            if (typeof val === 'number') return val.toFixed(4)
            return String(val)
          }
          const bodyHtml = shown.map(r => `<tr>${headers.map(h => `<td>${formatCell(r[h])}</td>`).join('')}</tr>`).join('')
          return `
            <table class="cpt-table">
              <thead><tr>${headerHtml}</tr></thead>
              <tbody>${bodyHtml}</tbody>
            </table>
            ${rows.length > MAX_ROWS ? `<div class="table-hint">仅展示前 ${MAX_ROWS} 行</div>` : ''}
          `
        }
        // CPT 状态信息：展示节点类型与父节点（不含完整CPT）
        const renderCPTStatus = (m) => {
          const mdata = methods[m] || {}
          const items = []
          const ntype = mdata.node_type
          const parents = mdata.parents
          if (ntype !== undefined && ntype !== null && ntype !== '') {
            items.push(`<li class="kv-item"><span class="kv-label">节点类型</span><span class="kv-value">${ntype}</span></li>`)
          }
          let pVal = null
          if (Array.isArray(parents) && parents.length) pVal = parents.join(', ')
          else if (typeof parents === 'string' && parents) pVal = parents
          else if (parents !== undefined && parents !== null) pVal = String(parents)
          if (pVal) {
            items.push(`<li class="kv-item"><span class="kv-label">父节点</span><span class="kv-value">${pVal}</span></li>`)
          }
          return items.length ? `<ul class="kv-grid" role="list">${items.join('')}</ul>` : ''
        }

        // SEM 参数卡片（仅在 SEM/结构方程模型时展示）
        const renderSEMParamCard = (m) => {
          const mdata = methods[m] || {}
          const isSEM = m === 'SEM' || (mdata.node_type && String(mdata.node_type).includes('structural_equation'))
          if (!isSEM) return ''
          const cpt = mdata.cpt_data || mdata.cpt || {}
          const mest = (methodEstimates && methodEstimates[m]) || {}
          const getVal = (k) => {
            // 优先从 CPT 对象取（常见情况），其次方法顶层，最后方法估计汇总
            if (cpt && typeof cpt === 'object' && !Array.isArray(cpt) && cpt[k] !== undefined) return cpt[k]
            if (mdata[k] !== undefined) return mdata[k]
            if (mest[k] !== undefined) return mest[k]
            return undefined
          }
          // 只展示你要求的六个字段
          const pairs = [
            ['系数', getVal('coefficient')],
            ['R²', getVal('r_squared')],
            ['标准误差', getVal('coefficient_std_error') ?? getVal('std_error')],
            ['t统计量', getVal('t_statistic') ?? getVal('t_value') ?? getVal('t')],
            ['MSE', getVal('mse') ?? getVal('mean_squared_error')],
            ['RMSE', getVal('rmse') ?? getVal('root_mean_squared_error')]
          ]
          const rows = pairs
            .filter(([,v]) => v !== undefined && v !== null)
            .map(([label, v]) => `<li class="kv-item"><span class="kv-label">${label}</span><span class="kv-value">${typeof v === 'number' ? v.toFixed(4) : fmt4(v)}</span></li>`)
            .join('')
          if (!rows) return ''
          return `
            <div class="info-card">
              <h5>结构方程模型参数</h5>
              <ul class="kv-grid" role="list">${rows}</ul>
            </div>
          `
        }
        const renderEdgeGains = (m) => {
          const gains = (methodEstimates?.[m]?.edge_likelihood_gain)
            || (methodEstimates?.edge_likelihood_gain?.[m])
            || null
          if (!gains || (typeof gains !== 'object')) return ''
          const entries = Object.entries(gains)
          if (!entries.length) return ''
          const rows = entries.map(([edge, val]) => `<tr><td>${edge}</td><td>${fmt4(val)}</td></tr>`).join('')
          return `
            <table class="metric-table">
              <thead><tr><th>边</th><th>似然增益</th></tr></thead>
              <tbody>${rows}</tbody>
            </table>
          `
        }
        const renderMethodStability = (m) => {
          const s = (stability?.method_scores?.[m]) || (stability?.by_method?.[m]) || null
          if (s === null || s === undefined) return '<p class="empty">无稳定性结果</p>'
          if (typeof s === 'number') {
            return `<ul class="kv-grid" role="list"><li class="kv-item"><span class="kv-label">分数</span><span class="kv-value">${fmt4(s)}</span></li></ul>`
          }
          const keys = Object.keys(s)
          if (!keys.length) return '<p class="empty">无稳定性结果</p>'
          const rows = keys.map(k => `<tr><td>${k}</td><td>${fmt4(s[k])}</td></tr>`).join('')
          return `
            <table class="metric-table">
              <thead><tr><th>指标</th><th>数值</th></tr></thead>
              <tbody>${rows}</tbody>
            </table>
          `
        }
        const renderCPTOverview = () => {
          const mnames = ['MLE','Bayesian','EM','SEM']
          const rows = mnames.map(m => {
            const mdata = methods[m] || {}
            const cpt = mdata.cpt_data
            const has = mdata.has_complete_cpt
            let rowCount = 'N/A'
            if (Array.isArray(cpt)) rowCount = cpt.length
            else if (typeof cpt === 'object' && Array.isArray(cpt?.table)) rowCount = cpt.table.length
            const pcount = Array.isArray(mdata.parents) ? mdata.parents.length : (mdata.parents ? 1 : 0)
            const ntype = mdata.node_type || '—'
            return `<tr>
              <td>${m}</td>
              <td>${has === undefined ? '未知' : (has ? '完整' : '不完整')}</td>
              <td>${fmtCount(rowCount)}</td>
              <td>${fmtCount(pcount)}</td>
              <td>${ntype}</td>
            </tr>`
          }).join('')
          return `
            <div class="detail-section">
              <h4>CPT 概览</h4>
              <table class="metric-table">
                <thead><tr><th>方法</th><th>完整性</th><th>行数</th><th>父节点数</th><th>节点类型</th></tr></thead>
                <tbody>${rows}</tbody>
              </table>
            </div>
          `
        }
        const renderEdgeCondProb = () => {
          if (!edgeCond || !Object.keys(edgeCond).length) return ''
          const rows = Object.entries(edgeCond).map(([edge, payload]) => {
            const prob = payload?.probability ?? payload?.cond_prob ?? payload?.p
            const influence = payload?.influence ?? payload?.type
            return `<tr><td>${edge}</td><td>${fmt4(prob)}</td><td>${influence || 'N/A'}</td></tr>`
          }).join('')
          return `
            <div class="detail-section">
              <h4>边条件概率与父影响</h4>
              <table class="metric-table">
                <thead><tr><th>边</th><th>P(目标|父)</th><th>影响类型</th></tr></thead>
                <tbody>${rows}</tbody>
              </table>
            </div>
          `
        }

        // 汇总：平均似然增益（跨方法）
        const computeAvgGainAll = () => {
          let values = []
          const mnames = ['MLE','Bayesian','EM','SEM']
          mnames.forEach(m => {
            const gains = (methodEstimates?.[m]?.edge_likelihood_gain)
              || (methodEstimates?.edge_likelihood_gain?.[m])
              || null
            if (gains && typeof gains === 'object') {
              Object.values(gains).forEach(v => {
                if (typeof v === 'number' && !isNaN(v)) values.push(v)
              })
            }
          })
          if (!values.length) return null
          const avg = values.reduce((a,b)=>a+b,0) / values.length
          return avg
        }

        // 方法级边际似然增益与SEM统计汇总
        const renderMethodGainSummary = () => {
          const m = methodEstimates || {}
          const mle = m?.MLE?.likelihood_gain
          const bayes = m?.Bayesian?.likelihood_gain
          const em = m?.EM?.likelihood_gain
          const semR2 = m?.SEM?.r_squared
          const semAdjR2 = m?.SEM?.adjusted_r_squared
          if ([mle, bayes, em, semR2, semAdjR2].every(v => v === undefined)) return ''
          // 按需求不展示该完整/边际似然汇总区块
          return ''
        }

        // 参数稳定性详细（总体）
        const renderStabilityOverallDetailed = () => {
          if (!stability || !Object.keys(stability).length) return ''
          const s = stability || {}
          const score = s.stability_score ?? s.overall_score ?? s.overall
          const mean = s.mean_score
          const std = s.std_score
          const cv = s.coefficient_of_variation
          const maxDiff = s.max_pairwise_diff
          const avgDiff = s.avg_pairwise_diff
          const nMethods = s.num_methods
          const level = s.consistency_level
          const ms = s.method_scores || {}
          const msHtml = ['MLE','Bayesian','EM','SEM']
            .filter(k => ms[k] !== undefined)
            .map(k => `<li class="kv-item"><span class="kv-label">${k}</span><span class="kv-value">${fmt4(ms[k])}</span></li>`)
            .join('')
          const levelBadge = level ? `<span class="consistency-badge ${String(level).toLowerCase()}">${level}</span>` : ''
          const overallCards = [
            score !== undefined ? { label: '稳定性评估', value: fmt4(score) } : null,
            mean !== undefined ? { label: '平均分数', value: fmt4(mean) } : null,
            std !== undefined ? { label: '标准差', value: fmt4(std) } : null,
            cv !== undefined ? { label: '变异系数', value: fmt4(cv) } : null,
            maxDiff !== undefined ? { label: '最大成对差异', value: fmt4(maxDiff) } : null,
            avgDiff !== undefined ? { label: '平均成对差异', value: fmt4(avgDiff) } : null,
            nMethods !== undefined ? { label: '方法数量', value: fmtCount(nMethods) } : null,
          ].filter(Boolean)
          const overallHtml = overallCards.map(item => `
            <li class="kv-item card">
              <span class="kv-label">${item.label}</span>
              <span class="kv-value">${item.value}</span>
            </li>
          `).join('')
          return `
            <div class="detail-section">
              <div class="section-head">
                <h4>参数稳定性</h4>
                ${levelBadge}
              </div>
              <ul class="kv-grid" role="list">${overallHtml}</ul>
              ${ms && Object.keys(ms).length ? `<h5 class="section-subtitle">方法分数</h5><ul class="kv-grid method-grid" role="list">${msHtml}</ul>` : ''}
            </div>
          `
        }

        // 顶部“参数总览”区块按需求移除
        let content = ``

        // 移除 CPT 概览区块
        // 参数稳定性（详细）移动到页面底部（在各方法详情之后）

        // 移除方法级似然增益汇总（完整/边际似然）
        // content += renderMethodGainSummary()

        // 各方法详情
        const methodsOrder = ['MLE','Bayesian','EM','SEM']
        methodsOrder.forEach(m => {
          const mdata = methods[m] || {}
          const cpt = mdata.cpt_data || mdata.cpt || null
          const cptRowCount = (() => {
            if (!cpt) return 0
            if (Array.isArray(cpt)) return cpt.length
            if (typeof cpt === 'object') {
              if (Array.isArray(cpt.table)) return cpt.table.length
              return Object.keys(cpt).length
            }
            return 0
          })()
          console.log('CPT调试: 参数方法CPT状态', { key: useKey ?? primary, method: m, hasCPT: !!cpt, rows: cptRowCount })
          const head = `
            <div class="method-header">
              <h4>${m} 方法</h4>
            </div>
          `
          const estimatorBlock = (m === 'SEM')
            ? ''
            : `
              <div class="subsection">
                <h5>估计器信息</h5>
                ${renderEstimatorInfo(m)}
              </div>
            `
          content += `
            <div class="detail-section">
              ${head}
              ${estimatorBlock}
              <div class="subsection">
                <h5>${m === 'SEM' ? 'SEM 条件概率表' : '条件概率表（CPT）'}</h5>
                ${renderCPTStatus(m)}
                ${renderCPTTable(cpt, m)}
                ${renderSEMParamCard(m)}
              </div>
              ${renderEdgeGains(m) ? `<div class="subsection">${renderEdgeGains(m)}</div>` : ''}
              <div class="subsection">
                <h5>参数稳定性</h5>
                ${renderMethodStability(m)}
              </div>
            </div>
          `
        })

        // 去掉 边条件概率与父影响 区块

        // 将总体“参数稳定性”区块追加到页面最底部
        content += renderStabilityOverallDetailed()

        currentParameterKey.value = useKey ?? primary
        modalTitle.value = `参数详情: ${useKey ?? primary}`
        modalContent.value = content
        showModal.value = true
      } catch (err) {
        const normalizeParamKey = (k) => {
          if (k == null) return ''
          const s = String(k).trim()
          return s.replace(/\s*(?:→|—>|->|=>|➝|⟶)\s*/g, '->')
        }
        const primary = normalizeParamKey(key)
        currentParameterKey.value = primary || key
        modalTitle.value = `参数详情: ${primary || key}`
        modalContent.value = `<div class="detail-section"><p>获取详情失败: ${err?.message || '未知错误'}</p></div>`
        showModal.value = true
      }
    }

    const onRefreshWeights = async () => {
      try {
        const { count } = await store.reloadWeights()
        store.statistics = { ...store.statistics, weights: count }
      } catch (e) {
        console.error('刷新权重集合失败', e)
      }
    }

    const onSearchParameters = async (q) => {
      try {
        const resp = await store.searchParameters(q)
        const result = resp?.data ?? resp ?? {}
        // 兼容数组或对象结果，统一转换为以参数键索引的对象
        const rawItems = result.items ?? result ?? {}
        let itemsObj = {}
        if (Array.isArray(rawItems)) {
          itemsObj = rawItems.reduce((acc, it, idx) => {
            const key = (it && (it.key ?? it.id ?? it.name)) ?? String(idx)
            const value = it?.data ?? it
            acc[key] = value
            return acc
          }, {})
        } else if (rawItems && typeof rawItems === 'object') {
          itemsObj = rawItems
        } else {
          itemsObj = {}
        }
        // 更新 store 的 parameters 以驱动子组件列表
        store.parameters = itemsObj
        // 同步统计数量（对象键数量）
        const count = Object.keys(itemsObj).length
        store.statistics = { ...store.statistics, parameters: count }
      } catch (error) {
        console.error('参数搜索失败:', error)
      }
    }

    const onRefreshParameters = async () => {
      try {
        const { count } = await store.reloadParameters()
        // 统计在 store.reloadParameters 内已同步，这里确保侧边栏立即反映
        store.statistics = { ...store.statistics, parameters: count }
      } catch (e) {
        console.error('刷新参数集合失败', e)
      }
    }

    const showPathwayDetails = async (payloadOrKey, maybePathway, fromBack = false) => {
      try {
        // 在切换前，记录当前上下文到通用历史栈
        if (showModal.value && !fromBack) {
          if (edgeHeaderSource.value && edgeHeaderTarget.value) {
            modalHistory.value.push({ kind: 'edge', source: edgeHeaderSource.value, target: edgeHeaderTarget.value })
          } else if (currentNodeId.value) {
            modalHistory.value.push({ kind: 'node', nodeId: currentNodeId.value })
          } else if (currentRelationType.value) {
            modalHistory.value.push({ kind: 'relation', type: currentRelationType.value })
          } else if (currentWeightKey.value) {
            modalHistory.value.push({ kind: 'weight', key: currentWeightKey.value })
          } else if (currentParameterKey.value) {
            modalHistory.value.push({ kind: 'parameter', key: currentParameterKey.value })
          }
        }

        const isObj = typeof payloadOrKey === 'object' && payloadOrKey !== null
        const rawKey = isObj ? payloadOrKey.key : payloadOrKey
        const summary = isObj ? (payloadOrKey.data || {}) : (maybePathway || {})

        const normalizePathKey = (k) => {
          if (k == null) return ''
          const s = String(k).trim()
          return s.replace(/\s*(?:→|—>|->|=>|➝|⟶)\s*/g, '->')
        }
        const uniq = (arr) => Array.from(new Set(arr.filter(Boolean)))
        const primary = normalizePathKey(rawKey)
        const candidates = uniq([
          primary,
          String(rawKey || primary).replace(/→/g, '->').trim(),
          String(rawKey || primary).replace(/->/g, '→').trim(),
        ])

        let useKey = null
        let resp = null
        let lastErr = null
        for (const k of candidates) {
          try {
            resp = await store.getPathwayDetails(k)
            useKey = k
            break
          } catch (e) {
            lastErr = e
          }
        }
        if (!resp) throw lastErr || new Error('无法获取路径详情')
        const data = resp?.data ?? resp ?? {}
        const core = data.core_paths ?? data.core ?? []
        const candidate = data.candidate_paths ?? data.candidate ?? []
        const coverage = data.coverage ?? summary.coverage
        const sig = data.significance_info || {}
        const mostSigId = data.most_significant_pathway_id ?? sig.most_significant_pathway
        const statsAll = data.effect_statistics || {}
        const hdiRanges = (data.confidence_intervals && data.confidence_intervals.hdi_ranges) || {}
        const dirSummary = (data.effect_directions && data.effect_directions.direction_summary) || {}
        const typeSummary = (data.mediation_types && data.mediation_types.type_summary) || {}
        const members = Array.isArray(data.pathway_membership) ? data.pathway_membership : []
        const effects = Array.isArray(data.mediation_effects_list) ? data.mediation_effects_list : []

        const formatEffect = (v) => {
          const n = Number(v)
          return Number.isFinite(n) ? n.toFixed(4) : 'N/A'
        }
        const formatConfidence = (v) => {
          if (v == null) return 'N/A'
          const n = Number(v)
          if (!Number.isFinite(n)) return 'N/A'
          const pct = n > 1 ? Math.min(100, Math.max(0, n)) : Math.min(1, Math.max(0, n)) * 100
          return `${Math.round(pct)}%`
        }
        const formatPath = (p) => Array.isArray(p) ? p.join(' → ') : String(p)
        const formatList = (arr) => Array.isArray(arr) && arr.length ? arr.join(', ') : 'N/A'
        // 表格构建工具
        const buildEffectSummaryTable = (list) => {
          if (!Array.isArray(list) || !list.length) return '<span class="empty">暂无</span>'
          const rows = list.map(e => `
            <div class=\"table-row cols-3\">
              <div class=\"table-cell\">${e.pathway_id ?? '—'}</div>
              <div class=\"table-cell\">${e.primary_effect_type || 'N/A'}</div>
              <div class=\"table-cell\">${e.effect_strength ?? 'N/A'}</div>
            </div>
          `).join('')
          return `
            <div class=\"simple-table\">
              <div class=\"table-header cols-3\">
                <div class=\"table-cell\">路径ID</div>
                <div class=\"table-cell\">主效应类型</div>
                <div class=\"table-cell\">效应强度</div>
              </div>
              <div class=\"table-body\">${rows}</div>
            </div>
          `
        }
        const buildHdiTable = (list) => {
          if (!Array.isArray(list) || !list.length) return '<span class=\"empty\">暂无</span>'
          const rows = list.map(x => `
            <div class=\"table-row cols-3\">
              <div class=\"table-cell\">${x.pathway_id ?? '—'}</div>
              <div class=\"table-cell\">${x.lower}</div>
              <div class=\"table-cell\">${x.upper}</div>
            </div>
          `).join('')
          return `
            <div class=\"simple-table\">
              <div class=\"table-header cols-3\">
                <div class=\"table-cell\">路径ID</div>
                <div class=\"table-cell\">下限</div>
                <div class=\"table-cell\">上限</div>
              </div>
              <div class=\"table-body\">${rows}</div>
            </div>
          `
        }

        const buildPathTable = (paths, title) => {
          if (!Array.isArray(paths) || !paths.length) return '<span class=\"empty\">暂无</span>'
          const rows = paths.map(p => `
            <div class=\"table-row cols-1\">
              <div class=\"table-cell\">${formatPath(p)}</div>
            </div>
          `).join('')
          return `
            <div class=\"simple-table\">
              <div class=\"table-header cols-1\">
                <div class=\"table-cell\">${title || '路径'}</div>
              </div>
              <div class=\"table-body\">${rows}</div>
            </div>
          `
        }

        const buildMembersTable = (members) => {
          if (!Array.isArray(members) || !members.length) return '<span class=\"empty\">暂无</span>'
          const rows = members.map(m => `
            <div class=\"table-row cols-3\">
              <div class=\"table-cell\">${m.pathway_id ?? '—'}</div>
              <div class=\"table-cell\">${m.role_in_pathway || '—'}</div>
              <div class=\"table-cell\">${formatConfidence(m.pathway_significance)}</div>
            </div>
          `).join('')
          return `
            <div class=\"simple-table\">
              <div class=\"table-header cols-3\">
                <div class=\"table-cell\">成员ID</div>
                <div class=\"table-cell\">角色</div>
                <div class=\"table-cell\">显著性</div>
              </div>
              <div class=\"table-body\">${rows}</div>
            </div>
          `
        }

        const coreHtml = (Array.isArray(core) && core.length)
          ? buildPathTable(core, '核心路径')
          : '<span class=\"empty\">暂无核心路径</span>'
        const candHtml = (Array.isArray(candidate) && candidate.length)
          ? buildPathTable(candidate, '候选路径')
          : '<span class=\"empty\">暂无候选路径</span>'

        const membersHtml = (Array.isArray(members) && members.length)
          ? buildMembersTable(members)
          : '<span class=\"empty\">暂无成员信息</span>'

        const effectsHtml = (Array.isArray(effects) && effects.length)
          ? effects.map(e => {
              const de = e.direct_effect || {}
              const ie = e.indirect_effect || {}
              const te = e.total_effect || {}
              const mrp = typeof e.mediation_ratio_percentage === 'number' ? `${e.mediation_ratio_percentage.toFixed(1)}%` : 'N/A'
              const mr = typeof e.mediation_ratio === 'number' ? e.mediation_ratio.toFixed(4) : 'N/A'
              return `
                <div class=\"detail-section\">
                  <h4>中介路径 ${e.pathway_id}${e.is_significant ? '（显著）' : ''}</h4>
                  <div class=\"kv-grid\">
                    <div class=\"kv-item\"><span class=\"kv-label\">描述:</span><span class=\"kv-value\">${e.description || '—'}</span></div>
                    <div class=\"kv-item\"><span class=\"kv-label\">显著性概率:</span><span class=\"kv-value\">${formatConfidence(e.significance_probability)}</span></div>
                    <div class=\"kv-item\"><span class=\"kv-label\">中介比例:</span><span class=\"kv-value\">${mrp}（${mr}）</span></div>
                    <div class=\"kv-item\"><span class=\"kv-label\">效应类型/强度:</span><span class=\"kv-value\">${e.primary_effect_type || 'N/A'} / ${e.effect_strength || 'N/A'}</span></div>
                    <div class=\"kv-item\"><span class=\"kv-label\">方向:</span><span class=\"kv-value\">间接: ${e.indirect_effect_direction || 'N/A'}，直接: ${e.direct_effect_direction || 'N/A'}</span></div>
                    <div class=\"kv-item\"><span class=\"kv-label\">中介类型:</span><span class=\"kv-value\">${e.mediation_type || 'N/A'}</span></div>
                    <div class=\"kv-item\"><span class=\"kv-label\">正/负概率:</span><span class=\"kv-value\">${formatConfidence(e.positive_effect_probability)} / ${formatConfidence(e.negative_effect_probability)}</span></div>
                    <div class=\"kv-item\"><span class=\"kv-label\">直接效应:</span><span class=\"kv-value\">${formatEffect(de.mean)} [${de.hdi_lower ?? '—'}, ${de.hdi_upper ?? '—'}]</span></div>
                    <div class=\"kv-item\"><span class=\"kv-label\">间接效应:</span><span class=\"kv-value\">${formatEffect(ie.mean)} [${ie.hdi_lower ?? '—'}, ${ie.hdi_upper ?? '—'}]</span></div>
                    <div class=\"kv-item\"><span class=\"kv-label\">总效应:</span><span class=\"kv-value\">${formatEffect(te.mean)} [${te.hdi_lower ?? '—'}, ${te.hdi_upper ?? '—'}]</span></div>
                    <div class=\"kv-item\"><span class=\"kv-label\">结论:</span><span class=\"kv-value\">${e.conclusion || '—'}</span></div>
                  </div>
                </div>
              `
            }).join('')
          : '<div class=\"detail-section\"><h4>中介路径详情</h4><span class=\"empty\">暂无详细记录</span></div>'

        currentPathwayKey.value = useKey ?? primary
        modalTitle.value = `路径详情: ${useKey ?? primary}`
        modalContent.value = `
          <div class=\"detail-section\">
            <h4>显著性与统计</h4>
            <div class=\"kv-grid\">
              <div class=\"kv-item\"><span class=\"kv-label\">最大显著性:</span><span class=\"kv-value\">${formatConfidence(sig.max_significance)}</span></div>
              <div class=\"kv-item\"><span class=\"kv-label\">最显著路径ID:</span><span class=\"kv-value\">${mostSigId ?? 'N/A'}</span></div>
              <div class=\"kv-item\"><span class=\"kv-label\">路径数:</span><span class=\"kv-value\">${statsAll.pathways_count ?? (effects.length || ((Array.isArray(core)?core.length:0) + (Array.isArray(candidate)?candidate.length:0)))}</span></div>
              <div class=\"kv-item\"><span class=\"kv-label\">显著路径数:</span><span class=\"kv-value\">${statsAll.significant_pathways_count ?? (Array.isArray(core)?core.length:0)}</span></div>
            </div>
          </div>
          <div class=\"detail-section\">
            <h4>路径效应综览</h4>
            ${buildEffectSummaryTable(effects)}
          </div>
          <div class=\"detail-section\">
            <h4>置信区间</h4>
            <div class=\"kv-item\"><span class=\"kv-label\">间接效应区间:</span></div>
            ${buildHdiTable(hdiRanges.indirect_effect)}
            <div class=\"kv-item\" style=\"margin-top:8px\"><span class=\"kv-label\">直接效应区间:</span></div>
            ${buildHdiTable(hdiRanges.direct_effect)}
            <div class=\"kv-item\" style=\"margin-top:8px\"><span class=\"kv-label\">总效应区间:</span></div>
            ${buildHdiTable(hdiRanges.total_effect)}
          </div>
          <div class=\"detail-section\">
            <h4>方向总结</h4>
            <div class=\"kv-grid\">
              <div class=\"kv-item\"><span class=\"kv-label\">间接方向:</span><span class=\"kv-value\">${formatList(dirSummary.indirect_directions)}</span></div>
              <div class=\"kv-item\"><span class=\"kv-label\">直接方向:</span><span class=\"kv-value\">${formatList(dirSummary.direct_directions)}</span></div>
              <div class=\"kv-item\"><span class=\"kv-label\">正向概率:</span><span class=\"kv-value\">${Array.isArray(dirSummary.positive_probabilities) ? dirSummary.positive_probabilities.map(formatConfidence).join(', ') : 'N/A'}</span></div>
              <div class=\"kv-item\"><span class=\"kv-label\">负向概率:</span><span class=\"kv-value\">${Array.isArray(dirSummary.negative_probabilities) ? dirSummary.negative_probabilities.map(formatConfidence).join(', ') : 'N/A'}</span></div>
            </div>
          </div>
          <div class=\"detail-section\">
            <h4>核心路径 (${Array.isArray(core) ? core.length : 0})</h4>
            ${coreHtml}
          </div>
          <div class=\"detail-section\">
            <h4>候选路径 (${Array.isArray(candidate) ? candidate.length : 0})</h4>
            ${candHtml}
          </div>
          <div class=\"detail-section\">
            <h4>路径成员 (${Array.isArray(members) ? members.length : 0})</h4>
            ${membersHtml}
          </div>
          ${effectsHtml}
        `
        showModal.value = true
        nextTick(() => {
          if (modalBodyEl.value && !isModalBodyListenerBound) {
            modalBodyEl.value.addEventListener('click', handleModalClick)
            isModalBodyListenerBound = true
          }
        })
      } catch (e) {
        console.error('获取路径详情失败', e)
        const normalizePathKey = (k) => {
          if (k == null) return ''
          const s = String(k).trim()
          return s.replace(/\s*(?:→|—>|->|=>|➝|⟶)\s*/g, '->')
        }
        const primary = normalizePathKey(isObj ? payloadOrKey?.key : payloadOrKey)
        currentPathwayKey.value = primary || (isObj ? payloadOrKey?.key : payloadOrKey)
        modalTitle.value = `路径详情: ${primary || (isObj ? payloadOrKey?.key : payloadOrKey)}`
        modalContent.value = '<p>无法获取路径详情，请稍后重试。</p>'
        showModal.value = true
      }
    }

    const closeModal = () => {
      // 移除事件委托，避免重复绑定
      if (modalBodyEl.value && isModalBodyListenerBound) {
        modalBodyEl.value.removeEventListener('click', handleModalClick)
        isModalBodyListenerBound = false
      }
      // 清理历史与当前节点
      modalHistory.value = []
      nodeHistory.value = []
      currentNodeId.value = ''
      currentRelationType.value = ''
      currentWeightKey.value = ''
      currentParameterKey.value = ''
      currentPathwayKey.value = ''
      edgeHeaderSource.value = ''
      edgeHeaderTarget.value = ''
      showModal.value = false
      modalTitle.value = ''
      modalContent.value = ''
    }

    const getNodeType = (node) => {
      if (node.startsWith('疾病_')) return '疾病'
      if (node.startsWith('药物_')) return '药物'
      if (node.startsWith('检验_')) return '检验'
      return '其他'
    }

    const userName = ref('')
    const loadUser = async () => {
      try {
        const res = await fetch('/api/auth/me')
        const json = await res.json()
        userName.value = (json && json.user && json.user.name) ? json.user.name : ''
      } catch (e) {}
    }

    // 生命周期
    onMounted(async () => {
      await loadCurrentDatasource()
      await refreshDatasourceList()
      await store.loadData()
      await loadUser()
    })

    // 图组件点击事件：节点与边详情（需在 setup 内定义并在 return 暴露）
    const onGraphNodeClick = (nodeId) => {
      showNodeDetails(nodeId)
    }
    const onGraphEdgeClick = (edgeObj) => {
      const e = edgeObj || {}
      showEdgeDetails({
        source: e.source,
        target: e.target,
        relation_type: e.relation_type,
        weight_ref: e.weight_ref,
        edge_hierarchy: e.edge_hierarchy
      })
    }

    return {
      activePanel,
      showModal,
      modalTitle,
      modalContent,
      modalBodyEl,
      nodeHistory,
      currentNodeId,
      NodeComponent,
      EdgeComponent,
      RelationComponent,
      WeightComponent,
      ParameterComponent,
      PathwayComponent,
      OverviewComponent,
      GraphComponent,
      AdminComponent,
      containerEl,
      sidebarWidth,
      startDrag,
      startTouchDrag,
      nodes,
      edges,
      relations,
      weights,
      parameters,
      pathways,
      nodeTypeStats,
      statistics,
      relationTypeStats,
      relationTypeStatsDetailed,
      hierarchyStats,
      availableRelationTypes,
      availableHierarchies,
      setActivePanel,
      showNodeDetails,
      showEdgeDetails,
      showRelationDetails,
      showWeightDetails,
      showParameterDetails,
      showPathwayDetails,
      onRefreshEdges,
      onRefreshRelations,
      onRefreshPathways,
      onRefreshParameters,
      onSearchParameters,
      onRefreshWeights,
      getNodeType,
      currentRelationType,
      relationExamplesMaster,
      relationExamplesPageSize,
      relationExamplesPage,
      goBack,
      canGoBack,
      closeModal,
      edgeHeaderSource,
      edgeHeaderTarget,
      onGraphNodeClick,
      onGraphEdgeClick,
      datasourceFiles,
      datasourceFilesDedup,
      selectedDatasourcePath,
      currentDatasource,
      fmtSize,
      refreshDatasourceList,
      applySelectedDatasource,
      onUploadChange,
      onDeleteDatasource,
      isUploadPath,
      dsOpen,
      selectedLabel,
      onSelectDatasource,
      dsConfirmVisible,
      dsConfirmName,
      onRequestDeleteDatasource,
      onCancelDeleteDatasource,
      onConfirmDeleteDatasource,
      onApplyFromDropdown,
      isCurrent,
      onLogout,
      goToUpload,
      userName
    }
  }
}
</script>

<style scoped>
.container {
  display: flex;
  height: 100vh;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', sans-serif;
}
.splitter {
  width: 8px;
  cursor: col-resize;
  background: linear-gradient(180deg, rgba(255,255,255,0.35), rgba(255,255,255,0.18));
}
.splitter:hover { background: linear-gradient(180deg, rgba(255,255,255,0.5), rgba(255,255,255,0.28)); }

.user-inline-bar {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 14px 18px;
  margin: 24px 12px 16px 12px;
  color: var(--sidebar-text);
  border: 1px solid var(--sidebar-border);
  background: linear-gradient(180deg, rgba(255,255,255,0.22), rgba(255,255,255,0.14));
  border-radius: 12px;
  box-shadow: 0 8px 18px rgba(0,0,0,0.18);
  transition: background .18s ease, transform .18s ease, box-shadow .18s ease, margin .18s ease;
}
.user-inline-bar:hover { box-shadow: 0 10px 22px rgba(0,0,0,0.22); }
.user-inline-bar:focus-within { outline: 2px solid var(--focus-outline); outline-offset: 2px; }
.user-inline-bar .avatar { width: 24px; height: 24px; border-radius: 50%; background: rgba(255,255,255,0.2); display: inline-flex; align-items: center; justify-content: center; font-size: 14px; }
.user-inline-bar .user-name { font-size: 14px; font-weight: 600; line-height: 1.4; opacity: .95; letter-spacing: .2px; }

@media (max-width: 640px) {
  .user-inline-bar { padding: 12px 14px; gap: 8px; margin: 20px 8px 12px 8px; }
  .user-inline-bar .user-name { font-size: 13px; }
}
.user-bar { position: fixed; top: 8px; right: 12px; display: flex; align-items: center; gap: 8px; z-index: 1000; }
.user-bar .avatar { width: 28px; height: 28px; border-radius: 50%; background: #f3f4f6; display: inline-flex; align-items: center; justify-content: center; font-size: 14px; }

.sidebar {
  width: clamp(260px, 24vw, 320px);
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  overflow-y: auto;
  --sidebar-text: #ffffff;
  --sidebar-muted: rgba(255,255,255,0.9);
  --sidebar-border: rgba(255,255,255,0.28);
  --panel-bg: rgba(255,255,255,0.12);
  --panel-border: rgba(255,255,255,0.18);
  --btn-bg: linear-gradient(180deg, rgba(255,255,255,0.22), rgba(255,255,255,0.14));
  --btn-border: rgba(255,255,255,0.35);
  --btn-bg-hover: rgba(255,255,255,0.28);
  --focus-outline: #93c5fd;
  --menu-bg: #ffffff;
  --menu-text: #111827;
  --menu-border: #e5e7eb;
  --menu-hover: #f5f7fb;
  --accent-blue: #1d4ed8;
  --accent-blue-weak: rgba(59,130,246,0.08);
  --accent-red: #b91c1c;
  --accent-red-weak: rgba(255, 80, 80, 0.12);
  --success-pill-bg: #ecfdf5;
  --success-pill-border: #86efac;
}

.datasource-panel {
  padding: 16px 14px;
  margin-bottom: 12px;
  background: rgba(255,255,255,0.12);
  border: 1px solid rgba(255,255,255,0.18);
  border-radius: 12px;
  backdrop-filter: saturate(140%) blur(6px);
}
.ds-row {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 6px;
}
.ds-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.sidebar-actions { display: flex; align-items: center; gap: 8px; }
.user-area { display: flex; align-items: center; gap: 8px; }
.avatar { width: 24px; height: 24px; border-radius: 50%; background: rgba(255,255,255,0.2); display: inline-flex; align-items: center; justify-content: center; font-size: 14px; }
.avatar.placeholder { color: #fff; }
.sidebar-actions { display: flex; align-items: center; gap: 8px; }
.user-area { display: flex; align-items: center; gap: 8px; }
.avatar { width: 24px; height: 24px; border-radius: 50%; background: rgba(255,255,255,0.2); display: inline-flex; align-items: center; justify-content: center; font-size: 14px; }
.avatar.placeholder { color: #fff; }
.ds-title {
  font-size: 13px;
  font-weight: 600;
  letter-spacing: .2px;
  text-transform: uppercase;
  color: rgba(255,255,255,0.92);
}
.ds-btn {
  padding: 6px 10px;
  border: 1px solid rgba(255, 255, 255, 0.35);
  background: linear-gradient(180deg, rgba(255,255,255,0.22), rgba(255,255,255,0.14));
  color: #fff;
  cursor: pointer;
  border-radius: 8px;
  transition: background .2s ease, transform .12s ease, box-shadow .2s ease;
}
.ds-btn:hover { background: rgba(255,255,255,0.28); box-shadow: 0 6px 14px rgba(0,0,0,0.18); }
.ds-btn:active { transform: translateY(1px); }
.ds-btn.sm { padding: 4px 8px; font-size: 12px; }
.ds-btn.back {
  border: 1px solid var(--btn-border);
  background: var(--btn-bg);
  color: #fff;
}
.ds-btn.back:hover {
  background: rgba(255,255,255,0.28);
  box-shadow: 0 6px 14px rgba(0,0,0,0.18);
}
.ds-btn.back:active { transform: translateY(1px); }
.current-path {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.85);
}
.ds-select {
  flex: 1;
  padding: 6px 8px;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.35);
  background: rgba(255, 255, 255, 0.2);
  color: #fff;
}
.ds-select option { color: #111827; }
.ds-dropdown { position: relative; flex: 1; }
.ds-dropdown-toggle { width: 100%; min-height: 40px; padding: 10px 12px; border: 1px solid var(--btn-border); background: var(--btn-bg); color: var(--sidebar-text); border-radius: 10px; text-align: left; display: flex; align-items: center; justify-content: space-between; transition: box-shadow .2s ease, border-color .2s ease; }
.ds-dropdown-toggle:hover { box-shadow: 0 6px 18px rgba(0,0,0,0.22); border-color: rgba(255,255,255,0.55); }
.ds-dropdown-toggle:hover { box-shadow: 0 6px 18px rgba(0,0,0,0.22); border-color: rgba(255,255,255,0.55); }
.ds-dropdown-toggle.open { box-shadow: 0 10px 24px rgba(0,0,0,0.28); }
.ds-caret { opacity: 0.9; transition: transform .16s ease; }
.ds-dropdown-toggle.open .ds-caret { transform: rotate(180deg); }
.ds-dropdown-menu { position: absolute; top: calc(100% + 6px); left: 0; right: 0; background: #ffffff; color: #111827; border-radius: 12px; box-shadow: 0 14px 28px rgba(0,0,0,0.14); padding: 8px; z-index: 10; max-height: 280px; overflow: auto; border: 1px solid #e5e7eb; animation: dsMenuIn .14s ease-out; }
@keyframes dsMenuIn { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: translateY(0); } }
.ds-dropdown-item { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 10px 12px; border-radius: 10px; }
.ds-dropdown-item:hover { background: #f5f7fb; }
.ds-item-select { background: transparent; border: none; color: #111827; text-align: left; flex: 1; cursor: pointer; font-size: 14px; }
.current-badge { margin-left: 8px; color: #16a34a; font-weight: 600; padding: 2px 8px; border-radius: 999px; border: 1px solid #86efac; background: #ecfdf5; font-size: 12px; }
.ds-item-actions { display: flex; gap: 8px; }
.ds-item-apply { padding: 6px 10px; border: 1px solid rgba(59,130,246,0.8); background: rgba(59,130,246,0.08); color: #1d4ed8; border-radius: 8px; cursor: pointer; transition: background .14s ease, transform .12s ease, box-shadow .14s ease; }
.ds-item-apply:hover { background: rgba(59,130,246,0.12); box-shadow: 0 3px 8px rgba(59,130,246,0.24); }
.ds-item-apply:active { transform: translateY(1px); }
.ds-item-apply.disabled { opacity: 0.6; cursor: not-allowed; box-shadow: none; }
.ds-item-delete { padding: 6px 10px; border: 1px solid rgba(255, 80, 80, 0.8); background: rgba(255, 80, 80, 0.12); color: #b91c1c; border-radius: 8px; cursor: pointer; transition: background .14s ease, transform .12s ease, box-shadow .14s ease; }
.ds-item-delete:hover { background: rgba(255, 80, 80, 0.18); box-shadow: 0 3px 8px rgba(185, 28, 28, 0.24); }
.ds-item-delete:active { transform: translateY(1px); }
.ds-confirm { margin-top: 10px; padding: 10px 12px; border-radius: 12px; background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.28); box-shadow: 0 8px 18px rgba(0,0,0,0.16) inset; }
.ds-confirm-text { font-size: 13px; color: #ffffff; margin-bottom: 6px; letter-spacing: .1px; }
.ds-confirm-name { font-size: 12px; color: #fff; opacity: 0.9; margin-bottom: 8px; }
.ds-confirm-actions { display: flex; gap: 8px; }
.ds-current { display: none; }
.ds-label { display: none; }
.ds-path { display: none; }

.header {
  padding: 28px 18px 22px 18px;
  text-align: center;
  border-bottom: 1px solid rgba(255, 255, 255, 0.12);
}

.header h1 {
  margin: 0 0 8px 0;
  font-size: 1.45em;
  font-weight: 700;
}

.header p {
  margin: 0;
  opacity: 0.85;
  font-size: 0.92em;
}

.nav-menu {
  padding: 20px 0;
}

.nav-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px 20px;
  cursor: pointer;
  transition: all 0.3s ease;
  border-left: 4px solid transparent;
}

.nav-item:hover {
  background: rgba(255, 255, 255, 0.1);
  border-left-color: rgba(255, 255, 255, 0.5);
}

.nav-item.active {
  background: rgba(255, 255, 255, 0.15);
  border-left-color: white;
}

.nav-item .icon {
  margin-right: 10px;
  font-size: 1.2em;
}

.count {
  background: rgba(255, 255, 255, 0.2);
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 0.8em;
  font-weight: 500;
}

.main-content {
  flex: 1;
  background: #f8f9fa;
  overflow: hidden;
}

.content-panel {
  height: 100%;
  overflow-y: auto;
  /* 去掉左右内边距，让内部白色面板铺满宽度 */
  padding: 0;
}

.panel {
  /* 让内部内容能够使用 100% 高度进行弹性布局 */
  height: 100%;
  display: flex;
  flex-direction: column;
  padding: 0;
  background: #ffffff;
  border: 1px solid #e9ecef;
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
}

.panel.active {
  display: block;
}

/* 面板顶部工具栏（通用） */
.panel .toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 16px 20px;
  border-bottom: 1px solid #f1f3f5;
  background: linear-gradient(180deg, #ffffff, #fbfbfb);
}

.toolbar .title {
  font-size: 16px;
  font-weight: 600;
  color: #2c3e50;
}

.toolbar .actions {
  display: flex;
  gap: 8px;
}

.toolbar .btn {
  padding: 8px 14px;
  border-radius: 8px;
  border: 1px solid #dee2e6;
  background: #fff;
  color: #34495e;
  font-size: 13px;
  cursor: pointer;
  transition: all .2s ease;
}

.toolbar .btn:hover {
  background: #f8f9fa;
}

.toolbar .btn.primary {
  border-color: #3b82f6;
  color: #fff;
  background: #3b82f6;
}

.toolbar .btn.primary:hover {
  background: #2563eb;
}

/* 模态框样式 */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.modal-content {
  background: white;
  border-radius: 8px;
  max-width: 600px;
  width: 90%;
  max-height: 80vh;
  overflow-y: auto;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  border-bottom: 1px solid #e9ecef;
}

.modal-header h3 {
  margin: 0;
  color: #2c3e50;
}

.title-with-badge {
  display: flex;
  align-items: baseline;
  gap: 8px;
}

.type-badge {
  display: inline-block;
  padding: 4px 8px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid #e5e7eb;
  background: #f9fafb;
  color: #374151;
}
.type-疾病 { border-color: #fecaca; background: #fff5f5; }
.type-药物 { border-color: #a5b4fc; background: #eef2ff; }
.type-检验 { border-color: #6ee7b7; background: #ecfdf5; }
.type-其他 { border-color: #e5e7eb; background: #f3f4f6; }

.modal-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.back-btn {
  padding: 6px 10px;
  border-radius: 8px;
  border: 1px solid #dee2e6;
  background: #fff;
  color: #34495e;
  font-size: 12px;
  cursor: pointer;
  transition: all .2s ease;
}
.back-btn:hover { background: #f8f9fa; }
.back-btn:disabled { opacity: .5; cursor: not-allowed; }

.close-btn {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: #6c757d;
  padding: 0;
  width: 30px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.close-btn:hover {
  color: #495057;
}

.modal-body {
  padding: 20px;
  transition: opacity .2s ease;
}
.modal-body.fade-out { opacity: 0; }
.modal-body.fade-in { opacity: 1; }

/* 让统计徽章可点击并提示交互 */
.modal-body :deep(.stat-badge) {
  cursor: pointer;
}
.modal-body :deep(.stat-badge:hover) {
  box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}

/* 深度选择器，确保作用到 v-html 内容 */
.modal-body :deep(.detail-section) {
  background: #ffffff;
  border: 1px solid #eef2f7;
  border-radius: 10px;
  padding: 14px 16px;
  margin-bottom: 14px;
  box-shadow: 0 6px 14px rgba(0,0,0,0.06);
  font-size: 14px;
}

/* 锚点滚动与临时高亮 */
.modal-body :deep(#sec-weight),
.modal-body :deep(#sec-parameter),
.modal-body :deep(#sec-mediation) {
  scroll-margin-top: 8px;
}
.modal-body :deep(.anchor-highlight) {
  box-shadow: 0 0 0 2px rgba(59,130,246,0.25) inset;
  transition: box-shadow .6s ease;
}

.modal-body :deep(.detail-section h4) {
  color: #374151;
  font-size: 20px;
  font-weight: 600;
  margin: 0 0 12px 0;
  border-bottom: 2px solid #e9ecef;
  padding-bottom: 6px;
}

.modal-body :deep(.section-subtitle) {
  color: #6b7280;
  font-size: 13px;
  margin: 6px 0 10px 0;
}

.detail-section p {
  margin: 8px 0;
  line-height: 1.5;
}

.detail-section strong {
  color: #2c3e50;
}

/* 详情弹窗增强样式 */
.modal-body :deep(.stats-row) {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.modal-body :deep(.stat-badge) {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid #e5e7eb;
  background: #f9fafb;
}
.modal-body :deep(.stat-badge .label) { color: #6b7280; }
.modal-body :deep(.stat-badge .value) { color: #111827; font-weight: 600; }
.modal-body :deep(.stat-badge.in) { border-color: #93c5fd; background: #eff6ff; }
.modal-body :deep(.stat-badge.out) { border-color: #fcd34d; background: #fffbeb; }
/* 关系详情配色：总边数/三角/非三角 */
.modal-body :deep(.stat-badge.all) { border-color: #93c5fd; background: #eff6ff; color: #0c4a6e; }
.modal-body :deep(.stat-badge.tri) { border-color: #86efac; background: #dcfce7; color: #065f46; }
.modal-body :deep(.stat-badge.non) { border-color: #fbbf24; background: #fef3c7; color: #92400e; }
.modal-body :deep(.stat-badge.active) { box-shadow: 0 0 0 2px rgba(59,130,246,0.15) inset; }

.modal-body :deep(.neighbor-group) { margin-top: 12px; }
.modal-body :deep(.group-title) { color: #374151; margin-bottom: 6px; }
.modal-body :deep(.neighbor-list) { display: flex; flex-wrap: wrap; gap: 10px; }
.modal-body :deep(.neighbor-list) { list-style: none; padding-left: 0; margin: 0; }
.modal-body :deep(.neighbor-chip) {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 10px;
  font-size: 12px;
  border: 1px solid #e5e7eb;
  background: #f9fafb;
  box-shadow: none;
  transition: background .15s ease;
  cursor: pointer;
}
.modal-body :deep(.neighbor-chip.in) { border-color: #93c5fd; background: #eff6ff; }
.modal-body :deep(.neighbor-chip.out) { border-color: #fcd34d; background: #fffbeb; }
.modal-body :deep(.neighbor-chip:hover) { background: #f3f4f6; }
.modal-body :deep(.empty) { color: #9ca3af; }

/* 三角验证边明细样式 */
.modal-body :deep(.edge-list) { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; list-style: none; padding-left: 0; }
.modal-body :deep(.edge-chip) {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 10px;
  font-size: 12px;
  border: 1px solid #e5e7eb;
  background: #f9fafb;
  box-shadow: none;
  cursor: default;
  transition: background .15s ease;
}
.modal-body :deep(.edge-chip:hover) { background: #f3f4f6; }
.modal-body :deep(.edge-chip.tri) { border-color: #a7f3d0; background: #ecfdf5; }

/* 详情弹窗——简洁表格与KV栅格 */
.modal-body :deep(.simple-table) { 
  width: 100%; 
  border: 1px solid #e5e7eb; 
  border-radius: 8px; 
  overflow: hidden; 
  background: #fff; 
}
.modal-body :deep(.table-header) { 
  display: grid; 
  gap: 0; 
  background: #f8fafc; 
  border-bottom: 1px solid #e5e7eb; 
  font-weight: 600; 
}
.modal-body :deep(.table-body) { display: block; }
.modal-body :deep(.table-row) { display: grid; gap: 0; }
.modal-body :deep(.table-cell) { 
  padding: 8px 10px; 
  font-size: 13px; 
  border-right: 1px solid #eef2f7; 
}
.modal-body :deep(.table-row .table-cell:last-child),
.modal-body :deep(.table-header .table-cell:last-child) { border-right: none; }
.modal-body :deep(.cols-1) { grid-template-columns: 1fr; }
.modal-body :deep(.cols-2) { grid-template-columns: 1fr 1fr; }
.modal-body :deep(.cols-3) { grid-template-columns: 140px 1fr 1fr; }

.modal-body :deep(.kv-grid) { 
  display: grid; 
  grid-template-columns: 1fr 1fr; 
  gap: 10px 24px; 
}
.modal-body :deep(ul.kv-grid) { list-style: none; padding-left: 0; margin: 0; }
.modal-body :deep(.kv-item) { display: flex; align-items: baseline; gap: 8px; }
.modal-body :deep(.kv-label) { color: #6b7280; font-size: 12px; }
.modal-body :deep(.kv-value) { font-size: 14px; font-weight: 600; color: #111827; }

.modal-body :deep(.metric-table) {
  width: 100%;
  border-collapse: collapse;
  border: none;
  border-radius: 8px;
  overflow: hidden;
  background: #fff;
}
.modal-body :deep(.metric-table thead th) {
  background: #f8fafc;
  border-bottom: 1px solid #e5e7eb;
  padding: 8px 10px;
  font-weight: 600;
  font-size: 13px;
  color: #374151;
  text-align: center;
}
.modal-body :deep(.metric-table td) {
  padding: 8px 10px;
  border-top: none;
  font-size: 13px;
  color: #111827;
  text-align: center;
  font-variant-numeric: tabular-nums;
}
.modal-body :deep(.metric-table tbody tr:nth-child(even) td) { background: #fafafa; }
.modal-body :deep(.metric-table tbody tr:hover td) { background: #f3f4f6; }
.modal-body :deep(.cpt-table) { width: 100%; border-collapse: collapse; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; background: #fff; }
.modal-body :deep(.cpt-table) { width: 100%; border-collapse: collapse; border: none; border-radius: 8px; overflow: hidden; background: #fff; }
.modal-body :deep(.cpt-table thead th) { background:#f8fafc; border-bottom:1px solid #e5e7eb; padding:8px 10px; font-weight:600; font-size:13px; color:#374151; text-align:center; }
.modal-body :deep(.cpt-table td) { padding:8px 10px; border-top:none; font-size:13px; color:#111827; text-align:center; font-variant-numeric: tabular-nums; }
.modal-body :deep(.cpt-table tbody tr:nth-child(even) td) { background: #fafafa; }
.modal-body :deep(.cpt-table tbody tr:hover td) { background: #f3f4f6; }
.modal-body :deep(.table-hint) { color:#6b7280; font-size:12px; margin-top:6px; }

/* 美化：方法头部与SEM信息卡 */
.modal-body :deep(.method-header) { display:flex; align-items:center; justify-content:space-between; gap:8px; }
.modal-body :deep(.method-header .sub) { color:#6b7280; font-size:12px; }
.modal-body :deep(.info-card) { background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px; padding:12px; margin-top:8px; }
.modal-body :deep(.info-card h5) { margin:0 0 8px 0; color:#1e3a8a; }

/* 美化：详情区块与子块 */
.modal-body :deep(.detail-section) {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 14px;
  margin-bottom: 16px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.05);
}
.modal-body :deep(.detail-section h4) {
  margin-top: 0;
  margin-bottom: 10px;
  font-weight: 600;
  color: #111827;
}
.modal-body :deep(.subsection) {
  background: #f9fafb;
  border: 1px solid #edf2f7;
  border-radius: 10px;
  padding: 12px;
  margin-top: 10px;
}

/* 美化：KV 网格项 */
.modal-body :deep(.kv-grid) { 
  display: grid; 
  grid-template-columns: 1fr 1fr; 
  gap: 12px 16px; 
}
.modal-body :deep(ul.kv-grid) { list-style: none; padding-left: 0; margin: 0; }
.modal-body :deep(.kv-item) {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 10px 12px;
  box-shadow: 0 4px 10px rgba(0,0,0,0.04);
}
.modal-body :deep(.kv-label) { color: #6b7280; font-size: 12px; }
.modal-body :deep(.kv-value) { font-size: 14px; font-weight: 600; color: #111827; }

/* 卡片式：方法分数与总体项区分间距 */
.modal-body :deep(.method-grid .kv-item) {
  background: #f9fafb;
}

/* 标题与一致性徽章 */
.modal-body :deep(.section-head) {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 8px;
}
.modal-body :deep(.consistency-badge) {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid #e5e7eb;
  background: #f9fafb;
  color: #374151;
}
.modal-body :deep(.consistency-badge.high) {
  border-color: #86efac;
  background: #ecfdf5;
  color: #065f46;
}
.modal-body :deep(.consistency-badge.medium) {
  border-color: #fde68a;
  background: #fffbeb;
  color: #92400e;
}
.modal-body :deep(.consistency-badge.low) {
  border-color: #fecaca;
  background: #fff5f5;
  color: #9f1239;
}

/* 简洁表格居中 */
.modal-body :deep(.simple-table .table-cell) { text-align: center; }

/* 页大小与分页 chips */
.modal-body :deep(.chip-group) { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
.modal-body :deep(.page-size-chip),
.modal-body :deep(.pager-chip) {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid #d1d5db;
  background: #f3f4f6;
  color: #374151;
  cursor: pointer;
  transition: all .15s ease;
}
.modal-body :deep(.page-size-chip.active),
.modal-body :deep(.pager-chip.active) {
  background: #e0f2fe;
  border-color: #7dd3fc;
  color: #0c4a6e;
}
.modal-body :deep(.chip-group.pager) { margin-top: 12px; }
.modal-body :deep(.pager-info) { color: #6b7280; font-size: 12px; margin-top: 4px; }

/* 模态标题中可点击的节点名 */
.link-node {
  color: #2563eb;
  cursor: pointer;
  padding: 2px 6px;
  border-radius: 6px;
  transition: all .15s ease;
}
.link-node:hover {
  background: #eff6ff;
}
.arrow { color: #6b7280; }

/* 操作按钮样式（权重/参数/中介） */
.modal-body :deep(.button-group) {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}
.modal-body :deep(.action-btn) {
  padding: 8px 14px;
  border-radius: 8px;
  border: 1px solid #dee2e6;
  background: #ffffff;
  color: #34495e;
  font-size: 13px;
  cursor: pointer;
  transition: all .2s ease;
}
.modal-body :deep(.action-btn:hover) {
  background: #f8f9fa;
  box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}

/* 彩色按钮主题 */
.modal-body :deep(.action-btn.weight) {
  background: #eef2ff;
  border-color: #c7d2fe;
  color: #3730a3;
}
.modal-body :deep(.action-btn.weight:hover) { background: #e0e7ff; }

.modal-body :deep(.action-btn.parameter) {
  background: #ecfdf5;
  border-color: #a7f3d0;
  color: #065f46;
}
.modal-body :deep(.action-btn.parameter:hover) { background: #d1fae5; }

.modal-body :deep(.action-btn.mediation) {
  background: #fff7ed;
  border-color: #fed7aa;
  color: #9a3412;
}
.modal-body :deep(.action-btn.mediation:hover) { background: #ffedd5; }
</style>
