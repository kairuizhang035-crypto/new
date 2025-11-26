<template>
  <div class="nodes-panel">
    <!-- 顶部工具栏 -->
    <div class="toolbar">
      <div class="title">节点集合 (V)</div>
      <div class="actions">
        <button class="btn" @click="expandAll">展开全部</button>
        <button class="btn" @click="collapseAll">折叠全部</button>
        <button class="btn" @click="clearFilters">清除筛选</button>
        <button class="btn primary" @click="refreshData">刷新数据</button>
      </div>
    </div>
    
    <div class="search-bar">
      <input 
        v-model="searchInput" 
        type="text" 
        class="search-input" 
        placeholder="搜索节点..."
      >
      <button class="search-btn" @click="onSearch">
        🔍 搜索
      </button>
    </div>
    
    <!-- 节点类型筛选 -->
    <div class="filter-section">
      <h3>节点类型筛选:</h3>
      <div class="filter-options">
        <span 
          class="filter-option all"
          :class="{ active: selectedNodeTypes.length === 0 }"
          @click="selectAllTypes"
        >
          全部节点
        </span>
        <span 
          v-for="type in availableNodeTypes" 
          :key="type"
          class="filter-option"
          :class="{ active: selectedNodeTypes.includes(type) }"
          @click="toggleNodeType(type)"
        >
          {{ getNodeTypeLabel(type) }}
        </span>
      </div>
    </div>

    <!-- 节点类型统计 -->
    <div v-if="nodeTypeStats" class="stats-section">
      <div 
        v-for="(typeData, type) in nodeTypeStats" 
        :key="type"
        class="node-type-section"
        v-show="selectedNodeTypes.length === 0 || selectedNodeTypes.includes(type)"
      >
        <div class="type-header" @click="toggleTypeExpansion(type)">
          <span class="type-icon">{{ getNodeTypeIcon(type) }}</span>
          <span class="type-name">{{ getNodeTypeLabel(type) }}</span>
          <span class="type-count">{{ typeData.count }}</span>
          <span class="expand-icon" :class="{ expanded: expandedTypes.includes(type) }">▼</span>
        </div>
        
        <div v-if="expandedTypes.includes(type)" class="type-content">
          <div class="type-stats">
            <div class="stat-item">
              <span class="stat-label">平均入度:</span>
              <span class="stat-value">{{ typeData.avg_in_degree?.toFixed(1) || 0 }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">平均出度:</span>
              <span class="stat-value">{{ typeData.avg_out_degree?.toFixed(1) || 0 }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">平均总度:</span>
              <span class="stat-value">{{ typeData.avg_total_degree?.toFixed(1) || 0 }}</span>
            </div>
          </div>
          
          <div class="node-list">
            <div 
              v-for="node in getPagedNodesByType(type)" 
              :key="node.id" 
              class="list-item node-item"
              @click="showNodeDetails(node.id)"
            >
              <div class="item-title">{{ node.name }}</div>
              <div class="item-details">
                <span class="degree-info">
                  入度: {{ node.in_degree }} | 出度: {{ node.out_degree }} | 总度: {{ node.total_degree }}
                </span>
              </div>
            </div>
          </div>

          <!-- 分页器：每页5个，滑动窗口显示5个页码，支持首页/末页与页码跳转 -->
          <div class="pager">
            <button class="btn" @click="goFirst(type)" :disabled="getCurrentPage(type) === 1">首页</button>
            <button class="btn" @click="goPrev(type)" :disabled="getCurrentPage(type) === 1">上一页</button>
            <span 
              v-for="p in getPageWindow(type)" 
              :key="`page-${type}-${p}`" 
              class="page-number" 
              :class="{ active: p === getCurrentPage(type) }" 
              @click="setPage(type, p)"
            >{{ p }}</span>
            <button class="btn" @click="goNext(type)" :disabled="getCurrentPage(type) >= getTotalPages(type)">下一页</button>
            <button class="btn" @click="goLast(type)" :disabled="getCurrentPage(type) >= getTotalPages(type)">末页</button>
            <div class="pager-jump">
              <input 
                class="pager-input" 
                v-model="pageJumpByType[type]" 
                type="number" 
                min="1" 
                :max="getTotalPages(type)" 
                placeholder="页码"
              >
              <button class="btn" @click="jumpToPage(type)">跳转</button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 简单节点列表（当没有类型统计时的后备显示） -->
    <div v-if="!nodeTypeStats" class="item-list">
      <div 
        v-for="node in pagedSimpleNodes" 
        :key="getNodeId(node)" 
        class="list-item"
        @click="showNodeDetails(getNodeId(node))"
      >
        <div class="item-title">{{ getNodeName(node) }}</div>
        <div class="item-details">
          类型: {{ getNodeType(node) }}
        </div>
      </div>

      <!-- 后备列表分页器 -->
      <div class="pager">
        <button class="btn" @click="goFirstSimple" :disabled="getCurrentPageSimple() === 1">首页</button>
        <button class="btn" @click="goPrevSimple" :disabled="getCurrentPageSimple() === 1">上一页</button>
        <span 
          v-for="p in getPageWindowSimple" 
          :key="`simple-page-${p}`" 
          class="page-number" 
          :class="{ active: p === getCurrentPageSimple() }" 
          @click="setPageSimple(p)"
        >{{ p }}</span>
        <button class="btn" @click="goNextSimple" :disabled="getCurrentPageSimple() >= getTotalPagesSimple">下一页</button>
        <button class="btn" @click="goLastSimple" :disabled="getCurrentPageSimple() >= getTotalPagesSimple">末页</button>
        <div class="pager-jump">
          <input 
            class="pager-input" 
            v-model="pageJumpSimple" 
            type="number" 
            min="1" 
            :max="getTotalPagesSimple" 
            placeholder="页码"
          >
          <button class="btn" @click="jumpToPageSimple">跳转</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'

export default {
  name: '节点管理组件',
  props: {
    nodes: {
      type: Array,
      default: () => []
    },
    nodeTypeStats: {
      type: Object,
      default: null
    }
  },
  emits: ['show-node-details', 'refresh-nodes'],
  setup(props, { emit }) {
    const searchInput = ref('')
    const appliedSearch = ref('')
    const selectedNodeTypes = ref([])
    const expandedTypes = ref([])

    // 统一获取节点的可读名称与ID（兼容字符串或对象）
    const getNodeId = (node) => {
      if (!node) return ''
      if (typeof node === 'string') return node
      return node.id || node.name || ''
    }

    const getNodeName = (node) => {
      if (!node) return ''
      if (typeof node === 'string') return node
      return node.name || node.id || ''
    }

    // 优先从 nodeTypeStats 推断类型集合，降级使用 nodes
    const availableNodeTypes = computed(() => {
      if (props.nodeTypeStats) {
        return Object.keys(props.nodeTypeStats)
      }
      const types = new Set()
      props.nodes.forEach(node => {
        const id = getNodeId(node)
        if (id.startsWith('疾病_')) types.add('疾病')
        else if (id.startsWith('药物_')) types.add('药物')
        else if (id.startsWith('检验_')) types.add('检验')
        else types.add('其他')
      })
      return Array.from(types)
    })

    // 当 nodeTypeStats 不可用时的简化过滤（点击“搜索”后应用关键词；类型筛选为空表示全部）
    const filteredNodes = computed(() => {
      let list = props.nodes
      const q = (appliedSearch.value || '').toLowerCase()
      if (q) {
        list = list.filter(node => {
          if (typeof node === 'string') {
            return node.toLowerCase().includes(q)
          }
          const name = (node.name || '').toLowerCase()
          const id = (node.id || '').toLowerCase()
          return name.includes(q) || id.includes(q)
        })
      }
      if (selectedNodeTypes.value.length > 0) {
        list = list.filter(node => selectedNodeTypes.value.includes(getNodeType(node)))
      }
      return list
    })

    const toggleNodeType = (type) => {
      // 多选逻辑：点击某类型即切换选中状态；再次点击取消该类型；
      // 若需要“全部节点”，点击上方“全部节点”按钮即可清空选择
      const idx = selectedNodeTypes.value.indexOf(type)
      if (idx > -1) {
        selectedNodeTypes.value.splice(idx, 1)
      } else {
        selectedNodeTypes.value.push(type)
      }
    }

    const toggleTypeExpansion = (type) => {
      const index = expandedTypes.value.indexOf(type)
      if (index > -1) {
        expandedTypes.value.splice(index, 1)
      } else {
        expandedTypes.value.push(type)
      }
    }
    const selectAllTypes = () => {
      selectedNodeTypes.value = []
    }

    const expandAll = () => {
      const types = availableNodeTypes.value
      expandedTypes.value = [...types]
    }

    const collapseAll = () => {
      expandedTypes.value = []
    }

    const clearFilters = () => {
      selectedNodeTypes.value = []
      searchInput.value = ''
      appliedSearch.value = ''
    }

    const refreshData = () => {
      emit('refresh-nodes')
    }
    const onSearch = () => {
      appliedSearch.value = (searchInput.value || '').trim()
    }

    const getNodeTypeLabel = (type) => {
      const labels = {
        '疾病': '疾病节点',
        '药物': '药物节点',
        '检验': '检验节点',
        '其他': '其他节点',
        'other': '其他节点'
      }
      return labels[type] || type
    }

    const getNodeTypeIcon = (type) => {
      const icons = {
        '疾病': '🏥',
        '药物': '💊',
        '检验': '🔬',
        '其他': '📋',
        'other': '📋'
      }
      return icons[type] || '●'
    }

    const getNodeType = (node) => {
      const id = getNodeId(node)
      if (id.startsWith('疾病_')) return '疾病'
      if (id.startsWith('药物_')) return '药物'
      if (id.startsWith('检验_')) return '检验'
      return '其他'
    }

    // 基于 nodeTypeStats 的带搜索与类型选中过滤（点击“搜索”后应用关键词）
    const getFilteredNodesByType = (type) => {
      if (!props.nodeTypeStats || !props.nodeTypeStats[type]) return []
      let nodes = props.nodeTypeStats[type].nodes || []
      // 应用搜索关键词（点击“搜索”后）
      if (appliedSearch.value) {
        const q = appliedSearch.value.toLowerCase()
        nodes = nodes.filter(n => 
          (n.name && n.name.toLowerCase().includes(q)) ||
          (n.id && n.id.toLowerCase().includes(q))
        )
      }
      // 应用类型多选（若有选择则只展示被选类型，否则展示全部）
      if (selectedNodeTypes.value.length > 0 && !selectedNodeTypes.value.includes(type)) {
        return []
      }
      return nodes
    }

    // =========================
    // 类型分组列表的分页（每页5个）
    // =========================
    const pageSize = 5
    const currentPageByType = ref({})
    const pageJumpByType = ref({})

    const getTotalPages = (type) => {
      const totalItems = getFilteredNodesByType(type).length
      return Math.max(1, Math.ceil(totalItems / pageSize))
    }

    const getCurrentPage = (type) => {
      const total = getTotalPages(type)
      const cur = currentPageByType.value[type] || 1
      return Math.min(Math.max(cur, 1), total)
    }

    const setPage = (type, page) => {
      const total = getTotalPages(type)
      currentPageByType.value[type] = Math.min(Math.max(page, 1), total)
    }

    const goFirst = (type) => setPage(type, 1)
    const goLast = (type) => setPage(type, getTotalPages(type))
    const goPrev = (type) => setPage(type, getCurrentPage(type) - 1)
    const goNext = (type) => setPage(type, getCurrentPage(type) + 1)

    const getPagedNodesByType = (type) => {
      const all = getFilteredNodesByType(type)
      const cur = getCurrentPage(type)
      const start = (cur - 1) * pageSize
      return all.slice(start, start + pageSize)
    }

    // 滑动窗口：显示从当前页开始的最多5个页码，例如：第2页显示 2 3 4 5 6
    const getPageWindow = (type) => {
      const total = getTotalPages(type)
      const start = getCurrentPage(type)
      const end = Math.min(start + 4, total)
      const pages = []
      for (let p = start; p <= end; p++) pages.push(p)
      return pages
    }

    const jumpToPage = (type) => {
      const v = parseInt(pageJumpByType.value[type], 10)
      if (!isNaN(v)) setPage(type, v)
      pageJumpByType.value[type] = ''
    }

    // =========================
    // 简化列表的分页（每页5个）
    // =========================
    const currentPageSimple = ref(1)
    const pageJumpSimple = ref('')

    const getTotalPagesSimple = computed(() => {
      const totalItems = filteredNodes.value.length
      return Math.max(1, Math.ceil(totalItems / pageSize))
    })

    const getCurrentPageSimple = () => {
      const total = getTotalPagesSimple.value
      const cur = currentPageSimple.value || 1
      return Math.min(Math.max(cur, 1), total)
    }

    const setPageSimple = (page) => {
      const total = getTotalPagesSimple.value
      currentPageSimple.value = Math.min(Math.max(page, 1), total)
    }

    const goFirstSimple = () => setPageSimple(1)
    const goLastSimple = () => setPageSimple(getTotalPagesSimple.value)
    const goPrevSimple = () => setPageSimple(getCurrentPageSimple() - 1)
    const goNextSimple = () => setPageSimple(getCurrentPageSimple() + 1)

    const pagedSimpleNodes = computed(() => {
      const cur = getCurrentPageSimple()
      const start = (cur - 1) * pageSize
      return filteredNodes.value.slice(start, start + pageSize)
    })

    const getPageWindowSimple = computed(() => {
      const start = getCurrentPageSimple()
      const end = Math.min(start + 4, getTotalPagesSimple.value)
      const arr = []
      for (let p = start; p <= end; p++) arr.push(p)
      return arr
    })

    const jumpToPageSimple = () => {
      const v = parseInt(pageJumpSimple.value, 10)
      if (!isNaN(v)) setPageSimple(v)
      pageJumpSimple.value = ''
    }

    const showNodeDetails = (nodeId) => {
      emit('show-node-details', nodeId)
    }

    return {
      searchInput,
      appliedSearch,
      selectedNodeTypes,
      expandedTypes,
      availableNodeTypes,
      filteredNodes,
      toggleNodeType,
      toggleTypeExpansion,
      selectAllTypes,
      getNodeTypeLabel,
      getNodeTypeIcon,
      getNodeType,
      getNodeId,
      getNodeName,
      getFilteredNodesByType,
      getPagedNodesByType,
      pageSize,
      currentPageByType,
      pageJumpByType,
      getTotalPages,
      getCurrentPage,
      setPage,
      goFirst,
      goLast,
      goPrev,
      goNext,
      getPageWindow,
      jumpToPage,
      // 简单列表分页
      currentPageSimple,
      pageJumpSimple,
      getTotalPagesSimple,
      getCurrentPageSimple,
      setPageSimple,
      goFirstSimple,
      goLastSimple,
      goPrevSimple,
      goNextSimple,
      pagedSimpleNodes,
      getPageWindowSimple,
      jumpToPageSimple,
      showNodeDetails,
      expandAll,
      collapseAll,
      clearFilters,
      refreshData,
      onSearch
    }
  }
}
</script>

<style scoped>
.nodes-panel {
  padding: 20px;
  /* 面板自身不产生横向滚动，由父页面统一滚动 */
  overflow-x: hidden !important;
  overflow-y: visible !important;
  background: #fff;
  /* 高度控制交由父级容器，避免局部滚动条影响视觉宽度 */
  max-height: unset;
}

.toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 16px 0 16px;
  /* 与下方搜索栏拉开距离，避免视觉拥挤 */
  margin-bottom: 12px;
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

.search-box {
  width: 100%;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 5px;
  margin-bottom: 20px;
  font-size: 14px;
}

.search-bar {
  display: flex;
  gap: 10px;
  margin-bottom: 16px;
}

.search-input {
  flex: 1;
  padding: 10px 12px;
  border: 1px solid #dee2e6;
  border-radius: 8px;
  font-size: 14px;
  transition: border-color .2s ease, box-shadow .2s ease;
}

.search-input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
}

.search-btn {
  padding: 10px 16px;
  border-radius: 8px;
  border: 1px solid #dee2e6;
  background: #fff;
  color: #34495e;
  font-size: 14px;
  cursor: pointer;
  transition: all .2s ease;
}

.search-btn:hover {
  background: #f8f9fa;
}

/* 新搜索栏样式 */
.search-bar {
  display: flex;
  gap: 10px;
  margin-bottom: 16px;
}

.search-input {
  flex: 1;
  padding: 10px 12px;
  border: 1px solid #dee2e6;
  border-radius: 8px;
  font-size: 14px;
  transition: border-color .2s ease, box-shadow .2s ease;
}

.search-input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
}

.search-btn {
  padding: 10px 16px;
  border-radius: 8px;
  border: 1px solid #dee2e6;
  background: #fff;
  color: #34495e;
  font-size: 14px;
  cursor: pointer;
  transition: all .2s ease;
}

.search-btn:hover {
  background: #f8f9fa;
}

.search-btn.primary {
  border-color: #3b82f6;
  color: #fff;
  background: #3b82f6;
}

.search-btn.primary:hover {
  background: #2563eb;
}

.filter-section {
  margin-bottom: 25px;
}

.filter-section h3 {
  margin-bottom: 10px;
  color: #2c3e50;
}

.filter-options {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.filter-option {
  padding: 8px 15px;
  background: #f8f9fa;
  border: 1px solid #dee2e6;
  border-radius: 20px;
  cursor: pointer;
  transition: all 0.2s;
  font-size: 0.9em;
}

.filter-option:hover {
  background: #e9ecef;
}

.filter-option.active {
  background: #007bff;
  color: white;
  border-color: #007bff;
}

/* 全部节点选项的特殊样式 */
.filter-option.all {
  background: linear-gradient(135deg, #f0f5ff 0%, #e6f0ff 100%);
  border-color: #cfe0ff;
}

.filter-option.all.active {
  background: #007bff;
  color: #fff;
  border-color: #007bff;
}

.filter-hint {
  align-self: center;
  color: #6c757d;
  font-size: 0.85em;
}

.stats-section {
  margin-top: 20px;
}

.node-type-section {
  margin-bottom: 20px;
  border: 1px solid #e9ecef;
  border-radius: 8px;
  overflow: hidden;
}

.type-header {
  display: flex;
  align-items: center;
  padding: 15px;
  background: #f8f9fa;
  cursor: pointer;
  transition: background-color 0.2s;
  position: sticky;
  top: 0;
  z-index: 1;
}

.type-header:hover {
  background: #e9ecef;
}

.type-icon {
  font-size: 1.2em;
  margin-right: 10px;
}

.type-name {
  flex: 1;
  font-weight: 500;
}

.type-count {
  background: #007bff;
  color: white;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 0.8em;
  margin-right: 10px;
}

.expand-icon {
  transition: transform 0.2s;
}

.expand-icon.expanded {
  transform: rotate(180deg);
}

.type-content {
  padding: 15px;
}

.type-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 15px;
  margin-bottom: 15px;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  padding: 10px;
  background: #f8f9fa;
  border-radius: 5px;
}

.stat-label {
  color: #6c757d;
  font-size: 0.9em;
}

.stat-value {
  font-weight: 500;
  color: #495057;
}

.node-list {
  /* 让类型块下的列表根据内容自然撑开，由外层滚动 */
  max-height: none;
}

.item-list {
  max-height: none;
}

.list-item {
  padding: 12px;
  border: 1px solid #e9ecef;
  border-radius: 5px;
  margin-bottom: 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.list-item:hover {
  background: #f8f9fa;
  border-color: #007bff;
}

.item-title {
  font-weight: 500;
  color: #2c3e50;
  margin-bottom: 5px;
}

.item-details {
  font-size: 0.9em;
  color: #6c757d;
}

.degree-info {
  font-family: monospace;
}

/* ========================= */
/* 分页器样式（与按钮风格保持一致） */
/* ========================= */
.pager {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center; /* 水平居中 */
  gap: 8px;
  margin: 48px 0 40px; /* 往下挪一些，并在底部留出空间 */
}

/* 通用按钮风格（用于分页器） */
.btn {
  padding: 8px 14px;
  border-radius: 8px;
  border: 1px solid #dee2e6;
  background: #fff;
  color: #34495e;
  font-size: 13px;
  cursor: pointer;
  transition: all .2s ease;
}

.btn:hover { background: #f8f9fa; }
.btn.primary { border-color: #3b82f6; color: #fff; background: #3b82f6; }
.btn.primary:hover { background: #2563eb; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }

.page-number {
  padding: 6px 10px;
  border: 1px solid #dee2e6;
  border-radius: 6px;
  cursor: pointer;
  color: #34495e;
  transition: all .2s ease;
}
.page-number:hover { background: #f8f9fa; }
.page-number.active { background: #3b82f6; border-color: #3b82f6; color: #fff; }

.pager-jump {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-left: 8px;
}

.pager-input {
  width: 70px;
  padding: 6px 8px;
  border: 1px solid #dee2e6;
  border-radius: 6px;
}
</style>
