<template>
  <div class="edges-panel">
    <!-- 顶部工具栏 -->
    <div class="toolbar">
      <div class="title">边集合 (E_core)</div>
      <div class="actions">
        <button class="btn" @click="clearFilters">清除筛选</button>
        <button class="btn primary" @click="refreshData">刷新数据</button>
      </div>
    </div>

    <!-- 搜索框 -->
    <div class="search-bar">
      <input 
        v-model="searchInput" 
        type="text" 
        class="search-input" 
        placeholder="搜索源/目标节点或权重引用..."
      >
      <button class="search-btn" @click="onSearch">
        🔍 搜索
      </button>
    </div>

    <div class="filter-section">
      <div class="filter-group">
        <label class="filter-label">关系类型筛选:</label>
        <div class="filter-options">
          <span 
            class="filter-option all"
            :class="{ active: selectedRelationTypes.length === 0 }"
            @click="selectAllRelationTypes"
          >
            全部
          </span>
          <span 
            v-for="type in relationTypes"
            :key="type"
            class="filter-option"
            :class="{ active: selectedRelationTypes.includes(type) }"
            @click="toggleRelationType(type)"
          >
            {{ type }}
          </span>
        </div>
      </div>
      
      <div class="filter-group">
        <label class="filter-label">边层次筛选:</label>
        <div class="filter-options">
          <span 
            class="filter-option all"
            :class="{ active: selectedHierarchies.length === 0 }"
            @click="selectAllHierarchies"
          >
            全部
          </span>
          <span 
            v-for="hierarchy in hierarchies"
            :key="hierarchy"
            class="filter-option"
            :class="{ active: selectedHierarchies.includes(hierarchy) }"
            @click="toggleHierarchy(hierarchy)"
          >
            <span :class="`hierarchy-indicator ${hierarchy}`"></span>
            {{ getHierarchyLabel(hierarchy) }}
          </span>
        </div>
      </div>
    </div>

    <div class="item-list">
      <div 
        v-for="edge in pagedEdges" 
        :key="`${edge.source}-${edge.target}`" 
        class="list-item"
        @click="showEdgeDetails(edge)"
      >
        <div class="item-title">
          <span :class="`hierarchy-indicator ${edge.edge_hierarchy}`"></span>
          {{ edge.source }} → {{ edge.target }}
        </div>
        <div class="item-details">
          <span :class="`relation-badge ${edge.relation_type.replace('_', '-')}`">
            {{ edge.relation_type }}
          </span>
          权重: {{ edge.weight_ref || 'N/A' }}
          {{ edge.is_direct ? '(直接)' : '(间接)' }}
        </div>
      </div>
    </div>

    <!-- 分页控件：统一为 5 页滑窗 + 首页/末页按钮，与权重系统一致 -->
    <div class="pager" v-if="totalPages > 1">
      <span class="pager-chip" :class="{ disabled: currentPage === 1 }" @click="goToFirst">首页</span>
      <span class="pager-chip" :class="{ disabled: currentPage === 1 }" @click="prevPage">上一页</span>
      <span
        v-for="p in displayPages"
        :key="`chip-${p}`"
        class="pager-chip"
        :class="{ active: p === currentPage }"
        @click="goToPage(p)"
      >{{ p }}</span>
      <span class="pager-chip" :class="{ disabled: currentPage === totalPages }" @click="nextPage">下一页</span>
      <span class="pager-chip" :class="{ disabled: currentPage === totalPages }" @click="goToLast">末页</span>
    </div>
    <div class="pager-jump" v-if="totalPages > 1">
      <input v-model="jumpInput" type="number" class="jump-input" :min="1" :max="totalPages" placeholder="页码" @keyup.enter="applyJump" />
      <button class="btn ghost" @click="applyJump">跳转</button>
      <span class="pager-info">共 {{ totalPages }} 页，每页 {{ pageSize }} 条</span>
    </div>
  </div>
</template>

<script>
import { ref, computed, watch } from 'vue'

export default {
  name: '边管理组件',
  props: {
    edges: {
      type: Array,
      default: () => []
    },
    // 主页面可传入已计算好的集合，未传时组件自行计算
    availableRelationTypes: {
      type: Array,
      default: () => []
    },
    availableHierarchies: {
      type: Array,
      default: () => []
    }
  },
  emits: ['show-edge-details', 'refresh-edges'],
  setup(props, { emit }) {
    const searchInput = ref('')
    const appliedSearch = ref('')
    const selectedRelationTypes = ref([])
    const selectedHierarchies = ref([])
    const pageSize = ref(10)
    const currentPage = ref(1)
    const jumpInput = ref('')

    // 合并主页面传入与本地推断
    const relationTypes = computed(() => {
      if (props.availableRelationTypes && props.availableRelationTypes.length > 0) {
        return props.availableRelationTypes
      }
      return [...new Set(props.edges.map(edge => edge.relation_type))]
    })

    const hierarchies = computed(() => {
      if (props.availableHierarchies && props.availableHierarchies.length > 0) {
        return props.availableHierarchies
      }
      return [...new Set(props.edges.map(edge => edge.edge_hierarchy))]
    })

    const filteredEdges = computed(() => {
      let filtered = props.edges

      // 关系类型筛选
      if (selectedRelationTypes.value.length > 0) {
        filtered = filtered.filter(edge => 
          selectedRelationTypes.value.includes(edge.relation_type)
        )
      }

      // 层次筛选
      if (selectedHierarchies.value.length > 0) {
        filtered = filtered.filter(edge => 
          selectedHierarchies.value.includes(edge.edge_hierarchy)
        )
      }

      // 搜索关键词筛选（源/目标/权重引用）
      const q = (appliedSearch.value || '').toLowerCase()
      if (q) {
        filtered = filtered.filter(edge => {
          const s = (edge.source || '').toLowerCase()
          const t = (edge.target || '').toLowerCase()
          const w = (edge.weight_ref || '').toLowerCase()
          return s.includes(q) || t.includes(q) || w.includes(q)
        })
      }

      return filtered
    })

    const totalPages = computed(() => {
      const total = Math.ceil(filteredEdges.value.length / pageSize.value)
      return total > 0 ? total : 1
    })

    const pagedEdges = computed(() => {
      const start = (currentPage.value - 1) * pageSize.value
      return filteredEdges.value.slice(start, start + pageSize.value)
    })

    // 统一 5 页滑动窗口
    const displayPages = computed(() => {
      const tp = totalPages.value
      const cur = currentPage.value
      const res = []
      if (tp <= 5) {
        for (let i = 1; i <= tp; i++) res.push(i)
        return res
      }
      let winStart = cur - 2
      let winEnd = cur + 2
      if (winStart < 1) {
        winStart = 1
        winEnd = 5
      } else if (winEnd > tp) {
        winEnd = tp
        winStart = tp - 4
      }
      for (let i = winStart; i <= winEnd; i++) res.push(i)
      return res
    })

    const toggleRelationType = (type) => {
      const index = selectedRelationTypes.value.indexOf(type)
      if (index > -1) {
        selectedRelationTypes.value.splice(index, 1)
      } else {
        selectedRelationTypes.value.push(type)
      }
      currentPage.value = 1
    }

    const toggleHierarchy = (hierarchy) => {
      const index = selectedHierarchies.value.indexOf(hierarchy)
      if (index > -1) {
        selectedHierarchies.value.splice(index, 1)
      } else {
        selectedHierarchies.value.push(hierarchy)
      }
      currentPage.value = 1
    }

    const selectAllRelationTypes = () => {
      selectedRelationTypes.value = []
      currentPage.value = 1
    }

    const selectAllHierarchies = () => {
      selectedHierarchies.value = []
      currentPage.value = 1
    }

    const clearFilters = () => {
      selectedRelationTypes.value = []
      selectedHierarchies.value = []
      searchInput.value = ''
      appliedSearch.value = ''
      currentPage.value = 1
    }

    const refreshData = () => {
      emit('refresh-edges')
    }

    const onSearch = () => {
      appliedSearch.value = (searchInput.value || '').trim()
      currentPage.value = 1
    }

    const getHierarchyLabel = (hierarchy) => {
      const labels = {
        'triangulated_verified': '三角验证',
        'non_triangulated': '非三角验证',
        'candidate_only': '候选边'
      }
      return labels[hierarchy] || hierarchy
    }

    const showEdgeDetails = (edge) => {
      emit('show-edge-details', edge)
    }

    // 约束当前页在过滤后的范围内
    watch(filteredEdges, () => {
      const max = totalPages.value
      if (currentPage.value > max) currentPage.value = max
      if (currentPage.value < 1) currentPage.value = 1
    })

    const goToPage = (page) => {
      if (page < 1 || page > totalPages.value) return
      currentPage.value = page
    }

    const prevPage = () => {
      if (currentPage.value > 1) currentPage.value -= 1
    }

    const nextPage = () => {
      if (currentPage.value < totalPages.value) currentPage.value += 1
    }

    const goToFirst = () => { currentPage.value = 1 }
    const goToLast = () => { currentPage.value = totalPages.value }

    const applyJump = () => {
      const n = parseInt(jumpInput.value, 10)
      if (!Number.isNaN(n)) {
        const to = Math.min(totalPages.value, Math.max(1, n))
        currentPage.value = to
      }
      jumpInput.value = ''
    }

    return {
      // 状态
      searchInput,
      appliedSearch,
      selectedRelationTypes,
      selectedHierarchies,
      // 供模板使用的集合
      relationTypes,
      hierarchies,
      // 计算与方法
      filteredEdges,
      pagedEdges,
      pageSize,
      currentPage,
      totalPages,
      toggleRelationType,
      toggleHierarchy,
      selectAllRelationTypes,
      selectAllHierarchies,
      clearFilters,
      refreshData,
      onSearch,
      getHierarchyLabel,
      showEdgeDetails,
      goToPage,
      prevPage,
      nextPage,
      displayPages,
      goToFirst,
      goToLast,
      jumpInput,
      applyJump
    }
  }
}
</script>

<style scoped>
.edges-panel {
  padding: 20px;
  background: #fff;
  /* 禁止组件自身出现横向滚动，内容交由外层页面滚动 */
  overflow-x: hidden !important;
  overflow-y: visible !important;
}

/* 工具栏与搜索样式，与节点集合保持一致 */
.toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 16px 0 16px;
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

.search-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px 12px 16px;
}

.search-input {
  flex: 1;
  padding: 10px 12px;
  border: 1px solid #dee2e6;
  border-radius: 8px;
  font-size: 13px;
}

.search-btn {
  padding: 8px 12px;
  border-radius: 8px;
  border: 1px solid #dee2e6;
  background: #fff;
  color: #34495e;
  font-size: 13px;
  cursor: pointer;
}

.search-btn:hover {
  background: #f8f9fa;
}

.filter-section {
  margin-bottom: 25px;
}

.filter-group {
  margin-bottom: 20px;
}

.filter-label {
  display: block;
  margin-bottom: 10px;
  font-weight: 500;
  color: #2c3e50;
}

.filter-options {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.filter-option {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 8px 15px;
  background: #f8f9fa;
  border: 1px solid #dee2e6;
  border-radius: 20px;
  cursor: pointer;
  transition: all 0.2s;
  font-size: 0.9em;
}

.filter-option.all {
  background: #fff;
}

.filter-option:hover {
  background: #e9ecef;
}

.filter-option.active {
  background: #007bff;
  color: white;
  border-color: #007bff;
}

.item-list {
  max-height: none;
  overflow: visible;
}

.list-item {
  padding: 15px;
  border: 1px solid #e9ecef;
  border-radius: 8px;
  margin-bottom: 10px;
  cursor: pointer;
  transition: all 0.2s;
}

.list-item:hover {
  background: #f8f9fa;
  border-color: #007bff;
  box-shadow: 0 2px 4px rgba(0,123,255,0.1);
}

.item-title {
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 500;
  color: #2c3e50;
  margin-bottom: 8px;
  font-size: 1.1em;
}

.item-details {
  display: flex;
  align-items: center;
  gap: 15px;
  font-size: 0.9em;
  color: #6c757d;
}

.hierarchy-indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  display: inline-block;
  flex-shrink: 0;
}

.hierarchy-indicator.triangulated_verified {
  background-color: #27ae60;
}

.hierarchy-indicator.non_triangulated {
  background-color: #f39c12;
}

.hierarchy-indicator.candidate_only {
  background-color: #95a5a6;
}

.relation-badge {
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 0.8em;
  font-weight: 500;
  background: #e9ecef;
  color: #495057;
}

.relation-badge.causal-relationship {
  background: #d4edda;
  color: #155724;
}

.relation-badge.correlation {
  background: #d1ecf1;
  color: #0c5460;
}

.relation-badge.association {
  background: #fff3cd;
  color: #856404;
}

.relation-badge.interaction {
  background: #f8d7da;
  color: #721c24;
}

/* 统一分页样式：与权重系统一致 */
.pager {
  display: flex;
  gap: 6px;
  justify-content: center;
  padding: 12px 16px;
}
.pager-chip {
  padding: 6px 10px;
  border-radius: 14px;
  border: 1px solid #dee2e6;
  background: #f8f9fa;
  cursor: pointer;
  font-size: 13px;
}
.pager-chip.active {
  background: #4c6ef5;
  color: #fff;
  border-color: #4c6ef5;
}
.pager-chip.disabled {
  opacity: .5;
  cursor: not-allowed;
}

.pager-jump {
  margin-top: 8px;
  display: flex;
  gap: 8px;
  justify-content: center;
  align-items: center;
}
.jump-input {
  width: 80px;
  padding: 6px 8px;
  border: 1px solid #dee2e6;
  border-radius: 8px;
  font-size: 13px;
}
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
.btn.ghost { background: transparent; color: #4c6ef5; border: 1px solid #4c6ef5; }
</style>