<template>
  <div class="weights-panel">
    <div class="panel-title">⚖️ 权重系统 (W)</div>
    <div class="search-bar">
      <input 
        v-model="searchInput" 
        type="text" 
        class="search-input" 
        placeholder="搜索权重关系..."
        @keyup.enter="onSearch"
      >
      <button class="search-btn" @click="onSearch">🔍 搜索</button>
      <button class="btn primary" @click="onRefresh">刷新数据</button>
    </div>
    <div class="filters-bar">
      <div class="filter-group">
        <label class="filter-label">质量等级:</label>
        <div class="filter-options">
          <span class="filter-option" :class="{ active: selectedQualities.length === 0 }" @click="clearQualities">全部</span>
          <span class="filter-option" :class="{ active: selectedQualities.includes('platinum') }" @click="toggleQuality('platinum')">
            <span class="quality-indicator quality-platinum"></span> platinum
          </span>
          <span class="filter-option" :class="{ active: selectedQualities.includes('gold') }" @click="toggleQuality('gold')">
            <span class="quality-indicator quality-gold"></span> gold
          </span>
          <span class="filter-option" :class="{ active: selectedQualities.includes('silver') }" @click="toggleQuality('silver')">
            <span class="quality-indicator quality-silver"></span> silver
          </span>
          <span class="filter-option" :class="{ active: selectedQualities.includes('bronze') }" @click="toggleQuality('bronze')">
            <span class="quality-indicator quality-bronze"></span> bronze
          </span>
        </div>
      </div>
      <div class="filter-group">
        <label class="filter-label">算法来源:</label>
        <div class="filter-options">
          <span class="filter-option" :class="{ active: selectedMethods.length === 0 }" @click="clearMethods"> 全部 </span>
          <span class="filter-option" :class="{ active: selectedMethods.includes('PC') }" @click="toggleMethod('PC')">PC算法</span>
          <span class="filter-option" :class="{ active: selectedMethods.includes('HillClimbing') }" @click="toggleMethod('HillClimbing')">爬山算法</span>
          <span class="filter-option" :class="{ active: selectedMethods.includes('GES') }" @click="toggleMethod('GES')">贪婪等价搜索</span>
          <span class="filter-option" :class="{ active: selectedMethods.includes('TAN') }" @click="toggleMethod('TAN')">树搜索</span>
          <span class="filter-option" :class="{ active: selectedMethods.includes('ExpertInLoop') }" @click="toggleMethod('ExpertInLoop')">专家在循环</span>
        </div>
      </div>
    </div>
    <div class="item-list">
      <div 
        v-for="item in pagedList" 
        :key="item.key" 
        class="list-item"
        @click="showWeightDetails(item.key, item.weight)"
      >
        <div class="item-title">{{ item.key }}</div>
        <div class="item-details">
          <div class="weight-info">
            <div class="quality-level">
              质量等级: <span :class="`quality-${getQualityValue(item.weight)}`">
                {{ item.weight.base_weight?.quality_level || item.weight.base_weight?.quality }}
              </span>
            </div>
            <div class="score-info">
              综合评分: <span class="score-value">{{ formatScore(item.weight.base_weight?.integrated_score) }}</span>
            </div>
            <div class="algorithm-info">
              支持算法:
              <span class="algorithm-list" v-if="getSupportAlgorithms(item.weight).length">
                {{ getSupportAlgorithms(item.weight).map(mapAlgCn).join('，') }}
              </span>
              <span class="algorithm-none" v-else>无</span>
            </div>
          </div>
        </div>
      </div>
      <div v-if="pagedList.length === 0" class="empty">暂无符合条件的权重</div>
    </div>

    <!-- 底部分页条：固定显示、支持跳页（5页滑窗 + 首页/末页） -->
    <div class="bottom-pager" v-if="totalPages > 1">
      <div class="pager">
        <span class="pager-chip" :class="{ disabled: currentPage === 1 }" @click="goToFirst">首页</span>
        <span class="pager-chip" :class="{ disabled: currentPage === 1 }" @click="prevPage">上一页</span>
        <template v-for="p in displayPages" :key="`chip-${p}`">
          <span class="pager-chip" :class="{ active: p === currentPage }" @click="goToPage(p)">{{ p }}</span>
        </template>
        <span class="pager-chip" :class="{ disabled: currentPage === totalPages }" @click="nextPage">下一页</span>
        <span class="pager-chip" :class="{ disabled: currentPage === totalPages }" @click="goToLast">末页</span>
      </div>
      <div class="pager-jump">
        <input v-model="jumpInput" type="number" class="jump-input" :min="1" :max="totalPages" placeholder="页码" @keyup.enter="applyJump" />
        <button class="btn ghost" @click="applyJump">跳转</button>
        <span class="pager-info">共 {{ totalPages }} 页，每页 10 条</span>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, watch } from 'vue'

export default {
  name: '权重系统组件',
  props: {
    weights: {
      type: Object,
      default: () => ({})
    }
  },
  emits: ['show-weight-details', 'refresh-weights'],
  setup(props, { emit }) {
    const searchInput = ref('')
    const appliedSearch = ref('')
    // 多选：为空表示不过滤（全部）
    const selectedQualities = ref([]) // ['platinum','gold','silver','bronze'] 可多选
    const selectedMethods = ref([])   // ['PC','HillClimbing','GES','TAN','ExpertInLoop'] 可多选
    const pageSize = ref(10)
    const currentPage = ref(1)
    const jumpInput = ref('')

    const filteredWeights = computed(() => {
      const query = (appliedSearch.value || '').toLowerCase()
      const out = {}
      Object.keys(props.weights).forEach(key => {
        const w = props.weights[key]
        // 搜索过滤（按键名）
        if (query && !key.toLowerCase().includes(query)) return
        // 质量等级多选过滤（兼容 quality 与 quality_level）
        if (selectedQualities.value.length && !matchesQualities(w, selectedQualities.value)) return
        // 算法来源多选过滤
        if (selectedMethods.value.length && !matchesMethods(w, selectedMethods.value)) return
        out[key] = w
      })
      return out
    })

    // 数组化并分页
    const filteredList = computed(() => {
      const obj = filteredWeights.value || {}
      return Object.keys(obj).map(k => ({ key: k, weight: obj[k] }))
    })
    const totalPages = computed(() => Math.max(1, Math.ceil(filteredList.value.length / pageSize.value)))
    const pagedList = computed(() => {
      const start = (currentPage.value - 1) * pageSize.value
      return filteredList.value.slice(start, start + pageSize.value)
    })

    // 搜索/筛选变化时重置到第1页
    watch([appliedSearch, selectedQualities, selectedMethods], () => {
      currentPage.value = 1
    }, { deep: true })

    const showWeightDetails = (key, weight) => {
      emit('show-weight-details', key, weight)
    }

    const onSearch = () => {
      appliedSearch.value = searchInput.value || ''
    }

    const onRefresh = () => {
      emit('refresh-weights')
    }

    // 多选切换/清空
    const toggleQuality = (val) => {
      const arr = selectedQualities.value
      const idx = arr.indexOf(val)
      if (idx >= 0) arr.splice(idx, 1)
      else arr.push(val)
    }
    const clearQualities = () => { selectedQualities.value = [] }
    const toggleMethod = (val) => {
      const arr = selectedMethods.value
      const idx = arr.indexOf(val)
      if (idx >= 0) arr.splice(idx, 1)
      else arr.push(val)
    }
    const clearMethods = () => { selectedMethods.value = [] }
    // 算法中文映射
    const mapAlgCn = (alg) => {
      const dict = {
        // 参数学习/统计方法
        'MLE': '极大似然',
        'Bayesian': '贝叶斯',
        'EM': 'EM',
        'SEM': '结构方程',
        'Pearson': '皮尔逊',
        'Spearman': '斯皮尔曼',
        // 因果发现方法
        'PC': 'PC算法',
        'HillClimbing': '爬山算法',
        'HillClimbing_AIC-D': '爬山算法',
        'GES': '贪婪等价搜索',
        'TAN': '树搜索',
        'ExpertInLoop': '专家在循环'
      }
      const k = String(alg || '')
      return dict[k] || k || '未知'
    }

    // 提取支持算法列表（兼容不同后端字段）
    const getSupportAlgorithms = (weight) => {
      if (!weight) return []
      const list = weight?.candidate_details?.support_algorithms
        || weight?.base_weight?.support_algorithms
        || []
      if (Array.isArray(list)) return list
      return list ? [list] : []
    }

    const getQualityValue = (weight) => {
      const q = weight?.base_weight?.quality_level ?? weight?.base_weight?.quality ?? ''
      return String(q).toLowerCase()
    }

    // 质量等级匹配（兼容 High/Medium/Low 与 Platinum/Gold/Silver/Bronze），支持多选
    const matchesQualities = (weight, selectedList) => {
      if (!selectedList || selectedList.length === 0) return true
      const q = getQualityValue(weight)
      const map = {
        'platinum': ['platinum', 'high', '高'],
        'gold': ['gold', 'medium', '中'],
        'silver': ['silver', 'unknown', '未知'],
        'bronze': ['bronze', 'low', '低']
      }
      return selectedList.some(sel => (map[sel] || [sel]).includes(q))
    }

    // 因果方法匹配（支持多种内部命名），支持多选
    const matchesMethods = (weight, selectedList) => {
      if (!selectedList || selectedList.length === 0) return true
      const map = {
        'PC': ['PC', 'PC算法'],
        'HillClimbing': ['HillClimbing', 'HillClimbing_AIC-D', '爬山算法'],
        'GES': ['GES', '贪婪等价搜索'],
        'TAN': ['TAN', '树搜索'],
        'ExpertInLoop': ['ExpertInLoop', '专家在循环']
      }
      const allowed = selectedList.flatMap(sel => map[sel] || [sel])
      const algs = getSupportAlgorithms(weight).map(a => String(a))
      return algs.some(a => allowed.includes(a))
    }

    // 分页操作与显示策略（1 2 3 4 5 ... N）
    const goToPage = (p) => {
      const tp = totalPages.value
      const to = Math.min(tp, Math.max(1, Number(p)))
      currentPage.value = to
    }
    const prevPage = () => { currentPage.value = Math.max(1, currentPage.value - 1) }
    const nextPage = () => { currentPage.value = Math.min(totalPages.value, currentPage.value + 1) }

    const displayPages = computed(() => {
      const tp = totalPages.value
      const cur = currentPage.value
      const res = []
      // 仅显示5页滑窗
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

    const applyJump = () => {
      const n = parseInt(jumpInput.value, 10)
      if (!Number.isNaN(n)) {
        goToPage(n)
      }
      jumpInput.value = ''
    }

    const goToFirst = () => { currentPage.value = 1 }
    const goToLast = () => { currentPage.value = totalPages.value }

    const formatScore = (v) => {
      const n = Number(v)
      return Number.isFinite(n) ? n.toFixed(4) : '—'
    }

    return {
      searchInput,
      appliedSearch,
      selectedQualities,
      selectedMethods,
      filteredWeights,
      filteredList,
      pagedList,
      pageSize,
      currentPage,
      totalPages,
      displayPages,
      jumpInput,
      applyJump,
      prevPage,
      nextPage,
      goToPage,
      goToFirst,
      goToLast,
      formatScore,
      mapAlgCn,
      getSupportAlgorithms,
      getQualityValue,
      matchesQualities,
      matchesMethods,
      toggleQuality,
      clearQualities,
      toggleMethod,
      clearMethods,
      showWeightDetails,
      onSearch,
      onRefresh
    }
  }
}
</script>

<style scoped>
.weights-panel {
  padding: 20px;
  background: #fff;
  /* 禁止组件自身出现横向滚动，内容交由外层页面滚动 */
  overflow-x: hidden !important;
  overflow-y: visible !important;
}

.panel-title {
  padding: 0 16px;
  margin-bottom: 12px;
  font-size: 16px;
  font-weight: 600;
  color: #2c3e50;
  display: flex;
  align-items: center;
  gap: 6px;
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
  transition: all .15s ease;
}
.search-btn:hover { background: #f8f9fa; }

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

.item-list {
  width: 100%;
  max-height: none;
  /* 禁止内部横向滚动，纵向不做局部滚动 */
  overflow-x: hidden !important;
  overflow-y: visible !important;
}


.filters-bar {
  display: flex;
  gap: 12px;
  align-items: center;
  padding: 0 16px 12px 16px;
  /* 窄屏换行，避免产生横向滚动 */
  flex-wrap: wrap;
}
.filter-group { display: flex; align-items: center; gap: 8px; }
.filter-label { color: #6c757d; font-size: 13px; }
.filter-options { display: flex; flex-wrap: wrap; gap: 8px; }
.filter-option {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border: 1px solid #dee2e6;
  border-radius: 999px;
  font-size: 13px;
  color: #374151;      /* 默认灰色文本 */
  background: #f3f4f6; /* 默认灰色背景 */
  cursor: pointer;
}
.filter-option.active {
  background: #007bff; /* 选中蓝色背景 */
  color: #fff;         /* 白色文本 */
  border-color: #007bff; /* 蓝色边框 */
}
.quality-indicator {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
}
.quality-platinum { background-color: #b7b7b7; }
.quality-gold { background-color: #ffd700; }
.quality-silver { background-color: #c0c0c0; }
.quality-bronze { background-color: #cd7f32; }

.list-item {
  padding: 15px;
  border: 1px solid #e9ecef;
  border-radius: 8px;
  margin-bottom: 10px;
  cursor: pointer;
  transition: all 0.2s;
  background: white;
  max-width: 100%;
  /* 长文本自动换行，避免产生横向滚动 */
  word-break: break-word;
  overflow-wrap: anywhere;
}

.list-item:hover {
  background: #f8f9fa;
  border-color: #007bff;
  box-shadow: 0 2px 4px rgba(0,123,255,0.1);
}

.item-title {
  font-weight: 500;
  color: #2c3e50;
  margin-bottom: 10px;
  font-size: 1.1em;
  /* 标题长文本换行，避免横向滚动 */
  word-break: break-word;
  overflow-wrap: anywhere;
}

.weight-info {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.quality-level, .score-info, .algorithm-info {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.9em;
  color: #6c757d;
  /* 明确允许内容换行，避免横向滚动 */
  flex-wrap: wrap;
}

.quality-high {
  color: #28a745;
  font-weight: 500;
}

.quality-medium {
  color: #ffc107;
  font-weight: 500;
}

.quality-low {
  color: #dc3545;
  font-weight: 500;
}

.score-value {
  font-weight: 500;
  color: #007bff;
}

.algorithm-count {
  display: none;
}

.algorithm-list {
  color: #0f766e;
  background: #ecfeff;
  border: 1px solid #99f6e4;
  padding: 2px 6px;
  border-radius: 6px;
  font-size: 0.85em;
  white-space: normal;
  word-break: break-word;
  overflow-wrap: anywhere;
}

.algorithm-none {
  color: #6b7280;
}
.empty { color: #9ca3af; text-align: center; padding: 24px 0; }
.bottom-pager {
  margin-top: 12px;
  background: #ffffff;
  border-top: 1px solid #e9ecef;
  padding: 10px 8px;
}
.pager {
  display: flex;
  gap: 6px;
  justify-content: center;
  align-items: center;
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
.pager-chip.ellipsis {
  cursor: default;
  color: #6c757d;
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
.btn.ghost {
  background: transparent;
  color: #4c6ef5;
  border: 1px solid #4c6ef5;
}
</style>