<template>
  <div class="parameters-panel">
    <div class="panel-title">📈 参数学习 (Θ)</div>
    <div class="search-bar">
      <input 
        v-model="searchQuery" 
        type="text" 
        class="search-input" 
        placeholder="搜索参数节点..."
        @keyup.enter="onEnterSearch"
      >
      <button class="search-btn" @click="onClickSearch">🔍 搜索</button>
      <button class="btn primary" @click="onRefresh">刷新数据</button>
    </div>
    <div class="filter-bar">
      <span class="filter-label">一致性筛选:</span>
      <div class="filter-chips">
        <div 
          v-for="opt in consistencyOptions" 
          :key="opt.value" 
          class="filter-chip" 
          :class="{ active: (opt.value === 'all' ? selectedConsistencies.length === 0 : selectedConsistencies.includes(opt.value)), [opt.class]: true }"
          @click="onClickConsistency(opt.value)"
        >{{ opt.label }}</div>
      </div>
    </div>
    <div class="item-list">
      <div 
        v-for="([key, param], idx) in pagedEntries" 
        :key="key" 
        class="list-item"
        @click="showParameterDetails(key, param)"
      >
        <div class="item-title">{{ key }}</div>
        <div class="item-subtitle">{{ buildSubtitle(param) }}</div>
        <div class="item-metrics">{{ buildMetricsLine(key, param) }}</div>
        <div class="item-details">
          <div class="parameter-methods">
            <div v-if="isAvailable(param, 'MLE')" class="method-badge mle">MLE</div>
            <div v-if="isAvailable(param, 'Bayesian')" class="method-badge bayesian">Bayesian</div>
            <div v-if="isAvailable(param, 'EM')" class="method-badge em">EM</div>
            <div v-if="isAvailable(param, 'SEM')" class="method-badge sem">SEM</div>
          </div>
          <div class="parameter-summary">
            可用方法: {{ getAvailableMethodsCount(param) }} / 4
          </div>
        </div>
      </div>
      <div v-if="parameterEntries.length === 0" class="empty">暂无参数数据</div>
    </div>
    <div class="pager" v-if="totalPages > 1">
      <span class="pager-chip" :class="{ disabled: currentPage === 1 }" @click="firstPage">首页</span>
      <span class="pager-chip" :class="{ disabled: currentPage === 1 }" @click="prevPage">上一页</span>
      <span 
        v-for="p in displayPages" 
        :key="p" 
        class="pager-chip" 
        :class="{ active: currentPage === p }" 
        @click="goToPage(p)"
      >{{ p }}</span>
      <span class="pager-chip" :class="{ disabled: currentPage === totalPages }" @click="nextPage">下一页</span>
      <span class="pager-chip" :class="{ disabled: currentPage === totalPages }" @click="lastPage">末页</span>
    </div>
    <div class="pager-jump" v-if="totalPages > 1">
      <input v-model="jumpInput" type="number" class="jump-input" :min="1" :max="totalPages" placeholder="页码" @keyup.enter="applyJump" />
      <button class="btn ghost" @click="applyJump">跳转</button>
      <span class="pager-info">共 {{ totalPages }} 页，每页 {{ pageSize }} 条</span>
    </div>
  </div>
</template>

<script>
import { ref, computed, watchEffect } from 'vue'
import axios from 'axios'

export default {
  name: '参数学习组件',
  props: {
    parameters: {
      type: Object,
      default: () => ({})
    }
  },
  emits: ['show-parameter-details', 'search-parameters', 'refresh-parameters'],
  setup(props, { emit }) {
    const searchQuery = ref('')
    const pageSize = ref(10)
    const currentPage = ref(1)
    const jumpInput = ref('')
    const selectedConsistencies = ref([]) // 多选：为空表示全部不过滤
    const consistencyOptions = [
      { value: 'all', label: '全部', class: 'chip-all' },
      { value: '高度一致', label: '高度一致', class: 'chip-high' },
      { value: '中等一致', label: '中等一致', class: 'chip-medium' },
      { value: '低度一致', label: '低度一致', class: 'chip-low' },
      { value: '不一致', label: '不一致', class: 'chip-none' },
    ]

    // 兼容Theta可能的嵌套结构：parameter_learning / 直接方法键
    const normalizedParameters = computed(() => {
      const src = props.parameters || {}
      const out = {}
      for (const [key, val] of Object.entries(src)) {
        if (val && typeof val === 'object') {
          const methods = val.parameter_learning && typeof val.parameter_learning === 'object'
            ? val.parameter_learning
            : val
          out[key] = {
            MLE: methods.MLE,
            Bayesian: methods.Bayesian,
            EM: methods.EM,
            SEM: methods.SEM,
            __raw: val
          }
        } else {
          out[key] = { __raw: val }
        }
      }
      return out
    })

    const parameterEntries = computed(() => {
      const obj = normalizedParameters.value || {}
      let entries = Object.entries(obj)
      // 一致性筛选（可多选，基于稳定性等级）：当选择不为空时过滤
      if (selectedConsistencies.value.length > 0) {
        entries = entries.filter(([key, param]) => {
          const cached = detailsCache.value[key]
          const hasCache = cached && (cached.score !== undefined || cached.level !== undefined)
          const { level: rawLevel } = getStabilitySummary(param)
          const level = String(hasCache ? (cached.level ?? rawLevel) : rawLevel || '').trim()
          return selectedConsistencies.value.includes(level)
        })
      }
      return entries
    })
    const totalPages = computed(() => {
      const total = parameterEntries.value.length
      return Math.max(1, Math.ceil(total / pageSize.value))
    })
    const pagedEntries = computed(() => {
      const start = (currentPage.value - 1) * pageSize.value
      return parameterEntries.value.slice(start, start + pageSize.value)
    })
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

    // 详情缓存：用于填充指标，避免显示“—”
    const detailsCache = ref({})
    const ensureVisibleDetails = async () => {
      const keys = pagedEntries.value.map(([k]) => k)
      await Promise.all(keys.map(async (k) => {
        if (detailsCache.value[k]) return
        try {
          const resp = await axios.get(`/api/parameters/${encodeURIComponent(k)}/details`)
          const data = resp?.data?.data ?? resp?.data ?? resp ?? {}
          const ps = data.parameter_stability || {}
          const me = data.method_estimates || {}
          // 解析稳定性
          const score = ps.stability_score ?? ps.overall_score ?? ps.overall
          const level = ps.consistency_level ?? ps.level ?? ps.category
          // 解析平均似然增益：直接取各方法显示的“边际似然增益”数值的平均
          // 优先使用方法级 point 值（likelihood_gain），与详情面板一致
          const methodPointVals = ['MLE','Bayesian','EM','SEM']
            .map(m => me?.[m]?.likelihood_gain)
            .filter(v => typeof v === 'number' && !isNaN(v))
          let avgGain = null
          if (methodPointVals.length) {
            avgGain = methodPointVals.reduce((a,b)=>a+b,0) / methodPointVals.length
          } else {
            // 回退：汇总边级增益的平均；再回退到方法级平均或总体平均
            const edgeVals = []
            ;['MLE','Bayesian','EM','SEM'].forEach(m => {
              const gains = (me?.[m]?.edge_likelihood_gain) || (me?.edge_likelihood_gain?.[m])
              if (gains && typeof gains === 'object') {
                Object.values(gains).forEach(v => { if (typeof v === 'number' && !isNaN(v)) edgeVals.push(v) })
              }
            })
            if (edgeVals.length) {
              avgGain = edgeVals.reduce((a,b)=>a+b,0) / edgeVals.length
            } else {
              const methodAvgVals = ['MLE','Bayesian','EM','SEM']
                .map(m => me?.[m]?.likelihood_gain_avg)
                .filter(v => typeof v === 'number' && !isNaN(v))
              if (methodAvgVals.length) {
                avgGain = methodAvgVals.reduce((a,b)=>a+b,0) / methodAvgVals.length
              } else {
                avgGain = me.likelihood_gain_avg ?? null
              }
            }
          }
          detailsCache.value[k] = { score, level, avgGain }
        } catch (e) {
          // 忽略错误，保持空
          detailsCache.value[k] = detailsCache.value[k] || null
        }
      }))
    }
    watchEffect(() => { ensureVisibleDetails() })

    const isAvailable = (param, method) => {
      const p = param || {}
      return !!p[method]
    }

    const getAvailableMethodsCount = (param) => {
      let count = 0
      if (isAvailable(param, 'MLE')) count++
      if (isAvailable(param, 'Bayesian')) count++
      if (isAvailable(param, 'EM')) count++
      if (isAvailable(param, 'SEM')) count++
      return count
    }

    const buildSubtitle = (param) => {
      const methods = []
      if (isAvailable(param, 'MLE')) methods.push('MLE')
      if (isAvailable(param, 'Bayesian')) methods.push('Bayesian')
      if (isAvailable(param, 'EM')) methods.push('EM')
      if (isAvailable(param, 'SEM')) methods.push('SEM')
      if (!methods.length) return '暂无可用方法'
      return `可用方法: ${methods.join(', ')}`
    }

    const fmt = (v, digits = 2) => {
      if (v === null || v === undefined) return '—'
      if (typeof v === 'number') {
        if (isNaN(v)) return '—'
        return v.toFixed(digits)
      }
      return String(v)
    }

    const getStabilitySummary = (param) => {
      const raw = (param && param.__raw) || {}
      const s = raw.stability 
        || (raw.parameter_learning && raw.parameter_learning.stability)
        || raw.stability_summary
        || null
      let score = undefined
      let level = undefined
      if (typeof s === 'number') {
        score = s
      } else if (s && typeof s === 'object') {
        score = s.stability_score ?? s.overall_score ?? s.overall
        level = s.consistency_level ?? s.level ?? s.category
      }
      return { score, level }
    }

    const getAvgLikelihoodGain = (param) => {
      const raw = (param && param.__raw) || {}
      const me = raw.method_estimates 
        || (raw.parameter_learning && raw.parameter_learning.method_estimates)
        || null
      // 优先用方法级 point 值（likelihood_gain）的平均
      const methodPointVals = ['MLE','Bayesian','EM','SEM']
        .map(m => me?.[m]?.likelihood_gain)
        .filter(v => typeof v === 'number' && !isNaN(v))
      if (methodPointVals.length) {
        return methodPointVals.reduce((a,b)=>a+b,0) / methodPointVals.length
      }
      // 次选：边级增益的平均
      const edgeVals = []
      ;['MLE','Bayesian','EM','SEM'].forEach(m => {
        const gains = (me?.[m]?.edge_likelihood_gain) 
          || (me?.edge_likelihood_gain?.[m])
          || null
        if (gains && typeof gains === 'object') {
          Object.values(gains).forEach(v => { if (typeof v === 'number' && !isNaN(v)) edgeVals.push(v) })
        }
      })
      if (edgeVals.length) {
        return edgeVals.reduce((a,b)=>a+b,0) / edgeVals.length
      }
      // 最后回退：方法级平均或总体平均
      const methodAvgVals = ['MLE','Bayesian','EM','SEM']
        .map(m => me?.[m]?.likelihood_gain_avg)
        .filter(v => typeof v === 'number' && !isNaN(v))
      if (methodAvgVals.length) {
        return methodAvgVals.reduce((a,b)=>a+b,0) / methodAvgVals.length
      }
      const fallback = me?.likelihood_gain_avg 
        ?? raw.likelihood_gain_avg 
        ?? raw.avg_likelihood_gain 
        ?? raw.likelihood_gain_mean
      return typeof fallback === 'number' ? fallback : undefined
    }

    const buildMetricsLine = (key, param) => {
      const cached = detailsCache.value[key]
      const hasCache = cached && (cached.score !== undefined || cached.avgGain !== undefined)
      const { score: rawScore, level: rawLevel } = getStabilitySummary(param)
      const rawAvgGain = getAvgLikelihoodGain(param)
      const score = hasCache ? cached.score ?? rawScore : rawScore
      const level = hasCache ? cached.level ?? rawLevel : rawLevel
      const avgGain = hasCache ? cached.avgGain ?? rawAvgGain : rawAvgGain
      const levelText = level ? `（${level}）` : ''
      return `稳定性: ${fmt(score, 3)} ${levelText} | 平均似然增益: ${fmt(avgGain, 2)}`
    }

    const showParameterDetails = (key, param) => {
      emit('show-parameter-details', key, param)
    }

    const onEnterSearch = () => {
      emit('search-parameters', searchQuery.value || '')
    }

    const onClickSearch = () => {
      emit('search-parameters', searchQuery.value || '')
      currentPage.value = 1
    }

    const onRefresh = () => {
      emit('refresh-parameters')
    }

    const onClickConsistency = (val) => {
      if (val === 'all') {
        selectedConsistencies.value = []
      } else {
        const arr = selectedConsistencies.value
        const idx = arr.indexOf(val)
        if (idx >= 0) arr.splice(idx, 1)
        else arr.push(val)
      }
      currentPage.value = 1
    }

    const goToPage = (p) => {
      if (typeof p !== 'number') return
      if (p < 1 || p > totalPages.value) return
      currentPage.value = p
    }
    const prevPage = () => { if (currentPage.value > 1) currentPage.value-- }
    const nextPage = () => { if (currentPage.value < totalPages.value) currentPage.value++ }
    const firstPage = () => { if (currentPage.value !== 1) currentPage.value = 1 }
    const lastPage = () => { if (currentPage.value !== totalPages.value) currentPage.value = totalPages.value }

    const applyJump = () => {
      const n = parseInt(jumpInput.value, 10)
      if (!Number.isNaN(n)) {
        const to = Math.min(totalPages.value, Math.max(1, n))
        currentPage.value = to
      }
      jumpInput.value = ''
    }

    return {
      searchQuery,
      pageSize,
      currentPage,
      selectedConsistencies,
      consistencyOptions,
      totalPages,
      parameterEntries,
      pagedEntries,
      displayPages,
      detailsCache,
      normalizedParameters,
      isAvailable,
      getAvailableMethodsCount,
      buildSubtitle,
      buildMetricsLine,
      showParameterDetails,
      onEnterSearch,
      onClickSearch,
      onRefresh,
      onClickConsistency,
      goToPage,
      prevPage,
      nextPage,
      firstPage,
      lastPage,
      jumpInput,
      applyJump
    }
  }
}
</script>

<style scoped>
.parameters-panel {
  padding: 20px;
  /* 面板自身不产生横向/纵向的局部滚动，由父页面统一滚动 */
  overflow-x: hidden !important;
  overflow-y: visible !important;
  background: #fff;
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
  /* 窄屏时自动换行，避免产生横向滚动 */
  flex-wrap: wrap;
}

.filter-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 16px 12px 16px;
  margin-top: 8px;
  /* 允许换行，避免横向滚动 */
  flex-wrap: wrap;
}
.filter-label { color: #6b7280; font-size: 13px; }
.filter-chips { display: flex; gap: 8px; flex-wrap: wrap; }
.filter-chip {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid #dee2e6;
  background: #f3f4f6; /* 默认灰色背景 */
  color: #374151;      /* 默认灰色文本 */
  cursor: pointer;
  transition: all .15s ease;
}
.filter-chip.active { background: #007bff; border-color: #007bff; color: #fff; }
.filter-chip.chip-high { border-color: #dee2e6; background: #f3f4f6; color: #374151; }
.filter-chip.chip-medium { border-color: #dee2e6; background: #f3f4f6; color: #374151; }
.filter-chip.chip-low { border-color: #dee2e6; background: #f3f4f6; color: #374151; }
.filter-chip.chip-none { border-color: #dee2e6; background: #f3f4f6; color: #374151; }
.filter-chip.chip-all.active { background: #007bff; border-color: #007bff; color: #fff; }

/* 选中态强制为蓝色，避免 chip-high/medium/low/none 的默认灰色覆盖 */
.filter-chip.active.chip-high,
.filter-chip.active.chip-medium,
.filter-chip.active.chip-low,
.filter-chip.active.chip-none {
  background: #007bff;
  border-color: #007bff;
  color: #fff;
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

.btn {
  padding: 8px 14px;
  border-radius: 8px;
  border: 1px solid #dee2e6;
  background: #fff;
  color: #34495e;
  font-size: 13px;
  cursor: pointer;
}
.btn.primary {
  border-color: #3b82f6;
  color: #fff;
  background: #3b82f6;
}
.btn.primary:hover {
  background: #2563eb;
}

.item-list {
  max-height: none;
  /* 禁止内部横向滚动，纵向不做局部滚动 */
  overflow-x: hidden !important;
  overflow-y: visible !important;
}

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

.item-metrics {
  color: #495057;
  font-size: 13px;
  margin: 6px 0 4px;
  /* 指标行允许换行，避免横向滚动 */
  word-break: break-word;
  overflow-wrap: anywhere;
}

.pager {
  display: flex;
  gap: 6px;
  justify-content: center;
  margin-top: 12px;
}

.pager-btn {
  padding: 6px 12px;
  border-radius: 14px;
  border: 1px solid #dee2e6;
  background: #fff;
  cursor: pointer;
  font-size: 13px;
}
.pager-btn:disabled {
  cursor: not-allowed;
  opacity: 0.5;
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
  opacity: 0.5;
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
.btn.ghost {
  background: transparent;
  color: #4c6ef5;
  border: 1px solid #4c6ef5;
}

.item-title {
  font-weight: 500;
  color: #2c3e50;
  margin-bottom: 6px;
  font-size: 1.1em;
  /* 标题长文本换行，避免横向滚动 */
  word-break: break-word;
  overflow-wrap: anywhere;
}

.item-subtitle {
  color: #6c757d;
  margin-bottom: 10px;
  font-size: 0.9em;
  /* 副标题长文本换行，避免横向滚动 */
  word-break: break-word;
  overflow-wrap: anywhere;
}

.parameter-methods {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 10px;
}

.method-badge {
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 0.8em;
  font-weight: 500;
}

.method-badge.mle {
  background: #d4edda;
  color: #155724;
}

.method-badge.bayesian {
  background: #d1ecf1;
  color: #0c5460;
}

.method-badge.em {
  background: #fff3cd;
  color: #856404;
}

.method-badge.sem {
  background: #f8d7da;
  color: #721c24;
}

.parameter-summary {
  font-size: 0.9em;
  color: #6c757d;
  font-weight: 500;
}

.empty {
  color: #adb5bd;
  text-align: center;
  padding: 24px 0;
}
</style>