<template>
  <div class="phi-wrapper">
    <!-- 顶部工具栏 -->
    <div class="toolbar">
      <div class="title">
        <span class="icon">🛤️</span>
        路径分析 (Φ)
      </div>
      <div class="actions">
        <button class="btn primary" @click="handleRefresh">刷新数据</button>
      </div>
    </div>

    <!-- 搜索框 -->
    <div class="search-row">
      <input
        v-model="searchInput"
        type="text"
        class="search-input"
        placeholder="搜索路径..."
        @keyup.enter="applySearch"
      />
      <button class="search-btn" @click="applySearch">🔍 搜索</button>
    </div>

    <!-- 筛选区：效果类型
    <div class="chip-group">
      <span
        class="chip"
        :class="{ active: effectFilters.length === 0 }"
        @click="toggleAllEffects"
      >全部效果</span>
      <span
        class="chip"
        :class="{ active: effectFilters.includes('direct_positive') }"
        @click="toggleEffect('direct_positive')"
      >正向直接</span>
      <span
        class="chip"
        :class="{ active: effectFilters.includes('direct_negative') }"
        @click="toggleEffect('direct_negative')"
      >负向直接</span>
      <span
        class="chip"
        :class="{ active: effectFilters.includes('indirect') }"
        @click="toggleEffect('indirect')"
      >存在间接</span>
    </div>

    筛选区：置信度 
    <div class="chip-group">
      <span
        class="chip"
        :class="{ active: confidencePreset === 'all' }"
        @click="setConfidence('all')"
      >全部置信度</span>
      <span
        class="chip"
        :class="{ active: confidencePreset === 'gte_0_6' }"
        @click="setConfidence('gte_0_6')"
      >置信度≥0.6</span>
      <span
        class="chip"
        :class="{ active: confidencePreset === 'lt_0_6' }"
        @click="setConfidence('lt_0_6')"
      >置信度<0.6</span>
    </div> -->

    <!-- 列表 -->
    <div class="list">
      <div
        v-for="item in pagedPathways"
        :key="item.key"
        class="card"
        @click="openDetails(item)"
      >
        <div class="card-header">
          <div class="card-title">中介效应分析 - {{ item.key }}</div>
        </div>
        <div class="card-body">
          <div class="summary-grid">
            <div class="summary-item">
              <span class="label">路径数:</span>
              <span class="value">{{ getPathCount(item) }}</span>
            </div>
            <div class="summary-item">
              <span class="label">显著路径:</span>
              <span class="value">{{ getSignificantCount(item) }}</span>
            </div>
            <div class="summary-item">
              <span class="label">最大显著性:</span>
              <span class="value">{{ formatSignificanceRaw(item.confidence) }}</span>
            </div>
            <div class="summary-item">
              <span class="label">最显著路径:</span>
              <span class="value">{{ getMostSignificantPathId(item) }}</span>
            </div>
            <div class="summary-item">
              <span class="label">主要效应类型:</span>
              <span class="value">{{ getPrimaryEffectTypes(item) }}</span>
            </div>
            <div class="summary-item">
              <span class="label">显著率:</span>
              <span class="value">{{ formatRate(getSignificantCount(item), getPathCount(item)) }}</span>
            </div>
          </div>
        </div>
      </div>

      <div v-if="pagedPathways.length === 0" class="empty">
        暂无符合条件的路径
      </div>
    </div>

    <!-- 分页：统一为 5 页滑窗 + 首页/末页，与权重系统保持一致 -->
    <div class="pager" v-if="totalPages > 1">
      <span class="pager-chip" :class="{ disabled: currentPage === 1 }" @click="goToFirst">首页</span>
      <span class="pager-chip" :class="{ disabled: currentPage === 1 }" @click="prevPage">上一页</span>
      <span
        v-for="p in displayPages"
        :key="`chip-${p}`"
        class="pager-chip"
        :class="{ active: p === currentPage }"
        @click="goPage(p)"
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

<script setup>
import { computed, ref, watch } from 'vue'
import { useKnowledgeGraphStore } from '../状态管理/知识图谱状态'

const props = defineProps({
  pathways: {
    type: [Object, Array],
    default: () => ({})
  }
})
const emit = defineEmits(['show-pathway-details', 'refresh-pathways'])

const searchQuery = ref('') // 实际用于过滤的关键词
const searchInput = ref('') // 输入框内容，仅在点击“搜索”时应用
const pageSize = ref(10)
const currentPage = ref(1)
const jumpInput = ref('')
const effectFilters = ref([]) // e.g. ['direct_positive','indirect']
const confidencePreset = ref('all')
const store = useKnowledgeGraphStore()

// 详情缓存（当前页预取）
const detailsByKey = ref({})
const inflightKeys = new Set()

// 归一化列表：支持对象或数组
const normalizedList = computed(() => {
  const src = props.pathways || {}
  if (Array.isArray(src)) {
    return src.map((v) => {
      if (typeof v === 'object' && v) {
        return { key: v.key || v.name || v.id || JSON.stringify(v), ...v }
      }
      return { key: String(v), raw: v }
    })
  }
  return Object.keys(src).map((k) => ({ key: k, ...(src[k] || {}) }))
})

// 过滤逻辑
const filtered = computed(() => {
  const q = searchQuery.value.trim().toLowerCase()
  return normalizedList.value.filter((item) => {
    // 搜索
    if (q) {
      const hay = `${item.key} ${JSON.stringify(item)}`.toLowerCase()
      const keyStr = String(item.key || '').toLowerCase()
      if (!keyStr.includes(q) && !hay.includes(q)) {
        const d = detailsByKey.value[item.key] || {}
        const effs = Array.isArray(d.mediation_effects_list) ? d.mediation_effects_list : []
        const matched = effs.some(e => String(e?.pathway_id ?? '').toLowerCase().includes(q))
        if (!matched) return false
      }
    }
    // 置信度
    const c = normalizeConfidence(item.confidence)
    if (confidencePreset.value === 'gte_0_6' && !(c >= 0.6)) return false
    if (confidencePreset.value === 'lt_0_6' && !(c < 0.6)) return false

    // 效果类型
    if (effectFilters.value.length > 0) {
      const d = numberOrZero(item.direct_effect)
      const ind = numberOrZero(item.indirect_effect)

      const hasDirectPos = d > 0
      const hasDirectNeg = d < 0
      const hasIndirect = Math.abs(ind) > 0

      const checks = {
        direct_positive: hasDirectPos,
        direct_negative: hasDirectNeg,
        indirect: hasIndirect
      }

      const ok = effectFilters.value.every((f) => checks[f])
      if (!ok) return false
    }
    return true
  })
})

// 分页
const totalPages = computed(() => Math.max(1, Math.ceil(filtered.value.length / pageSize.value)))
const pagedPathways = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value
  return filtered.value.slice(start, start + pageSize.value)
})
// 统一 5 页滑窗
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
watch([searchQuery, effectFilters, confidencePreset], () => {
  currentPage.value = 1
})

// 当分页条目变化时，预取详情以渲染摘要字段
watch(
  () => pagedPathways.value.map(i => i.key),
  async (keys) => {
    for (const key of keys) {
      if (!detailsByKey.value[key] && !inflightKeys.has(key)) {
        inflightKeys.add(key)
        try {
          const resp = await store.getPathwayDetails(key)
          const data = resp?.data ?? resp ?? {}
          detailsByKey.value[key] = data
        } catch (e) {
          console.error('加载路径详情失败:', key, e)
        } finally {
          inflightKeys.delete(key)
        }
      }
    }
  },
  { immediate: true }
)

// 事件
const openDetails = (item) => {
  emit('show-pathway-details', { key: item.key, data: item })
}
const handleRefresh = () => emit('refresh-pathways')
const applySearch = async () => {
  searchQuery.value = (searchInput.value || '').trim()
  if (searchQuery.value) {
    await prefetchAllDetails()
  }
}

// 效果筛选操作
const toggleEffect = (flag) => {
  const idx = effectFilters.value.indexOf(flag)
  if (idx >= 0) effectFilters.value.splice(idx, 1)
  else effectFilters.value.push(flag)
}
const toggleAllEffects = () => {
  effectFilters.value = []
}
const setConfidence = (preset) => {
  confidencePreset.value = preset
}

// 在搜索时预取所有详情，便于通过路径ID匹配
async function prefetchAllDetails() {
  const keys = normalizedList.value.map(i => i.key)
  const queue = keys.filter(k => !detailsByKey.value[k] && !inflightKeys.has(k))
  const concurrency = 4
  const runners = Array.from({ length: concurrency }, async () => {
    while (queue.length) {
      const key = queue.shift()
      inflightKeys.add(key)
      try {
        const resp = await store.getPathwayDetails(key)
        const data = resp?.data ?? resp ?? {}
        detailsByKey.value[key] = data
      } catch (e) {
        console.error('加载路径详情失败:', key, e)
      } finally {
        inflightKeys.delete(key)
      }
    }
  })
  await Promise.all(runners)
}

// 分页操作
const goPage = (p) => {
  const tp = totalPages.value
  const to = Math.min(tp, Math.max(1, Number(p)))
  currentPage.value = to
}
const prevPage = () => { currentPage.value = Math.max(1, currentPage.value - 1) }
const nextPage = () => { currentPage.value = Math.min(totalPages.value, currentPage.value + 1) }
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

// 辅助（使用函数声明，确保在上方计算属性中可用）
function numberOrZero(v) {
  const n = Number(v)
  return Number.isFinite(n) ? n : 0
}
function normalizeConfidence(v) {
  if (v == null) return 0
  const n = Number(v)
  if (!Number.isFinite(n)) return 0
  // 允许 0~1 和 0~100 两种格式
  return n > 1 ? Math.min(1, Math.max(0, n / 100)) : Math.min(1, Math.max(0, n))
}
const formatEffect = (v) => {
  const n = Number(v)
  return Number.isFinite(n) ? n.toFixed(4) : '—'
}
const formatConfidence = (v) => `${Math.round(normalizeConfidence(v) * 100)}%`
const formatSignificanceRaw = (v) => {
  const n = Number(v)
  return Number.isFinite(n) ? n.toFixed(3) : '—'
}
const formatRate = (sigCount, totalCount) => {
  const t = Number(totalCount)
  const s = Number(sigCount)
  if (!Number.isFinite(t) || t <= 0 || !Number.isFinite(s)) return '—'
  return `${((s / t) * 100).toFixed(1)}%`
}
const toCount = (val) => {
  if (Array.isArray(val)) return val.length
  const n = Number(val)
  return Number.isFinite(n) ? n : 0
}
const getCoreCount = (item) => toCount(item.core_count ?? item.core_paths ?? item.CorePaths ?? item.core)
const getCandidateCount = (item) => toCount(item.candidate_count ?? item.candidate_paths ?? item.CandidatePaths ?? item.candidate)

// 摘要数据来源：详情接口
const getDetailsObj = (item) => detailsByKey.value[item.key] || {}
const getPathCount = (item) => {
  const d = getDetailsObj(item)
  const stats = d.effect_statistics || {}
  if (Number.isFinite(stats.pathways_count)) return stats.pathways_count
  if (Array.isArray(d.mediation_effects_list)) return d.mediation_effects_list.length
  const core = Array.isArray(d.core_paths) ? d.core_paths.length : 0
  const cand = Array.isArray(d.candidate_paths) ? d.candidate_paths.length : 0
  return core + cand
}
const getSignificantCount = (item) => {
  const d = getDetailsObj(item)
  const stats = d.effect_statistics || {}
  if (Number.isFinite(stats.significant_pathways_count)) return stats.significant_pathways_count
  if (Array.isArray(d.core_paths)) return d.core_paths.length
  if (Array.isArray(d.mediation_effects_list)) return d.mediation_effects_list.filter(x => x && x.is_significant).length
  return 0
}
const getMostSignificantPathId = (item) => {
  const d = getDetailsObj(item)
  const id = d.most_significant_pathway_id ?? (d.significance_info && d.significance_info.most_significant_pathway)
  return (id ?? '—')
}
const getPrimaryEffectTypes = (item) => {
  const d = getDetailsObj(item)
  const stats = d.effect_statistics || {}
  const types = stats.primary_effect_types
  if (Array.isArray(types) && types.length) return types.join('，')
  if (Array.isArray(d.mediation_effects_list) && d.mediation_effects_list.length) {
    const arr = d.mediation_effects_list.map(e => e && e.primary_effect_type).filter(Boolean)
    if (arr.length) return [...new Set(arr)].join('，')
  }
  return '—'
}

// 子路径效应预览（最多3条）：路径ID + 主效应类型 + 平均效应强度
const getEffectsPreview = (item) => {
  const d = getDetailsObj(item)
  const list = Array.isArray(d.mediation_effects_list) ? d.mediation_effects_list : []
  return list.slice(0, 3).map(e => ({
    id: e?.pathway_id ?? '—',
    type: e?.primary_effect_type ?? '—',
    strength: e?.effect_strength ?? '—'
  }))
}
</script>

<style scoped>
.phi-wrapper {
  background: #fff;
  border: 1px solid #e9ecef;
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.06);
  padding: 20px;
  /* 面板自身不产生横向滚动，由父页面统一滚动 */
  overflow-x: hidden !important;
  overflow-y: visible !important;
}

.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}
.title {
  font-size: 18px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
}
.actions .btn {
  padding: 6px 12px;
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

.search-row {
  margin-bottom: 8px;
  display: flex;
  gap: 8px;
}
.search-input {
  width: 100%;
  flex: 1;
  border: 1px solid #dee2e6;
  border-radius: 8px;
  padding: 10px 12px;
  outline: none;
  font-size: 14px;
}

/* 统一搜索按钮样式，圆角、浅色边框与悬停反馈 */
.search-btn {
  padding: 10px 14px;
  border-radius: 18px;
  border: 1px solid #dee2e6;
  background: #fff;
  color: #34495e;
  font-size: 13px;
  cursor: pointer;
  transition: background .15s ease, box-shadow .15s ease, border-color .15s ease, transform .05s ease;
}
.search-btn:hover {
  background: #f8f9fa;
  border-color: #d8dee4;
  box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}
.search-btn:focus {
  outline: none;
  border-color: #4c6ef5;
  box-shadow: 0 0 0 3px rgba(76, 110, 245, 0.15);
}
.search-btn:active {
  transform: translateY(1px);
}

.chip-group {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 10px 0;
}
.chip {
  padding: 6px 10px;
  border-radius: 16px;
  background: #f1f3f5;
  color: #495057;
  font-size: 13px;
  cursor: pointer;
  border: 1px solid #e9ecef;
}
.chip.active {
  background: #4c6ef5;
  color: #fff;
  border-color: #4c6ef5;
}

.list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.card {
  border: 1px solid #e9ecef;
  border-radius: 10px;
  padding: 12px;
  background: #fff;
  cursor: pointer;
  transition: background .15s ease, box-shadow .15s ease, border-color .15s ease;
}
.card:hover {
  background: #f8f9fa;
  border-color: #dfe3e6;
  box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
.card-title {
  font-weight: 600;
  font-size: 15px;
}
.summary-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px 24px;
}
.summary-item .label {
  color: #868e96;
  font-size: 12px;
  margin-right: 6px;
}
.summary-item .value {
  font-size: 14px;
  font-weight: 600;
}
.effects-preview {
  margin-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.effects-preview .effect-row {
  font-size: 13px;
  color: #495057;
}
.effects-preview .effect-row .label {
  color: #868e96;
  margin-right: 4px;
}
.effects-preview .effect-row .sep {
  color: #ced4da;
  margin: 0 8px;
}
.empty {
  color: #adb5bd;
  text-align: center;
  padding: 24px 0;
}

.pager {
  display: flex;
  gap: 6px;
  justify-content: center;
  margin-top: 12px;
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
.btn.ghost {
  background: transparent;
  color: #4c6ef5;
  border: 1px solid #4c6ef5;
}
</style>
