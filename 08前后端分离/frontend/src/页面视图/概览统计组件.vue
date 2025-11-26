<template>
  <div class="overview">
    <div class="stats-grid">
      <div
        class="stat-card primary clickable"
        @click="$emit('navigate','nodes')"
        role="button"
        tabindex="0"
        title="跳转到 节点集合 (V)"
      >
        <div class="stat-number">{{ statistics.nodes ?? 0 }}</div>
        <div class="stat-label">节点总数</div>
      </div>
      <div
        class="stat-card success clickable"
        @click="$emit('navigate','edges')"
        role="button"
        tabindex="0"
        title="跳转到 边集合 (E_core)"
      >
        <div class="stat-number">{{ statistics.edges ?? 0 }}</div>
        <div class="stat-label">边总数</div>
      </div>
      <div
        class="stat-card info clickable"
        @click="$emit('navigate','relations')"
        role="button"
        tabindex="0"
        title="跳转到 关系类型 (R)"
      >
        <div class="stat-number">{{ statistics.relations ?? 0 }}</div>
        <div class="stat-label">关系类型</div>
      </div>
    </div>

    <div class="section-card">
      <h3 class="section-title">关系类型分布</h3>
      <div class="rel-list">
        <div v-for="row in relRows" :key="row.type" class="rel-item">
          <span class="rel-label">{{ row.type }}</span>
          <div class="rel-bar">
            <div class="rel-bar-fill" :style="{ width: barWidth(row.count) }"></div>
          </div>
          <span class="rel-count">{{ row.count }}</span>
        </div>
      </div>
    </div>

    <div class="section-card">
      <h3 class="section-title">边层次分布</h3>
      <div class="chip-row">
        <span class="chip green">三角验证 <span class="chip-count">{{ hierStats.triangulated_verified || 0 }}</span></span>
        <span class="chip orange">非三角验证 <span class="chip-count">{{ hierStats.non_triangulated || 0 }}</span></span>
      </div>
    </div>

    <!-- 在下方增加介绍区域 -->
    <div class="section-card">
      <h3 class="section-title">介绍</h3>
      <div class="intro-block">
        增强知识图谱可视化 基于
        <span class="intro-math">(V, E_core, R, W, Θ, Φ)</span>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'

export default {
  name: '概览统计组件',
  props: {
    statistics: {
      type: Object,
      default: () => ({})
    },
    relationTypeStats: {
      type: Object,
      default: null
    },
    hierarchyStats: {
      type: Object,
      default: null
    },
    edges: {
      type: Array,
      default: () => []
    }
  },
  emits: ['navigate'],
  setup(props) {
    const relStats = computed(() => {
      if (props.relationTypeStats && Object.keys(props.relationTypeStats).length > 0) {
        return props.relationTypeStats
      }
      const stats = {}
      props.edges.forEach(edge => {
        stats[edge.relation_type] = (stats[edge.relation_type] || 0) + 1
      })
      return stats
    })

    const relRows = computed(() => {
      const entries = Object.entries(relStats.value || {}).map(([type, count]) => ({ type, count }))
      entries.sort((a, b) => b.count - a.count)
      return entries
    })

    const maxRel = computed(() => {
      const vals = relRows.value.map(r => r.count)
      return vals.length ? Math.max(...vals) : 0
    })

    const hierStats = computed(() => {
      if (props.hierarchyStats && Object.keys(props.hierarchyStats).length > 0) {
        return props.hierarchyStats
      }
      const stats = {}
      props.edges.forEach(edge => {
        stats[edge.edge_hierarchy] = (stats[edge.edge_hierarchy] || 0) + 1
      })
      return stats
    })

    const barWidth = (count) => {
      return maxRel.value ? `${Math.round((count / maxRel.value) * 100)}%` : '0%'
    }

    return {
      relStats,
      relRows,
      hierStats,
      barWidth
    }
  }
}
</script>

<style scoped>
.overview {
  padding: 16px;
  background: linear-gradient(180deg, #f8fafc, #f1f5f9);
  border-radius: 12px;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}

.stat-card {
  border-radius: 12px;
  padding: 16px;
  text-align: center;
  color: #1f2937;
  box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}
.stat-card.primary { background: linear-gradient(135deg, #eef2ff, #e0e7ff); }
.stat-card.success { background: linear-gradient(135deg, #e7f5ff, #d0ebff); }
.stat-card.info { background: linear-gradient(135deg, #e6fcf5, #c3fae8); }

/* 使统计卡片可点击并提供交互反馈 */
.stat-card.clickable { cursor: pointer; transition: box-shadow 0.2s ease, transform 0.2s ease; }
.stat-card.clickable:hover { box-shadow: 0 10px 28px rgba(0,0,0,0.08); transform: translateY(-1px); }
.stat-card.clickable:active { transform: translateY(0); }

.stat-number { font-size: 28px; font-weight: 700; }
.stat-label { margin-top: 6px; color: #4b5563; }

.section-card {
  background: #fff;
  border: 1px solid #e9ecef;
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.06);
  padding: 14px 16px;
  margin-top: 12px;
}
.section-title {
  margin: 0 0 10px 0;
  font-size: 16px;
  font-weight: 600;
  color: #2c3e50;
}

/* 关系类型分布：进度条样式 */
.rel-list { display: flex; flex-direction: column; gap: 8px; }
.rel-item {
  display: grid;
  grid-template-columns: 160px 1fr auto;
  align-items: center;
  gap: 10px;
}
.rel-label { color: #334155; font-weight: 500; }
.rel-bar {
  position: relative;
  height: 10px;
  background: linear-gradient(180deg, #f1f5f9, #e2e8f0);
  border-radius: 6px;
  overflow: hidden;
  box-shadow: inset 0 1px 2px rgba(0,0,0,0.06);
}
.rel-bar-fill {
  position: absolute; left: 0; top: 0; bottom: 0;
  background: linear-gradient(90deg, #60a5fa, #34d399);
  border-radius: 6px;
  transition: width 300ms ease;
}
.rel-count {
  font-weight: 600; color: #111827;
  background: #f8fafc; border: 1px solid #e5e7eb;
  padding: 4px 8px; border-radius: 10px;
  min-width: 40px; text-align: right;
}

.chip-row { display: flex; gap: 8px; flex-wrap: wrap; }
.chip { display: inline-flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 18px; font-size: 13px; border: 1px solid #e9ecef; background: #f8f9fa; color: #374151; }
.chip.green { background: #e9f7ef; border-color: #d3f9d8; }
.chip.orange { background: #fff4e6; border-color: #ffe8cc; }
.chip-count { font-weight: 700; color: #111827; }

/* 介绍区域样式 */
.intro-block {
  padding: 10px 12px;
  background: linear-gradient(90deg, #f8fafc 0%, #eef2ff 100%);
  border: 1px solid #e9ecef;
  border-radius: 10px;
  color: #1f2937;
  font-size: 14px;
  display: inline-block;
}
.intro-math {
  font-weight: 600;
  color: #2563eb;
  margin-left: 6px;
}
</style>
