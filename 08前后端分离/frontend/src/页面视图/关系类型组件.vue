<template>
  <div class="relations-panel">
    <!-- 顶部工具栏，与 E_core/V 保持风格一致 -->
    <div class="toolbar">
      <div class="title">关系类型 (R)</div>
      <div class="actions">
        <button class="btn primary" @click="$emit('refresh-relations')">刷新数据</button>
      </div>
    </div>

  <div class="item-list">
      <div 
        v-for="(meta, key) in renderedRelations" 
        :key="key" 
        class="list-item"
        @click="onClickRelation(key)"
      >
        <div class="item-title">
          <span class="relation-name">{{ meta.name || key }}</span>
        </div>
        <div class="item-details">
          <div class="detail-row">
            <span class="detail-label">总边数:</span>
            <span class="detail-value count">{{ meta.stats?.total || 0 }}</span>
          </div>
          <div class="badge-row">
            <span class="hierarchy-badge tri">三角验证: {{ meta.stats?.triangulated_verified || 0 }}</span>
            <span class="hierarchy-badge non">非三角: {{ meta.stats?.non_triangulated || 0 }}</span>
            <span class="hierarchy-badge unk" v-if="meta.stats?.unknown_hierarchy">未知: {{ meta.stats?.unknown_hierarchy }}</span>
          </div>
          <div class="examples" v-if="meta.stats?.examples?.length">
            <div class="examples-title">示例边:</div>
            <div class="examples-list">
              <span
                v-for="ex in meta.stats.examples"
                :key="`${ex.source}-${ex.target}`"
                class="edge-chip"
              >
                {{ ex.source }} → {{ ex.target }}
              </span>
            </div>
          </div>
        </div>
      </div>
      <div v-if="Object.keys(renderedRelations).length === 0" class="empty">暂无关系类型</div>
    </div>
  </div>
</template>

<script>
export default {
  name: '关系类型组件',
  props: {
    relations: {
      type: Object,
      default: () => ({})
    },
    relationStats: {
      type: Object,
      default: () => ({})
    }
  },
  emits: ['refresh-relations', 'show-relation-details'],
  computed: {
    renderedRelations() {
      // 将基本元数据 relations 与统计 relationStats 进行合并用于渲染
      const result = {}
      const keys = new Set([
        ...Object.keys(this.relations || {}),
        ...Object.keys(this.relationStats || {})
      ])
      keys.forEach(k => {
        result[k] = {
          ...(this.relations?.[k] || {}),
          stats: this.relationStats?.[k] || {}
        }
      })
      return result
    }
  },
  methods: {
    onClickRelation(type) {
      this.$emit('show-relation-details', type)
    },
    // 关系类型列表页不直接跳转到边详情，示例边仅做展示
  }
}
</script>

<style scoped>
.relations-panel {
  padding: 20px;
  /* 面板自身不产生横向/纵向的局部滚动，由父页面统一滚动 */
  overflow-x: hidden !important;
  overflow-y: visible !important;
  background: #fff;
}

.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 14px;
  /* 窄屏换行，避免产生横向滚动 */
  flex-wrap: wrap;
}

.toolbar .title {
  font-size: 18px;
  font-weight: 600;
  color: #2c3e50;
}

.toolbar .actions { display: flex; gap: 8px; }
.toolbar .btn {
  padding: 6px 12px;
  border-radius: 8px;
  border: 1px solid #dee2e6;
  background: #fff;
  color: #34495e;
  font-size: 13px;
  cursor: pointer;
  transition: all .2s ease;
}
.toolbar .btn:hover { background: #f8f9fa; }
.toolbar .btn.primary {
  border-color: #3b82f6;
  color: #fff;
  background: #3b82f6;
}
.toolbar .btn.primary:hover { background: #2563eb; }

.item-list {
  max-height: none;
  /* 禁止内部横向滚动，纵向不做局部滚动 */
  overflow-x: hidden !important;
  overflow-y: visible !important;
}

.list-item {
  padding: 20px;
  border: 1px solid #e9ecef;
  border-radius: 8px;
  margin-bottom: 15px;
  background: white;
  transition: all 0.2s;
  max-width: 100%;
  /* 长文本自动换行，避免产生横向滚动 */
  word-break: break-word;
  overflow-wrap: anywhere;
}

.list-item:hover {
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  border-color: #007bff;
}

.item-title {
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 15px;
  font-size: 1.2em;
  border-bottom: 2px solid #f8f9fa;
  padding-bottom: 8px;
  /* 标题长文本换行，避免横向滚动 */
  word-break: break-word;
  overflow-wrap: anywhere;
}

.relation-name { color: #334155; }

.item-details {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.detail-row {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  /* 当标签和值过长时允许换行，避免横向滚动 */
  flex-wrap: wrap;
}

.detail-label {
  font-weight: 500;
  color: #6c757d;
  min-width: 60px;
  flex-shrink: 0;
}

.detail-value {
  color: #495057;
  flex: 1;
  line-height: 1.4;
  word-break: break-word;
  overflow-wrap: anywhere;
}

.badge-row { display: flex; gap: 8px; flex-wrap: wrap; }
.hierarchy-badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid #e5e7eb;
  background: linear-gradient(180deg,#f8fafc,#eef2f7);
  color: #374151;
}

/* 示例边列表：允许换行，避免横向滚动 */
.examples-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.edge-chip {
  display: inline-block;
  padding: 2px 6px;
  border-radius: 6px;
  background: #f1f5f9;
  border: 1px solid #e2e8f0;
  color: #334155;
  white-space: normal;
  word-break: break-word;
  overflow-wrap: anywhere;
}
.hierarchy-badge.tri { border-color: #a7f3d0; background: #ecfdf5; color: #065f46; }
.hierarchy-badge.non { border-color: #fcd34d; background: #fffbeb; color: #92400e; }
.hierarchy-badge.unk { border-color: #d1d5db; background: #f3f4f6; color: #374151; }

.examples-title { color: #374151; margin-bottom: 6px; }
.examples-list { display: flex; flex-wrap: wrap; gap: 8px; }
.edge-chip {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 10px;
  font-size: 12px;
  border: 1px solid #e5e7eb;
  background: linear-gradient(180deg,#f8fafc,#eef2f7);
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  cursor: default;
}
.hierarchy-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid #e5e7eb;
  background: #f9fafb;
}
.hierarchy-badge.tri { border-color: #93c5fd; background: #eff6ff; }
.hierarchy-badge.non { border-color: #fcd34d; background: #fffbeb; }
.hierarchy-badge.cand { border-color: #a7f3d0; background: #ecfdf5; }
.hierarchy-badge.unk { border-color: #e5e7eb; background: #f3f4f6; }

.examples { margin-top: 6px; }
.examples-title { color: #6b7280; font-size: 12px; margin-bottom: 4px; }
.examples-list { display: flex; gap: 6px; flex-wrap: wrap; }
.edge-chip {
  display: inline-block;
  padding: 4px 8px;
  border-radius: 10px;
  background: #f1f5f9;
  color: #334155;
  border: 1px solid #e2e8f0;
  cursor: pointer;
  transition: all .2s ease;
}
.edge-chip:hover { box-shadow: 0 2px 6px rgba(0,0,0,0.08); transform: translateY(-1px); }

.empty { color: #64748b; text-align: center; padding: 20px; }
</style>