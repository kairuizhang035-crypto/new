<template>
  <div class="container">
    <!-- 侧边栏 -->
    <div class="sidebar">
      <!-- 头部 -->
      <div class="header">
        <h1>增强知识图谱可视化</h1>
        <p>基于 (V, E_core, R, W, Θ, Φ)</p>
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
      </div>
    </div>

    <!-- 主内容区 -->
<div class="main-content">
      <div class="content-panel" :class="{ 'has-bottom': activePanel === 'weights' }">
        <!-- 概览统计面板 -->
        <div v-if="activePanel === 'overview'" class="panel active">
          <component 
            :is="OverviewComponent"
            :statistics="statistics"
            :relation-type-stats="relationTypeStats"
            :hierarchy-stats="hierarchyStats"
            :edges="edges"
          />
        </div>

        <!-- 节点面板 -->
        <div v-if="activePanel === 'nodes'" class="panel active">
          <input 
            v-model="nodeSearchQuery" 
            type="text" 
            class="search-box" 
            placeholder="搜索节点..."
          >
          
          <!-- 节点类型筛选 -->
          <div class="filter-section">
            <h3>节点类型筛选:</h3>
            <div class="filter-options">
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

                <!-- 每个节点类型的分页器（5页滑动窗口 + 首页/末页 + 跳转输入） -->
                <div class="pager" v-if="totalPagesByType(type) > 1">
                  <span
                    class="pager-chip"
                    :class="{ disabled: (currentPageByType[type] || 1) === 1 }"
                    @click="goPageType(type, 1)"
                  >首页</span>
                  <span
                    class="pager-chip"
                    :class="{ disabled: (currentPageByType[type] || 1) === 1 }"
                    @click="prevPageType(type)"
                  >上一页</span>
                  <span
                    v-for="p in getPageNumbersByType(type)"
                    :key="`type-${type}-p-${p}`"
                    class="pager-chip"
                    :class="{ active: (currentPageByType[type] || 1) === p }"
                    @click="goPageType(type, p)"
                  >{{ p }}</span>
                  <span
                    class="pager-chip"
                    :class="{ disabled: (currentPageByType[type] || 1) >= totalPagesByType(type) }"
                    @click="nextPageType(type)"
                  >下一页</span>
                  <span
                    class="pager-chip"
                    :class="{ disabled: (currentPageByType[type] || 1) >= totalPagesByType(type) }"
                    @click="goPageType(type, totalPagesByType(type))"
                  >末页</span>
                  <span class="pager-info">第 {{ currentPageByType[type] || 1 }} / {{ totalPagesByType(type) }} 页</span>
                  <span class="pager-jump">
                    跳转到:
                    <input
                      class="pager-input"
                      v-model="jumpInputByType[type]"
                      @keyup.enter="applyJumpType(type)"
                    />
                    <button class="pager-go" @click="applyJumpType(type)">确定</button>
                  </span>
                </div>
              </div>
            </div>
          </div>

          <!-- 简单节点列表（当没有类型统计时的后备显示，带分页与跳转） -->
          <div v-if="!nodeTypeStats" class="item-list">
            <div 
              v-for="node in pagedNodesSimple" 
              :key="node" 
              class="list-item"
              @click="showNodeDetails(node)"
            >
              <div class="item-title">{{ node }}</div>
              <div class="item-details">
                类型: {{ getNodeType(node) }}
              </div>
            </div>
            <div class="pager" v-if="totalPagesNodesSimple > 1">
              <span
                class="pager-chip"
                :class="{ disabled: currentPageNodesSimple === 1 }"
                @click="goToPageNodesSimple(1)"
              >首页</span>
              <span
                class="pager-chip"
                :class="{ disabled: currentPageNodesSimple === 1 }"
                @click="prevPageNodesSimple"
              >上一页</span>
              <span
                v-for="p in pageNumbersNodesSimple"
                :key="`nodes-simple-${p}`"
                class="pager-chip"
                :class="{ active: currentPageNodesSimple === p }"
                @click="goToPageNodesSimple(p)"
              >{{ p }}</span>
              <span
                class="pager-chip"
                :class="{ disabled: currentPageNodesSimple >= totalPagesNodesSimple }"
                @click="nextPageNodesSimple"
              >下一页</span>
              <span
                class="pager-chip"
                :class="{ disabled: currentPageNodesSimple >= totalPagesNodesSimple }"
                @click="goToPageNodesSimple(totalPagesNodesSimple)"
              >末页</span>
              <span class="pager-info">第 {{ currentPageNodesSimple }} / {{ totalPagesNodesSimple }} 页</span>
              <span class="pager-jump">
                跳转到:
                <input
                  class="pager-input"
                  v-model="jumpInputNodesSimple"
                  @keyup.enter="applyJumpNodesSimple"
                />
                <button class="pager-go" @click="applyJumpNodesSimple">确定</button>
              </span>
            </div>
          </div>
        </div>

        <!-- 边面板 -->
        <div v-if="activePanel === 'edges'" class="panel active">
          <div class="filter-section">
            <div class="filter-group">
              <label class="filter-label">关系类型筛选:</label>
              <div class="filter-options">
                <span 
                  v-for="type in availableRelationTypes" 
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
                  v-for="hierarchy in availableHierarchies" 
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
              v-for="edge in filteredEdges" 
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
        </div>

        <!-- 关系类型面板 -->
        <div v-if="activePanel === 'relations'" class="panel active">
          <div class="item-list">
            <div 
              v-for="(relation, key) in relations" 
              :key="key" 
              class="list-item"
            >
              <div class="item-title">{{ relation.name || key }}</div>
              <div class="item-details">
                <div>{{ relation.description }}</div>
                <div>语义: {{ relation.semantic }}</div>
              </div>
            </div>
          </div>
        </div>

        <!-- 权重系统面板 -->
        <div v-if="activePanel === 'weights'" class="panel active">
          <input 
            v-model="weightSearchQuery" 
            type="text" 
            class="search-box" 
            placeholder="搜索权重关系..."
          >
          <div class="item-list">
            <div 
              v-for="entry in pagedWeights" 
              :key="entry.key" 
              class="list-item"
              @click="showWeightDetails(entry.key, entry.value)"
            >
              <div class="item-title">{{ entry.key }}</div>
              <div class="item-details">
                <div>质量等级: <span :class="`quality-${entry.value.base_weight?.quality_level?.toLowerCase()}`">
                  {{ entry.value.base_weight?.quality_level }}
                </span></div>
                <div>综合评分: {{ entry.value.base_weight?.integrated_score?.toFixed(4) }}</div>
                <div>支持算法: {{ formatAlgList(entry.value) }}</div>
              </div>
            </div>
          </div>
        </div>

        <!-- 参数学习面板 -->
        <div v-if="activePanel === 'parameters'" class="panel active">
          <input 
            v-model="parameterSearchQuery" 
            type="text" 
            class="search-box" 
            placeholder="搜索参数节点..."
          >
          <div class="item-list">
            <div 
              v-for="(param, key) in filteredParameters" 
              :key="key" 
              class="list-item"
              @click="showParameterDetails(key, param)"
            >
              <div class="item-title">{{ key }}</div>
              <div class="item-details">
                <div v-if="param.MLE">MLE参数: 可用</div>
                <div v-if="param.Bayesian">贝叶斯参数: 可用</div>
                <div v-if="param.EM">EM参数: 可用</div>
                <div v-if="param.SEM">SEM参数: 可用</div>
              </div>
            </div>
          </div>
        </div>

        <!-- 路径分析面板 -->
        <div v-if="activePanel === 'pathways'" class="panel active">
          <input 
            v-model="pathwaySearchQuery" 
            type="text" 
            class="search-box" 
            placeholder="搜索路径..."
          >
          <div class="item-list">
            <div 
              v-for="(pathway, key) in filteredPathways" 
              :key="key" 
              class="list-item"
              @click="showPathwayDetails(key, pathway)"
            >
              <div class="item-title">{{ key }}</div>
              <div class="item-details">
                <div v-if="pathway.direct_effect">直接效应: {{ pathway.direct_effect.toFixed(4) }}</div>
                <div v-if="pathway.indirect_effect">间接效应: {{ pathway.indirect_effect.toFixed(4) }}</div>
                <div v-if="pathway.total_effect">总效应: {{ pathway.total_effect.toFixed(4) }}</div>
              </div>
            </div>
          </div>
        </div>

        <!-- 网络图谱面板 -->
        <div v-if="activePanel === 'graph'" class="panel active">
          <div class="graph-container">
            <div class="controls">
              <button class="control-btn" @click="resetGraph">重置视图</button>
              <button class="control-btn" @click="fitGraph">适应窗口</button>
              <button class="control-btn" @click="togglePhysics">
                {{ physicsEnabled ? '禁用' : '启用' }}物理引擎
              </button>
            </div>
            <div id="network-graph" ref="networkContainer"></div>
          </div>
        </div>
      </div>
      <!-- 权重系统底部分页栏：贴底且不覆盖左侧界面 -->
      <div v-if="activePanel === 'weights'" class="bottom-pager">
        <div class="pager">
          <span class="pager-chip" @click="goToPageW(1)">首页</span>
          <span class="pager-chip" @click="prevPageW">上一页</span>
          <span
            v-for="it in pageNumbersW"
            :key="`w-${it.type}-${it.page ?? it.idx}`"
            class="pager-chip"
            :class="{ active: it.type === 'page' && it.page === currentPageW }"
            @click="it.type === 'page' && goToPageW(it.page)"
          >{{ it.type === 'page' ? it.page : '...' }}</span>
          <span class="pager-chip" @click="nextPageW">下一页</span>
          <span class="pager-chip" @click="goToPageW(totalPagesW)">末页</span>
          <span class="pager-info">第 {{ currentPageW }} / {{ totalPagesW }} 页</span>
          <span class="pager-jump">
            跳转到:
            <input class="pager-input" v-model="goInputW" @keyup.enter="goToPageW(Number(goInputW))" />
            <button class="pager-go" @click="goToPageW(Number(goInputW))">确定</button>
          </span>
        </div>
      </div>
    </div>

    <!-- 详情模态框 -->
    <div v-if="showModal" class="detail-modal" @click="closeModal">
      <div class="modal-content" @click.stop>
        <div class="modal-header">
          <h3>{{ modalTitle }}</h3>
          <button class="close-btn" @click="closeModal">&times;</button>
        </div>
        <div v-html="modalContent"></div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, reactive, computed, onMounted, nextTick, watch } from 'vue'
import { useKnowledgeGraphStore } from '../状态管理/知识图谱状态'
import { Network } from 'vis-network/standalone'
import 概览统计组件 from './概览统计组件.vue'

export default {
  name: 'KnowledgeGraph',
  setup() {
    const store = useKnowledgeGraphStore()
    
    // 响应式数据
    const activePanel = ref('overview')
    const nodeSearchQuery = ref('')
    const weightSearchQuery = ref('')
    const parameterSearchQuery = ref('')
    const pathwaySearchQuery = ref('')
    const selectedRelationTypes = ref([])
    const selectedHierarchies = ref([])
    const selectedNodeTypes = ref([])
    const expandedTypes = ref(['疾病', '药物', '检验']) // 默认展开所有类型
    const nodeTypeStats = ref(null)
    const showModal = ref(false)
    const modalTitle = ref('')
    const modalContent = ref('')
    const physicsEnabled = ref(true)
    const networkContainer = ref(null)
    
    let network = null

    // 动态组件别名
    const OverviewComponent = 概览统计组件

    // 计算属性
    const statistics = computed(() => store.statistics)
    const nodes = computed(() => store.nodes)
    const edges = computed(() => store.edges)
    const relations = computed(() => store.relations)
    const weights = computed(() => store.weights)
    const parameters = computed(() => store.parameters)
    const pathways = computed(() => store.pathways)

    const relationTypeStats = computed(() => {
      const stats = {}
      edges.value.forEach(edge => {
        stats[edge.relation_type] = (stats[edge.relation_type] || 0) + 1
      })
      return stats
    })

    const hierarchyStats = computed(() => {
      const stats = {}
      edges.value.forEach(edge => {
        stats[edge.edge_hierarchy] = (stats[edge.edge_hierarchy] || 0) + 1
      })
      return stats
    })

    const availableRelationTypes = computed(() => {
      return [...new Set(edges.value.map(edge => edge.relation_type))]
    })

    const availableHierarchies = computed(() => {
      return [...new Set(edges.value.map(edge => edge.edge_hierarchy))]
    })

    const availableNodeTypes = computed(() => {
      const types = new Set()
      nodes.value.forEach(node => {
        if (node.startsWith('疾病_')) types.add('疾病')
        else if (node.startsWith('药物_')) types.add('药物')
        else if (node.startsWith('检验_')) types.add('检验')
        else types.add('其他')
      })
      return Array.from(types)
    })

    // 算法中文映射与列表格式化
    const mapAlgCn = (alg) => {
      const dict = {
        'MLE': '极大似然',
        'Bayesian': '贝叶斯',
        'EM': 'EM',
        'SEM': '结构方程',
        'Pearson': '皮尔逊',
        'Spearman': '斯皮尔曼'
      }
      const k = String(alg || '')
      return dict[k] || k || '未知'
    }
    const getSupportAlgorithmsFromWeight = (w) => {
      if (!w) return []
      const list = w?.candidate_details?.support_algorithms
        || w?.base_weight?.support_algorithms
        || []
      if (Array.isArray(list)) return list
      return list ? [list] : []
    }
    const formatAlgList = (w) => {
      const list = getSupportAlgorithmsFromWeight(w)
      if (!list.length) return '无'
      return list.map(mapAlgCn).join('，')
    }

    const filteredNodes = computed(() => {
      if (!nodeSearchQuery.value) return nodes.value
      return nodes.value.filter(node => 
        node.toLowerCase().includes(nodeSearchQuery.value.toLowerCase())
      )
    })

    // 简单节点列表分页（当没有类型统计时启用，5页滑窗 + 跳转）
    const pageSizeNodesSimple = ref(5)
    const currentPageNodesSimple = ref(1)
    const jumpInputNodesSimple = ref('')

    const totalPagesNodesSimple = computed(() => Math.max(1, Math.ceil((filteredNodes.value || []).length / pageSizeNodesSimple.value)))
    const pagedNodesSimple = computed(() => {
      const start = (currentPageNodesSimple.value - 1) * pageSizeNodesSimple.value
      return (filteredNodes.value || []).slice(start, start + pageSizeNodesSimple.value)
    })
    const pageNumbersNodesSimple = computed(() => {
      const total = totalPagesNodesSimple.value
      const current = currentPageNodesSimple.value
      const WINDOW = 5
      let start = current - Math.floor(WINDOW / 2)
      if (start < 1) start = 1
      let end = start + WINDOW - 1
      if (end > total) { end = total; start = Math.max(1, end - WINDOW + 1) }
      const res = []
      for (let p = start; p <= end; p++) res.push(p)
      return res
    })
    const goToPageNodesSimple = (p) => {
      const n = Number(p)
      if (!Number.isFinite(n)) return
      if (n < 1 || n > totalPagesNodesSimple.value) return
      currentPageNodesSimple.value = n
    }
    const prevPageNodesSimple = () => { if (currentPageNodesSimple.value > 1) currentPageNodesSimple.value -= 1 }
    const nextPageNodesSimple = () => { if (currentPageNodesSimple.value < totalPagesNodesSimple.value) currentPageNodesSimple.value += 1 }
    const applyJumpNodesSimple = () => { const n = Number(jumpInputNodesSimple.value); if (!Number.isFinite(n)) return; goToPageNodesSimple(n) }

    watch(filteredNodes, () => {
      // 搜索变化时重置并校正页码范围
      currentPageNodesSimple.value = 1
      const max = totalPagesNodesSimple.value
      if (currentPageNodesSimple.value > max) currentPageNodesSimple.value = max
    })

    const filteredEdges = computed(() => {
      let filtered = edges.value
      
      if (selectedRelationTypes.value.length > 0) {
        filtered = filtered.filter(edge => 
          selectedRelationTypes.value.includes(edge.relation_type)
        )
      }
      
      if (selectedHierarchies.value.length > 0) {
        filtered = filtered.filter(edge => 
          selectedHierarchies.value.includes(edge.edge_hierarchy)
        )
      }
      
      return filtered
    })

    const filteredWeights = computed(() => {
      if (!weightSearchQuery.value) return weights.value
      const query = weightSearchQuery.value.toLowerCase()
      const filtered = {}
      Object.keys(weights.value).forEach(key => {
        if (key.toLowerCase().includes(query)) {
          filtered[key] = weights.value[key]
        }
      })
      return filtered
    })

    // 权重系统分页（每页10条 + 省略号分页 + 跳页）
    const pageSizeW = ref(10)
    const currentPageW = ref(1)
    const goInputW = ref('')

    const normalizedFilteredWeights = computed(() => {
      const obj = filteredWeights.value || {}
      return Object.keys(obj).map(k => ({ key: k, value: obj[k] }))
    })

    const totalPagesW = computed(() => Math.max(1, Math.ceil(normalizedFilteredWeights.value.length / pageSizeW.value)))
    const pagedWeights = computed(() => {
      const start = (currentPageW.value - 1) * pageSizeW.value
      return normalizedFilteredWeights.value.slice(start, start + pageSizeW.value)
    })

    const pageNumbersW = computed(() => {
      const total = totalPagesW.value
      const current = currentPageW.value
      const WINDOW = 5
      let start = current - Math.floor(WINDOW / 2)
      if (start < 1) start = 1
      let end = start + WINDOW - 1
      if (end > total) {
        end = total
        start = Math.max(1, end - WINDOW + 1)
      }
      const res = []
      for (let p = start; p <= end; p++) res.push({ type: 'page', page: p })
      return res.map((it, idx) => ({ ...it, idx }))
    })

    const goToPageW = (p) => {
      const n = Number(p)
      if (!Number.isFinite(n)) return
      if (n < 1 || n > totalPagesW.value) return
      currentPageW.value = n
    }
    const prevPageW = () => { if (currentPageW.value > 1) currentPageW.value -= 1 }
    const nextPageW = () => { if (currentPageW.value < totalPagesW.value) currentPageW.value += 1 }

    // 搜索或源数据变化时，重置到第1页并校正页码范围
    watch([weightSearchQuery, weights], () => { currentPageW.value = 1 })
    watch(normalizedFilteredWeights, () => {
      const max = totalPagesW.value
      if (currentPageW.value > max) currentPageW.value = max
      if (currentPageW.value < 1) currentPageW.value = 1
    })

    const filteredParameters = computed(() => {
      if (!parameterSearchQuery.value) return parameters.value
      const query = parameterSearchQuery.value.toLowerCase()
      const filtered = {}
      Object.keys(parameters.value).forEach(key => {
        if (key.toLowerCase().includes(query)) {
          filtered[key] = parameters.value[key]
        }
      })
      return filtered
    })

    const filteredPathways = computed(() => {
      if (!pathwaySearchQuery.value) return pathways.value
      const query = pathwaySearchQuery.value.toLowerCase()
      const filtered = {}
      Object.keys(pathways.value).forEach(key => {
        if (key.toLowerCase().includes(query)) {
          filtered[key] = pathways.value[key]
        }
      })
      return filtered
    })

    // 方法
    const setActivePanel = (panel) => {
      activePanel.value = panel
      if (panel === 'graph') {
        nextTick(() => {
          initNetwork()
        })
      }
    }

    const toggleRelationType = (type) => {
      const index = selectedRelationTypes.value.indexOf(type)
      if (index > -1) {
        selectedRelationTypes.value.splice(index, 1)
      } else {
        selectedRelationTypes.value.push(type)
      }
    }

    const toggleHierarchy = (hierarchy) => {
      const index = selectedHierarchies.value.indexOf(hierarchy)
      if (index > -1) {
        selectedHierarchies.value.splice(index, 1)
      } else {
        selectedHierarchies.value.push(hierarchy)
      }
    }

    const toggleNodeType = (type) => {
      const index = selectedNodeTypes.value.indexOf(type)
      if (index > -1) {
        selectedNodeTypes.value.splice(index, 1)
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

    const getNodeType = (node) => {
      if (node.startsWith('疾病_')) return '疾病'
      if (node.startsWith('药物_')) return '药物'
      if (node.startsWith('检验_')) return '检验'
      return '其他'
    }

    const getNodeTypeLabel = (type) => {
      const labels = {
        '疾病': '疾病',
        '药物': '药物',
        '检验': '检验',
        '其他': '其他'
      }
      return labels[type] || type
    }

    const getNodeTypeIcon = (type) => {
      const icons = {
        '疾病': '🦠',
        '药物': '💊',
        '检验': '🔬',
        '其他': '❓'
      }
      return icons[type] || '●'
    }

    const getFilteredNodesByType = (type) => {
      if (!nodeTypeStats.value || !nodeTypeStats.value[type]) return []
      
      let nodes = nodeTypeStats.value[type].nodes || []
      
      // 应用搜索筛选
      if (nodeSearchQuery.value) {
        const query = nodeSearchQuery.value.toLowerCase()
        nodes = nodes.filter(node => 
          node.name.toLowerCase().includes(query) || 
          node.id.toLowerCase().includes(query)
        )
      }
      
      return nodes
    }

    // 节点集合分页（每个类型单独分页，5页滑动窗口 + 跳页）
    const pageSizeNodes = ref(5)
    const currentPageByType = reactive({})
    const jumpInputByType = reactive({})

    const totalPagesByType = (type) => {
      const list = getFilteredNodesByType(type)
      return Math.max(1, Math.ceil(list.length / pageSizeNodes.value))
    }

    const getPagedNodesByType = (type) => {
      const list = getFilteredNodesByType(type)
      const total = Math.max(1, Math.ceil(list.length / pageSizeNodes.value))
      const cur = Number(currentPageByType[type] || 1)
      const safeCur = Math.max(1, Math.min(cur, total))
      // 规范化当前页（例如筛选导致总页数变少）
      currentPageByType[type] = safeCur
      const start = (safeCur - 1) * pageSizeNodes.value
      return list.slice(start, start + pageSizeNodes.value)
    }

    const getPageNumbersByType = (type) => {
      const total = totalPagesByType(type)
      const current = Number(currentPageByType[type] || 1)
      const WINDOW = 5
      let start = current - Math.floor(WINDOW / 2)
      if (start < 1) start = 1
      let end = start + WINDOW - 1
      if (end > total) {
        end = total
        start = Math.max(1, end - WINDOW + 1)
      }
      const res = []
      for (let p = start; p <= end; p++) res.push(p)
      return res
    }

    const goPageType = (type, p) => {
      const n = Number(p)
      if (!Number.isFinite(n)) return
      const total = totalPagesByType(type)
      if (n < 1 || n > total) return
      currentPageByType[type] = n
    }
    const prevPageType = (type) => {
      const cur = Number(currentPageByType[type] || 1)
      if (cur > 1) currentPageByType[type] = cur - 1
    }
    const nextPageType = (type) => {
      const cur = Number(currentPageByType[type] || 1)
      const total = totalPagesByType(type)
      if (cur < total) currentPageByType[type] = cur + 1
    }
    const applyJumpType = (type) => {
      const val = jumpInputByType[type]
      const n = Number(val)
      if (!Number.isFinite(n)) return
      goPageType(type, n)
    }

    // 搜索或类型统计变化时：为每个类型重置/校正当前页
    watch([nodeSearchQuery, nodeTypeStats], () => {
      const stats = nodeTypeStats.value || {}
      Object.keys(stats).forEach(t => {
        const total = totalPagesByType(t)
        const cur = Number(currentPageByType[t] || 1)
        currentPageByType[t] = Math.min(Math.max(1, cur), total)
        // 如果筛选条件变化，默认回到第 1 页
        if (nodeSearchQuery.value) currentPageByType[t] = 1
      })
    })

    const getHierarchyLabel = (hierarchy) => {
      const labels = {
        'triangulated_verified': '三角验证',
        'non_triangulated': '非三角验证',
        'candidate_only': '候选边'
      }
      return labels[hierarchy] || hierarchy
    }

    const showNodeDetails = (node) => {
      modalTitle.value = `节点详情: ${node}`
      modalContent.value = `
        <div class="detail-section">
          <h4>基本信息</h4>
          <p><strong>节点名称:</strong> ${node}</p>
          <p><strong>节点类型:</strong> ${getNodeType(node)}</p>
        </div>
      `
      showModal.value = true
    }

    const showEdgeDetails = (edge) => {
      modalTitle.value = `边详情: ${edge.source} → ${edge.target}`
      modalContent.value = `
        <div class="detail-section">
          <h4>基本信息</h4>
          <p><strong>源节点:</strong> ${edge.source}</p>
          <p><strong>目标节点:</strong> ${edge.target}</p>
          <p><strong>关系类型:</strong> ${edge.relation_type}</p>
          <p><strong>边层次:</strong> ${getHierarchyLabel(edge.edge_hierarchy)}</p>
          <p><strong>是否直接:</strong> ${edge.is_direct ? '是' : '否'}</p>
          <p><strong>权重引用:</strong> ${edge.weight_ref || 'N/A'}</p>
        </div>
      `
      showModal.value = true
    }

    const showWeightDetails = (key, weight) => {
      modalTitle.value = `权重详情: ${key}`
      const baseWeight = weight.base_weight || {}
      const candidateDetails = weight.candidate_details || {}
      const algListText = formatAlgList(weight)
      
      modalContent.value = `
        <div class="detail-section">
          <h4>基础权重信息</h4>
          <p><strong>质量等级:</strong> <span class="quality-${baseWeight.quality_level?.toLowerCase()}">${baseWeight.quality_level}</span></p>
          <p><strong>综合评分:</strong> ${baseWeight.integrated_score?.toFixed(4)}</p>
          <p><strong>支持算法:</strong> ${algListText}</p>
        </div>
        <div class="detail-section">
          <h4>候选详情</h4>
          <p><strong>频率评分:</strong> ${candidateDetails.frequency_score?.toFixed(4)}</p>
          <p><strong>多样性评分:</strong> ${candidateDetails.diversity_score?.toFixed(4)}</p>
          <p><strong>综合评分:</strong> ${candidateDetails.comprehensive_score?.toFixed(4)}</p>
          <p><strong>算法一致性:</strong> ${candidateDetails.algorithm_consistency?.toFixed(4)}</p>
          <p><strong>网络拓扑:</strong> ${candidateDetails.network_topology?.toFixed(4)}</p>
          <p><strong>统计显著性:</strong> ${candidateDetails.statistical_significance?.toFixed(4)}</p>
        </div>
      `
      showModal.value = true
    }

    const showParameterDetails = (key, param) => {
      modalTitle.value = `参数详情: ${key}`
      let content = `<div class="detail-section"><h4>可用参数类型</h4>`
      
      if (param.MLE) content += `<p><strong>MLE参数:</strong> 可用</p>`
      if (param.Bayesian) content += `<p><strong>贝叶斯参数:</strong> 可用</p>`
      if (param.EM) content += `<p><strong>EM参数:</strong> 可用</p>`
      if (param.SEM) content += `<p><strong>SEM参数:</strong> 可用</p>`
      
      content += `</div>`
      modalContent.value = content
      showModal.value = true
    }

    const showPathwayDetails = (key, pathway) => {
      modalTitle.value = `路径详情: ${key}`
      modalContent.value = `
        <div class="detail-section">
          <h4>效应分析</h4>
          <p><strong>直接效应:</strong> ${pathway.direct_effect?.toFixed(4) || 'N/A'}</p>
          <p><strong>间接效应:</strong> ${pathway.indirect_effect?.toFixed(4) || 'N/A'}</p>
          <p><strong>总效应:</strong> ${pathway.total_effect?.toFixed(4) || 'N/A'}</p>
        </div>
      `
      showModal.value = true
    }

    const closeModal = () => {
      showModal.value = false
      modalTitle.value = ''
      modalContent.value = ''
    }

    // 颜色与宽度美化工具
    const getNodeColor = (node) => {
      if ((node || '').startsWith('疾病_')) return '#e74c3c'
      if ((node || '').startsWith('药物_')) return '#3498db'
      if ((node || '').startsWith('检验_')) return '#2ecc71'
      return '#95a5a6'
    }

    const getEdgeColor = (hierarchy) => {
      switch (hierarchy) {
        case 'triangulated_verified': return '#27ae60'
        case 'non_triangulated': return '#f39c12'
        case 'candidate_only': return '#95a5a6'
        default: return '#bdc3c7'
      }
    }

    const getEdgeWidth = (weight) => {
      if (!weight) return 1
      return Math.max(1, Math.min(5, (typeof weight === 'number' ? weight : Number(weight)) * 5))
    }

    const lightenColor = (hex, percent) => {
      try {
        const p = Math.max(-100, Math.min(100, percent || 0))
        const num = parseInt(String(hex).replace('#', ''), 16)
        let r = (num >> 16) & 0xff
        let g = (num >> 8) & 0xff
        let b = num & 0xff
        r = Math.min(255, Math.max(0, Math.round(r + (255 - r) * p / 100)))
        g = Math.min(255, Math.max(0, Math.round(g + (255 - g) * p / 100)))
        b = Math.min(255, Math.max(0, Math.round(b + (255 - b) * p / 100)))
        return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)
      } catch (e) { return hex }
    }

    const initNetwork = async () => {
      if (!networkContainer.value) return
      
      try {
        const graphData = await store.getGraphData()
        const rawNodes = graphData?.nodes || nodes.value || []
        const rawEdges = graphData?.edges || edges.value || []

        // 计算度数，用于节点大小与提示
        const degreeMap = new Map()
        rawNodes.forEach(n => degreeMap.set(n, { in: 0, out: 0 }))
        rawEdges.forEach(e => {
          const s = e.source ?? e.from
          const t = e.target ?? e.to
          const out = degreeMap.get(s)
          if (out) out.out += 1
          const inp = degreeMap.get(t)
          if (inp) inp.in += 1
        })

        // 转换为 vis-network 数据结构，并加入方向箭头与悬浮提示
        const visNodes = rawNodes.map(node => {
          const deg = degreeMap.get(node) || { in: 0, out: 0 }
          const size = Math.max(10, Math.min(26, 12 + (deg.in + deg.out) * 3))
          return {
            id: node,
            label: node,
            color: getNodeColor(node),
            size,
            font: { size: 14 },
            shadow: true,
            title: `节点: ${node}\n类型: ${getNodeType(node)}\n入度: ${deg.in} | 出度: ${deg.out}`
          }
        })

        const visEdges = rawEdges.map(edge => {
          const source = edge.source ?? edge.from
          const target = edge.target ?? edge.to
          const base = getEdgeColor(edge.edge_hierarchy)
          const hover = lightenColor(base, 10)
          const highlight = lightenColor(base, 20)
          return {
            id: `${source}-${target}`,
            from: source,
            to: target,
            label: edge.relation_type,
            color: { color: base, hover, highlight },
            width: getEdgeWidth(edge.weight_ref),
            shadow: true,
            arrows: 'to',
            title: `${source} → ${target}\n关系: ${edge.relation_type || '未知'}\n层次: ${getHierarchyLabel(edge.edge_hierarchy) || '未知'}\n直接关系: ${edge.is_direct ? '是' : '否'}`
          }
        })

        const options = {
          nodes: {
            shape: 'dot',
            size: 16,
            font: {
              size: 12,
              color: '#333333'
            },
            borderWidth: 2,
            shadow: true
          },
          edges: {
            width: 2,
            color: { inherit: 'from' },
            smooth: {
              type: 'continuous'
            },
            arrows: { to: { enabled: true, scaleFactor: 2, type: 'arrow' } },
            arrowStrikethrough: false
          },
          physics: {
            enabled: physicsEnabled.value,
            stabilization: { iterations: 100 }
          },
          interaction: {
            hover: true,
            hoverConnectedEdges: true,
            tooltipDelay: 200,
            navigationButtons: true
          }
        }

        network = new Network(networkContainer.value, { nodes: visNodes, edges: visEdges }, options)
        
        network.on('click', (params) => {
          if (params.nodes.length > 0) {
            const nodeId = params.nodes[0]
            showNodeDetails(nodeId)
          } else if (params.edges.length > 0) {
            const edgeId = params.edges[0]
            const edge = edges.value.find(e => `${e.source}-${e.target}` === edgeId)
            if (edge) showEdgeDetails(edge)
          }
        })
      } catch (error) {
        console.error('初始化网络图失败:', error)
      }
    }

    const resetGraph = () => {
      if (network) {
        network.fit()
      }
    }

    const fitGraph = () => {
      if (network) {
        network.fit()
      }
    }

    const togglePhysics = () => {
      physicsEnabled.value = !physicsEnabled.value
      if (network) {
        network.setOptions({ physics: { enabled: physicsEnabled.value } })
      }
    }

    // 加载节点类型统计数据
    const loadNodeTypeStats = async () => {
      try {
        const response = await store.api.get('/nodes/types')
        if (response.data.success) {
          nodeTypeStats.value = response.data.data
        }
      } catch (error) {
        console.error('加载节点类型统计失败:', error)
      }
    }

    // 生命周期
    onMounted(async () => {
      await store.loadData()
      await loadNodeTypeStats()
    })

    return {
      // 响应式数据
      activePanel,
      nodeSearchQuery,
      weightSearchQuery,
      parameterSearchQuery,
      pathwaySearchQuery,
      selectedRelationTypes,
      selectedHierarchies,
      selectedNodeTypes,
      expandedTypes,
      nodeTypeStats,
      showModal,
      modalTitle,
      modalContent,
      physicsEnabled,
      networkContainer,
      
      // 计算属性
      statistics,
      nodes,
      edges,
      relations,
      weights,
      parameters,
      pathways,
      relationTypeStats,
      hierarchyStats,
      availableRelationTypes,
      availableHierarchies,
      availableNodeTypes,
      filteredNodes,
      // 简单节点列表分页（无类型统计时）
      pageSizeNodesSimple,
      currentPageNodesSimple,
      totalPagesNodesSimple,
      pagedNodesSimple,
      pageNumbersNodesSimple,
      jumpInputNodesSimple,
      goToPageNodesSimple,
      prevPageNodesSimple,
      nextPageNodesSimple,
      applyJumpNodesSimple,
      filteredEdges,
      filteredWeights,
      // 权重分页相关
      pageSizeW,
      currentPageW,
      totalPagesW,
      pagedWeights,
      pageNumbersW,
      goInputW,
      goToPageW,
      prevPageW,
      nextPageW,
      // 节点集合分页相关（按类型）
      pageSizeNodes,
      currentPageByType,
      totalPagesByType,
      getPagedNodesByType,
      getPageNumbersByType,
      jumpInputByType,
      goPageType,
      prevPageType,
      nextPageType,
      applyJumpType,
      filteredParameters,
      filteredPathways,
      formatAlgList,
      
      // 方法
      setActivePanel,
      toggleRelationType,
      toggleHierarchy,
      toggleNodeType,
      toggleTypeExpansion,
      getNodeType,
      getNodeTypeLabel,
      getNodeTypeIcon,
      getFilteredNodesByType,
      getHierarchyLabel,
      showNodeDetails,
      showEdgeDetails,
      showWeightDetails,
      showParameterDetails,
      showPathwayDetails,
      closeModal,
      resetGraph,
      fitGraph,
      togglePhysics
    }
  }
}
</script>

<style scoped>
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

.panel.active { display: block; }

.content-panel.has-bottom { padding-bottom: 64px; /* 避免列表被底部分页栏遮挡 */ }
.bottom-pager {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: rgba(255, 255, 255, 0.95);
  border-top: 1px solid #dee2e6;
  box-shadow: 0 -4px 12px rgba(0,0,0,0.06);
  padding: 8px 12px;
}
.pager {
  display: flex;
  align-items: center;
  gap: 6px;
  justify-content: center;
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
.pager-info {
  margin-left: 8px;
  color: #6c757d;
  font-size: 12px;
}
.pager-jump {
  margin-left: 12px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #495057;
  font-size: 13px;
}
.pager-input {
  width: 56px;
  padding: 6px 8px;
  border: 1px solid #dee2e6;
  border-radius: 8px;
  font-size: 13px;
  outline: none;
}
.pager-go {
  padding: 6px 10px;
  border-radius: 10px;
  border: 1px solid #dee2e6;
  background: #fff;
  cursor: pointer;
  font-size: 13px;
}
.pager-chip.disabled {
  opacity: 0.5;
  cursor: not-allowed;
  pointer-events: none;
}
.content-panel.has-bottom {
  padding-bottom: 64px; /* 避免列表被底部分页栏遮挡 */
}
</style>
