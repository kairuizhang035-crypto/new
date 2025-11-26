<template>
  <div class="graph-panel" :class="[themeMode === 'neon' ? 'neon-theme' : 'light-theme']">
    <div class="graph-container" ref="graphContainer">
      <div class="controls">
        <button class="control-btn" @click="resetGraph">重置视图</button>
        <button class="control-btn" @click="fitGraph">适应窗口</button>
        <button class="control-btn" @click="togglePhysics">
          {{ physicsEnabled ? '禁用' : '启用' }}物理引擎
        </button>
        <button class="control-btn" @click="exportPng">导出 PNG</button>
        <button class="control-btn" @click="toggleLayout">{{ hierarchicalEnabled ? '禁用' : '启用' }}层次布局</button>
        <button class="control-btn" @click="toggleShadow">{{ shadowEnabled ? '禁用' : '启用' }}阴影</button>
        <button class="control-btn" @click="toggleTheme">主题: {{ themeMode === 'neon' ? '霓虹' : '轻盈' }}</button>
        <button class="control-btn" @click="toggleEdgeLabelMode">边标签: {{ edgeLabelMode === 'direction' ? '方向' : '关系' }}</button>
        <div class="direction-hint">方向: 源节点 → 目标节点</div>
        <div class="control-range-wrap">
          <label class="control-label">松散度: <span class="range-value">{{ loosenessPct }}%</span></label>
          <input class="control-range" type="range" min="0" max="100" step="1" v-model="loosenessPct" />
        </div>
        <div class="control-range-wrap">
          <label class="control-label">标签字号: <span class="range-value">{{ labelFontSize }}px</span></label>
          <input class="control-range" type="range" min="12" max="28" step="1" v-model="labelFontSize" />
        </div>
        <div class="control-range-wrap">
          <label class="control-label">箭头大小: <span class="range-value">{{ arrowScale.toFixed(1) }}</span></label>
          <input class="control-range" type="range" min="1" max="4" step="0.1" v-model.number="arrowScale" />
        </div>
        <div class="control-range-wrap">
          <label class="control-label">发光强度: <span class="range-value">{{ glowStrength }}</span></label>
          <input class="control-range" type="range" min="0" max="100" step="1" v-model.number="glowStrength" />
        </div>
        <div class="control-range-wrap">
          <label class="control-label">边曲率: <span class="range-value">{{ edgeCurvature.toFixed(2) }}</span></label>
          <input class="control-range" type="range" min="0" max="0.7" step="0.01" v-model.number="edgeCurvature" />
        </div>
      </div>
      <!-- 背景层：极光与粒子光斑以及网格覆盖 -->
      <div class="bg-layer" aria-hidden="true">
        <div class="aurora"></div>
        <div class="grid-overlay"></div>
        <div class="particles">
          <span v-for="(ps, idx) in particleStyles" :key="idx" class="particle" :style="ps"></span>
        </div>
      </div>
      <div id="network-graph" ref="networkContainer" class="network-container"></div>
      <!-- 介绍卡片（替代右下角旧图例） -->
      <div
        class="intro-card"
        ref="introCardRef"
        :style="{ top: introCardPos.top + 'px', left: introCardPos.left + 'px' }"
        :class="{ dragging: introDragging }"
        @pointerdown="startDragIntro"
      >
        <div class="intro-title">知识图谱图例</div>
        <div class="intro-section">节点类型:</div>
        <div class="intro-item"><span class="dot dot-disease"></span> 疾病 ({{ diseaseCount }})</div>
        <div class="intro-item"><span class="dot dot-drug"></span> 药物 ({{ drugCount }})</div>
        <div class="intro-item"><span class="dot dot-exam"></span> 检验 ({{ examCount }})</div>
        <div class="intro-sep"></div>
        <div class="intro-meta">总节点: {{ totalNodes }}</div>
        <div class="intro-meta">总边数: {{ totalEdges }}</div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, onBeforeUnmount, nextTick, watch, computed } from 'vue'
import { Network } from 'vis-network/standalone'

export default {
  name: '网络图谱组件',
  props: {
    nodes: {
      type: Array,
      default: () => []
    },
    edges: {
      type: Array,
      default: () => []
    }
  },
  emits: ['show-node-details', 'show-edge-details'],
  setup(props, { emit }) {
    const networkContainer = ref(null)
    const graphContainer = ref(null)
    const introCardRef = ref(null)
    const introCardPos = ref({ top: 220, left: 12 })
    const introDragging = ref(false)
    let dragOffsetX = 0
    let dragOffsetY = 0
    const physicsEnabled = ref(true)
    const loosenessPct = ref(50)
    const labelFontSize = ref(18)
    const hierarchicalEnabled = ref(false)
    const shadowEnabled = ref(true)
    const arrowScale = ref(2.0)
    const edgeLabelMode = ref('direction')
    const themeMode = ref((() => { try { return localStorage.getItem('graph_theme') || 'light' } catch(e) { return 'light' } })())
    const glowStrength = ref(40)
    const edgeCurvature = ref(0.25)
    const particleStyles = ref([])
    let network = null
    let resizeObserver = null
    let legendResizeObserver = null

    // 响应式统计信息：节点/边类型与总量
    const totalNodes = computed(() => (Array.isArray(props.nodes) ? props.nodes.length : 0))
    const totalEdges = computed(() => (Array.isArray(props.edges) ? props.edges.length : 0))
    const diseaseCount = computed(() =>
      (Array.isArray(props.nodes) ? props.nodes : []).reduce((acc, n) => acc + (typeof n === 'string' && n.startsWith('疾病_') ? 1 : 0), 0)
    )
    const drugCount = computed(() =>
      (Array.isArray(props.nodes) ? props.nodes : []).reduce((acc, n) => acc + (typeof n === 'string' && n.startsWith('药物_') ? 1 : 0), 0)
    )
    const examCount = computed(() =>
      (Array.isArray(props.nodes) ? props.nodes : []).reduce((acc, n) => acc + (typeof n === 'string' && n.startsWith('检验_') ? 1 : 0), 0)
    )
    const triangulatedCount = computed(() =>
      (Array.isArray(props.edges) ? props.edges : []).reduce((acc, e) => acc + (e && e.edge_hierarchy === 'triangulated_verified' ? 1 : 0), 0)
    )
    const candidateCount = computed(() =>
      (Array.isArray(props.edges) ? props.edges : []).reduce((acc, e) => acc + (e && e.edge_hierarchy === 'candidate_only' ? 1 : 0), 0)
    )
    const weightDataCount = computed(() =>
      (Array.isArray(props.edges) ? props.edges : []).reduce((acc, e) => {
        const n = Number(e && e.weight_ref)
        return acc + (Number.isFinite(n) ? 1 : 0)
      }, 0)
    )

    const buildData = () => {
      // 计算节点入度/出度用于悬浮提示
      const degreeMap = new Map()
      ;(props.nodes || []).forEach(n => degreeMap.set(n, { in: 0, out: 0 }))
      ;(props.edges || []).forEach(e => {
        const s = e?.source ?? e?.from
        const t = e?.target ?? e?.to
        const out = degreeMap.get(s)
        if (out) out.out += 1
        const inp = degreeMap.get(t)
        if (inp) inp.in += 1
      })

      // 转换节点/边数据并加入悬浮提示
      const visNodes = (props.nodes || []).map(node => {
        const deg = degreeMap.get(node) || { in: 0, out: 0 }
        const computedSize = Math.max(10, Math.min(26, 12 + (deg.in + deg.out) * 3))
        const totalDeg = deg.in + deg.out
        const nodeFont = themeMode.value === 'neon'
          ? { size: labelFontSize.value, color: '#ffffff', strokeWidth: 3, strokeColor: '#0b1020' }
          : { size: labelFontSize.value, color: '#2c3e50', strokeWidth: 0 }
        return {
          id: node,
          label: node,
          color: getNodeColorObj(node, totalDeg),
          borderWidth: Math.min(7, Math.max(1, 1 + totalDeg)),
          shadow: shadowEnabled.value ? { enabled: true, size: Math.round(glowStrength.value / 5 + Math.min(12, totalDeg * 1.5)), x: 0, y: 0, color: themeMode.value === 'neon' ? 'rgba(0,255,255,0.25)' : 'rgba(0,0,0,0.10)' } : false,
          size: computedSize,
          font: nodeFont,
          title: `节点: ${node}\n类型: ${getNodeTypeLabel(node)}\n入度: ${deg.in} | 出度: ${deg.out}`
        }
      })

      const visEdges = (props.edges || []).map(edge => {
        const source = edge?.source ?? edge?.from
        const target = edge?.target ?? edge?.to
        const base = getEdgeColor(edge.edge_hierarchy)
        const hover = lightenColor(base, 10)
        const highlight = lightenColor(base, 20)
        const labelText = edgeLabelMode.value === 'direction' 
          ? `${source} → ${target}` 
          : (edge.relation_type || '')
        const edgeFont = themeMode.value === 'neon'
          ? { size: labelFontSize.value, color: '#ffffff', strokeWidth: 3, strokeColor: '#0b1020' }
          : { size: labelFontSize.value, color: '#2c3e50', strokeWidth: 0 }
        return {
          id: `${source}->${target}`,
          from: source,
          to: target,
          label: labelText,
          color: { color: base, hover, highlight },
          width: getEdgeWidth(edge.weight_ref),
          // 使用字符串形式确保在所有版本的 vis-network 中都启用指向目标的箭头
          arrows: 'to',
          font: edgeFont,
          title: `${source} → ${target}\n关系: ${edge.relation_type || '未知'}\n层次: ${edge.edge_hierarchy || '未知'}\n直接关系: ${edge.is_direct ? '是' : '否'}`
        }
      })

      return { nodes: visNodes, edges: visEdges }
    }

    const lerp = (a, b, t) => a + (b - a) * t
    const repulsionForLooseness = (pct) => {
      // 反转映射：0% 对应最松散（原先 100% 的效果）
      const t = 1 - Math.max(0, Math.min(1, (pct || 0) / 100))
      const centralGravity = lerp(0.05, 0.01, t)
      const springLength = lerp(180, 380, t)
      const springConstant = lerp(0.014, 0.006, t)
      const nodeDistance = lerp(220, 420, t)
      const damping = lerp(0.18, 0.06, t)
      return { centralGravity, springLength, springConstant, nodeDistance, damping }
    }

    const applyLooseness = () => {
      if (!network) return
      const repulsion = repulsionForLooseness(loosenessPct.value)
      network.setOptions({
        physics: {
          enabled: physicsEnabled.value,
          solver: 'repulsion',
          repulsion
        }
      })
      try { network.stabilize() } catch (e) {}
    }

    const applyLabelFontSize = () => {
      if (!network) return
      // 更新边标签字体大小（通过全局选项）
      network.setOptions({
        edges: {
          font: { size: labelFontSize.value }
        },
        nodes: {
          font: { size: labelFontSize.value }
        }
      })
      // 更新节点数据以应用新的节点标签字号
      try {
        const data = buildData()
        network.setData(data)
        network.redraw()
      } catch (e) {}
    }

    // 图例拖拽：限制在 graph-container 内部
    const clampIntroPos = (top, left) => {
      try {
        const containerEl = graphContainer.value
        const cardEl = introCardRef.value
        if (!containerEl || !cardEl) return { top, left }
        const cRect = containerEl.getBoundingClientRect()
        const cardRect = cardEl.getBoundingClientRect()
        const maxLeft = Math.max(0, cRect.width - cardRect.width)
        const maxTop = Math.max(0, cRect.height - cardRect.height)
        const clampedLeft = Math.min(Math.max(0, left), maxLeft)
        const clampedTop = Math.min(Math.max(0, top), maxTop)
        return { top: clampedTop, left: clampedLeft }
      } catch (e) { return { top, left } }
    }

    const loadIntroPos = () => {
      try {
        const saved = localStorage.getItem('graph_intro_card_pos')
        if (saved) {
          const obj = JSON.parse(saved)
          if (obj && typeof obj.top === 'number' && typeof obj.left === 'number') {
            const c = clampIntroPos(obj.top, obj.left)
            introCardPos.value = c
          }
        }
      } catch (e) {}
    }

    const startDragIntro = (e) => {
      try { e.preventDefault() } catch (_) {}
      const cardEl = introCardRef.value
      const containerEl = graphContainer.value
      if (!cardEl || !containerEl) return
      const cardRect = cardEl.getBoundingClientRect()
      dragOffsetX = e.clientX - cardRect.left
      dragOffsetY = e.clientY - cardRect.top
      introDragging.value = true
      const onMove = (evt) => {
        const cRect = containerEl.getBoundingClientRect()
        const nextLeft = evt.clientX - cRect.left - dragOffsetX
        const nextTop = evt.clientY - cRect.top - dragOffsetY
        const c = clampIntroPos(nextTop, nextLeft)
        introCardPos.value = c
      }
      const onUp = () => {
        introDragging.value = false
        try { window.removeEventListener('pointermove', onMove) } catch (e) {}
        try { window.removeEventListener('pointerup', onUp) } catch (e) {}
        try { localStorage.setItem('graph_intro_card_pos', JSON.stringify(introCardPos.value)) } catch (e) {}
      }
      window.addEventListener('pointermove', onMove)
      window.addEventListener('pointerup', onUp)
    }

    const initNetwork = () => {
      if (!networkContainer.value || !props.nodes.length) return

      const data = buildData()

      const options = {
        physics: {
          enabled: physicsEnabled.value && !hierarchicalEnabled.value,
          solver: 'repulsion',
          repulsion: {
            ...repulsionForLooseness(loosenessPct.value)
          },
          minVelocity: 0.1,
          maxVelocity: 50,
          stabilization: { iterations: 200 }
        },
        layout: hierarchicalEnabled.value ? {
          hierarchical: {
            enabled: true,
            direction: 'LR',
            sortMethod: 'directed',
            nodeSpacing: 180,
            levelSeparation: 220
          }
        } : {
          improvedLayout: true
        },
        interaction: {
          hover: true,
          selectConnectedEdges: false,
          hoverConnectedEdges: true,
          tooltipDelay: 200,
          navigationButtons: true
        },
        nodes: {
          shape: 'dot',
          scaling: {
            min: 10,
            max: 26
          },
          font: themeMode.value === 'neon' 
            ? { size: labelFontSize.value, color: '#ffffff', strokeWidth: 3, strokeColor: '#0b1020' }
            : { size: labelFontSize.value, color: '#2c3e50', strokeWidth: 0 },
          shadow: shadowEnabled.value ? { enabled: true, size: Math.round(glowStrength.value / 4 + 2), x: 0, y: 0, color: themeMode.value === 'neon' ? 'rgba(0,255,255,0.3)' : 'rgba(0,0,0,0.12)' } : false
        },
        edges: {
          smooth: {
            enabled: true,
            type: hierarchicalEnabled.value ? 'cubicBezier' : 'continuous',
            roundness: edgeCurvature.value
          },
          arrows: { to: { enabled: true, scaleFactor: arrowScale.value, type: 'arrow' } },
          arrowStrikethrough: false,
          font: themeMode.value === 'neon' 
            ? { size: labelFontSize.value, color: '#ffffff', strokeWidth: 3, strokeColor: '#0b1020' }
            : { size: labelFontSize.value, color: '#2c3e50', strokeWidth: 0 },
          shadow: shadowEnabled.value ? { enabled: true, size: Math.round(glowStrength.value / 4 + 2), x: 0, y: 0, color: themeMode.value === 'neon' ? 'rgba(0,255,255,0.3)' : 'rgba(0,0,0,0.12)' } : false
        }
      }

      network = new Network(networkContainer.value, data, options)
      // 初始化后应用一次箭头设置，确保兼容性
      applyArrowOptions()
      // 绑定点击事件用于详情展示
      network.on('click', (params) => {
        if (params.nodes && params.nodes.length) {
          const nodeId = params.nodes[0]
          emit('show-node-details', nodeId)
          return
        }
        if (params.edges && params.edges.length) {
          const edgeId = params.edges[0]
          const [source, target] = String(edgeId).split('->')
          const orig = (props.edges || []).find(e => e.source === source && e.target === target) || { source, target }
          emit('show-edge-details', orig)
        }
      })
      // 显式设置尺寸，避免百分比高度在父元素未设置明确高度时失效
      try {
        const h = networkContainer.value.clientHeight || networkContainer.value.offsetHeight || 500
        network.setSize('100%', h + 'px')
      } catch (e) {}
      // 初次绘制后适配容器
      try {
        network.redraw()
        network.fit()
      } catch (e) {}
    }

    const updateNetwork = () => {
      if (!network) {
        // 当数据从空变为非空时初始化
        if (networkContainer.value && props.nodes && props.nodes.length) {
          initNetwork()
        }
        return
      }
      const data = buildData()
      network.setData(data)
      applyArrowOptions()
    }

    const getNodeColor = (node) => {
      if (node.startsWith('疾病_')) return '#e74c3c'
      if (node.startsWith('药物_')) return '#3498db'
      if (node.startsWith('检验_')) return '#2ecc71'
      return '#95a5a6'
    }

    // 新增：根据节点类型与度数生成对象颜色（含边框/背景/高亮/悬浮）
    const getNodeColorObj = (node, totalDeg) => {
      const base = getNodeColor(node)
      const intensity = Math.min(30, Math.max(0, Math.round((totalDeg || 0) * 4)))
      const plus = themeMode.value === 'neon' ? intensity + 8 : Math.max(0, intensity - 6)
      const background = lightenColor(base, plus)
      const border = lightenColor(base, Math.min(40, plus + 10))
      return {
        border,
        background,
        highlight: {
          border: lightenColor(base, Math.min(60, plus + 28)),
          background: lightenColor(base, Math.min(50, plus + 22))
        },
        hover: {
          border: lightenColor(base, Math.min(50, plus + 18)),
          background: lightenColor(base, Math.min(40, plus + 14))
        }
      }
    }

    const getNodeTypeLabel = (node) => {
      if (!node || typeof node !== 'string') return '未知'
      if (node.startsWith('疾病_')) return '疾病'
      if (node.startsWith('药物_')) return '药物'
      if (node.startsWith('检验_')) return '检验'
      return '其他'
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
      // 兼容字符串引用的权重键（例如 weight_ref 为某个权重对象的 key）
      // 当无法解析为数字时，使用一个稳健的默认宽度，避免 NaN 导致边不渲染
      const n = typeof weight === 'number' ? weight : Number(weight)
      if (!Number.isFinite(n)) return 2
      return Math.max(1, Math.min(5, n * 5))
    }

    const lightenColor = (hex, percent) => {
      try {
        const p = Math.max(-100, Math.min(100, percent || 0))
        const num = parseInt(hex.replace('#', ''), 16)
        let r = (num >> 16) & 0xff
        let g = (num >> 8) & 0xff
        let b = num & 0xff
        r = Math.min(255, Math.max(0, Math.round(r + (255 - r) * p / 100)))
        g = Math.min(255, Math.max(0, Math.round(g + (255 - g) * p / 100)))
        b = Math.min(255, Math.max(0, Math.round(b + (255 - b) * p / 100)))
        return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)
      } catch (e) { return hex }
    }

    const toggleEdgeLabelMode = () => {
      edgeLabelMode.value = edgeLabelMode.value === 'direction' ? 'relation' : 'direction'
      updateNetwork()
    }

    const resetGraph = () => {
      if (network) {
        network.fit()
      }
    }

    const fitGraph = () => {
      if (network) {
        network.fit({
          animation: {
            duration: 1000,
            easingFunction: 'easeInOutQuad'
          }
        })
      }
    }

    const togglePhysics = () => {
      physicsEnabled.value = !physicsEnabled.value
      if (network) {
        network.setOptions({
          physics: {
            enabled: physicsEnabled.value,
            solver: 'repulsion',
            repulsion: repulsionForLooseness(loosenessPct.value)
          }
        })
      }
    }

    const applyArrowOptions = () => {
      if (!network) return
      network.setOptions({
        edges: {
          arrows: { to: { enabled: true, scaleFactor: arrowScale.value, type: 'arrow' } },
          arrowStrikethrough: false
        }
      })
    }

    const applyThemeOptions = () => {
      if (!network) return
      network.setOptions({
        nodes: { font: themeMode.value === 'neon' 
          ? { size: labelFontSize.value, color: '#ffffff', strokeWidth: 3, strokeColor: '#0b1020' } 
          : { size: labelFontSize.value, color: '#2c3e50', strokeWidth: 0 } },
        edges: { font: themeMode.value === 'neon' 
          ? { size: labelFontSize.value, color: '#ffffff', strokeWidth: 3, strokeColor: '#0b1020' } 
          : { size: labelFontSize.value, color: '#2c3e50', strokeWidth: 0 } }
      })
      try { network.redraw() } catch (e) {}
    }

    const applyLayoutOptions = () => {
      if (!network) return
      const opts = {
        physics: {
          enabled: physicsEnabled.value && !hierarchicalEnabled.value,
          solver: 'repulsion',
          repulsion: repulsionForLooseness(loosenessPct.value)
        },
        layout: hierarchicalEnabled.value ? {
          hierarchical: {
            enabled: true,
            direction: 'LR',
            sortMethod: 'directed',
            nodeSpacing: 180,
            levelSeparation: 220
          }
        } : {
          improvedLayout: true
        },
        edges: {
          smooth: {
            enabled: true,
            type: hierarchicalEnabled.value ? 'cubicBezier' : 'continuous',
            roundness: edgeCurvature.value
          },
          arrows: { to: { enabled: true, scaleFactor: arrowScale.value, type: 'arrow' } },
          arrowStrikethrough: false
        }
      }
      network.setOptions(opts)
      try { network.fit() } catch (e) {}
    }

    const toggleLayout = () => {
      hierarchicalEnabled.value = !hierarchicalEnabled.value
      applyLayoutOptions()
    }

    const applyShadowOptions = () => {
      if (!network) return
      const globalShadow = shadowEnabled.value ? { enabled: true, size: Math.round(glowStrength.value / 4 + 2), x: 0, y: 0, color: themeMode.value === 'neon' ? 'rgba(0,255,255,0.3)' : 'rgba(0,0,0,0.12)' } : false
      network.setOptions({
        nodes: { shadow: globalShadow },
        edges: { shadow: globalShadow }
      })
      try { network.redraw() } catch (e) {}
    }

    const toggleShadow = () => {
      shadowEnabled.value = !shadowEnabled.value
      applyShadowOptions()
    }

    const exportPng = () => {
      if (!network) return
      const canvas = network.canvas.frame.canvas
      const link = document.createElement('a')
      link.download = 'knowledge-graph.png'
      link.href = canvas.toDataURL('image/png')
      link.click()
    }

    const toggleTheme = () => {
      themeMode.value = themeMode.value === 'neon' ? 'light' : 'neon'
      try { localStorage.setItem('graph_theme', themeMode.value) } catch (e) {}
      particleStyles.value = createParticleStyles()
      applyThemeOptions()
      applyShadowOptions()
      updateNetwork()
    }

    const createParticleStyles = () => {
      const arr = []
      const palette = themeMode.value === 'neon'
        ? ['rgba(0,255,255,0.18)','rgba(255,80,120,0.14)','rgba(160,120,255,0.14)']
        : ['rgba(0,140,255,0.08)','rgba(255,255,255,0.10)']
      for (let i = 0; i < 12; i++) {
        const top = Math.random() * 80 + 5
        const left = Math.random() * 80 + 10
        const size = Math.random() * 200 + 120
        const color = palette[i % palette.length]
        arr.push({
          position: 'absolute',
          top: top + '%',
          left: left + '%',
          width: size + 'px',
          height: size + 'px',
          borderRadius: '50%',
          background: `radial-gradient(circle at 50% 50%, ${color} 0%, transparent 70%)`,
          filter: 'blur(18px)',
          transition: 'opacity 0.8s ease'
        })
      }
      return arr
    }

    onMounted(() => {
      nextTick(() => {
        initNetwork()
        particleStyles.value = createParticleStyles()

        // 载入图例位置（若有持久化）并确保不越界
        loadIntroPos()
        introCardPos.value = clampIntroPos(introCardPos.value.top, introCardPos.value.left)

        // 监听容器尺寸变化，确保画布自适应填充
        if (window.ResizeObserver && networkContainer.value) {
          resizeObserver = new ResizeObserver(() => {
            if (network) {
              try {
                const h = networkContainer.value.clientHeight || networkContainer.value.offsetHeight || 500
                network.setSize('100%', h + 'px')
                network.redraw()
                network.fit()
              } catch (e) {}
            } else {
              // 如果还未初始化，尝试初始化
              initNetwork()
            }
          })
          resizeObserver.observe(networkContainer.value)
        }

        // 监听图容器尺寸变化，避免图例拖拽后在缩放/窗口变化时越界
        if (window.ResizeObserver && graphContainer.value) {
          legendResizeObserver = new ResizeObserver(() => {
            introCardPos.value = clampIntroPos(introCardPos.value.top, introCardPos.value.left)
          })
          legendResizeObserver.observe(graphContainer.value)
        }
      })
    })

    // 监听节点/边数据变化，动态更新图谱
    watch(() => props.nodes, () => updateNetwork(), { deep: false })
    watch(() => props.edges, () => updateNetwork(), { deep: false })
    watch(() => loosenessPct.value, () => applyLooseness())
    watch(() => labelFontSize.value, () => applyLabelFontSize())
    watch(() => hierarchicalEnabled.value, () => applyLayoutOptions())
    watch(() => shadowEnabled.value, () => applyShadowOptions())
    watch(() => arrowScale.value, () => applyArrowOptions())
    watch(() => edgeLabelMode.value, () => updateNetwork())
    watch(() => themeMode.value, () => { try { localStorage.setItem('graph_theme', themeMode.value) } catch (e) {} ; applyThemeOptions() })
    watch(() => glowStrength.value, () => { applyShadowOptions(); updateNetwork() })
    watch(() => edgeCurvature.value, () => applyLayoutOptions())

    onBeforeUnmount(() => {
      if (network) {
        network.destroy()
        network = null
      }
      if (resizeObserver) {
        try { resizeObserver.disconnect() } catch (e) {}
        resizeObserver = null
      }
      if (legendResizeObserver) {
        try { legendResizeObserver.disconnect() } catch (e) {}
        legendResizeObserver = null
      }
    })

    return {
      networkContainer,
      graphContainer,
      introCardRef,
      introCardPos,
      introDragging,
      startDragIntro,
      physicsEnabled,
      loosenessPct,
      labelFontSize,
      arrowScale,
      edgeLabelMode,
      hierarchicalEnabled,
      shadowEnabled,
      themeMode,
      toggleTheme,
      glowStrength,
      edgeCurvature,
      particleStyles,
      diseaseCount,
      drugCount,
      examCount,
      triangulatedCount,
      candidateCount,
      totalNodes,
      totalEdges,
      weightDataCount,
      resetGraph,
      fitGraph,
      togglePhysics,
      toggleLayout,
      toggleShadow,
      toggleEdgeLabelMode,
      exportPng
    }
  }
}
</script>

<style scoped>
.graph-panel {
  padding: 20px;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.graph-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  position: relative;
}

.controls {
  display: flex;
  gap: 10px;
  margin-bottom: 15px;
  padding: 10px 12px;
  background: linear-gradient(90deg, #f8f9fa 0%, #f3f6ff 100%);
  border-radius: 10px;
  position: relative;
  z-index: 10;
  box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}

.control-btn {
  padding: 8px 16px;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.9em;
  transition: background-color 0.2s;
}

.control-btn:hover {
  background: #0056b3;
}

.control-range-wrap {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-left: auto;
}

.control-label {
  font-size: 0.9em;
  color: #333;
}

.control-range {
  width: 180px;
}

.range-value {
  color: #007bff;
}

.network-container {
  flex: 1;
  min-height: 500px;
  border: 1px solid #e9ecef;
  border-radius: 12px;
  background: transparent;
  position: relative;
  z-index: 1;
  box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

.direction-hint {
  margin-left: auto;
  margin-right: 8px;
  align-self: center;
  color: #6c757d;
  font-size: 0.9em;
}

/* 介绍卡片样式 */
.intro-card {
  position: absolute;
  top: 168px;
  left: 12px;
  width: 210px;
  background: rgba(255,255,255,0.92);
  border: 1px solid #e9ecef;
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.08);
  padding: 10px 12px;
  z-index: 5;
  cursor: grab;
  user-select: none;
  touch-action: none;
}
.intro-card.dragging { cursor: grabbing; }
.graph-panel.neon-theme .intro-card {
  background: rgba(24,28,40,0.55);
  color: #eaf4ff;
  border: 1px solid rgba(140,180,255,0.28);
  box-shadow: 0 8px 26px rgba(0,220,255,0.12);
}
.intro-title { font-weight: 700; font-size: 14px; margin-bottom: 8px; }
.intro-section { font-weight: 600; font-size: 12px; margin-top: 4px; margin-bottom: 4px; }
.intro-item { display: flex; align-items: center; gap: 8px; font-size: 12px; color: #555; }
.graph-panel.neon-theme .intro-item { color: #eaf4ff; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.dot-disease { background: #e74c3c; }
.dot-drug { background: #3498db; }
.dot-exam { background: #2ecc71; }
.line { display: inline-block; width: 26px; height: 0; border-top: 2px solid #1f2937; }
.line-dashed { border-top-style: dashed; opacity: 0.7; }
.line-solid { border-top-style: solid; }
.intro-sep { margin: 6px 0; height: 1px; background: #e9ecef; }
.graph-panel.neon-theme .intro-sep { background: rgba(180,200,255,0.24); }
.intro-meta { font-size: 12px; color: #666; }
.graph-panel.neon-theme .intro-meta { color: #eaf4ff; }

/* 霓虹主题下控件文本对比度提升 */
.graph-panel.neon-theme .control-label,
.graph-panel.neon-theme .direction-hint { color: #eaf4ff; }

/* 背景层：极光与粒子光斑 */
.bg-layer {
  position: absolute;
  inset: 0;
  z-index: 0;
  pointer-events: none;
}

.aurora {
  position: absolute;
  inset: -10% -20% -10% -20%;
  background: radial-gradient(1200px 500px at 10% 20%, rgba(0, 196, 255, 0.15), transparent 60%),
              radial-gradient(900px 400px at 90% 30%, rgba(255, 120, 160, 0.12), transparent 65%),
              radial-gradient(1000px 600px at 50% 80%, rgba(120, 80, 255, 0.10), transparent 70%);
  filter: blur(30px) saturate(110%);
  animation: auroraFlow 16s ease-in-out infinite alternate;
}

@keyframes auroraFlow {
  0% { transform: translate3d(0,0,0) scale(1); opacity: 0.85; }
  50% { transform: translate3d(2%, -2%, 0) scale(1.03); opacity: 1; }
  100% { transform: translate3d(-2%, 2%, 0) scale(1.02); opacity: 0.9; }
}

.grid-overlay {
  position: absolute;
  inset: 0;
  background-image: 
    linear-gradient(rgba(120,120,120,0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(120,120,120,0.05) 1px, transparent 1px);
  background-size: 24px 24px, 24px 24px;
  mix-blend-mode: soft-light;
}

.particles {
  position: absolute;
  inset: 0;
}

.particle {
  position: absolute;
  border-radius: 50%;
  will-change: transform, opacity;
  animation: particleFloat 18s ease-in-out infinite;
}

@keyframes particleFloat {
  0% { transform: translate3d(0, 0, 0); opacity: 0.8; }
  50% { transform: translate3d(10px, -8px, 0); opacity: 1; }
  100% { transform: translate3d(-8px, 10px, 0); opacity: 0.85; }
}

/* 主题样式增强 */
.graph-panel.light-theme .controls {
  background: linear-gradient(90deg, #f8fbff 0%, #eef5ff 100%);
  box-shadow: 0 6px 20px rgba(90,120,180,0.12);
}
.graph-panel.neon-theme .controls {
  background: rgba(22, 26, 40, 0.45);
  backdrop-filter: blur(10px);
  box-shadow: 0 10px 28px rgba(0, 255, 255, 0.12);
}

.graph-panel.neon-theme .control-btn {
  background: linear-gradient(135deg, #00d1ff 0%, #7b5fff 100%);
}
.graph-panel.neon-theme .control-btn:hover {
  background: linear-gradient(135deg, #00a8ff 0%, #6749ff 100%);
  box-shadow: 0 0 12px rgba(0, 200, 255, 0.6);
}

.graph-panel.neon-theme .range-value {
  color: #00d1ff;
}
</style>
    // 统计信息用于介绍卡片
    const diseaseCount = computed(() => (props.nodes || []).filter(n => String(n).startsWith('疾病_')).length)
    const drugCount = computed(() => (props.nodes || []).filter(n => String(n).startsWith('药物_')).length)
    const examCount = computed(() => (props.nodes || []).filter(n => String(n).startsWith('检验_')).length)
    const triangulatedCount = computed(() => (props.edges || []).filter(e => e?.edge_hierarchy === 'triangulated_verified').length)
    const candidateCount = computed(() => (props.edges || []).filter(e => e?.edge_hierarchy === 'candidate_only').length)
    const totalNodes = computed(() => (props.nodes || []).length)
    const totalEdges = computed(() => (props.edges || []).length)
    const weightDataCount = computed(() => (props.edges || []).filter(e => {
      const v = typeof e?.weight_ref === 'number' ? e.weight_ref : Number(e?.weight_ref)
      return Number.isFinite(v)
    }).length)