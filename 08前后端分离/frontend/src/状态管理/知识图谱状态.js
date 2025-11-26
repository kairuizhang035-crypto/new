import { defineStore } from 'pinia'
import axios from 'axios'

const API_BASE_URL = '/api'

export const useKnowledgeGraphStore = defineStore('knowledgeGraph', {
  state: () => ({
    nodes: [],
    edges: [],
    relations: {},
    weights: {},
    parameters: {},
    pathways: {},
    // 新增：节点类型详细统计（含每个节点的度数）
    nodeTypeStats: null,
    // 新增：关系类型与层次统计的缓存（便于主页面展示）
    relationTypeStats: {},
    // 新增：关系类型详细统计（后端计算）
    relationTypeStatsDetailed: {},
    hierarchyStats: {},
    statistics: {
      nodes: 0,
      edges: 0,
      relations: 0,
      weights: 0,
      parameters: 0,
      pathways: 0
    },
    loading: false,
    error: null
  }),

  getters: {
    getNodesByType: (state) => (type) => {
      return state.nodes.filter(node => {
        if (type === 'disease') return node.startsWith('疾病_')
        if (type === 'drug') return node.startsWith('药物_')
        if (type === 'test') return node.startsWith('检验_')
        return true
      })
    },

    getEdgesByRelationType: (state) => (relationType) => {
      return state.edges.filter(edge => edge.relation_type === relationType)
    },

    getEdgesByHierarchy: (state) => (hierarchy) => {
      return state.edges.filter(edge => edge.edge_hierarchy === hierarchy)
    }
  },

  actions: {
    async listDatasources() {
      try {
        const res = await axios.get(`${API_BASE_URL}/datasource/list`)
        return res.data?.data || []
      } catch (error) {
        console.error('获取数据源列表失败:', error)
        throw error
      }
    },

    async getCurrentDatasource() {
      try {
        const res = await axios.get(`${API_BASE_URL}/datasource/current`)
        return res.data?.data || {}
      } catch (error) {
        console.error('获取当前数据源失败:', error)
        throw error
      }
    },

    async selectDatasource(path) {
      try {
        const res = await axios.post(`${API_BASE_URL}/datasource/select`, { path })
        // 选择成功后重新加载数据
        await this.loadData()
        return res.data?.data || {}
      } catch (error) {
        console.error('选择数据源失败:', error)
        throw error
      }
    },

    async uploadDatasource(file, autoSelect = true) {
      try {
        const form = new FormData()
        form.append('file', file)
        form.append('select', autoSelect ? 'true' : 'false')
        const res = await axios.post(`${API_BASE_URL}/datasource/upload`, form, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })
        // 如果后端已自动选择，刷新数据
        await this.loadData()
        return res.data?.data || {}
      } catch (error) {
        console.error('上传数据源失败:', error)
        throw error
      }
    },
    async loadData() {
      this.loading = true
      this.error = null
      
      try {
        // 并行加载所有数据
        const [
          nodesResponse,
          edgesResponse,
          relationsResponse,
          weightsResponse,
          parametersResponse,
          pathwaysResponse,
          statisticsResponse,
          nodeTypesResponse,
          relationStatsResponse
        ] = await Promise.all([
          axios.get(`${API_BASE_URL}/nodes`),
          axios.get(`${API_BASE_URL}/edges`),
          axios.get(`${API_BASE_URL}/relations`),
          axios.get(`${API_BASE_URL}/weights`),
          axios.get(`${API_BASE_URL}/parameters`),
          axios.get(`${API_BASE_URL}/pathways`),
          axios.get(`${API_BASE_URL}/statistics`),
          axios.get(`${API_BASE_URL}/nodes/types`),
          axios.get(`${API_BASE_URL}/relations/stats`)
        ])

        // 更新状态（后端统一返回 { success, data, ... }，需取 data）
        this.nodes = nodesResponse.data?.data || []
        this.edges = edgesResponse.data?.data || []
        this.relations = relationsResponse.data?.data || {}
        this.weights = weightsResponse.data?.data || {}
        this.parameters = parametersResponse.data?.data || {}
        this.pathways = pathwaysResponse.data?.data || {}
        this.statistics = statisticsResponse.data?.data || {
          nodes: 0, edges: 0, relations: 0, weights: 0, parameters: 0, pathways: 0
        }
        this.nodeTypeStats = nodeTypesResponse.data?.data || null
        this.relationTypeStatsDetailed = relationStatsResponse.data?.data || {}

        // 计算并缓存关系类型与层次统计
        this.relationTypeStats = this.getRelationTypeStats()
        this.hierarchyStats = this.getHierarchyStats()

        console.log('数据加载完成:', {
          nodes: this.nodes.length,
          edges: this.edges.length,
          relations: Object.keys(this.relations).length,
          weights: Object.keys(this.weights).length,
          parameters: Object.keys(this.parameters).length,
          pathways: Object.keys(this.pathways).length,
          nodeTypeGroups: this.nodeTypeStats ? Object.keys(this.nodeTypeStats).length : 0,
          relationTypes: Object.keys(this.relations || {}).length
        })

      } catch (error) {
        console.error('加载数据失败:', error)
        this.error = error.message || '加载数据失败'
        throw error
      } finally {
        this.loading = false
      }
    },

    // 仅刷新参数数据，并更新统计
    async reloadParameters() {
      this.loading = true
      this.error = null
      try {
        const res = await axios.get(`${API_BASE_URL}/parameters`)
        const data = res.data?.data ?? res.data ?? {}
        this.parameters = data
        const count = Array.isArray(data) ? data.length : Object.keys(data).length
        this.statistics = { ...this.statistics, parameters: count }
        return { success: true, count }
      } catch (error) {
        console.error('刷新参数数据失败:', error)
        this.error = error.message || '刷新参数数据失败'
        throw error
      } finally {
        this.loading = false
      }
    },

    // 参数详情：调用后端统一详情接口
    async getParameterDetails(key) {
      try {
        const response = await axios.get(`${API_BASE_URL}/parameters/${encodeURIComponent(key)}/details`)
        return response.data
      } catch (error) {
        console.error('获取参数详情失败:', error)
        throw error
      }
    },

    // 参数搜索：支持后端按关键字搜索Θ集合
    async searchParameters(query) {
      try {
        const response = await axios.get(`${API_BASE_URL}/search/parameters`, { params: { q: query } })
        return response.data
      } catch (error) {
        console.error('搜索参数失败:', error)
        throw error
      }
    },

    // 仅刷新边数据，并更新相关统计缓存
    async reloadEdges() {
      try {
        const edgesResponse = await axios.get(`${API_BASE_URL}/edges`)
        this.edges = edgesResponse.data?.data || []
        // 重新计算并缓存关系类型与层次统计
        this.relationTypeStats = this.getRelationTypeStats()
        this.hierarchyStats = this.getHierarchyStats()
        return { success: true, count: this.edges.length }
      } catch (error) {
        console.error('刷新边数据失败:', error)
        this.error = error.message || '刷新边数据失败'
        throw error
      }
    },

    // 仅刷新权重数据，并同步统计
    async reloadWeights() {
      this.loading = true
      this.error = null
      try {
        const res = await axios.get(`${API_BASE_URL}/weights`)
        const data = res.data?.data ?? res.data ?? {}
        this.weights = data
        const count = Array.isArray(data) ? data.length : Object.keys(data).length
        this.statistics = { ...this.statistics, weights: count }
        return { success: true, count }
      } catch (error) {
        console.error('刷新权重数据失败:', error)
        this.error = error.message || '刷新权重数据失败'
        throw error
      } finally {
        this.loading = false
      }
    },

    // 仅刷新路径数据，并更新统计
    async reloadPathways() {
      this.loading = true
      this.error = null
      try {
        const res = await axios.get(`${API_BASE_URL}/pathways`)
        const data = res.data?.data ?? res.data ?? {}
        this.pathways = data
        const count = Array.isArray(data) ? data.length : Object.keys(data).length
        this.statistics = { ...this.statistics, pathways: count }
        return { success: true, count }
      } catch (error) {
        console.error('刷新路径数据失败:', error)
        this.error = error.message || '刷新路径数据失败'
        throw error
      } finally {
        this.loading = false
      }
    },

    // 获取边详情（前后端分离：调用后端详情接口）
    async getEdgeDetails(source, target) {
      try {
        const response = await axios.get(`${API_BASE_URL}/edges/${encodeURIComponent(source)}/${encodeURIComponent(target)}/details`)
        return response.data
      } catch (error) {
        console.error('获取边详情失败:', error)
        throw error
      }
    },

    // 权重详情：调用后端统一详情接口
    async getWeightDetails(key) {
      try {
        const response = await axios.get(`${API_BASE_URL}/weights/${encodeURIComponent(key)}/details`)
        return response.data
      } catch (error) {
        console.error('获取权重详情失败:', error)
        throw error
      }
    },

    // 获取关系类型详细统计（单独调用）
    async fetchRelationTypeStats() {
      try {
        const response = await axios.get(`${API_BASE_URL}/relations/stats`)
        this.relationTypeStatsDetailed = response.data?.data || {}
        return { success: true, count: Object.keys(this.relationTypeStatsDetailed).length }
      } catch (error) {
        console.error('获取关系类型统计失败:', error)
        this.error = error.message || '获取关系类型统计失败'
        throw error
      }
    },

    async searchNodes(query) {
      try {
        const response = await axios.get(`${API_BASE_URL}/search/nodes`, {
          params: { q: query }
        })
        return response.data
      } catch (error) {
        console.error('搜索节点失败:', error)
        throw error
      }
    },

    async getNodeDetails(nodeId) {
      try {
        const response = await axios.get(`${API_BASE_URL}/nodes/${nodeId}/details`)
        return response.data
      } catch (error) {
        console.error('获取节点详情失败:', error)
        throw error
      }
    },

    async getGraphData() {
      try {
        const response = await axios.get(`${API_BASE_URL}/graph/data`)
        return response.data
      } catch (error) {
        console.error('获取图谱数据失败:', error)
        throw error
      }
    },

    // 路径详情：调用后端统一详情接口
    async getPathwayDetails(key) {
      try {
        const response = await axios.get(`${API_BASE_URL}/pathways/${encodeURIComponent(key)}/details`)
        return response.data
      } catch (error) {
        console.error('获取路径详情失败:', error)
        throw error
      }
    },

    // 路径搜索：支持后端按关键字搜索Phi集合
    async searchPathways(query) {
      try {
        const response = await axios.get(`${API_BASE_URL}/search/pathways`, {
          params: { q: query }
        })
        return response.data
      } catch (error) {
        console.error('搜索路径失败:', error)
        throw error
      }
    },

    // 仅刷新路径数据，并同步统计
    async reloadPathways() {
      this.loading = true
      this.error = null
      try {
        const response = await axios.get(`${API_BASE_URL}/pathways`)
        const data = response.data?.data ?? response.data ?? {}
        this.pathways = data
        const count = Array.isArray(data) ? data.length : Object.keys(data).length
        this.statistics = { ...this.statistics, pathways: count }
        return { success: true, count }
      } catch (error) {
        console.error('刷新路径数据失败:', error)
        this.error = error.message || '刷新路径数据失败'
        throw error
      } finally {
        this.loading = false
      }
    },

    // 筛选方法
    filterEdgesByRelationTypes(relationTypes) {
      if (!relationTypes || relationTypes.length === 0) {
        return this.edges
      }
      return this.edges.filter(edge => relationTypes.includes(edge.relation_type))
    },

    filterEdgesByHierarchies(hierarchies) {
      if (!hierarchies || hierarchies.length === 0) {
        return this.edges
      }
      return this.edges.filter(edge => hierarchies.includes(edge.edge_hierarchy))
    },

    // 统计方法
    getRelationTypeStats() {
      const stats = {}
      this.edges.forEach(edge => {
        stats[edge.relation_type] = (stats[edge.relation_type] || 0) + 1
      })
      return stats
    },

    getHierarchyStats() {
      const stats = {}
      this.edges.forEach(edge => {
        stats[edge.edge_hierarchy] = (stats[edge.edge_hierarchy] || 0) + 1
      })
      return stats
    },

    getNodeTypeStats() {
      const stats = {
        disease: 0,
        drug: 0,
        test: 0,
        other: 0
      }
      
      this.nodes.forEach(node => {
        if (node.startsWith('疾病_')) {
          stats.disease++
        } else if (node.startsWith('药物_')) {
          stats.drug++
        } else if (node.startsWith('检验_')) {
          stats.test++
        } else {
          stats.other++
        }
      })
      
      return stats
    },

    // 质量分析方法
    getWeightQualityDistribution() {
      const distribution = {
        platinum: 0,
        gold: 0,
        silver: 0,
        bronze: 0
      }
      
      Object.values(this.weights).forEach(weight => {
        const quality = weight.base_weight?.quality_level?.toLowerCase()
        if (quality && distribution.hasOwnProperty(quality)) {
          distribution[quality]++
        }
      })
      
      return distribution
    },

    // 获取高质量边
    getHighQualityEdges(minScore = 0.8) {
      return this.edges.filter(edge => {
        if (!edge.weight_ref) return false
        const weight = this.weights[edge.weight_ref]
        return weight?.base_weight?.integrated_score >= minScore
      })
    },

    // 获取支持算法最多的边
    getMostSupportedEdges(minAlgorithms = 3) {
      return this.edges.filter(edge => {
        if (!edge.weight_ref) return false
        const weight = this.weights[edge.weight_ref]
        return weight?.base_weight?.support_algorithm_count >= minAlgorithms
      })
    }
  }
})
