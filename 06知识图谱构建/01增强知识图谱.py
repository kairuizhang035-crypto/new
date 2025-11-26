#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强知识图谱构建脚本 - 基于63条精简边
实现完整的 KG = (V, E_core, R, W, Θ, Φ) 结构
体现250条候选边、63条精简边、23条三角测量边的层次关系

V: 变量节点集合
E_core: 核心因果边集合（63条精简边）
R: 关系类型集合
W: 边级权重属性（完整的Conf(e)分解）
Θ: 参数属性（CPT、边条件概率、各方法估计结果）
Φ: 效应属性（中介分析的IE/TE/MR、HDI、显著性概率）

作者: AI助手
创建时间: 2025-10-16
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import warnings
from collections import defaultdict, Counter
import matplotlib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import pickle
import glob
import http.server
import socketserver
import threading
import webbrowser
import time

# 设置中文字体
plt.rcParams['font.family'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

def safe_float_conversion(value, default=0.0):
    """
    安全地将值转换为浮点数，处理NaN和null值
    
    Args:
        value: 要转换的值
        default: 当值为NaN或null时的默认值
    
    Returns:
        float: 转换后的浮点数或默认值
    """
    if value is None:
        return default
    
    # 处理字符串形式的NaN
    if isinstance(value, str) and value.lower() in ['nan', 'null', 'none']:
        return default
    
    try:
        # 尝试转换为浮点数
        float_val = float(value)
        # 检查是否为NaN
        if np.isnan(float_val) or np.isinf(float_val):
            return default
        return float_val
    except (ValueError, TypeError):
        return default

class EnhancedKnowledgeGraph:
    """增强知识图谱构建器 - 基于63条精简边"""
    
    def __init__(self, base_dir=None):
        """初始化知识图谱构建器"""
        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.base_dir = base_dir
        
        # 设置输出目录
        self.output_dir = os.path.join(self.base_dir, "01增强知识图谱")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化知识图谱组件 KG = (V, E_core, R, W, Θ, Φ)
        self.V = set()  # 变量节点集合
        self.E_core = []  # 核心因果边集合（63条精简边）
        self.R = {}  # 关系类型集合
        self.W = {}  # 边级权重属性
        self.Theta = {}  # 参数属性
        self.Phi = {}  # 效应属性
        
        # 数据存储
        self.candidate_edges_df = None  # 250条候选边
        self.refined_edges_df = None    # 63条精简边
        self.triangulation_edges_df = None  # 23条三角测量边
        self.parameter_data = {}
        self.mediation_data = None
        self.cpt_data = {}
        
        # 图结构
        self.knowledge_graph = nx.DiGraph()
        self.graph_statistics = {}
        
    def load_all_data(self):
        """加载所有相关数据"""
        print("=== 加载所有数据源 ===")
        
        success_count = 0
        total_count = 6
        
        # 1. 加载250条候选边数据
        if self._load_candidate_edges():
            success_count += 1
        
        # 2. 加载63条精简边数据
        if self._load_refined_edges():
            success_count += 1
        
        # 3. 加载23条三角测量边数据
        if self._load_triangulation_edges():
            success_count += 1
        
        # 4. 加载参数学习结果
        if self._load_parameter_data():
            success_count += 1
        
        # 5. 加载中介分析结果
        if self._load_mediation_data():
            success_count += 1
        
        # 6. 加载CPT数据
        if self._load_cpt_data():
            success_count += 1
        
        print(f"\n数据加载完成: {success_count}/{total_count} 个数据源加载成功")
        
        # 只要有精简边数据就可以继续
        if hasattr(self, 'refined_edges_df') and self.refined_edges_df is not None and not self.refined_edges_df.empty:
            print("✓ 核心数据已加载，可以继续构建知识图谱")
            return True
        else:
            print("✗ 精简边数据加载失败，无法构建知识图谱")
            return False
    
    def _load_candidate_edges(self):
        """加载250条候选边数据"""
        try:
            candidate_file = os.path.join(os.path.dirname(self.base_dir), 
                                        "02因果发现/06候选因果边集合/因果边综合评分结果.csv")
            self.candidate_edges_df = pd.read_csv(candidate_file, encoding='utf-8')
            print(f"✓ 成功加载候选边数据: {len(self.candidate_edges_df)} 条边")
            return True
        except Exception as e:
            print(f"✗ 加载候选边数据失败: {e}")
            return False
    
    def _load_refined_edges(self):
        """加载63条精简边数据（主体）"""
        try:
            refined_file = os.path.join(os.path.dirname(self.base_dir), 
                                      "02因果发现/06候选因果边集合/精简因果边列表.csv")
            self.refined_edges_df = pd.read_csv(refined_file, encoding='utf-8')
            print(f"✓ 成功加载精简边数据（主体）: {len(self.refined_edges_df)} 条边")
            return True
        except Exception as e:
            print(f"✗ 加载精简边数据失败: {e}")
            return False
    
    def _load_triangulation_edges(self):
        """加载23条三角测量边数据"""
        try:
            triangulation_file = os.path.join(os.path.dirname(self.base_dir), 
                                            "05三角测量/三角验证结果/核心因果边集合.csv")
            self.triangulation_edges_df = pd.read_csv(triangulation_file, encoding='utf-8')
            print(f"✓ 成功加载三角测量边数据: {len(self.triangulation_edges_df)} 条边")
            return True
        except Exception as e:
            print(f"✗ 加载三角测量边数据失败: {e}")
            return False
    
    def _load_parameter_data(self):
        """加载参数学习结果"""
        try:
            # 初始化参数数据字典
            self.parameter_data = {}
            
            # 加载SEM结构方程结果
            sem_file = os.path.join(os.path.dirname(self.base_dir), 
                                  "03多方法参数学习/04SEM_结果/SEM_结构方程.json")
            if os.path.exists(sem_file):
                with open(sem_file, 'r', encoding='utf-8') as f:
                    self.sem_data = json.load(f)
                print(f"✓ 成功加载SEM结构方程结果: {len(self.sem_data)} 条记录")
            else:
                print("⚠ SEM结构方程结果文件不存在")
                self.sem_data = {}
                
            # 加载边级似然增益结果
            likelihood_file = os.path.join(os.path.dirname(self.base_dir), 
                                         "03多方法参数学习/05边级似然增益结果/边级似然增益结果.json")
            if os.path.exists(likelihood_file):
                with open(likelihood_file, 'r', encoding='utf-8') as f:
                    self.likelihood_data = json.load(f)
                print(f"✓ 成功加载边级似然增益结果: {len(self.likelihood_data)} 个方法")
            else:
                print("⚠ 边级似然增益结果文件不存在")
                self.likelihood_data = {}
                
            # 加载参数稳定性结果
            stability_file = os.path.join(os.path.dirname(self.base_dir), 
                                        "03多方法参数学习/06参数稳定性结果/参数稳定性详细结果.json")
            if os.path.exists(stability_file):
                with open(stability_file, 'r', encoding='utf-8') as f:
                    self.stability_data = json.load(f)
                print(f"✓ 成功加载参数稳定性结果")
            else:
                print("⚠ 参数稳定性结果文件不存在")
                self.stability_data = {}
                
            return True
        except Exception as e:
            print(f"⚠ 加载参数学习结果失败: {e}，但继续执行")
            self.parameter_data = {}
            self.sem_data = {}
            self.likelihood_data = {}
            self.stability_data = {}
            return True
    
    def _load_mediation_data(self):
        """加载贝叶斯中介分析结果（详细版本）"""
        try:
            # 优先加载详细结果文件
            detailed_mediation_file = os.path.join(os.path.dirname(self.base_dir), 
                                                 "04贝叶斯中介分析/02贝叶斯中介分析结果/贝叶斯中介分析详细结果.csv")
            
            if os.path.exists(detailed_mediation_file):
                self.mediation_data_df = pd.read_csv(detailed_mediation_file, encoding='utf-8')
                print(f"✓ 成功加载详细中介分析结果: {len(self.mediation_data_df)} 条路径")
                print(f"  - 包含详细字段: HDI置信区间、标准差、效应方向、中介类型等")
                return True
            else:
                # 如果详细文件不存在，回退到汇总文件
                summary_mediation_file = os.path.join(os.path.dirname(self.base_dir), 
                                                    "04贝叶斯中介分析/02贝叶斯中介分析结果/贝叶斯中介分析汇总.csv")
                if os.path.exists(summary_mediation_file):
                    self.mediation_data_df = pd.read_csv(summary_mediation_file, encoding='utf-8')
                    print(f"✓ 成功加载汇总中介分析结果: {len(self.mediation_data_df)} 条路径")
                    print("  - 注意: 使用的是汇总版本，字段相对简化")
                    return True
                else:
                    print("⚠ 中介分析结果文件不存在")
                    self.mediation_data_df = None
                    return True
        except Exception as e:
            print(f"⚠ 加载中介分析结果失败: {e}，但继续执行")
            self.mediation_data_df = None
            return True
    
    def _load_cpt_data(self):
        """加载条件概率表数据"""
        try:
            # 初始化CPT数据字典
            self.cpt_data = {}
            
            cpt_methods = [
                ('MLE', '01MLE_CPT结果'),
                ('Bayesian', '02Bayesian_CPT结果'),
                ('EM', '03EM_CPT结果')
            ]
            cpt_dir = os.path.join(os.path.dirname(self.base_dir), "03多方法参数学习")
            
            for method, folder_name in cpt_methods:
                try:
                    cpt_file = os.path.join(cpt_dir, f"{folder_name}/{method}_CPTs.json")
                    if os.path.exists(cpt_file):
                        with open(cpt_file, 'r', encoding='utf-8') as f:
                            self.cpt_data[method] = json.load(f)
                        print(f"✓ 成功加载 {method} CPT数据: {len(self.cpt_data[method])} 个节点")
                    else:
                        print(f"⚠ CPT文件不存在: {cpt_file}")
                        self.cpt_data[method] = {}
                except Exception as e:
                    print(f"⚠ 加载 {method} CPT数据失败: {e}")
                    self.cpt_data[method] = {}
            
            return True
        except Exception as e:
            print(f"⚠ 加载CPT数据失败: {e}，但继续执行")
            self.cpt_data = {'MLE': {}, 'Bayesian': {}, 'EM': {}}
            return True
    
    def build_enhanced_knowledge_graph(self):
        """构建增强知识图谱 KG = (V, E_core, R, W, Θ, Φ)"""
        print("\n=== 构建增强知识图谱 ===")
        
        # 1. 构建节点集合 V
        self._build_node_set()
        
        # 2. 预构建权重属性 W（为关系类型判断提供依据）
        self._build_weight_attributes()
        
        # 3. 构建核心边集合 E_core（基于63条精简边，现在可以正确判断关系类型）
        self._build_core_edge_set()
        
        # 4. 定义关系类型集合 R
        self._define_relation_types()
        
        # 5. 构建参数属性 Θ
        self._build_parameter_attributes()
        
        # 6. 构建效应属性 Φ
        self._build_effect_attributes()
        
        # 7. 构建NetworkX图结构
        self._build_networkx_graph()
        
        # 8. 计算图统计信息
        self._calculate_graph_statistics()
        
        print("✓ 增强知识图谱构建完成")
    
    def _build_node_set(self):
        """构建节点集合 V"""
        print("构建节点集合 V...")
        
        # 从63条精简边中提取所有节点
        for _, row in self.refined_edges_df.iterrows():
            self.V.add(row['源节点'])
            self.V.add(row['目标节点'])
        
        print(f"✓ 节点集合 V: {len(self.V)} 个节点")
    
    def _build_core_edge_set(self):
        """构建核心边集合 E_core（基于63条精简边）- 优化版本，去除与W重复的字段"""
        print("构建核心边集合 E_core...")
        
        for _, row in self.refined_edges_df.iterrows():
            # 检查是否为三角测量验证边
            is_triangulated = False
            
            if self.triangulation_edges_df is not None:
                triangulated_edge = self.triangulation_edges_df[
                    (self.triangulation_edges_df['源节点'] == row['源节点']) & 
                    (self.triangulation_edges_df['目标节点'] == row['目标节点'])
                ]
                if not triangulated_edge.empty:
                    is_triangulated = True
            
            # 获取权重引用键
            edge_key = f"{row['源节点']}->{row['目标节点']}"
            
            # 判断因果关系的直接性（使用临时数据进行判断）
            temp_edge_data = {
                'edge_hierarchy': self._determine_edge_hierarchy(row, is_triangulated),
                'weight_attributes': {
                    'base_weight': row.get('集成评分', 0),
                    'triangulation_weight': 0  # 临时值，实际从W中获取
                },
                'is_triangulated': is_triangulated,
                '集成评分': row.get('集成评分', 0),
                '支持算法数量': row.get('支持算法数量', 0)
            }
            is_direct = self._determine_causality_type(row['源节点'], row['目标节点'], temp_edge_data)
            
            # 构建优化后的边结构（去除重复字段）
            edge = {
                'source': row['源节点'],
                'target': row['目标节点'],
                'weight_ref': edge_key,  # 引用W中的完整权重信息
                'edge_hierarchy': temp_edge_data['edge_hierarchy'],
                'relation_type': self._determine_relation_type(row['源节点'], row['目标节点'], is_direct),
                'is_direct': is_direct,
                'identification_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self.E_core.append(edge)
        
        print(f"✓ 核心边集合 E_core: {len(self.E_core)} 条边（63条精简边，其中{sum(1 for e in self.E_core if self._is_edge_triangulated(e['weight_ref']))}条经过三角测量验证）")
        
        # 统计直接和间接关系
        direct_count = sum(1 for e in self.E_core if e.get('is_direct', True))
        indirect_count = len(self.E_core) - direct_count
        print(f"  - 直接关系: {direct_count} 条")
        print(f"  - 间接关系: {indirect_count} 条")
    
    def _is_edge_triangulated(self, weight_ref):
        """检查边是否经过三角测量验证"""
        weight_info = self.W.get(weight_ref, {})
        triangulation_weights = weight_info.get('triangulation_weights', {})
        return bool(triangulation_weights)
    
    def get_edge_quality_info(self, edge):
        """获取边的质量信息（从W中获取）"""
        weight_ref = edge.get('weight_ref')
        if not weight_ref or weight_ref not in self.W:
            return {}
        
        weight_info = self.W[weight_ref]
        base_weight = weight_info.get('base_weight', {})
        
        return {
            'quality_level': base_weight.get('quality_level'),
            'integrated_score': base_weight.get('integrated_score'),
            'support_algorithm_count': base_weight.get('support_algorithm_count')
        }
    
    def get_edge_triangulation_info(self, edge):
        """获取边的三角测量信息（从W中获取）"""
        weight_ref = edge.get('weight_ref')
        if not weight_ref or weight_ref not in self.W:
            return {}
        
        weight_info = self.W[weight_ref]
        return weight_info.get('triangulation_weights', {})
    
    def get_complete_edge_info(self, edge):
        """获取边的完整信息（结合E_core和W）"""
        edge_info = edge.copy()
        edge_info.update(self.get_edge_quality_info(edge))
        edge_info['triangulation_info'] = self.get_edge_triangulation_info(edge)
        edge_info['is_triangulated'] = self._is_edge_triangulated(edge.get('weight_ref'))
        return edge_info
    
    def _determine_edge_hierarchy(self, row, is_triangulated):
        """确定边的层次关系 - 简化为两种类型"""
        if is_triangulated:
            return "triangulated_verified"  # 经过三角测量验证的边
        else:
            return "non_triangulated"  # 未经过三角测量验证的边
    
    def _determine_relation_type(self, source, target, is_direct=True):
        """根据节点类型确定关系类型"""
        source_type = source.split('_')[0]
        target_type = target.split('_')[0]
        
        # 关系类型映射
        type_mapping = {
            ('疾病', '疾病'): 'disease_disease',
            ('疾病', '药物'): 'disease_drug',
            ('疾病', '检验'): 'disease_test',
            ('药物', '疾病'): 'drug_disease',
            ('药物', '药物'): 'drug_drug',
            ('药物', '检验'): 'drug_test',
            ('检验', '疾病'): 'test_disease',
            ('检验', '药物'): 'test_drug',
            ('检验', '检验'): 'test_test'
        }
        
        # 获取关系类型
        relation_type = type_mapping.get((source_type, target_type))
        
        if relation_type:
            return relation_type
        else:
            # 默认返回疾病-疾病关系
            return 'disease_disease'
    
    def _determine_causality_type(self, source, target, edge_data):
        """判断因果关系是直接还是间接 - 仅基于中介路径分析"""
        # 仅使用中介路径分析作为判断标准
        if hasattr(self, 'mediation_data_df') and self.mediation_data_df is not None:
            # 查找从source到target的中介路径
            mediation_paths = self.mediation_data_df[
                (self.mediation_data_df['路径描述'].str.contains(f'{source}.*→.*{target}', na=False)) &
                (self.mediation_data_df['显著性概率'] > 0.95)  # 显著中介路径
            ]
            
            if not mediation_paths.empty:
                # 存在显著中介路径，判定为间接关系
                return False  # 间接关系
            else:
                # 不存在显著中介路径，判定为直接关系
                return True  # 直接关系
        
        # 如果没有中介数据，默认为直接关系
        return True
    
    def _define_relation_types(self):
        """定义关系类型集合 R"""
        print("定义关系类型集合 R...")
        
        self.R = {
            # 疾病-疾病关系
            'disease_disease': {
                'name': '疾病-疾病关系',
                'description': '疾病间的因果关系或共病关系',
                'semantic': '疾病A影响疾病B'
            },
            
            # 疾病-药物关系
            'disease_drug': {
                'name': '疾病-药物关系',
                'description': '疾病对药物使用的需求关系',
                'semantic': '疾病需要药物治疗'
            },
            
            # 疾病-检验关系
            'disease_test': {
                'name': '疾病-检验关系',
                'description': '疾病对检验指标的影响',
                'semantic': '疾病影响检验指标'
            },
            
            # 药物-疾病关系
            'drug_disease': {
                'name': '药物-疾病关系',
                'description': '药物对疾病的治疗效果',
                'semantic': '药物治疗疾病'
            },
            
            # 药物-药物关系
            'drug_drug': {
                'name': '药物-药物关系',
                'description': '药物间的相互作用',
                'semantic': '药物A与药物B相互作用'
            },
            
            # 药物-检验关系
            'drug_test': {
                'name': '药物-检验关系',
                'description': '药物对检验指标的影响',
                'semantic': '药物影响检验指标'
            },
            
            # 检验-疾病关系
            'test_disease': {
                'name': '检验-疾病关系',
                'description': '检验指标异常指示疾病',
                'semantic': '检验指标反映疾病状态'
            },
            
            # 检验-药物关系
            'test_drug': {
                'name': '检验-药物关系',
                'description': '检验结果指导药物使用',
                'semantic': '检验结果指导药物选择'
            },
            
            # 检验-检验关系
            'test_test': {
                'name': '检验-检验关系',
                'description': '检验指标间的生理关联',
                'semantic': '检验指标A影响检验指标B'
            }
        }
        
        print(f"✓ 关系类型集合 R: {len(self.R)} 种关系类型")
    
    def _build_weight_attributes(self):
        """构建权重属性 W - 基于63条精简边"""
        print("构建权重属性 W...")
        
        for _, row in self.refined_edges_df.iterrows():
            edge_key = f"{row['源节点']}->{row['目标节点']}"
            
            # 获取候选边的详细信息
            candidate_info = {}
            if self.candidate_edges_df is not None:
                candidate_edge = self.candidate_edges_df[
                    (self.candidate_edges_df['源节点'] == row['源节点']) & 
                    (self.candidate_edges_df['目标节点'] == row['目标节点'])
                ]
                if not candidate_edge.empty:
                    cand_row = candidate_edge.iloc[0]
                    candidate_info = {
                        'frequency_score': cand_row.get('频次评分', 0.0),
                        'diversity_score': cand_row.get('多样性评分', 0.0),
                        'comprehensive_score': cand_row.get('综合评分', 0.0),
                        'algorithm_consistency': cand_row.get('算法一致性评分', 0.0),
                        'network_topology': cand_row.get('网络拓扑评分', 0.0),
                        'statistical_significance': cand_row.get('统计显著性评分', 0.0),
                        'support_algorithms': cand_row.get('支持算法', '').split(', ') if cand_row.get('支持算法') else []
                    }
            
            # 获取三角测量信息
            triangulation_weights = {}
            if self.triangulation_edges_df is not None:
                triangulated_edge = self.triangulation_edges_df[
                    (self.triangulation_edges_df['源节点'] == row['源节点']) & 
                    (self.triangulation_edges_df['目标节点'] == row['目标节点'])
                ]
                if not triangulated_edge.empty:
                    tri_row = triangulated_edge.iloc[0]
                    triangulation_weights = {
                        'joint_confidence': tri_row.get('联合置信度', 0.0),
                        'quality_adjusted_confidence': tri_row.get('质量调整置信度', 0.0),
                        'four_dimension_scores': {
                            'structural_consistency': tri_row.get('结构一致性分数', 0.0),
                            'parameter_fitting': tri_row.get('参数拟合分数', 0.0),
                            'mediation_support': tri_row.get('中介支持分数', 0.0),
                            'expert_direction': tri_row.get('专家定向分数', 0.0)
                        }
                    }
            
            self.W[edge_key] = {
                # 基础权重（来自精简边）
                'base_weight': {
                    'quality_level': row['质量等级'],
                    'integrated_score': row['集成评分'],
                    'support_algorithm_count': row['支持算法数量']
                },
                
                # 候选边详细信息（来自250条候选边）
                'candidate_details': candidate_info,
                
                # 三角测量权重（来自23条三角测量边）
                'triangulation_weights': triangulation_weights,
                
                # 层次权重
                'hierarchy_weight': self._calculate_hierarchy_weight(row, bool(triangulation_weights)),
                
                # 权重计算参数
                'weight_params': {
                    'base_weight': 0.4,      # 基础权重
                    'candidate_weight': 0.3,  # 候选边权重
                    'triangulation_weight': 0.3  # 三角测量权重
                }
            }
        
        print(f"✓ 权重属性 W: {len(self.W)} 条边的权重属性")
    
    def _calculate_hierarchy_weight(self, row, is_triangulated):
        """计算层次权重"""
        base_score = row['集成评分']
        
        # 质量等级权重
        quality_weights = {
            'platinum': 1.0,
            'gold': 0.8,
            'silver': 0.6,
            'bronze': 0.4
        }
        quality_weight = quality_weights.get(row['质量等级'], 0.2)
        
        # 三角测量加权
        triangulation_bonus = 0.2 if is_triangulated else 0.0
        
        # 算法支持加权
        algorithm_weight = min(row['支持算法数量'] / 5.0, 1.0)
        
        return {
            'base_score': base_score,
            'quality_weight': quality_weight,
            'triangulation_bonus': triangulation_bonus,
            'algorithm_weight': algorithm_weight,
            'final_weight': (base_score * quality_weight + triangulation_bonus) * algorithm_weight
        }
    
    def _build_parameter_attributes(self):
        """构建参数属性 Θ - 基于63条精简边的参数学习结果"""
        print("构建参数属性 Θ...")
        
        # 统计NaN处理情况
        nan_stats = {
            'total_sem_edges': 0,
            'nan_std_error_count': 0,
            'nan_t_stat_count': 0,
            'complete_data_count': 0
        }
        
        for edge_key in self.W.keys():
            source, target = edge_key.split('->')
            
            self.Theta[edge_key] = {
                # 边的参数学习结果
                'parameter_learning': {},
                
                # 条件概率表信息
                'cpt_info': {},
                
                # 边条件概率
                'edge_conditional_prob': {},
                
                # 各方法估计结果汇总
                'method_estimates': {}
            }
            
            # 提取CPT相关信息
            for method in ['MLE', 'Bayesian', 'EM', 'SEM']:
                if method == 'SEM':
                    # SEM方法的特殊处理
                    if target in self.sem_data:
                        sem_info = self.sem_data[target]
                        parents = sem_info.get('parents', [])
                        
                        self.Theta[edge_key]['cpt_info'][method] = {
                            'target_node': target,
                            'parents': parents,
                            'cpt_shape': f"({len(parents)}, 1)" if parents else "(0, 1)",
                            'has_cpt': len(parents) > 0
                        }
                        
                        # 如果source是target的父节点，记录条件概率
                        if source in parents:
                            self.Theta[edge_key]['edge_conditional_prob'][method] = {
                                'conditional_dependency': True,
                                'parent_influence': 'linear_regression'
                            }
                            
                            # 填充parameter_learning信息
                            if method not in self.Theta[edge_key]['parameter_learning']:
                                self.Theta[edge_key]['parameter_learning'][method] = {}
                            
                            parent_index = parents.index(source)
                            
                            # 安全获取SEM参数
                            coefficients = sem_info.get('coefficients', [])
                            coefficient = safe_float_conversion(
                                coefficients[parent_index] if parent_index < len(coefficients) else None,
                                default=0.0
                            )
                            intercept = safe_float_conversion(sem_info.get('intercept', 0), 0.0)
                            r_squared = safe_float_conversion(sem_info.get('r_squared', 0), 0.0)
                            
                            self.Theta[edge_key]['parameter_learning'][method] = {
                                'cpt_data': {
                                    'coefficient': coefficient,
                                    'intercept': intercept,
                                    'r_squared': r_squared
                                },
                                'node_type': sem_info.get('type', 'continuous'),
                                'parents': parents,
                                'has_complete_cpt': True
                            }
                    else:
                        # 即使没有SEM数据，也要初始化结构
                        self.Theta[edge_key]['cpt_info'][method] = {
                            'target_node': target,
                            'parents': [],
                            'cpt_shape': "unknown",
                            'has_cpt': False
                        }
                elif method in self.cpt_data and target in self.cpt_data[method]:
                    cpt_node_data = self.cpt_data[method][target]
                    
                    # 获取CPT形状信息
                    cpt_shape = "unknown"
                    has_cpt = False
                    
                    # 检查实际的CPT数据键名（probabilities 或 cpt）
                    cpt_key = None
                    if 'probabilities' in cpt_node_data:
                        cpt_key = 'probabilities'
                    elif 'cpt' in cpt_node_data:
                        cpt_key = 'cpt'
                    
                    if cpt_key and cpt_node_data[cpt_key]:
                        has_cpt = True
                        cpt_values = cpt_node_data[cpt_key]
                        parents = cpt_node_data.get('parents', [])
                        
                        if isinstance(cpt_values, dict):
                            # 计算CPT的形状
                            if parents:
                                # 根据父节点数量和CPT条目数量计算形状
                                num_entries = len(cpt_values)
                                parent_states = 2 ** len(parents)
                                target_states = 2
                                cpt_shape = f"({parent_states}, {target_states})"
                            else:
                                cpt_shape = "(1, 2)"  # 无父节点的情况
                        elif isinstance(cpt_values, list):
                            cpt_shape = f"({len(cpt_values)},)"
                    
                    self.Theta[edge_key]['cpt_info'][method] = {
                        'target_node': target,
                        'parents': cpt_node_data.get('parents', []),
                        'cpt_shape': cpt_shape,
                        'has_cpt': has_cpt
                    }
                    
                    # 如果source是target的父节点，记录条件概率
                    if source in cpt_node_data.get('parents', []):
                        self.Theta[edge_key]['edge_conditional_prob'][method] = {
                            'conditional_dependency': True,
                            'parent_influence': 'direct'
                        }
                        
                        # 填充parameter_learning信息
                        if method not in self.Theta[edge_key]['parameter_learning']:
                            self.Theta[edge_key]['parameter_learning'][method] = {}
                        
                        # 使用正确的CPT数据键名
                        cpt_data = {}
                        if cpt_key and cpt_node_data[cpt_key]:
                            cpt_data = cpt_node_data[cpt_key]
                        
                        self.Theta[edge_key]['parameter_learning'][method] = {
                            'cpt_data': cpt_data,
                            'node_type': cpt_node_data.get('type', 'unknown'),
                            'parents': cpt_node_data.get('parents', []),
                            'has_complete_cpt': has_cpt
                        }
                        
                        # 对于Bayesian方法，添加额外信息
                        if method == 'Bayesian' and 'alpha' in cpt_node_data:
                            self.Theta[edge_key]['parameter_learning'][method]['alpha'] = cpt_node_data['alpha']
                        if method == 'Bayesian' and 'sample_count' in cpt_node_data:
                            self.Theta[edge_key]['parameter_learning'][method]['sample_count'] = cpt_node_data['sample_count']
                else:
                    # 即使没有CPT数据，也要初始化结构
                    self.Theta[edge_key]['cpt_info'][method] = {
                        'target_node': target,
                        'parents': [],
                        'cpt_shape': "unknown",
                        'has_cpt': False
                    }
            
            # 从边级似然增益结果中提取方法估计结果
            for method in ['MLE', 'Bayesian', 'EM', 'SEM']:
                if method == 'SEM':
                    # SEM方法从sem_data中提取
                    if target in self.sem_data:
                        sem_info = self.sem_data[target]
                        if source in sem_info.get('parents', []):
                            parent_index = sem_info['parents'].index(source)
                            if parent_index < len(sem_info.get('coefficients', [])):
                                # 安全获取coefficient_std_error
                                std_errors = sem_info.get('coefficient_std_errors', [])
                                original_std_error = std_errors[parent_index] if parent_index < len(std_errors) else None
                                std_error = safe_float_conversion(original_std_error, default=0.0)
                                
                                # 安全获取t_statistic
                                t_stats = sem_info.get('t_statistics', {}).get('coefficients', [])
                                original_t_stat = t_stats[parent_index] if parent_index < len(t_stats) else None
                                t_stat = safe_float_conversion(original_t_stat, default=0.0)
                                
                                # 统计NaN情况
                                nan_stats['total_sem_edges'] += 1
                                if original_std_error is None or (isinstance(original_std_error, float) and np.isnan(original_std_error)):
                                    nan_stats['nan_std_error_count'] += 1
                                if original_t_stat is None or (isinstance(original_t_stat, float) and np.isnan(original_t_stat)):
                                    nan_stats['nan_t_stat_count'] += 1
                                if std_error != 0.0 and t_stat != 0.0:
                                    nan_stats['complete_data_count'] += 1
                                
                                # 安全获取其他统计量
                                coefficient = safe_float_conversion(sem_info['coefficients'][parent_index], 0.0)
                                r_squared = safe_float_conversion(sem_info.get('r_squared', 0.0), 0.0)
                                adjusted_r_squared = safe_float_conversion(sem_info.get('adjusted_r_squared', 0.0), 0.0)
                                mse = safe_float_conversion(sem_info.get('mse', 0.0), 0.0)
                                rmse = safe_float_conversion(sem_info.get('rmse', 0.0), 0.0)
                                
                                self.Theta[edge_key]['method_estimates'][method] = {
                                    'coefficient': coefficient,
                                    'coefficient_std_error': std_error,
                                    't_statistic': t_stat,
                                    'r_squared': r_squared,
                                    'adjusted_r_squared': adjusted_r_squared,
                                    'mse': mse,
                                    'rmse': rmse,
                                    # 添加数据质量标记
                                    'has_valid_std_error': std_error != 0.0,
                                    'has_valid_t_statistic': t_stat != 0.0,
                                    'data_quality': 'complete' if (std_error != 0.0 and t_stat != 0.0) else 'incomplete'
                                }
                elif method in self.likelihood_data and edge_key in self.likelihood_data[method]:
                    likelihood_info = self.likelihood_data[method][edge_key]
                    self.Theta[edge_key]['method_estimates'][method] = {
                        'likelihood_gain': likelihood_info.get('likelihood_gain', 0.0),
                        'full_likelihood': likelihood_info.get('full_likelihood', 0.0),
                        'drop_likelihood': likelihood_info.get('drop_likelihood', 0.0),
                        'S_param': likelihood_info.get('S_param', 0.0)
                    }
            
            # 添加参数稳定性信息
            if hasattr(self, 'stability_data') and edge_key in self.stability_data:
                self.Theta[edge_key]['parameter_stability'] = self.stability_data[edge_key]
        
        # 输出NaN处理统计
        print(f"✓ 参数属性 Θ: {len(self.Theta)} 条边的参数信息")
        if nan_stats['total_sem_edges'] > 0:
            print(f"  SEM数据质量统计:")
            print(f"    - 总SEM边数: {nan_stats['total_sem_edges']}")
            print(f"    - NaN标准误差: {nan_stats['nan_std_error_count']} ({nan_stats['nan_std_error_count']/nan_stats['total_sem_edges']*100:.1f}%)")
            print(f"    - NaN t统计量: {nan_stats['nan_t_stat_count']} ({nan_stats['nan_t_stat_count']/nan_stats['total_sem_edges']*100:.1f}%)")
            print(f"    - 完整数据: {nan_stats['complete_data_count']} ({nan_stats['complete_data_count']/nan_stats['total_sem_edges']*100:.1f}%)")
    
    def _build_effect_attributes(self):
        """构建效应属性 Φ - 基于63条精简边的详细中介分析结果"""
        print("构建效应属性 Φ（详细版本）...")
        
        # 为每条精简边初始化效应属性
        for edge_key in self.W.keys():
            self.Phi[edge_key] = {
                'mediation_effects': {},
                'pathway_membership': [],
                'significance_info': {},
                'effect_statistics': {},  # 新增：效应统计信息
                'confidence_intervals': {},  # 新增：置信区间信息
                'effect_directions': {},  # 新增：效应方向信息
                'mediation_types': {}  # 新增：中介类型信息
            }
        
        # 处理中介分析结果
        if self.mediation_data_df is not None:
            # 检查是否为详细版本（包含HDI等字段）
            is_detailed_version = '间接效应HDI下限' in self.mediation_data_df.columns
            
            for _, row in self.mediation_data_df.iterrows():
                pathway_desc = row['路径描述']
                
                # 解析路径中的边
                edges_in_pathway = self._parse_mediation_pathway(pathway_desc)
                
                # 构建基础中介信息
                mediation_info = {
                    'pathway_id': row['路径ID'],
                    'pathway_description': pathway_desc,
                    'indirect_effect': {
                        'mean': row['间接效应均值']
                    },
                    'direct_effect': {
                        'mean': row['直接效应均值']
                    },
                    'total_effect': {
                        'mean': row['总效应均值']
                    },
                    'mediation_ratio': row['中介比例'],
                    'significance_probability': row['显著性概率'],
                    'is_significant': row['是否显著'] == '是'
                }
                
                # 如果是详细版本，添加更多信息
                if is_detailed_version:
                    # 添加HDI置信区间
                    mediation_info['indirect_effect'].update({
                        'hdi_lower': row.get('间接效应HDI下限', None),
                        'hdi_upper': row.get('间接效应HDI上限', None),
                        'std': row.get('间接效应标准差', None)
                    })
                    
                    mediation_info['direct_effect'].update({
                        'hdi_lower': row.get('直接效应HDI下限', None),
                        'hdi_upper': row.get('直接效应HDI上限', None),
                        'std': row.get('直接效应标准差', None)
                    })
                    
                    mediation_info['total_effect'].update({
                        'hdi_lower': row.get('总效应HDI下限', None),
                        'hdi_upper': row.get('总效应HDI上限', None),
                        'std': row.get('总效应标准差', None)
                    })
                    
                    # 添加详细的中介分析信息
                    mediation_info.update({
                        'mediation_ratio_percentage': row.get('中介比例百分比', None),
                        'primary_effect_type': row.get('主要影响类型', None),
                        'effect_strength': row.get('影响强度', None),
                        'indirect_effect_direction': row.get('间接效应方向', None),
                        'direct_effect_direction': row.get('直接效应方向', None),
                        'mediation_type': row.get('中介类型', None),
                        'positive_effect_probability': row.get('正向效应概率', None),
                        'negative_effect_probability': row.get('负向效应概率', None),
                        'conclusion': row.get('结论', None)
                    })
                else:
                    # 兼容汇总版本
                    mediation_info['indirect_effect']['hdi_95'] = row.get('间接效应95%HDI', None)
                
                # 为路径中的每条边添加中介效应信息（仅限精简边）
                for edge in edges_in_pathway:
                    edge_key = f"{edge['source']}->{edge['target']}"
                    if edge_key in self.Phi:  # 只处理精简边
                        # 添加中介效应信息
                        pathway_id = row['路径ID']
                        self.Phi[edge_key]['mediation_effects'][pathway_id] = mediation_info
                        
                        # 添加路径归属
                        self.Phi[edge_key]['pathway_membership'].append({
                            'pathway_id': pathway_id,
                            'role_in_pathway': edge['role'],
                            'pathway_significance': row['显著性概率']
                        })
                        
                        # 更新显著性信息
                        if 'max_significance' not in self.Phi[edge_key]['significance_info']:
                            self.Phi[edge_key]['significance_info']['max_significance'] = 0
                        
                        current_sig = row['显著性概率']
                        if current_sig > self.Phi[edge_key]['significance_info']['max_significance']:
                            self.Phi[edge_key]['significance_info']['max_significance'] = current_sig
                            self.Phi[edge_key]['significance_info']['most_significant_pathway'] = pathway_id
                        
                        # 如果是详细版本，添加统计和方向信息
                        if is_detailed_version:
                            # 更新效应统计信息
                            self._update_effect_statistics(edge_key, row)
                            
                            # 更新置信区间信息
                            self._update_confidence_intervals(edge_key, row)
                            
                            # 更新效应方向信息
                            self._update_effect_directions(edge_key, row)
                            
                            # 更新中介类型信息
                            self._update_mediation_types(edge_key, row)
        
        # 输出构建结果
        if self.mediation_data_df is not None:
            is_detailed = '间接效应HDI下限' in self.mediation_data_df.columns
            version_info = "详细版本" if is_detailed else "汇总版本"
            print(f"✓ 效应属性 Φ: {len(self.Phi)} 条边的效应信息（{version_info}）")
            if is_detailed:
                print("  - 包含HDI置信区间、标准差、效应方向、中介类型等详细信息")
        else:
             print(f"✓ 效应属性 Φ: {len(self.Phi)} 条边（无中介分析数据）")
    
    def _update_effect_statistics(self, edge_key, row):
        """更新边的效应统计信息"""
        if 'pathways_count' not in self.Phi[edge_key]['effect_statistics']:
            self.Phi[edge_key]['effect_statistics']['pathways_count'] = 0
            self.Phi[edge_key]['effect_statistics']['significant_pathways_count'] = 0
            self.Phi[edge_key]['effect_statistics']['avg_effect_strength'] = []
            self.Phi[edge_key]['effect_statistics']['primary_effect_types'] = []
        
        self.Phi[edge_key]['effect_statistics']['pathways_count'] += 1
        
        if row.get('是否显著') == '是':
            self.Phi[edge_key]['effect_statistics']['significant_pathways_count'] += 1
        
        # 收集效应强度
        effect_strength = row.get('影响强度')
        if effect_strength and effect_strength != 'nan':
            self.Phi[edge_key]['effect_statistics']['avg_effect_strength'].append(effect_strength)
        
        # 收集主要效应类型
        primary_type = row.get('主要影响类型')
        if primary_type and primary_type != 'nan':
            self.Phi[edge_key]['effect_statistics']['primary_effect_types'].append(primary_type)
    
    def _update_confidence_intervals(self, edge_key, row):
        """更新边的置信区间信息"""
        if 'hdi_ranges' not in self.Phi[edge_key]['confidence_intervals']:
            self.Phi[edge_key]['confidence_intervals']['hdi_ranges'] = {
                'indirect_effect': [],
                'direct_effect': [],
                'total_effect': []
            }
        
        # 收集间接效应HDI
        indirect_lower = row.get('间接效应HDI下限')
        indirect_upper = row.get('间接效应HDI上限')
        if indirect_lower is not None and indirect_upper is not None:
            self.Phi[edge_key]['confidence_intervals']['hdi_ranges']['indirect_effect'].append({
                'lower': indirect_lower,
                'upper': indirect_upper,
                'pathway_id': row.get('路径ID')
            })
        
        # 收集直接效应HDI
        direct_lower = row.get('直接效应HDI下限')
        direct_upper = row.get('直接效应HDI上限')
        if direct_lower is not None and direct_upper is not None:
            self.Phi[edge_key]['confidence_intervals']['hdi_ranges']['direct_effect'].append({
                'lower': direct_lower,
                'upper': direct_upper,
                'pathway_id': row.get('路径ID')
            })
        
        # 收集总效应HDI
        total_lower = row.get('总效应HDI下限')
        total_upper = row.get('总效应HDI上限')
        if total_lower is not None and total_upper is not None:
            self.Phi[edge_key]['confidence_intervals']['hdi_ranges']['total_effect'].append({
                'lower': total_lower,
                'upper': total_upper,
                'pathway_id': row.get('路径ID')
            })
    
    def _update_effect_directions(self, edge_key, row):
        """更新边的效应方向信息"""
        if 'direction_summary' not in self.Phi[edge_key]['effect_directions']:
            self.Phi[edge_key]['effect_directions']['direction_summary'] = {
                'indirect_directions': [],
                'direct_directions': [],
                'positive_probabilities': [],
                'negative_probabilities': []
            }
        
        # 收集间接效应方向
        indirect_dir = row.get('间接效应方向')
        if indirect_dir and indirect_dir != 'nan':
            self.Phi[edge_key]['effect_directions']['direction_summary']['indirect_directions'].append(indirect_dir)
        
        # 收集直接效应方向
        direct_dir = row.get('直接效应方向')
        if direct_dir and direct_dir != 'nan':
            self.Phi[edge_key]['effect_directions']['direction_summary']['direct_directions'].append(direct_dir)
        
        # 收集正向效应概率
        pos_prob = row.get('正向效应概率')
        if pos_prob is not None:
            self.Phi[edge_key]['effect_directions']['direction_summary']['positive_probabilities'].append(pos_prob)
        
        # 收集负向效应概率
        neg_prob = row.get('负向效应概率')
        if neg_prob is not None:
            self.Phi[edge_key]['effect_directions']['direction_summary']['negative_probabilities'].append(neg_prob)
    
    def _update_mediation_types(self, edge_key, row):
        """更新边的中介类型信息"""
        if 'type_summary' not in self.Phi[edge_key]['mediation_types']:
            self.Phi[edge_key]['mediation_types']['type_summary'] = {
                'mediation_types': [],
                'mediation_ratios': [],
                'conclusions': []
            }
        
        # 收集中介类型
        mediation_type = row.get('中介类型')
        if mediation_type and mediation_type != 'nan':
            self.Phi[edge_key]['mediation_types']['type_summary']['mediation_types'].append(mediation_type)
        
        # 收集中介比例
        mediation_ratio = row.get('中介比例')
        if mediation_ratio is not None:
            self.Phi[edge_key]['mediation_types']['type_summary']['mediation_ratios'].append(mediation_ratio)
        
        # 收集结论
        conclusion = row.get('结论')
        if conclusion and conclusion != 'nan':
            self.Phi[edge_key]['mediation_types']['type_summary']['conclusions'].append(conclusion)
    
    def _parse_mediation_pathway(self, pathway_desc):
        """解析中介路径描述，提取边信息"""
        edges = []
        
        # 解析路径描述，例如: "疾病_糖尿病 → 检验_葡萄糖 → 疾病_混合痔"
        nodes = [node.strip() for node in pathway_desc.split('→')]
        
        for i in range(len(nodes) - 1):
            source = nodes[i]
            target = nodes[i + 1]
            
            # 确定边在路径中的角色
            if i == 0:
                role = 'X->M' if len(nodes) == 3 else 'initial'
            elif i == len(nodes) - 2:
                role = 'M->Y' if len(nodes) == 3 else 'final'
            else:
                role = 'mediation'
            
            edges.append({
                'source': source,
                'target': target,
                'role': role
            })
        
        return edges
    
    def _build_networkx_graph(self):
        """构建NetworkX图结构 - 优化版本，使用weight_ref访问权重信息"""
        print("构建NetworkX图结构...")
        
        # 添加节点
        for node in self.V:
            node_type = node.split('_')[0]
            self.knowledge_graph.add_node(node, 
                                        node_type=node_type,
                                        label=node)
        
        # 添加边
        for edge in self.E_core:
            source = edge['source']
            target = edge['target']
            weight_ref = edge['weight_ref']
            
            # 获取权重信息
            weight_info = self.W.get(weight_ref, {})
            hierarchy_weight = weight_info.get('hierarchy_weight', {})
            base_weight = weight_info.get('base_weight', {})
            
            # 获取质量信息
            quality_info = self.get_edge_quality_info(edge)
            
            self.knowledge_graph.add_edge(
                source, target,
                weight=hierarchy_weight.get('final_weight', quality_info.get('integrated_score', 0)),
                quality_level=quality_info.get('quality_level'),
                integrated_score=quality_info.get('integrated_score'),
                is_triangulated=self._is_edge_triangulated(weight_ref),
                relation_type=edge['relation_type'],
                edge_hierarchy=edge['edge_hierarchy'],
                support_algorithm_count=quality_info.get('support_algorithm_count'),
                weight_ref=weight_ref  # 保留引用以便后续访问
            )
        
        print(f"✓ NetworkX图结构: {len(self.knowledge_graph.nodes)} 个节点, {len(self.knowledge_graph.edges)} 条边")
    
    def _calculate_graph_statistics(self):
        """计算图统计信息"""
        print("计算图统计信息...")
        
        self.graph_statistics = {
            'basic_stats': {
                'num_nodes': len(self.knowledge_graph.nodes),
                'num_edges': len(self.knowledge_graph.edges),
                'density': nx.density(self.knowledge_graph),
                'is_connected': nx.is_weakly_connected(self.knowledge_graph),
                'num_components': nx.number_weakly_connected_components(self.knowledge_graph)
            },
            'centrality_stats': {
                'avg_in_degree': np.mean([d for n, d in self.knowledge_graph.in_degree()]),
                'avg_out_degree': np.mean([d for n, d in self.knowledge_graph.out_degree()]),
                'max_in_degree': max([d for n, d in self.knowledge_graph.in_degree()]),
                'max_out_degree': max([d for n, d in self.knowledge_graph.out_degree()])
            },
            'quality_distribution': {},
            'relation_type_distribution': {},
            'hierarchy_distribution': {}
        }
        
        # 质量等级分布
        quality_levels = []
        for edge in self.E_core:
            quality_info = self.get_edge_quality_info(edge)
            quality_levels.append(quality_info['quality_level'])
        quality_counts = Counter(quality_levels)
        self.graph_statistics['quality_distribution'] = dict(quality_counts)
        
        # 关系类型分布
        relation_counts = Counter([edge['relation_type'] for edge in self.E_core])
        self.graph_statistics['relation_type_distribution'] = dict(relation_counts)
        
        # 层次分布
        hierarchy_counts = Counter([edge['edge_hierarchy'] for edge in self.E_core])
        self.graph_statistics['hierarchy_distribution'] = dict(hierarchy_counts)
        
        print("✓ 图统计信息计算完成")
    
    def save_knowledge_graph(self):
        """保存知识图谱结果"""
        print("\n=== 保存知识图谱结果 ===")
        
        # 1. 保存完整JSON结构
        self._save_complete_json()
        
        # 2. 保存GraphML网络结构
        self._save_graphml()
        
        # 3. 保存构建报告
        self._save_build_report()
        
        print("✓ 知识图谱结果保存完成")
    
    def _save_complete_json(self):
        """保存完整的知识图谱JSON结构"""
        print("保存完整JSON结构...")
        
        kg_structure = {
            'metadata': {
                'creation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'description': '增强知识图谱 - 基于63条精简边',
                'data_sources': {
                    'candidate_edges': len(self.candidate_edges_df) if self.candidate_edges_df is not None else 0,
                    'refined_edges': len(self.refined_edges_df),
                    'triangulation_edges': len(self.triangulation_edges_df) if self.triangulation_edges_df is not None else 0
                }
            },
            'V': list(self.V),  # 节点集合
            'E_core': self.E_core,  # 核心边集合
            'R': self.R,  # 关系类型集合
            'W': self.W,  # 权重属性
            'Theta': self.Theta,  # 参数属性
            'Phi': self.Phi,  # 效应属性
            'graph_statistics': self.graph_statistics
        }
        
        json_file = os.path.join(self.output_dir, "增强知识图谱完整结构.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(kg_structure, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✓ 完整JSON结构已保存: {json_file}")
    
    def _save_graphml(self):
        """保存GraphML网络结构"""
        print("保存GraphML网络结构...")
        
        graphml_file = os.path.join(self.output_dir, "增强知识图谱网络结构.graphml")
        nx.write_graphml(self.knowledge_graph, graphml_file, encoding='utf-8')
        
        print(f"✓ GraphML网络结构已保存: {graphml_file}")
    
    def _save_build_report(self):
        """保存构建报告"""
        print("保存构建报告...")
        
        report_content = f"""
# 增强知识图谱构建报告

## 构建时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 数据源统计
- 候选边数据: {len(self.candidate_edges_df) if self.candidate_edges_df is not None else 0} 条
- 精简边数据（主体）: {len(self.refined_edges_df)} 条
- 三角测量验证边: {len(self.triangulation_edges_df) if self.triangulation_edges_df is not None else 0} 条

## 知识图谱结构 KG = (V, E_core, R, W, Θ, Φ)
- V (节点集合): {len(self.V)} 个节点
- E_core (核心边集合): {len(self.E_core)} 条边
- R (关系类型集合): {len(self.R)} 种关系类型
- W (权重属性): {len(self.W)} 条边的权重信息
- Θ (参数属性): {len(self.Theta)} 条边的参数信息
- Φ (效应属性): {len(self.Phi)} 条边的效应信息

## 图统计信息
- 节点数: {self.graph_statistics['basic_stats']['num_nodes']}
- 边数: {self.graph_statistics['basic_stats']['num_edges']}
- 图密度: {self.graph_statistics['basic_stats']['density']:.4f}
- 连通性: {'是' if self.graph_statistics['basic_stats']['is_connected'] else '否'}
- 平均聚类系数: {nx.average_clustering(self.knowledge_graph.to_undirected()):.4f}

## 质量等级分布
{chr(10).join([f"- {level}: {count} 条边" for level, count in self.graph_statistics['quality_distribution'].items()])}

## 关系类型分布
{chr(10).join([f"- {rel_type}: {count} 条边" for rel_type, count in self.graph_statistics['relation_type_distribution'].items()])}

## 层次分布
{chr(10).join([f"- {hierarchy}: {count} 条边" for hierarchy, count in self.graph_statistics['hierarchy_distribution'].items()])}

## 输出文件
- 完整JSON结构: 增强知识图谱完整结构.json
- GraphML网络结构: 增强知识图谱网络结构.graphml
- 交互式网络图: 增强知识图谱交互式网络图.html
- 构建报告: 增强知识图谱构建报告.txt
"""
        
        report_file = os.path.join(self.output_dir, "增强知识图谱构建报告.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✓ 构建报告已保存: {report_file}")
    
    def create_interactive_visualization(self):
        """创建交互式知识图谱结构展示界面"""
        print("\n=== 创建交互式知识图谱结构展示界面 ===")
        
        # 创建新的目录式界面
        html_file = self._create_kg_structure_dashboard()
        
        print(f"✓ 知识图谱结构展示界面已保存: {html_file}")
        
        return html_file
    
    def _create_kg_structure_dashboard(self):
        """创建KG=(V,E_core,R,W,Θ,Φ)结构展示界面"""
        
        # 准备各个参数的数据
        v_data = self._prepare_v_data()  # 节点集合V
        ecore_data = self._prepare_ecore_data()  # 核心边集合E_core
        r_data = self._prepare_r_data()  # 关系类型集合R
        w_data = self._prepare_w_data()  # 权重属性W
        theta_data = self._prepare_theta_data()  # 参数属性Θ
        phi_data = self._prepare_phi_data()  # 效应属性Φ
        
        # 创建HTML内容
        html_content = self._generate_dashboard_html(v_data, ecore_data, r_data, w_data, theta_data, phi_data)
        
        # 保存HTML文件
        html_file = os.path.join(self.output_dir, "增强知识图谱交互式网络图.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_file
    
    def _prepare_v_data(self):
        """准备节点集合V的数据"""
        v_data = {
            'total_nodes': len(self.V),
            'node_types': {},
            'nodes_by_type': {},
            'node_statistics': {}
        }
        
        # 按类型统计节点
        for node in self.V:
            node_type = node.split('_')[0]
            if node_type not in v_data['node_types']:
                v_data['node_types'][node_type] = 0
                v_data['nodes_by_type'][node_type] = []
            
            v_data['node_types'][node_type] += 1
            v_data['nodes_by_type'][node_type].append(node)
        
        # 计算节点统计信息
        for node in self.V:
            degree = self.knowledge_graph.degree(node) if hasattr(self, 'knowledge_graph') else 0
            in_degree = self.knowledge_graph.in_degree(node) if hasattr(self, 'knowledge_graph') else 0
            out_degree = self.knowledge_graph.out_degree(node) if hasattr(self, 'knowledge_graph') else 0
            
            v_data['node_statistics'][node] = {
                'degree': degree,
                'in_degree': in_degree,
                'out_degree': out_degree
            }
        
        return v_data
    
    def _prepare_ecore_data(self):
        """准备核心边集合E_core的数据"""
        ecore_data = {
            'total_edges': len(self.E_core),
            'edges_by_quality': {},
            'edges_by_hierarchy': {},
            'edge_details': []
        }
        
        # 统计边的质量等级和层次
        for edge in self.E_core:
            source = edge['source']
            target = edge['target']
            quality = edge.get('quality_level', 'unknown')
            hierarchy = edge.get('hierarchy', 'unknown')
            
            if quality not in ecore_data['edges_by_quality']:
                ecore_data['edges_by_quality'][quality] = 0
            ecore_data['edges_by_quality'][quality] += 1
            
            if hierarchy not in ecore_data['edges_by_hierarchy']:
                ecore_data['edges_by_hierarchy'][hierarchy] = 0
            ecore_data['edges_by_hierarchy'][hierarchy] += 1
            
            # 添加边的详细信息
            ecore_data['edge_details'].append({
                'source': source,
                'target': target,
                'quality': quality,
                'hierarchy': hierarchy,
                'is_triangulated': edge.get('is_triangulated', False)
            })
        
        return ecore_data
    
    def _prepare_r_data(self):
        """准备关系类型集合R的数据"""
        return {
            'relation_types': self.R,
            'total_types': len(self.R),
            'type_descriptions': {
                rel_type: info.get('description', '') 
                for rel_type, info in self.R.items()
            }
        }
    
    def _prepare_w_data(self):
        """准备权重属性W的数据"""
        w_data = {
            'weight_categories': {},
            'weight_statistics': {},
            'total_weighted_edges': 0
        }
        
        if hasattr(self, 'W') and self.W:
            for edge_key, weight_info in self.W.items():
                w_data['total_weighted_edges'] += 1
                
                # 统计权重类别
                for weight_type, weight_value in weight_info.items():
                    if weight_type not in w_data['weight_categories']:
                        w_data['weight_categories'][weight_type] = []
                    
                    if isinstance(weight_value, dict):
                        w_data['weight_categories'][weight_type].append(weight_value.get('value', 0))
                    else:
                        w_data['weight_categories'][weight_type].append(weight_value)
            
            # 计算权重统计
            for weight_type, values in w_data['weight_categories'].items():
                if values:
                    w_data['weight_statistics'][weight_type] = {
                        'count': len(values),
                        'mean': np.mean(values) if values else 0,
                        'std': np.std(values) if len(values) > 1 else 0,
                        'min': min(values) if values else 0,
                        'max': max(values) if values else 0
                    }
        
        return w_data
    
    def _prepare_theta_data(self):
        """准备参数属性Θ的数据"""
        theta_data = {
            'parameter_methods': {},
            'total_parameterized_edges': 0,
            'method_statistics': {}
        }
        
        if hasattr(self, 'Theta') and self.Theta:
            for edge_key, param_info in self.Theta.items():
                theta_data['total_parameterized_edges'] += 1
                
                # 统计参数方法
                for method, method_data in param_info.items():
                    if method not in theta_data['parameter_methods']:
                        theta_data['parameter_methods'][method] = 0
                    theta_data['parameter_methods'][method] += 1
                    
                    # 收集方法统计信息
                    if method not in theta_data['method_statistics']:
                        theta_data['method_statistics'][method] = {
                            'total_edges': 0,
                            'complete_data': 0,
                            'incomplete_data': 0
                        }
                    
                    theta_data['method_statistics'][method]['total_edges'] += 1
                    
                    if isinstance(method_data, dict):
                        data_quality = method_data.get('data_quality', 'unknown')
                        if data_quality == 'complete':
                            theta_data['method_statistics'][method]['complete_data'] += 1
                        else:
                            theta_data['method_statistics'][method]['incomplete_data'] += 1
        
        return theta_data
    
    def _prepare_phi_data(self):
        """准备效应属性Φ的数据"""
        phi_data = {
            'effect_types': {},
            'total_effect_edges': 0,
            'effect_statistics': {}
        }
        
        if hasattr(self, 'Phi') and self.Phi:
            for edge_key, effect_info in self.Phi.items():
                phi_data['total_effect_edges'] += 1
                
                # 统计效应类型
                for effect_type, effect_data in effect_info.items():
                    if effect_type not in phi_data['effect_types']:
                        phi_data['effect_types'][effect_type] = 0
                    phi_data['effect_types'][effect_type] += 1
                    
                    # 收集效应统计信息
                    if effect_type not in phi_data['effect_statistics']:
                        phi_data['effect_statistics'][effect_type] = {
                            'total_edges': 0,
                            'significant_effects': 0,
                            'effect_values': []
                        }
                    
                    phi_data['effect_statistics'][effect_type]['total_edges'] += 1
                    
                    if isinstance(effect_data, dict):
                        # 检查效应显著性
                        if effect_data.get('is_significant', False):
                            phi_data['effect_statistics'][effect_type]['significant_effects'] += 1
                        
                        # 收集效应值
                        effect_value = effect_data.get('effect_size', 0)
                        if effect_value != 0:
                            phi_data['effect_statistics'][effect_type]['effect_values'].append(effect_value)
        
        return phi_data
    
    def _generate_dashboard_html(self, v_data, ecore_data, r_data, w_data, theta_data, phi_data):
        """生成知识图谱结构展示界面的HTML内容"""
        
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>增强知识图谱结构展示 - KG=(V,E_core,R,W,Θ,Φ)</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Microsoft YaHei', 'Arial', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.8;
            font-weight: 300;
        }}
        
        .nav-tabs {{
            display: flex;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .nav-tab {{
            flex: 1;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            border: none;
            background: transparent;
            font-size: 1.1em;
            font-weight: 500;
            color: #495057;
            transition: all 0.3s ease;
            position: relative;
        }}
        
        .nav-tab:hover {{
            background: #e9ecef;
            color: #007bff;
        }}
        
        .nav-tab.active {{
            background: #007bff;
            color: white;
        }}
        
        .nav-tab.active::after {{
            content: '';
            position: absolute;
            bottom: -1px;
            left: 0;
            right: 0;
            height: 3px;
            background: #007bff;
        }}
        
        .tab-content {{
            display: none;
            padding: 30px;
            animation: fadeIn 0.5s ease-in;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .section-title {{
            font-size: 1.8em;
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #3498db;
            display: flex;
            align-items: center;
        }}
        
        .section-title .icon {{
            margin-right: 15px;
            font-size: 1.2em;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 25px;
            border-radius: 10px;
            border-left: 5px solid #007bff;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transition: transform 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }}
        
        .stat-number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #007bff;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            font-size: 1.1em;
            color: #6c757d;
            font-weight: 500;
        }}
        
        .detail-section {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        
        .detail-title {{
            font-size: 1.3em;
            color: #495057;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        
        .detail-list {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
        }}
        
        .detail-item {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            border-left: 3px solid #28a745;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        
        .detail-item strong {{
            color: #495057;
        }}
        
        .table-container {{
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }}
        
        .table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        .table th {{
            background: #007bff;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        .table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .table tr:hover {{
            background: #f8f9fa;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: 500;
        }}
        
        .badge-primary {{ background: #007bff; color: white; }}
        .badge-success {{ background: #28a745; color: white; }}
        .badge-warning {{ background: #ffc107; color: #212529; }}
        .badge-danger {{ background: #dc3545; color: white; }}
        .badge-info {{ background: #17a2b8; color: white; }}
        
        .progress-bar {{
            background: #e9ecef;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #007bff, #0056b3);
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 0.8em;
            font-weight: 500;
        }}
        
        .network-preview {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }}
        
        .network-preview svg {{
            max-width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>增强知识图谱结构展示</h1>
            <div class="subtitle">KG = (V, E_core, R, W, Θ, Φ) 完整结构分析</div>
        </div>
        
        <div class="nav-tabs">
            <button class="nav-tab active" onclick="showTab('v-tab')">
                <span class="icon">🔵</span> 节点集合 V
            </button>
            <button class="nav-tab" onclick="showTab('ecore-tab')">
                <span class="icon">🔗</span> 核心边集合 E_core
            </button>
            <button class="nav-tab" onclick="showTab('r-tab')">
                <span class="icon">🔄</span> 关系类型 R
            </button>
            <button class="nav-tab" onclick="showTab('w-tab')">
                <span class="icon">⚖️</span> 权重属性 W
            </button>
            <button class="nav-tab" onclick="showTab('theta-tab')">
                <span class="icon">📊</span> 参数属性 Θ
            </button>
            <button class="nav-tab" onclick="showTab('phi-tab')">
                <span class="icon">⚡</span> 效应属性 Φ
            </button>
        </div>
        
        <!-- V 节点集合 -->
        <div id="v-tab" class="tab-content active">
            <div class="section-title">
                <span class="icon">🔵</span>
                节点集合 V - 知识图谱的基础实体
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{v_data['total_nodes']}</div>
                    <div class="stat-label">总节点数</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(v_data['node_types'])}</div>
                    <div class="stat-label">节点类型数</div>
                </div>
            </div>
            
            <div class="detail-section">
                <div class="detail-title">节点类型分布</div>
                <div class="detail-list">
                    {self._generate_node_type_items(v_data)}
                </div>
            </div>
            
            <div class="table-container">
                <table class="table">
                    <thead>
                        <tr>
                            <th>节点名称</th>
                            <th>节点类型</th>
                            <th>总度数</th>
                            <th>入度</th>
                            <th>出度</th>
                        </tr>
                    </thead>
                    <tbody>
                        {self._generate_node_table_rows(v_data)}
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- E_core 核心边集合 -->
        <div id="ecore-tab" class="tab-content">
            <div class="section-title">
                <span class="icon">🔗</span>
                核心边集合 E_core - 知识图谱的关系骨架
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{ecore_data['total_edges']}</div>
                    <div class="stat-label">总边数</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(ecore_data['edges_by_quality'])}</div>
                    <div class="stat-label">质量等级数</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(ecore_data['edges_by_hierarchy'])}</div>
                    <div class="stat-label">层次类型数</div>
                </div>
            </div>
            
            <div class="detail-section">
                <div class="detail-title">边质量分布</div>
                <div class="detail-list">
                    {self._generate_quality_items(ecore_data)}
                </div>
            </div>
            
            <div class="detail-section">
                <div class="detail-title">边层次分布</div>
                <div class="detail-list">
                    {self._generate_hierarchy_items(ecore_data)}
                </div>
            </div>
            
            <div class="table-container">
                <table class="table">
                    <thead>
                        <tr>
                            <th>源节点</th>
                            <th>目标节点</th>
                            <th>质量等级</th>
                            <th>层次</th>
                            <th>三角化</th>
                        </tr>
                    </thead>
                    <tbody>
                        {self._generate_edge_table_rows(ecore_data)}
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- R 关系类型 -->
        <div id="r-tab" class="tab-content">
            <div class="section-title">
                <span class="icon">🔄</span>
                关系类型集合 R - 定义节点间的语义关系
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{r_data['total_types']}</div>
                    <div class="stat-label">关系类型总数</div>
                </div>
            </div>
            
            <div class="table-container">
                <table class="table">
                    <thead>
                        <tr>
                            <th>关系类型</th>
                            <th>描述</th>
                            <th>因果性</th>
                            <th>方向性</th>
                        </tr>
                    </thead>
                    <tbody>
                        {self._generate_relation_table_rows(r_data)}
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- W 权重属性 -->
        <div id="w-tab" class="tab-content">
            <div class="section-title">
                <span class="icon">⚖️</span>
                权重属性 W - 量化边的重要性和强度
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{w_data['total_weighted_edges']}</div>
                    <div class="stat-label">带权重边数</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(w_data['weight_categories'])}</div>
                    <div class="stat-label">权重类型数</div>
                </div>
            </div>
            
            <div class="detail-section">
                <div class="detail-title">权重统计信息</div>
                {self._generate_weight_statistics(w_data)}
            </div>
        </div>
        
        <!-- Θ 参数属性 -->
        <div id="theta-tab" class="tab-content">
            <div class="section-title">
                <span class="icon">📊</span>
                参数属性 Θ - 统计学习的模型参数
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{theta_data['total_parameterized_edges']}</div>
                    <div class="stat-label">参数化边数</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(theta_data['parameter_methods'])}</div>
                    <div class="stat-label">参数方法数</div>
                </div>
            </div>
            
            <div class="detail-section">
                <div class="detail-title">参数方法分布</div>
                <div class="detail-list">
                    {self._generate_parameter_method_items(theta_data)}
                </div>
            </div>
            
            <div class="table-container">
                <table class="table">
                    <thead>
                        <tr>
                            <th>参数方法</th>
                            <th>总边数</th>
                            <th>完整数据</th>
                            <th>不完整数据</th>
                            <th>完整率</th>
                        </tr>
                    </thead>
                    <tbody>
                        {self._generate_parameter_statistics_rows(theta_data)}
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Φ 效应属性 -->
        <div id="phi-tab" class="tab-content">
            <div class="section-title">
                <span class="icon">⚡</span>
                效应属性 Φ - 因果效应的量化结果
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{phi_data['total_effect_edges']}</div>
                    <div class="stat-label">效应边数</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(phi_data['effect_types'])}</div>
                    <div class="stat-label">效应类型数</div>
                </div>
            </div>
            
            <div class="detail-section">
                <div class="detail-title">效应类型分布</div>
                <div class="detail-list">
                    {self._generate_effect_type_items(phi_data)}
                </div>
            </div>
            
            <div class="table-container">
                <table class="table">
                    <thead>
                        <tr>
                            <th>效应类型</th>
                            <th>总边数</th>
                            <th>显著效应</th>
                            <th>显著率</th>
                            <th>平均效应值</th>
                        </tr>
                    </thead>
                    <tbody>
                        {self._generate_effect_statistics_rows(phi_data)}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        function showTab(tabId) {{
            // 隐藏所有标签页内容
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // 移除所有标签页按钮的激活状态
            const tabs = document.querySelectorAll('.nav-tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // 显示选中的标签页内容
            document.getElementById(tabId).classList.add('active');
            
            // 激活对应的标签页按钮
            event.target.classList.add('active');
        }}
        
        // 页面加载完成后的初始化
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('知识图谱结构展示界面已加载');
        }});
    </script>
</body>
</html>
        """
        
        return html_content
    
    def _generate_node_type_items(self, v_data):
        """生成节点类型项目HTML"""
        items = []
        for node_type, count in v_data['node_types'].items():
            percentage = (count / v_data['total_nodes']) * 100
            items.append(f"""
                <div class="detail-item">
                    <strong>{node_type}</strong><br>
                    数量: {count} ({percentage:.1f}%)
                </div>
            """)
        return ''.join(items)
    
    def _generate_node_table_rows(self, v_data):
        """生成节点表格行HTML"""
        rows = []
        for node, stats in list(v_data['node_statistics'].items())[:20]:  # 限制显示前20个
            node_type = node.split('_')[0]
            rows.append(f"""
                <tr>
                    <td>{node}</td>
                    <td><span class="badge badge-primary">{node_type}</span></td>
                    <td>{stats['degree']}</td>
                    <td>{stats['in_degree']}</td>
                    <td>{stats['out_degree']}</td>
                </tr>
            """)
        if len(v_data['node_statistics']) > 20:
            rows.append(f"""
                <tr>
                    <td colspan="5" style="text-align: center; color: #6c757d; font-style: italic;">
                        ... 还有 {len(v_data['node_statistics']) - 20} 个节点未显示
                    </td>
                </tr>
            """)
        return ''.join(rows)
    
    def _generate_quality_items(self, ecore_data):
        """生成质量项目HTML"""
        items = []
        quality_colors = {
            'high': 'badge-success',
            'medium': 'badge-warning', 
            'low': 'badge-danger',
            'unknown': 'badge-info'
        }
        for quality, count in ecore_data['edges_by_quality'].items():
            percentage = (count / ecore_data['total_edges']) * 100
            badge_class = quality_colors.get(quality, 'badge-info')
            items.append(f"""
                <div class="detail-item">
                    <strong><span class="badge {badge_class}">{quality}</span></strong><br>
                    数量: {count} ({percentage:.1f}%)
                </div>
            """)
        return ''.join(items)
    
    def _generate_hierarchy_items(self, ecore_data):
        """生成层次项目HTML"""
        items = []
        for hierarchy, count in ecore_data['edges_by_hierarchy'].items():
            percentage = (count / ecore_data['total_edges']) * 100
            items.append(f"""
                <div class="detail-item">
                    <strong>{hierarchy}</strong><br>
                    数量: {count} ({percentage:.1f}%)
                </div>
            """)
        return ''.join(items)
    
    def _generate_edge_table_rows(self, ecore_data):
        """生成边表格行HTML"""
        rows = []
        quality_colors = {
            'high': 'badge-success',
            'medium': 'badge-warning',
            'low': 'badge-danger',
            'unknown': 'badge-info'
        }
        
        for edge in ecore_data['edge_details'][:20]:  # 限制显示前20条
            quality_badge = quality_colors.get(edge['quality'], 'badge-info')
            triangulated_badge = 'badge-success' if edge['is_triangulated'] else 'badge-danger'
            triangulated_text = '是' if edge['is_triangulated'] else '否'
            
            rows.append(f"""
                <tr>
                    <td>{edge['source']}</td>
                    <td>{edge['target']}</td>
                    <td><span class="badge {quality_badge}">{edge['quality']}</span></td>
                    <td><span class="badge badge-info">{edge['hierarchy']}</span></td>
                    <td><span class="badge {triangulated_badge}">{triangulated_text}</span></td>
                </tr>
            """)
        
        if len(ecore_data['edge_details']) > 20:
            rows.append(f"""
                <tr>
                    <td colspan="5" style="text-align: center; color: #6c757d; font-style: italic;">
                        ... 还有 {len(ecore_data['edge_details']) - 20} 条边未显示
                    </td>
                </tr>
            """)
        return ''.join(rows)
    
    def _generate_relation_table_rows(self, r_data):
        """生成关系类型表格行HTML"""
        rows = []
        for rel_type, rel_info in r_data['relation_types'].items():
            causality = rel_info.get('causality', 'unknown')
            directionality = rel_info.get('directionality', 'unknown')
            description = rel_info.get('description', '无描述')
            
            causality_badge = 'badge-success' if causality == 'causal' else 'badge-warning'
            direction_badge = 'badge-primary' if directionality == 'directed' else 'badge-info'
            
            rows.append(f"""
                <tr>
                    <td><strong>{rel_type}</strong></td>
                    <td>{description}</td>
                    <td><span class="badge {causality_badge}">{causality}</span></td>
                    <td><span class="badge {direction_badge}">{directionality}</span></td>
                </tr>
            """)
        return ''.join(rows)
    
    def _generate_weight_statistics(self, w_data):
        """生成权重统计HTML"""
        if not w_data['weight_statistics']:
            return '<div class="detail-item">暂无权重统计数据</div>'
        
        stats_html = []
        for weight_type, stats in w_data['weight_statistics'].items():
            stats_html.append(f"""
                <div class="detail-item">
                    <strong>{weight_type}</strong><br>
                    数量: {stats['count']}<br>
                    均值: {stats['mean']:.4f}<br>
                    标准差: {stats['std']:.4f}<br>
                    范围: [{stats['min']:.4f}, {stats['max']:.4f}]
                </div>
            """)
        
        return f'<div class="detail-list">{"".join(stats_html)}</div>'
    
    def _generate_parameter_method_items(self, theta_data):
        """生成参数方法项目HTML"""
        items = []
        for method, count in theta_data['parameter_methods'].items():
            percentage = (count / theta_data['total_parameterized_edges']) * 100 if theta_data['total_parameterized_edges'] > 0 else 0
            items.append(f"""
                <div class="detail-item">
                    <strong>{method}</strong><br>
                    数量: {count} ({percentage:.1f}%)
                </div>
            """)
        return ''.join(items)
    
    def _generate_parameter_statistics_rows(self, theta_data):
        """生成参数统计表格行HTML"""
        rows = []
        for method, stats in theta_data['method_statistics'].items():
            complete_rate = (stats['complete_data'] / stats['total_edges']) * 100 if stats['total_edges'] > 0 else 0
            
            rows.append(f"""
                <tr>
                    <td><strong>{method}</strong></td>
                    <td>{stats['total_edges']}</td>
                    <td><span class="badge badge-success">{stats['complete_data']}</span></td>
                    <td><span class="badge badge-warning">{stats['incomplete_data']}</span></td>
                    <td>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {complete_rate}%">
                                {complete_rate:.1f}%
                            </div>
                        </div>
                    </td>
                </tr>
            """)
        return ''.join(rows)
    
    def _generate_effect_type_items(self, phi_data):
        """生成效应类型项目HTML"""
        items = []
        for effect_type, count in phi_data['effect_types'].items():
            percentage = (count / phi_data['total_effect_edges']) * 100 if phi_data['total_effect_edges'] > 0 else 0
            items.append(f"""
                <div class="detail-item">
                    <strong>{effect_type}</strong><br>
                    数量: {count} ({percentage:.1f}%)
                </div>
            """)
        return ''.join(items)
    
    def _generate_effect_statistics_rows(self, phi_data):
        """生成效应统计表格行HTML"""
        rows = []
        for effect_type, stats in phi_data['effect_statistics'].items():
            significant_rate = (stats['significant_effects'] / stats['total_edges']) * 100 if stats['total_edges'] > 0 else 0
            avg_effect = np.mean(stats['effect_values']) if stats['effect_values'] else 0
            
            rows.append(f"""
                <tr>
                    <td><strong>{effect_type}</strong></td>
                    <td>{stats['total_edges']}</td>
                    <td><span class="badge badge-success">{stats['significant_effects']}</span></td>
                    <td>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {significant_rate}%">
                                {significant_rate:.1f}%
                            </div>
                        </div>
                    </td>
                    <td>{avg_effect:.4f}</td>
                </tr>
            """)
        return ''.join(rows)
    
    def _prepare_node_trace(self):
        """准备节点轨迹数据"""
        # 使用spring layout布局
        pos = nx.spring_layout(self.knowledge_graph, k=3, iterations=50)
        
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        node_color = []
        node_size = []
        
        # 节点类型颜色映射
        node_type_colors = {
            '疾病': '#FF6B6B',
            '药物': '#4ECDC4', 
            '检验': '#45B7D1'
        }
        
        for node in self.knowledge_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # 节点信息
            node_type = node.split('_')[0]
            degree = self.knowledge_graph.degree(node)
            in_degree = self.knowledge_graph.in_degree(node)
            out_degree = self.knowledge_graph.out_degree(node)
            
            node_text.append(node)
            node_info.append(f"节点: {node}<br>"
                           f"类型: {node_type}<br>"
                           f"总度数: {degree}<br>"
                           f"入度: {in_degree}<br>"
                           f"出度: {out_degree}")
            
            node_color.append(node_type_colors.get(node_type, '#95A5A6'))
            node_size.append(max(10, degree * 3))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            textfont=dict(size=8),
            hovertext=node_info,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            name='节点'
        )
        
        return node_trace
    
    def _prepare_edge_traces(self):
        """准备边轨迹数据"""
        pos = nx.spring_layout(self.knowledge_graph, k=3, iterations=50)
        
        # 按质量等级分组边
        quality_groups = {
            'platinum': {'edges': [], 'color': '#FFD700', 'name': 'Platinum级'},
            'gold': {'edges': [], 'color': '#FFA500', 'name': 'Gold级'},
            'silver': {'edges': [], 'color': '#C0C0C0', 'name': 'Silver级'},
            'bronze': {'edges': [], 'color': '#CD7F32', 'name': 'Bronze级'}
        }
        
        # 分组边
        for edge in self.knowledge_graph.edges(data=True):
            source, target, data = edge
            quality = data.get('quality_level', 'bronze')
            if quality in quality_groups:
                quality_groups[quality]['edges'].append(edge)
        
        edge_traces = []
        
        for quality, group_info in quality_groups.items():
            if not group_info['edges']:
                continue
                
            edge_x = []
            edge_y = []
            edge_info = []
            edge_widths = []
            
            for source, target, data in group_info['edges']:
                x0, y0 = pos[source]
                x1, y1 = pos[target]
                
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                # 边信息
                edge_key = f"{source}->{target}"
                weight_info = self.W.get(edge_key, {})
                param_info = self.Theta.get(edge_key, {})
                effect_info = self.Phi.get(edge_key, {})
                relation_type = data.get('relation_type', 'unknown')
                
                info_text = f"<b>边: {source} → {target}</b><br>"
                info_text += f"<b>基本信息:</b><br>"
                info_text += f"• 质量等级: {data.get('quality_level', 'unknown')}<br>"
                info_text += f"• 集成评分: {data.get('integrated_score', 0):.4f}<br>"
                info_text += f"• 支持算法数: {data.get('support_algorithm_count', 0)}<br>"
                info_text += f"• 是否三角测量: {'是' if data.get('is_triangulated', False) else '否'}<br>"
                info_text += f"• 边层次: {data.get('edge_hierarchy', 'unknown')}<br><br>"
                
                # R - 关系类型详细信息
                info_text += f"<b>R - 关系类型:</b><br>"
                relation_details = self.R.get(relation_type, {})
                info_text += f"• 类型: {relation_details.get('name', relation_type)}<br>"
                info_text += f"• 语义: {relation_details.get('semantic', '未定义')}<br>"
                info_text += f"• 描述: {relation_details.get('description', '无描述')}<br><br>"
                
                # W - 权重属性详细信息
                info_text += f"<b>W - 权重属性:</b><br>"
                if weight_info:
                    base_weight = weight_info.get('base_weight', {})
                    hierarchy_weight = weight_info.get('hierarchy_weight', {})
                    candidate_details = weight_info.get('candidate_details', {})
                    triangulation_weights = weight_info.get('triangulation_weights', {})
                    
                    info_text += f"• 基础权重: {base_weight.get('integrated_score', 0):.4f}<br>"
                    info_text += f"• 层次权重: {hierarchy_weight.get('final_weight', 0):.4f}<br>"
                    info_text += f"• 质量权重: {hierarchy_weight.get('quality_weight', 0):.4f}<br>"
                    info_text += f"• 算法权重: {hierarchy_weight.get('algorithm_weight', 0):.4f}<br>"
                    
                    if candidate_details:
                        info_text += f"• 频次评分: {candidate_details.get('frequency_score', 0):.4f}<br>"
                        info_text += f"• 多样性评分: {candidate_details.get('diversity_score', 0):.4f}<br>"
                        info_text += f"• 综合评分: {candidate_details.get('comprehensive_score', 0):.4f}<br>"
                    
                    if triangulation_weights:
                        info_text += f"• 联合置信度: {triangulation_weights.get('joint_confidence', 0):.4f}<br>"
                        info_text += f"• 质量调整置信度: {triangulation_weights.get('quality_adjusted_confidence', 0):.4f}<br>"
                        four_dim = triangulation_weights.get('four_dimension_scores', {})
                        if four_dim:
                            info_text += f"• 结构一致性: {four_dim.get('structural_consistency', 0):.4f}<br>"
                            info_text += f"• 参数拟合: {four_dim.get('parameter_fitting', 0):.4f}<br>"
                            info_text += f"• 中介支持: {four_dim.get('mediation_support', 0):.4f}<br>"
                            info_text += f"• 专家定向: {four_dim.get('expert_direction', 0):.4f}<br>"
                info_text += "<br>"
                
                # Θ - 参数属性详细信息
                info_text += f"<b>Θ - 参数属性:</b><br>"
                if param_info:
                    method_estimates = param_info.get('method_estimates', {})
                    cpt_summary = param_info.get('cpt_summary', {})
                    edge_parameters = param_info.get('edge_parameters', {})
                    
                    if method_estimates:
                        info_text += f"• 参数估计方法数: {len(method_estimates)}<br>"
                        for method, estimate in method_estimates.items():
                            if isinstance(estimate, dict) and 'mean_probability' in estimate:
                                info_text += f"  - {method}: {estimate['mean_probability']:.4f}<br>"
                    
                    if cpt_summary:
                        info_text += f"• CPT变量数: {cpt_summary.get('variable_count', 0)}<br>"
                        info_text += f"• CPT总条目数: {cpt_summary.get('total_entries', 0)}<br>"
                    
                    if edge_parameters:
                        info_text += f"• 边条件概率: {edge_parameters.get('conditional_probability', 0):.4f}<br>"
                        info_text += f"• 参数置信区间: [{edge_parameters.get('confidence_interval', [0,0])[0]:.3f}, {edge_parameters.get('confidence_interval', [0,0])[1]:.3f}]<br>"
                info_text += "<br>"
                
                # Φ - 效应属性详细信息
                info_text += f"<b>Φ - 效应属性:</b><br>"
                if effect_info:
                    mediation_effects = effect_info.get('mediation_effects', [])
                    significance_info = effect_info.get('significance_info', {})
                    pathway_analysis = effect_info.get('pathway_analysis', {})
                    
                    if mediation_effects:
                         info_text += f"• 中介路径数: {len(mediation_effects)}<br>"
                         # 确保mediation_effects是列表类型
                         if isinstance(mediation_effects, list):
                             for i, effect in enumerate(mediation_effects[:3]):  # 显示前3个
                                 if isinstance(effect, dict):
                                     ie = effect.get('indirect_effect', 0)
                                     te = effect.get('total_effect', 0)
                                     mr = effect.get('mediation_ratio', 0)
                                     info_text += f"  路径{i+1} - IE: {ie:.4f}, TE: {te:.4f}, MR: {mr:.4f}<br>"
                         else:
                             info_text += f"  中介效应数据格式: {type(mediation_effects)}<br>"
                    
                    if significance_info:
                        info_text += f"• 最大显著性: {significance_info.get('max_significance', 0):.4f}<br>"
                        info_text += f"• 平均显著性: {significance_info.get('avg_significance', 0):.4f}<br>"
                        info_text += f"• 显著路径数: {significance_info.get('significant_pathways', 0)}<br>"
                    
                    if pathway_analysis:
                        info_text += f"• 路径长度: {pathway_analysis.get('pathway_length', 0)}<br>"
                        info_text += f"• 路径强度: {pathway_analysis.get('pathway_strength', 0):.4f}<br>"
                
                edge_info.append(info_text)
                
                # 边宽度基于权重
                weight = data.get('weight', 0.5)
                edge_widths.append(max(1, weight * 5))
            
            # 创建边轨迹
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color=group_info['color']),
                hoverinfo='text',
                hovertext=edge_info,
                mode='lines',
                name=group_info['name'],
                opacity=0.7
            )
            
            edge_traces.append(edge_trace)
        
        return edge_traces
    
    def start_web_server(self, port=8000):
        """启动Web服务器预览可视化结果"""
        print(f"\n=== 启动Web服务器 (端口 {port}) ===")
        
        def run_server():
            os.chdir(self.output_dir)
            handler = http.server.SimpleHTTPRequestHandler
            with socketserver.TCPServer(("", port), handler) as httpd:
                print(f"✓ Web服务器已启动: http://localhost:{port}")
                print(f"✓ 可视化文件: http://localhost:{port}/增强知识图谱交互式网络图.html")
                httpd.serve_forever()
        
        # 在后台线程中启动服务器
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # 等待服务器启动
        time.sleep(2)
        
        # 自动打开浏览器
        try:
            webbrowser.open(f'http://localhost:{port}/增强知识图谱交互式网络图.html')
            print("✓ 浏览器已自动打开")
        except:
            print("⚠ 无法自动打开浏览器，请手动访问上述URL")
        
        return f"http://localhost:{port}/增强知识图谱交互式网络图.html"

def main():
    """主函数"""
    print("=" * 60)
    print("增强知识图谱构建 - 基于63条精简边")
    print("=" * 60)
    
    # 创建知识图谱构建器
    kg_builder = EnhancedKnowledgeGraph()
    
    # 加载数据
    if not kg_builder.load_all_data():
        print("数据加载失败，程序退出")
        return
    
    # 构建知识图谱
    kg_builder.build_enhanced_knowledge_graph()
    
    # 保存结果
    kg_builder.save_knowledge_graph()
    
    # 创建可视化
    html_file = kg_builder.create_interactive_visualization()
    
    print("\n" + "=" * 60)
    print("增强知识图谱构建完成！")
    print("=" * 60)
    print(f"输出目录: {kg_builder.output_dir}")
    print(f"可视化文件: {html_file}")
    print("\n如需预览可视化结果，请运行以下命令启动Web服务器:")
    print(f"cd {kg_builder.output_dir}")
    print("python -m http.server 8000")
    print("然后在浏览器中访问: http://localhost:8000/增强知识图谱交互式网络图.html")
    print("=" * 60)

def start_preview_server(output_dir, port=8000):
    """独立的Web服务器启动函数，供手动调用"""
    print(f"\n=== 启动Web服务器预览 (端口 {port}) ===")
    
    def run_server():
        os.chdir(output_dir)
        handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"✓ Web服务器已启动: http://localhost:{port}")
            print(f"✓ 可视化文件: http://localhost:{port}/增强知识图谱交互式网络图.html")
            print("按 Ctrl+C 停止服务器")
            httpd.serve_forever()
    
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"启动服务器失败: {e}")

if __name__ == "__main__":
    import sys
    
    # 检查是否有命令行参数启动预览服务器
    if len(sys.argv) > 1 and sys.argv[1] == "--preview":
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "01增强知识图谱")
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        start_preview_server(output_dir, port)
    else:
        main()