#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因果发现结果三角验证脚本
基于证据三角测量的四维评分体系：结构一致性、参数拟合、中介支持、专家定向
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from datetime import datetime
import os
import json
import matplotlib
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
import matplotlib
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = [
    'SimHei', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 
    'Noto Sans CJK SC', 'Source Han Sans SC', 'Microsoft YaHei',
    'DejaVu Sans', 'Arial Unicode MS', 'Liberation Sans'
]
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.unicode_minus'] = False

class EvidenceTriangulation:
    """证据三角测量核心边识别系统"""
    
    def __init__(self, base_dir="/home/zkr/因果发现3"):
        self.base_dir = base_dir
        self.methods = [
            'PC算法', 'HillClimbing_AIC-D', 'GreedyEquivalence_AIC-D', 
            'TAN', 'ExpertInLoop_LLM'
        ]
        
        # 初始化各维度分数
        self.structural_scores = {}
        self.parameter_scores = {}
        self.mediation_scores = {}
        self.expert_scores = {}
        self.confidence_scores = {}
        
        # 创建输出目录
        self.output_dir = os.path.join(base_dir, "05三角测量/三角验证结果")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_structural_data(self):
        """加载结构发现阶段的数据"""
        print("=== 加载结构发现数据 ===")
        
        try:
            # 加载高质量因果边候选集
            structure_file = os.path.join(self.base_dir, "02因果发现/06候选因果边集合/高质量因果边候选集.csv")
            self.structure_df = pd.read_csv(structure_file, encoding='utf-8')
            print(f"✓ 成功加载结构发现数据: {len(self.structure_df)} 条边")
            
            return True
        except Exception as e:
            print(f"✗ 加载结构发现数据失败: {e}")
            return False
    
    def load_parameter_data(self):
        """加载参数学习阶段的数据"""
        print("=== 加载参数学习数据 ===")
        
        try:
            # 加载边级似然增益汇总
            param_file = os.path.join(self.base_dir, "03多方法参数学习/05边级似然增益结果/边级似然增益汇总.csv")
            self.parameter_df = pd.read_csv(param_file, encoding='utf-8')
            print(f"✓ 成功加载参数学习数据: {len(self.parameter_df)} 条记录")
            
            return True
        except Exception as e:
            print(f"✗ 加载参数学习数据失败: {e}")
            return False
    
    def load_mediation_data(self):
        """加载中介分析阶段的数据"""
        print("=== 加载中介分析数据 ===")
        
        try:
            # 加载贝叶斯中介分析汇总
            mediation_file = os.path.join(self.base_dir, "04贝叶斯中介分析/02贝叶斯中介分析结果/贝叶斯中介分析汇总.csv")
            self.mediation_df = pd.read_csv(mediation_file, encoding='utf-8')
            print(f"✓ 成功加载中介分析数据: {len(self.mediation_df)} 条路径")
            
            return True
        except Exception as e:
            print(f"✗ 加载中介分析数据失败: {e}")
            return False
    
    def calculate_structural_consistency_score(self):
        """计算结构一致性分数 S_struct(e)"""
        print("\n=== 计算结构一致性分数 ===")
        
        total_methods = len(self.methods)
        
        for _, row in self.structure_df.iterrows():
            edge = (row['源节点'], row['目标节点'])
            
            # 基础频次分数
            frequency_score = row['频次评分']
            
            # 算法多样性分数
            diversity_score = row['多样性评分']
            
            # 算法一致性分数
            consistency_score = row['算法一致性评分']
            
            # 综合结构一致性分数
            structural_score = (frequency_score * 0.4 + 
                              diversity_score * 0.3 + 
                              consistency_score * 0.3)
            
            self.structural_scores[edge] = {
                'score': structural_score,
                'frequency_score': frequency_score,
                'diversity_score': diversity_score,
                'consistency_score': consistency_score,
                'appearance_count': row['出现频次'],
                'support_methods': row['支持算法数量'],
                'methods': row['支持算法'].split('; ') if pd.notna(row['支持算法']) else []
            }
        
        print(f"✓ 计算了 {len(self.structural_scores)} 条边的结构一致性分数")
        return self.structural_scores
    
    def calculate_parameter_fitting_score(self):
        """计算参数拟合分数 S_param(e)"""
        print("\n=== 计算参数拟合分数 ===")
        
        # 按边分组计算平均归一化分数
        edge_param_scores = {}
        for _, row in self.parameter_df.iterrows():
            # 解析边格式 "源节点->目标节点"
            edge_str = row['Edge']
            if '->' in edge_str:
                source, target = edge_str.split('->')
                edge = (source.strip(), target.strip())
                
                # 计算多方法的平均S参数作为归一化分数
                s_params = []
                for method in ['MLE', 'Bayesian', 'EM', 'SEM']:
                    s_col = f'{method}_S_param'
                    if s_col in row and pd.notna(row[s_col]) and row[s_col] != '':
                        s_params.append(float(row[s_col]))
                
                if s_params:
                    score = np.mean(s_params)
                    if edge not in edge_param_scores:
                        edge_param_scores[edge] = []
                    edge_param_scores[edge].append(score)
        
        # 为所有结构边计算参数拟合分数
        for edge in self.structural_scores.keys():
            if edge in edge_param_scores:
                # 使用多方法的平均分数
                param_score = np.mean(edge_param_scores[edge])
                param_std = np.std(edge_param_scores[edge])
                param_count = len(edge_param_scores[edge])
            else:
                # 如果没有参数学习数据，基于结构分数估算
                struct_info = self.structural_scores[edge]
                param_score = struct_info['score'] * 0.5  # 降权处理
                param_std = 0.0
                param_count = 0
            
            self.parameter_scores[edge] = {
                'score': param_score,
                'std': param_std,
                'method_count': param_count,
                'has_real_data': edge in edge_param_scores
            }
        
        print(f"✓ 计算了 {len(self.parameter_scores)} 条边的参数拟合分数")
        print(f"✓ 其中 {len(edge_param_scores)} 条边有真实参数学习数据")
        return self.parameter_scores
    
    def calculate_mediation_support_score(self):
        """计算中介支持分数 S_mediation(e)"""
        print("\n=== 计算中介支持分数 ===")
        
        # 解析中介路径，提取边信息
        edge_mediation_info = {}
        
        for _, row in self.mediation_df.iterrows():
            path_desc = row['路径描述']
            significance_prob = row['显著性概率']
            is_significant = row['是否显著']
            indirect_effect = abs(row['间接效应均值'])
            
            # 解析路径描述，提取边
            if ' → ' in path_desc:
                nodes = path_desc.split(' → ')
                for i in range(len(nodes) - 1):
                    edge = (nodes[i], nodes[i + 1])
                    
                    # 计算中介支持强度
                    if is_significant == '是':
                        mediation_strength = significance_prob * (1 + indirect_effect)
                    else:
                        mediation_strength = significance_prob * 0.3  # 非显著路径大幅降权
                    
                    if edge not in edge_mediation_info:
                        edge_mediation_info[edge] = []
                    edge_mediation_info[edge].append({
                        'strength': mediation_strength,
                        'significance_prob': significance_prob,
                        'is_significant': is_significant,
                        'effect_size': indirect_effect
                    })
        
        # 为所有结构边计算中介支持分数
        for edge in self.structural_scores.keys():
            if edge in edge_mediation_info:
                # 使用最强的中介支持
                mediation_info = edge_mediation_info[edge]
                max_strength = max([info['strength'] for info in mediation_info])
                avg_significance = np.mean([info['significance_prob'] for info in mediation_info])
                significant_count = sum([1 for info in mediation_info if info['is_significant'] == '是'])
                
                mediation_score = max_strength
            else:
                # 基于节点类型和结构分数估算中介可能性
                source, target = edge
                struct_info = self.structural_scores[edge]
                
                # 节点类型权重
                if source.startswith('疾病_') and target.startswith('药物_'):
                    type_weight = 0.9
                elif (source.startswith('检验_') and target.startswith('疾病_')) or \
                     (source.startswith('疾病_') and target.startswith('检验_')):
                    type_weight = 0.7
                elif source.startswith('检验_') and target.startswith('药物_'):
                    type_weight = 0.6
                else:
                    type_weight = 0.4
                
                mediation_score = struct_info['score'] * type_weight * 0.3  # 估算分数降权
                avg_significance = 0.5
                significant_count = 0
            
            self.mediation_scores[edge] = {
                'score': mediation_score,
                'avg_significance': avg_significance,
                'significant_paths': significant_count,
                'total_paths': len(edge_mediation_info.get(edge, [])),
                'has_real_data': edge in edge_mediation_info
            }
        
        print(f"✓ 计算了 {len(self.mediation_scores)} 条边的中介支持分数")
        print(f"✓ 其中 {len(edge_mediation_info)} 条边有真实中介分析数据")
        return self.mediation_scores
    
    def calculate_expert_orientation_score(self):
        """计算专家定向分数 S_expert(e)"""
        print("\n=== 计算专家定向分数 ===")
        
        for edge in self.structural_scores.keys():
            struct_info = self.structural_scores[edge]
            methods = struct_info['methods']
            
            # 检查专家方法支持
            expert_support = 0.0
            if 'ExpertInLoop_LLM' in methods:
                expert_support = 1.0
            
            # 基于领域知识的节点类型评分
            source, target = edge
            domain_score = 0.0
            
            # 医学领域知识规则
            if source.startswith('疾病_') and target.startswith('药物_'):
                domain_score = 0.95  # 疾病→药物，高度合理
            elif source.startswith('疾病_') and target.startswith('检验_'):
                domain_score = 0.90  # 疾病→检验，高度合理
            elif source.startswith('检验_') and target.startswith('疾病_'):
                domain_score = 0.85  # 检验→疾病，较为合理
            elif source.startswith('药物_') and target.startswith('检验_'):
                domain_score = 0.80  # 药物→检验，较为合理
            elif source.startswith('检验_') and target.startswith('药物_'):
                domain_score = 0.70  # 检验→药物，中等合理
            elif source.startswith('药物_') and target.startswith('疾病_'):
                domain_score = 0.60  # 药物→疾病，需谨慎
            else:
                domain_score = 0.50  # 其他情况，中性
            
            # 综合专家定向分数
            expert_score = expert_support * 0.6 + domain_score * 0.4
            
            self.expert_scores[edge] = {
                'score': expert_score,
                'expert_support': expert_support,
                'domain_score': domain_score,
                'has_expert_support': expert_support > 0
            }
        
        print(f"✓ 计算了 {len(self.expert_scores)} 条边的专家定向分数")
        return self.expert_scores
    
    def calculate_triangulation_confidence(self, alpha=0.35, beta=0.25, gamma=0.25, delta=0.15):
        """计算三角测量联合置信度分数"""
        print(f"\n=== 计算三角测量联合置信度 ===")
        print(f"权重设置: 结构一致性α={alpha}, 参数拟合β={beta}, 中介支持γ={gamma}, 专家定向δ={delta}")
        
        for edge in self.structural_scores.keys():
            s_struct = self.structural_scores[edge]['score']
            s_param = self.parameter_scores[edge]['score']
            s_med = self.mediation_scores[edge]['score']
            s_expert = self.expert_scores[edge]['score']
            
            # 联合置信度公式
            confidence = alpha * s_struct + beta * s_param + gamma * s_med + delta * s_expert
            
            # 计算各维度的权重贡献
            struct_contrib = alpha * s_struct
            param_contrib = beta * s_param
            med_contrib = gamma * s_med
            expert_contrib = delta * s_expert
            
            # 数据质量评估
            has_real_param = self.parameter_scores[edge]['has_real_data']
            has_real_med = self.mediation_scores[edge]['has_real_data']
            has_expert = self.expert_scores[edge]['has_expert_support']
            
            quality_score = (
                1.0 +  # 结构数据总是真实的
                (1.0 if has_real_param else 0.3) +
                (1.0 if has_real_med else 0.3) +
                (1.0 if has_expert else 0.5)
            ) / 4.0
            
            self.confidence_scores[edge] = {
                'confidence': confidence,
                'quality_adjusted_confidence': confidence * quality_score,
                'struct_contrib': struct_contrib,
                'param_contrib': param_contrib,
                'med_contrib': med_contrib,
                'expert_contrib': expert_contrib,
                'quality_score': quality_score,
                'has_real_param': has_real_param,
                'has_real_med': has_real_med,
                'has_expert': has_expert
            }
        
        print(f"✓ 计算了 {len(self.confidence_scores)} 条边的联合置信度分数")
        return self.confidence_scores
    
    def identify_core_causal_edges(self, confidence_threshold=0.6, quality_threshold=0.6):
        """识别核心因果边"""
        print(f"\n=== 识别核心因果边 ===")
        print(f"置信度阈值: {confidence_threshold}, 质量阈值: {quality_threshold}")
        
        core_edges = []
        
        for edge, conf_info in self.confidence_scores.items():
            confidence = conf_info['confidence']
            quality_score = conf_info['quality_score']
            
            # 核心边标准：置信度≥0.6 且 质量≥0.6
            if confidence >= confidence_threshold and quality_score >= quality_threshold:
                core_edges.append(edge)
        
        print(f"✓ 识别出 {len(core_edges)} 条核心因果边")
        
        return core_edges
    
    def save_detailed_results(self):
        """保存详细的三角验证结果"""
        print("\n=== 保存详细结果 ===")
        
        # 准备详细结果数据
        detailed_results = []
        
        for edge in self.structural_scores.keys():
            source, target = edge
            
            struct_info = self.structural_scores[edge]
            param_info = self.parameter_scores[edge]
            med_info = self.mediation_scores[edge]
            expert_info = self.expert_scores[edge]
            conf_info = self.confidence_scores[edge]
            
            result_row = {
                '源节点': source,
                '目标节点': target,
                '边标识': f"{source} -> {target}",
                
                # 结构一致性维度
                '结构一致性分数': round(struct_info['score'], 4),
                '频次评分': round(struct_info['frequency_score'], 4),
                '多样性评分': round(struct_info['diversity_score'], 4),
                '算法一致性评分': round(struct_info['consistency_score'], 4),
                '出现频次': struct_info['appearance_count'],
                '支持方法数': struct_info['support_methods'],
                
                # 参数拟合维度
                '参数拟合分数': round(param_info['score'], 4),
                '参数标准差': round(param_info['std'], 4),
                '参数方法数': param_info['method_count'],
                '有真实参数数据': param_info['has_real_data'],
                
                # 中介支持维度
                '中介支持分数': round(med_info['score'], 4),
                '平均显著性概率': round(med_info['avg_significance'], 4),
                '显著路径数': med_info['significant_paths'],
                '总路径数': med_info['total_paths'],
                '有真实中介数据': med_info['has_real_data'],
                
                # 专家定向维度
                '专家定向分数': round(expert_info['score'], 4),
                '专家支持': expert_info['expert_support'],
                '领域知识评分': round(expert_info['domain_score'], 4),
                '有专家支持': expert_info['has_expert_support'],
                
                # 联合置信度
                '联合置信度': round(conf_info['confidence'], 4),
                '质量调整置信度': round(conf_info['quality_adjusted_confidence'], 4),
                '数据质量评分': round(conf_info['quality_score'], 4),
                '结构贡献': round(conf_info['struct_contrib'], 4),
                '参数贡献': round(conf_info['param_contrib'], 4),
                '中介贡献': round(conf_info['med_contrib'], 4),
                '专家贡献': round(conf_info['expert_contrib'], 4)
            }
            
            detailed_results.append(result_row)
        
        # 转换为DataFrame并排序
        results_df = pd.DataFrame(detailed_results)
        results_df = results_df.sort_values('联合置信度', ascending=False)
        
        # 保存详细结果
        detailed_file = os.path.join(self.output_dir, "三角验证详细结果.csv")
        results_df.to_csv(detailed_file, index=False, encoding='utf-8')
        print(f"✓ 详细结果已保存: {detailed_file}")
        
        return results_df
    
    def save_core_edges(self, core_edges):
        """保存核心因果边结果"""
        print("\n=== 保存核心因果边 ===")
        
        # 核心边结果
        core_results = []
        for edge in core_edges:
            source, target = edge
            conf_info = self.confidence_scores[edge]
            
            # 获取各维度分数
            struct_score = self.structural_scores.get(edge, {}).get('score', 0)
            param_score = self.parameter_scores.get(edge, {}).get('score', 0)
            mediation_score = self.mediation_scores.get(edge, {}).get('score', 0)
            expert_score = self.expert_scores.get(edge, {}).get('score', 0)
            
            core_results.append({
                '边标识': f"{source} -> {target}",
                '源节点': source,
                '目标节点': target,
                '联合置信度': round(conf_info['confidence'], 4),
                '质量调整置信度': round(conf_info['quality_adjusted_confidence'], 4),
                '数据质量评分': round(conf_info['quality_score'], 4),
                '结构一致性分数': round(struct_score, 4),
                '参数拟合分数': round(param_score, 4),
                '中介支持分数': round(mediation_score, 4),
                '专家定向分数': round(expert_score, 4),
                '边类型': '核心边'
            })
        
        # 保存核心边
        if core_results:
            core_df = pd.DataFrame(core_results)
            core_df = core_df.sort_values('联合置信度', ascending=False)
            core_file = os.path.join(self.output_dir, "核心因果边集合.csv")
            core_df.to_csv(core_file, index=False, encoding='utf-8')
            print(f"✓ 核心因果边已保存: {core_file} ({len(core_results)}条)")
        
        return core_results
    
    def save_visualizations(self, core_edges):
        """保存可视化图表"""
        print("\n=== 生成可视化图表 ===")
        
        # 1. 置信度分布图
        self._save_confidence_distribution()
        
        # 2. 四维评分雷达图（核心边）
        self._save_radar_charts(core_edges)
        
        # 3. 四维评分热力图
        self._save_heatmap()
        
        # 4. 置信度vs质量散点图
        self._save_confidence_quality_scatter()
        
        # 5. 各维度贡献度饼图
        self._save_dimension_contribution_pie()
        
        print("✓ 所有可视化图表已保存")
    
    def _save_confidence_distribution(self):
        """保存置信度分布图"""
        confidences = [info['confidence'] for info in self.confidence_scores.values()]
        quality_scores = [info['quality_score'] for info in self.confidence_scores.values()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 置信度分布直方图
        ax1.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(0.6, color='red', linestyle='--', label='核心边阈值 (0.6)')
        ax1.set_xlabel('联合置信度')
        ax1.set_ylabel('边数量')
        ax1.set_title('联合置信度分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 质量评分分布直方图
        ax2.hist(quality_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.axvline(0.6, color='red', linestyle='--', label='质量阈值 (0.6)')
        ax2.set_xlabel('数据质量评分')
        ax2.set_ylabel('边数量')
        ax2.set_title('数据质量评分分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "置信度分布图.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_radar_charts(self, core_edges):
        """保存核心边的四维评分雷达图"""
        if not core_edges:
            return
            
        # 选择前6条核心边进行雷达图展示
        top_edges = sorted(core_edges, 
                          key=lambda e: self.confidence_scores[e]['confidence'], 
                          reverse=True)[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(projection='polar'))
        axes = axes.flatten()
        
        categories = ['结构一致性', '参数拟合', '中介支持', '专家定向']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合雷达图
        
        for i, edge in enumerate(top_edges):
            if i >= 6:
                break
                
            ax = axes[i]
            source, target = edge
            
            # 获取四维分数
            scores = [
                self.structural_scores[edge]['score'],
                self.parameter_scores[edge]['score'],
                self.mediation_scores[edge]['score'],
                self.expert_scores[edge]['score']
            ]
            scores += scores[:1]  # 闭合雷达图
            
            # 绘制雷达图
            ax.plot(angles, scores, 'o-', linewidth=2, label=f'{source} → {target}')
            ax.fill(angles, scores, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_title(f'{source} → {target}\n置信度: {self.confidence_scores[edge]["confidence"]:.3f}', 
                        size=10, pad=20)
            ax.grid(True)
        
        # 隐藏多余的子图
        for i in range(len(top_edges), 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "核心边四维评分雷达图.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_heatmap(self):
        """保存四维评分热力图"""
        # 准备热力图数据
        edges_list = list(self.structural_scores.keys())
        edge_labels = [f"{s} → {t}" for s, t in edges_list]
        
        # 构建评分矩阵
        scores_matrix = []
        for edge in edges_list:
            scores = [
                self.structural_scores[edge]['score'],
                self.parameter_scores[edge]['score'],
                self.mediation_scores[edge]['score'],
                self.expert_scores[edge]['score']
            ]
            scores_matrix.append(scores)
        
        scores_matrix = np.array(scores_matrix)
        
        # 只显示前20条边（按置信度排序）
        confidences = [self.confidence_scores[edge]['confidence'] for edge in edges_list]
        top_indices = np.argsort(confidences)[-20:][::-1]
        
        plt.figure(figsize=(10, 12))
        sns.heatmap(scores_matrix[top_indices], 
                   xticklabels=['结构一致性', '参数拟合', '中介支持', '专家定向'],
                   yticklabels=[edge_labels[i] for i in top_indices],
                   annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   cbar_kws={'label': '评分'})
        plt.title('前20条边的四维评分热力图')
        plt.xlabel('评分维度')
        plt.ylabel('因果边')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "四维评分热力图.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_confidence_quality_scatter(self):
        """保存置信度vs质量散点图"""
        confidences = []
        qualities = []
        edge_types = []
        
        for edge, conf_info in self.confidence_scores.items():
            confidences.append(conf_info['confidence'])
            qualities.append(conf_info['quality_score'])
            
            # 判断边类型
            if conf_info['confidence'] >= 0.6 and conf_info['quality_score'] >= 0.6:
                edge_types.append('核心边')
            else:
                edge_types.append('普通边')
        
        plt.figure(figsize=(10, 8))
        
        # 分类绘制散点图
        for edge_type in ['普通边', '核心边']:
            mask = np.array(edge_types) == edge_type
            color = 'red' if edge_type == '核心边' else 'lightblue'
            plt.scatter(np.array(confidences)[mask], np.array(qualities)[mask], 
                       c=color, label=edge_type, alpha=0.7, s=50)
        
        # 添加阈值线
        plt.axvline(0.6, color='gray', linestyle='--', alpha=0.7, label='置信度阈值')
        plt.axhline(0.6, color='gray', linestyle='--', alpha=0.7, label='质量阈值')
        
        plt.xlabel('联合置信度')
        plt.ylabel('数据质量评分')
        plt.title('置信度 vs 质量评分散点图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "置信度质量散点图.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_dimension_contribution_pie(self):
        """保存各维度贡献度饼图"""
        # 计算各维度的平均贡献
        struct_contribs = [info['struct_contrib'] for info in self.confidence_scores.values()]
        param_contribs = [info['param_contrib'] for info in self.confidence_scores.values()]
        med_contribs = [info['med_contrib'] for info in self.confidence_scores.values()]
        expert_contribs = [info['expert_contrib'] for info in self.confidence_scores.values()]
        
        avg_contributions = [
            np.mean(struct_contribs),
            np.mean(param_contribs),
            np.mean(med_contribs),
            np.mean(expert_contribs)
        ]
        
        labels = ['结构一致性', '参数拟合', '中介支持', '专家定向']
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        plt.figure(figsize=(10, 8))
        plt.pie(avg_contributions, labels=labels, colors=colors, autopct='%1.1f%%', 
                startangle=90, explode=(0.05, 0.05, 0.05, 0.05))
        plt.title('各维度平均贡献度分布')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "维度贡献度饼图.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_structured_data(self, core_edges):
        """保存JSON格式的结构化数据"""
        print("\n=== 保存结构化数据 ===")
        
        # 1. 完整分析结果
        analysis_results = {
            "metadata": {
                "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_edges": len(self.confidence_scores),
                "core_edges_count": len(core_edges),
                "filtering_criteria": {
                    "confidence_threshold": 0.6,
                    "quality_threshold": 0.6
                }
            },
            "scoring_weights": {
                "structural_consistency": 0.35,
                "parameter_fitting": 0.25,
                "mediation_support": 0.25,
                "expert_direction": 0.15
            },
            "all_edges": {},
            "core_edges": {},
            "statistics": {}
        }
        
        # 2. 所有边的详细信息
        for edge, conf_info in self.confidence_scores.items():
            source, target = edge
            edge_key = f"{source} -> {target}"
            
            analysis_results["all_edges"][edge_key] = {
                "source": source,
                "target": target,
                "confidence": conf_info['confidence'],
                "quality_adjusted_confidence": conf_info['quality_adjusted_confidence'],
                "quality_score": conf_info['quality_score'],
                "dimension_scores": {
                    "structural_consistency": self.structural_scores[edge]['score'],
                    "parameter_fitting": self.parameter_scores[edge]['score'],
                    "mediation_support": self.mediation_scores[edge]['score'],
                    "expert_direction": self.expert_scores[edge]['score']
                },
                "dimension_contributions": {
                    "structural_consistency": conf_info['struct_contrib'],
                    "parameter_fitting": conf_info['param_contrib'],
                    "mediation_support": conf_info['med_contrib'],
                    "expert_direction": conf_info['expert_contrib']
                },
                "is_core_edge": edge in core_edges
            }
        
        # 3. 核心边的详细信息
        for edge in core_edges:
            source, target = edge
            edge_key = f"{source} -> {target}"
            analysis_results["core_edges"][edge_key] = analysis_results["all_edges"][edge_key]
        
        # 4. 统计信息
        confidences = [info['confidence'] for info in self.confidence_scores.values()]
        qualities = [info['quality_score'] for info in self.confidence_scores.values()]
        
        analysis_results["statistics"] = {
            "confidence_stats": {
                "mean": float(np.mean(confidences)),
                "std": float(np.std(confidences)),
                "min": float(np.min(confidences)),
                "max": float(np.max(confidences)),
                "median": float(np.median(confidences)),
                "percentiles": {
                    "25th": float(np.percentile(confidences, 25)),
                    "75th": float(np.percentile(confidences, 75)),
                    "90th": float(np.percentile(confidences, 90))
                }
            },
            "quality_stats": {
                "mean": float(np.mean(qualities)),
                "std": float(np.std(qualities)),
                "min": float(np.min(qualities)),
                "max": float(np.max(qualities)),
                "median": float(np.median(qualities)),
                "percentiles": {
                    "25th": float(np.percentile(qualities, 25)),
                    "75th": float(np.percentile(qualities, 75)),
                    "90th": float(np.percentile(qualities, 90))
                }
            },
            "dimension_stats": {
                "structural_consistency": self._calculate_dimension_stats("structural"),
                "parameter_fitting": self._calculate_dimension_stats("parameter"),
                "mediation_support": self._calculate_dimension_stats("mediation"),
                "expert_direction": self._calculate_dimension_stats("expert")
            }
        }
        
        # 保存完整分析结果
        json_file = os.path.join(self.output_dir, "三角验证完整结果.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        print(f"✓ 完整分析结果已保存: {json_file}")
        
        # 5. 保存核心边简化版本
        core_edges_simple = {
            "metadata": analysis_results["metadata"],
            "core_edges": analysis_results["core_edges"],
            "statistics": {
                "core_edges_confidence_stats": self._calculate_core_edges_stats(core_edges, "confidence"),
                "core_edges_quality_stats": self._calculate_core_edges_stats(core_edges, "quality")
            }
        }
        
        core_json_file = os.path.join(self.output_dir, "核心因果边结构化数据.json")
        with open(core_json_file, 'w', encoding='utf-8') as f:
            json.dump(core_edges_simple, f, ensure_ascii=False, indent=2)
        print(f"✓ 核心边结构化数据已保存: {core_json_file}")
        
        return analysis_results
    
    def _calculate_dimension_stats(self, dimension_type):
        """计算特定维度的统计信息"""
        if dimension_type == "structural":
            scores = [info['score'] for info in self.structural_scores.values()]
        elif dimension_type == "parameter":
            scores = [info['score'] for info in self.parameter_scores.values()]
        elif dimension_type == "mediation":
            scores = [info['score'] for info in self.mediation_scores.values()]
        elif dimension_type == "expert":
            scores = [info['score'] for info in self.expert_scores.values()]
        else:
            return {}
        
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores))
        }
    
    def _calculate_core_edges_stats(self, core_edges, stat_type):
        """计算核心边的统计信息"""
        if stat_type == "confidence":
            values = [self.confidence_scores[edge]['confidence'] for edge in core_edges]
        elif stat_type == "quality":
            values = [self.confidence_scores[edge]['quality_score'] for edge in core_edges]
        else:
            return {}
        
        if not values:
            return {}
        
        return {
             "mean": float(np.mean(values)),
             "std": float(np.std(values)),
             "min": float(np.min(values)),
             "max": float(np.max(values)),
             "median": float(np.median(values))
         }
    
    def save_detailed_edge_reports(self, core_edges):
        """保存边级详细分析报告"""
        print("\n=== 生成边级详细分析报告 ===")
        
        # 1. 核心边详细报告
        core_report_lines = []
        core_report_lines.append("=" * 80)
        core_report_lines.append("核心因果边详细分析报告")
        core_report_lines.append("=" * 80)
        core_report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        core_report_lines.append(f"核心边数量: {len(core_edges)}")
        core_report_lines.append(f"筛选标准: 置信度≥0.6 且 质量≥0.6")
        core_report_lines.append("")
        
        # 按置信度排序核心边
        sorted_core_edges = sorted(core_edges, 
                                 key=lambda e: self.confidence_scores[e]['confidence'], 
                                 reverse=True)
        
        for i, edge in enumerate(sorted_core_edges, 1):
            source, target = edge
            conf_info = self.confidence_scores[edge]
            
            core_report_lines.append(f"【核心边 {i:02d}】 {source} → {target}")
            core_report_lines.append("-" * 60)
            
            # 基本信息
            core_report_lines.append("▶ 基本评分信息:")
            core_report_lines.append(f"  • 联合置信度: {conf_info['confidence']:.4f}")
            core_report_lines.append(f"  • 质量调整置信度: {conf_info['quality_adjusted_confidence']:.4f}")
            core_report_lines.append(f"  • 数据质量评分: {conf_info['quality_score']:.4f}")
            core_report_lines.append("")
            
            # 四维评分详情
            core_report_lines.append("▶ 四维评分详情:")
            struct_score = self.structural_scores[edge]['score']
            param_score = self.parameter_scores[edge]['score']
            med_score = self.mediation_scores[edge]['score']
            expert_score = self.expert_scores[edge]['score']
            
            core_report_lines.append(f"  • 结构一致性: {struct_score:.4f} (权重: 35%)")
            core_report_lines.append(f"    - 贡献度: {conf_info['struct_contrib']:.4f}")
            if edge in self.structural_scores:
                struct_detail = self.structural_scores[edge]
                if 'details' in struct_detail:
                    core_report_lines.append(f"    - 详细信息: {struct_detail['details']}")
            
            core_report_lines.append(f"  • 参数拟合: {param_score:.4f} (权重: 25%)")
            core_report_lines.append(f"    - 贡献度: {conf_info['param_contrib']:.4f}")
            if edge in self.parameter_scores:
                param_detail = self.parameter_scores[edge]
                if 'details' in param_detail:
                    core_report_lines.append(f"    - 详细信息: {param_detail['details']}")
            
            core_report_lines.append(f"  • 中介支持: {med_score:.4f} (权重: 25%)")
            core_report_lines.append(f"    - 贡献度: {conf_info['med_contrib']:.4f}")
            if edge in self.mediation_scores:
                med_detail = self.mediation_scores[edge]
                if 'details' in med_detail:
                    core_report_lines.append(f"    - 详细信息: {med_detail['details']}")
            
            core_report_lines.append(f"  • 专家定向: {expert_score:.4f} (权重: 15%)")
            core_report_lines.append(f"    - 贡献度: {conf_info['expert_contrib']:.4f}")
            if edge in self.expert_scores:
                expert_detail = self.expert_scores[edge]
                if 'details' in expert_detail:
                    core_report_lines.append(f"    - 详细信息: {expert_detail['details']}")
            
            core_report_lines.append("")
            
            # 质量评估
            core_report_lines.append("▶ 质量评估:")
            if conf_info['confidence'] >= 0.8:
                quality_level = "极高置信度"
            elif conf_info['confidence'] >= 0.7:
                quality_level = "高置信度"
            else:
                quality_level = "中等置信度"
            
            core_report_lines.append(f"  • 置信度等级: {quality_level}")
            core_report_lines.append(f"  • 综合评价: 该因果边具有{quality_level}，")
            
            # 找出最强和最弱的维度
            scores_dict = {
                '结构一致性': struct_score,
                '参数拟合': param_score,
                '中介支持': med_score,
                '专家定向': expert_score
            }
            max_dim = max(scores_dict, key=scores_dict.get)
            min_dim = min(scores_dict, key=scores_dict.get)
            
            core_report_lines.append(f"    最强支持维度为{max_dim}({scores_dict[max_dim]:.3f})，")
            core_report_lines.append(f"    最弱支持维度为{min_dim}({scores_dict[min_dim]:.3f})")
            
            core_report_lines.append("")
            core_report_lines.append("=" * 60)
            core_report_lines.append("")
        
        # 保存核心边详细报告
        core_report_file = os.path.join(self.output_dir, "核心边详细分析报告.txt")
        with open(core_report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(core_report_lines))
        print(f"✓ 核心边详细分析报告已保存: {core_report_file}")
        
        # 2. 所有边的简化报告
        all_edges_report = []
        all_edges_report.append("=" * 80)
        all_edges_report.append("所有候选边分析摘要")
        all_edges_report.append("=" * 80)
        all_edges_report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        all_edges_report.append(f"总边数: {len(self.confidence_scores)}")
        all_edges_report.append(f"核心边数: {len(core_edges)}")
        all_edges_report.append("")
        
        # 按置信度排序所有边
        all_edges_sorted = sorted(self.confidence_scores.items(), 
                                key=lambda x: x[1]['confidence'], 
                                reverse=True)
        
        all_edges_report.append("边标识\t\t\t置信度\t质量\t结构\t参数\t中介\t专家\t类型")
        all_edges_report.append("-" * 80)
        
        for edge, conf_info in all_edges_sorted:
            source, target = edge
            edge_type = "核心边" if edge in core_edges else "普通边"
            
            struct_score = self.structural_scores[edge]['score']
            param_score = self.parameter_scores[edge]['score']
            med_score = self.mediation_scores[edge]['score']
            expert_score = self.expert_scores[edge]['score']
            
            edge_label = f"{source} → {target}"
            if len(edge_label) > 20:
                edge_label = edge_label[:17] + "..."
            
            all_edges_report.append(
                f"{edge_label:<20}\t{conf_info['confidence']:.3f}\t"
                f"{conf_info['quality_score']:.3f}\t{struct_score:.3f}\t"
                f"{param_score:.3f}\t{med_score:.3f}\t{expert_score:.3f}\t{edge_type}"
            )
        
        # 保存所有边摘要报告
        all_report_file = os.path.join(self.output_dir, "所有边分析摘要.txt")
        with open(all_report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_edges_report))
        print(f"✓ 所有边分析摘要已保存: {all_report_file}")
    
    def save_network_visualizations(self, core_edges):
        """保存网络图可视化"""
        print("\n=== 生成网络图可视化 ===")
        
        # 1. 核心因果边网络图
        self._save_core_edges_network(core_edges)
        
        # 2. 全网络图（所有候选边）
        self._save_full_network()
        
        # 3. 分层网络图（按置信度分层）
        self._save_layered_network()
        
        # 4. 交互式网络图数据
        self._save_interactive_network_data(core_edges)
        
        print("✓ 所有网络图可视化已保存完成")
    
    def _save_core_edges_network(self, core_edges):
        """保存核心因果边网络图"""
        if not core_edges:
            print("⚠ 没有核心边，跳过核心边网络图")
            return
            
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加边和权重
        for edge in core_edges:
            # edge是元组格式 (source, target)
            if isinstance(edge, tuple):
                source, target = edge
            else:
                # 如果是字符串格式，则分割
                source, target = edge.split(' -> ')
            confidence = self.confidence_scores[edge]['confidence']
            quality = self.confidence_scores[edge]['quality_score']
            G.add_edge(source, target, weight=confidence, quality=quality)
        
        # 设置图形大小
        plt.figure(figsize=(16, 12))
        
        # 使用spring布局
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # 绘制节点
        node_sizes = [300 + G.degree(node) * 100 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.8)
        
        # 绘制边，边的粗细和颜色表示置信度
        edges = G.edges()
        confidences = [self.confidence_scores[(u, v)]['confidence'] for u, v in edges]
        
        # 边的粗细
        edge_widths = [conf * 5 for conf in confidences]
        
        # 边的颜色映射
        edge_colors = plt.cm.Reds([conf for conf in confidences])
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors,
                              arrows=True, arrowsize=20, arrowstyle='->')
        
        # 绘制节点标签
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        # 添加边标签（置信度）
        edge_labels = {(u, v): f"{self.confidence_scores[(u, v)]['confidence']:.3f}" 
                      for u, v in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
        
        plt.title('核心因果边网络图\n(边的粗细和颜色表示置信度)', fontsize=16, fontweight='bold')
        plt.axis('off')
        
        # 添加颜色条
        if confidences:  # 只有当有边时才添加颜色条
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, 
                                      norm=plt.Normalize(vmin=min(confidences), vmax=max(confidences)))
            sm.set_array([])
            cbar = plt.colorbar(sm, shrink=0.8, ax=plt.gca())
            cbar.set_label('置信度', fontsize=12)
        
        plt.tight_layout()
        network_file = os.path.join(self.output_dir, "核心因果边网络图.png")
        plt.savefig(network_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 核心因果边网络图已保存: {network_file}")
    
    def _save_full_network(self):
        """保存全网络图（所有候选边）"""
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加所有边
        for edge, info in self.confidence_scores.items():
            # edge是元组格式 (source, target)
            if isinstance(edge, tuple):
                source, target = edge
            else:
                source, target = edge.split(' -> ')
            G.add_edge(source, target, 
                      confidence=info['confidence'],
                      quality=info['quality_score'])
        
        # 设置图形大小
        plt.figure(figsize=(20, 16))
        
        # 使用分层布局
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # 按置信度分类边
        high_conf_edges = [(u, v) for u, v in G.edges() 
                          if G[u][v]['confidence'] >= 0.8]
        med_conf_edges = [(u, v) for u, v in G.edges() 
                         if 0.6 <= G[u][v]['confidence'] < 0.8]
        low_conf_edges = [(u, v) for u, v in G.edges() 
                         if G[u][v]['confidence'] < 0.6]
        
        # 绘制节点
        node_sizes = [200 + G.degree(node) * 50 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightgray', alpha=0.7)
        
        # 分别绘制不同置信度的边
        if high_conf_edges:
            nx.draw_networkx_edges(G, pos, edgelist=high_conf_edges, 
                                  edge_color='red', width=3, alpha=0.8,
                                  arrows=True, arrowsize=15)
        if med_conf_edges:
            nx.draw_networkx_edges(G, pos, edgelist=med_conf_edges, 
                                  edge_color='orange', width=2, alpha=0.6,
                                  arrows=True, arrowsize=12)
        if low_conf_edges:
            nx.draw_networkx_edges(G, pos, edgelist=low_conf_edges, 
                                  edge_color='gray', width=1, alpha=0.4,
                                  arrows=True, arrowsize=10)
        
        # 绘制节点标签
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        plt.title('完整因果网络图\n(红色: 高置信度≥0.8, 橙色: 中等置信度0.6-0.8, 灰色: 低置信度<0.6)', 
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        
        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='red', lw=3, label='高置信度 (≥0.8)'),
                          Line2D([0], [0], color='orange', lw=2, label='中等置信度 (0.6-0.8)'),
                          Line2D([0], [0], color='gray', lw=1, label='低置信度 (<0.6)')]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        full_network_file = os.path.join(self.output_dir, "完整因果网络图.png")
        plt.savefig(full_network_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 完整因果网络图已保存: {full_network_file}")
    
    def _save_layered_network(self):
        """保存分层网络图"""
        # 按置信度分层
        high_edges = {edge: info for edge, info in self.confidence_scores.items() 
                     if info['confidence'] >= 0.8}
        med_edges = {edge: info for edge, info in self.confidence_scores.items() 
                    if 0.6 <= info['confidence'] < 0.8}
        low_edges = {edge: info for edge, info in self.confidence_scores.items() 
                    if info['confidence'] < 0.6}
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        layers = [
            (high_edges, '高置信度层 (≥0.8)', 'red', axes[0]),
            (med_edges, '中等置信度层 (0.6-0.8)', 'orange', axes[1]),
            (low_edges, '低置信度层 (<0.6)', 'gray', axes[2])
        ]
        
        for edges_dict, title, color, ax in layers:
            if not edges_dict:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.axis('off')
                continue
                
            # 创建子图
            G = nx.DiGraph()
            for edge in edges_dict:
                # edge是元组格式 (source, target)
                if isinstance(edge, tuple):
                    source, target = edge
                else:
                    source, target = edge.split(' -> ')
                G.add_edge(source, target)
            
            # 布局
            pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
            
            # 绘制
            nx.draw(G, pos, ax=ax, with_labels=True, 
                   node_color='lightblue', node_size=300,
                   edge_color=color, width=2, arrows=True,
                   font_size=8, font_weight='bold')
            
            ax.set_title(f'{title}\n({len(edges_dict)}条边)', 
                        fontsize=14, fontweight='bold')
        
        plt.suptitle('分层因果网络图', fontsize=18, fontweight='bold')
        plt.tight_layout()
        layered_file = os.path.join(self.output_dir, "分层因果网络图.png")
        plt.savefig(layered_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 分层因果网络图已保存: {layered_file}")
    
    def _save_interactive_network_data(self, core_edges):
        """保存交互式网络图数据（JSON格式）"""
        # 准备节点数据
        all_nodes = set()
        for edge in self.confidence_scores:
            # edge是元组格式 (source, target)
            if isinstance(edge, tuple):
                source, target = edge
            else:
                source, target = edge.split(' -> ')
            all_nodes.add(source)
            all_nodes.add(target)
        
        nodes_data = []
        for node in all_nodes:
            # 计算节点的入度和出度
            in_degree = sum(1 for edge in self.confidence_scores 
                           if (isinstance(edge, tuple) and edge[1] == node) or 
                              (isinstance(edge, str) and edge.endswith(f' -> {node}')))
            out_degree = sum(1 for edge in self.confidence_scores 
                            if (isinstance(edge, tuple) and edge[0] == node) or
                               (isinstance(edge, str) and edge.startswith(f'{node} -> ')))
            
            nodes_data.append({
                'id': node,
                'label': node,
                'in_degree': in_degree,
                'out_degree': out_degree,
                'total_degree': in_degree + out_degree,
                'is_core_node': any(node in edge for edge in core_edges)
            })
        
        # 准备边数据
        edges_data = []
        for edge, info in self.confidence_scores.items():
            # edge是元组格式 (source, target)
            if isinstance(edge, tuple):
                source, target = edge
            else:
                source, target = edge.split(' -> ')
            edges_data.append({
                'source': source,
                'target': target,
                'confidence': info['confidence'],
                'quality': info['quality_score'],
                'is_core_edge': edge in core_edges,
                'struct_score': self.structural_scores[edge]['score'],
                'param_score': self.parameter_scores[edge]['score'],
                'mediation_score': self.mediation_scores[edge]['score'],
                'expert_score': self.expert_scores[edge]['score']
            })
        
        # 网络统计信息
        network_stats = {
            'total_nodes': len(all_nodes),
            'total_edges': len(self.confidence_scores),
            'core_edges': len(core_edges),
            'avg_confidence': np.mean([info['confidence'] for info in self.confidence_scores.values()]),
            'avg_quality': np.mean([info['quality_score'] for info in self.confidence_scores.values()]),
            'density': len(self.confidence_scores) / (len(all_nodes) * (len(all_nodes) - 1)) if len(all_nodes) > 1 else 0
        }
        
        # 组合数据
        interactive_data = {
            'metadata': {
                'title': '因果发现网络图数据',
                'description': '用于交互式可视化的网络数据',
                'created_at': datetime.now().isoformat(),
                'statistics': network_stats
            },
            'nodes': nodes_data,
            'edges': edges_data,
            'layout_suggestions': {
                'force_directed': {
                    'description': '力导向布局，适合展示整体结构',
                    'parameters': {'k': 2, 'iterations': 50}
                },
                'hierarchical': {
                    'description': '分层布局，适合展示因果流向',
                    'parameters': {'direction': 'TB'}
                },
                'circular': {
                    'description': '环形布局，适合展示节点关系',
                    'parameters': {'scale': 1}
                }
            }
        }
        
        # 保存数据
        interactive_file = os.path.join(self.output_dir, "交互式网络图数据.json")
        with open(interactive_file, 'w', encoding='utf-8') as f:
            json.dump(interactive_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 交互式网络图数据已保存: {interactive_file}")
        
        # 同时保存简化版的Gephi格式数据
        self._save_gephi_format(nodes_data, edges_data)
    
    def _save_gephi_format(self, nodes_data, edges_data):
        """保存Gephi格式的网络数据"""
        # 节点文件
        nodes_df = pd.DataFrame(nodes_data)
        nodes_file = os.path.join(self.output_dir, "网络节点数据_Gephi格式.csv")
        nodes_df.to_csv(nodes_file, index=False, encoding='utf-8')
        
        # 边文件
        edges_df = pd.DataFrame(edges_data)
        edges_df = edges_df.rename(columns={'source': 'Source', 'target': 'Target'})
        edges_file = os.path.join(self.output_dir, "网络边数据_Gephi格式.csv")
        edges_df.to_csv(edges_file, index=False, encoding='utf-8')
        
        print(f"✓ Gephi格式数据已保存: {nodes_file}, {edges_file}")
    
    def save_statistical_summary(self, core_edges):
        print("\n=== 生成统计摘要文件 ===")
        
        # 1. 综合统计摘要
        stats_lines = []
        stats_lines.append("=" * 80)
        stats_lines.append("因果发现三角验证统计摘要")
        stats_lines.append("=" * 80)
        stats_lines.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        stats_lines.append("")
        
        # 基本统计
        total_edges = len(self.confidence_scores)
        core_edges_count = len(core_edges)
        core_rate = (core_edges_count / total_edges * 100) if total_edges > 0 else 0
        
        stats_lines.append("【基本统计】")
        stats_lines.append(f"总候选边数: {total_edges}")
        stats_lines.append(f"核心因果边数: {core_edges_count}")
        stats_lines.append(f"核心边比例: {core_rate:.2f}%")
        stats_lines.append(f"筛选标准: 置信度≥0.6 且 质量≥0.6")
        stats_lines.append("")
        
        # 置信度统计
        confidences = [info['confidence'] for info in self.confidence_scores.values()]
        qualities = [info['quality_score'] for info in self.confidence_scores.values()]
        
        stats_lines.append("【置信度分布统计】")
        stats_lines.append(f"平均置信度: {np.mean(confidences):.4f}")
        stats_lines.append(f"置信度标准差: {np.std(confidences):.4f}")
        stats_lines.append(f"置信度中位数: {np.median(confidences):.4f}")
        stats_lines.append(f"置信度范围: [{np.min(confidences):.4f}, {np.max(confidences):.4f}]")
        
        # 置信度分段统计
        high_conf = sum(1 for c in confidences if c >= 0.8)
        med_conf = sum(1 for c in confidences if 0.6 <= c < 0.8)
        low_conf = sum(1 for c in confidences if c < 0.6)
        
        stats_lines.append(f"高置信度边(≥0.8): {high_conf} ({high_conf/total_edges*100:.1f}%)")
        stats_lines.append(f"中等置信度边(0.6-0.8): {med_conf} ({med_conf/total_edges*100:.1f}%)")
        stats_lines.append(f"低置信度边(<0.6): {low_conf} ({low_conf/total_edges*100:.1f}%)")
        stats_lines.append("")
        
        # 质量评分统计
        stats_lines.append("【质量评分分布统计】")
        stats_lines.append(f"平均质量评分: {np.mean(qualities):.4f}")
        stats_lines.append(f"质量评分标准差: {np.std(qualities):.4f}")
        stats_lines.append(f"质量评分中位数: {np.median(qualities):.4f}")
        stats_lines.append(f"质量评分范围: [{np.min(qualities):.4f}, {np.max(qualities):.4f}]")
        stats_lines.append("")
        
        # 四维评分统计
        struct_scores = [info['score'] for info in self.structural_scores.values()]
        param_scores = [info['score'] for info in self.parameter_scores.values()]
        med_scores = [info['score'] for info in self.mediation_scores.values()]
        expert_scores = [info['score'] for info in self.expert_scores.values()]
        
        stats_lines.append("【四维评分统计】")
        stats_lines.append("维度\t\t平均值\t标准差\t中位数\t最小值\t最大值")
        stats_lines.append("-" * 60)
        stats_lines.append(f"结构一致性\t{np.mean(struct_scores):.4f}\t{np.std(struct_scores):.4f}\t"
                          f"{np.median(struct_scores):.4f}\t{np.min(struct_scores):.4f}\t{np.max(struct_scores):.4f}")
        stats_lines.append(f"参数拟合\t{np.mean(param_scores):.4f}\t{np.std(param_scores):.4f}\t"
                          f"{np.median(param_scores):.4f}\t{np.min(param_scores):.4f}\t{np.max(param_scores):.4f}")
        stats_lines.append(f"中介支持\t{np.mean(med_scores):.4f}\t{np.std(med_scores):.4f}\t"
                          f"{np.median(med_scores):.4f}\t{np.min(med_scores):.4f}\t{np.max(med_scores):.4f}")
        stats_lines.append(f"专家定向\t{np.mean(expert_scores):.4f}\t{np.std(expert_scores):.4f}\t"
                          f"{np.median(expert_scores):.4f}\t{np.min(expert_scores):.4f}\t{np.max(expert_scores):.4f}")
        stats_lines.append("")
        
        # 核心边统计
        if core_edges:
            core_confidences = [self.confidence_scores[edge]['confidence'] for edge in core_edges]
            core_qualities = [self.confidence_scores[edge]['quality_score'] for edge in core_edges]
            
            stats_lines.append("【核心边专项统计】")
            stats_lines.append(f"核心边平均置信度: {np.mean(core_confidences):.4f}")
            stats_lines.append(f"核心边平均质量: {np.mean(core_qualities):.4f}")
            stats_lines.append(f"核心边置信度范围: [{np.min(core_confidences):.4f}, {np.max(core_confidences):.4f}]")
            
            # 核心边四维评分
            core_struct = [self.structural_scores[edge]['score'] for edge in core_edges]
            core_param = [self.parameter_scores[edge]['score'] for edge in core_edges]
            core_med = [self.mediation_scores[edge]['score'] for edge in core_edges]
            core_expert = [self.expert_scores[edge]['score'] for edge in core_edges]
            
            stats_lines.append("核心边四维评分平均值:")
            stats_lines.append(f"  结构一致性: {np.mean(core_struct):.4f}")
            stats_lines.append(f"  参数拟合: {np.mean(core_param):.4f}")
            stats_lines.append(f"  中介支持: {np.mean(core_med):.4f}")
            stats_lines.append(f"  专家定向: {np.mean(core_expert):.4f}")
            stats_lines.append("")
        
        # 维度贡献度统计
        struct_contribs = [info['struct_contrib'] for info in self.confidence_scores.values()]
        param_contribs = [info['param_contrib'] for info in self.confidence_scores.values()]
        med_contribs = [info['med_contrib'] for info in self.confidence_scores.values()]
        expert_contribs = [info['expert_contrib'] for info in self.confidence_scores.values()]
        
        stats_lines.append("【维度贡献度统计】")
        stats_lines.append(f"结构一致性平均贡献: {np.mean(struct_contribs):.4f}")
        stats_lines.append(f"参数拟合平均贡献: {np.mean(param_contribs):.4f}")
        stats_lines.append(f"中介支持平均贡献: {np.mean(med_contribs):.4f}")
        stats_lines.append(f"专家定向平均贡献: {np.mean(expert_contribs):.4f}")
        stats_lines.append("")
        
        # 相关性分析
        stats_lines.append("【维度间相关性分析】")
        correlation_matrix = np.corrcoef([struct_scores, param_scores, med_scores, expert_scores])
        dimensions = ['结构一致性', '参数拟合', '中介支持', '专家定向']
        
        for i, dim1 in enumerate(dimensions):
            for j, dim2 in enumerate(dimensions):
                if i < j:  # 只显示上三角
                    corr = correlation_matrix[i, j]
                    stats_lines.append(f"{dim1} vs {dim2}: {corr:.4f}")
        stats_lines.append("")
        
        # 质量分层统计
        stats_lines.append("【质量分层统计】")
        excellent_edges = sum(1 for edge in self.confidence_scores 
                            if self.confidence_scores[edge]['confidence'] >= 0.8 
                            and self.confidence_scores[edge]['quality_score'] >= 0.8)
        good_edges = sum(1 for edge in self.confidence_scores 
                        if 0.7 <= self.confidence_scores[edge]['confidence'] < 0.8 
                        and self.confidence_scores[edge]['quality_score'] >= 0.7)
        fair_edges = sum(1 for edge in self.confidence_scores 
                        if 0.6 <= self.confidence_scores[edge]['confidence'] < 0.7 
                        and self.confidence_scores[edge]['quality_score'] >= 0.6)
        
        stats_lines.append(f"优秀边(置信度≥0.8, 质量≥0.8): {excellent_edges}")
        stats_lines.append(f"良好边(置信度0.7-0.8, 质量≥0.7): {good_edges}")
        stats_lines.append(f"一般边(置信度0.6-0.7, 质量≥0.6): {fair_edges}")
        stats_lines.append("")
        
        # 保存统计摘要
        stats_file = os.path.join(self.output_dir, "统计摘要报告.txt")
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(stats_lines))
        print(f"✓ 统计摘要报告已保存: {stats_file}")
        
        # 2. 保存CSV格式的统计数据
        stats_data = {
            '统计项目': [],
            '数值': [],
            '说明': []
        }
        
        # 添加基本统计数据
        stats_data['统计项目'].extend(['总边数', '核心边数', '核心边比例'])
        stats_data['数值'].extend([total_edges, core_edges_count, f"{core_rate:.2f}%"])
        stats_data['说明'].extend(['候选因果边总数', '通过筛选的核心边数', '核心边占总边数的比例'])
        
        # 添加置信度统计
        stats_data['统计项目'].extend(['平均置信度', '置信度标准差', '置信度中位数'])
        stats_data['数值'].extend([f"{np.mean(confidences):.4f}", 
                                 f"{np.std(confidences):.4f}", 
                                 f"{np.median(confidences):.4f}"])
        stats_data['说明'].extend(['所有边的平均置信度', '置信度的标准差', '置信度的中位数'])
        
        # 添加四维评分统计
        for dim_name, scores in [('结构一致性', struct_scores), ('参数拟合', param_scores), 
                               ('中介支持', med_scores), ('专家定向', expert_scores)]:
            stats_data['统计项目'].extend([f'{dim_name}_平均', f'{dim_name}_标准差'])
            stats_data['数值'].extend([f"{np.mean(scores):.4f}", f"{np.std(scores):.4f}"])
            stats_data['说明'].extend([f'{dim_name}维度的平均评分', f'{dim_name}维度评分的标准差'])
        
        stats_df = pd.DataFrame(stats_data)
        stats_csv_file = os.path.join(self.output_dir, "统计数据表.csv")
        stats_df.to_csv(stats_csv_file, index=False, encoding='utf-8')
        print(f"✓ 统计数据表已保存: {stats_csv_file}")
        
        return {
            'total_edges': total_edges,
            'core_edges_count': core_edges_count,
            'core_rate': core_rate,
            'avg_confidence': np.mean(confidences),
            'avg_quality': np.mean(qualities)
        }
    
    def generate_summary_report(self, core_edges):
        print("\n=== 生成汇总报告 ===")
        
        total_edges = len(self.structural_scores)
        core_count = len(core_edges)
        
        # 统计各维度数据覆盖情况
        real_param_count = sum([1 for info in self.parameter_scores.values() if info['has_real_data']])
        real_med_count = sum([1 for info in self.mediation_scores.values() if info['has_real_data']])
        expert_count = sum([1 for info in self.expert_scores.values() if info['has_expert_support']])
        
        # 置信度分布统计
        confidences = [info['confidence'] for info in self.confidence_scores.values()]
        
        report = f"""
# 因果发现三角验证汇总报告

## 分析概览
- 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 总候选边数: {total_edges}
- 核心因果边数: {core_count} ({core_count/total_edges*100:.1f}%)
- 筛选标准: 置信度≥0.6 且 质量≥0.6

## 四维评分体系数据覆盖
- 结构一致性: {total_edges}/{total_edges} (100.0%) - 基于因果发现阶段
- 参数拟合: {real_param_count}/{total_edges} ({real_param_count/total_edges*100:.1f}%) - 基于参数学习阶段
- 中介支持: {real_med_count}/{total_edges} ({real_med_count/total_edges*100:.1f}%) - 基于中介分析阶段
- 专家定向: {expert_count}/{total_edges} ({expert_count/total_edges*100:.1f}%) - 基于专家方法

## 置信度分布统计
- 平均置信度: {np.mean(confidences):.4f}
- 置信度标准差: {np.std(confidences):.4f}
- 最高置信度: {np.max(confidences):.4f}
- 最低置信度: {np.min(confidences):.4f}
- 中位数置信度: {np.median(confidences):.4f}

## 质量评估
- 高质量数据边数: {sum([1 for info in self.confidence_scores.values() if info['quality_score'] >= 0.8])}
- 中等质量数据边数: {sum([1 for info in self.confidence_scores.values() if 0.6 <= info['quality_score'] < 0.8])}
- 低质量数据边数: {sum([1 for info in self.confidence_scores.values() if info['quality_score'] < 0.6])}

## 核心发现
1. 通过四维评分体系成功整合了多阶段分析结果
2. 识别出{core_count}条高置信度且高质量的核心因果边
3. 建立了完整的证据三角测量框架
4. 为后续因果推断和临床应用提供了可靠证据

## 文件输出
- 详细结果: 三角验证详细结果.csv
- 核心边集合: 核心因果边集合.csv
"""
        
        # 保存报告
        report_file = os.path.join(self.output_dir, "三角验证汇总报告.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ 汇总报告已保存: {report_file}")
        return report
    
    def run_triangulation_analysis(self):
        """运行完整的三角验证分析"""
        print("=" * 60)
        print("因果发现结果三角验证分析")
        print("=" * 60)
        
        # 1. 加载各阶段数据
        if not self.load_structural_data():
            return False
        if not self.load_parameter_data():
            return False
        if not self.load_mediation_data():
            return False
        
        # 2. 计算四维评分
        self.calculate_structural_consistency_score()
        self.calculate_parameter_fitting_score()
        self.calculate_mediation_support_score()
        self.calculate_expert_orientation_score()
        
        # 3. 计算联合置信度
        self.calculate_triangulation_confidence()
        
        # 4. 识别核心因果边
        core_edges = self.identify_core_causal_edges()
        
        # 5. 保存结果
        detailed_df = self.save_detailed_results()
        core_results = self.save_core_edges(core_edges)
        
        # 6. 保存结构化数据
        structured_data = self.save_structured_data(core_edges)
        
        # 7. 保存边级详细分析报告
        self.save_detailed_edge_reports(core_edges)
        
        # 8. 保存统计摘要文件
        stats_summary = self.save_statistical_summary(core_edges)
        
        # 9. 生成可视化图表
        self.save_visualizations(core_edges)
        
        # 10. 保存网络图可视化
        self.save_network_visualizations(core_edges)
        
        # 11. 生成报告
        report = self.generate_summary_report(core_edges)
        
        print("\n" + "=" * 60)
        print("三角验证分析完成!")
        print(f"结果保存在: {self.output_dir}")
        print("=" * 60)
        
        return True

def main():
    """主函数"""
    # 创建三角验证实例
    triangulation = EvidenceTriangulation()
    
    # 运行分析
    success = triangulation.run_triangulation_analysis()
    
    if success:
        print("\n✓ 三角验证分析成功完成!")
    else:
        print("\n✗ 三角验证分析失败!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())