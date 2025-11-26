#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级因果边筛选算法 - 专业版
Advanced Causal Edge Selection Algorithm - Professional Edition

基于多维度评估和机器学习的智能因果边筛选系统
Multi-dimensional Assessment and Machine Learning-based Intelligent Causal Edge Selection System

核心特性：
1. 算法一致性评分 - 所有算法权重相等
2. 完整网络拓扑评分 - 包含度中心性、介数中心性、接近中心性
3. 统计显著性评分 - 基于频次分布的统计检验
4. 自适应阈值选择 - 肘部法则、聚类分析、异常检测
5. 质量分层管理 - 铂金、黄金、白银、青铜四级质量体系

作者: 因果发现系统
版本: Professional Edition v2.0
日期: 2025年
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import networkx as nx
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings('ignore')

class ProfessionalCausalEdgeSelector:
    """专业级因果边筛选器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.network_graph = None
        
    def load_algorithm_results(self):
        """直接加载所有算法的因果边结果"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 定义算法结果文件路径
        algorithm_configs = [
            {
                "name": "PC算法",
                "folder": "01PC算法结果",
                "csv_file": "PC_因果边列表.csv"
            },
            {
                "name": "爬山算法",
                "folder": "02爬山算法结果",
                "csv_file": "HillClimbing_AIC-D_因果边列表.csv"
            },
            {
                "name": "贪婪等价搜索",
                "folder": "03贪婪等价搜索结果",
                "csv_file": "GreedyEquivalence_AIC-D_因果边列表.csv"
            },
            {
                "name": "树搜索",
                "folder": "04树搜索结果",
                "csv_file": "TAN_因果边列表.csv"
            },
            {
                "name": "专家在循环",
                "folder": "05专家在循环结果",
                "csv_file": "ExpertInLoop_因果边列表.csv"
            }
        ]
        
        all_edges = []
        algorithm_stats = {}
        
        print("正在加载各算法的因果边结果...")
        
        for config in algorithm_configs:
            try:
                # 构建文件路径
                folder_path = os.path.join(script_dir, config["folder"])
                csv_path = os.path.join(folder_path, config["csv_file"])
                
                if os.path.exists(csv_path):
                    # 加载CSV文件
                    df = pd.read_csv(csv_path, encoding='utf-8-sig')
                    
                    # 标准化列名
                    if len(df.columns) >= 2:
                        df.columns = ['源节点', '目标节点']
                        
                        # 添加算法标识
                        df['算法'] = config["name"]
                        
                        # 创建边的唯一标识
                        df['边标识'] = df['源节点'] + ' -> ' + df['目标节点']
                        
                        all_edges.append(df)
                        
                        # 记录统计信息
                        algorithm_stats[config["name"]] = {
                            "边数量": len(df),
                            "文件路径": csv_path,
                            "加载状态": "成功"
                        }
                        
                        print(f"✓ {config['name']}: 加载 {len(df)} 条因果边")
                    else:
                        print(f"⚠ {config['name']}: CSV文件格式不正确")
                        algorithm_stats[config["name"]] = {
                            "边数量": 0,
                            "文件路径": csv_path,
                            "加载状态": "格式错误"
                        }
                else:
                    print(f"⚠ {config['name']}: 文件不存在 - {csv_path}")
                    algorithm_stats[config["name"]] = {
                        "边数量": 0,
                        "文件路径": csv_path,
                        "加载状态": "文件不存在"
                    }
                    
            except Exception as e:
                print(f"❌ {config['name']}: 加载失败 - {str(e)}")
                algorithm_stats[config["name"]] = {
                    "边数量": 0,
                    "文件路径": "N/A",
                    "加载状态": f"错误: {str(e)}"
                }
        
        return all_edges, algorithm_stats
    
    def merge_and_analyze_edges(self, all_edges):
        """合并和分析因果边"""
        if not all_edges:
            print("❌ 没有找到任何因果边数据")
            return None
        
        # 合并所有边
        combined_df = pd.concat(all_edges, ignore_index=True)
        
        print(f"\n总共收集到 {len(combined_df)} 条因果边（包含重复）")
        
        # 统计每条边出现的频次
        edge_counts = combined_df['边标识'].value_counts()
        
        # 创建详细的边分析
        edge_analysis = []
        
        for edge_id, count in edge_counts.items():
            # 获取该边的所有记录
            edge_records = combined_df[combined_df['边标识'] == edge_id]
            
            # 提取源节点和目标节点
            source = edge_records.iloc[0]['源节点']
            target = edge_records.iloc[0]['目标节点']
            
            # 获取支持该边的算法
            supporting_algorithms = edge_records['算法'].tolist()
            
            # 计算综合评分（基于出现频次和算法多样性）
            frequency_score = count / len(all_edges)  # 频次评分
            diversity_score = len(set(supporting_algorithms)) / 5  # 算法多样性评分（最多5种算法）
            comprehensive_score = (frequency_score * 0.6 + diversity_score * 0.4) * 100
            
            edge_analysis.append({
                '边标识': edge_id,
                '源节点': source,
                '目标节点': target,
                '出现频次': count,
                '支持算法数量': len(set(supporting_algorithms)),
                '支持算法': ', '.join(set(supporting_algorithms)),
                '频次评分': frequency_score,
                '多样性评分': diversity_score,
                '综合评分': comprehensive_score
            })
        
        # 转换为DataFrame并排序
        edge_df = pd.DataFrame(edge_analysis)
        edge_df = edge_df.sort_values('综合评分', ascending=False).reset_index(drop=True)
        
        print(f"发现 {len(edge_df)} 条唯一因果边")
        
        return edge_df
    
    def calculate_advanced_scores(self, edge_df):
        """计算高级评分指标"""
        
        # 1. 算法一致性评分（所有算法权重相等）
        algorithm_weights = {
            "PC算法": 1.0,
            "爬山算法": 1.0,
            "贪婪等价搜索": 1.0,
            "树搜索": 1.0,
            "专家在循环": 1.0
        }
        
        consistency_scores = []
        for _, row in edge_df.iterrows():
            supporting_algs = row['支持算法'].split(', ')
            weighted_support = sum(algorithm_weights.get(alg, 1.0) for alg in supporting_algs)
            max_possible = sum(algorithm_weights.values())
            consistency_scores.append(weighted_support / max_possible)
        
        edge_df['算法一致性评分'] = consistency_scores
        
        # 2. 网络拓扑评分（完整版）
        topology_scores = self._calculate_comprehensive_topology_scores(edge_df)
        edge_df['网络拓扑评分'] = topology_scores
        
        # 3. 统计显著性评分（基于频次的统计检验）
        significance_scores = self._calculate_significance_scores(edge_df)
        edge_df['统计显著性评分'] = significance_scores
        
        return edge_df
    
    def _calculate_comprehensive_topology_scores(self, edge_df):
        """计算完整的网络拓扑评分"""
        # 构建网络图
        G = nx.DiGraph()
        
        for _, row in edge_df.iterrows():
            G.add_edge(row['源节点'], row['目标节点'])
        
        self.network_graph = G
        
        # 计算各种中心性指标
        try:
            # 度中心性
            degree_centrality = nx.degree_centrality(G)
            
            # 介数中心性
            betweenness_centrality = nx.betweenness_centrality(G)
            
            # 接近中心性
            closeness_centrality = nx.closeness_centrality(G)
            
            # PageRank中心性
            pagerank_centrality = nx.pagerank(G, alpha=0.85)
            
            # 特征向量中心性（对于强连通图）
            try:
                eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
            except:
                eigenvector_centrality = {node: 0.5 for node in G.nodes()}
            
        except Exception as e:
            print(f"网络拓扑计算警告: {e}")
            # 如果计算失败，使用默认值
            nodes = list(G.nodes())
            degree_centrality = {node: 0.5 for node in nodes}
            betweenness_centrality = {node: 0.5 for node in nodes}
            closeness_centrality = {node: 0.5 for node in nodes}
            pagerank_centrality = {node: 0.5 for node in nodes}
            eigenvector_centrality = {node: 0.5 for node in nodes}
        
        topology_scores = []
        
        for _, row in edge_df.iterrows():
            source, target = row['源节点'], row['目标节点']
            
            # 获取各种中心性指标
            source_degree = degree_centrality.get(source, 0)
            target_degree = degree_centrality.get(target, 0)
            
            source_betweenness = betweenness_centrality.get(source, 0)
            target_betweenness = betweenness_centrality.get(target, 0)
            
            source_closeness = closeness_centrality.get(source, 0)
            target_closeness = closeness_centrality.get(target, 0)
            
            source_pagerank = pagerank_centrality.get(source, 0)
            target_pagerank = pagerank_centrality.get(target, 0)
            
            source_eigenvector = eigenvector_centrality.get(source, 0)
            target_eigenvector = eigenvector_centrality.get(target, 0)
            
            # 计算综合拓扑评分
            # 权重分配：度中心性(0.25) + 介数中心性(0.25) + 接近中心性(0.2) + PageRank(0.2) + 特征向量(0.1)
            topology_score = (
                (source_degree + target_degree) / 2 * 0.25 +
                (source_betweenness + target_betweenness) / 2 * 0.25 +
                (source_closeness + target_closeness) / 2 * 0.2 +
                (source_pagerank + target_pagerank) / 2 * 0.2 +
                (source_eigenvector + target_eigenvector) / 2 * 0.1
            )
            
            topology_scores.append(min(topology_score, 1.0))
        
        return topology_scores
    
    def _calculate_significance_scores(self, edge_df):
        """计算统计显著性评分"""
        # 基于频次分布的Z-score
        frequencies = edge_df['出现频次'].values
        mean_freq = np.mean(frequencies)
        std_freq = np.std(frequencies) if np.std(frequencies) > 0 else 1
        
        significance_scores = []
        for freq in frequencies:
            z_score = (freq - mean_freq) / std_freq
            # 将Z-score转换为0-1范围的评分
            significance = 1 / (1 + np.exp(-z_score))  # Sigmoid函数
            significance_scores.append(significance)
        
        return significance_scores
    
    def ensemble_scoring(self, edge_df):
        """集成评分（移除医学合理性评分）"""
        # 权重配置（重新分配权重）
        weights = {
            '频次评分': 0.30,           # 增加频次权重
            '多样性评分': 0.25,         # 增加多样性权重
            '算法一致性评分': 0.25,     # 增加一致性权重
            '网络拓扑评分': 0.20,       # 网络拓扑权重
        }
        
        # 计算加权平均
        edge_df['集成评分'] = (
            edge_df['频次评分'] * weights['频次评分'] +
            edge_df['多样性评分'] * weights['多样性评分'] +
            edge_df['算法一致性评分'] * weights['算法一致性评分'] +
            edge_df['网络拓扑评分'] * weights['网络拓扑评分']
        )
        
        # 按集成评分排序
        edge_df = edge_df.sort_values('集成评分', ascending=False).reset_index(drop=True)
        
        return edge_df
    
    def adaptive_threshold_selection(self, edge_df):
        """自适应阈值选择"""
        scores = edge_df['集成评分'].values
        
        # 方法1: 肘部法则
        elbow_threshold = 0.6  # 默认值
        try:
            # 简化的肘部检测
            sorted_scores = np.sort(scores)[::-1]
            diffs = np.diff(sorted_scores)
            if len(diffs) > 0:
                elbow_idx = np.argmax(diffs) + 1
                elbow_threshold = sorted_scores[elbow_idx] if elbow_idx < len(sorted_scores) else 0.6
        except:
            pass
        
        # 方法2: 聚类分析
        cluster_threshold = 0.65  # 默认值
        try:
            if len(scores) > 5:
                scores_reshaped = scores.reshape(-1, 1)
                scaler = StandardScaler()
                scores_scaled = scaler.fit_transform(scores_reshaped)
                
                dbscan = DBSCAN(eps=0.3, min_samples=2)
                clusters = dbscan.fit_predict(scores_scaled)
                
                if len(set(clusters)) > 1:
                    # 找到最高质量的聚类
                    cluster_means = {}
                    for cluster_id in set(clusters):
                        if cluster_id != -1:  # 排除噪声点
                            cluster_scores = scores[clusters == cluster_id]
                            cluster_means[cluster_id] = np.mean(cluster_scores)
                    
                    if cluster_means:
                        best_cluster = max(cluster_means.keys(), key=lambda x: cluster_means[x])
                        cluster_threshold = np.min(scores[clusters == best_cluster])
        except:
            pass
        
        # 方法3: 异常检测
        anomaly_threshold = 0.7  # 默认值
        try:
            if len(scores) > 10:
                isolation_forest = IsolationForest(contamination=0.3, random_state=42)
                scores_reshaped = scores.reshape(-1, 1)
                outliers = isolation_forest.fit_predict(scores_reshaped)
                
                # 正常点的最低分数作为阈值
                normal_scores = scores[outliers == 1]
                if len(normal_scores) > 0:
                    anomaly_threshold = np.min(normal_scores)
        except:
            pass
        
        # 综合阈值
        final_threshold = np.mean([elbow_threshold, cluster_threshold, anomaly_threshold])
        
        return {
            "肘部法则阈值": float(elbow_threshold),
            "聚类分析阈值": float(cluster_threshold),
            "异常检测阈值": float(anomaly_threshold),
            "最终阈值": float(final_threshold)
        }
    
    def quality_based_selection(self, edge_df, quality_tiers=None):
        """基于质量分层的选择"""
        if quality_tiers is None:
            quality_tiers = {
                "platinum": {"min_score": 0.8, "min_algorithms": 4, "min_frequency": 3},
                "gold": {"min_score": 0.65, "min_algorithms": 3, "min_frequency": 2},
                "silver": {"min_score": 0.5, "min_algorithms": 2, "min_frequency": 2},
                "bronze": {"min_score": 0.3, "min_algorithms": 1, "min_frequency": 1}
            }
        
        selected_edges = []
        
        for _, row in edge_df.iterrows():
            score = row['集成评分']
            algorithms = row['支持算法数量']
            frequency = row['出现频次']
            
            quality_level = None
            
            # 按质量等级从高到低检查
            for level, criteria in quality_tiers.items():
                if (score >= criteria["min_score"] and 
                    algorithms >= criteria["min_algorithms"] and 
                    frequency >= criteria["min_frequency"]):
                    quality_level = level
                    break
            
            if quality_level:
                row_dict = row.to_dict()
                row_dict['质量等级'] = quality_level
                selected_edges.append(row_dict)
        
        return pd.DataFrame(selected_edges) if selected_edges else pd.DataFrame()

def run_professional_causal_edge_selection():
    """运行专业级因果边筛选算法"""
    print("=" * 80)
    print("高级因果边筛选算法 - 专业版 - Professional Edition")
    print("Advanced Causal Edge Selection Algorithm - Professional Edition")
    print("=" * 80)
    
    # 1. 初始化筛选器
    selector = ProfessionalCausalEdgeSelector()
    
    # 2. 直接加载算法结果
    all_edges, algorithm_stats = selector.load_algorithm_results()
    
    if not all_edges:
        print("❌ 没有找到任何算法结果，请先运行01-05算法脚本")
        return None
    
    # 3. 合并和分析边
    edge_df = selector.merge_and_analyze_edges(all_edges)
    
    if edge_df is None:
        print("❌ 边分析失败")
        return None
    
    # 4. 计算高级评分
    print("\n正在计算高级评分指标...")
    edge_df = selector.calculate_advanced_scores(edge_df)
    
    # 5. 集成评分
    print("正在计算集成评分...")
    edge_df = selector.ensemble_scoring(edge_df)
    
    # 6. 自适应阈值选择
    print("正在进行自适应阈值选择...")
    thresholds = selector.adaptive_threshold_selection(edge_df)
    
    # 7. 基于质量分层的选择
    print("正在进行质量分层选择...")
    selected_edges = selector.quality_based_selection(edge_df)
    
    # 8. 保存结果（使用专业命名）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "06候选因果边集合")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存完整评分结果
    comprehensive_results_file = '因果边综合评分结果.csv'
    edge_df.to_csv(os.path.join(output_dir, comprehensive_results_file), 
                   index=False, encoding='utf-8-sig')
    
    # 保存筛选结果
    if len(selected_edges) > 0:
        # 高质量因果边候选集
        high_quality_edges_file = '高质量因果边候选集.csv'
        selected_edges.to_csv(os.path.join(output_dir, high_quality_edges_file), 
                             index=False, encoding='utf-8-sig')
        
        # 精简版因果边列表
        simplified_edges = selected_edges[['源节点', '目标节点', '质量等级', '集成评分', '支持算法数量']].copy()
        simplified_edges_file = '精简因果边列表.csv'
        simplified_edges.to_csv(os.path.join(output_dir, simplified_edges_file), 
                               index=False, encoding='utf-8-sig')
    
    # 保存分析报告
    report = {
        "筛选信息": {
            "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "算法版本": "Professional Edition v2.0",
            "输入边数量": len(edge_df),
            "筛选后边数量": len(selected_edges),
            "筛选率": f"{len(selected_edges)/len(edge_df)*100:.2f}%"
        },
        "算法统计": algorithm_stats,
        "阈值分析": thresholds,
        "质量等级分布": {
            level: len(selected_edges[selected_edges['质量等级'] == level])
            for level in ['platinum', 'gold', 'silver', 'bronze']
            if level in selected_edges['质量等级'].values
        } if len(selected_edges) > 0 else {},
        "集成评分统计": {
            "最高分": float(edge_df['集成评分'].max()),
            "最低分": float(edge_df['集成评分'].min()),
            "平均分": float(edge_df['集成评分'].mean()),
            "标准差": float(edge_df['集成评分'].std())
        },
        "网络拓扑统计": {
            "节点数量": selector.network_graph.number_of_nodes() if selector.network_graph else 0,
            "边数量": selector.network_graph.number_of_edges() if selector.network_graph else 0,
            "平均度": float(np.mean([d for n, d in selector.network_graph.degree()])) if selector.network_graph and selector.network_graph.nodes() else 0
        }
    }
    # 保存分析报告
    analysis_report_file = '因果边筛选分析报告.json'
    with open(os.path.join(output_dir, analysis_report_file), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 9. 输出结果摘要
    print("\n" + "=" * 60)
    print("筛选结果摘要 | Selection Results Summary")
    print("=" * 60)
    print(f"输入边数量: {len(edge_df)}")
    print(f"筛选后边数量: {len(selected_edges)}")
    print(f"筛选率: {len(selected_edges)/len(edge_df)*100:.2f}%")
    
    if len(selected_edges) > 0:
        print("\n质量等级分布:")
        quality_dist = selected_edges['质量等级'].value_counts()
        for level in ['platinum', 'gold', 'silver', 'bronze']:
            if level in quality_dist.index:
                print(f"  {level.capitalize()}: {quality_dist[level]} 条边")
        
        print(f"\n前5条高质量因果边:")
        top_edges = selected_edges.head(5)
        for idx, row in top_edges.iterrows():
            print(f"  {idx+1}. {row['边标识']} (评分: {row['集成评分']:.3f}, 等级: {row['质量等级']})")
    
    print(f"\n生成文件:")
    print(f"  - 完整评分结果: {comprehensive_results_file}")
    if len(selected_edges) > 0:
        print(f"  - 高质量候选边: {high_quality_edges_file}")
        print(f"  - 精简边列表: {simplified_edges_file}")
    print(f"  - 分析报告: {analysis_report_file}")
    
    print("\n✅ 专业级因果边筛选完成!")
    
    return {
        "edge_df": edge_df,
        "selected_edges": selected_edges,
        "thresholds": thresholds,
        "report": report
    }

if __name__ == "__main__":
    run_professional_causal_edge_selection()