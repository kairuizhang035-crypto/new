#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因果发现结果可视化脚本
对所有因果发现算法的结果进行综合可视化分析

作者: 因果发现系统
日期: 2025年
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
# 设置中文字体
import matplotlib
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = [
    'SimHei', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 
    'Noto Sans CJK SC', 'Source Han Sans SC', 'Microsoft YaHei',
    'DejaVu Sans', 'Arial Unicode MS', 'Liberation Sans'
]
matplotlib.rcParams['axes.unicode_minus'] = False

class CausalDiscoveryVisualizer:
    """因果发现结果可视化器"""
    
    def __init__(self, base_dir):
        """初始化可视化器"""
        self.base_dir = base_dir
        self.output_dir = os.path.join(base_dir, "可视化")
        self.algorithms = {
            "PC算法": "01PC算法结果",
            "爬山算法": "02爬山算法结果", 
            "贪婪等价搜索": "03贪婪等价搜索结果",
            "树搜索": "04树搜索结果",
            "专家在循环": "05专家在循环结果"
        }
        self.results_data = {}
        self.edge_data = {}
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_all_results(self):
        """加载所有算法结果"""
        print("🔄 正在加载所有算法结果...")
        
        for alg_name, folder_name in self.algorithms.items():
            folder_path = os.path.join(self.base_dir, folder_name)
            
            if not os.path.exists(folder_path):
                print(f"⚠️  {alg_name} 结果文件夹不存在: {folder_path}")
                continue
                
            # 查找JSON结果文件
            json_files = [f for f in os.listdir(folder_path) if f.endswith('_因果结果.json')]
            
            if not json_files:
                print(f"⚠️  {alg_name} 未找到结果JSON文件")
                continue
                
            json_file = os.path.join(folder_path, json_files[0])
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.results_data[alg_name] = data
                    
                    # 提取边数据 - 处理不同的数据结构
                    edges = []
                    
                    # 方式1: 标准结构 - 网络结构.因果边列表
                    if '网络结构' in data and '因果边列表' in data['网络结构']:
                        for edge in data['网络结构']['因果边列表']:
                            if '源节点' in edge and '目标节点' in edge:
                                edges.append((edge['源节点'], edge['目标节点']))
                    
                    # 方式2: 专家在循环结构 - 直接edges字段
                    elif 'edges' in data:
                        for edge in data['edges']:
                            if isinstance(edge, list) and len(edge) >= 2:
                                edges.append((edge[0], edge[1]))
                    
                    # 方式3: 其他可能的结构
                    elif 'causal_edges' in data:
                        for edge in data['causal_edges']:
                            if isinstance(edge, dict) and '源节点' in edge and '目标节点' in edge:
                                edges.append((edge['源节点'], edge['目标节点']))
                            elif isinstance(edge, list) and len(edge) >= 2:
                                edges.append((edge[0], edge[1]))
                    
                    self.edge_data[alg_name] = edges
                        
                print(f"✅ {alg_name}: 加载成功 ({len(self.edge_data.get(alg_name, []))} 条边)")
                
            except Exception as e:
                print(f"❌ {alg_name} 加载失败: {str(e)}")
                
        print(f"📊 总共加载了 {len(self.results_data)} 个算法的结果")
        
    def create_algorithm_comparison(self):
        """创建算法对比图表"""
        print("📊 创建算法对比图表...")
        
        # 准备对比数据
        comparison_data = []
        
        for alg_name, data in self.results_data.items():
            # 处理不同的数据结构
            nodes_count = 0
            edges_count = 0
            max_in_degree = 0
            max_out_degree = 0
            avg_degree = 0
            
            # 标准结构
            if '网络结构' in data and '统计信息' in data:
                network = data['网络结构']
                stats = data['统计信息']
                
                nodes_count = network.get('节点总数', 0)
                edges_count = network.get('边总数', 0)
                max_in_degree = stats.get('最大入度', 0)
                max_out_degree = stats.get('最大出度', 0)
                avg_degree = round(stats.get('平均度数', 0), 2)
            
            # 专家在循环结构
            elif 'nodes_count' in data and 'edges_count' in data:
                nodes_count = data.get('nodes_count', 0)
                edges_count = data.get('edges_count', 0)
                
                # 从边数据计算度数统计
                if alg_name in self.edge_data:
                    edges = self.edge_data[alg_name]
                    in_degrees = defaultdict(int)
                    out_degrees = defaultdict(int)
                    
                    for source, target in edges:
                        out_degrees[source] += 1
                        in_degrees[target] += 1
                    
                    if in_degrees:
                        max_in_degree = max(in_degrees.values())
                    if out_degrees:
                        max_out_degree = max(out_degrees.values())
                    
                    all_degrees = list(in_degrees.values()) + list(out_degrees.values())
                    if all_degrees:
                        avg_degree = round(sum(all_degrees) / len(all_degrees), 2)
            
            comparison_data.append({
                '算法': alg_name,
                '节点数': nodes_count,
                '边数': edges_count,
                '最大入度': max_in_degree,
                '最大出度': max_out_degree,
                '平均度数': avg_degree
            })
        
        if not comparison_data:
            print("⚠️  没有可用的对比数据")
            return
            
        df = pd.DataFrame(comparison_data)
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('因果发现算法对比分析', fontsize=16, fontweight='bold')
        
        # 1. 边数对比
        axes[0, 0].bar(df['算法'], df['边数'], color='skyblue', alpha=0.7)
        axes[0, 0].set_title('各算法发现的边数对比')
        axes[0, 0].set_ylabel('边数')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 最大入度对比
        axes[0, 1].bar(df['算法'], df['最大入度'], color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('最大入度对比')
        axes[0, 1].set_ylabel('最大入度')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 最大出度对比
        axes[0, 2].bar(df['算法'], df['最大出度'], color='salmon', alpha=0.7)
        axes[0, 2].set_title('最大出度对比')
        axes[0, 2].set_ylabel('最大出度')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. 平均度数对比
        axes[1, 0].bar(df['算法'], df['平均度数'], color='gold', alpha=0.7)
        axes[1, 0].set_title('平均度数对比')
        axes[1, 0].set_ylabel('平均度数')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. 综合雷达图
        categories = ['边数', '最大入度', '最大出度', '平均度数']
        
        # 标准化数据用于雷达图
        normalized_data = df[categories].copy()
        for col in categories:
            max_val = normalized_data[col].max()
            if max_val > 0:
                normalized_data[col] = normalized_data[col] / max_val
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        ax_radar = plt.subplot(2, 3, 5, projection='polar')
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, (_, row) in enumerate(normalized_data.iterrows()):
            values = row.tolist()
            values += values[:1]  # 闭合
            
            ax_radar.plot(angles, values, 'o-', linewidth=2, 
                         label=df.iloc[i]['算法'], color=colors[i % len(colors)])
            ax_radar.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories)
        ax_radar.set_title('算法性能雷达图')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 6. 数据表格
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        table = axes[1, 2].table(cellText=df.values, colLabels=df.columns,
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        axes[1, 2].set_title('详细数据表')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '算法对比分析.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存对比数据
        df.to_csv(os.path.join(self.output_dir, '算法对比数据.csv'), 
                 index=False, encoding='utf-8')
        
        print("✅ 算法对比图表创建完成")
        
    def create_edge_overlap_analysis(self):
        """创建边重叠分析"""
        print("🔗 创建边重叠分析...")
        
        if len(self.edge_data) < 2:
            print("⚠️  需要至少2个算法的结果才能进行重叠分析")
            return
            
        # 计算边的重叠情况
        all_edges = set()
        for edges in self.edge_data.values():
            all_edges.update(edges)
            
        edge_counts = defaultdict(int)
        edge_algorithms = defaultdict(list)
        
        for alg_name, edges in self.edge_data.items():
            for edge in edges:
                edge_counts[edge] += 1
                edge_algorithms[edge].append(alg_name)
        
        # 创建重叠矩阵
        algorithms = list(self.edge_data.keys())
        n_algs = len(algorithms)
        overlap_matrix = np.zeros((n_algs, n_algs))
        
        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms):
                if i == j:
                    overlap_matrix[i, j] = len(self.edge_data[alg1])
                else:
                    common_edges = set(self.edge_data[alg1]) & set(self.edge_data[alg2])
                    overlap_matrix[i, j] = len(common_edges)
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('因果边重叠分析', fontsize=16, fontweight='bold')
        
        # 1. 重叠热力图
        sns.heatmap(overlap_matrix, annot=True, fmt='.0f', 
                   xticklabels=algorithms, yticklabels=algorithms,
                   cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('算法间边重叠数量热力图')
        
        # 2. 边支持度分布
        support_counts = Counter(edge_counts.values())
        support_levels = list(support_counts.keys())
        support_nums = list(support_counts.values())
        
        axes[0, 1].bar(support_levels, support_nums, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('边支持度分布')
        axes[0, 1].set_xlabel('支持算法数量')
        axes[0, 1].set_ylabel('边数量')
        
        # 3. 高置信度边（被多个算法支持）
        high_confidence_edges = [(edge, count) for edge, count in edge_counts.items() 
                               if count >= max(2, len(algorithms) // 2)]
        
        if high_confidence_edges:
            high_conf_df = pd.DataFrame(high_confidence_edges, 
                                      columns=['边', '支持算法数'])
            high_conf_df['边标签'] = high_conf_df['边'].apply(
                lambda x: f"{x[0][:10]}...→{x[1][:10]}..." if len(x[0]) > 10 or len(x[1]) > 10 
                else f"{x[0]}→{x[1]}")
            
            axes[1, 0].barh(range(len(high_conf_df)), high_conf_df['支持算法数'], 
                           color='green', alpha=0.7)
            axes[1, 0].set_yticks(range(len(high_conf_df)))
            axes[1, 0].set_yticklabels(high_conf_df['边标签'], fontsize=8)
            axes[1, 0].set_title(f'高置信度边 (≥{max(2, len(algorithms) // 2)}个算法支持)')
            axes[1, 0].set_xlabel('支持算法数量')
        else:
            axes[1, 0].text(0.5, 0.5, '没有高置信度边', ha='center', va='center',
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_title('高置信度边')
        
        # 4. 算法独特性分析
        unique_edges = {}
        for alg_name, edges in self.edge_data.items():
            unique = set(edges)
            for other_alg, other_edges in self.edge_data.items():
                if other_alg != alg_name:
                    unique -= set(other_edges)
            unique_edges[alg_name] = len(unique)
        
        axes[1, 1].bar(unique_edges.keys(), unique_edges.values(), 
                      color='orange', alpha=0.7)
        axes[1, 1].set_title('各算法独特边数量')
        axes[1, 1].set_ylabel('独特边数')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '边重叠分析.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存详细的重叠数据
        overlap_df = pd.DataFrame(overlap_matrix, 
                                 index=algorithms, columns=algorithms)
        overlap_df.to_csv(os.path.join(self.output_dir, '算法重叠矩阵.csv'), 
                         encoding='utf-8')
        
        # 保存高置信度边
        if high_confidence_edges:
            high_conf_detailed = []
            for edge, count in high_confidence_edges:
                supporting_algs = edge_algorithms[edge]
                high_conf_detailed.append({
                    '源节点': edge[0],
                    '目标节点': edge[1],
                    '支持算法数': count,
                    '支持算法': ', '.join(supporting_algs)
                })
            
            pd.DataFrame(high_conf_detailed).to_csv(
                os.path.join(self.output_dir, '高置信度边详情.csv'), 
                index=False, encoding='utf-8')
        
        print("✅ 边重叠分析完成")
        
    def create_network_topology_analysis(self):
        """创建网络拓扑分析"""
        print("🕸️ 创建网络拓扑分析...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('网络拓扑结构分析', fontsize=16, fontweight='bold')
        
        for idx, (alg_name, data) in enumerate(self.results_data.items()):
            if idx >= 6:  # 最多显示6个算法
                break
                
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            if '网络结构' in data and '因果边列表' in data['网络结构']:
                # 创建网络图
                G = nx.DiGraph()
                
                # 添加节点
                if '节点列表' in data['网络结构']:
                    G.add_nodes_from(data['网络结构']['节点列表'])
                
                # 添加边
                edges = data['网络结构']['因果边列表']
                for edge in edges:
                    if '源节点' in edge and '目标节点' in edge:
                        G.add_edge(edge['源节点'], edge['目标节点'])
                
                if len(G.nodes()) > 0:
                    # 计算布局
                    try:
                        pos = nx.spring_layout(G, k=1, iterations=50)
                    except:
                        pos = nx.random_layout(G)
                    
                    # 绘制网络
                    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', 
                                         node_size=100, alpha=0.7)
                    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', 
                                         arrows=True, arrowsize=10, alpha=0.6)
                    
                    # 只为重要节点添加标签（度数较高的节点）
                    degrees = dict(G.degree())
                    important_nodes = {node: pos[node] for node, degree in degrees.items() 
                                     if degree >= np.percentile(list(degrees.values()), 75)}
                    
                    if important_nodes:
                        # 简化节点标签
                        simplified_labels = {}
                        for node in important_nodes:
                            if len(node) > 8:
                                simplified_labels[node] = node[:8] + "..."
                            else:
                                simplified_labels[node] = node
                        
                        nx.draw_networkx_labels(G, important_nodes, simplified_labels, 
                                              ax=ax, font_size=6)
                
            ax.set_title(f'{alg_name}\n节点:{len(G.nodes())}, 边:{len(G.edges())}', 
                        fontsize=10)
            ax.axis('off')
        
        # 如果算法少于6个，隐藏多余的子图
        for idx in range(len(self.results_data), 6):
            row = idx // 3
            col = idx % 3
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '网络拓扑结构.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ 网络拓扑分析完成")
        
    def create_node_analysis(self):
        """创建节点分析"""
        print("🔍 创建节点分析...")
        
        # 收集所有节点的统计信息
        all_nodes = set()
        node_stats = defaultdict(lambda: {
            'total_appearances': 0,
            'in_degree_sum': 0,
            'out_degree_sum': 0,
            'algorithms': []
        })
        
        for alg_name, data in self.results_data.items():
            if '网络结构' in data and '统计信息' in data:
                nodes = data['网络结构'].get('节点列表', [])
                in_degrees = data['统计信息'].get('入度统计', {})
                out_degrees = data['统计信息'].get('出度统计', {})
                
                all_nodes.update(nodes)
                
                for node in nodes:
                    node_stats[node]['total_appearances'] += 1
                    node_stats[node]['in_degree_sum'] += in_degrees.get(node, 0)
                    node_stats[node]['out_degree_sum'] += out_degrees.get(node, 0)
                    node_stats[node]['algorithms'].append(alg_name)
        
        # 按节点类型分类
        node_types = {
            '疾病': [],
            '药物': [],
            '检验': [],
            '其他': []
        }
        
        for node in all_nodes:
            if node.startswith('疾病_'):
                node_types['疾病'].append(node)
            elif node.startswith('药物_'):
                node_types['药物'].append(node)
            elif node.startswith('检验_'):
                node_types['检验'].append(node)
            else:
                node_types['其他'].append(node)
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('节点分析', fontsize=16, fontweight='bold')
        
        # 1. 节点类型分布
        type_counts = {k: len(v) for k, v in node_types.items() if v}
        axes[0, 0].pie(type_counts.values(), labels=type_counts.keys(), 
                      autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('节点类型分布')
        
        # 2. 最活跃节点（出现次数最多）
        most_active = sorted(node_stats.items(), 
                           key=lambda x: x[1]['total_appearances'], reverse=True)[:10]
        
        if most_active:
            nodes, stats = zip(*most_active)
            appearances = [s['total_appearances'] for s in stats]
            
            # 简化节点名称用于显示
            simplified_nodes = [node.replace('疾病_', '').replace('药物_', '').replace('检验_', '')[:10] 
                              for node in nodes]
            
            axes[0, 1].barh(range(len(simplified_nodes)), appearances, color='green', alpha=0.7)
            axes[0, 1].set_yticks(range(len(simplified_nodes)))
            axes[0, 1].set_yticklabels(simplified_nodes, fontsize=8)
            axes[0, 1].set_title('最活跃节点 (出现次数)')
            axes[0, 1].set_xlabel('出现次数')
        
        # 3. 最高入度节点
        highest_in_degree = sorted(node_stats.items(), 
                                 key=lambda x: x[1]['in_degree_sum'], reverse=True)[:10]
        
        if highest_in_degree:
            nodes, stats = zip(*highest_in_degree)
            in_degrees = [s['in_degree_sum'] for s in stats]
            
            simplified_nodes = [node.replace('疾病_', '').replace('药物_', '').replace('检验_', '')[:10] 
                              for node in nodes]
            
            axes[1, 0].barh(range(len(simplified_nodes)), in_degrees, color='blue', alpha=0.7)
            axes[1, 0].set_yticks(range(len(simplified_nodes)))
            axes[1, 0].set_yticklabels(simplified_nodes, fontsize=8)
            axes[1, 0].set_title('最高入度节点')
            axes[1, 0].set_xlabel('总入度')
        
        # 4. 最高出度节点
        highest_out_degree = sorted(node_stats.items(), 
                                  key=lambda x: x[1]['out_degree_sum'], reverse=True)[:10]
        
        if highest_out_degree:
            nodes, stats = zip(*highest_out_degree)
            out_degrees = [s['out_degree_sum'] for s in stats]
            
            simplified_nodes = [node.replace('疾病_', '').replace('药物_', '').replace('检验_', '')[:10] 
                              for node in nodes]
            
            axes[1, 1].barh(range(len(simplified_nodes)), out_degrees, color='red', alpha=0.7)
            axes[1, 1].set_yticks(range(len(simplified_nodes)))
            axes[1, 1].set_yticklabels(simplified_nodes, fontsize=8)
            axes[1, 1].set_title('最高出度节点')
            axes[1, 1].set_xlabel('总出度')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '节点分析.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存节点统计数据
        node_analysis_data = []
        for node, stats in node_stats.items():
            node_analysis_data.append({
                '节点': node,
                '节点类型': '疾病' if node.startswith('疾病_') else 
                          '药物' if node.startswith('药物_') else 
                          '检验' if node.startswith('检验_') else '其他',
                '出现次数': stats['total_appearances'],
                '总入度': stats['in_degree_sum'],
                '总出度': stats['out_degree_sum'],
                '支持算法': ', '.join(stats['algorithms'])
            })
        
        pd.DataFrame(node_analysis_data).to_csv(
            os.path.join(self.output_dir, '节点统计分析.csv'), 
            index=False, encoding='utf-8')
        
        print("✅ 节点分析完成")
        
    def create_interactive_network(self):
        """创建交互式网络图"""
        print("🌐 创建交互式网络图...")
        
        # 合并所有算法的边，计算权重
        all_edges = defaultdict(int)
        for alg_name, edges in self.edge_data.items():
            for edge in edges:
                all_edges[edge] += 1
        
        if not all_edges:
            print("⚠️  没有可用的边数据")
            return
        
        # 创建网络图
        G = nx.DiGraph()
        
        # 添加边（权重为支持算法数量）
        for (source, target), weight in all_edges.items():
            G.add_edge(source, target, weight=weight)
        
        # 计算布局
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # 准备Plotly数据
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            weight = G[edge[0]][edge[1]]['weight']
            edge_info.append(f"{edge[0]} → {edge[1]}<br>支持算法数: {weight}")
        
        # 创建边的轨迹
        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                               line=dict(width=0.5, color='#888'),
                               hoverinfo='none',
                               mode='lines')
        
        # 准备节点数据
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        node_colors = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # 节点颜色根据类型
            if node.startswith('疾病_'):
                color = 'red'
            elif node.startswith('药物_'):
                color = 'blue'
            elif node.startswith('检验_'):
                color = 'green'
            else:
                color = 'gray'
            node_colors.append(color)
            
            # 节点信息
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            node_text.append(node.replace('疾病_', '').replace('药物_', '').replace('检验_', ''))
            node_info.append(f"{node}<br>入度: {in_degree}<br>出度: {out_degree}")
        
        # 创建节点的轨迹
        node_trace = go.Scatter(x=node_x, y=node_y,
                               mode='markers+text',
                               hoverinfo='text',
                               text=node_text,
                               textposition="middle center",
                               hovertext=node_info,
                               marker=dict(showscale=True,
                                         colorscale='YlOrRd',
                                         reversescale=True,
                                         color=node_colors,
                                         size=10,
                                         colorbar=dict(
                                             thickness=15,
                                             len=0.5,
                                             x=1.02,
                                             title="节点类型"
                                         ),
                                         line=dict(width=2)))
        
        # 创建图形
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='因果发现综合网络图',
                           title_font_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="节点颜色: 红色=疾病, 蓝色=药物, 绿色=检验",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        
        # 保存交互式图形
        pyo.plot(fig, filename=os.path.join(self.output_dir, '交互式网络图.html'), 
                auto_open=False)
        
        print("✅ 交互式网络图创建完成")
        
    def create_summary_report(self):
        """创建总结报告"""
        print("📋 创建总结报告...")
        
        report_lines = []
        report_lines.append("# 因果发现结果可视化分析报告")
        report_lines.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 算法概览
        report_lines.append("## 算法执行概览")
        report_lines.append(f"- 成功执行的算法数量: {len(self.results_data)}")
        report_lines.append(f"- 算法列表: {', '.join(self.results_data.keys())}")
        report_lines.append("")
        
        # 网络统计
        report_lines.append("## 网络结构统计")
        for alg_name, data in self.results_data.items():
            if '网络结构' in data:
                network = data['网络结构']
                report_lines.append(f"### {alg_name}")
                report_lines.append(f"- 节点数: {network.get('节点总数', 0)}")
                report_lines.append(f"- 边数: {network.get('边总数', 0)}")
                
                if '统计信息' in data:
                    stats = data['统计信息']
                    report_lines.append(f"- 最大入度: {stats.get('最大入度', 0)}")
                    report_lines.append(f"- 最大出度: {stats.get('最大出度', 0)}")
                    report_lines.append(f"- 平均度数: {stats.get('平均度数', 0):.2f}")
                report_lines.append("")
        
        # 边重叠分析
        if len(self.edge_data) >= 2:
            report_lines.append("## 边重叠分析")
            all_edges = set()
            for edges in self.edge_data.values():
                all_edges.update(edges)
            
            edge_counts = defaultdict(int)
            for edges in self.edge_data.values():
                for edge in edges:
                    edge_counts[edge] += 1
            
            high_confidence = sum(1 for count in edge_counts.values() 
                                if count >= max(2, len(self.edge_data) // 2))
            
            report_lines.append(f"- 总边数（去重）: {len(all_edges)}")
            report_lines.append(f"- 高置信度边数: {high_confidence}")
            report_lines.append(f"- 平均边重叠度: {np.mean(list(edge_counts.values())):.2f}")
            report_lines.append("")
        
        # 生成的文件列表
        report_lines.append("## 生成的可视化文件")
        output_files = [
            "算法对比分析.png - 各算法性能对比图表",
            "边重叠分析.png - 算法间边重叠分析",
            "网络拓扑结构.png - 各算法网络拓扑图",
            "节点分析.png - 节点统计分析图表",
            "交互式网络图.html - 可交互的综合网络图",
            "算法对比数据.csv - 详细对比数据",
            "算法重叠矩阵.csv - 边重叠矩阵数据",
            "高置信度边详情.csv - 高置信度边详细信息",
            "节点统计分析.csv - 节点统计数据"
        ]
        
        for file_desc in output_files:
            report_lines.append(f"- {file_desc}")
        
        report_lines.append("")
        report_lines.append("## 使用说明")
        report_lines.append("1. PNG图片文件可直接查看分析结果")
        report_lines.append("2. HTML文件需要在浏览器中打开查看交互式图表")
        report_lines.append("3. CSV文件包含详细数据，可用于进一步分析")
        
        # 保存报告
        with open(os.path.join(self.output_dir, '可视化分析报告.md'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print("✅ 总结报告创建完成")
        
    def run_all_visualizations(self):
        """运行所有可视化分析"""
        print("🚀 开始因果发现结果可视化分析...")
        print(f"📂 输出目录: {self.output_dir}")
        
        try:
            # 加载数据
            self.load_all_results()
            
            if not self.results_data:
                print("❌ 没有找到可用的结果数据")
                return False
            
            # 执行各种可视化分析
            self.create_algorithm_comparison()
            self.create_edge_overlap_analysis()
            self.create_network_topology_analysis()
            self.create_node_analysis()
            self.create_interactive_network()
            self.create_summary_report()
            
            print("🎉 所有可视化分析完成！")
            print(f"📁 结果保存在: {self.output_dir}")
            
            return True
            
        except Exception as e:
            print(f"❌ 可视化分析过程中出现错误: {str(e)}")
            return False

def main():
    """主函数"""
    print("=" * 80)
    print(" 因果发现结果可视化分析 ")
    print("=" * 80)
    
    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建可视化器
    visualizer = CausalDiscoveryVisualizer(script_dir)
    
    # 运行可视化分析
    success = visualizer.run_all_visualizations()
    
    if success:
        print("\n✅ 可视化脚本执行成功！")
        return True
    else:
        print("\n❌ 可视化脚本执行失败！")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  用户中断执行")
        sys.exit(2)
    except Exception as e:
        print(f"\n❌ 程序异常: {str(e)}")
        sys.exit(3)