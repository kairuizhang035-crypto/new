#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
贝叶斯中介分析可视化脚本
对中介路径分析和贝叶斯中介分析结果进行丰富的可视化分析
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import networkx as nx
from datetime import datetime
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

class MediationAnalysisVisualizer:
    """贝叶斯中介分析可视化器"""
    
    def __init__(self, base_dir=None):
        """
        初始化可视化器
        
        Args:
            base_dir: 基础目录路径，默认为脚本所在目录
        """
        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.base_dir = base_dir
            
        self.mediation_paths_file = os.path.join(self.base_dir, "01中介路径分析结果", "完整中介路径结果.csv")
        self.bayesian_results_file = os.path.join(self.base_dir, "02贝叶斯中介分析结果", "贝叶斯中介分析汇总.csv")
        self.output_dir = os.path.join(self.base_dir, "可视化")
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 数据存储
        self.mediation_paths = None
        self.bayesian_results = None
        
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        
        # 加载中介路径数据
        if os.path.exists(self.mediation_paths_file):
            self.mediation_paths = pd.read_csv(self.mediation_paths_file)
            print(f"✓ 成功加载中介路径数据: {len(self.mediation_paths)} 条路径")
        else:
            print(f"✗ 中介路径文件不存在: {self.mediation_paths_file}")
            
        # 加载贝叶斯分析结果
        if os.path.exists(self.bayesian_results_file):
            self.bayesian_results = pd.read_csv(self.bayesian_results_file)
            print(f"✓ 成功加载贝叶斯分析结果: {len(self.bayesian_results)} 条结果")
        else:
            print(f"✗ 贝叶斯分析结果文件不存在: {self.bayesian_results_file}")
    
    def create_mediation_network_graph(self):
        """创建中介路径网络图"""
        if self.mediation_paths is None:
            print("中介路径数据未加载，跳过网络图生成")
            return
            
        print("正在生成中介路径网络图...")
        
        # 创建网络图
        G = nx.DiGraph()
        
        # 添加节点和边
        for _, row in self.mediation_paths.iterrows():
            start_node = row['起始节点']
            mediator = row['中介变量']
            end_node = row['终点节点']
            
            # 添加边
            G.add_edge(start_node, mediator, path_id=row['路径ID'])
            G.add_edge(mediator, end_node, path_id=row['路径ID'])
        
        # 计算布局
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 创建matplotlib图
        plt.figure(figsize=(20, 16))
        
        # 绘制节点
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            if '疾病_' in node:
                node_colors.append('#FF6B6B')  # 红色 - 疾病
                node_sizes.append(1000)
            elif '药物_' in node:
                node_colors.append('#4ECDC4')  # 青色 - 药物
                node_sizes.append(800)
            elif '检验_' in node:
                node_colors.append('#45B7D1')  # 蓝色 - 检验
                node_sizes.append(600)
            else:
                node_colors.append('#96CEB4')  # 绿色 - 其他
                node_sizes.append(400)
        
        # 绘制网络
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.6, arrows=True, arrowsize=20)
        
        # 添加标签（只显示简化的节点名）
        labels = {node: node.split('_')[-1] if '_' in node else node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
        
        plt.title('中介路径网络图', fontsize=20, fontweight='bold', pad=20)
        plt.axis('off')
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', markersize=15, label='疾病'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4', markersize=15, label='药物'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#45B7D1', markersize=15, label='检验'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#96CEB4', markersize=15, label='其他')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '中介路径网络图.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 中介路径网络图已保存")
    
    def create_mediation_statistics(self):
        """创建中介路径统计分析"""
        if self.mediation_paths is None:
            print("中介路径数据未加载，跳过统计分析")
            return
            
        print("正在生成中介路径统计分析...")
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('中介路径统计分析', fontsize=16, fontweight='bold')
        
        # 1. 节点类型分布
        node_types = {'疾病': 0, '药物': 0, '检验': 0, '其他': 0}
        all_nodes = set()
        
        for _, row in self.mediation_paths.iterrows():
            all_nodes.add(row['起始节点'])
            all_nodes.add(row['中介变量'])
            all_nodes.add(row['终点节点'])
        
        for node in all_nodes:
            if '疾病_' in node:
                node_types['疾病'] += 1
            elif '药物_' in node:
                node_types['药物'] += 1
            elif '检验_' in node:
                node_types['检验'] += 1
            else:
                node_types['其他'] += 1
        
        axes[0, 0].pie(node_types.values(), labels=node_types.keys(), autopct='%1.1f%%', 
                       colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[0, 0].set_title('节点类型分布')
        
        # 2. 中介变量频次分析
        mediator_counts = self.mediation_paths['中介变量'].value_counts().head(10)
        axes[0, 1].barh(range(len(mediator_counts)), mediator_counts.values, 
                        color='skyblue')
        axes[0, 1].set_yticks(range(len(mediator_counts)))
        axes[0, 1].set_yticklabels([name.split('_')[-1] if '_' in name else name 
                                   for name in mediator_counts.index], fontsize=10)
        axes[0, 1].set_title('Top 10 中介变量频次')
        axes[0, 1].set_xlabel('频次')
        
        # 3. 起始节点分析
        start_counts = self.mediation_paths['起始节点'].value_counts().head(10)
        axes[1, 0].bar(range(len(start_counts)), start_counts.values, 
                       color='lightcoral')
        axes[1, 0].set_xticks(range(len(start_counts)))
        axes[1, 0].set_xticklabels([name.split('_')[-1] if '_' in name else name 
                                   for name in start_counts.index], 
                                  rotation=45, ha='right', fontsize=10)
        axes[1, 0].set_title('Top 10 起始节点频次')
        axes[1, 0].set_ylabel('频次')
        
        # 4. 终点节点分析
        end_counts = self.mediation_paths['终点节点'].value_counts().head(10)
        axes[1, 1].bar(range(len(end_counts)), end_counts.values, 
                       color='lightgreen')
        axes[1, 1].set_xticks(range(len(end_counts)))
        axes[1, 1].set_xticklabels([name.split('_')[-1] if '_' in name else name 
                                   for name in end_counts.index], 
                                  rotation=45, ha='right', fontsize=10)
        axes[1, 1].set_title('Top 10 终点节点频次')
        axes[1, 1].set_ylabel('频次')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '中介路径统计分析.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 中介路径统计分析已保存")
    
    def create_bayesian_effects_analysis(self):
        """创建贝叶斯效应分析图"""
        if self.bayesian_results is None:
            print("贝叶斯分析结果未加载，跳过效应分析")
            return
            
        print("正在生成贝叶斯效应分析图...")
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('贝叶斯中介效应分析', fontsize=16, fontweight='bold')
        
        # 1. 间接效应分布
        axes[0, 0].hist(self.bayesian_results['间接效应均值'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.8)
        axes[0, 0].set_title('间接效应分布')
        axes[0, 0].set_xlabel('间接效应均值')
        axes[0, 0].set_ylabel('频次')
        
        # 2. 直接效应分布
        axes[0, 1].hist(self.bayesian_results['直接效应均值'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.8)
        axes[0, 1].set_title('直接效应分布')
        axes[0, 1].set_xlabel('直接效应均值')
        axes[0, 1].set_ylabel('频次')
        
        # 3. 总效应分布
        axes[0, 2].hist(self.bayesian_results['总效应均值'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 2].axvline(0, color='red', linestyle='--', alpha=0.8)
        axes[0, 2].set_title('总效应分布')
        axes[0, 2].set_xlabel('总效应均值')
        axes[0, 2].set_ylabel('频次')
        
        # 4. 显著性分析
        significance_counts = self.bayesian_results['是否显著'].value_counts()
        axes[1, 0].pie(significance_counts.values, labels=significance_counts.index, 
                       autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'])
        axes[1, 0].set_title('显著性分布')
        
        # 5. 中介比例分析
        # 过滤极端值
        mediation_ratios = self.bayesian_results['中介比例']
        filtered_ratios = mediation_ratios[(mediation_ratios >= -10) & (mediation_ratios <= 10)]
        axes[1, 1].hist(filtered_ratios, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_title('中介比例分布（过滤极端值）')
        axes[1, 1].set_xlabel('中介比例')
        axes[1, 1].set_ylabel('频次')
        
        # 6. 显著性概率分布
        axes[1, 2].hist(self.bayesian_results['显著性概率'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 2].axvline(0.95, color='red', linestyle='--', alpha=0.8, label='95%阈值')
        axes[1, 2].set_title('显著性概率分布')
        axes[1, 2].set_xlabel('显著性概率')
        axes[1, 2].set_ylabel('频次')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '贝叶斯效应分析.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 贝叶斯效应分析图已保存")
    
    def create_significant_pathways_analysis(self):
        """创建显著中介路径分析"""
        if self.bayesian_results is None:
            print("贝叶斯分析结果未加载，跳过显著路径分析")
            return
            
        print("正在生成显著中介路径分析...")
        
        # 筛选显著路径
        significant_paths = self.bayesian_results[self.bayesian_results['是否显著'] == '是'].copy()
        
        if len(significant_paths) == 0:
            print("没有发现显著的中介路径")
            return
        
        # 按间接效应绝对值排序
        significant_paths['间接效应绝对值'] = significant_paths['间接效应均值'].abs()
        significant_paths = significant_paths.sort_values('间接效应绝对值', ascending=False)
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'显著中介路径分析 (共{len(significant_paths)}条)', fontsize=16, fontweight='bold')
        
        # 1. Top 15 显著路径效应大小
        top_paths = significant_paths.head(15)
        y_pos = np.arange(len(top_paths))
        
        colors = ['red' if x < 0 else 'blue' for x in top_paths['间接效应均值']]
        bars = axes[0, 0].barh(y_pos, top_paths['间接效应均值'], color=colors, alpha=0.7)
        axes[0, 0].set_yticks(y_pos)
        axes[0, 0].set_yticklabels([f"路径{pid}" for pid in top_paths['路径ID']], fontsize=10)
        axes[0, 0].set_title('Top 15 显著路径间接效应')
        axes[0, 0].set_xlabel('间接效应均值')
        axes[0, 0].axvline(0, color='black', linestyle='-', alpha=0.3)
        
        # 2. 效应类型散点图
        axes[0, 1].scatter(significant_paths['间接效应均值'], significant_paths['直接效应均值'], 
                          c=significant_paths['显著性概率'], cmap='viridis', alpha=0.7, s=60)
        axes[0, 1].axhline(0, color='black', linestyle='--', alpha=0.3)
        axes[0, 1].axvline(0, color='black', linestyle='--', alpha=0.3)
        axes[0, 1].set_xlabel('间接效应均值')
        axes[0, 1].set_ylabel('直接效应均值')
        axes[0, 1].set_title('间接效应 vs 直接效应')
        
        # 添加颜色条
        cbar = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
        cbar.set_label('显著性概率')
        
        # 3. 中介比例分析
        # 过滤极端值
        filtered_ratios = significant_paths['中介比例']
        filtered_ratios = filtered_ratios[(filtered_ratios >= -5) & (filtered_ratios <= 5)]
        
        axes[1, 0].hist(filtered_ratios, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_title('显著路径中介比例分布')
        axes[1, 0].set_xlabel('中介比例')
        axes[1, 0].set_ylabel('频次')
        
        # 4. 显著性概率分布
        axes[1, 1].hist(significant_paths['显著性概率'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].axvline(0.95, color='red', linestyle='--', alpha=0.8, label='95%阈值')
        axes[1, 1].set_title('显著路径概率分布')
        axes[1, 1].set_xlabel('显著性概率')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '显著中介路径分析.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 显著中介路径分析已保存")
        
        # 保存显著路径详细信息
        significant_paths_output = significant_paths[['路径ID', '路径描述', '间接效应均值', 
                                                    '直接效应均值', '总效应均值', '中介比例', 
                                                    '显著性概率']].copy()
        significant_paths_output.to_csv(os.path.join(self.output_dir, '显著中介路径详细.csv'), 
                                      index=False, encoding='utf-8-sig')
        print(f"✓ 显著中介路径详细信息已保存")
    
    def create_interactive_dashboard(self):
        """创建交互式仪表板"""
        if self.bayesian_results is None:
            print("贝叶斯分析结果未加载，跳过交互式仪表板")
            return
            
        print("正在生成交互式仪表板...")
        
        # 创建子图
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('间接效应 vs 直接效应', '效应大小分布', 
                          '显著性概率分布', '中介比例分析',
                          '路径效应热力图', '综合效应分析'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"colspan": 2}, None]],
            vertical_spacing=0.08
        )
        
        # 1. 间接效应 vs 直接效应散点图
        fig.add_trace(
            go.Scatter(
                x=self.bayesian_results['间接效应均值'],
                y=self.bayesian_results['直接效应均值'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=self.bayesian_results['显著性概率'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="显著性概率")
                ),
                text=self.bayesian_results['路径描述'],
                hovertemplate='<b>%{text}</b><br>' +
                            '间接效应: %{x:.4f}<br>' +
                            '直接效应: %{y:.4f}<br>' +
                            '<extra></extra>',
                name='路径'
            ),
            row=1, col=1
        )
        
        # 2. 效应大小分布
        fig.add_trace(
            go.Histogram(
                x=self.bayesian_results['间接效应均值'],
                name='间接效应',
                opacity=0.7,
                nbinsx=30
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=self.bayesian_results['直接效应均值'],
                name='直接效应',
                opacity=0.7,
                nbinsx=30
            ),
            row=1, col=2
        )
        
        # 3. 显著性概率分布
        fig.add_trace(
            go.Histogram(
                x=self.bayesian_results['显著性概率'],
                name='显著性概率',
                marker_color='orange',
                opacity=0.7,
                nbinsx=20
            ),
            row=2, col=1
        )
        
        # 4. 中介比例分析（过滤极端值）
        filtered_ratios = self.bayesian_results['中介比例']
        filtered_ratios = filtered_ratios[(filtered_ratios >= -10) & (filtered_ratios <= 10)]
        
        fig.add_trace(
            go.Histogram(
                x=filtered_ratios,
                name='中介比例',
                marker_color='green',
                opacity=0.7,
                nbinsx=25
            ),
            row=2, col=2
        )
        
        # 5. 综合效应分析
        significant_paths = self.bayesian_results[self.bayesian_results['是否显著'] == '是']
        top_significant = significant_paths.nlargest(10, '间接效应均值')
        
        fig.add_trace(
            go.Bar(
                x=[f"路径{pid}" for pid in top_significant['路径ID']],
                y=top_significant['间接效应均值'],
                name='Top 10 显著间接效应',
                marker_color='red',
                text=top_significant['间接效应均值'].round(4),
                textposition='auto'
            ),
            row=3, col=1
        )
        
        # 更新布局
        fig.update_layout(
            height=1200,
            title_text="贝叶斯中介分析交互式仪表板",
            title_x=0.5,
            showlegend=True
        )
        
        # 更新坐标轴标签
        fig.update_xaxes(title_text="间接效应均值", row=1, col=1)
        fig.update_yaxes(title_text="直接效应均值", row=1, col=1)
        fig.update_xaxes(title_text="效应值", row=1, col=2)
        fig.update_yaxes(title_text="频次", row=1, col=2)
        fig.update_xaxes(title_text="显著性概率", row=2, col=1)
        fig.update_yaxes(title_text="频次", row=2, col=1)
        fig.update_xaxes(title_text="中介比例", row=2, col=2)
        fig.update_yaxes(title_text="频次", row=2, col=2)
        fig.update_xaxes(title_text="路径ID", row=3, col=1)
        fig.update_yaxes(title_text="间接效应均值", row=3, col=1)
        
        # 保存交互式图表
        output_file = os.path.join(self.output_dir, '交互式中介分析仪表板.html')
        pyo.plot(fig, filename=output_file, auto_open=False)
        
        print(f"✓ 交互式仪表板已保存")
    
    def create_pathway_network_interactive(self):
        """创建交互式路径网络图"""
        if self.mediation_paths is None or self.bayesian_results is None:
            print("数据未完全加载，跳过交互式网络图")
            return
            
        print("正在生成交互式路径网络图...")
        
        # 合并数据
        merged_data = self.mediation_paths.merge(
            self.bayesian_results[['路径ID', '间接效应均值', '是否显著', '显著性概率']], 
            on='路径ID', how='left'
        )
        
        # 创建网络图
        G = nx.DiGraph()
        
        # 添加节点和边，包含效应信息
        for _, row in merged_data.iterrows():
            start_node = row['起始节点']
            mediator = row['中介变量']
            end_node = row['终点节点']
            
            # 添加节点属性
            for node in [start_node, mediator, end_node]:
                if node not in G.nodes():
                    node_type = 'disease' if '疾病_' in node else 'drug' if '药物_' in node else 'test' if '检验_' in node else 'other'
                    G.add_node(node, type=node_type)
            
            # 添加边属性
            effect = row.get('间接效应均值', 0)
            significant = row.get('是否显著', '否')
            
            G.add_edge(start_node, mediator, 
                      path_id=row['路径ID'], 
                      effect=effect,
                      significant=significant)
            G.add_edge(mediator, end_node, 
                      path_id=row['路径ID'], 
                      effect=effect,
                      significant=significant)
        
        # 计算布局
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # 准备边数据
        significant_x = []
        significant_y = []
        normal_x = []
        normal_y = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # 根据显著性分类边
            if edge[2].get('significant') == '是':
                significant_x.extend([x0, x1, None])
                significant_y.extend([y0, y1, None])
            else:
                normal_x.extend([x0, x1, None])
                normal_y.extend([y0, y1, None])
        
        # 创建显著边trace
        significant_edge_trace = go.Scatter(
            x=significant_x,
            y=significant_y, 
            mode='lines',
            line=dict(width=3, color='red'),
            hoverinfo='none',
            name='显著路径'
        )
        
        # 创建普通边trace  
        normal_edge_trace = go.Scatter(
            x=normal_x,
            y=normal_y,
            mode='lines', 
            line=dict(width=1, color='gray'),
            hoverinfo='none',
            name='普通路径'
        )
        
        # 准备节点数据
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        # 添加节点
        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            
            # 设置节点颜色和大小
            node_type = node[1].get('type', 'other')
            if node_type == 'disease':
                node_colors.append('red')
                node_sizes.append(20)
            elif node_type == 'drug':
                node_colors.append('blue')
                node_sizes.append(15)
            elif node_type == 'test':
                node_colors.append('green')
                node_sizes.append(12)
            else:
                node_colors.append('gray')
                node_sizes.append(10)
            
            # 简化节点标签
            label = node[0].split('_')[-1] if '_' in node[0] else node[0]
            node_text.append(label)
        
        # 创建节点trace
        node_trace = go.Scatter(
            x=node_x, 
            y=node_y, 
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(size=node_sizes, color=node_colors, line=dict(width=2))
        )
        
        # 创建图表
        traces = [normal_edge_trace, significant_edge_trace, node_trace]
        fig = go.Figure(data=traces,
                       layout=go.Layout(
                           title=dict(text='交互式中介路径网络图', font=dict(size=16)),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="红色边表示显著路径，节点颜色：红色=疾病，蓝色=药物，绿色=检验",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        # 保存交互式网络图
        output_file = os.path.join(self.output_dir, '交互式路径网络图.html')
        pyo.plot(fig, filename=output_file, auto_open=False)
        
        print(f"✓ 交互式路径网络图已保存")
    
    def generate_summary_report(self):
        """生成汇总报告"""
        print("正在生成汇总报告...")
        
        report_lines = []
        report_lines.append("# 贝叶斯中介分析可视化报告")
        report_lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("\n" + "="*60)
        
        # 中介路径分析
        if self.mediation_paths is not None:
            report_lines.append(f"\n## 中介路径分析")
            report_lines.append(f"- 总路径数量: {len(self.mediation_paths)}")
            
            # 节点统计
            all_nodes = set()
            for _, row in self.mediation_paths.iterrows():
                all_nodes.add(row['起始节点'])
                all_nodes.add(row['中介变量'])
                all_nodes.add(row['终点节点'])
            
            node_types = {'疾病': 0, '药物': 0, '检验': 0, '其他': 0}
            for node in all_nodes:
                if '疾病_' in node:
                    node_types['疾病'] += 1
                elif '药物_' in node:
                    node_types['药物'] += 1
                elif '检验_' in node:
                    node_types['检验'] += 1
                else:
                    node_types['其他'] += 1
            
            report_lines.append(f"- 总节点数量: {len(all_nodes)}")
            for node_type, count in node_types.items():
                report_lines.append(f"  - {node_type}节点: {count}")
            
            # Top中介变量
            top_mediators = self.mediation_paths['中介变量'].value_counts().head(5)
            report_lines.append(f"\n### Top 5 中介变量:")
            for mediator, count in top_mediators.items():
                report_lines.append(f"- {mediator}: {count}次")
        
        # 贝叶斯分析结果
        if self.bayesian_results is not None:
            report_lines.append(f"\n## 贝叶斯中介分析结果")
            report_lines.append(f"- 总分析路径: {len(self.bayesian_results)}")
            
            # 显著性统计
            significant_count = len(self.bayesian_results[self.bayesian_results['是否显著'] == '是'])
            report_lines.append(f"- 显著路径数量: {significant_count}")
            report_lines.append(f"- 显著率: {significant_count/len(self.bayesian_results)*100:.1f}%")
            
            # 效应统计
            report_lines.append(f"\n### 效应统计:")
            report_lines.append(f"- 间接效应均值: {self.bayesian_results['间接效应均值'].mean():.4f}")
            report_lines.append(f"- 直接效应均值: {self.bayesian_results['直接效应均值'].mean():.4f}")
            report_lines.append(f"- 总效应均值: {self.bayesian_results['总效应均值'].mean():.4f}")
            
            # 显著路径Top 5
            if significant_count > 0:
                significant_paths = self.bayesian_results[self.bayesian_results['是否显著'] == '是']
                top_significant = significant_paths.nlargest(5, '间接效应均值')
                
                report_lines.append(f"\n### Top 5 显著间接效应路径:")
                for _, row in top_significant.iterrows():
                    report_lines.append(f"- 路径{row['路径ID']}: {row['间接效应均值']:.4f}")
                    report_lines.append(f"  {row['路径描述']}")
        
        # 生成的文件列表
        report_lines.append(f"\n## 生成的可视化文件")
        output_files = [
            "中介路径网络图.png",
            "中介路径统计分析.png", 
            "贝叶斯效应分析.png",
            "显著中介路径分析.png",
            "交互式中介分析仪表板.html",
            "交互式路径网络图.html",
            "显著中介路径详细.csv"
        ]
        
        for file_name in output_files:
            file_path = os.path.join(self.output_dir, file_name)
            if os.path.exists(file_path):
                report_lines.append(f"✓ {file_name}")
            else:
                report_lines.append(f"✗ {file_name}")
        
        report_lines.append(f"\n## 分析建议")
        if self.bayesian_results is not None and significant_count > 0:
            report_lines.append("- 重点关注显著的中介路径，这些路径在因果关系中起重要作用")
            report_lines.append("- 分析间接效应和直接效应的相对大小，了解中介机制的重要性")
            report_lines.append("- 查看交互式图表以获得更深入的洞察")
        else:
            report_lines.append("- 当前分析中显著路径较少，可能需要调整分析参数或收集更多数据")
        
        # 保存报告
        report_content = "\n".join(report_lines)
        report_file = os.path.join(self.output_dir, "贝叶斯中介分析可视化报告.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✓ 汇总报告已保存")
    
    def run_all_visualizations(self):
        """运行所有可视化分析"""
        print("="*60)
        print("开始贝叶斯中介分析可视化")
        print("="*60)
        
        # 加载数据
        self.load_data()
        
        if self.mediation_paths is None and self.bayesian_results is None:
            print("✗ 没有可用的数据文件，无法进行可视化")
            return False
        
        # 执行各种可视化
        try:
            self.create_mediation_network_graph()
            self.create_mediation_statistics()
            self.create_bayesian_effects_analysis()
            self.create_significant_pathways_analysis()
            self.create_interactive_dashboard()
            self.create_pathway_network_interactive()
            self.generate_summary_report()
            
            print("\n" + "="*60)
            print("✓ 所有可视化分析完成！")
            print(f"✓ 结果已保存到: {self.output_dir}")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"\n✗ 可视化过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    try:
        # 创建可视化器
        visualizer = MediationAnalysisVisualizer()
        
        # 运行所有可视化
        success = visualizer.run_all_visualizations()
        
        if success:
            print("\n🎉 贝叶斯中介分析可视化完成！")
            print(f"\n📁 查看结果文件夹: {visualizer.output_dir}")
            print("\n📊 推荐查看文件:")
            print("- 交互式中介分析仪表板.html (交互式分析)")
            print("- 交互式路径网络图.html (网络可视化)")
            print("- 贝叶斯中介分析可视化报告.md (详细报告)")
        else:
            print("\n❌ 可视化过程中出现错误，请检查数据文件和错误信息")
            return 1
            
        return 0
        
    except Exception as e:
        print(f"\n❌ 程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())