#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多方法参数学习结果可视化脚本
对MLE、Bayesian、EM、SEM、边级似然增益、参数稳定性结果进行丰富的可视化分析
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
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

class ParameterLearningVisualizer:
    """参数学习结果可视化器"""
    
    def __init__(self, base_dir=None):
        """
        初始化可视化器
        
        Args:
            base_dir: 基础目录路径，默认为脚本所在目录
        """
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(self.base_dir, "可视化")
        
        # 结果文件夹路径
        self.result_dirs = {
            'MLE': os.path.join(self.base_dir, "01MLE_CPT结果"),
            'Bayesian': os.path.join(self.base_dir, "02Bayesian_CPT结果"),
            'EM': os.path.join(self.base_dir, "03EM_CPT结果"),
            'SEM': os.path.join(self.base_dir, "04SEM_结果"),
            'EdgeGain': os.path.join(self.base_dir, "05边级似然增益结果"),
            'Stability': os.path.join(self.base_dir, "06参数稳定性结果")
        }
        
        # 数据存储
        self.data = {}
        
    def create_output_dir(self):
        """创建输出目录"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"创建输出目录: {self.output_dir}")
        
    def load_data(self):
        """加载所有结果数据"""
        print("加载参数学习结果数据...")
        
        # 加载MLE结果
        mle_summary_path = os.path.join(self.result_dirs['MLE'], "MLE_条件概率表汇总.csv")
        if os.path.exists(mle_summary_path):
            self.data['mle_summary'] = pd.read_csv(mle_summary_path)
            print(f"✓ 加载MLE汇总数据: {len(self.data['mle_summary'])} 条记录")
        
        # 加载Bayesian结果
        bayesian_summary_path = os.path.join(self.result_dirs['Bayesian'], "Bayesian_条件概率表汇总.csv")
        if os.path.exists(bayesian_summary_path):
            self.data['bayesian_summary'] = pd.read_csv(bayesian_summary_path)
            print(f"✓ 加载Bayesian汇总数据: {len(self.data['bayesian_summary'])} 条记录")
        
        # 加载EM结果
        em_summary_path = os.path.join(self.result_dirs['EM'], "EM_条件概率表汇总.csv")
        if os.path.exists(em_summary_path):
            self.data['em_summary'] = pd.read_csv(em_summary_path)
            print(f"✓ 加载EM汇总数据: {len(self.data['em_summary'])} 条记录")
        
        # 加载SEM结果
        sem_coeff_path = os.path.join(self.result_dirs['SEM'], "SEM_系数表.csv")
        if os.path.exists(sem_coeff_path):
            self.data['sem_coeffs'] = pd.read_csv(sem_coeff_path)
            print(f"✓ 加载SEM系数数据: {len(self.data['sem_coeffs'])} 条记录")
        
        # 加载边级似然增益结果
        edge_gain_path = os.path.join(self.result_dirs['EdgeGain'], "边级似然增益汇总.csv")
        if os.path.exists(edge_gain_path):
            self.data['edge_gains'] = pd.read_csv(edge_gain_path)
            print(f"✓ 加载边级似然增益数据: {len(self.data['edge_gains'])} 条记录")
        
        # 加载参数稳定性结果
        stability_path = os.path.join(self.result_dirs['Stability'], "参数稳定性汇总.csv")
        if os.path.exists(stability_path):
            self.data['stability'] = pd.read_csv(stability_path)
            print(f"✓ 加载参数稳定性数据: {len(self.data['stability'])} 条记录")
    
    def create_method_comparison_chart(self):
        """创建方法对比图表"""
        print("创建方法对比图表...")
        
        # 统计各方法的节点数量
        method_stats = {}
        
        if 'mle_summary' in self.data:
            method_stats['MLE'] = {
                'nodes': len(self.data['mle_summary']['节点'].unique()),
                'conditions': len(self.data['mle_summary'])
            }
        
        if 'bayesian_summary' in self.data:
            method_stats['Bayesian'] = {
                'nodes': len(self.data['bayesian_summary']['节点'].unique()),
                'conditions': len(self.data['bayesian_summary'])
            }
        
        if 'em_summary' in self.data:
            method_stats['EM'] = {
                'nodes': len(self.data['em_summary']['节点'].unique()),
                'conditions': len(self.data['em_summary'])
            }
        
        if 'sem_coeffs' in self.data:
            method_stats['SEM'] = {
                'nodes': len(self.data['sem_coeffs']['方程'].unique()),
                'conditions': len(self.data['sem_coeffs'][self.data['sem_coeffs']['变量类型'] == '解释变量'])
            }
        
        # 创建对比图
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        methods = list(method_stats.keys())
        nodes_count = [method_stats[m]['nodes'] for m in methods]
        conditions_count = [method_stats[m]['conditions'] for m in methods]
        
        # 节点数量对比
        bars1 = axes[0].bar(methods, nodes_count, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[0].set_title('各方法处理的节点数量', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('节点数量')
        axes[0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        
        # 条件/系数数量对比
        bars2 = axes[1].bar(methods, conditions_count, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[1].set_title('各方法的条件/系数数量', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('条件/系数数量')
        axes[1].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "方法对比统计.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        return method_stats
    
    def create_probability_distribution_analysis(self):
        """创建概率分布分析"""
        print("创建概率分布分析...")
        
        if 'mle_summary' not in self.data:
            print("警告: 缺少MLE数据，跳过概率分布分析")
            return
        
        # 分析P(0)和P(1)的分布
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # MLE概率分布
        if 'mle_summary' in self.data:
            axes[0, 0].hist(self.data['mle_summary']['P(0)'], bins=30, alpha=0.7, color='#FF6B6B', label='P(0)')
            axes[0, 0].hist(self.data['mle_summary']['P(1)'], bins=30, alpha=0.7, color='#4ECDC4', label='P(1)')
            axes[0, 0].set_title('MLE 概率分布', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('概率值')
            axes[0, 0].set_ylabel('频次')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Bayesian概率分布
        if 'bayesian_summary' in self.data:
            axes[0, 1].hist(self.data['bayesian_summary']['P(0)'], bins=30, alpha=0.7, color='#FF6B6B', label='P(0)')
            axes[0, 1].hist(self.data['bayesian_summary']['P(1)'], bins=30, alpha=0.7, color='#4ECDC4', label='P(1)')
            axes[0, 1].set_title('Bayesian 概率分布', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('概率值')
            axes[0, 1].set_ylabel('频次')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # EM概率分布
        if 'em_summary' in self.data:
            axes[1, 0].hist(self.data['em_summary']['P(0)'], bins=30, alpha=0.7, color='#FF6B6B', label='P(0)')
            axes[1, 0].hist(self.data['em_summary']['P(1)'], bins=30, alpha=0.7, color='#4ECDC4', label='P(1)')
            axes[1, 0].set_title('EM 概率分布', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('概率值')
            axes[1, 0].set_ylabel('频次')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 概率差异分析
        if 'mle_summary' in self.data and 'bayesian_summary' in self.data:
            # 计算MLE和Bayesian的概率差异
            mle_p1 = self.data['mle_summary']['P(1)'].values
            bayesian_p1 = self.data['bayesian_summary']['P(1)'].values
            
            if len(mle_p1) == len(bayesian_p1):
                diff = np.abs(mle_p1 - bayesian_p1)
                axes[1, 1].hist(diff, bins=30, alpha=0.7, color='#45B7D1')
                axes[1, 1].set_title('MLE vs Bayesian P(1)差异分布', fontsize=12, fontweight='bold')
                axes[1, 1].set_xlabel('|P(1)_MLE - P(1)_Bayesian|')
                axes[1, 1].set_ylabel('频次')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "概率分布分析.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_edge_gain_analysis(self):
        """创建边级似然增益分析"""
        print("创建边级似然增益分析...")
        
        if 'edge_gains' not in self.data:
            print("警告: 缺少边级似然增益数据，跳过分析")
            return
        
        df = self.data['edge_gains']
        
        # 创建多子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 各方法似然增益分布
        gain_columns = [col for col in df.columns if 'Gain' in col and col != 'Edge']
        if gain_columns:
            for i, col in enumerate(gain_columns):
                if col in df.columns:
                    valid_data = df[col].dropna()
                    if len(valid_data) > 0:
                        axes[0, 0].hist(valid_data, bins=20, alpha=0.6, label=col.replace('_Gain', ''))
            
            axes[0, 0].set_title('各方法似然增益分布', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('似然增益值')
            axes[0, 0].set_ylabel('频次')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 参数S值分布
        s_columns = [col for col in df.columns if 'S_param' in col]
        if s_columns:
            for i, col in enumerate(s_columns):
                if col in df.columns:
                    valid_data = df[col].dropna()
                    if len(valid_data) > 0:
                        axes[0, 1].hist(valid_data, bins=20, alpha=0.6, label=col.replace('_S_param', ''))
            
            axes[0, 1].set_title('各方法参数S值分布', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('参数S值')
            axes[0, 1].set_ylabel('频次')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. MLE vs Bayesian 似然增益对比
        if 'MLE_Gain' in df.columns and 'Bayesian_Gain' in df.columns:
            valid_mask = df['MLE_Gain'].notna() & df['Bayesian_Gain'].notna()
            if valid_mask.sum() > 0:
                axes[1, 0].scatter(df.loc[valid_mask, 'MLE_Gain'], 
                                 df.loc[valid_mask, 'Bayesian_Gain'], 
                                 alpha=0.6, color='#45B7D1')
                axes[1, 0].plot([df['MLE_Gain'].min(), df['MLE_Gain'].max()], 
                               [df['MLE_Gain'].min(), df['MLE_Gain'].max()], 
                               'r--', alpha=0.8)
                axes[1, 0].set_title('MLE vs Bayesian 似然增益对比', fontsize=12, fontweight='bold')
                axes[1, 0].set_xlabel('MLE 似然增益')
                axes[1, 0].set_ylabel('Bayesian 似然增益')
                axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 边的似然增益排名
        if 'MLE_Gain' in df.columns:
            top_edges = df.nlargest(10, 'MLE_Gain')
            y_pos = np.arange(len(top_edges))
            axes[1, 1].barh(y_pos, top_edges['MLE_Gain'], color='#96CEB4')
            axes[1, 1].set_yticks(y_pos)
            axes[1, 1].set_yticklabels([edge[:20] + '...' if len(edge) > 20 else edge 
                                       for edge in top_edges['Edge']], fontsize=8)
            axes[1, 1].set_title('Top 10 边的似然增益 (MLE)', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('似然增益值')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "边级似然增益分析.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_stability_analysis(self):
        """创建参数稳定性分析"""
        print("创建参数稳定性分析...")
        
        if 'stability' not in self.data:
            print("警告: 缺少参数稳定性数据，跳过分析")
            return
        
        df = self.data['stability']
        
        # 创建多子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 稳定性分数分布
        if '稳定性分数' in df.columns:
            axes[0, 0].hist(df['稳定性分数'], bins=30, color='#FF6B6B', alpha=0.7)
            axes[0, 0].set_title('稳定性分数分布', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('稳定性分数')
            axes[0, 0].set_ylabel('频次')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_score = df['稳定性分数'].mean()
            axes[0, 0].axvline(mean_score, color='red', linestyle='--', 
                              label=f'平均值: {mean_score:.3f}')
            axes[0, 0].legend()
        
        # 2. 变异系数分布
        if '变异系数' in df.columns:
            # 过滤掉无穷大值
            cv_data = df['变异系数'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(cv_data) > 0:
                axes[0, 1].hist(cv_data, bins=30, color='#4ECDC4', alpha=0.7)
                axes[0, 1].set_title('变异系数分布', fontsize=12, fontweight='bold')
                axes[0, 1].set_xlabel('变异系数')
                axes[0, 1].set_ylabel('频次')
                axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 一致性水平分布
        if '一致性水平' in df.columns:
            consistency_counts = df['一致性水平'].value_counts()
            colors = ['#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
            axes[1, 0].pie(consistency_counts.values, labels=consistency_counts.index, 
                          autopct='%1.1f%%', colors=colors[:len(consistency_counts)])
            axes[1, 0].set_title('一致性水平分布', fontsize=12, fontweight='bold')
        
        # 4. 方法间分数对比（选择前10个边）
        method_columns = [col for col in df.columns if col.endswith('_分数')]
        if len(method_columns) >= 2:
            top_edges = df.head(10)
            x = np.arange(len(top_edges))
            width = 0.8 / len(method_columns)
            
            for i, col in enumerate(method_columns):
                if col in top_edges.columns:
                    values = top_edges[col].fillna(0)
                    axes[1, 1].bar(x + i * width, values, width, 
                                  label=col.replace('_分数', ''), alpha=0.8)
            
            axes[1, 1].set_title('前10个边的方法间分数对比', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('边索引')
            axes[1, 1].set_ylabel('分数')
            axes[1, 1].set_xticks(x + width * (len(method_columns) - 1) / 2)
            axes[1, 1].set_xticklabels([f'边{i+1}' for i in range(len(top_edges))])
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "参数稳定性分析.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_sem_analysis(self):
        """创建SEM分析"""
        print("创建SEM分析...")
        
        if 'sem_coeffs' not in self.data:
            print("警告: 缺少SEM数据，跳过分析")
            return
        
        df = self.data['sem_coeffs']
        
        # 过滤解释变量
        explanatory_vars = df[df['变量类型'] == '解释变量'].copy()
        
        if len(explanatory_vars) == 0:
            print("警告: 没有找到解释变量数据")
            return
        
        # 创建多子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 系数分布
        if '系数' in explanatory_vars.columns:
            axes[0, 0].hist(explanatory_vars['系数'], bins=30, color='#FF6B6B', alpha=0.7)
            axes[0, 0].set_title('SEM系数分布', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('系数值')
            axes[0, 0].set_ylabel('频次')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 添加零线
            axes[0, 0].axvline(0, color='black', linestyle='--', alpha=0.5)
        
        # 2. t统计量分布
        if 't统计量' in explanatory_vars.columns:
            t_stats = explanatory_vars['t统计量'].dropna()
            if len(t_stats) > 0:
                axes[0, 1].hist(t_stats, bins=30, color='#4ECDC4', alpha=0.7)
                axes[0, 1].set_title('t统计量分布', fontsize=12, fontweight='bold')
                axes[0, 1].set_xlabel('t统计量')
                axes[0, 1].set_ylabel('频次')
                axes[0, 1].grid(True, alpha=0.3)
                
                # 添加显著性线
                axes[0, 1].axvline(1.96, color='red', linestyle='--', alpha=0.7, label='p=0.05')
                axes[0, 1].axvline(-1.96, color='red', linestyle='--', alpha=0.7)
                axes[0, 1].legend()
        
        # 3. 系数 vs t统计量散点图
        if '系数' in explanatory_vars.columns and 't统计量' in explanatory_vars.columns:
            valid_mask = explanatory_vars['系数'].notna() & explanatory_vars['t统计量'].notna()
            if valid_mask.sum() > 0:
                axes[1, 0].scatter(explanatory_vars.loc[valid_mask, '系数'], 
                                 explanatory_vars.loc[valid_mask, 't统计量'], 
                                 alpha=0.6, color='#45B7D1')
                axes[1, 0].set_title('系数 vs t统计量', fontsize=12, fontweight='bold')
                axes[1, 0].set_xlabel('系数值')
                axes[1, 0].set_ylabel('t统计量')
                axes[1, 0].grid(True, alpha=0.3)
                
                # 添加显著性区域
                axes[1, 0].axhline(1.96, color='red', linestyle='--', alpha=0.5)
                axes[1, 0].axhline(-1.96, color='red', linestyle='--', alpha=0.5)
        
        # 4. 各方程的系数数量
        if '方程' in explanatory_vars.columns:
            equation_counts = explanatory_vars['方程'].value_counts().head(10)
            axes[1, 1].barh(range(len(equation_counts)), equation_counts.values, color='#96CEB4')
            axes[1, 1].set_yticks(range(len(equation_counts)))
            axes[1, 1].set_yticklabels([eq[:20] + '...' if len(eq) > 20 else eq 
                                       for eq in equation_counts.index], fontsize=8)
            axes[1, 1].set_title('各方程的解释变量数量 (Top 10)', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('解释变量数量')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "SEM分析.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_interactive_dashboard(self):
        """创建交互式仪表板"""
        print("创建交互式仪表板...")
        
        # 创建子图
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('方法对比', '概率分布对比', '似然增益分析', 
                          '稳定性分析', 'SEM系数分析', '综合统计'),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "box"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        # 1. 方法对比
        if 'mle_summary' in self.data and 'bayesian_summary' in self.data:
            methods = ['MLE', 'Bayesian', 'EM', 'SEM']
            node_counts = []
            
            for method in methods:
                if method.lower() + '_summary' in self.data:
                    count = len(self.data[method.lower() + '_summary']['节点'].unique())
                    node_counts.append(count)
                elif method == 'SEM' and 'sem_coeffs' in self.data:
                    count = len(self.data['sem_coeffs']['方程'].unique())
                    node_counts.append(count)
                else:
                    node_counts.append(0)
            
            fig.add_trace(
                go.Bar(x=methods, y=node_counts, name='节点数量',
                      marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']),
                row=1, col=1
            )
        
        # 2. 概率分布对比
        if 'mle_summary' in self.data:
            fig.add_trace(
                go.Histogram(x=self.data['mle_summary']['P(1)'], name='MLE P(1)',
                           opacity=0.7, marker_color='#FF6B6B'),
                row=1, col=2
            )
        
        if 'bayesian_summary' in self.data:
            fig.add_trace(
                go.Histogram(x=self.data['bayesian_summary']['P(1)'], name='Bayesian P(1)',
                           opacity=0.7, marker_color='#4ECDC4'),
                row=1, col=2
            )
        
        # 3. 似然增益分析
        if 'edge_gains' in self.data and 'MLE_Gain' in self.data['edge_gains'].columns:
            df = self.data['edge_gains']
            if 'Bayesian_Gain' in df.columns:
                valid_mask = df['MLE_Gain'].notna() & df['Bayesian_Gain'].notna()
                if valid_mask.sum() > 0:
                    fig.add_trace(
                        go.Scatter(x=df.loc[valid_mask, 'MLE_Gain'], 
                                 y=df.loc[valid_mask, 'Bayesian_Gain'],
                                 mode='markers', name='MLE vs Bayesian增益',
                                 marker=dict(color='#45B7D1', opacity=0.6)),
                        row=2, col=1
                    )
        
        # 4. 稳定性分析
        if 'stability' in self.data and '稳定性分数' in self.data['stability'].columns:
            method_columns = [col for col in self.data['stability'].columns if col.endswith('_分数')]
            for col in method_columns[:3]:  # 限制显示前3个方法
                if col in self.data['stability'].columns:
                    fig.add_trace(
                        go.Box(y=self.data['stability'][col].dropna(), name=col.replace('_分数', '')),
                        row=2, col=2
                    )
        
        # 5. SEM系数分析
        if 'sem_coeffs' in self.data:
            explanatory_vars = self.data['sem_coeffs'][self.data['sem_coeffs']['变量类型'] == '解释变量']
            if len(explanatory_vars) > 0 and '系数' in explanatory_vars.columns:
                top_coeffs = explanatory_vars.nlargest(10, '系数')
                fig.add_trace(
                    go.Bar(x=top_coeffs['系数'], y=top_coeffs['变量'], 
                          orientation='h', name='Top 10 系数',
                          marker_color='#96CEB4'),
                    row=3, col=1
                )
        
        # 6. 综合统计表
        stats_data = []
        if 'mle_summary' in self.data:
            stats_data.append(['MLE', len(self.data['mle_summary']), 
                             len(self.data['mle_summary']['节点'].unique())])
        if 'bayesian_summary' in self.data:
            stats_data.append(['Bayesian', len(self.data['bayesian_summary']), 
                             len(self.data['bayesian_summary']['节点'].unique())])
        if 'edge_gains' in self.data:
            stats_data.append(['边级增益', len(self.data['edge_gains']), '-'])
        if 'stability' in self.data:
            stats_data.append(['稳定性', len(self.data['stability']), '-'])
        
        if stats_data:
            fig.add_trace(
                go.Table(
                    header=dict(values=['方法', '记录数', '节点数'],
                              fill_color='paleturquoise',
                              align='left'),
                    cells=dict(values=list(zip(*stats_data)),
                             fill_color='lavender',
                             align='left')
                ),
                row=3, col=2
            )
        
        # 更新布局
        fig.update_layout(
            height=1200,
            title_text="参数学习结果综合分析仪表板",
            title_x=0.5,
            showlegend=True
        )
        
        # 保存交互式图表
        pyo.plot(fig, filename=os.path.join(self.output_dir, "交互式分析仪表板.html"), 
                auto_open=False)
    
    def create_summary_report(self):
        """创建汇总报告"""
        print("创建汇总报告...")
        
        report_path = os.path.join(self.output_dir, "参数学习可视化分析报告.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 参数学习结果可视化分析报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 数据概览\n\n")
            
            # 统计各方法的数据量
            if 'mle_summary' in self.data:
                f.write(f"- **MLE结果**: {len(self.data['mle_summary'])} 条概率记录，")
                f.write(f"{len(self.data['mle_summary']['节点'].unique())} 个节点\n")
            
            if 'bayesian_summary' in self.data:
                f.write(f"- **Bayesian结果**: {len(self.data['bayesian_summary'])} 条概率记录，")
                f.write(f"{len(self.data['bayesian_summary']['节点'].unique())} 个节点\n")
            
            if 'em_summary' in self.data:
                f.write(f"- **EM结果**: {len(self.data['em_summary'])} 条概率记录，")
                f.write(f"{len(self.data['em_summary']['节点'].unique())} 个节点\n")
            
            if 'sem_coeffs' in self.data:
                explanatory_count = len(self.data['sem_coeffs'][self.data['sem_coeffs']['变量类型'] == '解释变量'])
                f.write(f"- **SEM结果**: {explanatory_count} 个解释变量系数，")
                f.write(f"{len(self.data['sem_coeffs']['方程'].unique())} 个方程\n")
            
            if 'edge_gains' in self.data:
                f.write(f"- **边级似然增益**: {len(self.data['edge_gains'])} 条边的增益分析\n")
            
            if 'stability' in self.data:
                f.write(f"- **参数稳定性**: {len(self.data['stability'])} 条边的稳定性分析\n")
                if '一致性水平' in self.data['stability'].columns:
                    consistency_counts = self.data['stability']['一致性水平'].value_counts()
                    f.write(f"  - 一致性水平分布: {dict(consistency_counts)}\n")
            
            f.write("\n## 生成的可视化文件\n\n")
            f.write("1. **方法对比统计.png** - 各参数学习方法的统计对比\n")
            f.write("2. **概率分布分析.png** - MLE、Bayesian、EM的概率分布分析\n")
            f.write("3. **边级似然增益分析.png** - 边级似然增益的详细分析\n")
            f.write("4. **参数稳定性分析.png** - 参数稳定性的多维度分析\n")
            f.write("5. **SEM分析.png** - 结构方程模型的系数分析\n")
            f.write("6. **交互式分析仪表板.html** - 综合交互式分析仪表板\n")
            
            f.write("\n## 主要发现\n\n")
            
            # 稳定性分析发现
            if 'stability' in self.data and '稳定性分数' in self.data['stability'].columns:
                mean_stability = self.data['stability']['稳定性分数'].mean()
                f.write(f"- 平均参数稳定性分数: {mean_stability:.4f}\n")
                
                high_stability = (self.data['stability']['稳定性分数'] > 0.95).sum()
                total_edges = len(self.data['stability'])
                f.write(f"- 高稳定性边数量: {high_stability}/{total_edges} ({high_stability/total_edges*100:.1f}%)\n")
            
            # 似然增益分析发现
            if 'edge_gains' in self.data and 'MLE_Gain' in self.data['edge_gains'].columns:
                max_gain = self.data['edge_gains']['MLE_Gain'].max()
                min_gain = self.data['edge_gains']['MLE_Gain'].min()
                f.write(f"- MLE似然增益范围: {min_gain:.2f} ~ {max_gain:.2f}\n")
            
            f.write("\n---\n")
            f.write("*本报告由参数学习可视化脚本自动生成*\n")
    
    def run_visualization(self):
        """运行完整的可视化分析"""
        print("开始参数学习结果可视化分析...")
        print("=" * 60)
        
        # 创建输出目录
        self.create_output_dir()
        
        # 加载数据
        self.load_data()
        
        if not self.data:
            print("错误: 没有找到任何数据文件")
            return False
        
        # 创建各种可视化
        try:
            method_stats = self.create_method_comparison_chart()
            self.create_probability_distribution_analysis()
            self.create_edge_gain_analysis()
            self.create_stability_analysis()
            self.create_sem_analysis()
            self.create_interactive_dashboard()
            self.create_summary_report()
            
            print("\n" + "=" * 60)
            print("参数学习结果可视化分析完成！")
            print(f"结果保存在: {self.output_dir}")
            print("\n生成的文件:")
            
            # 列出生成的文件
            if os.path.exists(self.output_dir):
                files = os.listdir(self.output_dir)
                for file in sorted(files):
                    print(f"  - {file}")
            
            return True
            
        except Exception as e:
            print(f"可视化过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    print("参数学习结果可视化脚本")
    print("=" * 50)
    
    # 创建可视化器
    visualizer = ParameterLearningVisualizer()
    
    # 运行可视化
    success = visualizer.run_visualization()
    
    if success:
        print("\n✓ 可视化分析成功完成！")
    else:
        print("\n✗ 可视化分析失败")
        sys.exit(1)

if __name__ == "__main__":
    main()