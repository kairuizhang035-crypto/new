#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数稳定性分析器
评估同一边在不同参数学习方法下的一致性
计算稳定性指标和一致性水平
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import warnings
from itertools import combinations
import networkx as nx

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.family'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ParameterStabilityAnalyzer:
    """参数稳定性分析器"""
    
    def __init__(self, data_file=None):
        """
        初始化参数稳定性分析器
        
        Args:
            data_file: 数据文件路径
        """
        self.data_file = data_file
        self.causal_edges = []
        self.edge_gains = {}  # 存储各方法的边级似然增益结果
        
    def load_causal_edges(self):
        """加载因果边"""
        # 使用脚本所在目录作为基础目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 尝试多个可能的因果边文件路径
        possible_paths = [
            os.path.join(base_dir, "因果边.txt"),
            os.path.join(base_dir, "..", "02因果发现", "06候选因果边集合", "精简因果边列表.csv"),
            os.path.join(base_dir, "..", "02因果发现", "因果边.txt")
        ]
        
        edges_file = None
        for path in possible_paths:
            if os.path.exists(path):
                edges_file = path
                break
        
        if edges_file:
            try:
                if edges_file.endswith('.csv'):
                    # 处理CSV格式的因果边文件
                    df = pd.read_csv(edges_file)
                    self.causal_edges = []
                    for _, row in df.iterrows():
                        if 'source' in df.columns and 'target' in df.columns:
                            self.causal_edges.append((row['source'], row['target']))
                        elif 'Source' in df.columns and 'Target' in df.columns:
                            self.causal_edges.append((row['Source'], row['Target']))
                        elif len(df.columns) >= 2:
                            self.causal_edges.append((row.iloc[0], row.iloc[1]))
                else:
                    # 处理TXT格式的因果边文件
                    with open(edges_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    self.causal_edges = []
                    for line in lines:
                        line = line.strip()
                        if '->' in line:
                            source, target = line.split('->')
                            self.causal_edges.append((source.strip(), target.strip()))
                
                print(f"成功加载 {len(self.causal_edges)} 条因果边")
                return True
            except Exception as e:
                print(f"加载因果边失败: {e}")
                return False
        else:
            print("因果边文件不存在")
            return False
    
    def load_edge_likelihood_gains(self):
        """加载边级似然增益结果"""
        # 使用脚本所在目录作为基础目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        gains_file = os.path.join(base_dir, "05边级似然增益结果", "边级似然增益结果.json")
        
        if os.path.exists(gains_file):
            try:
                with open(gains_file, 'r', encoding='utf-8') as f:
                    self.edge_gains = json.load(f)
                print(f"成功加载边级似然增益结果，包含 {len(self.edge_gains)} 个方法")
                print(f"成功加载边级似然增益结果，包含方法: {list(self.edge_gains.keys())}")
                return True
            except Exception as e:
                print(f"加载边级似然增益结果失败: {e}")
                return False
        else:
            print(f"边级似然增益结果文件不存在: {gains_file}")
            # 尝试从各个方法的结果文件夹分别加载
            print("尝试从各方法结果文件夹加载边级似然增益...")
            return self._load_gains_from_method_folders()
    
    def _load_gains_from_method_folders(self):
        """从各方法文件夹分别加载边级似然增益结果"""
        # 使用当前脚本所在目录作为基础目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 首先尝试从边级似然增益结果文件夹加载
        gains_folder = os.path.join(base_dir, "05边级似然增益结果")
        gains_file = os.path.join(gains_folder, "边级似然增益汇总.json")
        
        if os.path.exists(gains_file):
            try:
                with open(gains_file, 'r', encoding='utf-8') as f:
                    gains_data = json.load(f)
                
                # 将数据转换为所需格式
                for method in ['MLE', 'Bayesian', 'EM', 'SEM']:
                    if method in gains_data:
                        self.edge_gains[method] = gains_data[method]
                
                print("从边级似然增益结果文件夹加载数据")
                return len(self.edge_gains) > 0
            except Exception as e:
                print(f"从边级似然增益结果文件夹加载失败: {e}")
        
        # 如果没有边级似然增益结果，尝试从各方法文件夹加载
        methods = {
            'MLE': ('01MLE_CPT结果', 'MLE_CPTs.json'),
            'Bayesian': ('02Bayesian_CPT结果', 'Bayesian_CPTs.json'),
            'EM': ('03EM_CPT结果', 'EM_CPTs.json'),
            'SEM': ('04SEM_结果', 'SEM_结构方程.json')
        }
        
        loaded_any = False
        for method, (folder, filename) in methods.items():
            result_file = os.path.join(base_dir, folder, filename)
            if os.path.exists(result_file):
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        method_result = json.load(f)
                    
                    # 如果结果中包含边级似然增益，提取出来
                    if 'edge_likelihood_gains' in method_result:
                        self.edge_gains[method] = method_result['edge_likelihood_gains']
                        loaded_any = True
                        print(f"从 {method} 结果中加载边级似然增益")
                except Exception as e:
                    print(f"从 {method} 加载边级似然增益失败: {e}")
        
        return loaded_any
    
    def calculate_parameter_stability(self):
        """
        计算参数稳定性指标
        评估同一边在不同参数学习方法下的一致性
        
        Returns:
            dict: 包含每条边的稳定性分数和详细信息
        """
        print("\n开始计算参数稳定性指标...")
        
        try:
            stability_results = {}
            
            if not self.edge_gains:
                print("没有可用的边级似然增益结果")
                return {}
            
            # 计算每条边的稳定性
            edge_stability = {}
            processed_edges = 0
            
            for edge in self.causal_edges:
                edge_key = f"{edge[0]}->{edge[1]}"
                
                # 收集该边在所有方法中的S_param分数
                scores = []
                method_scores = {}
                
                for method, gains in self.edge_gains.items():
                    if edge_key in gains and 'S_param' in gains[edge_key]:
                        score = gains[edge_key]['S_param']
                        scores.append(score)
                        method_scores[method] = score
                
                if len(scores) >= 2:  # 至少需要两个方法的结果
                    # 计算稳定性指标
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    cv = std_score / mean_score if mean_score > 0 else float('inf')
                    
                    # 稳定性分数：变异系数的倒数（归一化）
                    stability_score = 1 / (1 + cv) if cv != float('inf') else 1.0
                    
                    # 计算方法间的成对差异
                    pairwise_diffs = []
                    method_names = list(method_scores.keys())
                    for i in range(len(method_names)):
                        for j in range(i+1, len(method_names)):
                            diff = abs(method_scores[method_names[i]] - method_scores[method_names[j]])
                            pairwise_diffs.append(diff)
                    
                    max_diff = max(pairwise_diffs) if pairwise_diffs else 0
                    avg_diff = np.mean(pairwise_diffs) if pairwise_diffs else 0
                    
                    # 计算方法间相关性
                    correlations = {}
                    if len(method_names) >= 2:
                        for method1, method2 in combinations(method_names, 2):
                            # 收集两个方法在所有边上的分数
                            scores1, scores2 = [], []
                            for e in self.causal_edges:
                                ek = f"{e[0]}->{e[1]}"
                                if (ek in self.edge_gains.get(method1, {}) and 
                                    ek in self.edge_gains.get(method2, {})):
                                    s1 = self.edge_gains[method1][ek].get('S_param', 0)
                                    s2 = self.edge_gains[method2][ek].get('S_param', 0)
                                    scores1.append(s1)
                                    scores2.append(s2)
                            
                            if len(scores1) >= 2:
                                corr = np.corrcoef(scores1, scores2)[0, 1]
                                correlations[f"{method1}-{method2}"] = corr if not np.isnan(corr) else 0
                    
                    stability_results[edge_key] = {
                        'edge': edge,
                        'method_scores': method_scores,
                        'mean_score': mean_score,
                        'std_score': std_score,
                        'coefficient_of_variation': cv,
                        'stability_score': stability_score,
                        'max_pairwise_diff': max_diff,
                        'avg_pairwise_diff': avg_diff,
                        'num_methods': len(scores),
                        'consistency_level': self._classify_consistency(stability_score),
                        'method_correlations': correlations
                    }
            
            # 计算整体稳定性统计
            if stability_results:
                all_stability_scores = [result['stability_score'] for result in stability_results.values()]
                all_cv_scores = [result['coefficient_of_variation'] for result in stability_results.values() 
                               if result['coefficient_of_variation'] != float('inf')]
                
                # 计算一致性水平分布
                consistency_counts = {}
                for result in stability_results.values():
                    level = result['consistency_level']
                    consistency_counts[level] = consistency_counts.get(level, 0) + 1
                
                overall_stats = {
                    'mean_stability': np.mean(all_stability_scores),
                    'std_stability': np.std(all_stability_scores),
                    'min_stability': min(all_stability_scores),
                    'max_stability': max(all_stability_scores),
                    'mean_cv': np.mean(all_cv_scores) if all_cv_scores else 0,
                    'num_edges': len(stability_results),
                    'consistency_distribution': consistency_counts,
                    'high_consistency_ratio': consistency_counts.get('高度一致', 0) / len(stability_results)
                }
                
                stability_results['_overall_statistics'] = overall_stats
            
            print(f"完成 {len(stability_results)-1 if '_overall_statistics' in stability_results else len(stability_results)} 条边的稳定性计算")
            return stability_results
            
        except Exception as e:
            print(f"计算参数稳定性失败: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _classify_consistency(self, stability_score):
        """根据稳定性分数分类一致性水平"""
        if stability_score >= 0.8:
            return "高度一致"
        elif stability_score >= 0.6:
            return "中等一致"
        elif stability_score >= 0.4:
            return "低度一致"
        else:
            return "不一致"
    
    def create_output_folder(self):
        """创建输出文件夹"""
        # 使用当前脚本所在目录作为基础目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_folder = os.path.join(base_dir, "06参数稳定性结果")
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"创建输出文件夹: {output_folder}")
        
        return output_folder
    
    def save_results(self, stability_results):
        """保存参数稳定性结果"""
        output_folder = self.create_output_folder()
        
        # 保存完整的参数稳定性结果
        stability_file = os.path.join(output_folder, "参数稳定性详细结果.json")
        with open(stability_file, 'w', encoding='utf-8') as f:
            json.dump(stability_results, f, ensure_ascii=False, indent=2, default=str)
        
        # 创建稳定性汇总CSV
        self._create_stability_summary_csv(stability_results, output_folder)
        
        # 创建详细报告
        self._create_detailed_report(stability_results, output_folder)
        
        # 创建可视化
        self._create_visualizations(stability_results, output_folder)
        
        print(f"参数稳定性结果已保存到: {output_folder}")
    
    def _create_stability_summary_csv(self, stability_results, output_folder):
        """创建稳定性汇总CSV"""
        try:
            summary_data = []
            
            for edge_key, edge_info in stability_results.items():
                if edge_key == '_overall_statistics':
                    continue
                    
                if isinstance(edge_info, dict) and 'edge' in edge_info:
                    edge = edge_info['edge']
                    method_scores = edge_info.get('method_scores', {})
                    
                    row_data = {
                        '边': edge_key,
                        '源节点': edge[0],
                        '目标节点': edge[1],
                        '平均分数': edge_info.get('mean_score', 0),
                        '标准差': edge_info.get('std_score', 0),
                        '变异系数': edge_info.get('coefficient_of_variation', 0),
                        '稳定性分数': edge_info.get('stability_score', 0),
                        '最大成对差异': edge_info.get('max_pairwise_diff', 0),
                        '平均成对差异': edge_info.get('avg_pairwise_diff', 0),
                        '方法数量': edge_info.get('num_methods', 0),
                        '一致性水平': edge_info.get('consistency_level', '未知')
                    }
                    
                    # 添加各方法的分数
                    for method in ['MLE', 'Bayesian', 'EM', 'SEM']:
                        row_data[f'{method}_分数'] = method_scores.get(method, None)
                    
                    summary_data.append(row_data)
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                
                # 按稳定性分数排序
                summary_df = summary_df.sort_values('稳定性分数', ascending=False)
                
                summary_file = os.path.join(output_folder, "参数稳定性汇总.csv")
                summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
                
                # 创建按一致性水平分组的文件
                for level in summary_df['一致性水平'].unique():
                    level_df = summary_df[summary_df['一致性水平'] == level]
                    level_file = os.path.join(output_folder, f"参数稳定性_{level}.csv")
                    level_df.to_csv(level_file, index=False, encoding='utf-8-sig')
                
                print("稳定性汇总CSV已保存")
            
        except Exception as e:
            print(f"创建稳定性汇总CSV失败: {e}")
    
    def _create_detailed_report(self, stability_results, output_folder):
        """创建详细报告"""
        try:
            report_file = os.path.join(output_folder, "参数稳定性详细报告.txt")
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("参数稳定性详细报告\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 整体统计
                if '_overall_statistics' in stability_results:
                    stats = stability_results['_overall_statistics']
                    f.write("整体稳定性统计:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"边数量: {stats['num_edges']}\n")
                    f.write(f"平均稳定性分数: {stats['mean_stability']:.4f}\n")
                    f.write(f"稳定性分数标准差: {stats['std_stability']:.4f}\n")
                    f.write(f"最小稳定性分数: {stats['min_stability']:.4f}\n")
                    f.write(f"最大稳定性分数: {stats['max_stability']:.4f}\n")
                    f.write(f"平均变异系数: {stats['mean_cv']:.4f}\n")
                    f.write(f"高度一致比例: {stats['high_consistency_ratio']:.2%}\n\n")
                    
                    f.write("一致性水平分布:\n")
                    for level, count in stats['consistency_distribution'].items():
                        ratio = count / stats['num_edges']
                        f.write(f"  {level}: {count} ({ratio:.2%})\n")
                    f.write("\n")
                
                # 按稳定性分数排序的边详情
                edge_items = [(k, v) for k, v in stability_results.items() 
                             if k != '_overall_statistics']
                edge_items.sort(key=lambda x: x[1]['stability_score'], reverse=True)
                
                f.write("边级稳定性详情 (按稳定性分数排序):\n")
                f.write("-" * 40 + "\n")
                
                for edge_key, info in edge_items:
                    f.write(f"\n边: {edge_key}\n")
                    f.write(f"  稳定性分数: {info['stability_score']:.4f}\n")
                    f.write(f"  一致性水平: {info['consistency_level']}\n")
                    f.write(f"  平均分数: {info['mean_score']:.4f}\n")
                    f.write(f"  标准差: {info['std_score']:.4f}\n")
                    f.write(f"  变异系数: {info['coefficient_of_variation']:.4f}\n")
                    f.write(f"  最大成对差异: {info['max_pairwise_diff']:.4f}\n")
                    f.write(f"  平均成对差异: {info['avg_pairwise_diff']:.4f}\n")
                    f.write(f"  参与方法数: {info['num_methods']}\n")
                    
                    f.write("  各方法分数:\n")
                    for method, score in info['method_scores'].items():
                        f.write(f"    {method}: {score:.4f}\n")
                    
                    if 'method_correlations' in info and info['method_correlations']:
                        f.write("  方法间相关性:\n")
                        for pair, corr in info['method_correlations'].items():
                            f.write(f"    {pair}: {corr:.4f}\n")
            
            print(f"详细报告已保存: {report_file}")
            
        except Exception as e:
            print(f"创建详细报告失败: {e}")
    
    def _create_visualizations(self, stability_results, output_folder):
        """创建可视化图表"""
        try:
            # 1. 稳定性分数分布直方图
            plt.figure(figsize=(10, 6))
            
            stability_scores = [info['stability_score'] for k, info in stability_results.items() 
                              if k != '_overall_statistics']
            
            plt.hist(stability_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('稳定性分数')
            plt.ylabel('频数')
            plt.title('参数稳定性分数分布')
            plt.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_score = np.mean(stability_scores)
            plt.axvline(mean_score, color='red', linestyle='--', 
                       label=f'平均值: {mean_score:.3f}')
            plt.legend()
            
            plt.tight_layout()
            hist_file = os.path.join(output_folder, "稳定性分数分布.png")
            plt.savefig(hist_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 一致性水平饼图
            if '_overall_statistics' in stability_results:
                plt.figure(figsize=(8, 8))
                
                consistency_dist = stability_results['_overall_statistics']['consistency_distribution']
                labels = list(consistency_dist.keys())
                sizes = list(consistency_dist.values())
                colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
                
                plt.pie(sizes, labels=labels, colors=colors[:len(labels)], autopct='%1.1f%%', 
                       startangle=90)
                plt.title('一致性水平分布')
                
                pie_file = os.path.join(output_folder, "一致性水平分布.png")
                plt.savefig(pie_file, dpi=300, bbox_inches='tight')
                plt.close()
            
            # 3. 方法间分数对比散点图
            methods = list(set().union(*[info['method_scores'].keys() 
                                       for k, info in stability_results.items() 
                                       if k != '_overall_statistics']))
            
            if len(methods) >= 2:
                # 创建方法间的成对比较图
                method_pairs = list(combinations(methods, 2))
                
                if method_pairs:
                    n_pairs = len(method_pairs)
                    n_cols = min(3, n_pairs)
                    n_rows = (n_pairs + n_cols - 1) // n_cols
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
                    
                    # 确保axes是二维数组
                    if n_rows == 1 and n_cols == 1:
                        axes = [[axes]]
                    elif n_rows == 1:
                        axes = [axes]
                    elif n_cols == 1:
                        axes = [[ax] for ax in axes]
                    
                    for idx, (method1, method2) in enumerate(method_pairs):
                        row = idx // n_cols
                        col = idx % n_cols
                        
                        scores1, scores2 = [], []
                        for k, info in stability_results.items():
                            if k != '_overall_statistics':
                                if (method1 in info['method_scores'] and 
                                    method2 in info['method_scores']):
                                    scores1.append(info['method_scores'][method1])
                                    scores2.append(info['method_scores'][method2])
                        
                        if scores1 and scores2:
                            axes[row][col].scatter(scores1, scores2, alpha=0.6)
                            axes[row][col].set_xlabel(f'{method1} 分数')
                            axes[row][col].set_ylabel(f'{method2} 分数')
                            axes[row][col].set_title(f'{method1} vs {method2}')
                            
                            # 添加对角线
                            min_val = min(min(scores1), min(scores2))
                            max_val = max(max(scores1), max(scores2))
                            axes[row][col].plot([min_val, max_val], [min_val, max_val], 
                                              'r--', alpha=0.5)
                            
                            # 计算相关系数
                            corr = np.corrcoef(scores1, scores2)[0, 1]
                            if not np.isnan(corr):
                                axes[row][col].text(0.05, 0.95, f'r={corr:.3f}', 
                                                   transform=axes[row][col].transAxes,
                                                   verticalalignment='top')
                    
                    # 隐藏多余的子图
                    for idx in range(len(method_pairs), n_rows * n_cols):
                        row = idx // n_cols
                        col = idx % n_cols
                        axes[row][col].axis('off')
                
                plt.tight_layout()
                scatter_file = os.path.join(output_folder, "方法间分数对比.png")
                plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
                plt.close()
            
            # 4. 稳定性分数热力图
            edge_keys = [k for k in stability_results.keys() if k != '_overall_statistics']
            if edge_keys and methods:
                heatmap_data = []
                edge_labels = []
                
                for edge_key in edge_keys:
                    info = stability_results[edge_key]
                    row = []
                    for method in methods:
                        score = info['method_scores'].get(method, np.nan)
                        row.append(score)
                    heatmap_data.append(row)
                    edge_labels.append(edge_key)
                
                plt.figure(figsize=(max(8, len(methods)*2), max(6, len(edge_keys)*0.3)))
                
                heatmap_df = pd.DataFrame(heatmap_data, index=edge_labels, columns=methods)
                sns.heatmap(heatmap_df, annot=True, cmap='RdYlBu_r', fmt='.3f', 
                           cbar_kws={'label': '归一化分数'})
                plt.title('各边在不同方法下的参数分数')
                plt.tight_layout()
                
                heatmap_file = os.path.join(output_folder, "参数分数热力图.png")
                plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
                plt.close()
            
            print("可视化图表已生成")
            
        except Exception as e:
            print(f"创建可视化失败: {e}")

def main():
    """主函数"""
    print("开始参数稳定性分析...")
    
    # 使用脚本所在目录作为基础目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 数据文件路径 - 使用绝对路径
    data_file = os.path.join(base_dir, "..", "01数据预处理", "processed_data.csv")
    
    # 创建分析器实例
    analyzer = ParameterStabilityAnalyzer(data_file)
    
    # 加载因果边
    if not analyzer.load_causal_edges():
        print("因果边加载失败，程序退出")
        return
    
    # 加载边级似然增益结果
    if not analyzer.load_edge_likelihood_gains():
        print("边级似然增益结果加载失败，程序退出")
        return
    
    # 计算参数稳定性
    stability_results = analyzer.calculate_parameter_stability()
    
    if stability_results:
        # 保存结果
        analyzer.save_results(stability_results)
        print("参数稳定性分析完成！")
    else:
        print("未能计算出参数稳定性结果")

if __name__ == "__main__":
    main()