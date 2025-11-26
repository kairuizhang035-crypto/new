#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边级似然增益计算器
计算不同参数学习方法下每条边的似然增益 ΔLL(e) = LL_full - LL_drop(e)
支持MLE、Bayesian、EM、SEM四种方法
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import warnings
from itertools import product
import networkx as nx
import pickle
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.family'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class EdgeLikelihoodGainCalculator:
    """边级似然增益计算器"""
    
    def __init__(self, data_file=None):
        """
        初始化边级似然增益计算器
        
        Args:
            data_file: 数据文件路径
        """
        self.data_file = data_file
        self.data = None
        self.processed_data = None
        self.causal_edges = []
        self.graph = None
        self.results = {}
        
    def load_data(self):
        """加载数据"""
        if self.data_file and os.path.exists(self.data_file):
            try:
                self.data = pd.read_csv(self.data_file)
                print(f"成功加载数据: {self.data.shape}")
                return True
            except Exception as e:
                print(f"加载数据失败: {e}")
                return False
        else:
            print("数据文件不存在")
            return False
    
    def load_causal_edges(self):
        """加载因果边"""
        # 尝试多个可能的因果边文件路径
        possible_paths = [
            os.path.join(os.path.dirname(self.data_file), "因果边.txt"),
            "/home/zkr/因果发现3/02因果发现/06候选因果边集合/精简因果边列表.csv",
            "/home/zkr/因果发现3/02因果发现/因果边.txt"
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
                
                # 创建有向图
                self.graph = nx.DiGraph()
                self.graph.add_edges_from(self.causal_edges)
                
                print(f"成功加载 {len(self.causal_edges)} 条因果边")
                return True
            except Exception as e:
                print(f"加载因果边失败: {e}")
                return False
        else:
            print("因果边文件不存在")
            return False
    
    def preprocess_data(self):
        """数据预处理"""
        if self.data is None:
            return False
        
        try:
            # 移除ID列
            self.processed_data = self.data.copy()
            id_columns = [col for col in self.processed_data.columns if 'id' in col.lower()]
            if id_columns:
                self.processed_data = self.processed_data.drop(columns=id_columns)
            
            # 二值化处理
            for col in self.processed_data.columns:
                if self.processed_data[col].dtype in ['object', 'category']:
                    # 分类变量转换为数值
                    unique_vals = self.processed_data[col].unique()
                    if len(unique_vals) == 2:
                        self.processed_data[col] = pd.Categorical(self.processed_data[col]).codes
                else:
                    # 数值变量二值化（使用中位数）
                    median_val = self.processed_data[col].median()
                    self.processed_data[col] = (self.processed_data[col] > median_val).astype(int)
            
            print(f"数据预处理完成: {self.processed_data.shape}")
            return True
        except Exception as e:
            print(f"数据预处理失败: {e}")
            return False
    
    def load_method_results(self):
        """加载各方法的结果"""
        # 使用当前脚本所在目录作为基础目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 加载各方法的结果
        methods = {
            'MLE': ('01MLE_CPT结果', 'MLE_CPTs.json'),
            'Bayesian': ('02Bayesian_CPT结果', 'Bayesian_CPTs.json'), 
            'EM': ('03EM_CPT结果', 'EM_CPTs.json'),
            'SEM': ('04SEM_结果', 'SEM_结构方程.json')
        }
        
        for method, (folder, filename) in methods.items():
            result_file = os.path.join(base_dir, folder, filename)
            if os.path.exists(result_file):
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        self.results[method] = json.load(f)
                    print(f"成功加载 {method} 方法结果")
                except Exception as e:
                    print(f"加载 {method} 方法结果失败: {e}")
            else:
                print(f"{method} 方法结果文件不存在: {result_file}")
    
    def calculate_edge_likelihood_gain(self, method_name='MLE'):
        """
        计算边级似然增益 ΔLL(e) = LL_full - LL_drop(e)
        
        Args:
            method_name: 参数学习方法名称 ('MLE', 'Bayesian', 'EM', 'SEM')
        
        Returns:
            dict: 包含每条边的似然增益和归一化分数
        """
        print(f"\n开始计算 {method_name} 方法的边级似然增益...")
        
        try:
            # 获取完整图的对数似然
            if method_name in self.results:
                if method_name == 'SEM':
                    # SEM方法使用R²作为似然度量
                    full_likelihood = self._calculate_sem_likelihood()
                else:
                    full_likelihood = self.results[method_name].get('log_likelihood', 0)
            else:
                print(f"方法 {method_name} 的结果不存在")
                return {}
            
            edge_gains = {}
            
            # 对每条边计算似然增益
            for edge in self.causal_edges:
                source, target = edge
                
                # 创建移除该边的临时图
                temp_graph = self.graph.copy()
                temp_graph.remove_edge(source, target)
                
                # 保存原图并替换为临时图
                original_graph = self.graph
                self.graph = temp_graph
                
                # 重新估计参数并计算似然
                if method_name == 'MLE':
                    drop_likelihood = self._calculate_mle_likelihood_for_edge_removal()
                elif method_name == 'Bayesian':
                    drop_likelihood = self._calculate_bayesian_likelihood_for_edge_removal()
                elif method_name == 'EM':
                    drop_likelihood = self._calculate_em_likelihood_for_edge_removal()
                elif method_name == 'SEM':
                    drop_likelihood = self._calculate_sem_likelihood_for_edge_removal()
                else:
                    drop_likelihood = 0
                
                # 恢复原图
                self.graph = original_graph
                
                # 计算似然增益
                likelihood_gain = full_likelihood - drop_likelihood
                edge_gains[f"{source}->{target}"] = {
                    'full_likelihood': full_likelihood,
                    'drop_likelihood': drop_likelihood,
                    'likelihood_gain': likelihood_gain,
                    'edge': edge
                }
            
            # Min-Max归一化
            if edge_gains:
                gains = [info['likelihood_gain'] for info in edge_gains.values()]
                min_gain = min(gains)
                max_gain = max(gains)
                
                if max_gain > min_gain:
                    for edge_key in edge_gains:
                        gain = edge_gains[edge_key]['likelihood_gain']
                        normalized_score = (gain - min_gain) / (max_gain - min_gain)
                        edge_gains[edge_key]['S_param'] = normalized_score
                else:
                    # 所有增益相同的情况
                    for edge_key in edge_gains:
                        edge_gains[edge_key]['S_param'] = 1.0
            
            print(f"完成 {len(edge_gains)} 条边的似然增益计算")
            return edge_gains
            
        except Exception as e:
            print(f"计算边级似然增益失败: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _calculate_mle_likelihood_for_edge_removal(self):
        """为边移除计算MLE似然"""
        try:
            total_log_likelihood = 0
            graph_nodes = set(self.graph.nodes())
            
            for node in graph_nodes:
                if node not in self.processed_data.columns:
                    continue
                
                parents = list(self.graph.predecessors(node))
                parents = [p for p in parents if p in self.processed_data.columns]
                
                if not parents:
                    # 无父节点，计算边际概率
                    node_counts = self.processed_data[node].value_counts()
                    total_count = len(self.processed_data)
                    
                    for value, count in node_counts.items():
                        prob = count / total_count
                        if prob > 0:
                            total_log_likelihood += count * np.log(prob)
                else:
                    # 有父节点，计算条件概率
                    parent_combinations = list(product(*[self.processed_data[p].unique() for p in parents]))
                    
                    for parent_combo in parent_combinations:
                        # 创建父节点条件
                        condition = pd.Series([True] * len(self.processed_data))
                        for i, parent in enumerate(parents):
                            condition &= (self.processed_data[parent] == parent_combo[i])
                        
                        if condition.sum() == 0:
                            continue
                        
                        # 在给定父节点值的条件下，计算子节点的条件概率
                        subset = self.processed_data[condition]
                        node_counts = subset[node].value_counts()
                        total_count = len(subset)
                        
                        for value, count in node_counts.items():
                            prob = count / total_count
                            if prob > 0:
                                total_log_likelihood += count * np.log(prob)
            
            return total_log_likelihood
        except Exception as e:
            print(f"计算MLE似然失败: {e}")
            return 0
    
    def _calculate_bayesian_likelihood_for_edge_removal(self):
        """为边移除计算贝叶斯似然"""
        try:
            total_log_likelihood = 0
            alpha = 1.0  # Dirichlet先验参数
            graph_nodes = set(self.graph.nodes())
            
            for node in graph_nodes:
                if node not in self.processed_data.columns:
                    continue
                
                parents = list(self.graph.predecessors(node))
                parents = [p for p in parents if p in self.processed_data.columns]
                
                if not parents:
                    # 无父节点
                    node_counts = self.processed_data[node].value_counts()
                    total_count = len(self.processed_data)
                    
                    for value, count in node_counts.items():
                        prob = (count + alpha) / (total_count + alpha * len(node_counts))
                        if prob > 0:
                            total_log_likelihood += count * np.log(prob)
                else:
                    # 有父节点
                    parent_combinations = list(product(*[self.processed_data[p].unique() for p in parents]))
                    
                    for parent_combo in parent_combinations:
                        condition = pd.Series([True] * len(self.processed_data))
                        for i, parent in enumerate(parents):
                            condition &= (self.processed_data[parent] == parent_combo[i])
                        
                        if condition.sum() == 0:
                            continue
                        
                        subset = self.processed_data[condition]
                        node_counts = subset[node].value_counts()
                        total_count = len(subset)
                        
                        for value, count in node_counts.items():
                            prob = (count + alpha) / (total_count + alpha * len(node_counts))
                            if prob > 0:
                                total_log_likelihood += count * np.log(prob)
            
            return total_log_likelihood
        except Exception as e:
            print(f"计算贝叶斯似然失败: {e}")
            return 0
    
    def _calculate_em_likelihood_for_edge_removal(self):
        """为边移除计算EM似然"""
        # 简化实现，使用MLE似然作为近似
        return self._calculate_mle_likelihood_for_edge_removal()
    
    def _calculate_sem_likelihood_for_edge_removal(self):
        """为边移除计算SEM似然"""
        try:
            total_r_squared = 0
            node_count = 0
            graph_nodes = set(self.graph.nodes())
            
            for node in graph_nodes:
                if node not in self.processed_data.columns:
                    continue
                
                parents = list(self.graph.predecessors(node))
                parents = [p for p in parents if p in self.processed_data.columns]
                
                if parents:
                    # 有父节点，进行线性回归
                    X = self.processed_data[parents].values
                    y = self.processed_data[node].values
                    
                    if len(np.unique(y)) > 1:  # 确保y有变化
                        model = LinearRegression()
                        model.fit(X, y)
                        r_squared = model.score(X, y)
                        total_r_squared += max(0, r_squared)  # 确保R²非负
                        node_count += 1
            
            # 返回平均R²作为似然度量
            return total_r_squared / max(1, node_count)
        except Exception as e:
            print(f"计算SEM似然失败: {e}")
            return 0
    
    def _calculate_sem_likelihood(self):
        """计算完整SEM的似然"""
        try:
            if 'SEM' in self.results and 'average_r_squared' in self.results['SEM']:
                return self.results['SEM']['average_r_squared']
            else:
                # 重新计算
                return self._calculate_sem_likelihood_for_edge_removal()
        except Exception as e:
            print(f"获取SEM似然失败: {e}")
            return 0
    
    def calculate_all_methods_gains(self):
        """计算所有方法的边级似然增益"""
        all_gains = {}
        methods = ['MLE', 'Bayesian', 'EM', 'SEM']
        
        for method in methods:
            if method in self.results:
                gains = self.calculate_edge_likelihood_gain(method)
                if gains:
                    all_gains[method] = gains
                    print(f"{method} 方法边级似然增益计算完成")
            else:
                print(f"跳过 {method} 方法（结果不存在）")
        
        return all_gains
    
    def create_output_folder(self):
        """创建输出文件夹"""
        # 使用当前脚本所在目录作为基础目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_folder = os.path.join(base_dir, "05边级似然增益结果")
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"创建输出文件夹: {output_folder}")
        
        return output_folder
    
    def save_results(self, all_gains):
        """保存边级似然增益结果"""
        output_folder = self.create_output_folder()
        
        # 保存总体结果
        results_file = os.path.join(output_folder, "边级似然增益结果.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_gains, f, ensure_ascii=False, indent=2)
        
        # 创建汇总CSV
        self._create_summary_csv(all_gains, output_folder)
        
        # 创建详细报告
        self._create_detailed_report(all_gains, output_folder)
        
        # 创建可视化
        self._create_visualizations(all_gains, output_folder)
        
        print(f"边级似然增益结果已保存到: {output_folder}")
    
    def _create_summary_csv(self, all_gains, output_folder):
        """创建汇总CSV文件"""
        try:
            summary_data = []
            
            # 收集所有边
            all_edges = set()
            for method_gains in all_gains.values():
                all_edges.update(method_gains.keys())
            
            for edge in sorted(all_edges):
                row = {'Edge': edge}
                for method in ['MLE', 'Bayesian', 'EM', 'SEM']:
                    if method in all_gains and edge in all_gains[method]:
                        row[f'{method}_Gain'] = all_gains[method][edge]['likelihood_gain']
                        row[f'{method}_S_param'] = all_gains[method][edge]['S_param']
                    else:
                        row[f'{method}_Gain'] = None
                        row[f'{method}_S_param'] = None
                summary_data.append(row)
            
            df = pd.DataFrame(summary_data)
            csv_file = os.path.join(output_folder, "边级似然增益汇总.csv")
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            print(f"汇总CSV已保存: {csv_file}")
            
        except Exception as e:
            print(f"创建汇总CSV失败: {e}")
    
    def _create_detailed_report(self, all_gains, output_folder):
        """创建详细报告"""
        try:
            report_file = os.path.join(output_folder, "边级似然增益详细报告.txt")
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("边级似然增益详细报告\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for method in ['MLE', 'Bayesian', 'EM', 'SEM']:
                    if method in all_gains:
                        f.write(f"\n{method} 方法边级似然增益:\n")
                        f.write("-" * 30 + "\n")
                        
                        gains = all_gains[method]
                        sorted_edges = sorted(gains.items(), 
                                            key=lambda x: x[1]['S_param'], 
                                            reverse=True)
                        
                        for edge, info in sorted_edges:
                            f.write(f"边: {edge}\n")
                            f.write(f"  似然增益: {info['likelihood_gain']:.6f}\n")
                            f.write(f"  归一化分数: {info['S_param']:.6f}\n")
                            f.write(f"  完整似然: {info['full_likelihood']:.6f}\n")
                            f.write(f"  移除后似然: {info['drop_likelihood']:.6f}\n\n")
            
            print(f"详细报告已保存: {report_file}")
            
        except Exception as e:
            print(f"创建详细报告失败: {e}")
    
    def _create_visualizations(self, all_gains, output_folder):
        """创建可视化图表"""
        try:
            # 边级似然增益对比图
            plt.figure(figsize=(12, 8))
            
            methods = [m for m in ['MLE', 'Bayesian', 'EM', 'SEM'] if m in all_gains]
            edges = list(set().union(*[gains.keys() for gains in all_gains.values()]))
            
            x = np.arange(len(edges))
            width = 0.2
            
            for i, method in enumerate(methods):
                gains_values = []
                for edge in edges:
                    if edge in all_gains[method]:
                        gains_values.append(all_gains[method][edge]['S_param'])
                    else:
                        gains_values.append(0)
                
                plt.bar(x + i * width, gains_values, width, label=method)
            
            plt.xlabel('边')
            plt.ylabel('归一化似然增益分数')
            plt.title('各方法边级似然增益对比')
            plt.xticks(x + width * 1.5, edges, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            
            plot_file = os.path.join(output_folder, "边级似然增益对比图.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 热力图
            if len(methods) > 1:
                plt.figure(figsize=(10, 8))
                
                heatmap_data = []
                for edge in edges:
                    row = []
                    for method in methods:
                        if method in all_gains and edge in all_gains[method]:
                            row.append(all_gains[method][edge]['S_param'])
                        else:
                            row.append(0)
                    heatmap_data.append(row)
                
                heatmap_df = pd.DataFrame(heatmap_data, index=edges, columns=methods)
                
                sns.heatmap(heatmap_df, annot=True, cmap='YlOrRd', fmt='.3f')
                plt.title('边级似然增益热力图')
                plt.tight_layout()
                
                heatmap_file = os.path.join(output_folder, "边级似然增益热力图.png")
                plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
                plt.close()
            
            print("可视化图表已生成")
            
        except Exception as e:
            print(f"创建可视化失败: {e}")

def main():
    """主函数"""
    print("开始边级似然增益计算...")
    
    # 数据文件路径
    data_file = "/home/zkr/临时/01数据预处理/缩减数据_规格.csv"
    
    # 创建计算器实例
    calculator = EdgeLikelihoodGainCalculator(data_file)
    
    # 加载数据和因果边
    if not calculator.load_data():
        print("数据加载失败，程序退出")
        return
    
    if not calculator.load_causal_edges():
        print("因果边加载失败，程序退出")
        return
    
    # 数据预处理
    if not calculator.preprocess_data():
        print("数据预处理失败，程序退出")
        return
    
    # 加载各方法结果
    calculator.load_method_results()
    
    # 计算所有方法的边级似然增益
    all_gains = calculator.calculate_all_methods_gains()
    
    if all_gains:
        # 保存结果
        calculator.save_results(all_gains)
        print("边级似然增益计算完成！")
    else:
        print("未能计算出任何边级似然增益结果")

if __name__ == "__main__":
    main()