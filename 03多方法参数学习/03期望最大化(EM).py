#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03期望最大化(EM) (Expectation-Maximization Parameter Estimator)
实现EM算法进行参数学习
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

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.family'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class EMParameterEstimator:
    """
    期望最大化(EM)参数估计器
    """
    
    def __init__(self, data_file=None, max_iterations=100, tolerance=1e-6):
        # 设置数据文件路径
        if data_file is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            self.data_file = os.path.join(parent_dir, '01数据预处理/缩减数据_规格.csv')
        else:
            self.data_file = data_file
        self.data = None
        self.causal_edges = []
        self.graph = nx.DiGraph()
        self.results = {}
        self.output_folder = None
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
    def load_data(self):
        """加载数据文件"""
        try:
            print(f"正在加载数据文件: {self.data_file}")
            if os.path.exists(self.data_file):
                self.data = pd.read_csv(self.data_file, encoding='utf-8')
                print(f"成功加载数据: {self.data.shape}")
                return True
            else:
                print(f"数据文件不存在: {self.data_file}")
                return False
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False

    def load_causal_edges(self):
        """加载因果边"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        
        edge_file = os.path.join(parent_dir, '02因果发现/06候选因果边集合/精简因果边列表.csv')
        
        if os.path.exists(edge_file):
            try:
                print(f"正在加载因果边: {edge_file}")
                df = pd.read_csv(edge_file, encoding='utf-8')
                
                # 根据文件格式确定列名
                if '源节点' in df.columns and '目标节点' in df.columns:
                    source_col, target_col = '源节点', '目标节点'
                elif '父节点' in df.columns and '子节点' in df.columns:
                    source_col, target_col = '父节点', '子节点'
                elif 'Source' in df.columns and 'Target' in df.columns:
                    source_col, target_col = 'Source', 'Target'
                else:
                    print(f"未识别的列格式: {df.columns.tolist()}")
                    return False
                
                data_columns = set(self.data.columns)
                edges_added = 0
                
                for _, row in df.iterrows():
                    source = str(row[source_col]).strip()
                    target = str(row[target_col]).strip()
                    
                    if source in data_columns and target in data_columns:
                        self.causal_edges.append((source, target))
                        self.graph.add_edge(source, target)
                        edges_added += 1
                
                print(f"成功加载 {edges_added} 条有效因果边")
                return True
                
            except Exception as e:
                print(f"加载因果边失败 {edge_file}: {e}")
                return False
        else:
            print(f"因果边文件不存在: {edge_file}")
            return False
    
    def preprocess_data(self):
        """数据预处理：排除ID列并二值化"""
        print("正在进行数据预处理...")
        
        # 排除ID列和其他非特征列
        exclude_columns = ['RECORD_ID', 'ID', 'id', 'record_id']
        columns_to_drop = [col for col in self.data.columns if col in exclude_columns]
        
        if columns_to_drop:
            print(f"排除列: {columns_to_drop}")
            self.data = self.data.drop(columns=columns_to_drop)
        
        # 二值化处理
        for col in self.data.columns:
            if self.data[col].dtype in ['float64', 'int64']:
                median_val = self.data[col].median()
                self.data[col] = (self.data[col] > median_val).astype(int)
        
        print(f"数据预处理完成，最终数据维度: {self.data.shape}")

    def em_estimation(self):
        """EM算法参数估计"""
        print(f"\n开始EM算法参数估计 (最大迭代次数: {self.max_iterations}, 收敛阈值: {self.tolerance})...")
        
        try:
            # 获取图中的所有节点
            graph_nodes = set(self.graph.nodes())
            
            # 如果图为空，使用所有数据列
            if not graph_nodes:
                nodes_to_process = self.data.columns
                print("图为空，处理所有数据列")
            else:
                # 只处理图中存在的节点
                nodes_to_process = [node for node in self.data.columns if node in graph_nodes]
                print(f"处理图中的 {len(nodes_to_process)} 个节点")
            
            # 初始化参数
            cpts = self._initialize_parameters(nodes_to_process)
            
            # EM迭代
            log_likelihood_history = []
            
            for iteration in range(self.max_iterations):
                # E步：计算期望
                expected_counts = self._e_step(cpts, nodes_to_process)
                
                # M步：更新参数
                new_cpts = self._m_step(expected_counts, nodes_to_process)
                
                # 计算对数似然
                log_likelihood = self._calculate_likelihood(new_cpts, nodes_to_process)
                log_likelihood_history.append(log_likelihood)
                
                # 检查收敛
                if iteration > 0:
                    improvement = log_likelihood - log_likelihood_history[-2]
                    print(f"迭代 {iteration + 1}: 对数似然 = {log_likelihood:.6f}, 改进 = {improvement:.6f}")
                    
                    if abs(improvement) < self.tolerance:
                        print(f"在第 {iteration + 1} 次迭代后收敛")
                        break
                else:
                    print(f"迭代 {iteration + 1}: 对数似然 = {log_likelihood:.6f}")
                
                cpts = new_cpts
            
            self.results['EM'] = {
                'cpts': cpts,
                'method': 'Expectation-Maximization',
                'iterations': iteration + 1,
                'final_log_likelihood': log_likelihood,
                'log_likelihood_history': log_likelihood_history,
                'max_iterations': self.max_iterations,
                'tolerance': self.tolerance,
                'converged': abs(improvement) < self.tolerance if iteration > 0 else False,
                'timestamp': datetime.now().isoformat()
            }
            
            print("EM算法估计完成")
            return True
            
        except Exception as e:
            print(f"EM算法估计失败: {e}")
            return False

    def _initialize_parameters(self, nodes_to_process):
        """初始化参数"""
        cpts = {}
        
        for node in nodes_to_process:
            parents = list(self.graph.predecessors(node))
            
            if not parents:
                # 边际概率：随机初始化
                prob_1 = np.random.uniform(0.3, 0.7)
                cpts[node] = {
                    'type': 'marginal',
                    'parents': [],
                    'probabilities': [1 - prob_1, prob_1]
                }
            else:
                # 条件概率：随机初始化
                parent_combinations = list(product([0, 1], repeat=len(parents)))
                conditional_probs = {}
                
                for combo in parent_combinations:
                    prob_1 = np.random.uniform(0.3, 0.7)
                    key = ','.join(map(str, combo))
                    conditional_probs[key] = [1 - prob_1, prob_1]
                
                cpts[node] = {
                    'type': 'conditional',
                    'parents': parents,
                    'probabilities': conditional_probs
                }
        
        return cpts

    def _e_step(self, cpts, nodes_to_process):
        """E步：计算期望计数"""
        expected_counts = {}
        
        for node in nodes_to_process:
            parents = list(self.graph.predecessors(node))
            
            if not parents:
                # 边际概率的期望计数
                counts = [0.0, 0.0]
                for _, row in self.data.iterrows():
                    value = int(row[node])
                    counts[value] += 1.0
                
                expected_counts[node] = {
                    'type': 'marginal',
                    'counts': counts
                }
            else:
                # 条件概率的期望计数
                parent_combinations = list(product([0, 1], repeat=len(parents)))
                conditional_counts = {}
                
                for combo in parent_combinations:
                    key = ','.join(map(str, combo))
                    conditional_counts[key] = [0.0, 0.0]
                
                for _, row in self.data.iterrows():
                    # 获取父节点的值
                    parent_values = [int(row[parent]) for parent in parents]
                    combo_key = ','.join(map(str, parent_values))
                    
                    # 获取子节点的值
                    child_value = int(row[node])
                    conditional_counts[combo_key][child_value] += 1.0
                
                expected_counts[node] = {
                    'type': 'conditional',
                    'parents': parents,
                    'counts': conditional_counts
                }
        
        return expected_counts

    def _m_step(self, expected_counts, nodes_to_process):
        """M步：更新参数"""
        new_cpts = {}
        
        for node in nodes_to_process:
            count_info = expected_counts[node]
            
            if count_info['type'] == 'marginal':
                # 更新边际概率
                counts = count_info['counts']
                total = sum(counts)
                
                if total > 0:
                    probs = [count / total for count in counts]
                else:
                    probs = [0.5, 0.5]
                
                new_cpts[node] = {
                    'type': 'marginal',
                    'parents': [],
                    'probabilities': probs
                }
            else:
                # 更新条件概率
                parents = count_info['parents']
                conditional_probs = {}
                
                for combo_key, counts in count_info['counts'].items():
                    total = sum(counts)
                    
                    if total > 0:
                        probs = [count / total for count in counts]
                    else:
                        probs = [0.5, 0.5]
                    
                    conditional_probs[combo_key] = probs
                
                new_cpts[node] = {
                    'type': 'conditional',
                    'parents': parents,
                    'probabilities': conditional_probs
                }
        
        return new_cpts

    def _calculate_likelihood(self, cpts, nodes_to_process):
        """计算对数似然"""
        log_likelihood = 0.0
        
        for _, row in self.data.iterrows():
            row_likelihood = 0.0
            
            for node in nodes_to_process:
                if node not in cpts:
                    continue
                
                cpt_info = cpts[node]
                node_value = int(row[node])
                
                if cpt_info['type'] == 'marginal':
                    prob = cpt_info['probabilities'][node_value]
                else:
                    parents = cpt_info['parents']
                    parent_values = [int(row[parent]) for parent in parents]
                    combo_key = ','.join(map(str, parent_values))
                    prob = cpt_info['probabilities'][combo_key][node_value]
                
                # 避免对数为负无穷
                if prob > 0:
                    row_likelihood += np.log(prob)
                else:
                    row_likelihood += np.log(1e-10)
            
            log_likelihood += row_likelihood
        
        return log_likelihood

    def create_output_folder(self):
        """创建输出文件夹"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_folder = os.path.join(script_dir, "03EM_CPT结果")
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"输出目录: {self.output_folder}")

    def save_results(self):
        """保存结果"""
        print("\n保存EM算法估计结果...")
        
        if 'EM' not in self.results:
            print("没有EM算法估计结果可保存")
            return
        
        result = self.results['EM']
        
        # 保存CPT为JSON格式
        cpt_file = os.path.join(self.output_folder, "EM_CPTs.json")
        with open(cpt_file, 'w', encoding='utf-8') as f:
            json.dump(result['cpts'], f, ensure_ascii=False, indent=2, default=str)
        
        # 保存为pickle格式
        pkl_file = os.path.join(self.output_folder, "EM_CPTs.pkl")
        with open(pkl_file, 'wb') as f:
            pickle.dump(result['cpts'], f)
        
        # 创建汇总CSV
        self.create_summary_csv()
        
        # 创建详细结果文件
        self.create_detailed_results()
        
        # 创建收敛图
        self.create_convergence_plot()
        
        print(f"EM算法估计结果已保存到: {self.output_folder}")

    def create_summary_csv(self):
        """创建条件概率表汇总CSV"""
        try:
            result = self.results['EM']
            cpts = result['cpts']
            
            summary_data = []
            
            for node, cpt_info in cpts.items():
                if cpt_info['type'] == 'marginal':
                    # 边际概率
                    probs = cpt_info['probabilities']
                    summary_data.append({
                        '节点': node,
                        '类型': '边际概率',
                        '父节点': '',
                        'P(0)': f"{probs[0]:.6f}",
                        'P(1)': f"{probs[1]:.6f}",
                        '条件': '无条件',
                        '迭代次数': result['iterations'],
                        '最终对数似然': f"{result['final_log_likelihood']:.6f}",
                        '是否收敛': '是' if result['converged'] else '否'
                    })
                else:
                    # 条件概率
                    parents = cpt_info['parents']
                    parent_str = ', '.join(parents)
                    
                    for condition, probs in cpt_info['probabilities'].items():
                        summary_data.append({
                            '节点': node,
                            '类型': '条件概率',
                            '父节点': parent_str,
                            'P(0)': f"{probs[0]:.6f}",
                            'P(1)': f"{probs[1]:.6f}",
                            '条件': condition,
                            '迭代次数': result['iterations'],
                            '最终对数似然': f"{result['final_log_likelihood']:.6f}",
                            '是否收敛': '是' if result['converged'] else '否'
                        })
            
            df_summary = pd.DataFrame(summary_data)
            summary_file = os.path.join(self.output_folder, "EM_条件概率表汇总.csv")
            df_summary.to_csv(summary_file, index=False, encoding='utf-8')
            
            print(f"汇总CSV已保存: {summary_file}")
            
        except Exception as e:
            print(f"创建汇总CSV失败: {e}")

    def create_detailed_results(self):
        """创建详细结果文件"""
        try:
            result = self.results['EM']
            
            # 创建详细文本报告
            report_file = os.path.join(self.output_folder, "EM_条件概率表详细结果.txt")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=== EM算法参数估计详细结果 ===\n\n")
                f.write(f"估计方法: {result['method']}\n")
                f.write(f"时间戳: {result['timestamp']}\n")
                f.write(f"最大迭代次数: {result['max_iterations']}\n")
                f.write(f"收敛阈值: {result['tolerance']}\n")
                f.write(f"实际迭代次数: {result['iterations']}\n")
                f.write(f"最终对数似然: {result['final_log_likelihood']:.6f}\n")
                f.write(f"是否收敛: {'是' if result['converged'] else '否'}\n\n")
                
                # 对数似然历史
                f.write("对数似然历史:\n")
                for i, ll in enumerate(result['log_likelihood_history']):
                    f.write(f"迭代 {i+1}: {ll:.6f}\n")
                f.write("\n")
                
                cpts = result['cpts']
                f.write(f"条件概率表数量: {len(cpts)}\n\n")
                
                for node, cpt_info in cpts.items():
                    f.write(f"节点: {node}\n")
                    f.write(f"类型: {cpt_info['type']}\n")
                    
                    if cpt_info['type'] == 'marginal':
                        probs = cpt_info['probabilities']
                        f.write(f"P({node}=0) = {probs[0]:.6f}\n")
                        f.write(f"P({node}=1) = {probs[1]:.6f}\n")
                    else:
                        parents = cpt_info['parents']
                        f.write(f"父节点: {', '.join(parents)}\n")
                        
                        for condition, probs in cpt_info['probabilities'].items():
                            parent_values = condition.split(',')
                            condition_str = ', '.join([f"{p}={v}" for p, v in zip(parents, parent_values)])
                            f.write(f"\n条件: {condition_str}\n")
                            f.write(f"P({node}=0|{condition_str}) = {probs[0]:.6f}\n")
                            f.write(f"P({node}=1|{condition_str}) = {probs[1]:.6f}\n")
                    
                    f.write("\n" + "-"*50 + "\n\n")
            
            print(f"详细结果已保存: {report_file}")
            
        except Exception as e:
            print(f"创建详细结果失败: {e}")

    def create_convergence_plot(self):
        """创建收敛图"""
        try:
            result = self.results['EM']
            log_likelihood_history = result['log_likelihood_history']
            
            if len(log_likelihood_history) > 1:
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(log_likelihood_history) + 1), log_likelihood_history, 'b-o', linewidth=2, markersize=4)
                plt.xlabel('迭代次数')
                plt.ylabel('对数似然')
                plt.title('EM算法收敛过程')
                plt.grid(True, alpha=0.3)
                
                # 添加收敛信息
                final_ll = log_likelihood_history[-1]
                plt.axhline(y=final_ll, color='r', linestyle='--', alpha=0.7, label=f'最终对数似然: {final_ll:.2f}')
                plt.legend()
                
                plot_file = os.path.join(self.output_folder, "EM_收敛过程.png")
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"收敛图已保存: {plot_file}")
            
        except Exception as e:
            print(f"创建收敛图失败: {e}")

    def run(self):
        """运行EM算法参数估计"""
        print("=== EM算法参数估计器 ===")
        
        # 加载数据
        if not self.load_data():
            print("数据加载失败，程序退出")
            return False
        
        # 数据预处理
        self.preprocess_data()
        
        # 加载因果边
        self.load_causal_edges()
        
        # 运行EM算法估计
        if not self.em_estimation():
            print("EM算法估计失败")
            return False
        
        # 保存结果
        self.create_output_folder()
        self.save_results()
        
        print(f"\n=== EM算法参数估计完成 ===")
        print(f"结果已保存到: {self.output_folder}")
        return True

def main():
    # 可以通过参数调整最大迭代次数和收敛阈值
    estimator = EMParameterEstimator(max_iterations=100, tolerance=1e-6)
    estimator.run()

if __name__ == "__main__":
    main()