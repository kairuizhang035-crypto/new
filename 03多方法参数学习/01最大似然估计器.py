#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01最大似然估计器 (MLE Parameter Estimator)
实现最大似然估计方法进行参数学习
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

class MLEParameterEstimator:
    """
    最大似然估计参数学习器
    """
    
    def __init__(self, data_file=None):
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

    def mle_estimation(self):
        """MLE参数估计"""
        print("\n开始MLE参数估计...")
        
        try:
            cpts = {}
            total_log_likelihood = 0
            
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
            
            for node in nodes_to_process:
                parents = list(self.graph.predecessors(node))
                
                if not parents:
                    # 边际概率
                    counts = self.data[node].value_counts().sort_index()
                    total_count = len(self.data)
                    probs = [counts.get(i, 0) / total_count for i in [0, 1]]
                    
                    cpts[node] = {
                        'type': 'marginal',
                        'parents': [],
                        'probabilities': probs
                    }
                    
                    # 计算对数似然
                    for i, prob in enumerate(probs):
                        if prob > 0:
                            count = counts.get(i, 0)
                            total_log_likelihood += count * np.log(prob)
                
                else:
                    # 条件概率
                    parent_combinations = list(product([0, 1], repeat=len(parents)))
                    conditional_probs = {}
                    
                    for combo in parent_combinations:
                        condition = dict(zip(parents, combo))
                        mask = pd.Series([True] * len(self.data))
                        
                        for parent, value in condition.items():
                            mask &= (self.data[parent] == value)
                        
                        subset = self.data[mask]
                        
                        if len(subset) > 0:
                            counts = subset[node].value_counts().sort_index()
                            total_count = len(subset)
                            probs = [counts.get(i, 0) / total_count for i in [0, 1]]
                        else:
                            probs = [0.5, 0.5]
                        
                        key = ','.join(map(str, combo))
                        conditional_probs[key] = probs
                        
                        # 计算对数似然
                        for i, prob in enumerate(probs):
                            if prob > 0:
                                count = counts.get(i, 0) if len(subset) > 0 else 0
                                if count > 0:
                                    total_log_likelihood += count * np.log(prob)
                    
                    cpts[node] = {
                        'type': 'conditional',
                        'parents': parents,
                        'probabilities': conditional_probs
                    }
            
            self.results['MLE'] = {
                'cpts': cpts,
                'log_likelihood': total_log_likelihood,
                'method': 'Maximum Likelihood Estimation',
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"MLE估计完成，对数似然: {total_log_likelihood:.4f}")
            return True
            
        except Exception as e:
            print(f"MLE估计失败: {e}")
            return False

    def create_output_folder(self):
        """创建输出文件夹"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_folder = os.path.join(script_dir, "01MLE_CPT结果")
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"输出目录: {self.output_folder}")

    def save_results(self):
        """保存结果"""
        print("\n保存MLE结果...")
        
        if 'MLE' not in self.results:
            print("没有MLE结果可保存")
            return
        
        result = self.results['MLE']
        
        # 保存CPT为JSON格式
        cpt_file = os.path.join(self.output_folder, "MLE_CPTs.json")
        with open(cpt_file, 'w', encoding='utf-8') as f:
            json.dump(result['cpts'], f, ensure_ascii=False, indent=2, default=str)
        
        # 保存为pickle格式
        pkl_file = os.path.join(self.output_folder, "MLE_CPTs.pkl")
        with open(pkl_file, 'wb') as f:
            pickle.dump(result['cpts'], f)
        
        # 创建汇总CSV
        self.create_summary_csv()
        
        # 创建详细结果文件
        self.create_detailed_results()
        
        print(f"MLE结果已保存到: {self.output_folder}")

    def create_summary_csv(self):
        """创建条件概率表汇总CSV"""
        try:
            result = self.results['MLE']
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
                        'P(0)': f"{probs[0]:.4f}",
                        'P(1)': f"{probs[1]:.4f}",
                        '条件': '无条件'
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
                            'P(0)': f"{probs[0]:.4f}",
                            'P(1)': f"{probs[1]:.4f}",
                            '条件': condition
                        })
            
            df_summary = pd.DataFrame(summary_data)
            summary_file = os.path.join(self.output_folder, "MLE_条件概率表汇总.csv")
            df_summary.to_csv(summary_file, index=False, encoding='utf-8')
            
            print(f"汇总CSV已保存: {summary_file}")
            
        except Exception as e:
            print(f"创建汇总CSV失败: {e}")

    def create_detailed_results(self):
        """创建详细结果文件"""
        try:
            result = self.results['MLE']
            
            # 创建详细文本报告
            report_file = os.path.join(self.output_folder, "MLE_条件概率表详细结果.txt")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=== MLE参数估计详细结果 ===\n\n")
                f.write(f"估计方法: {result['method']}\n")
                f.write(f"时间戳: {result['timestamp']}\n")
                f.write(f"对数似然: {result['log_likelihood']:.6f}\n\n")
                
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
                            f.write(f"P({node}=0|{condition_str}) = {probs[0]:.6f}\n")
                            f.write(f"P({node}=1|{condition_str}) = {probs[1]:.6f}\n")
                    
                    f.write("\n" + "-"*50 + "\n\n")
            
            # 创建详细数据CSV
            detailed_data = []
            cpts = result['cpts']
            
            for node, cpt_info in cpts.items():
                if cpt_info['type'] == 'marginal':
                    probs = cpt_info['probabilities']
                    detailed_data.append({
                        '节点': node,
                        '父节点': '',
                        '条件': '无条件',
                        '目标值': 0,
                        '概率': probs[0]
                    })
                    detailed_data.append({
                        '节点': node,
                        '父节点': '',
                        '条件': '无条件',
                        '目标值': 1,
                        '概率': probs[1]
                    })
                else:
                    parents = cpt_info['parents']
                    parent_str = ', '.join(parents)
                    
                    for condition, probs in cpt_info['probabilities'].items():
                        detailed_data.append({
                            '节点': node,
                            '父节点': parent_str,
                            '条件': condition,
                            '目标值': 0,
                            '概率': probs[0]
                        })
                        detailed_data.append({
                            '节点': node,
                            '父节点': parent_str,
                            '条件': condition,
                            '目标值': 1,
                            '概率': probs[1]
                        })
            
            df_detailed = pd.DataFrame(detailed_data)
            detailed_file = os.path.join(self.output_folder, "MLE_条件概率表详细数据.csv")
            df_detailed.to_csv(detailed_file, index=False, encoding='utf-8')
            
            print(f"详细结果已保存: {report_file}")
            print(f"详细数据CSV已保存: {detailed_file}")
            
        except Exception as e:
            print(f"创建详细结果失败: {e}")

    def run(self):
        """运行MLE参数估计"""
        print("=== MLE参数估计器 ===")
        
        # 加载数据
        if not self.load_data():
            print("数据加载失败，程序退出")
            return False
        
        # 数据预处理
        self.preprocess_data()
        
        # 加载因果边
        self.load_causal_edges()
        
        # 运行MLE估计
        if not self.mle_estimation():
            print("MLE估计失败")
            return False
        
        # 保存结果
        self.create_output_folder()
        self.save_results()
        
        print(f"\n=== MLE参数估计完成 ===")
        print(f"结果已保存到: {self.output_folder}")
        return True

def main():
    estimator = MLEParameterEstimator()
    estimator.run()

if __name__ == "__main__":
    main()