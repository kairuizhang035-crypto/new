#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04结构方程模型估计器 (Structural Equation Model Parameter Estimator)
实现结构方程模型(SEM)进行参数学习
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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.family'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SEMParameterEstimator:
    """
    结构方程模型(SEM)参数估计器
    """
    
    def __init__(self, data_file=None, standardize=True):
        # 设置数据文件路径
        if data_file is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            self.data_file = os.path.join(parent_dir, '01数据预处理/缩减数据_规格.csv')
        else:
            self.data_file = data_file
        self.data = None
        self.original_data = None
        self.causal_edges = []
        self.graph = nx.DiGraph()
        self.results = {}
        self.output_folder = None
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None
        
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
        
        # 保存原始数据
        self.original_data = self.data.copy()
        
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
        
        # 标准化处理（如果需要）
        if self.standardize and self.scaler is not None:
            print("进行数据标准化...")
            scaled_data = self.scaler.fit_transform(self.data)
            self.data = pd.DataFrame(scaled_data, columns=self.data.columns, index=self.data.index)
        
        print(f"数据预处理完成，最终数据维度: {self.data.shape}")

    def sem_estimation(self):
        """结构方程模型参数估计"""
        print(f"\n开始SEM参数估计 (标准化: {'是' if self.standardize else '否'})...")
        
        try:
            sem_results = {}
            model_fit_stats = {}
            
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
            
            total_r2 = 0
            total_nodes_with_parents = 0
            
            for node in nodes_to_process:
                parents = list(self.graph.predecessors(node))
                
                if parents:
                    # 确保所有父节点都在数据中
                    valid_parents = [p for p in parents if p in self.data.columns]
                    if not valid_parents:
                        continue
                    
                    X = self.data[valid_parents].values
                    y = self.data[node].values
                    
                    # 拟合线性模型
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # 计算预测值和残差
                    y_pred = model.predict(X)
                    residuals = y - y_pred
                    
                    # 计算各种统计量
                    r2 = model.score(X, y)
                    adjusted_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - len(valid_parents) - 1)
                    mse = np.mean(residuals ** 2)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(residuals))
                    
                    # 计算系数的标准误差（简化版）
                    residual_var = np.var(residuals, ddof=len(valid_parents) + 1)
                    X_with_intercept = np.column_stack([np.ones(len(X)), X])
                    try:
                        cov_matrix = residual_var * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
                        std_errors = np.sqrt(np.diag(cov_matrix))
                        intercept_se = std_errors[0]
                        coef_se = std_errors[1:].tolist()
                    except:
                        intercept_se = np.nan
                        coef_se = [np.nan] * len(valid_parents)
                    
                    # 计算t统计量和p值（简化版）
                    try:
                        t_intercept = model.intercept_ / intercept_se if not np.isnan(intercept_se) else np.nan
                        t_coefs = [coef / se for coef, se in zip(model.coef_, coef_se) if not np.isnan(se)]
                    except:
                        t_intercept = np.nan
                        t_coefs = [np.nan] * len(valid_parents)
                    
                    sem_results[node] = {
                        'type': 'structural_equation',
                        'parents': valid_parents,
                        'coefficients': model.coef_.tolist(),
                        'intercept': float(model.intercept_),
                        'coefficient_std_errors': coef_se,
                        'intercept_std_error': float(intercept_se) if not np.isnan(intercept_se) else None,
                        't_statistics': {
                            'intercept': float(t_intercept) if not np.isnan(t_intercept) else None,
                            'coefficients': t_coefs
                        },
                        'r_squared': float(r2),
                        'adjusted_r_squared': float(adjusted_r2),
                        'residual_variance': float(np.var(residuals)),
                        'mse': float(mse),
                        'rmse': float(rmse),
                        'mae': float(mae),
                        'sample_size': len(y),
                        'degrees_of_freedom': len(y) - len(valid_parents) - 1
                    }
                    
                    total_r2 += r2
                    total_nodes_with_parents += 1
                    
                else:
                    # 无父节点的情况（外生变量）
                    mean_val = float(self.data[node].mean())
                    var_val = float(self.data[node].var())
                    std_val = float(self.data[node].std())
                    
                    sem_results[node] = {
                        'type': 'exogenous_variable',
                        'parents': [],
                        'mean': mean_val,
                        'variance': var_val,
                        'standard_deviation': std_val,
                        'sample_size': len(self.data[node])
                    }
            
            # 计算整体模型拟合统计量
            avg_r2 = total_r2 / max(total_nodes_with_parents, 1)
            
            model_fit_stats = {
                'average_r_squared': float(avg_r2),
                'total_equations': total_nodes_with_parents,
                'total_exogenous_variables': len(nodes_to_process) - total_nodes_with_parents,
                'total_parameters': sum([len(eq.get('parents', [])) + 1 for eq in sem_results.values() if eq['type'] == 'structural_equation']),
                'sample_size': len(self.data)
            }
            
            self.results['SEM'] = {
                'structural_equations': sem_results,
                'model_fit_statistics': model_fit_stats,
                'method': 'Structural Equation Model',
                'standardized': self.standardize,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"SEM估计完成，平均R²: {avg_r2:.4f}")
            return True
            
        except Exception as e:
            print(f"SEM估计失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_output_folder(self):
        """创建输出文件夹"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_folder = os.path.join(script_dir, "04SEM_结果")
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"输出目录: {self.output_folder}")

    def save_results(self):
        """保存结果"""
        print("\n保存SEM估计结果...")
        
        if 'SEM' not in self.results:
            print("没有SEM估计结果可保存")
            return
        
        result = self.results['SEM']
        
        # 保存结构方程为JSON格式
        sem_file = os.path.join(self.output_folder, "SEM_结构方程.json")
        with open(sem_file, 'w', encoding='utf-8') as f:
            json.dump(result['structural_equations'], f, ensure_ascii=False, indent=2, default=str)
        
        # 保存为pickle格式
        pkl_file = os.path.join(self.output_folder, "SEM_结构方程.pkl")
        with open(pkl_file, 'wb') as f:
            pickle.dump(result['structural_equations'], f)
        
        # 创建汇总CSV
        self.create_summary_csv()
        
        # 创建详细结果文件
        self.create_detailed_results()
        
        # 创建系数表
        self.create_coefficient_table()
        
        # 创建模型拟合报告
        self.create_model_fit_report()
        
        print(f"SEM估计结果已保存到: {self.output_folder}")

    def create_summary_csv(self):
        """创建结构方程汇总CSV"""
        try:
            result = self.results['SEM']
            equations = result['structural_equations']
            
            summary_data = []
            
            for node, eq_info in equations.items():
                if eq_info['type'] == 'structural_equation':
                    # 结构方程
                    parents = eq_info['parents']
                    parent_str = ', '.join(parents)
                    
                    summary_data.append({
                        '节点': node,
                        '类型': '结构方程',
                        '父节点': parent_str,
                        '截距': f"{eq_info['intercept']:.6f}",
                        'R²': f"{eq_info['r_squared']:.6f}",
                        '调整R²': f"{eq_info['adjusted_r_squared']:.6f}",
                        'RMSE': f"{eq_info['rmse']:.6f}",
                        'MAE': f"{eq_info['mae']:.6f}",
                        '样本量': eq_info['sample_size'],
                        '自由度': eq_info['degrees_of_freedom'],
                        '系数数量': len(parents)
                    })
                else:
                    # 外生变量
                    summary_data.append({
                        '节点': node,
                        '类型': '外生变量',
                        '父节点': '',
                        '截距': '',
                        'R²': '',
                        '调整R²': '',
                        'RMSE': '',
                        'MAE': '',
                        '样本量': eq_info['sample_size'],
                        '自由度': '',
                        '系数数量': 0,
                        '均值': f"{eq_info['mean']:.6f}",
                        '方差': f"{eq_info['variance']:.6f}",
                        '标准差': f"{eq_info['standard_deviation']:.6f}"
                    })
            
            df_summary = pd.DataFrame(summary_data)
            summary_file = os.path.join(self.output_folder, "SEM_结构方程汇总.csv")
            df_summary.to_csv(summary_file, index=False, encoding='utf-8')
            
            print(f"汇总CSV已保存: {summary_file}")
            
        except Exception as e:
            print(f"创建汇总CSV失败: {e}")

    def create_coefficient_table(self):
        """创建系数表"""
        try:
            result = self.results['SEM']
            equations = result['structural_equations']
            
            coef_data = []
            
            for node, eq_info in equations.items():
                if eq_info['type'] == 'structural_equation':
                    parents = eq_info['parents']
                    coefficients = eq_info['coefficients']
                    coef_se = eq_info['coefficient_std_errors']
                    t_stats = eq_info['t_statistics']['coefficients']
                    
                    # 截距
                    intercept_se = eq_info['intercept_std_error']
                    intercept_t = eq_info['t_statistics']['intercept']
                    
                    coef_data.append({
                        '方程': node,
                        '变量': '(截距)',
                        '系数': f"{eq_info['intercept']:.6f}",
                        '标准误': f"{intercept_se:.6f}" if intercept_se is not None else 'N/A',
                        't统计量': f"{intercept_t:.4f}" if intercept_t is not None else 'N/A',
                        '变量类型': '截距'
                    })
                    
                    # 系数
                    for i, (parent, coef, se, t_stat) in enumerate(zip(parents, coefficients, coef_se, t_stats)):
                        coef_data.append({
                            '方程': node,
                            '变量': parent,
                            '系数': f"{coef:.6f}",
                            '标准误': f"{se:.6f}" if not np.isnan(se) else 'N/A',
                            't统计量': f"{t_stat:.4f}" if not np.isnan(t_stat) else 'N/A',
                            '变量类型': '解释变量'
                        })
            
            df_coef = pd.DataFrame(coef_data)
            coef_file = os.path.join(self.output_folder, "SEM_系数表.csv")
            df_coef.to_csv(coef_file, index=False, encoding='utf-8')
            
            print(f"系数表已保存: {coef_file}")
            
        except Exception as e:
            print(f"创建系数表失败: {e}")

    def create_detailed_results(self):
        """创建详细结果文件"""
        try:
            result = self.results['SEM']
            
            # 创建详细文本报告
            report_file = os.path.join(self.output_folder, "SEM_详细结果.txt")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=== 结构方程模型(SEM)参数估计详细结果 ===\n\n")
                f.write(f"估计方法: {result['method']}\n")
                f.write(f"时间戳: {result['timestamp']}\n")
                f.write(f"数据标准化: {'是' if result['standardized'] else '否'}\n\n")
                
                # 模型拟合统计
                fit_stats = result['model_fit_statistics']
                f.write("=== 模型拟合统计 ===\n")
                f.write(f"平均R²: {fit_stats['average_r_squared']:.6f}\n")
                f.write(f"结构方程数量: {fit_stats['total_equations']}\n")
                f.write(f"外生变量数量: {fit_stats['total_exogenous_variables']}\n")
                f.write(f"总参数数量: {fit_stats['total_parameters']}\n")
                f.write(f"样本量: {fit_stats['sample_size']}\n\n")
                
                equations = result['structural_equations']
                f.write(f"结构方程详细信息 (共 {len(equations)} 个节点):\n\n")
                
                for node, eq_info in equations.items():
                    f.write(f"节点: {node}\n")
                    f.write(f"类型: {eq_info['type']}\n")
                    
                    if eq_info['type'] == 'structural_equation':
                        parents = eq_info['parents']
                        f.write(f"父节点: {', '.join(parents)}\n")
                        f.write(f"方程: {node} = {eq_info['intercept']:.6f}")
                        
                        for parent, coef in zip(parents, eq_info['coefficients']):
                            f.write(f" + {coef:.6f}*{parent}")
                        f.write(" + ε\n")
                        
                        f.write(f"截距: {eq_info['intercept']:.6f}")
                        if eq_info['intercept_std_error'] is not None:
                            f.write(f" (SE: {eq_info['intercept_std_error']:.6f})")
                        f.write("\n")
                        
                        f.write("系数:\n")
                        for i, (parent, coef, se) in enumerate(zip(parents, eq_info['coefficients'], eq_info['coefficient_std_errors'])):
                            f.write(f"  {parent}: {coef:.6f}")
                            if not np.isnan(se):
                                f.write(f" (SE: {se:.6f})")
                            f.write("\n")
                        
                        f.write(f"R²: {eq_info['r_squared']:.6f}\n")
                        f.write(f"调整R²: {eq_info['adjusted_r_squared']:.6f}\n")
                        f.write(f"残差方差: {eq_info['residual_variance']:.6f}\n")
                        f.write(f"RMSE: {eq_info['rmse']:.6f}\n")
                        f.write(f"MAE: {eq_info['mae']:.6f}\n")
                        f.write(f"样本量: {eq_info['sample_size']}\n")
                        f.write(f"自由度: {eq_info['degrees_of_freedom']}\n")
                        
                    else:
                        f.write(f"均值: {eq_info['mean']:.6f}\n")
                        f.write(f"方差: {eq_info['variance']:.6f}\n")
                        f.write(f"标准差: {eq_info['standard_deviation']:.6f}\n")
                        f.write(f"样本量: {eq_info['sample_size']}\n")
                    
                    f.write("\n" + "-"*50 + "\n\n")
            
            print(f"详细结果已保存: {report_file}")
            
        except Exception as e:
            print(f"创建详细结果失败: {e}")

    def create_model_fit_report(self):
        """创建模型拟合报告"""
        try:
            result = self.results['SEM']
            fit_stats = result['model_fit_statistics']
            equations = result['structural_equations']
            
            # 收集所有R²值
            r2_values = []
            for eq_info in equations.values():
                if eq_info['type'] == 'structural_equation':
                    r2_values.append(eq_info['r_squared'])
            
            if r2_values:
                # 创建R²分布图
                plt.figure(figsize=(10, 6))
                plt.hist(r2_values, bins=min(10, len(r2_values)), alpha=0.7, edgecolor='black')
                plt.xlabel('R²值')
                plt.ylabel('频数')
                plt.title('结构方程R²分布')
                plt.axvline(np.mean(r2_values), color='red', linestyle='--', label=f'平均值: {np.mean(r2_values):.4f}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plot_file = os.path.join(self.output_folder, "SEM_R²分布.png")
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"R²分布图已保存: {plot_file}")
            
            # 创建拟合报告
            report_file = os.path.join(self.output_folder, "SEM_模型拟合报告.txt")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=== SEM模型拟合报告 ===\n\n")
                f.write(f"模型估计时间: {result['timestamp']}\n")
                f.write(f"数据标准化: {'是' if result['standardized'] else '否'}\n\n")
                
                f.write("=== 整体拟合统计 ===\n")
                f.write(f"平均R²: {fit_stats['average_r_squared']:.6f}\n")
                f.write(f"结构方程数量: {fit_stats['total_equations']}\n")
                f.write(f"外生变量数量: {fit_stats['total_exogenous_variables']}\n")
                f.write(f"总参数数量: {fit_stats['total_parameters']}\n")
                f.write(f"样本量: {fit_stats['sample_size']}\n\n")
                
                if r2_values:
                    f.write("=== R²统计 ===\n")
                    f.write(f"最小R²: {min(r2_values):.6f}\n")
                    f.write(f"最大R²: {max(r2_values):.6f}\n")
                    f.write(f"平均R²: {np.mean(r2_values):.6f}\n")
                    f.write(f"R²标准差: {np.std(r2_values):.6f}\n")
                    f.write(f"R²中位数: {np.median(r2_values):.6f}\n\n")
                
                # 拟合质量评估
                f.write("=== 拟合质量评估 ===\n")
                if r2_values:
                    avg_r2 = np.mean(r2_values)
                    if avg_r2 >= 0.7:
                        f.write("整体拟合质量: 优秀 (平均R² ≥ 0.7)\n")
                    elif avg_r2 >= 0.5:
                        f.write("整体拟合质量: 良好 (0.5 ≤ 平均R² < 0.7)\n")
                    elif avg_r2 >= 0.3:
                        f.write("整体拟合质量: 一般 (0.3 ≤ 平均R² < 0.5)\n")
                    else:
                        f.write("整体拟合质量: 较差 (平均R² < 0.3)\n")
                
                f.write(f"\n模型复杂度: {fit_stats['total_parameters']} 个参数\n")
                f.write(f"参数密度: {fit_stats['total_parameters'] / fit_stats['sample_size']:.4f} (参数/样本)\n")
            
            print(f"模型拟合报告已保存: {report_file}")
            
        except Exception as e:
            print(f"创建模型拟合报告失败: {e}")

    def run(self):
        """运行SEM参数估计"""
        print("=== 结构方程模型(SEM)参数估计器 ===")
        
        # 加载数据
        if not self.load_data():
            print("数据加载失败，程序退出")
            return False
        
        # 数据预处理
        self.preprocess_data()
        
        # 加载因果边
        self.load_causal_edges()
        
        # 运行SEM估计
        if not self.sem_estimation():
            print("SEM估计失败")
            return False
        
        # 保存结果
        self.create_output_folder()
        self.save_results()
        
        print(f"\n=== SEM参数估计完成 ===")
        print(f"结果已保存到: {self.output_folder}")
        return True

def main():
    # 可以通过参数控制是否标准化数据
    estimator = SEMParameterEstimator(standardize=True)
    estimator.run()

if __name__ == "__main__":
    main()