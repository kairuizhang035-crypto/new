import json
import pandas as pd
import numpy as np
import os
from datetime import datetime
from itertools import product
import re

class SEMRegressionPredictor:
    def __init__(self, sem_json_path, data_path):
        """
        初始化SEM回归预测器
        
        Args:
            sem_json_path: SEM结构方程JSON文件路径
            data_path: 原始数据CSV文件路径
        """
        self.sem_json_path = sem_json_path
        self.data_path = data_path
        self.sem_equations = None
        self.data = None
        self.results = {}
        
    def load_data(self):
        """加载SEM结构方程和原始数据"""
        # 加载SEM结构方程
        with open(self.sem_json_path, 'r', encoding='utf-8') as f:
            self.sem_equations = json.load(f)
        print(f"成功加载SEM结构方程，包含 {len(self.sem_equations)} 个节点")
        
        # 加载原始数据
        self.data = pd.read_csv(self.data_path)
        print(f"成功加载数据，包含 {len(self.data)} 行，{len(self.data.columns)} 列")
        
    def sigmoid(self, x):
        """Sigmoid函数，将预测值转换为概率"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def predict_node_probability(self, target_node, parent_values):
        """
        使用SEM线性方程预测目标节点的概率
        
        Args:
            target_node: 目标节点名称
            parent_values: 父节点值的字典 {parent_name: value}
        
        Returns:
            tuple: (predicted_value, probability_0, probability_1)
        """
        if target_node not in self.sem_equations:
            return None, None, None
            
        equation = self.sem_equations[target_node]
        
        # 计算线性预测值
        predicted_value = equation['intercept']
        
        # coefficients是列表，parents也是列表，需要按索引对应
        parents = equation['parents']
        coefficients = equation['coefficients']
        
        for i, parent in enumerate(parents):
            if parent in parent_values and i < len(coefficients):
                predicted_value += coefficients[i] * parent_values[parent]
        
        # 使用sigmoid函数转换为概率
        prob_1 = self.sigmoid(predicted_value)
        prob_0 = 1 - prob_1
        
        return predicted_value, prob_0, prob_1
    
    def generate_conditional_probability_tables(self):
        """生成类似MLE的条件概率表"""
        cpt_results = {}
        summary_data = []
        detailed_results = []
        
        print("开始生成条件概率表...")
        
        for target_node, equation in self.sem_equations.items():
            if not equation['parents']:
                continue  # 跳过没有父节点的节点
                
            parents = equation['parents']
            print(f"处理节点: {target_node}, 父节点: {parents}")
            
            # 生成所有可能的父节点组合 (0, 1)
            parent_combinations = list(product([0, 1], repeat=len(parents)))
            
            node_probabilities = {}
            
            for combination in parent_combinations:
                # 创建父节点值字典
                parent_values = {parent: value for parent, value in zip(parents, combination)}
                
                # 预测概率
                predicted_value, prob_0, prob_1 = self.predict_node_probability(target_node, parent_values)
                
                if predicted_value is not None:
                    # 创建条件键 (如 "0,1,0")
                    condition_key = ",".join(map(str, combination))
                    node_probabilities[condition_key] = [prob_0, prob_1]
                    
                    # 添加到汇总数据
                    summary_data.append({
                        '节点': target_node,
                        '类型': '条件概率',
                        '父节点': ', '.join(parents),
                        'P(0)': f"{prob_0:.4f}",
                        'P(1)': f"{prob_1:.4f}",
                        '条件': condition_key
                    })
                    
                    # 添加到详细结果
                    parent_condition = ', '.join([f"{parent}={value}" for parent, value in zip(parents, combination)])
                    detailed_results.append({
                        'node': target_node,
                        'parents': parents,
                        'condition': parent_condition,
                        'prob_0': prob_0,
                        'prob_1': prob_1,
                        'predicted_value': predicted_value
                    })
            
            # 保存节点的条件概率表
            if node_probabilities:
                cpt_results[target_node] = {
                    "type": "conditional",
                    "parents": parents,
                    "probabilities": node_probabilities
                }
        
        self.results = {
            'cpt_json': cpt_results,
            'summary_csv': summary_data,
            'detailed_results': detailed_results
        }
        
        print(f"成功生成 {len(cpt_results)} 个节点的条件概率表")
        return self.results
    
    def save_results(self, output_dir):
        """保存结果到指定目录"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 保存JSON格式 (类似MLE_CPTs.json)
        json_path = os.path.join(output_dir, 'SEM_CPTs.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results['cpt_json'], f, ensure_ascii=False, indent=2)
        print(f"已保存JSON结果到: {json_path}")
        
        # 2. 保存CSV汇总 (类似MLE_条件概率表汇总.csv)
        csv_summary_path = os.path.join(output_dir, 'SEM_条件概率表汇总.csv')
        summary_df = pd.DataFrame(self.results['summary_csv'])
        summary_df.to_csv(csv_summary_path, index=False, encoding='utf-8')
        print(f"已保存CSV汇总到: {csv_summary_path}")
        
        # 3. 保存详细CSV数据 (类似MLE_条件概率表详细数据.csv)
        csv_detailed_path = os.path.join(output_dir, 'SEM_条件概率表详细数据.csv')
        detailed_df = pd.DataFrame(self.results['detailed_results'])
        detailed_df.to_csv(csv_detailed_path, index=False, encoding='utf-8')
        print(f"已保存详细CSV数据到: {csv_detailed_path}")
        
        # 4. 保存详细TXT结果 (类似MLE_条件概率表详细结果.txt)
        txt_path = os.path.join(output_dir, 'SEM_条件概率表详细结果.txt')
        self._save_detailed_txt(txt_path)
        print(f"已保存详细TXT结果到: {txt_path}")
        
        # 5. 生成分析报告
        report_path = os.path.join(output_dir, 'SEM_条件概率表分析报告.txt')
        self._generate_analysis_report(report_path)
        print(f"已保存分析报告到: {report_path}")
    
    def _save_detailed_txt(self, file_path):
        """保存详细的TXT格式结果"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("=== SEM回归预测详细结果 ===\n\n")
            f.write("估计方法: Structural Equation Modeling (SEM)\n")
            f.write(f"时间戳: {datetime.now().isoformat()}\n")
            f.write(f"条件概率表数量: {len(self.results['cpt_json'])}\n\n")
            
            for node_name, cpt_data in self.results['cpt_json'].items():
                f.write(f"节点: {node_name}\n")
                f.write(f"类型: {cpt_data['type']}\n")
                f.write(f"父节点: {', '.join(cpt_data['parents'])}\n")
                
                # 写入每个条件的概率
                for condition, probs in cpt_data['probabilities'].items():
                    parent_values = condition.split(',')
                    parent_condition = ', '.join([f"{parent}={value}" for parent, value in zip(cpt_data['parents'], parent_values)])
                    
                    f.write(f"P({node_name}=0|{parent_condition}) = {probs[0]:.6f}\n")
                    f.write(f"P({node_name}=1|{parent_condition}) = {probs[1]:.6f}\n")
                
                f.write("\n" + "-"*50 + "\n\n")
    
    def _generate_analysis_report(self, file_path):
        """生成分析报告"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("=== SEM条件概率表分析报告 ===\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"处理的节点数量: {len(self.results['cpt_json'])}\n")
            f.write(f"生成的条件概率记录数: {len(self.results['detailed_results'])}\n\n")
            
            # 统计信息
            all_probs_0 = [item['prob_0'] for item in self.results['detailed_results']]
            all_probs_1 = [item['prob_1'] for item in self.results['detailed_results']]
            all_predicted = [item['predicted_value'] for item in self.results['detailed_results']]
            
            f.write("=== 统计信息 ===\n")
            f.write(f"P(0)概率范围: {min(all_probs_0):.4f} - {max(all_probs_0):.4f}\n")
            f.write(f"P(1)概率范围: {min(all_probs_1):.4f} - {max(all_probs_1):.4f}\n")
            f.write(f"预测值范围: {min(all_predicted):.4f} - {max(all_predicted):.4f}\n")
            f.write(f"平均P(0): {np.mean(all_probs_0):.4f}\n")
            f.write(f"平均P(1): {np.mean(all_probs_1):.4f}\n\n")
            
            # 节点详细信息
            f.write("=== 节点详细信息 ===\n")
            for node_name, cpt_data in self.results['cpt_json'].items():
                node_results = [item for item in self.results['detailed_results'] if item['node'] == node_name]
                node_probs_1 = [item['prob_1'] for item in node_results]
                
                f.write(f"\n节点: {node_name}\n")
                f.write(f"  父节点: {', '.join(cpt_data['parents'])}\n")
                f.write(f"  条件组合数: {len(cpt_data['probabilities'])}\n")
                f.write(f"  P(1)范围: {min(node_probs_1):.4f} - {max(node_probs_1):.4f}\n")
                f.write(f"  平均P(1): {np.mean(node_probs_1):.4f}\n")

def main():
    # 文件路径
    sem_json_path = "/home/zkr/因果发现/03多方法参数学习/04SEM_结果/SEM_结构方程.json"
    data_path = "/home/zkr/因果发现/01数据预处理/缩减数据_规格.csv"
    output_dir = "/home/zkr/因果发现/03多方法参数学习/04SEM_结果/SEM_CPT结果"
    
    # 创建预测器
    predictor = SEMRegressionPredictor(sem_json_path, data_path)
    
    try:
        # 加载数据
        predictor.load_data()
        
        # 生成条件概率表
        results = predictor.generate_conditional_probability_tables()
        
        # 保存结果
        predictor.save_results(output_dir)
        
        print("\n=== 处理完成 ===")
        print(f"结果已保存到: {output_dir}")
        print(f"生成的文件:")
        print("  - SEM_CPTs.json (JSON格式条件概率表)")
        print("  - SEM_条件概率表汇总.csv (CSV汇总)")
        print("  - SEM_条件概率表详细数据.csv (详细CSV数据)")
        print("  - SEM_条件概率表详细结果.txt (详细TXT结果)")
        print("  - SEM_条件概率表分析报告.txt (分析报告)")
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()