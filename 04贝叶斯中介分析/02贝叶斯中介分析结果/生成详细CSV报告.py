#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
贝叶斯中介分析详细报告CSV转换器
从详细分析报告.txt文件中提取所有路径信息并生成详细的CSV文件
"""

import re
import pandas as pd
import os
from datetime import datetime

def parse_detailed_report(file_path):
    """
    解析详细分析报告文件，提取所有路径的详细信息
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取基本信息
    analysis_time_match = re.search(r'分析时间: (.+)', content)
    total_paths_match = re.search(r'分析路径总数: (\d+)', content)
    significant_paths_match = re.search(r'显著中介路径数量: (\d+)/(\d+)', content)
    significant_ratio_match = re.search(r'显著路径比例: (.+)%', content)
    
    analysis_time = analysis_time_match.group(1) if analysis_time_match else ""
    total_paths = int(total_paths_match.group(1)) if total_paths_match else 0
    significant_paths = int(significant_paths_match.group(1)) if significant_paths_match else 0
    significant_ratio = float(significant_ratio_match.group(1)) if significant_ratio_match else 0.0
    
    # 分割每个路径的详细分析
    path_sections = re.split(r'============================================================\n路径 \d+ 详细分析\n============================================================', content)[1:]
    
    paths_data = []
    
    for i, section in enumerate(path_sections, 1):
        path_data = {'路径ID': i}
        
        # 提取路径描述
        path_desc_match = re.search(r'路径描述: (.+)', section)
        if path_desc_match:
            path_data['路径描述'] = path_desc_match.group(1).strip()
        
        # 提取间接效应信息
        indirect_mean_match = re.search(r'间接效应（中介效应）:\s*均值: ([-\d.]+)', section)
        indirect_hdi_match = re.search(r'间接效应（中介效应）:.*?95% HDI: \[([-\d.]+), ([-\d.]+)\]', section, re.DOTALL)
        indirect_std_match = re.search(r'间接效应（中介效应）:.*?标准差: ([-\d.]+)', section, re.DOTALL)
        
        if indirect_mean_match:
            path_data['间接效应均值'] = float(indirect_mean_match.group(1))
        if indirect_hdi_match:
            path_data['间接效应HDI下限'] = float(indirect_hdi_match.group(1))
            path_data['间接效应HDI上限'] = float(indirect_hdi_match.group(2))
        if indirect_std_match:
            path_data['间接效应标准差'] = float(indirect_std_match.group(1))
        
        # 提取直接效应信息
        direct_mean_match = re.search(r'直接效应:\s*均值: ([-\d.]+)', section)
        direct_hdi_match = re.search(r'直接效应:.*?95% HDI: \[([-\d.]+), ([-\d.]+)\]', section, re.DOTALL)
        direct_std_match = re.search(r'直接效应:.*?标准差: ([-\d.]+)', section, re.DOTALL)
        
        if direct_mean_match:
            path_data['直接效应均值'] = float(direct_mean_match.group(1))
        if direct_hdi_match:
            path_data['直接效应HDI下限'] = float(direct_hdi_match.group(1))
            path_data['直接效应HDI上限'] = float(direct_hdi_match.group(2))
        if direct_std_match:
            path_data['直接效应标准差'] = float(direct_std_match.group(1))
        
        # 提取总效应信息
        total_mean_match = re.search(r'总效应:\s*均值: ([-\d.]+)', section)
        total_hdi_match = re.search(r'总效应:.*?95% HDI: \[([-\d.]+), ([-\d.]+)\]', section, re.DOTALL)
        total_std_match = re.search(r'总效应:.*?标准差: ([-\d.]+)', section, re.DOTALL)
        
        if total_mean_match:
            path_data['总效应均值'] = float(total_mean_match.group(1))
        if total_hdi_match:
            path_data['总效应HDI下限'] = float(total_hdi_match.group(1))
            path_data['总效应HDI上限'] = float(total_hdi_match.group(2))
        if total_std_match:
            path_data['总效应标准差'] = float(total_std_match.group(1))
        
        # 提取中介比例
        mediation_ratio_match = re.search(r'中介比例: ([-\d.]+) \(([-\d.]+)%\)', section)
        if mediation_ratio_match:
            path_data['中介比例'] = float(mediation_ratio_match.group(1))
            path_data['中介比例百分比'] = float(mediation_ratio_match.group(2))
        
        # 提取影响类型分析
        main_influence_match = re.search(r'主要影响类型: (.+)', section)
        influence_strength_match = re.search(r'影响强度: (.+)', section)
        indirect_direction_match = re.search(r'间接效应方向: (.+)', section)
        direct_direction_match = re.search(r'直接效应方向: (.+)', section)
        mediation_type_match = re.search(r'中介类型: (.+)', section)
        
        if main_influence_match:
            path_data['主要影响类型'] = main_influence_match.group(1).strip()
        if influence_strength_match:
            path_data['影响强度'] = influence_strength_match.group(1).strip()
        if indirect_direction_match:
            path_data['间接效应方向'] = indirect_direction_match.group(1).strip()
        if direct_direction_match:
            path_data['直接效应方向'] = direct_direction_match.group(1).strip()
        if mediation_type_match:
            path_data['中介类型'] = mediation_type_match.group(1).strip()
        
        # 提取显著性分析
        significance_prob_match = re.search(r'显著性概率: ([\d.]+)', section)
        is_significant_match = re.search(r'是否显著: (.+)', section)
        positive_prob_match = re.search(r'正向效应概率: ([\d.]+)', section)
        negative_prob_match = re.search(r'负向效应概率: ([\d.]+)', section)
        
        if significance_prob_match:
            path_data['显著性概率'] = float(significance_prob_match.group(1))
        if is_significant_match:
            path_data['是否显著'] = is_significant_match.group(1).strip()
        if positive_prob_match:
            path_data['正向效应概率'] = float(positive_prob_match.group(1))
        if negative_prob_match:
            path_data['负向效应概率'] = float(negative_prob_match.group(1))
        
        # 提取结论
        conclusion_match = re.search(r'结论:\s*-+\s*(.+?)(?=============================================================|$)', section, re.DOTALL)
        if conclusion_match:
            conclusion_text = conclusion_match.group(1).strip()
            # 清理结论文本，移除多余的换行和空格
            conclusion_text = re.sub(r'\n+', ' ', conclusion_text)
            conclusion_text = re.sub(r'\s+', ' ', conclusion_text)
            path_data['结论'] = conclusion_text
        
        paths_data.append(path_data)
    
    return paths_data, {
        '分析时间': analysis_time,
        '总路径数': total_paths,
        '显著路径数': significant_paths,
        '显著路径比例': significant_ratio
    }

def create_detailed_csv(input_file, output_file):
    """
    创建详细的CSV文件
    """
    print(f"正在解析文件: {input_file}")
    
    # 解析报告
    paths_data, summary_info = parse_detailed_report(input_file)
    
    print(f"成功解析 {len(paths_data)} 个路径")
    
    # 创建DataFrame
    df = pd.DataFrame(paths_data)
    
    # 定义列的顺序
    column_order = [
        '路径ID', '路径描述',
        '间接效应均值', '间接效应HDI下限', '间接效应HDI上限', '间接效应标准差',
        '直接效应均值', '直接效应HDI下限', '直接效应HDI上限', '直接效应标准差',
        '总效应均值', '总效应HDI下限', '总效应HDI上限', '总效应标准差',
        '中介比例', '中介比例百分比',
        '主要影响类型', '影响强度', '间接效应方向', '直接效应方向', '中介类型',
        '显著性概率', '是否显著', '正向效应概率', '负向效应概率',
        '结论'
    ]
    
    # 重新排列列的顺序
    df = df.reindex(columns=column_order)
    
    # 保存CSV文件
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"详细CSV文件已保存至: {output_file}")
    print(f"文件包含 {len(df)} 行数据，{len(df.columns)} 列")
    
    # 打印摘要信息
    print("\n=== 分析摘要 ===")
    print(f"分析时间: {summary_info['分析时间']}")
    print(f"总路径数: {summary_info['总路径数']}")
    print(f"显著路径数: {summary_info['显著路径数']}")
    print(f"显著路径比例: {summary_info['显著路径比例']}%")
    
    # 统计中介类型分布
    if '中介类型' in df.columns:
        mediation_type_counts = df['中介类型'].value_counts()
        print(f"\n=== 中介类型分布 ===")
        for mediation_type, count in mediation_type_counts.items():
            print(f"{mediation_type}: {count} 个路径")
    
    # 统计显著性分布
    if '是否显著' in df.columns:
        significance_counts = df['是否显著'].value_counts()
        print(f"\n=== 显著性分布 ===")
        for significance, count in significance_counts.items():
            print(f"{significance}: {count} 个路径")
    
    return df, summary_info

def main():
    """
    主函数
    """
    # 文件路径
    input_file = "/home/zkr/因果发现3/04贝叶斯中介分析/02贝叶斯中介分析结果/详细分析报告.txt"
    output_file = "/home/zkr/因果发现3/04贝叶斯中介分析/02贝叶斯中介分析结果/贝叶斯中介分析详细结果.csv"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在 - {input_file}")
        return
    
    try:
        # 创建详细CSV文件
        df, summary_info = create_detailed_csv(input_file, output_file)
        
        print(f"\n=== 处理完成 ===")
        print(f"详细CSV文件已成功生成: {output_file}")
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()