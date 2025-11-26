#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取完整中介路径脚本
完整中介路径定义：
1. 存在间接路径：X → Z → Y
这样Z才是真正的中介变量
"""

import os

def create_output_directory(dir_name):
    """
    创建输出目录
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"创建输出目录: {dir_name}")
    else:
        print(f"输出目录已存在: {dir_name}")
    return dir_name

def extract_complete_mediation_paths(input_file, output_dir):
    """
    从最终因果边文件中提取完整的中介路径
    
    Args:
        input_file: 输入文件路径（最终因果边完整列表.csv）
        output_dir: 输出目录路径
    """
    import pandas as pd
    
    # 读取所有因果边
    edges = []
    try:
        # 读取CSV文件
        df = pd.read_csv(input_file, encoding='utf-8')
        
        # 提取源节点和目标节点列
        for _, row in df.iterrows():
            source = row['源节点']
            target = row['目标节点']
            edges.append((source, target))
            
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    print(f"总共读取了 {len(edges)} 条因果边")
    
    # 构建邻接表表示的有向图
    graph = {}
    edge_set = set()  # 用于快速查找直接边是否存在
    
    for source, target in edges:
        if source not in graph:
            graph[source] = []
        graph[source].append(target)
        edge_set.add((source, target))
    
    # 查找完整的中介路径
    complete_mediation_paths = []
    
    for x in graph:
        # X的所有直接后继节点作为潜在的中介变量Z
        for z in graph[x]:
            # 检查Z是否有后继节点Y
            if z in graph:
                for y in graph[z]:
                    # 确保不是自环或重复
                    if x != y and x != z and z != y:
                        # 修改：只需要间接路径即可，不再检查直接路径
                        complete_path = {
                            'start': x,
                            'mediator': z,
                            'end': y,
                            'indirect_path': f"{x} → {z} → {y}"
                        }
                        complete_mediation_paths.append(complete_path)
    
    print(f"找到 {len(complete_mediation_paths)} 条完整中介路径")
    
    # 保存完整中介路径到文件
    output_file = os.path.join(output_dir, "完整中介路径结果.txt")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 完整中介路径分析\n")
            f.write("# 定义：存在间接路径 X → Z → Y\n")
            f.write(f"# 总共找到 {len(complete_mediation_paths)} 条完整中介路径\n\n")
            
            for i, path in enumerate(complete_mediation_paths, 1):
                f.write(f"{i:4d}. 中介路径:\n")
                f.write(f"      间接路径: {path['indirect_path']}\n")
                f.write(f"      中介变量: {path['mediator']}\n")
                f.write("\n")
        
        print(f"完整中介路径已保存到: {output_file}")
        
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return
    
    # 保存完整中介路径到CSV文件
    csv_output_file = os.path.join(output_dir, "完整中介路径结果.csv")
    try:
        import pandas as pd
        
        # 准备CSV数据
        csv_data = []
        for i, path in enumerate(complete_mediation_paths, 1):
            csv_data.append({
                '路径ID': i,
                '起始节点': path['start'],
                '中介变量': path['mediator'],
                '终点节点': path['end'],
                '完整路径': path['indirect_path']
            })
        
        # 创建DataFrame并保存为CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_output_file, index=False, encoding='utf-8')
        
        print(f"完整中介路径CSV文件已保存到: {csv_output_file}")
        
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")
    
    # 统计信息
    print("\n=== 统计信息 ===")
    
    # 统计作为中介变量的节点
    mediators = set()
    start_nodes = set()
    end_nodes = set()
    
    for path in complete_mediation_paths:
        start_nodes.add(path['start'])
        mediators.add(path['mediator'])
        end_nodes.add(path['end'])
    
    print(f"起始节点数量: {len(start_nodes)}")
    print(f"中介变量数量: {len(mediators)}")
    print(f"终点节点数量: {len(end_nodes)}")
    
    # 显示前5个完整中介路径作为示例
    print("\n=== 前5个完整中介路径示例 ===")
    for i, path in enumerate(complete_mediation_paths[:5], 1):
        print(f"{i}. 间接: {path['indirect_path']}")
        print(f"   中介: {path['mediator']}")
        print()
    
    if len(complete_mediation_paths) > 5:
        print(f"... 还有 {len(complete_mediation_paths) - 5} 条完整中介路径")
    
    return complete_mediation_paths

def analyze_complete_mediation_patterns(complete_paths, output_dir):
    """
    分析完整中介路径的模式
    """
    print("\n=== 完整中介路径模式分析 ===")
    
    # 统计不同类型的中介路径
    disease_mediation = []  # 疾病作为中介
    drug_mediation = []     # 药物作为中介
    test_mediation = []     # 检验作为中介
    mixed_mediation = []    # 混合类型
    
    for path in complete_paths:
        x = path['start']
        z = path['mediator']
        y = path['end']
        
        x_type = x.split('_')[0] if '_' in x else 'unknown'
        z_type = z.split('_')[0] if '_' in z else 'unknown'
        y_type = y.split('_')[0] if '_' in y else 'unknown'
        
        if z_type == '疾病':
            disease_mediation.append(path)
        elif z_type == '药物':
            drug_mediation.append(path)
        elif z_type == '检验':
            test_mediation.append(path)
        else:
            mixed_mediation.append(path)
    
    print(f"疾病作为中介变量: {len(disease_mediation)} 条")
    print(f"药物作为中介变量: {len(drug_mediation)} 条")
    print(f"检验作为中介变量: {len(test_mediation)} 条")
    print(f"其他类型中介: {len(mixed_mediation)} 条")
    
    patterns = {
        '疾病中介路径': disease_mediation,
        '药物中介路径': drug_mediation,
        '检验中介路径': test_mediation,
        '其他中介路径': mixed_mediation
    }
    
    return patterns

def save_mediation_by_type(patterns, output_dir):
    """
    按中介变量类型保存到不同文件
    """
    for mediation_type, paths in patterns.items():
        if not paths:
            continue
            
        filename = os.path.join(output_dir, f"{mediation_type}.txt")
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# {mediation_type}\n")
                f.write(f"# 总共 {len(paths)} 条路径\n\n")
                
                for i, path in enumerate(paths, 1):
                    f.write(f"{i:3d}. 间接路径: {path['indirect_path']}\n")
                    f.write(f"     直接路径: {path['direct_path']}\n")
                    f.write(f"     中介变量: {path['mediator']}\n")
                    f.write("\n")
            
            print(f"{mediation_type}已保存到: {filename}")
            
        except Exception as e:
            print(f"保存 {mediation_type} 文件时出错: {e}")

if __name__ == "__main__":
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建输入文件的绝对路径 - 修改为精简因果边列表.csv
    input_file = os.path.join(os.path.dirname(script_dir), "02因果发现/06候选因果边集合/精简因果边列表.csv")
    output_dir = os.path.join(script_dir, "01中介路径分析结果")
    
    print("开始提取完整中介路径...")
    print("完整中介路径定义：存在 X → Z → Y")
    print()
    
    # 创建输出目录
    create_output_directory(output_dir)
    
    # 提取完整中介路径
    complete_paths = extract_complete_mediation_paths(input_file, output_dir)
    
    if complete_paths:
        # 分析中介路径模式
        patterns = analyze_complete_mediation_patterns(complete_paths, output_dir)
        
        # 注释掉按类型保存到不同文件的功能
        # save_mediation_by_type(patterns, output_dir)
        
        # 生成汇总报告
        summary_file = os.path.join(output_dir, "中介路径汇总报告.txt")
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("# 完整中介路径汇总报告\n\n")
                f.write(f"## 总体统计\n")
                f.write(f"- 总路径数: {len(complete_paths)}\n")
                f.write(f"- 疾病中介: {len(patterns['疾病中介路径'])}\n")
                f.write(f"- 药物中介: {len(patterns['药物中介路径'])}\n")
                f.write(f"- 检验中介: {len(patterns['检验中介路径'])}\n")
                f.write(f"- 其他中介: {len(patterns['其他中介路径'])}\n\n")
                
                f.write("## 中介效应强度排序（按出现频次）\n")
                
                # 统计每个中介变量的出现频次
                mediator_count = {}
                for path in complete_paths:
                    mediator = path['mediator']
                    mediator_count[mediator] = mediator_count.get(mediator, 0) + 1
                
                # 按频次排序
                sorted_mediators = sorted(mediator_count.items(), key=lambda x: x[1], reverse=True)
                
                for i, (mediator, count) in enumerate(sorted_mediators[:20], 1):
                    f.write(f"{i:2d}. {mediator}: {count} 次\n")
                
                if len(sorted_mediators) > 20:
                    f.write(f"... 还有 {len(sorted_mediators) - 20} 个中介变量\n")
                
                # 添加详细的路径类型分析
                f.write("\n## 路径类型详细分析\n\n")
                
                for pattern_name, pattern_paths in patterns.items():
                    if pattern_paths:
                        f.write(f"### {pattern_name} ({len(pattern_paths)} 条)\n")
                        
                        # 统计该类型中最常见的中介变量
                        type_mediator_count = {}
                        for path in pattern_paths:
                            mediator = path['mediator']
                            type_mediator_count[mediator] = type_mediator_count.get(mediator, 0) + 1
                        
                        sorted_type_mediators = sorted(type_mediator_count.items(), key=lambda x: x[1], reverse=True)
                        
                        f.write("最常见的中介变量：\n")
                        for i, (mediator, count) in enumerate(sorted_type_mediators[:5], 1):
                            f.write(f"  {i}. {mediator}: {count} 次\n")
                        f.write("\n")
            
            print(f"\n汇总报告已保存到: {summary_file}")
            
        except Exception as e:
            print(f"保存汇总报告时出错: {e}")
    
    print("\n处理完成！")
    print(f"\n所有结果已保存到文件夹: {output_dir}")
    print("\n生成的文件：")
    print("1. 完整中介路径结果.txt - 所有完整中介路径")
    print("2. 完整中介路径结果.csv - CSV格式的中介路径数据")
    print("3. 中介路径汇总报告.txt - 详细统计分析")
    # 注释掉不再生成的文件说明
    # print("2. 疾病中介路径.txt - 疾病作为中介的路径")
    # print("3. 药物中介路径.txt - 药物作为中介的路径")
    # print("4. 检验中介路径.txt - 检验作为中介的路径")
    # print("5. 其他中介路径.txt - 其他类型中介路径")
    print("6. 中介路径汇总报告.txt - 详细统计分析")