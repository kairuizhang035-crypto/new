#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01 PC算法 (基于约束的估计器)
非交互式版本，用于统一执行流程

作者: 因果发现系统
日期: 2025年
"""

import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
import numpy as np
from pgmpy.estimators import PC

# 设置中文字体
import matplotlib
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = [
    'SimHei', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 
    'Noto Sans CJK SC', 'Source Han Sans SC', 'Microsoft YaHei',
    'DejaVu Sans', 'Arial Unicode MS', 'Liberation Sans'
]
matplotlib.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载数据"""
    input_file = "/home/zkr/因果发现3/01数据预处理/缩减数据_规格.csv"
    
    # 尝试使用utf-8编码
    try:
        df = pd.read_csv(input_file, encoding='utf-8', header=0, index_col=0)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(input_file, encoding='utf-8-sig', header=0, index_col=0)
        except UnicodeDecodeError:
            df = pd.read_csv(input_file, encoding='latin-1', header=0, index_col=0)
    
    df = df.dropna(axis=1, how='all')
    df = df.astype('float32')
    
    print(f"✓ 数据加载完成: {df.shape}")
    return df

def create_output_folder():
    base_root = "/home/zkr/因果发现3/02因果发现"
    output_dir = os.path.join(base_root, "01PC算法结果")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def create_causal_network_graph(edges, output_dir):
    """创建因果网络图并保存为PNG"""
    G = nx.DiGraph()
    
    # 添加边
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    
    plt.figure(figsize=(16, 12))
    
    # 使用spring布局
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, 
                          node_color='lightblue', 
                          node_size=2000,
                          alpha=0.8)
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, 
                          edge_color='gray',
                          arrows=True,
                          arrowsize=20,
                          arrowstyle='->',
                          width=1.5,
                          alpha=0.7)
    
    # 绘制标签
    nx.draw_networkx_labels(G, pos, 
                           font_size=10,
                           font_weight='bold',
                           font_family='sans-serif')
    
    plt.title(f"PC算法因果网络图\n共{len(edges)}条因果边", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    graph_file = os.path.join(output_dir, "PC_因果网络图.png")
    plt.savefig(graph_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    return graph_file

def create_detailed_json_results(estimated_model, df, output_dir):
    """创建详细的因果发现结果JSON文件"""
    nodes = list(estimated_model.nodes())
    edges = list(estimated_model.edges())
    
    # 计算网络统计信息
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    # 节点度数统计
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    # 创建详细结果字典
    results = {
        "算法信息": {
            "算法名称": "PC算法",
            "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "数据维度": {
                "样本数": int(df.shape[0]),
                "变量数": int(df.shape[1])
            },
            "参数设置": {
                "独立性检验": "chi_square",
                "显著性水平": 0.05,
                "变体": "stable"
            }
        },
        "网络结构": {
            "节点总数": len(nodes),
            "边总数": len(edges),
            "节点列表": nodes,
            "因果边列表": [{"源节点": edge[0], "目标节点": edge[1]} for edge in edges]
        },
        "统计信息": {
            "入度统计": {node: in_degrees.get(node, 0) for node in nodes},
            "出度统计": {node: out_degrees.get(node, 0) for node in nodes},
            "最大入度": max(in_degrees.values()) if in_degrees else 0,
            "最大出度": max(out_degrees.values()) if out_degrees else 0,
            "平均度数": sum(dict(G.degree()).values()) / len(nodes) if nodes else 0
        },
        "节点分析": {
            "根节点": [node for node in nodes if in_degrees.get(node, 0) == 0],
            "叶节点": [node for node in nodes if out_degrees.get(node, 0) == 0],
            "中介节点": [node for node in nodes if in_degrees.get(node, 0) > 0 and out_degrees.get(node, 0) > 0]
        }
    }
    
    # 保存JSON文件
    json_file = os.path.join(output_dir, "PC_因果结果.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return json_file, results

def run_pc_algorithm():
    """运行PC算法"""
    print("=" * 60)
    print("01 PC算法 (基于约束的估计器) - 开始执行")
    print("=" * 60)
    
    # 1. 加载数据
    df = load_data()
    
    # 2. 创建输出文件夹
    output_dir = create_output_folder()
    
    # 3. 初始化PC算法估计器
    print("正在运行PC算法...")
    est = PC(data=df)
    
    # 4. 运行估计算法
    estimated_model = est.estimate(variant="stable", ci_test="chi_square", significance_level=0.05)
    
    # 5. 获取结果
    edges_list = list(estimated_model.edges())
    print(f"✓ PC算法完成，发现 {len(edges_list)} 条因果边")
    
    # 6. 保存结果文件
    # 保存TXT格式
    output_file_txt = os.path.join(output_dir, "PC_因果边完整.txt")
    with open(output_file_txt, 'w', encoding='utf-8') as f:
        f.write("PC算法发现的因果边\n")
        f.write("=" * 30 + "\n")
        for i, edge in enumerate(edges_list, 1):
            f.write(f"{i:3d}. {edge[0]} -> {edge[1]}\n")
    
    # 保存CSV格式
    df_edges = pd.DataFrame(edges_list, columns=["源节点", "目标节点"])
    output_file_csv = os.path.join(output_dir, "PC_因果边列表.csv")
    df_edges.to_csv(output_file_csv, index=False, encoding="utf-8-sig")
    
    # 7. 生成网络图
    print("正在生成因果网络图...")
    graph_file = create_causal_network_graph(edges_list, output_dir)
    
    # 8. 生成JSON结果
    print("正在生成详细JSON结果...")
    json_file, results = create_detailed_json_results(estimated_model, df, output_dir)
    
    # 9. 输出结果摘要
    print("\n" + "=" * 60)
    print("PC算法执行完成 - 结果摘要")
    print("=" * 60)
    print(f"数据维度: {results['算法信息']['数据维度']['样本数']} × {results['算法信息']['数据维度']['变量数']}")
    print(f"发现的因果边数量: {results['网络结构']['边总数']}")
    print(f"网络节点数量: {results['网络结构']['节点总数']}")
    print(f"根节点数量: {len(results['节点分析']['根节点'])}")
    print(f"叶节点数量: {len(results['节点分析']['叶节点'])}")
    print(f"中介节点数量: {len(results['节点分析']['中介节点'])}")
    print(f"平均节点度数: {results['统计信息']['平均度数']:.2f}")
    
    print(f"\n📁 结果保存位置:")
    print(f"  - TXT文件: {output_file_txt}")
    print(f"  - CSV文件: {output_file_csv}")
    print(f"  - 网络图: {graph_file}")
    print(f"  - JSON结果: {json_file}")
    
    return output_dir, len(edges_list)

if __name__ == "__main__":
    try:
        output_dir, edge_count = run_pc_algorithm()
        print(f"\n✅ 01 PC算法执行成功！发现 {edge_count} 条因果边")
    except Exception as e:
        print(f"\n❌ 01 PC算法执行失败: {str(e)}")
        raise
