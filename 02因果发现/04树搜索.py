#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04 树搜索 (Tree Search)
非交互式版本，使用TAN方法，默认第一个节点作为根节点

作者: 因果发现系统
日期: 2025年
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.estimators import TreeSearch
import os
import time
import json
from datetime import datetime

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
    """创建输出文件夹"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "04树搜索结果")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_tree_results(model, output_folder, df_columns, root_node):
    """保存树搜索结果到文件"""
    edges = list(model.edges())
    
    # 保存TXT格式
    txt_file = os.path.join(output_folder, "TAN_因果边完整.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("树搜索 (TAN) 发现的因果边\n")
        f.write("=" * 40 + "\n")
        f.write(f"根节点: {root_node}\n")
        f.write("=" * 40 + "\n")
        for i, edge in enumerate(edges, 1):
            f.write(f"{i:3d}. {edge[0]} -> {edge[1]}\n")
    
    # 保存CSV格式
    df_edges = pd.DataFrame(edges, columns=["源节点", "目标节点"])
    csv_file = os.path.join(output_folder, "TAN_因果边列表.csv")
    df_edges.to_csv(csv_file, index=False, encoding="utf-8-sig")
    
    # 生成网络图
    plt.figure(figsize=(16, 12))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # 绘制节点，根节点用不同颜色
    node_colors = ['red' if node == root_node else 'lightyellow' for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors, 
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
    
    plt.title(f"树搜索 (TAN) 因果网络图\n根节点: {root_node}\n共{len(edges)}条因果边", 
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    graph_file = os.path.join(output_folder, "TAN_因果网络图.png")
    plt.savefig(graph_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    # 创建详细JSON结果
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    results = {
        "算法信息": {
            "算法名称": "树搜索 (Tree Search - TAN)",
            "估计器类型": "TAN (Tree Augmented Naive Bayes)",
            "根节点": root_node,
            "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "数据维度": {
                "样本数": len(df_columns),
                "变量数": len(df_columns)
            }
        },
        "网络结构": {
            "节点总数": len(model.nodes()),
            "边总数": len(edges),
            "节点列表": list(model.nodes()),
            "因果边列表": [{"源节点": edge[0], "目标节点": edge[1]} for edge in edges]
        },
        "统计信息": {
            "入度统计": {node: in_degrees.get(node, 0) for node in model.nodes()},
            "出度统计": {node: out_degrees.get(node, 0) for node in model.nodes()},
            "最大入度": max(in_degrees.values()) if in_degrees else 0,
            "最大出度": max(out_degrees.values()) if out_degrees else 0,
            "平均度数": sum(dict(G.degree()).values()) / len(model.nodes()) if model.nodes() else 0
        },
        "节点分析": {
            "根节点": [node for node in model.nodes() if in_degrees.get(node, 0) == 0],
            "叶节点": [node for node in model.nodes() if out_degrees.get(node, 0) == 0],
            "中介节点": [node for node in model.nodes() if in_degrees.get(node, 0) > 0 and out_degrees.get(node, 0) > 0]
        }
    }
    
    json_file = os.path.join(output_folder, "TAN_因果结果.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return txt_file, csv_file, graph_file, json_file, results

def run_tree_search_algorithm():
    """运行树搜索算法"""
    print("=" * 60)
    print("04 树搜索 (Tree Search - TAN) - 开始执行")
    print("=" * 60)
    
    # 1. 加载数据
    df = load_data()
    
    # 2. 创建输出文件夹
    output_dir = create_output_folder()
    
    # 3. 使用第一个节点作为根节点
    root_node = df.columns[0]
    print(f"使用根节点: {root_node}")
    
    # 4. 运行TAN算法
    print("正在运行树搜索 (TAN算法)...")
    start_time = time.time()
    
    try:
        ts = TreeSearch(df)
        model = ts.estimate(estimator_type='tan', class_node=root_node)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"✓ 树搜索完成，耗时: {execution_time:.2f}秒")
        print(f"✓ 发现 {len(model.edges())} 条因果边")
        
        # 5. 保存结果
        txt_file, csv_file, graph_file, json_file, results = save_tree_results(model, output_dir, df.columns, root_node)
        
        # 6. 输出结果摘要
        print("\n" + "=" * 60)
        print("树搜索执行完成 - 结果摘要")
        print("=" * 60)
        print(f"算法类型: TAN (Tree Augmented Naive Bayes)")
        print(f"根节点: {root_node}")
        print(f"执行时间: {execution_time:.2f}秒")
        print(f"数据维度: {df.shape[0]} × {df.shape[1]}")
        print(f"发现的因果边数量: {results['网络结构']['边总数']}")
        print(f"网络节点数量: {results['网络结构']['节点总数']}")
        print(f"根节点数量: {len(results['节点分析']['根节点'])}")
        print(f"叶节点数量: {len(results['节点分析']['叶节点'])}")
        print(f"中介节点数量: {len(results['节点分析']['中介节点'])}")
        print(f"平均节点度数: {results['统计信息']['平均度数']:.2f}")
        
        print(f"\n📁 结果保存位置:")
        print(f"  - TXT文件: {txt_file}")
        print(f"  - CSV文件: {csv_file}")
        print(f"  - 网络图: {graph_file}")
        print(f"  - JSON结果: {json_file}")
        
        return output_dir, len(model.edges())
        
    except Exception as e:
        print(f"❌ 树搜索执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        output_dir, edge_count = run_tree_search_algorithm()
        print(f"\n✅ 04 树搜索执行成功！发现 {edge_count} 条因果边")
    except Exception as e:
        print(f"\n❌ 04 树搜索执行失败: {str(e)}")
        raise