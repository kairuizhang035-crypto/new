#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03 贪婪等价搜索 (Greedy Equivalence Search - GES)
非交互式版本，使用AIC-D评分标准

作者: 因果发现系统
日期: 2025年

环境依赖:
- 可选GPU加速: 需要安装`torch>=2.1`，NVIDIA驱动与CUDA(建议11.8或12.x)
- 无GPU或显存不足时将自动回退到CPU并继续运行
- 可通过环境变量`GES_FORCE_CPU=1`强制使用CPU
"""

import pandas as pd
import numpy as np
from pgmpy.estimators import GES, StructureScore
import matplotlib.pyplot as plt
import networkx as nx
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

def _detect_compute_mode():
    try:
        import torch
        import os as _os
        if _os.environ.get("GES_FORCE_CPU", "0") == "1":
            return "cpu"
        if torch.cuda.is_available():
            return "gpu"
    except Exception:
        pass
    return "cpu"

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

class AICDScoreGPU(StructureScore):
    def __init__(self, data, mode="cpu", **kwargs):
        super(AICDScoreGPU, self).__init__(data=data, **kwargs)
        self.mode = mode
        self.cols = list(self.data.columns)
        self.codes = []
        self.card = []
        for c in self.cols:
            cat = pd.Categorical(self.data[c])
            code = cat.codes.astype(np.int64)
            self.codes.append(code)
            self.card.append(int((code.max() + 1) if code.size else 0))
        self.codes = np.vstack(self.codes).T if len(self.codes) > 0 else np.empty((0, 0), dtype=np.int64)
        self.card = np.array(self.card, dtype=np.int64)
        self.cache = {}
        self.device = None
        self.use_gpu = False
        if self.mode == "gpu":
            try:
                import torch
                if torch.cuda.is_available() and self.codes.size > 0:
                    self.device = torch.device("cuda")
                    try:
                        self.tcodes = torch.tensor(self.codes, dtype=torch.int64, device=self.device)
                        self.tcard = torch.tensor(self.card, dtype=torch.int64, device=self.device)
                        self.use_gpu = True
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                            self.use_gpu = False
                else:
                    self.use_gpu = False
            except Exception:
                self.use_gpu = False

    def _local_score_cpu(self, vi, pidx):
        r = int(self.card[vi])
        if r == 0:
            return 0.0
        if len(pidx) == 0:
            x = self.codes[:, vi]
            counts = np.bincount(x, minlength=r).astype(np.float64)
            total = counts.sum()
            if total == 0:
                return 0.0
            mask = counts > 0
            ll = float((counts[mask] * (np.log(counts[mask]) - np.log(total))).sum())
            num_parents_states = 1
            return ll - num_parents_states * (r - 1)
        strides = []
        q = 1
        for idx in pidx:
            strides.append(q)
            q *= int(self.card[idx])
        if q == 0:
            return 0.0
        gp = np.zeros(self.codes.shape[0], dtype=np.int64)
        for s, idx in zip(strides, pidx):
            gp = gp + self.codes[:, idx] * int(s)
        combined = gp * r + self.codes[:, vi]
        counts = np.bincount(combined, minlength=q * r).astype(np.float64)
        mat = counts.reshape(q, r)
        n_pa = mat.sum(axis=1)
        rows = n_pa > 0
        if np.any(rows):
            mat2 = mat[rows]
            npa2 = n_pa[rows][:, None]
            ll = float((mat2 * (np.log(np.clip(mat2, 1, None)) - np.log(npa2))).sum())
        else:
            ll = 0.0
        num_parents_states = q
        return ll - num_parents_states * (r - 1)

    def _local_score_gpu(self, vi, pidx):
        import torch
        r = int(self.card[vi])
        if r == 0:
            return 0.0
        try:
            if len(pidx) == 0:
                tx = self.tcodes[:, vi]
                counts = torch.bincount(tx, minlength=r).double()
                total = counts.sum()
                if float(total.item()) == 0.0:
                    return 0.0
                mask = counts > 0
                ll = float((counts[mask] * (torch.log(counts[mask]) - torch.log(total))).sum().item())
                num_parents_states = 1
                return ll - num_parents_states * (r - 1)
            strides = []
            q = 1
            for idx in pidx:
                strides.append(q)
                q *= int(self.card[idx])
            if q == 0:
                return 0.0
            gp = torch.zeros(self.tcodes.shape[0], dtype=torch.int64, device=self.device)
            for s, idx in zip(strides, pidx):
                gp = gp + self.tcodes[:, idx] * int(s)
            combined = gp * r + self.tcodes[:, vi]
            counts = torch.bincount(combined, minlength=q * r).double()
            mat = counts.view(q, r)
            n_pa = mat.sum(dim=1)
            rows_mask = n_pa > 0
            if torch.any(rows_mask):
                mat2 = mat[rows_mask]
                npa2 = n_pa[rows_mask].unsqueeze(1)
                ll = float((mat2 * (torch.log(mat2.clamp_min(1)) - torch.log(npa2))).sum().item())
            else:
                ll = 0.0
            num_parents_states = q
            return ll - num_parents_states * (r - 1)
        except RuntimeError as e:
            if "CUDA" in str(e):
                self.use_gpu = False
                return self._local_score_cpu(vi, pidx)
            raise

    def local_score(self, variable, parents):
        vi = self.cols.index(variable) if isinstance(variable, str) else int(variable)
        pidx = [(self.cols.index(p) if isinstance(p, str) else int(p)) for p in (parents or [])]
        key = (vi, tuple(sorted(pidx)))
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if self.use_gpu:
            val = self._local_score_gpu(vi, pidx)
        else:
            val = self._local_score_cpu(vi, pidx)
        self.cache[key] = val
        return val

    def structure_prior(self, model):
        return 0

    def structure_prior_ratio(self, operation):
        return 0

def create_output_folder():
    """创建输出文件夹"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "03贪婪等价搜索结果")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_dag_results(dag, output_folder, df_columns):
    """保存DAG结果到文件"""
    edges = list(dag.edges())
    
    # 保存TXT格式
    txt_file = os.path.join(output_folder, "GreedyEquivalence_AIC-D_因果边完整.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("贪婪等价搜索 (GES-AIC-D) 发现的因果边\n")
        f.write("=" * 40 + "\n")
        for i, edge in enumerate(edges, 1):
            f.write(f"{i:3d}. {edge[0]} -> {edge[1]}\n")
    
    # 保存CSV格式
    df_edges = pd.DataFrame(edges, columns=["源节点", "目标节点"])
    csv_file = os.path.join(output_folder, "GreedyEquivalence_AIC-D_因果边列表.csv")
    df_edges.to_csv(csv_file, index=False, encoding="utf-8-sig")
    
    # 生成网络图
    plt.figure(figsize=(16, 12))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, 
                          node_color='lightgreen', 
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
    
    plt.title(f"贪婪等价搜索 (GES-AIC-D) 因果网络图\n共{len(edges)}条因果边", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    graph_file = os.path.join(output_folder, "GreedyEquivalence_AIC-D_因果网络图.png")
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
            "算法名称": "贪婪等价搜索 (Greedy Equivalence Search - GES)",
            "评分方法": "AIC-D (Akaike Information Criterion - Discrete)",
            "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "数据维度": {
                "样本数": len(df_columns),
                "变量数": len(df_columns)
            }
        },
        "网络结构": {
            "节点总数": len(dag.nodes()),
            "边总数": len(edges),
            "节点列表": list(dag.nodes()),
            "因果边列表": [{"源节点": edge[0], "目标节点": edge[1]} for edge in edges]
        },
        "统计信息": {
            "入度统计": {node: in_degrees.get(node, 0) for node in dag.nodes()},
            "出度统计": {node: out_degrees.get(node, 0) for node in dag.nodes()},
            "最大入度": max(in_degrees.values()) if in_degrees else 0,
            "最大出度": max(out_degrees.values()) if out_degrees else 0,
            "平均度数": sum(dict(G.degree()).values()) / len(dag.nodes()) if dag.nodes() else 0
        },
        "节点分析": {
            "根节点": [node for node in dag.nodes() if in_degrees.get(node, 0) == 0],
            "叶节点": [node for node in dag.nodes() if out_degrees.get(node, 0) == 0],
            "中介节点": [node for node in dag.nodes() if in_degrees.get(node, 0) > 0 and out_degrees.get(node, 0) > 0]
        }
    }
    
    json_file = os.path.join(output_folder, "GreedyEquivalence_AIC-D_因果结果.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return txt_file, csv_file, graph_file, json_file, results

def run_ges_algorithm():
    """运行贪婪等价搜索算法"""
    print("=" * 60)
    print("03 贪婪等价搜索 (GES) - 开始执行")
    print("=" * 60)
    
    # 1. 加载数据
    df = load_data()
    
    # 2. 创建输出文件夹
    output_dir = create_output_folder()
    
    # 3. 使用AIC-D评分方法运行GES算法
    mode = _detect_compute_mode()
    print(f"计算模式: {mode}")
    print("正在运行贪婪等价搜索 (GES-AIC-D评分)...")
    if mode == "gpu":
        print("GPU加速启用: 需要安装`torch`并且可用CUDA驱动 (建议CUDA 11.8/12.x)。显存不足将自动回退CPU。")
    start_time = time.time()
    
    try:
        ges = GES(df)
        if mode == "gpu":
            score = AICDScoreGPU(df, mode="gpu")
            dag = ges.estimate(scoring_method=score)
        else:
            dag = ges.estimate(scoring_method='aic-d')
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"✓ 贪婪等价搜索完成，耗时: {execution_time:.2f}秒")
        print(f"✓ 发现 {len(dag.edges())} 条因果边")
        
        # 4. 保存结果
        txt_file, csv_file, graph_file, json_file, results = save_dag_results(dag, output_dir, df.columns)
        
        # 5. 输出结果摘要
        print("\n" + "=" * 60)
        print("贪婪等价搜索执行完成 - 结果摘要")
        print("=" * 60)
        print(f"评分方法: AIC-D")
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
        
        return output_dir, len(dag.edges())
        
    except Exception as e:
        print(f"❌ 贪婪等价搜索执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        output_dir, edge_count = run_ges_algorithm()
        print(f"\n✅ 03 贪婪等价搜索执行成功！发现 {edge_count} 条因果边")
    except Exception as e:
        print(f"\n❌ 03 贪婪等价搜索执行失败: {str(e)}")
        raise
