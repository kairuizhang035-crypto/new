#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05 专家在循环 (Expert In The Loop)
使用LLM进行智能因果推断的完整版本

作者: 因果发现系统
日期: 2025年
"""

from pgmpy.utils import get_example_model, llm_pairwise_orient
from pgmpy.estimators import ExpertInLoop, ExpertKnowledge
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import json
from datetime import datetime
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import re
from litellm import completion
import pgmpy.utils as pg_utils
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# 过滤警告
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', category=ConvergenceWarning)

os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
LLM_MODEL = os.environ.get("LLM_MODEL", "ollama/qwen2.5:7b")
MODEL_CHOICES = [
    "ollama/qwen2.5:7b",
    "ollama/qwen2.5:32b",
    "ollama/mannix/qwen2-57b:latest",
    "ollama/huihui_ai/deepseek-r1-abliterated:70b",
    "ollama/deepseek-r1:32b",
]
EIL_PVAL_THRESHOLD = float(os.environ.get("EIL_PVAL_THRESHOLD", "0.1"))
EIL_EFFECT_SIZE = float(os.environ.get("EIL_EFFECT_SIZE", "0.2"))
ORIENT_CACHE = {}
LLM_WORKERS = int(os.environ.get("LLM_WORKERS", "10"))
SELECT_PAIRS_THRESHOLD = float(os.environ.get("SELECT_PAIRS_THRESHOLD", "0.45"))
MAX_LLM_PAIRS = int(os.environ.get("MAX_LLM_PAIRS", "120"))
CANDIDATE_SET = set()
LLM_BATCH_SIZE = int(os.environ.get("LLM_BATCH_SIZE", "30"))
FAST_MODE = os.environ.get("FAST_MODE", "1") == "1"
LLM_BATCH_WORKERS = int(os.environ.get("LLM_BATCH_WORKERS", "10"))
SELECTED_PAIRS = []
EDGE_SOURCE = {}
LLM_ONLY = os.environ.get("LLM_ONLY", "1") == "1"

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
    output_dir = os.path.join(script_dir, "05专家在循环结果")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def preprocess_data(df):
    """数据预处理"""
    print("正在进行数据质量检查...")
    
    # 处理NaN值
    if df.isnull().values.any():
        print("数据中存在 NaN 值，使用均值填充")
        df = df.fillna(df.mean())
    
    # 移除零方差列
    zero_var_cols = df.columns[df.var() == 0]
    if not zero_var_cols.empty:
        print(f"移除零方差列: {list(zero_var_cols)}")
        df = df.drop(columns=zero_var_cols)
    
    # 处理多重共线性
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"移除高度共线列: {to_drop}")
    
    # 方差阈值过滤
    selector = VarianceThreshold(threshold=0.01)
    df_transformed = selector.fit_transform(df)
    
    if df_transformed.shape[1] < df.shape[1]:
        retained_cols = df.columns[selector.get_support()]
        df = pd.DataFrame(df_transformed, columns=retained_cols, index=df.index)
        print(f"VarianceThreshold移除了 {df.shape[1] - df_transformed.shape[1]} 个低方差列")
    
    print(f"✓ 数据预处理完成，最终维度: {df.shape}")
    return df

def create_variable_descriptions(df):
    """创建变量描述字典"""
    variable_descriptions = {}
    for col in df.columns:
        variable_descriptions[col] = f"Binary indicator: {col} (yes/no)"
    return variable_descriptions

def robust_llm_orient(u, v, variable_descriptions=None, llm_model=LLM_MODEL, **kwargs):
    """稳健的LLM定向函数"""
    if variable_descriptions is None:
        variable_descriptions = {}
    
    try:
        # 使用原始的LLM定向函数
        result = llm_pairwise_orient(u, v, variable_descriptions, llm_model)
        return result
    except Exception as e:
        print(f"LLM定向失败 ({u} <-> {v}): {e}")
        # 使用字典序作为回退
        return (u, v) if str(u) < str(v) else (v, u)

def _category(x):
    if isinstance(x, str):
        if x.startswith("疾病_"):
            return "疾病"
        if x.startswith("药物_"):
            return "药物"
        if x.startswith("检验_"):
            return "检验"
    return "其他"

def rule_based_orient(u, v):
    cu, cv = _category(u), _category(v)
    if cu == "药物" and cv == "检验":
        return (u, v)
    if cu == "检验" and cv == "药物":
        return (v, u)
    if cu == "疾病" and cv == "检验":
        return (u, v)
    if cu == "检验" and cv == "疾病":
        return (v, u)
    if cu == "疾病" and cv == "药物":
        return (u, v)
    if cu == "药物" and cv == "疾病":
        return (v, u)
    return None

def fast_llm_pairwise_orient(u, v, variable_descriptions=None, llm_model=LLM_MODEL):
    key = (str(u), str(v)) if str(u) < str(v) else (str(v), str(u))
    if key in ORIENT_CACHE:
        return ORIENT_CACHE[key]
    if key not in CANDIDATE_SET:
        rb = rule_based_orient(u, v)
        if rb is not None:
            ORIENT_CACHE[key] = rb
            EDGE_SOURCE[key] = "rule"
            return ORIENT_CACHE[key]
        ORIENT_CACHE[key] = (u, v) if str(u) < str(v) else (v, u)
        EDGE_SOURCE[key] = "fallback"
        return ORIENT_CACHE[key]
    sys_msg = {
        "role": "system",
        "content": "只输出一行，格式严格为 'A->B' 或 'B->A'，不添加任何解释。"
    }
    desc_u = variable_descriptions.get(u, str(u)) if variable_descriptions else str(u)
    desc_v = variable_descriptions.get(v, str(v)) if variable_descriptions else str(v)
    user_msg = {
        "role": "user",
        "content": f"变量A: {u}；描述: {desc_u}\n变量B: {v}；描述: {desc_v}\n仅输出一个结果: '{u}->{v}' 或 '{v}->{u}'。"
    }
    try:
        resp = completion(model=llm_model, messages=[sys_msg, user_msg], temperature=0, max_tokens=16, timeout=15)
        text = resp["choices"][0]["message"]["content"].strip()
        p1 = rf"{re.escape(str(u))}\s*(?:->|=>|→)\s*{re.escape(str(v))}"
        p2 = rf"{re.escape(str(v))}\s*(?:->|=>|→)\s*{re.escape(str(u))}"
        if re.search(p1, text, re.IGNORECASE):
            ORIENT_CACHE[key] = (u, v)
            EDGE_SOURCE[key] = "llm"
            return ORIENT_CACHE[key]
        if re.search(p2, text, re.IGNORECASE):
            ORIENT_CACHE[key] = (v, u)
            EDGE_SOURCE[key] = "llm"
            return ORIENT_CACHE[key]
        rb = rule_based_orient(u, v)
        if rb is not None:
            ORIENT_CACHE[key] = rb
            EDGE_SOURCE[key] = "rule"
            return ORIENT_CACHE[key]
        ORIENT_CACHE[key] = (u, v) if str(u) < str(v) else (v, u)
        EDGE_SOURCE[key] = "fallback"
        return ORIENT_CACHE[key]
    except Exception:
        rb = rule_based_orient(u, v)
        if rb is not None:
            ORIENT_CACHE[key] = rb
            EDGE_SOURCE[key] = "rule"
            return ORIENT_CACHE[key]
        ORIENT_CACHE[key] = (u, v) if str(u) < str(v) else (v, u)
        EDGE_SOURCE[key] = "fallback"
        return ORIENT_CACHE[key]

def batched_llm_orient(pairs, variable_descriptions=None, llm_model=LLM_MODEL):
    lines = []
    for u, v in pairs:
        du = variable_descriptions.get(u, str(u)) if variable_descriptions else str(u)
        dv = variable_descriptions.get(v, str(v)) if variable_descriptions else str(v)
        lines.append(f"A={u};desc={du} | B={v};desc={dv}")
    sys_msg = {"role": "system", "content": "仅输出若干行，每行格式 'X->Y' 或 'Y->X'，不含其它内容。"}
    user_msg = {"role": "user", "content": "\n".join(lines)}
    try:
        resp = completion(model=llm_model, messages=[sys_msg, user_msg], temperature=0, max_tokens=LLM_BATCH_SIZE*16, timeout=45)
        text = resp["choices"][0]["message"]["content"].strip()
        outs = [t.strip() for t in text.splitlines() if t.strip()]
        assigned = set()
        for out in outs:
            for u, v in pairs:
                k = (str(u), str(v)) if str(u) < str(v) else (str(v), str(u))
                if k in assigned:
                    continue
                p1 = rf"{re.escape(str(u))}\s*(?:->|=>|→)\s*{re.escape(str(v))}"
                p2 = rf"{re.escape(str(v))}\s*(?:->|=>|→)\s*{re.escape(str(u))}"
                if re.search(p1, out, re.IGNORECASE):
                    ORIENT_CACHE[k] = (u, v)
                    EDGE_SOURCE[k] = "llm"
                    assigned.add(k)
                elif re.search(p2, out, re.IGNORECASE):
                    ORIENT_CACHE[k] = (v, u)
                    EDGE_SOURCE[k] = "llm"
                    assigned.add(k)
        for u, v in pairs:
            k = (str(u), str(v)) if str(u) < str(v) else (str(v), str(u))
            if k in assigned:
                continue
            rb = rule_based_orient(u, v)
            if rb is not None:
                ORIENT_CACHE[k] = rb
                EDGE_SOURCE[k] = "rule"
            else:
                ORIENT_CACHE[k] = (u, v) if str(u) < str(v) else (v, u)
                EDGE_SOURCE[k] = "fallback"
    except Exception:
        for u, v in pairs:
            k = (str(u), str(v)) if str(u) < str(v) else (str(v), str(u))
            rb = rule_based_orient(u, v)
            if rb is not None:
                ORIENT_CACHE[k] = rb
                EDGE_SOURCE[k] = "rule"
            else:
                ORIENT_CACHE[k] = (u, v) if str(u) < str(v) else (v, u)
                EDGE_SOURCE[k] = "fallback"

def precompute_orientations(df, variable_descriptions):
    corr = df.corr().abs()
    pairs = []
    cols = list(df.columns)
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            s = corr.loc[cols[i], cols[j]]
            if s >= SELECT_PAIRS_THRESHOLD:
                pairs.append((cols[i], cols[j], float(s)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    global SELECTED_PAIRS
    SELECTED_PAIRS = [(u, v) for u, v, _ in pairs[:MAX_LLM_PAIRS]]
    for u, v in SELECTED_PAIRS:
        k = (str(u), str(v)) if str(u) < str(v) else (str(v), str(u))
        CANDIDATE_SET.add(k)
    raw_count = len(pairs)
    if SELECTED_PAIRS:
        chunks = [SELECTED_PAIRS[i:i+LLM_BATCH_SIZE] for i in range(0, len(SELECTED_PAIRS), LLM_BATCH_SIZE)]
        total = len(chunks)
        print(f"相关阈值{SELECT_PAIRS_THRESHOLD}下共{raw_count}对；进入LLM定向{len(SELECTED_PAIRS)}对（上限{MAX_LLM_PAIRS}）")
        print(f"预定向批次: {total}，候选对数: {len(SELECTED_PAIRS)}")
        done = 0
        with ThreadPoolExecutor(max_workers=LLM_BATCH_WORKERS) as ex:
            futs = [ex.submit(batched_llm_orient, ch, variable_descriptions, LLM_MODEL) for ch in chunks]
            for _ in as_completed(futs):
                done += 1
                if done % 2 == 0 or done == total:
                    print(f"预定向进度: {done}/{total}")

def save_dag_results(dag, output_folder, df_columns):
    """保存DAG结果到文件"""
    edges = list(dag.edges())
    
    # 保存TXT格式
    txt_file = os.path.join(output_folder, "ExpertInLoop_因果边完整.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("专家在循环 (Expert In The Loop) 发现的因果边\n")
        f.write("=" * 50 + "\n")
        for i, edge in enumerate(edges, 1):
            f.write(f"{i:3d}. {edge[0]} -> {edge[1]}\n")
    
    # 保存CSV格式
    df_edges = pd.DataFrame(edges, columns=["源节点", "目标节点"])
    csv_file = os.path.join(output_folder, "ExpertInLoop_因果边列表.csv")
    df_edges.to_csv(csv_file, index=False, encoding="utf-8-sig")
    
    # 生成网络图
    plt.figure(figsize=(16, 12))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    if len(edges) > 0:
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, 
                              node_color='lightpink', 
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
    
    plt.title(f"专家在循环 (Expert In The Loop) 因果网络图\n共{len(edges)}条因果边", 
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    graph_file = os.path.join(output_folder, "ExpertInLoop_因果网络图.png")
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
            "算法名称": "专家在循环 (Expert In The Loop)",
            "策略": "LLM智能定向",
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
    
    json_file = os.path.join(output_folder, "ExpertInLoop_因果结果.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return txt_file, csv_file, graph_file, json_file, results

def run_expert_in_loop_algorithm():
    global LLM_MODEL, SELECT_PAIRS_THRESHOLD, MAX_LLM_PAIRS, LLM_ONLY
    """运行专家在循环算法"""
    print("=" * 60)
    print("05 专家在循环 (Expert In The Loop) - 开始执行")
    print("使用LLM进行智能因果推断")
    print("=" * 60)
    try:
        print("模型选择: 0:qwen2.5:7b  1:qwen2.5:32b  2:mannix/qwen2-57b:latest  3:huihui_ai/deepseek-r1-abliterated:70b  4:deepseek-r1:32b")
        s = input("请选择模型编号(默认0): ").strip()
        idx = 0 if s == "" else int(s)
        if 0 <= idx < len(MODEL_CHOICES):
            LLM_MODEL = MODEL_CHOICES[idx]
            os.environ["LLM_MODEL"] = LLM_MODEL
            print(f"使用模型: {LLM_MODEL}")
        t = input(f"候选筛选阈值(当前{SELECT_PAIRS_THRESHOLD}): ").strip()
        if t:
            try:
                v = float(t)
                SELECT_PAIRS_THRESHOLD = v
                os.environ["SELECT_PAIRS_THRESHOLD"] = str(v)
                print(f"设定阈值为: {SELECT_PAIRS_THRESHOLD}")
            except:
                pass
        m = input(f"候选数量上限(当前{MAX_LLM_PAIRS}): ").strip()
        if m:
            try:
                mv = int(m)
                MAX_LLM_PAIRS = mv
                os.environ["MAX_LLM_PAIRS"] = str(mv)
                print(f"设定上限为: {MAX_LLM_PAIRS}")
            except:
                pass
        yn = input(f"仅添加来源为LLM的边(当前{'是' if LLM_ONLY else '否'}) [Y/n]: ").strip().lower()
        if yn in ("", "y", "yes"):
            LLM_ONLY = True
            os.environ["LLM_ONLY"] = "1"
        elif yn in ("n", "no"):
            LLM_ONLY = False
            os.environ["LLM_ONLY"] = "0"
        print(f"仅添加LLM边: {'是' if LLM_ONLY else '否'}")
    except Exception:
        pass
    
    # 1. 加载数据
    df = load_data()
    
    # 2. 创建输出文件夹
    output_dir = create_output_folder()
    
    # 3. 数据预处理
    df_processed = preprocess_data(df)
    
    # 4. 创建变量描述
    variable_descriptions = create_variable_descriptions(df_processed)
    print(f"✓ 创建了{len(variable_descriptions)}个变量的描述")
    
    # 5. 使用Expert-in-the-Loop进行因果发现
    print("使用Expert-in-the-Loop方法，结合LLM进行边定向...")
    start_time = time.time()
    
    try:
        # 创建ExpertInLoop估计器
        precompute_orientations(df_processed, variable_descriptions)
        pg_utils.llm_pairwise_orient = fast_llm_pairwise_orient
        if FAST_MODE:
            from pgmpy.base import DAG
            dag = DAG()
            dag.add_nodes_from(df_processed.columns)
            processed = 0
            total_pairs = len(SELECTED_PAIRS)
            for u, v in SELECTED_PAIRS:
                k = (str(u), str(v)) if str(u) < str(v) else (str(v), str(u))
                o = ORIENT_CACHE.get(k)
                o = o if o is not None else fast_llm_pairwise_orient(u, v, variable_descriptions, LLM_MODEL)
                src = EDGE_SOURCE.get(k, "unknown")
                if LLM_ONLY and src != "llm":
                    processed += 1
                    if processed % 20 == 0 or processed == total_pairs:
                        print(f"构图进度: 已处理候选对 {processed}/{total_pairs}")
                    continue
                try:
                    dag.add_edge(o[0], o[1])
                except:
                    pass
                processed += 1
                if processed % 20 == 0 or processed == total_pairs:
                    print(f"构图进度: 已处理候选对 {processed}/{total_pairs}")
            keys = [(str(u), str(v)) if str(u) < str(v) else (str(v), str(u)) for u, v in SELECTED_PAIRS]
            c_llm = sum(1 for k in keys if EDGE_SOURCE.get(k) == "llm")
            c_rule = sum(1 for k in keys if EDGE_SOURCE.get(k) == "rule")
            c_fb = sum(1 for k in keys if EDGE_SOURCE.get(k) == "fallback")
            print(f"定向来源: LLM={c_llm} 规则={c_rule} 回退={c_fb}")
            learned_dag = dag
        else:
            estimator = ExpertInLoop(df_processed)
            learned_dag = estimator.estimate(
                pval_threshold=EIL_PVAL_THRESHOLD,
                effect_size_threshold=EIL_EFFECT_SIZE,
                variable_descriptions=variable_descriptions,
                llm_model=LLM_MODEL,
                use_cache=True,
                show_progress=True
            )
        
        if learned_dag is None:
            raise ValueError("ExpertInLoop.estimate() 返回了 None")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"✓ 专家在循环完成，耗时: {execution_time:.2f}秒")
        print(f"✓ 发现 {len(learned_dag.edges())} 条因果边")
        
        # 6. 保存结果
        txt_file, csv_file, graph_file, json_file, results = save_dag_results(learned_dag, output_dir, df_processed.columns)
        
        # 7. 输出结果摘要
        print("\n" + "=" * 60)
        print("专家在循环执行完成 - 结果摘要")
        print("=" * 60)
        print(f"策略: LLM智能定向")
        print(f"执行时间: {execution_time:.2f}秒")
        print(f"数据维度: {df_processed.shape[0]} × {df_processed.shape[1]}")
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
        
        return output_dir, len(learned_dag.edges())
        
    except Exception as e:
        print(f"❌ 专家在循环执行失败: {str(e)}")
        # 使用快速回退策略
        print("使用快速回退策略...")
        from pgmpy.base import DAG
        
        dag = DAG()
        dag.add_nodes_from(df_processed.columns)
        
        # 基于相关性添加边
        corr_matrix = df_processed.corr().abs()
        edges_added = 0
        max_edges = 50
        
        for i, col1 in enumerate(df_processed.columns):
            for j, col2 in enumerate(df_processed.columns):
                if i < j and edges_added < max_edges:
                    corr_val = corr_matrix.loc[col1, col2]
                    if corr_val >= 0.3:
                        try:
                            dag.add_edge(col1, col2)
                            edges_added += 1
                        except:
                            continue
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"✓ 快速回退完成，耗时: {execution_time:.2f}秒")
        print(f"✓ 发现 {len(dag.edges())} 条因果边")
        
        txt_file, csv_file, graph_file, json_file, results = save_dag_results(dag, output_dir, df_processed.columns)
        
        return output_dir, len(dag.edges())

if __name__ == "__main__":
    import time
    try:
        output_dir, edge_count = run_expert_in_loop_algorithm()
        print(f"\n✅ 05 专家在循环执行成功！发现 {edge_count} 条因果边")
    except Exception as e:
        print(f"\n❌ 05 专家在循环执行失败: {str(e)}")
        raise
