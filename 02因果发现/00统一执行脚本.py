#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06 统一执行脚本
按顺序运行01-05所有因果发现算法

作者: 因果发现系统
日期: 2025年
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_header(title):
    """打印标题"""
    print("\n" + "=" * 80)
    print(f" {title} ")
    print("=" * 80)

def print_step(step_num, total_steps, algorithm_name):
    """打印步骤信息"""
    print(f"\n[步骤 {step_num}/{total_steps}] 正在执行: {algorithm_name}")
    print("-" * 60)

def run_algorithm_script(script_path, algorithm_name):
    """运行单个算法脚本"""
    start_time = time.time()
    
    try:
        # 检查脚本文件是否存在
        if not os.path.exists(script_path):
            print(f"❌ 脚本文件不存在: {script_path}")
            return False, 0, f"脚本文件不存在: {script_path}"
        
        print(f"🚀 开始执行: {algorithm_name}")
        print(f"📄 脚本路径: {script_path}")
        
        # 运行脚本
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {algorithm_name} 执行成功")
            print(f"⏱️  执行时间: {execution_time:.2f}秒")
            
            # 显示输出的最后几行（如果有的话）
            if result.stdout:
                output_lines = result.stdout.strip().split('\n')
                if len(output_lines) > 0:
                    print("📋 执行输出:")
                    for line in output_lines[-5:]:  # 显示最后5行
                        if line.strip():
                            print(f"    {line}")
            
            return True, execution_time, "成功"
        else:
            print(f"❌ {algorithm_name} 执行失败")
            print(f"⏱️  执行时间: {execution_time:.2f}秒")
            print(f"🔍 错误代码: {result.returncode}")
            
            if result.stderr:
                print("❗ 错误信息:")
                error_lines = result.stderr.strip().split('\n')
                for line in error_lines[-10:]:  # 显示最后10行错误
                    if line.strip():
                        print(f"    {line}")
            
            return False, execution_time, f"执行失败 (代码: {result.returncode})"
            
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"❌ {algorithm_name} 执行异常: {str(e)}")
        return False, execution_time, f"执行异常: {str(e)}"

def check_results(script_dir):
    """检查各算法的结果文件"""
    result_folders = [
        "01PC算法结果",
        "02爬山算法结果", 
        "03贪婪等价搜索结果",
        "04树搜索结果",
        "05专家在循环结果",
        "可视化"
    ]
    
    results_summary = {}
    
    print("\n📊 检查算法结果:")
    print("-" * 40)
    
    for folder in result_folders:
        folder_path = os.path.join(script_dir, folder)
        algorithm_name = folder.replace("结果", "")
        
        if os.path.exists(folder_path):
            # 统计文件数量
            files = os.listdir(folder_path)
            csv_files = [f for f in files if f.endswith('.csv')]
            json_files = [f for f in files if f.endswith('.json')]
            png_files = [f for f in files if f.endswith('.png')]
            
            results_summary[algorithm_name] = {
                "状态": "成功",
                "文件总数": len(files),
                "CSV文件": len(csv_files),
                "JSON文件": len(json_files),
                "图片文件": len(png_files),
                "路径": folder_path
            }
            
            print(f"✅ {algorithm_name}: {len(files)} 个文件 (CSV:{len(csv_files)}, JSON:{len(json_files)}, PNG:{len(png_files)})")
        else:
            results_summary[algorithm_name] = {
                "状态": "失败",
                "文件总数": 0,
                "路径": folder_path
            }
            print(f"❌ {algorithm_name}: 结果文件夹不存在")
    
    return results_summary

def run_all_algorithms():
    """运行所有算法"""
    print_header("02阶段 因果发现算法 统一执行")
    
    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义要执行的算法脚本
    algorithms = [
        {
            "name": "01 PC算法 (基于约束的估计器)",
            "script": "01PC算法.py",
            "description": "使用PC算法进行因果发现，基于条件独立性测试",
            "required": True
        },
        {
            "name": "02 爬山算法 (Hill Climbing)",
            "script": "02爬山算法.py", 
            "description": "使用爬山搜索算法，基于AIC-D评分标准",
            "required": True
        },
        {
            "name": "03 贪婪等价搜索 (GES)",
            "script": "03贪婪等价搜索.py",
            "description": "使用贪婪等价搜索算法，基于AIC-D评分标准",
            "required": True
        },
        {
            "name": "04 树搜索 (TAN方法)",
            "script": "04树搜索.py",
            "description": "使用树增强朴素贝叶斯方法，以第一个节点为根",
            "required": True
        },
        {
            "name": "05 专家在循环 (Expert In The Loop)",
            "script": "05专家在循环.py",
            "description": "使用专家在循环方法，基于LLM的智能因果推断",
            "required": False  # 可选执行
        },
        {
            "name": "06 因果边筛选算法",
            "script": "06因果边筛选算法.py",
            "description": "对所有算法发现的因果边进行筛选和评分",
            "required": True
        },
        {
            "name": "可视化分析",
            "script": "可视化.py",
            "description": "对因果发现结果进行综合可视化分析",
            "required": True
        }
    ]
    
    print(f"📂 工作目录: {script_dir}")
    print(f"🔢 发现 {len(algorithms)} 个算法")
    
    # 询问是否执行专家在循环算法
    expert_algorithm = next((alg for alg in algorithms if "专家在循环" in alg["name"]), None)
    if expert_algorithm:
        print(f"\n⚠️  注意: {expert_algorithm['name']} 执行时间较长（可能需要几分钟）")
        print("   该算法使用LLM进行智能因果推断，但执行速度较慢")
        
        while True:
            choice = input(f"\n是否执行 {expert_algorithm['name']}? (y/n): ").lower().strip()
            if choice in ['y', 'yes', '是', '执行']:
                run_expert = True
                break
            elif choice in ['n', 'no', '否', '跳过']:
                run_expert = False
                break
            else:
                print("请输入 y/yes/是/执行 或 n/no/否/跳过")
        
        if not run_expert:
            print(f"⏭️  跳过执行 {expert_algorithm['name']}")
            algorithms = [alg for alg in algorithms if "专家在循环" not in alg["name"]]
    
    total_start_time = time.time()
    successful_count = 0
    execution_results = []
    
    print(f"\n🚀 开始执行 {len(algorithms)} 个算法")
    
    # 逐个执行算法
    for i, algorithm in enumerate(algorithms, 1):
        print_step(i, len(algorithms), algorithm["name"])
        
        script_path = os.path.join(script_dir, algorithm["script"])
        
        # 执行算法
        success, exec_time, status = run_algorithm_script(script_path, algorithm["name"])
        
        # 记录执行结果
        execution_results.append({
            "算法名称": algorithm["name"],
            "执行状态": "成功" if success else "失败",
            "执行时间": exec_time,
            "状态信息": status
        })
        
        if success:
            successful_count += 1
        
        # 短暂暂停，避免资源冲突
        if i < len(algorithms):
            time.sleep(1)
    
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    
    # 检查结果文件
    results_summary = check_results(script_dir)
    
    # 输出总结
    print_header("执行完成 - 总结报告")
    
    print(f"📊 执行统计:")
    print(f"  - 总算法数量: {len(algorithms)}")
    print(f"  - 成功执行: {successful_count}")
    print(f"  - 执行失败: {len(algorithms) - successful_count}")
    print(f"  - 成功率: {(successful_count / len(algorithms)) * 100:.1f}%")
    print(f"  - 总执行时间: {total_execution_time:.2f}秒")
    
    print(f"\n📋 各算法执行情况:")
    for result in execution_results:
        status_icon = "✅" if result["执行状态"] == "成功" else "❌"
        print(f"  {status_icon} {result['算法名称']}: {result['执行时间']:.2f}秒 - {result['状态信息']}")
    
    print(f"\n📁 结果文件:")
    for alg_name, summary in results_summary.items():
        if summary["状态"] == "成功":
            print(f"  ✅ {alg_name}: {summary['文件总数']} 个文件")
        else:
            print(f"  ❌ {alg_name}: 无结果文件")
    
    if successful_count == len(algorithms):
        print(f"\n🎉 所有算法执行成功！可以继续执行 07初步合并因果边.py")
        return True
    else:
        print(f"\n⚠️  有 {len(algorithms) - successful_count} 个算法执行失败，请检查错误信息")
        return False

if __name__ == "__main__":
    try:
        success = run_all_algorithms()
        
        if success:
            print(f"\n✅ 06 统一执行脚本完成！所有算法执行成功")
            sys.exit(0)
        else:
            print(f"\n❌ 06 统一执行脚本完成，但有算法执行失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断执行")
        sys.exit(2)
    except Exception as e:
        print(f"\n❌ 06 统一执行脚本异常: {str(e)}")
        sys.exit(3)