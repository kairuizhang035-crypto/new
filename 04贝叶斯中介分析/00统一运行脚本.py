#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
贝叶斯中介分析统一运行脚本
依次执行：
1. 01提取完整中介路径.py - 从精简因果边列表提取中介路径
2. 02贝叶斯中介分析.py - 进行贝叶斯中介效应分析
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_script(script_path, script_name, interactive=False):
    """
    运行指定的Python脚本
    
    Args:
        script_path: 脚本的完整路径
        script_name: 脚本名称（用于显示）
        interactive: 是否为交互式脚本
    
    Returns:
        bool: 是否成功运行
    """
    print(f"\n{'='*60}")
    print(f"开始运行: {script_name}")
    print(f"脚本路径: {script_path}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if interactive:
        print(f"模式: 交互式运行")
    print(f"{'='*60}")
    
    try:
        # 检查脚本是否存在
        if not os.path.exists(script_path):
            print(f"错误：脚本文件不存在 - {script_path}")
            return False
        
        # 运行脚本
        start_time = time.time()
        
        if interactive:
            # 交互式运行：不捕获输出，允许用户交互
            result = subprocess.run([sys.executable, script_path])
            returncode = result.returncode
            stdout = ""
            stderr = ""
        else:
            # 非交互式运行：捕获输出
            result = subprocess.run([sys.executable, script_path], 
                                  capture_output=True, 
                                  text=True, 
                                  encoding='utf-8')
            returncode = result.returncode
            stdout = result.stdout
            stderr = result.stderr
        
        end_time = time.time()
        
        # 显示运行结果
        print(f"\n运行时间: {end_time - start_time:.2f} 秒")
        
        if returncode == 0:
            print(f"✓ {script_name} 运行成功！")
            if stdout:
                print(f"\n标准输出:")
                print(stdout)
        else:
            print(f"✗ {script_name} 运行失败！")
            print(f"返回码: {returncode}")
            if stderr:
                print(f"\n错误信息:")
                print(stderr)
            if stdout:
                print(f"\n标准输出:")
                print(stdout)
            return False
        
        return True
        
    except Exception as e:
        print(f"运行 {script_name} 时发生异常: {str(e)}")
        return False

def main():
    """
    主函数：依次运行贝叶斯中介分析的两个脚本
    """
    print("="*80)
    print("贝叶斯中介分析统一运行脚本")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义要运行的脚本列表
    scripts = [
        {
            'path': os.path.join(script_dir, '01提取完整中介路径.py'),
            'name': '01提取完整中介路径.py',
            'description': '从精简因果边列表提取完整中介路径',
            'interactive': False
        },
        {
            'path': os.path.join(script_dir, '02贝叶斯中介分析.py'),
            'name': '02贝叶斯中介分析.py',
            'description': '基于中介路径进行贝叶斯中介效应分析（交互式选择路径数量）',
            'interactive': True
        },
        {
            'path': os.path.join(script_dir, '可视化.py'),
            'name': '可视化.py',
            'description': '贝叶斯中介分析结果可视化',
            'interactive': False
        },
        {
            'path': os.path.join(script_dir, '02贝叶斯中介分析结果', '生成详细CSV报告.py'),
            'name': '生成详细CSV报告.py',
            'description': '生成贝叶斯中介分析详细CSV报告',
            'interactive': False
        }
    ]
    
    # 显示运行计划
    print(f"\n运行计划:")
    for i, script in enumerate(scripts, 1):
        print(f"{i}. {script['name']} - {script['description']}")
    
    # 依次运行脚本
    success_count = 0
    total_start_time = time.time()
    
    for i, script in enumerate(scripts, 1):
        print(f"\n{'='*80}")
        print(f"步骤 {i}/{len(scripts)}: {script['description']}")
        print(f"{'='*80}")
        
        success = run_script(script['path'], script['name'], script.get('interactive', False))
        
        if success:
            success_count += 1
            print(f"\n✓ 步骤 {i} 完成")
        else:
            print(f"\n✗ 步骤 {i} 失败")
            print(f"由于步骤 {i} 失败，后续步骤可能无法正常运行")
            # 询问是否继续
            try:
                continue_choice = input("\n是否继续运行后续步骤？(y/n): ").lower().strip()
                if continue_choice not in ['y', 'yes', '是']:
                    print("用户选择停止运行")
                    break
            except KeyboardInterrupt:
                print("\n用户中断运行")
                break
        
        # 在步骤之间添加短暂延迟
        if i < len(scripts):
            print(f"\n等待 2 秒后继续下一步...")
            time.sleep(2)
    
    # 显示最终结果
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print(f"\n{'='*80}")
    print("运行完成汇总")
    print(f"{'='*80}")
    print(f"总运行时间: {total_time:.2f} 秒")
    print(f"成功运行: {success_count}/{len(scripts)} 个脚本")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_count == len(scripts):
        print(f"\n🎉 所有脚本运行成功！")
        print(f"\n生成的结果文件夹:")
        print(f"1. 01中介路径分析结果/ - 中介路径提取结果")
        print(f"2. 02贝叶斯中介分析结果/ - 贝叶斯中介分析结果")
        print(f"3. 可视化/ - 贝叶斯中介分析可视化结果")
        print(f"4. 02贝叶斯中介分析结果/贝叶斯中介分析详细结果.csv - 详细CSV报告")
    else:
        print(f"\n⚠️  有 {len(scripts) - success_count} 个脚本运行失败")
        print(f"请检查上述错误信息并修复问题后重新运行")
    
    return success_count == len(scripts)

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\n用户中断运行")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n运行过程中发生未预期的错误: {str(e)}")
        sys.exit(1)
