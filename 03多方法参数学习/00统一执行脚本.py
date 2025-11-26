#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多方法参数学习统一执行脚本
依次运行01-06所有参数学习方法和分析脚本
包括：MLE、Bayesian、EM、SEM、边级似然增益、参数稳定性
"""

import os
import sys
import subprocess
import time
from datetime import datetime
import json

class ParameterLearningPipeline:
    """参数学习流水线执行器"""
    
    def __init__(self, base_dir=None):
        """
        初始化流水线执行器
        
        Args:
            base_dir: 基础目录路径，默认为脚本所在目录
        """
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.scripts = [
            ("01最大似然估计器.py", "最大似然估计 (MLE)"),
            ("02贝叶斯估计器.py", "贝叶斯估计 (Bayesian)"),
            ("03期望最大化(EM).py", "期望最大化 (EM)"),
            ("04结构方程模型估计器.py", "结构方程模型 (SEM)"),
            ("05边级似然增益.py", "边级似然增益分析"),
            ("06参数稳定性.py", "参数稳定性分析"),
            ("可视化.py", "参数学习结果可视化")
        ]
        self.execution_log = []
        
    def check_prerequisites(self):
        """检查执行前提条件"""
        print("检查执行前提条件...")
        
        # 检查数据文件 - 使用实际的数据文件路径
        data_file_paths = [
            os.path.join(self.base_dir, "processed_data.csv"),  # 原始期望路径
            os.path.join(os.path.dirname(self.base_dir), "01数据预处理", "整合转置数据_完整.csv"),  # 实际路径
            os.path.join(os.path.dirname(self.base_dir), "01数据预处理", "缩减数据_规格.csv")  # 备选路径
        ]
        
        data_file = None
        for path in data_file_paths:
            if os.path.exists(path):
                data_file = path
                print(f"✓ 找到数据文件: {data_file}")
                break
        
        if not data_file:
            print(f"错误: 数据文件不存在，检查了以下路径:")
            for path in data_file_paths:
                print(f"  - {path}")
            return False
        
        # 检查因果边文件 - 使用实际的因果边文件路径
        edges_file_paths = [
            os.path.join(self.base_dir, "因果边.txt"),  # 原始期望路径
            os.path.join(os.path.dirname(self.base_dir), "02因果发现", "06候选因果边集合", "高质量因果边候选集.csv"),  # 实际路径
            os.path.join(os.path.dirname(self.base_dir), "02因果发现", "06候选因果边集合", "精简因果边列表.csv")  # 备选路径
        ]
        
        edges_file = None
        for path in edges_file_paths:
            if os.path.exists(path):
                edges_file = path
                print(f"✓ 找到因果边文件: {edges_file}")
                break
        
        if not edges_file:
            print(f"错误: 因果边文件不存在，检查了以下路径:")
            for path in edges_file_paths:
                print(f"  - {path}")
            return False
        
        # 检查脚本文件
        missing_scripts = []
        for script_name, _ in self.scripts:
            script_path = os.path.join(self.base_dir, script_name)
            if not os.path.exists(script_path):
                missing_scripts.append(script_name)
        
        if missing_scripts:
            print(f"错误: 以下脚本文件不存在:")
            for script in missing_scripts:
                print(f"  - {script}")
            return False
        
        print("✓ 所有前提条件检查通过")
        
        # 将找到的文件路径保存为实例变量，供后续使用
        self.data_file = data_file
        self.edges_file = edges_file
        
        return True
    
    def run_script(self, script_name, description):
        """
        运行单个脚本
        
        Args:
            script_name: 脚本文件名
            description: 脚本描述
            
        Returns:
            bool: 执行是否成功
        """
        script_path = os.path.join(self.base_dir, script_name)
        
        print(f"\n{'='*60}")
        print(f"开始执行: {description}")
        print(f"脚本: {script_name}")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # 使用subprocess运行脚本
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=1800  # 30分钟超时
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 记录执行日志
            log_entry = {
                'script': script_name,
                'description': description,
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.fromtimestamp(end_time).isoformat(),
                'execution_time': execution_time,
                'return_code': result.returncode,
                'success': result.returncode == 0
            }
            
            if result.returncode == 0:
                print(f"✓ {description} 执行成功")
                print(f"执行时间: {execution_time:.2f} 秒")
                
                # 显示输出的最后几行
                if result.stdout:
                    stdout_lines = result.stdout.strip().split('\n')
                    if len(stdout_lines) > 5:
                        print("输出摘要 (最后5行):")
                        for line in stdout_lines[-5:]:
                            print(f"  {line}")
                    else:
                        print("输出:")
                        for line in stdout_lines:
                            print(f"  {line}")
                
                log_entry['stdout'] = result.stdout
                log_entry['stderr'] = result.stderr
                
            else:
                print(f"✗ {description} 执行失败")
                print(f"返回代码: {result.returncode}")
                print(f"执行时间: {execution_time:.2f} 秒")
                
                if result.stderr:
                    print("错误信息:")
                    for line in result.stderr.strip().split('\n'):
                        print(f"  {line}")
                
                if result.stdout:
                    print("标准输出:")
                    for line in result.stdout.strip().split('\n'):
                        print(f"  {line}")
                
                log_entry['stdout'] = result.stdout
                log_entry['stderr'] = result.stderr
            
            self.execution_log.append(log_entry)
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            end_time = time.time()
            execution_time = end_time - start_time
            
            print(f"✗ {description} 执行超时 (超过30分钟)")
            print(f"执行时间: {execution_time:.2f} 秒")
            
            log_entry = {
                'script': script_name,
                'description': description,
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.fromtimestamp(end_time).isoformat(),
                'execution_time': execution_time,
                'return_code': -1,
                'success': False,
                'error': 'Timeout (30 minutes)'
            }
            self.execution_log.append(log_entry)
            return False
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            print(f"✗ {description} 执行异常: {e}")
            print(f"执行时间: {execution_time:.2f} 秒")
            
            log_entry = {
                'script': script_name,
                'description': description,
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.fromtimestamp(end_time).isoformat(),
                'execution_time': execution_time,
                'return_code': -2,
                'success': False,
                'error': str(e)
            }
            self.execution_log.append(log_entry)
            return False
    
    def run_all(self):
        """运行所有脚本"""
        print("多方法参数学习流水线开始执行")
        print(f"基础目录: {self.base_dir}")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 检查前提条件
        if not self.check_prerequisites():
            print("前提条件检查失败，程序退出")
            return False
        
        # 记录总体开始时间
        pipeline_start_time = time.time()
        
        # 依次执行所有脚本
        success_count = 0
        total_count = len(self.scripts)
        
        for i, (script_name, description) in enumerate(self.scripts, 1):
            print(f"\n进度: {i}/{total_count}")
            
            success = self.run_script(script_name, description)
            if success:
                success_count += 1
            else:
                print(f"警告: {description} 执行失败，但继续执行后续脚本")
        
        # 记录总体结束时间
        pipeline_end_time = time.time()
        total_execution_time = pipeline_end_time - pipeline_start_time
        
        # 不生成执行报告文件，仅保留控制台输出
        
        # 输出总结
        print(f"\n{'='*60}")
        print("多方法参数学习流水线执行完成")
        print(f"{'='*60}")
        print(f"总执行时间: {total_execution_time:.2f} 秒 ({total_execution_time/60:.1f} 分钟)")
        print(f"成功执行: {success_count}/{total_count} 个脚本")
        print(f"成功率: {success_count/total_count*100:.1f}%")
        
        if success_count == total_count:
            print("✓ 所有脚本执行成功！")
            return True
        else:
            print(f"✗ {total_count - success_count} 个脚本执行失败")
            return False
    
    def generate_execution_report(self, start_time, end_time, success_count, total_count):
        """生成执行报告（已禁用）"""
        # 不再生成报告文件，仅保留方法以避免代码错误
        pass
    
    def check_results(self):
        """检查执行结果"""
        print("\n检查执行结果...")
        
        expected_folders = [
            "01MLE_CPT结果",
            "02Bayesian_CPT结果", 
            "03EM_CPT结果",
            "04SEM_结果",
            "05边级似然增益结果",
            "06参数稳定性结果",
            "可视化"
        ]
        
        existing_folders = []
        missing_folders = []
        
        for folder in expected_folders:
            folder_path = os.path.join(self.base_dir, folder)
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                existing_folders.append(folder)
                # 检查文件夹内容
                files = os.listdir(folder_path)
                print(f"✓ {folder} (包含 {len(files)} 个文件)")
            else:
                missing_folders.append(folder)
                print(f"✗ {folder} (不存在)")
        
        print(f"\n结果文件夹统计:")
        print(f"存在: {len(existing_folders)}/{len(expected_folders)}")
        print(f"缺失: {len(missing_folders)}")
        
        return len(missing_folders) == 0

def main():
    """主函数"""
    print("多方法参数学习统一执行脚本")
    print("=" * 50)
    
    # 创建流水线执行器
    pipeline = ParameterLearningPipeline()
    
    # 运行所有脚本
    success = pipeline.run_all()
    
    # 检查结果
    results_complete = pipeline.check_results()
    
    # 最终状态
    print(f"\n{'='*60}")
    print("最终执行状态")
    print(f"{'='*60}")
    
    if success and results_complete:
        print("✓ 多方法参数学习流水线执行完全成功！")
        print("✓ 所有结果文件夹已生成")
        print("\n可以查看以下结果:")
        print("  - 01MLE_CPT结果/")
        print("  - 02Bayesian_CPT结果/")
        print("  - 03EM_CPT结果/")
        print("  - 04SEM_结果/")
        print("  - 05边级似然增益结果/")
        print("  - 06参数稳定性结果/")
        print("  - 可视化/")
    else:
        print("✗ 流水线执行存在问题")
        if not success:
            print("  - 部分脚本执行失败")
        if not results_complete:
            print("  - 部分结果文件夹缺失")
        print("\n请查看控制台输出了解详细信息")

if __name__ == "__main__":
    main()