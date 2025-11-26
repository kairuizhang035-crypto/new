import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# 更强制性的中文字体配置
import matplotlib
import matplotlib.font_manager as fm

# 清除matplotlib字体缓存
try:
    fm._rebuild()
except:
    pass

# 设置中文字体 - 更强制性的配置
plt.rcParams['font.family'] = ['WenQuanYi Micro Hei',  'DejaVu Sans']
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei',  'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class BayesianMediationAnalysis:
    """
    贝叶斯中介分析类
    基于完整中介路径结果.txt中的路径进行贝叶斯中介效应分析
    """
    
    def __init__(self, data_path, mediation_paths_file=None, max_paths=None):
        self.data_path = data_path
        
        # 获取脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 如果未指定中介路径文件，使用脚本目录下的文件
        if mediation_paths_file is None:
            self.mediation_paths_file = os.path.join(script_dir, '01中介路径分析结果', '完整中介路径结果.txt')
        else:
            self.mediation_paths_file = mediation_paths_file
            
        self.max_paths = max_paths  # 新增：限制分析的路径数量
        self.data = None
        self.mediation_paths = self.load_mediation_paths()
        self.results = {}
        
        # 设置输出目录为脚本所在目录下的02贝叶斯中介分析结果文件夹
        self.output_dir = os.path.join(script_dir, '02贝叶斯中介分析结果')
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_mediation_paths(self):
        """
        从完整中介路径结果.txt文件中加载中介路径
        
        Returns:
        --------
        paths : list, 中介路径列表
        """
        paths = []
        
        try:
            with open(self.mediation_paths_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            current_path = {}
            path_id = 0
            
            for line in lines:
                line = line.strip()
                
                # 跳过注释行和空行
                if line.startswith('#') or not line:
                    continue
                
                # 检测路径编号行
                if line.endswith('. 中介路径:'):
                    if current_path:  # 保存前一个路径
                        paths.append(current_path)
                        # 检查是否达到路径数量限制
                        if self.max_paths and len(paths) >= self.max_paths:
                            break
                    
                    path_id += 1
                    current_path = {'id': path_id}
                
                # 解析间接路径
                elif line.startswith('间接路径:'):
                    indirect_path = line.replace('间接路径:', '').strip()
                    # 解析 X → M → Y 格式
                    parts = indirect_path.split(' → ')
                    if len(parts) == 3:
                        current_path['X'] = parts[0].strip()
                        current_path['M'] = parts[1].strip()
                        current_path['Y'] = parts[2].strip()
                        current_path['description'] = indirect_path
                
                # 解析中介变量（用于验证）
                elif line.startswith('中介变量:'):
                    mediator = line.replace('中介变量:', '').strip()
                    # 验证中介变量是否一致
                    if 'M' in current_path and current_path['M'] != mediator:
                        print(f"警告：路径{path_id}中介变量不一致：{current_path['M']} vs {mediator}")
            
            # 添加最后一个路径（如果未达到限制）
            if current_path and (not self.max_paths or len(paths) < self.max_paths):
                paths.append(current_path)
            
            # 显示加载信息
            if self.max_paths:
                print(f"限制分析路径数量为 {self.max_paths} 条")
                print(f"实际加载了 {len(paths)} 条中介路径")
            else:
                print(f"加载了所有 {len(paths)} 条中介路径")
            
            # 显示前5条路径作为验证
            print("\n选择的路径预览：")
            for i, path in enumerate(paths[:5], 1):
                print(f"{i}. {path['description']}")
            
            if len(paths) > 5:
                print(f"... 还有 {len(paths) - 5} 条路径")
            
            return paths
            
        except Exception as e:
            print(f"加载中介路径文件失败: {str(e)}")
            print("无法加载中介路径，程序将退出")
            raise FileNotFoundError(f"无法加载中介路径文件: {self.mediation_paths_file}")

    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        self.data = pd.read_csv(self.data_path)
        print(f"数据加载完成，共{len(self.data)}行，{len(self.data.columns)}列")
        return self.data
    
    def preprocess_data(self):
        """数据预处理"""
        print("正在进行数据预处理...")
        
        # 检查中介路径变量是否存在
        all_variables = set()
        for path in self.mediation_paths:
            all_variables.update([path['X'], path['M'], path['Y']])
        
        missing_vars = [var for var in all_variables if var not in self.data.columns]
        if missing_vars:
            print(f"警告：以下变量在数据中不存在：{missing_vars}")
        
        # 标准化数据
        for var in all_variables:
            if var in self.data.columns:
                self.data[f'{var}_std'] = (self.data[var] - self.data[var].mean()) / self.data[var].std()
        
        print("数据预处理完成")
    
    def define_bayesian_mediation_model(self, X, M, Y, prior_strength=1.0):
        """
        定义贝叶斯中介模型
        
        Parameters:
        -----------
        X : str, 自变量名称
        M : str, 中介变量名称  
        Y : str, 因变量名称
        prior_strength : float, 先验强度
        
        Returns:
        --------
        model : pymc.Model, 贝叶斯模型
        """
        
        # 获取标准化数据
        X_data = self.data[f'{X}_std'].values
        M_data = self.data[f'{M}_std'].values
        Y_data = self.data[f'{Y}_std'].values
        
        with pm.Model() as model:
            # 先验分布设置
            # 路径系数的先验分布（正态分布）
            alpha = pm.Normal('alpha', mu=0, sigma=prior_strength)  # X对M的效应
            beta = pm.Normal('beta', mu=0, sigma=prior_strength)    # M对Y的效应（控制X）
            tau_prime = pm.Normal('tau_prime', mu=0, sigma=prior_strength)  # X对Y的直接效应
            
            # 截距项
            intercept_M = pm.Normal('intercept_M', mu=0, sigma=1)
            intercept_Y = pm.Normal('intercept_Y', mu=0, sigma=1)
            
            # 误差项的精度（逆方差）
            sigma_M = pm.HalfNormal('sigma_M', sigma=1)
            sigma_Y = pm.HalfNormal('sigma_Y', sigma=1)
            
            # 中介模型方程
            # M = intercept_M + alpha * X + error_M
            mu_M = intercept_M + alpha * X_data
            M_obs = pm.Normal('M_obs', mu=mu_M, sigma=sigma_M, observed=M_data)
            
            # Y = intercept_Y + tau_prime * X + beta * M + error_Y
            mu_Y = intercept_Y + tau_prime * X_data + beta * M_data
            Y_obs = pm.Normal('Y_obs', mu=mu_Y, sigma=sigma_Y, observed=Y_data)
            
            # 计算中介效应
            indirect_effect = pm.Deterministic('indirect_effect', alpha * beta)
            total_effect = pm.Deterministic('total_effect', tau_prime + indirect_effect)
            
            # 中介效应比例
            mediation_ratio = pm.Deterministic('mediation_ratio', 
                                             pm.math.switch(pm.math.abs(total_effect) > 1e-6,
                                                          indirect_effect / total_effect, 0))
        
        return model
    
    def run_bayesian_inference(self, model, draws=500, tune=500, chains=4):
        """
        运行贝叶斯推断
        
        Parameters:
        -----------
        model : pymc.Model, 贝叶斯模型
        draws : int, MCMC采样数
        tune : int, 调优步数
        chains : int, 链数
        
        Returns:
        --------
        trace : arviz.InferenceData, MCMC采样结果
        """
        
        with model:
            # 使用NUTS采样器
            trace = pm.sample(draws=draws, tune=tune, chains=4, 
                            target_accept=0.95, random_seed=42)
        
        return trace
    
    def analyze_single_path(self, path_info, prior_strength=1.0):
        """
        分析单个中介路径
        
        Parameters:
        -----------
        path_info : dict, 路径信息
        prior_strength : float, 先验强度
        
        Returns:
        --------
        results : dict, 分析结果
        """
        
        print(f"\n正在分析路径 {path_info['id']}: {path_info['description']}")
        
        X, M, Y = path_info['X'], path_info['M'], path_info['Y']
        
        # 检查变量是否存在
        required_vars = [f'{X}_std', f'{M}_std', f'{Y}_std']
        missing_vars = [var for var in required_vars if var not in self.data.columns]
        
        if missing_vars:
            print(f"跳过路径 {path_info['id']}：缺少变量 {missing_vars}")
            return None
        
        try:
            # 定义模型
            model = self.define_bayesian_mediation_model(X, M, Y, prior_strength)
            
            # 运行推断
            trace = self.run_bayesian_inference(model)
            
            # 提取结果
            posterior = trace.posterior
            
            results = {
                'path_id': path_info['id'],
                'description': path_info['description'],
                'X': X, 'M': M, 'Y': Y,
                'trace': trace,
                'model': model,
                'posterior_summary': az.summary(trace),
                'indirect_effect': {
                    'mean': float(posterior['indirect_effect'].mean()),
                    'std': float(posterior['indirect_effect'].std()),
                    'hdi_95': az.hdi(trace, var_names=['indirect_effect'])['indirect_effect'].values.tolist()
                },
                'direct_effect': {
                    'mean': float(posterior['tau_prime'].mean()),
                    'std': float(posterior['tau_prime'].std()),
                    'hdi_95': az.hdi(trace, var_names=['tau_prime'])['tau_prime'].values.tolist()
                },
                'total_effect': {
                    'mean': float(posterior['total_effect'].mean()),
                    'std': float(posterior['total_effect'].std()),
                    'hdi_95': az.hdi(trace, var_names=['total_effect'])['total_effect'].values.tolist()
                },
                'mediation_ratio': {
                    'mean': float(posterior['mediation_ratio'].mean()),
                    'std': float(posterior['mediation_ratio'].std()),
                    'hdi_95': az.hdi(trace, var_names=['mediation_ratio'])['mediation_ratio'].values.tolist()
                }
            }
            
            # 计算贝叶斯因子（间接效应显著性）
            indirect_samples = posterior['indirect_effect'].values.flatten()
            prob_positive = np.mean(indirect_samples > 0)
            prob_negative = np.mean(indirect_samples < 0)
            prob_significant = max(prob_positive, prob_negative)
            
            results['bayesian_significance'] = {
                'prob_positive': prob_positive,
                'prob_negative': prob_negative,
                'prob_significant': prob_significant,
                'is_significant': prob_significant > 0.95
            }
            
            print(f"路径 {path_info['id']} 分析完成")
            print(f"间接效应均值: {results['indirect_effect']['mean']:.4f}")
            print(f"显著性概率: {prob_significant:.4f}")
            
            return results
            
        except Exception as e:
            print(f"路径 {path_info['id']} 分析失败: {str(e)}")
            return None
    
    def run_full_analysis(self, prior_strength=1.0):
        """
        运行完整的贝叶斯中介分析
        
        Parameters:
        -----------
        prior_strength : float, 先验强度
        """
        
        print("开始贝叶斯中介分析...")
        print(f"共有 {len(self.mediation_paths)} 条中介路径需要分析")
        
        # 加载和预处理数据
        self.load_data()
        self.preprocess_data()
        
        # 分析每条路径
        for path_info in self.mediation_paths:
            result = self.analyze_single_path(path_info, prior_strength)
            if result:
                self.results[path_info['id']] = result
        
        print(f"\n分析完成！成功分析了 {len(self.results)} 条路径")
    
    def generate_summary_report(self):
        """
        生成汇总报告
        """
        
        if not self.results:
            print("没有分析结果可供汇总")
            return
        
        print("\n" + "="*80)
        print("贝叶斯中介分析汇总报告")
        print("="*80)
        
        summary_data = []
        
        for path_id, result in self.results.items():
            summary_data.append({
                '路径ID': path_id,
                '路径描述': result['description'],
                '间接效应均值': f"{result['indirect_effect']['mean']:.4f}",
                '间接效应95%HDI': f"[{result['indirect_effect']['hdi_95'][0]:.4f}, {result['indirect_effect']['hdi_95'][1]:.4f}]",
                '直接效应均值': f"{result['direct_effect']['mean']:.4f}",
                '总效应均值': f"{result['total_effect']['mean']:.4f}",
                '中介比例': f"{result['mediation_ratio']['mean']:.4f}",
                '显著性概率': f"{result['bayesian_significance']['prob_significant']:.4f}",
                '是否显著': '是' if result['bayesian_significance']['is_significant'] else '否'
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # 保存汇总结果
        summary_df.to_csv(os.path.join(self.output_dir, '贝叶斯中介分析汇总.csv'), 
                         index=False, encoding='utf-8-sig')
        
        # 生成详细的txt报告
        self.generate_detailed_txt_report()
        
        return summary_df
    
    def generate_detailed_txt_report(self):
        """
        生成详细的txt分析报告
        """
        if not self.results:
            print("没有分析结果可供生成详细报告")
            return
        
        report_path = os.path.join(self.output_dir, '详细分析报告.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("贝叶斯中介分析详细报告\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"分析路径总数: {len(self.results)}\n")
            f.write(f"数据文件: {self.data_path}\n\n")
            
            # 整体分析摘要
            f.write("="*60 + "\n")
            f.write("整体分析摘要\n")
            f.write("="*60 + "\n")
            
            significant_paths = sum(1 for result in self.results.values() 
                                  if result['bayesian_significance']['is_significant'])
            f.write(f"显著中介路径数量: {significant_paths}/{len(self.results)}\n")
            f.write(f"显著路径比例: {significant_paths/len(self.results)*100:.1f}%\n\n")
            
            # 各路径详细分析
            for path_id, result in self.results.items():
                f.write("="*60 + "\n")
                f.write(f"路径 {path_id} 详细分析\n")
                f.write("="*60 + "\n")
                
                f.write(f"路径描述: {result['description']}\n\n")
                
                # 效应分析
                f.write("效应分析:\n")
                f.write("-" * 40 + "\n")
                
                # 间接效应（中介效应）
                indirect_mean = result['indirect_effect']['mean']
                indirect_hdi = result['indirect_effect']['hdi_95']
                f.write(f"间接效应（中介效应）:\n")
                f.write(f"  均值: {indirect_mean:.4f}\n")
                f.write(f"  95% HDI: [{indirect_hdi[0]:.4f}, {indirect_hdi[1]:.4f}]\n")
                f.write(f"  标准差: {result['indirect_effect']['std']:.4f}\n")
                
                # 直接效应
                direct_mean = result['direct_effect']['mean']
                direct_hdi = result['direct_effect']['hdi_95']
                f.write(f"\n直接效应:\n")
                f.write(f"  均值: {direct_mean:.4f}\n")
                f.write(f"  95% HDI: [{direct_hdi[0]:.4f}, {direct_hdi[1]:.4f}]\n")
                f.write(f"  标准差: {result['direct_effect']['std']:.4f}\n")
                
                # 总效应
                total_mean = result['total_effect']['mean']
                total_hdi = result['total_effect']['hdi_95']
                f.write(f"\n总效应:\n")
                f.write(f"  均值: {total_mean:.4f}\n")
                f.write(f"  95% HDI: [{total_hdi[0]:.4f}, {total_hdi[1]:.4f}]\n")
                f.write(f"  标准差: {result['total_effect']['std']:.4f}\n")
                
                # 中介比例
                mediation_ratio = result['mediation_ratio']['mean']
                f.write(f"\n中介比例: {mediation_ratio:.4f} ({mediation_ratio*100:.1f}%)\n")
                
                # 影响类型分析
                f.write("\n影响类型分析:\n")
                f.write("-" * 40 + "\n")
                
                # 判断主要影响类型
                abs_indirect = abs(indirect_mean)
                abs_direct = abs(direct_mean)
                
                if abs_indirect > abs_direct:
                    primary_effect = "间接影响（中介效应）"
                    effect_strength = "强"
                elif abs_direct > abs_indirect:
                    primary_effect = "直接影响"
                    effect_strength = "强"
                else:
                    primary_effect = "直接和间接影响并重"
                    effect_strength = "中等"
                
                f.write(f"主要影响类型: {primary_effect}\n")
                f.write(f"影响强度: {effect_strength}\n")
                
                # 效应方向分析
                if indirect_mean > 0:
                    indirect_direction = "正向中介"
                elif indirect_mean < 0:
                    indirect_direction = "负向中介"
                else:
                    indirect_direction = "无中介效应"
                
                if direct_mean > 0:
                    direct_direction = "正向直接效应"
                elif direct_mean < 0:
                    direct_direction = "负向直接效应"
                else:
                    direct_direction = "无直接效应"
                
                f.write(f"间接效应方向: {indirect_direction}\n")
                f.write(f"直接效应方向: {direct_direction}\n")
                
                # 中介类型判断
                if abs(mediation_ratio) > 0.8:
                    mediation_type = "完全中介"
                elif abs(mediation_ratio) > 0.2:
                    mediation_type = "部分中介"
                else:
                    mediation_type = "弱中介"
                
                f.write(f"中介类型: {mediation_type}\n")
                
                # 显著性分析
                f.write("\n显著性分析:\n")
                f.write("-" * 40 + "\n")
                
                sig_prob = result['bayesian_significance']['prob_significant']
                is_significant = result['bayesian_significance']['is_significant']
                prob_positive = result['bayesian_significance']['prob_positive']
                prob_negative = result['bayesian_significance']['prob_negative']
                
                f.write(f"显著性概率: {sig_prob:.4f}\n")
                f.write(f"是否显著: {'是' if is_significant else '否'}\n")
                f.write(f"正向效应概率: {prob_positive:.4f}\n")
                f.write(f"负向效应概率: {prob_negative:.4f}\n")
                
                # 结论
                f.write("\n结论:\n")
                f.write("-" * 40 + "\n")
                
                if is_significant:
                    if abs_indirect > abs_direct:
                        conclusion = f"该路径存在显著的{indirect_direction}，主要通过间接影响发挥作用。"
                    elif abs_direct > abs_indirect:
                        conclusion = f"该路径存在显著的{direct_direction}，主要通过直接影响发挥作用。"
                    else:
                        conclusion = f"该路径同时存在显著的直接和间接影响。"
                else:
                    conclusion = "该路径的中介效应不显著。"
                
                f.write(f"{conclusion}\n")
                f.write(f"中介效应强度为{mediation_type}，中介比例为{mediation_ratio*100:.1f}%。\n\n")
            
            # 总体结论
            f.write("="*80 + "\n")
            f.write("总体结论\n")
            f.write("="*80 + "\n")
            
            f.write(f"本次分析共检验了{len(self.results)}条中介路径，其中{significant_paths}条路径显示出显著的中介效应。\n")
            
            if significant_paths > 0:
                f.write("\n显著路径特征：\n")
                for path_id, result in self.results.items():
                    if result['bayesian_significance']['is_significant']:
                        indirect_mean = result['indirect_effect']['mean']
                        mediation_ratio = result['mediation_ratio']['mean']
                        f.write(f"- 路径{path_id}: 间接效应={indirect_mean:.4f}, 中介比例={mediation_ratio*100:.1f}%\n")
            
            f.write("\n分析完成。详细的图表结果请查看同目录下的PNG文件。\n")
        
        print(f"\n详细分析报告已保存至: {report_path}")
    
    def plot_results(self):
        """
        绘制分析结果的综合图表（简化合理版）
        """
        
        if not self.results:
            print("没有结果可供绘制")
            return
        
        import os
        os.makedirs('/home/zkr/yinguo/贝叶斯中介分析结果', exist_ok=True)
        
        # 保留的合理图表
        # 1. HDI区间比较图
        self._plot_hdi_comparison()
        
        # 2. 路径系数网络图
        self._plot_path_network()
        
        # 3. 贝叶斯统计热图
        self._plot_bayesian_heatmap()
        
        # 4. 间接效应森林图
        self._plot_forest_plot()
        
        # 5. 效应分布箱线图（简化版）
        self._plot_effect_boxplots_simple()
        
    def _plot_comprehensive_effects(self):
        """效应分析综合图（优化版）"""
        path_ids = list(self.results.keys())
        
        # 根据路径数量动态调整显示策略
        if len(path_ids) <= 6:
            # 少量路径：使用原有布局
            self._plot_comprehensive_effects_single_page(path_ids)
        else:
            # 大量路径：分页显示
            paths_per_page = 6
            num_pages = (len(path_ids) + paths_per_page - 1) // paths_per_page
            
            for page in range(num_pages):
                start_idx = page * paths_per_page
                end_idx = min((page + 1) * paths_per_page, len(path_ids))
                page_path_ids = path_ids[start_idx:end_idx]
                
                filename = f'/home/zkr/yinguo/贝叶斯中介分析结果/效应分析综合图_第{page+1}页.png'
                self._plot_comprehensive_effects_single_page(page_path_ids, page+1, num_pages, filename)
    
    def _plot_comprehensive_effects_single_page(self, path_ids, page_num=None, total_pages=None, filename=None):
        """绘制单页效应分析综合图"""
        if filename is None:
            filename = '/home/zkr/yinguo/贝叶斯中介分析结果/效应分析综合图.png'
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        title_suffix = f' (第{page_num}/{total_pages}页)' if page_num else ''
        
        # 间接效应分布
        ax1 = axes[0, 0]
        for path_id in path_ids:
            result = self.results[path_id]
            trace = result['trace']
            indirect_samples = trace.posterior['indirect_effect'].values.flatten()
            ax1.hist(indirect_samples, alpha=0.6, label=f'路径{path_id}', bins=30)
        ax1.set_xlabel('间接效应')
        ax1.set_ylabel('频率')
        ax1.set_title(f'间接效应后验分布{title_suffix}')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 直接效应分布
        ax2 = axes[0, 1]
        for path_id in path_ids:
            result = self.results[path_id]
            trace = result['trace']
            direct_samples = trace.posterior['tau_prime'].values.flatten()
            ax2.hist(direct_samples, alpha=0.6, label=f'路径{path_id}', bins=30)
        ax2.set_xlabel('直接效应')
        ax2.set_ylabel('频率')
        ax2.set_title(f'直接效应后验分布{title_suffix}')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 中介比例
        ax3 = axes[1, 0]
        mediation_ratios = [self.results[pid]['mediation_ratio']['mean'] for pid in path_ids]
        bars = ax3.bar([f'路径{pid}' for pid in path_ids], mediation_ratios, 
                      color=plt.cm.Set3(np.linspace(0, 1, len(path_ids))))
        ax3.set_ylabel('中介比例')
        ax3.set_title(f'各路径中介比例{title_suffix}')
        ax3.set_ylim(-1, 1)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.grid(True, alpha=0.3)
        
        # 旋转x轴标签以避免重叠
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, ratio in zip(bars, mediation_ratios):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.03,
                    f'{ratio:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
        
        # 显著性概率
        ax4 = axes[1, 1]
        sig_probs = [self.results[pid]['bayesian_significance']['prob_significant'] for pid in path_ids]
        bars = ax4.bar([f'路径{pid}' for pid in path_ids], sig_probs,
                      color=plt.cm.viridis(np.linspace(0, 1, len(path_ids))))
        ax4.set_ylabel('显著性概率')
        ax4.set_title(f'各路径显著性概率{title_suffix}')
        ax4.set_ylim(0, 1)
        ax4.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='显著性阈值(0.95)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 旋转x轴标签
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, prob in zip(bars, sig_probs):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_forest_plot(self):
        """间接效应森林图"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        path_ids = list(self.results.keys())
        y_positions = range(len(path_ids))
        
        means = []
        lower_bounds = []
        upper_bounds = []
        
        for path_id in path_ids:
            result = self.results[path_id]
            means.append(result['indirect_effect']['mean'])
            lower_bounds.append(result['indirect_effect']['hdi_95'][0])
            upper_bounds.append(result['indirect_effect']['hdi_95'][1])
        
        # 绘制置信区间
        for i, (mean, lower, upper) in enumerate(zip(means, lower_bounds, upper_bounds)):
            ax.plot([lower, upper], [i, i], 'b-', linewidth=2)
            ax.plot([lower, lower], [i-0.1, i+0.1], 'b-', linewidth=2)
            ax.plot([upper, upper], [i-0.1, i+0.1], 'b-', linewidth=2)
            ax.plot(mean, i, 'ro', markersize=8)
            
            # 添加数值标签
            ax.text(mean + 0.01, i, f'{mean:.4f}', va='center', fontsize=10)
        
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f'路径{pid}' for pid in path_ids])
        ax.set_xlabel('间接效应')
        ax.set_title('间接效应森林图 (95% HDI)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/zkr/yinguo/贝叶斯中介分析结果/间接效应森林图.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_effect_comparison(self):
        """效应大小比较图（优化版）"""
        path_ids = list(self.results.keys())
        
        # 根据路径数量动态调整布局
        if len(path_ids) <= 8:
            self._plot_effect_comparison_single_page(path_ids)
        else:
            # 分页显示
            paths_per_page = 8
            num_pages = (len(path_ids) + paths_per_page - 1) // paths_per_page
            
            for page in range(num_pages):
                start_idx = page * paths_per_page
                end_idx = min((page + 1) * paths_per_page, len(path_ids))
                page_path_ids = path_ids[start_idx:end_idx]
                
                filename = f'/home/zkr/yinguo/贝叶斯中介分析结果/效应大小比较图_第{page+1}页.png'
                self._plot_effect_comparison_single_page(page_path_ids, page+1, num_pages, filename)
    
    def _plot_effect_comparison_single_page(self, path_ids, page_num=None, total_pages=None, filename=None):
        """绘制单页效应大小比较图"""
        if filename is None:
            filename = '/home/zkr/yinguo/贝叶斯中介分析结果/效应大小比较图.png'
        
        # 动态调整布局和尺寸
        n_paths = len(path_ids)
        if n_paths <= 3:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            pie_axes = [ax2]  # 只使用右侧子图显示第一个路径的饼图
        else:
            fig = plt.figure(figsize=(20, 12))
            ax1 = plt.subplot(2, 4, (1, 4))  # 占据上方整行
            # 下方显示前4个路径的饼图
            pie_axes = [plt.subplot(2, 4, i+5) for i in range(min(4, n_paths))]
        
        title_suffix = f' (第{page_num}/{total_pages}页)' if page_num else ''
        
        indirect_effects = [self.results[pid]['indirect_effect']['mean'] for pid in path_ids]
        direct_effects = [self.results[pid]['direct_effect']['mean'] for pid in path_ids]
        total_effects = [self.results[pid]['total_effect']['mean'] for pid in path_ids]
        
        x = np.arange(len(path_ids))
        width = 0.25
        
        # 效应大小对比
        bars1 = ax1.bar(x - width, indirect_effects, width, label='间接效应', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x, direct_effects, width, label='直接效应', alpha=0.8, color='lightcoral')
        bars3 = ax1.bar(x + width, total_effects, width, label='总效应', alpha=0.8, color='lightgreen')
        
        ax1.set_xlabel('路径', fontsize=12)
        ax1.set_ylabel('效应大小', fontsize=12)
        ax1.set_title(f'各路径效应大小对比{title_suffix}', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'路径{pid}' for pid in path_ids], rotation=45)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 添加数值标签
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.01,
                        f'{height:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        # 效应比例饼图
        for i, path_id in enumerate(path_ids):
            if i >= len(pie_axes):  # 不超过可用的饼图位置
                break
                
            ax_pie = pie_axes[i]
            
            indirect = abs(indirect_effects[i])
            direct = abs(direct_effects[i])
            
            if indirect + direct > 0:
                sizes = [indirect, direct]
                labels = ['间接效应', '直接效应']
                colors = ['skyblue', 'lightcoral']
                
                wedges, texts, autotexts = ax_pie.pie(sizes, labels=labels, colors=colors, 
                                                     autopct='%1.1f%%', startangle=90, 
                                                     textprops={'fontsize': 10})
                ax_pie.set_title(f'路径{path_id}效应构成', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_posterior_distributions(self):
        """后验分布密度图（优化版）"""
        path_ids = list(self.results.keys())
        
        # 每页最多显示6个路径
        paths_per_page = 6
        num_pages = (len(path_ids) + paths_per_page - 1) // paths_per_page
        
        for page in range(num_pages):
            start_idx = page * paths_per_page
            end_idx = min((page + 1) * paths_per_page, len(path_ids))
            page_path_ids = path_ids[start_idx:end_idx]
            
            filename = f'/home/zkr/yinguo/贝叶斯中介分析结果/后验分布密度图_第{page+1}页.png' if num_pages > 1 else '/home/zkr/yinguo/贝叶斯中介分析结果/后验分布密度图.png'
            self._plot_posterior_single_page(page_path_ids, page+1, num_pages, filename)
    
    def _plot_posterior_single_page(self, path_ids, page_num=None, total_pages=None, filename=None):
        """绘制单页后验分布图"""
        if filename is None:
            filename = '/home/zkr/yinguo/贝叶斯中介分析结果/后验分布密度图.png'
        
        # 动态调整子图布局
        n_paths = len(path_ids)
        if n_paths <= 2:
            rows, cols = 1, n_paths
            figsize = (9 * cols, 6)
        elif n_paths <= 4:
            rows, cols = 2, 2
            figsize = (18, 12)
        else:
            rows, cols = 2, 3
            figsize = (18, 12)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_paths == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_paths > 1 else axes
        
        title_suffix = f' (第{page_num}/{total_pages}页)' if page_num and total_pages > 1 else ''
        
        for i, path_id in enumerate(path_ids):
            ax = axes[i]
            result = self.results[path_id]
            trace = result['trace']
            
            # 获取后验样本
            indirect_samples = trace.posterior['indirect_effect'].values.flatten()
            direct_samples = trace.posterior['tau_prime'].values.flatten()
            
            # 绘制密度图
            ax.hist(indirect_samples, bins=50, alpha=0.7, label='间接效应', density=True, color='skyblue')
            ax.hist(direct_samples, bins=50, alpha=0.7, label='直接效应', density=True, color='lightcoral')
            
            ax.axvline(result['indirect_effect']['mean'], color='blue', linestyle='--', alpha=0.8, linewidth=2)
            ax.axvline(result['direct_effect']['mean'], color='red', linestyle='--', alpha=0.8, linewidth=2)
            
            ax.set_title(f'路径{path_id}后验分布', fontsize=14, fontweight='bold')
            ax.set_xlabel('效应大小')
            ax.set_ylabel('密度')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(path_ids), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'各路径后验分布密度图{title_suffix}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_mediation_radar(self):
        """中介比例雷达图（优化版）"""
        path_ids = list(self.results.keys())
        
        # 限制显示路径数量，避免雷达图过于拥挤
        max_paths_per_radar = 8
        
        if len(path_ids) <= max_paths_per_radar:
            # 单个雷达图
            self._plot_single_radar(path_ids, '中介比例雷达图')
        else:
            # 分页显示多个雷达图
            num_pages = (len(path_ids) + max_paths_per_radar - 1) // max_paths_per_radar
            
            for page in range(num_pages):
                start_idx = page * max_paths_per_radar
                end_idx = min((page + 1) * max_paths_per_radar, len(path_ids))
                page_path_ids = path_ids[start_idx:end_idx]
                
                title = f'中介比例雷达图 (第{page+1}/{num_pages}页)'
                filename = f'/home/zkr/yinguo/贝叶斯中介分析结果/中介比例雷达图_第{page+1}页.png'
                self._plot_single_radar(page_path_ids, title, filename)
    
    def _plot_single_radar(self, path_ids, title, filename=None):
        """绘制单个雷达图"""
        if filename is None:
            filename = '/home/zkr/yinguo/贝叶斯中介分析结果/中介比例雷达图.png'
        
        # 准备数据
        categories = [f'路径{pid}' for pid in path_ids]
        values = [abs(self.results[pid]['mediation_ratio']['mean']) for pid in path_ids]
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 闭合图形
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, values, 'o-', linewidth=3, color='blue', markersize=8)
        ax.fill(angles, values, alpha=0.25, color='blue')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title(title, size=16, fontweight='bold', pad=30)
        ax.grid(True, alpha=0.7)
        
        # 添加数值标签
        for angle, value, category in zip(angles[:-1], values[:-1], categories):
            ax.text(angle, value + 0.05, f'{value:.3f}', 
                   horizontalalignment='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_bayesian_heatmap(self):
        """贝叶斯统计热图"""
        path_ids = list(self.results.keys())
        
        # 准备数据矩阵
        data_matrix = []
        row_labels = []
        
        for path_id in path_ids:
            result = self.results[path_id]
            row_data = [
                result['indirect_effect']['mean'],
                result['direct_effect']['mean'],
                result['total_effect']['mean'],
                result['mediation_ratio']['mean'],
                result['bayesian_significance']['prob_significant']
            ]
            data_matrix.append(row_data)
            row_labels.append(f'路径{path_id}')
        
        col_labels = ['间接效应', '直接效应', '总效应', '中介比例', '显著性概率']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto')
        
        # 设置标签
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        
        # 添加数值标签
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(j, i, f'{data_matrix[i][j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('贝叶斯中介分析统计热图', fontsize=14, fontweight='bold')
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig('/home/zkr/yinguo/贝叶斯中介分析结果/贝叶斯统计热图.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_path_network(self):
        """路径系数网络图"""
        try:
            import networkx as nx
        except ImportError:
            print("需要安装networkx库来绘制网络图: pip install networkx")
            return
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        G = nx.DiGraph()
        
        # 添加节点和边
        for path_id, result in self.results.items():
            X, M, Y = result['X'], result['M'], result['Y']
            
            # 添加节点
            G.add_node(X, node_type='X')
            G.add_node(M, node_type='M')
            G.add_node(Y, node_type='Y')
            
            # 添加边（权重为效应大小）
            indirect_effect = abs(result['indirect_effect']['mean'])
            direct_effect = abs(result['direct_effect']['mean'])
            
            G.add_edge(X, M, weight=indirect_effect, path_id=path_id, effect_type='X->M')
            G.add_edge(M, Y, weight=indirect_effect, path_id=path_id, effect_type='M->Y')
            G.add_edge(X, Y, weight=direct_effect, path_id=path_id, effect_type='direct')
        
        # 布局
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 绘制节点
        node_colors = {'X': 'lightblue', 'M': 'lightgreen', 'Y': 'lightcoral'}
        for node_type in ['X', 'M', 'Y']:
            nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == node_type]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                                 node_color=node_colors[node_type], 
                                 node_size=1000, alpha=0.8)
        
        # 绘制边
        edges = G.edges(data=True)
        for edge in edges:
            weight = edge[2]['weight']
            nx.draw_networkx_edges(G, pos, edgelist=[(edge[0], edge[1])], 
                                 width=weight*10, alpha=0.6, 
                                 edge_color='gray', arrows=True, arrowsize=20)
        
        # 绘制标签
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        ax.set_title('中介路径网络图', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # 添加图例
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, markersize=10, label=label)
                         for label, color in node_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig('/home/zkr/yinguo/贝叶斯中介分析结果/路径网络图.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_hdi_comparison(self):
        """HDI区间比较图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        path_ids = list(self.results.keys())
        
        # 间接效应HDI比较
        for i, path_id in enumerate(path_ids):
            result = self.results[path_id]
            mean = result['indirect_effect']['mean']
            hdi_lower, hdi_upper = result['indirect_effect']['hdi_95']
            
            ax1.errorbar(i, mean, yerr=[[mean-hdi_lower], [hdi_upper-mean]], 
                        fmt='o', capsize=5, capthick=2, markersize=8)
            ax1.text(i, mean + 0.01, f'{mean:.4f}', ha='center', va='bottom')
        
        ax1.set_xticks(range(len(path_ids)))
        ax1.set_xticklabels([f'路径{pid}' for pid in path_ids], rotation=90, ha='center')
        ax1.set_ylabel('间接效应')
        ax1.set_title('间接效应95% HDI区间比较')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # 直接效应HDI比较
        for i, path_id in enumerate(path_ids):
            result = self.results[path_id]
            mean = result['direct_effect']['mean']
            hdi_lower, hdi_upper = result['direct_effect']['hdi_95']
            
            ax2.errorbar(i, mean, yerr=[[mean-hdi_lower], [hdi_upper-mean]], 
                        fmt='s', capsize=5, capthick=2, markersize=8, color='orange')
            ax2.text(i, mean + 0.01, f'{mean:.4f}', ha='center', va='bottom')
        
        ax2.set_xticks(range(len(path_ids)))
        ax2.set_xticklabels([f'路径{pid}' for pid in path_ids], rotation=90, ha='center')
        ax2.set_ylabel('直接效应')
        ax2.set_title('直接效应95% HDI区间比较')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('/home/zkr/yinguo/贝叶斯中介分析结果/HDI区间比较图.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_effect_boxplots(self):
        """效应强度分布箱线图（优化版）"""
        path_ids = list(self.results.keys())
        
        # 根据路径数量决定显示策略
        max_paths_per_plot = 10
        
        if len(path_ids) <= max_paths_per_plot:
            self._plot_boxplots_single_page(path_ids)
        else:
            # 分页显示
            num_pages = (len(path_ids) + max_paths_per_plot - 1) // max_paths_per_plot
            
            for page in range(num_pages):
                start_idx = page * max_paths_per_plot
                end_idx = min((page + 1) * max_paths_per_plot, len(path_ids))
                page_path_ids = path_ids[start_idx:end_idx]
                
                filename = f'/home/zkr/yinguo/贝叶斯中介分析结果/效应分布箱线图_第{page+1}页.png'
                self._plot_boxplots_single_page(page_path_ids, page+1, num_pages, filename)
    
    def _plot_boxplots_single_page(self, path_ids, page_num=None, total_pages=None, filename=None):
        """绘制单页箱线图"""
        if filename is None:
            filename = '/home/zkr/yinguo/贝叶斯中介分析结果/效应分布箱线图.png'
        
        # 动态调整图表尺寸
        width = max(12, len(path_ids) * 2)
        fig, axes = plt.subplots(1, 3, figsize=(width, 8))
        
        title_suffix = f' (第{page_num}/{total_pages}页)' if page_num else ''
        
        # 收集所有效应数据
        indirect_data = []
        direct_data = []
        total_data = []
        labels = []
        
        for path_id in path_ids:
            result = self.results[path_id]
            trace = result['trace']
            indirect_samples = trace.posterior['indirect_effect'].values.flatten()
            direct_samples = trace.posterior['tau_prime'].values.flatten()
            total_samples = indirect_samples + direct_samples
            
            indirect_data.append(indirect_samples)
            direct_data.append(direct_samples)
            total_data.append(total_samples)
            labels.append(f'路径{path_id}')
        
        # 间接效应箱线图
        bp1 = axes[0].boxplot(indirect_data, labels=labels, patch_artist=True)
        axes[0].set_title(f'间接效应分布{title_suffix}')
        axes[0].set_ylabel('效应大小')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0].tick_params(axis='x', rotation=45)
        
        # 直接效应箱线图
        bp2 = axes[1].boxplot(direct_data, labels=labels, patch_artist=True)
        axes[1].set_title(f'直接效应分布{title_suffix}')
        axes[1].set_ylabel('效应大小')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1].tick_params(axis='x', rotation=45)
        
        # 总效应箱线图
        bp3 = axes[2].boxplot(total_data, labels=labels, patch_artist=True)
        axes[2].set_title(f'总效应分布{title_suffix}')
        axes[2].set_ylabel('效应大小')
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[2].tick_params(axis='x', rotation=45)
        
        # 设置颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(path_ids)))
        for bp, color_set in [(bp1, colors), (bp2, colors), (bp3, colors)]:
            for patch, color in zip(bp['boxes'], color_set):
                patch.set_facecolor(color)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_effect_boxplots_simple(self):
        """效应分布箱线图（优化合理版）"""
        path_ids = list(self.results.keys())
        total_paths = len(path_ids)
        
        # 根据路径数量采用不同的显示策略
        if total_paths <= 20:
            # 少于20条路径：单图显示所有路径
            self._plot_single_boxplot(path_ids)
        elif total_paths <= 50:
            # 20-50条路径：分为2-3个子图
            self._plot_multi_panel_boxplot(path_ids)
        else:
            # 超过50条路径：显示最重要的路径 + 统计摘要
            self._plot_selective_boxplot(path_ids)
    
    def _plot_single_boxplot(self, path_ids):
        """单图显示所有路径的箱线图"""
        # 动态调整图表尺寸
        width = max(15, len(path_ids) * 0.6)
        height = max(8, 10)
        fig, ax = plt.subplots(1, 1, figsize=(width, height))
        
        # 收集间接效应数据
        indirect_data = []
        labels = []
        colors = []
        
        for path_id in path_ids:
            result = self.results[path_id]
            trace = result['trace']
            indirect_samples = trace.posterior['indirect_effect'].values.flatten()
            indirect_data.append(indirect_samples)
            labels.append(f'路径{path_id}')
            
            # 根据显著性设置颜色
            sig_prob = result['bayesian_significance']['prob_significant']
            if sig_prob > 0.95:
                colors.append('#2E86AB')  # 显著 - 蓝色
            elif sig_prob > 0.8:
                colors.append('#F18F01')  # 边缘显著 - 橙色
            else:
                colors.append('#A23B72')  # 不显著 - 红色
        
        # 绘制箱线图
        bp = ax.boxplot(indirect_data, labels=labels, patch_artist=True)
        
        # 设置颜色
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(f'间接效应分布箱线图 (共{len(path_ids)}条路径)', fontsize=14, fontweight='bold')
        ax.set_ylabel('间接效应大小')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='零效应线')
        
        # 优化标签显示
        if len(path_ids) > 10:
            ax.tick_params(axis='x', rotation=90, labelsize=8)
        else:
            ax.tick_params(axis='x', rotation=45)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E86AB', alpha=0.7, label='显著 (p>0.95)'),
            Patch(facecolor='#F18F01', alpha=0.7, label='边缘显著 (p>0.8)'),
            Patch(facecolor='#A23B72', alpha=0.7, label='不显著 (p<0.8)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig('/home/zkr/yinguo/贝叶斯中介分析结果/效应分布箱线图.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_multi_panel_boxplot(self, path_ids):
        """多面板显示中等数量路径的箱线图"""
        total_paths = len(path_ids)
        paths_per_panel = 20
        num_panels = (total_paths + paths_per_panel - 1) // paths_per_panel
        
        fig, axes = plt.subplots(num_panels, 1, figsize=(16, 6 * num_panels))
        if num_panels == 1:
            axes = [axes]
        
        for panel_idx in range(num_panels):
            start_idx = panel_idx * paths_per_panel
            end_idx = min((panel_idx + 1) * paths_per_panel, total_paths)
            panel_path_ids = path_ids[start_idx:end_idx]
            
            ax = axes[panel_idx]
            
            # 收集数据
            indirect_data = []
            labels = []
            colors = []
            
            for path_id in panel_path_ids:
                result = self.results[path_id]
                trace = result['trace']
                indirect_samples = trace.posterior['indirect_effect'].values.flatten()
                indirect_data.append(indirect_samples)
                labels.append(f'路径{path_id}')
                
                # 根据显著性设置颜色
                sig_prob = result['bayesian_significance']['prob_significant']
                if sig_prob > 0.95:
                    colors.append('#2E86AB')
                elif sig_prob > 0.8:
                    colors.append('#F18F01')
                else:
                    colors.append('#A23B72')
            
            # 绘制箱线图
            bp = ax.boxplot(indirect_data, labels=labels, patch_artist=True)
            
            # 设置颜色
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(f'间接效应分布 - 第{panel_idx+1}组 (路径{start_idx+1}-{end_idx})', 
                        fontsize=12, fontweight='bold')
            ax.set_ylabel('间接效应大小')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax.tick_params(axis='x', rotation=45, labelsize=9)
        
        plt.suptitle(f'间接效应分布箱线图 (共{total_paths}条路径)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('/home/zkr/yinguo/贝叶斯中介分析结果/效应分布箱线图.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_selective_boxplot(self, path_ids):
        """选择性显示重要路径的箱线图 + 统计摘要"""
        total_paths = len(path_ids)
        
        # 按显著性概率排序，选择最重要的30条路径
        sorted_paths = sorted(path_ids, 
                            key=lambda x: self.results[x]['bayesian_significance']['prob_significant'], 
                            reverse=True)
        top_paths = sorted_paths[:30]
        
        fig = plt.figure(figsize=(18, 12))
        
        # 上半部分：重要路径的箱线图
        ax1 = plt.subplot(2, 1, 1)
        
        indirect_data = []
        labels = []
        colors = []
        
        for path_id in top_paths:
            result = self.results[path_id]
            trace = result['trace']
            indirect_samples = trace.posterior['indirect_effect'].values.flatten()
            indirect_data.append(indirect_samples)
            labels.append(f'路径{path_id}')
            
            sig_prob = result['bayesian_significance']['prob_significant']
            if sig_prob > 0.95:
                colors.append('#2E86AB')
            elif sig_prob > 0.8:
                colors.append('#F18F01')
            else:
                colors.append('#A23B72')
        
        bp = ax1.boxplot(indirect_data, labels=labels, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title(f'前30条最重要路径的间接效应分布 (共{total_paths}条路径)', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('间接效应大小')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax1.tick_params(axis='x', rotation=90, labelsize=8)
        
        # 下半部分：统计摘要
        ax2 = plt.subplot(2, 1, 2)
        
        # 计算所有路径的统计摘要
        all_effects = []
        sig_counts = {'显著': 0, '边缘显著': 0, '不显著': 0}
        
        for path_id in path_ids:
            result = self.results[path_id]
            all_effects.append(result['indirect_effect']['mean'])
            
            sig_prob = result['bayesian_significance']['prob_significant']
            if sig_prob > 0.95:
                sig_counts['显著'] += 1
            elif sig_prob > 0.8:
                sig_counts['边缘显著'] += 1
            else:
                sig_counts['不显著'] += 1
        
        # 绘制整体分布直方图
        ax2.hist(all_effects, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(all_effects), color='red', linestyle='-', linewidth=2, 
                   label=f'平均值: {np.mean(all_effects):.4f}')
        ax2.axvline(np.median(all_effects), color='orange', linestyle='--', linewidth=2, 
                   label=f'中位数: {np.median(all_effects):.4f}')
        ax2.axvline(0, color='black', linestyle=':', alpha=0.7)
        
        ax2.set_xlabel('间接效应大小')
        ax2.set_ylabel('路径数量')
        ax2.set_title(f'所有{total_paths}条路径的间接效应分布统计', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息文本
        stats_text = f"""统计摘要:
显著路径: {sig_counts['显著']}条 ({sig_counts['显著']/total_paths*100:.1f}%)
边缘显著: {sig_counts['边缘显著']}条 ({sig_counts['边缘显著']/total_paths*100:.1f}%)
不显著: {sig_counts['不显著']}条 ({sig_counts['不显著']/total_paths*100:.1f}%)

效应大小:
最大值: {max(all_effects):.4f}
最小值: {min(all_effects):.4f}
标准差: {np.std(all_effects):.4f}"""
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
        
        plt.tight_layout()
        plt.savefig('/home/zkr/yinguo/贝叶斯中介分析结果/效应分布箱线图.png', dpi=300, bbox_inches='tight')
        plt.show()

def main(max_paths=None, interactive=True):
    """
    主函数：执行贝叶斯中介分析
    
    Parameters:
    -----------
    max_paths : int or None, 要分析的路径数量，None表示分析所有路径
    interactive : bool, 是否启用交互式选择路径数量
    """
    print("=" * 60)
    print("贝叶斯中介分析")
    print("=" * 60)
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置数据文件路径（相对于因果发现根目录）
    data_path = os.path.join(os.path.dirname(script_dir), '01数据预处理', '缩减数据_规格.csv')
    
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        print(f"错误：数据文件不存在 - {data_path}")
        print("请确保数据文件存在于正确位置")
        return None
    
    print(f"使用数据文件: {data_path}")
    
    # 首先创建分析器实例来获取总路径数
    temp_analyzer = BayesianMediationAnalysis(data_path, max_paths=None)
    total_paths = len(temp_analyzer.mediation_paths)
    
    print(f"\n总共发现 {total_paths} 条完整中介路径")
    
    # 如果启用交互式选择
    if interactive:
        print("\n路径分析选项:")
        print("1. 分析所有路径")
        print("2. 分析指定数量的路径")
        print("3. 分析前N条路径（快速测试）")
        
        while True:
            try:
                choice = input("\n请选择分析选项 (1/2/3): ").strip()
                
                if choice == '1':
                    max_paths = None
                    print(f"将分析所有 {total_paths} 条路径")
                    break
                elif choice == '2':
                    while True:
                        try:
                            max_paths = int(input(f"请输入要分析的路径数量 (1-{total_paths}): "))
                            if 1 <= max_paths <= total_paths:
                                print(f"将分析前 {max_paths} 条路径")
                                break
                            else:
                                print(f"请输入1到{total_paths}之间的数字")
                        except ValueError:
                            print("请输入有效的数字")
                    break
                elif choice == '3':
                    # 提供一些快速测试选项
                    test_options = [5, 10, 20, min(50, total_paths)]
                    print("\n快速测试选项:")
                    for i, option in enumerate(test_options, 1):
                        print(f"{i}. 分析前 {option} 条路径")
                    
                    while True:
                        try:
                            test_choice = int(input(f"请选择测试选项 (1-{len(test_options)}): "))
                            if 1 <= test_choice <= len(test_options):
                                max_paths = test_options[test_choice - 1]
                                print(f"将分析前 {max_paths} 条路径")
                                break
                            else:
                                print(f"请输入1到{len(test_options)}之间的数字")
                        except ValueError:
                            print("请输入有效的数字")
                    break
                else:
                    print("请输入1、2或3")
            except KeyboardInterrupt:
                print("\n\n用户中断操作")
                return None
    else:
        # 非交互模式，使用传入的max_paths参数
        if max_paths is None:
            print(f"选择：分析所有 {total_paths} 条路径")
        else:
            print(f"选择：分析前 {max_paths} 条路径")
    
    # 创建最终的分析器实例
    analyzer = BayesianMediationAnalysis(data_path, max_paths=max_paths)
    
    print(f"\n最终将分析 {len(analyzer.mediation_paths)} 条中介路径")
    
    # 显示将要分析的路径预览
    print("\n将要分析的路径预览:")
    for i, path in enumerate(analyzer.mediation_paths[:5], 1):
        print(f"{i}. {path['description']}")
    
    if len(analyzer.mediation_paths) > 5:
        print(f"... 还有 {len(analyzer.mediation_paths) - 5} 条路径")
    
    # 确认开始分析
    if interactive:
        try:
            confirm = input("\n确认开始分析？(y/n): ").lower().strip()
            if confirm not in ['y', 'yes', '是', '确认']:
                print("用户取消分析")
                return None
        except KeyboardInterrupt:
            print("\n\n用户中断操作")
            return None
    
    # 运行完整分析
    analyzer.run_full_analysis(prior_strength=1.0)
    
    # 生成汇总报告
    summary_df = analyzer.generate_summary_report()
    
    # 绘制结果图表
    analyzer.plot_results()
    
    print("\n贝叶斯中介分析完成！")
    print(f"结果已保存到 {analyzer.output_dir} 目录")
    
    return analyzer


if __name__ == "__main__":
    # 运行分析 - 启用交互式选择
    analyzer = main(max_paths=None, interactive=True)
    
    # 暂时注释掉敏感性分析部分
    """
    print("\n" + "="*60)
    print("敏感性分析：不同先验强度的影响")
    print("="*60)
    
    prior_strengths = [0.5, 1.0, 2.0]
    sensitivity_results = {}
    
    for prior_strength in prior_strengths:
        print(f"\n使用先验强度: {prior_strength}")
        
        # 创建新的分析器实例进行敏感性分析
        sens_analyzer = BayesianMediationAnalysis(data_path)
        sens_analyzer.load_data()
        sens_analyzer.preprocess_data()
        
        # 只分析第一条路径作为示例
        result = sens_analyzer.analyze_single_path(sens_analyzer.mediation_paths[0], prior_strength)
        
        if result:
            sensitivity_results[prior_strength] = {
                'indirect_effect_mean': result['indirect_effect']['mean'],
                'significance_prob': result['bayesian_significance']['prob_significant']
            }
    
    # 保存敏感性分析结果
    if sensitivity_results:
        print("\n敏感性分析结果（路径1）:")
        sens_df = pd.DataFrame(sensitivity_results).T
        sens_df.index.name = '先验强度'
        print(sens_df)
        
        sens_df.to_csv('/home/zkr/yinguo/贝叶斯中介分析结果/敏感性分析结果.csv', 
                      encoding='utf-8-sig')
    """