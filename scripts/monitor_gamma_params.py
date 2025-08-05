#!/usr/bin/env python3
"""
TRINUE 双向交叉注意力门控参数监控脚本
=====================================

用于监控和记录训练过程中的可学习门控参数 γ_d2r 和 γ_r2d 的变化情况。


"""

import torch
import torch.nn as nn
import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

class GammaParameterMonitor:
    """双向交叉注意力门控参数监控器"""
    
    def __init__(self, log_dir: str = None, save_plots: bool = True):
        """
        初始化监控器
        
        Args:
            log_dir: 日志保存目录
            save_plots: 是否保存参数变化图表
        """
        self.log_dir = log_dir or "./gamma_logs"
        self.save_plots = save_plots
        
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 参数历史记录
        self.gamma_history = {
            'd2r': [],  # Depth→RGB 门控参数历史
            'r2d': []   # RGB→Depth 门控参数历史
        }
        
        # 步数记录
        self.step_history = []
        
        # 设置日志
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """设置专用的gamma参数日志记录器"""
        logger = logging.getLogger('gamma_monitor')
        logger.setLevel(logging.INFO)
        
        # 避免重复添加handler
        if not logger.handlers:
            # 文件handler
            log_file = os.path.join(self.log_dir, f"gamma_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # 控制台handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 格式化
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
        return logger
    
    def extract_gamma_params(self, model: nn.Module) -> Tuple[List[float], List[float]]:
        """
        从模型中提取门控参数
        
        Args:
            model: TRINUE模型实例
            
        Returns:
            (d2r_gammas, r2d_gammas): 两个方向的门控参数列表
        """
        # 处理DataParallel或DistributedDataParallel包装的模型
        if hasattr(model, 'module'):
            actual_model = model.module
        else:
            actual_model = model
            
        d2r_gammas = []
        r2d_gammas = []
        
        # 检查是否有encoder和门控参数
        if hasattr(actual_model, 'encoder'):
            encoder = actual_model.encoder
            
            if hasattr(encoder, 'd2r_gamma') and hasattr(encoder, 'r2d_gamma'):
                # 提取所有层级的门控参数
                for i in range(len(encoder.d2r_gamma)):
                    d2r_val = encoder.d2r_gamma[i].item()
                    r2d_val = encoder.r2d_gamma[i].item()
                    
                    d2r_gammas.append(d2r_val)
                    r2d_gammas.append(r2d_val)
            else:
                self.logger.warning("模型中未找到门控参数 d2r_gamma 或 r2d_gamma")
        else:
            self.logger.warning("模型中未找到 encoder 属性")
            
        return d2r_gammas, r2d_gammas
    
    def log_gamma_params(self, model: nn.Module, step: int, epoch: int = None):
        """
        记录当前的门控参数值
        
        Args:
            model: TRINUE模型实例
            step: 当前训练步数
            epoch: 当前训练轮次（可选）
        """
        d2r_gammas, r2d_gammas = self.extract_gamma_params(model)
        
        if not d2r_gammas or not r2d_gammas:
            self.logger.warning(f"Step {step}: 未能提取到门控参数")
            return
            
        # 记录到历史
        self.step_history.append(step)
        self.gamma_history['d2r'].append(d2r_gammas.copy())
        self.gamma_history['r2d'].append(r2d_gammas.copy())
        
        # 格式化输出
        epoch_str = f"Epoch {epoch}, " if epoch is not None else ""
        
        # 详细日志
        self.logger.info(f"{epoch_str}Step {step}: 门控参数更新")
        
        for i, (d2r, r2d) in enumerate(zip(d2r_gammas, r2d_gammas)):
            self.logger.info(f"  Level {i}: γ_d2r={d2r:.6f}, γ_r2d={r2d:.6f}")
            
        # 统计信息
        d2r_mean = np.mean(d2r_gammas)
        d2r_std = np.std(d2r_gammas)
        r2d_mean = np.mean(r2d_gammas)
        r2d_std = np.std(r2d_gammas)
        
        self.logger.info(f"  统计: γ_d2r均值={d2r_mean:.6f}±{d2r_std:.6f}, γ_r2d均值={r2d_mean:.6f}±{r2d_std:.6f}")
        
        # 保存JSON格式的历史记录
        self._save_history_json()
        
    def _save_history_json(self):
        """保存参数历史到JSON文件"""
        history_data = {
            'steps': self.step_history,
            'gamma_d2r': self.gamma_history['d2r'],
            'gamma_r2d': self.gamma_history['r2d'],
            'last_updated': datetime.now().isoformat()
        }
        
        json_file = os.path.join(self.log_dir, "gamma_history.json")
        with open(json_file, 'w') as f:
            json.dump(history_data, f, indent=2)
            
    def plot_gamma_evolution(self, save_path: str = None):
        """
        绘制门控参数的演化图表
        
        Args:
            save_path: 图表保存路径，如果为None则保存到log_dir
        """
        if not self.step_history:
            self.logger.warning("没有历史数据可以绘制")
            return
            
        if save_path is None:
            save_path = os.path.join(self.log_dir, f"gamma_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            
        # 转换数据格式
        steps = np.array(self.step_history)
        d2r_data = np.array(self.gamma_history['d2r'])  # [num_steps, num_levels]
        r2d_data = np.array(self.gamma_history['r2d'])  # [num_steps, num_levels]
        
        num_levels = d2r_data.shape[1] if len(d2r_data.shape) > 1 else 1
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('TRINUE 双向交叉注意力门控参数演化', fontsize=16)
        
        # 1. Depth→RGB 门控参数演化
        ax1 = axes[0, 0]
        for level in range(num_levels):
            if len(d2r_data.shape) > 1:
                ax1.plot(steps, d2r_data[:, level], label=f'Level {level}', marker='o', markersize=2)
            else:
                ax1.plot(steps, d2r_data, label='Single Level', marker='o', markersize=2)
        ax1.set_title('γ_d2r (Depth→RGB) 演化')
        ax1.set_xlabel('训练步数')
        ax1.set_ylabel('γ_d2r 值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. RGB→Depth 门控参数演化
        ax2 = axes[0, 1]
        for level in range(num_levels):
            if len(r2d_data.shape) > 1:
                ax2.plot(steps, r2d_data[:, level], label=f'Level {level}', marker='s', markersize=2)
            else:
                ax2.plot(steps, r2d_data, label='Single Level', marker='s', markersize=2)
        ax2.set_title('γ_r2d (RGB→Depth) 演化')
        ax2.set_xlabel('训练步数')
        ax2.set_ylabel('γ_r2d 值')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 门控参数均值对比
        ax3 = axes[1, 0]
        if len(d2r_data.shape) > 1:
            d2r_mean = np.mean(d2r_data, axis=1)
            r2d_mean = np.mean(r2d_data, axis=1)
        else:
            d2r_mean = d2r_data
            r2d_mean = r2d_data
            
        ax3.plot(steps, d2r_mean, label='γ_d2r 均值', color='blue', linewidth=2)
        ax3.plot(steps, r2d_mean, label='γ_r2d 均值', color='red', linewidth=2)
        ax3.set_title('门控参数均值对比')
        ax3.set_xlabel('训练步数')
        ax3.set_ylabel('门控参数均值')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 最新参数值分布
        ax4 = axes[1, 1]
        if self.gamma_history['d2r'] and self.gamma_history['r2d']:
            latest_d2r = self.gamma_history['d2r'][-1]
            latest_r2d = self.gamma_history['r2d'][-1]
            
            levels = list(range(len(latest_d2r)))
            x_pos = np.arange(len(levels))
            
            width = 0.35
            ax4.bar(x_pos - width/2, latest_d2r, width, label='γ_d2r', alpha=0.8)
            ax4.bar(x_pos + width/2, latest_r2d, width, label='γ_r2d', alpha=0.8)
            
            ax4.set_title(f'最新门控参数值 (Step {self.step_history[-1]})')
            ax4.set_xlabel('编码器层级')
            ax4.set_ylabel('门控参数值')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels([f'Level {i}' for i in levels])
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, (d2r, r2d) in enumerate(zip(latest_d2r, latest_r2d)):
                ax4.text(i - width/2, d2r + 0.001, f'{d2r:.3f}', ha='center', va='bottom', fontsize=8)
                ax4.text(i + width/2, r2d + 0.001, f'{r2d:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"门控参数演化图表已保存到: {save_path}")
        
    def get_summary_stats(self) -> Dict:
        """获取门控参数的统计摘要"""
        if not self.gamma_history['d2r'] or not self.gamma_history['r2d']:
            return {}
            
        latest_d2r = self.gamma_history['d2r'][-1]
        latest_r2d = self.gamma_history['r2d'][-1]
        
        stats = {
            'latest_step': self.step_history[-1] if self.step_history else 0,
            'num_levels': len(latest_d2r),
            'gamma_d2r': {
                'values': latest_d2r,
                'mean': float(np.mean(latest_d2r)),
                'std': float(np.std(latest_d2r)),
                'min': float(np.min(latest_d2r)),
                'max': float(np.max(latest_d2r))
            },
            'gamma_r2d': {
                'values': latest_r2d,
                'mean': float(np.mean(latest_r2d)),
                'std': float(np.std(latest_r2d)),
                'min': float(np.min(latest_r2d)),
                'max': float(np.max(latest_r2d))
            }
        }
        
        return stats


def analyze_checkpoint_gamma_params(checkpoint_path: str, output_dir: str = None):
    """
    分析检查点文件中的门控参数
    
    Args:
        checkpoint_path: 检查点文件路径
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = os.path.dirname(checkpoint_path)
        
    print(f"正在分析检查点: {checkpoint_path}")
    
    try:
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 提取模型状态
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # 查找门控参数
        d2r_params = []
        r2d_params = []
        
        for key, value in state_dict.items():
            if 'encoder.d2r_gamma' in key:
                level = int(key.split('.')[-1])
                d2r_params.append((level, value.item()))
            elif 'encoder.r2d_gamma' in key:
                level = int(key.split('.')[-1])
                r2d_params.append((level, value.item()))
                
        # 排序
        d2r_params.sort(key=lambda x: x[0])
        r2d_params.sort(key=lambda x: x[0])
        
        # 输出结果
        print("\n=== 门控参数分析结果 ===")
        print(f"检查点文件: {os.path.basename(checkpoint_path)}")
        if 'epoch' in checkpoint:
            print(f"训练轮次: {checkpoint['epoch']}")
        if 'step' in checkpoint:
            print(f"训练步数: {checkpoint['step']}")
            
        print(f"\n找到 {len(d2r_params)} 个 γ_d2r 参数:")
        for level, value in d2r_params:
            print(f"  Level {level}: γ_d2r = {value:.6f}")
            
        print(f"\n找到 {len(r2d_params)} 个 γ_r2d 参数:")
        for level, value in r2d_params:
            print(f"  Level {level}: γ_r2d = {value:.6f}")
            
        # 统计信息
        if d2r_params and r2d_params:
            d2r_values = [v for _, v in d2r_params]
            r2d_values = [v for _, v in r2d_params]
            
            print(f"\n=== 统计信息 ===")
            print(f"γ_d2r: 均值={np.mean(d2r_values):.6f}, 标准差={np.std(d2r_values):.6f}")
            print(f"γ_r2d: 均值={np.mean(r2d_values):.6f}, 标准差={np.std(r2d_values):.6f}")
            
            # 保存结果到文件
            result_file = os.path.join(output_dir, f"gamma_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(result_file, 'w') as f:
                f.write(f"门控参数分析结果\n")
                f.write(f"检查点: {checkpoint_path}\n")
                f.write(f"分析时间: {datetime.now()}\n\n")
                
                f.write("γ_d2r 参数:\n")
                for level, value in d2r_params:
                    f.write(f"  Level {level}: {value:.6f}\n")
                    
                f.write("\nγ_r2d 参数:\n")
                for level, value in r2d_params:
                    f.write(f"  Level {level}: {value:.6f}\n")
                    
                f.write(f"\n统计信息:\n")
                f.write(f"γ_d2r: 均值={np.mean(d2r_values):.6f}, 标准差={np.std(d2r_values):.6f}\n")
                f.write(f"γ_r2d: 均值={np.mean(r2d_values):.6f}, 标准差={np.std(r2d_values):.6f}\n")
                
            print(f"\n分析结果已保存到: {result_file}")
        
    except Exception as e:
        print(f"分析检查点时出错: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TRINUE 门控参数监控工具")
    parser.add_argument("--checkpoint", type=str, help="要分析的检查点文件路径")
    parser.add_argument("--output_dir", type=str, help="输出目录")
    
    args = parser.parse_args()
    
    if args.checkpoint:
        analyze_checkpoint_gamma_params(args.checkpoint, args.output_dir)
    else:
        print("请提供检查点文件路径，例如:")
        print("python monitor_gamma_params.py --checkpoint /path/to/checkpoint.pth") 