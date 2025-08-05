#!/usr/bin/env python3
"""
参数解析模块

负责解析训练脚本的命令行参数
"""

import argparse


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='UnderwaterEnhanceNet 训练脚本')
    parser.add_argument('--config', type=str, default='configs/train.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', action='store_true',
                        help='从最新检查点恢复训练')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='分布式训练的本地排名 (DDP)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--no_cuda', action='store_true',
                        help='禁用CUDA')
    parser.add_argument('--eval_only', action='store_true',
                        help='仅运行验证')
    parser.add_argument('--distributed', action='store_true',
                        help='是否使用分布式训练')
    parser.add_argument('--log_level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        help='设置控制台日志级别')
    return parser.parse_args()


def set_seed(seed, gpu_config=None):
    """设置随机种子以确保可重复性
    
    Args:
        seed: 随机种子
        gpu_config: GPU配置字典，包含optimization相关设置
    """
    import torch
    import numpy as np
    import random
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 从配置中读取GPU优化设置
    if gpu_config and 'optimization' in gpu_config:
        opt_config = gpu_config['optimization']
        
        # 设置TF32
        tf32_enabled = opt_config.get('tf32_enabled', True)  # 默认启用TF32
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled
        
        # 设置CUDNN选项
        torch.backends.cudnn.deterministic = opt_config.get('cudnn_deterministic', True)
        torch.backends.cudnn.benchmark = opt_config.get('cudnn_benchmark', False)
        
        print(f"🚀 GPU优化设置:")
        print(f"   - TF32加速: {'✅ 启用' if tf32_enabled else '❌ 禁用'}")
        print(f"   - CUDNN确定性: {'✅ 启用' if opt_config.get('cudnn_deterministic', True) else '❌ 禁用'}")
        print(f"   - CUDNN基准测试: {'✅ 启用' if opt_config.get('cudnn_benchmark', False) else '❌ 禁用'}")
    else:
        # 默认设置（兼容旧配置）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("⚠️  使用默认GPU设置（未找到optimization配置）") 