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


def set_seed(seed):
    """设置随机种子以确保可重复性"""
    import torch
    import numpy as np
    import random
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 