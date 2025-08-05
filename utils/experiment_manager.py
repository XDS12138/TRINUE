#!/usr/bin/env python3
"""
实验管理模块

负责实验目录的创建、查找和配置保存
"""

import os
import yaml
from datetime import datetime


def find_latest_experiment_with_checkpoints(output_dir, base_name):
    """
    在输出目录中查找最新的包含检查点的实验目录
    
    Args:
        output_dir: 实验输出根目录
        base_name: 实验基础名称 (如 'underwater_enhance_run')
        
    Returns:
        str or None: 找到的最新实验目录路径，如果没有找到返回None
    """
    if not os.path.exists(output_dir):
        return None
    
    # 查找所有匹配的实验目录
    matching_dirs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith(base_name):
            # 检查是否有检查点目录且不为空
            checkpoint_dir = os.path.join(item_path, 'checkpoints')
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(('.pth', '.pth.tar'))]
                if checkpoints:
                    # 记录目录修改时间用于排序
                    mtime = os.path.getmtime(item_path)
                    matching_dirs.append((item_path, mtime))
    
    # 按修改时间排序，返回最新的
    if matching_dirs:
        matching_dirs.sort(key=lambda x: x[1], reverse=True)
        latest_dir = matching_dirs[0][0]
        return latest_dir
    
    return None


def setup_experiment_dir(config, resume_mode=False):
    """
    设置实验目录并保存配置，支持自动添加时间戳或序号避免覆盖
    
    Args:
        config: 配置字典
        resume_mode: 是否为恢复训练模式
        
    Returns:
        str: 实验目录路径
    """
    exp_name = config['experiment']['name']
    output_dir = os.path.expanduser(config['experiment']['output_dir'])  # 🔧 修复：展开用户目录
    
    # 🔥 智能恢复逻辑：如果是恢复模式，优先查找现有实验
    if resume_mode:
        latest_exp_dir = find_latest_experiment_with_checkpoints(output_dir, exp_name)
        if latest_exp_dir:
            print(f"[智能恢复] 找到包含检查点的最新实验目录: {latest_exp_dir}")
            print(f"[智能恢复] 将恢复训练而不是创建新目录")
            return latest_exp_dir
        else:
            print(f"[智能恢复] 未找到包含检查点的实验目录，将创建新实验")
    
    # 原有逻辑：创建新实验目录
    if config['experiment'].get('auto_naming', True):
        timestamp_format = config['experiment'].get('timestamp_format', "%Y%m%d_%H%M%S")
        timestamp = datetime.now().strftime(timestamp_format)
        exp_name = f"{exp_name}_{timestamp}"
    else:
        # 检查是否已存在相同名称的实验目录，如果存在则添加序号
        base_dir = os.path.join(output_dir, exp_name)
        if os.path.exists(base_dir):
            i = 1
            while os.path.exists(os.path.join(output_dir, f"{exp_name}_{i}")):
                i += 1
            exp_name = f"{exp_name}_{i}"
    
    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 保存配置
    config_save_path = os.path.join(exp_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return exp_dir 