#!/usr/bin/env python3
"""
检查点管理模块

负责模型检查点的保存、加载和恢复
"""

import os
import logging
import traceback
import torch

logger = logging.getLogger(__name__)


def resume_from_checkpoint(checkpoint_dir: str,
                         model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer = None,
                         scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                         device = None,
                         scaler = None,
                         train_sampler = None) -> tuple:
    """
    从检查点目录中找到最新的检查点并恢复模型、优化器、调度器状态
    
    Args:
        checkpoint_dir: 包含检查点文件的目录
        model: 模型实例
        optimizer: 优化器实例
        scheduler: 学习率调度器实例
        device: 模型应该加载到的设备
        scaler: GradScaler实例（用于混合精度训练）
        
    Returns:
        A tuple containing:
        - model: The loaded model
        - optimizer: The loaded optimizer state
        - scheduler: The loaded scheduler state
        - scaler: The loaded GradScaler state
        - start_epoch: The epoch to start training from
        - best_metric: The best metric value from previous training
    """
    # 查找最新的检查点文件
    if not os.path.isdir(checkpoint_dir):
        logger.warning(f"检查点目录 '{checkpoint_dir}' 不存在，从头开始训练。")
        return model, optimizer, scheduler, scaler, 0, 0.0

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        logger.warning(f"在 '{checkpoint_dir}' 中没有找到检查点文件，从头开始训练。")
        return model, optimizer, scheduler, scaler, 0, 0.0

    # 按修改时间排序，找到最新的文件
    latest_checkpoint_path = max([os.path.join(checkpoint_dir, f) for f in checkpoints], key=os.path.getmtime)
    logger.info(f"从最新的检查点恢复训练: {latest_checkpoint_path}")

    try:
        # 加载检查点
        checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
        
        # 恢复模型权重
        # 兼容DDP和非DDP模型
        model_to_load = model.module if hasattr(model, 'module') else model
        
        # 处理可能的状态字典键不匹配问题
        state_dict = checkpoint['model_state_dict']
        # 移除 'module.' 前缀 (如果存在)
        if all(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        try:
            model_to_load.load_state_dict(state_dict)
        except RuntimeError as e:
            logger.warning(f"加载模型状态时遇到严格模式错误: {e}")
            logger.info("尝试以非严格模式加载...")
            model_to_load.load_state_dict(state_dict, strict=False)

        # 恢复优化器状态
        if optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                logger.error(f"无法恢复优化器状态: {e}, 优化器将从头开始。")

        # 恢复调度器状态
        if scheduler and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                logger.error(f"无法恢复调度器状态: {e}, 调度器将从头开始。")

        # 恢复混合精度scaler状态
        if scaler and 'scaler_state_dict' in checkpoint:
            try:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            except Exception as e:
                logger.error(f"无法恢复混合精度scaler状态: {e}")

        # 恢复训练采样器状态（用于分布式训练的epoch同步）
        if train_sampler and 'sampler_epoch' in checkpoint:
            try:
                if hasattr(train_sampler, 'set_epoch'):
                    train_sampler.set_epoch(checkpoint['sampler_epoch'])
                    logger.info(f"恢复训练采样器epoch状态: {checkpoint['sampler_epoch']}")
            except Exception as e:
                logger.error(f"无法恢复训练采样器状态: {e}")

        # 恢复训练轮次和最佳指标
        start_epoch = checkpoint.get('epoch', 0)
        best_metric = checkpoint.get('best_metric', 0.0)

        logger.info(f"成功从 epoch {start_epoch} 恢复训练, 最佳PSNR为 {best_metric:.4f}")

        return model, optimizer, scheduler, scaler, start_epoch, best_metric

    except Exception as e:
        logger.error(f"加载检查点 '{latest_checkpoint_path}' 失败: {e}")
        logger.error(traceback.format_exc())
        logger.warning("将从头开始训练。")
        return model, optimizer, scheduler, scaler, 0, 0.0


def save_checkpoint_extended(state, is_best, checkpoint_dir, filename=None, keep_last_n=5):
    """
    扩展的检查点保存功能，支持保留最近N个检查点
    
    Args:
        state: 包含模型状态的字典
        is_best: 是否为最佳模型
        checkpoint_dir: 检查点保存目录
        filename: 检查点文件名
        keep_last_n: 保留最近N个检查点
    """
    from utils.checkpoint import save_checkpoint
    
    # 使用原有的保存功能
    save_checkpoint(state, is_best, checkpoint_dir, filename)
    
    # 清理旧的检查点
    if keep_last_n > 0:
        _cleanup_old_checkpoints(checkpoint_dir, keep_last_n)


def _cleanup_old_checkpoints(checkpoint_dir, keep_last_n):
    """清理旧的检查点文件，保留最近的N个"""
    try:
        checkpoints = [f for f in os.listdir(checkpoint_dir) 
                      if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
        
        if len(checkpoints) <= keep_last_n:
            return
        
        # 按修改时间排序
        checkpoint_paths = [os.path.join(checkpoint_dir, f) for f in checkpoints]
        checkpoint_paths.sort(key=os.path.getmtime, reverse=True)
        
        # 删除最旧的检查点
        for old_checkpoint in checkpoint_paths[keep_last_n:]:
            os.remove(old_checkpoint)
            logger.info(f"删除旧检查点: {os.path.basename(old_checkpoint)}")
            
    except Exception as e:
        logger.warning(f"清理旧检查点时出错: {e}")


def validate_checkpoint(checkpoint_path):
    """验证检查点文件的有效性"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        required_keys = ['model_state_dict', 'epoch']
        
        for key in required_keys:
            if key not in checkpoint:
                return False, f"缺少必要的键: {key}"
        
        return True, "检查点有效"
        
    except Exception as e:
        return False, f"加载检查点失败: {e}"


def get_checkpoint_info(checkpoint_path):
    """获取检查点的基本信息"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        info = {
            'epoch': checkpoint.get('epoch', 'Unknown'),
            'best_metric': checkpoint.get('best_metric', 'Unknown'),
            'file_size': os.path.getsize(checkpoint_path) / (1024 * 1024),  # MB
            'creation_time': os.path.getctime(checkpoint_path)
        }
        
        return info
        
    except Exception as e:
        logger.error(f"获取检查点信息失败: {e}")
        return None 


def validate_config_compatibility(checkpoint_path, current_config):
    """
    验证checkpoint与当前配置的兼容性
    
    Args:
        checkpoint_path: 检查点文件路径
        current_config: 当前训练配置
        
    Returns:
        bool: 是否兼容
        str: 兼容性信息
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # 检查是否保存了配置信息
        if 'config' not in checkpoint:
            return True, "检查点未包含配置信息，跳过兼容性检查"
        
        saved_config = checkpoint['config']
        
        # 检查关键配置项
        critical_keys = [
            ['model', 'type'],
            ['model', 'params'],
            ['data', 'input_size'],
            ['train', 'mixed_precision']
        ]
        
        warnings = []
        errors = []
        
        for key_path in critical_keys:
            saved_val = saved_config
            current_val = current_config
            
            try:
                for key in key_path:
                    saved_val = saved_val[key]
                    current_val = current_val[key]
                
                if saved_val != current_val:
                    key_str = '.'.join(key_path)
                    if key_path[0] == 'model':  # 模型相关是错误
                        errors.append(f"{key_str}: saved={saved_val}, current={current_val}")
                    else:  # 其他是警告
                        warnings.append(f"{key_str}: saved={saved_val}, current={current_val}")
                        
            except KeyError:
                # 配置键不存在，跳过
                continue
        
        if errors:
            error_msg = "配置不兼容:\n" + "\n".join([f"  ❌ {e}" for e in errors])
            if warnings:
                error_msg += "\n配置警告:\n" + "\n".join([f"  ⚠️  {w}" for w in warnings])
            return False, error_msg
        
        if warnings:
            warning_msg = "配置警告:\n" + "\n".join([f"  ⚠️  {w}" for w in warnings])
            return True, warning_msg
        
        return True, "✅ 配置完全兼容"
        
    except Exception as e:
        return True, f"配置兼容性检查失败: {e}"


def get_checkpoint_summary(checkpoint_dir):
    """
    获取检查点目录的摘要信息
    
    Args:
        checkpoint_dir: 检查点目录
        
    Returns:
        dict: 摘要信息
    """
    if not os.path.isdir(checkpoint_dir):
        return {"error": "目录不存在"}
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        return {"error": "无检查点文件"}
    
    # 获取最新和最佳检查点信息
    latest_checkpoint = max([os.path.join(checkpoint_dir, f) for f in checkpoints], key=os.path.getmtime)
    best_checkpoint = os.path.join(checkpoint_dir, 'model_best.pth') if 'model_best.pth' in checkpoints else None
    
    summary = {
        "total_checkpoints": len(checkpoints),
        "latest_checkpoint": {
            "path": latest_checkpoint,
            "info": get_checkpoint_info(latest_checkpoint)
        }
    }
    
    if best_checkpoint and os.path.exists(best_checkpoint):
        summary["best_checkpoint"] = {
            "path": best_checkpoint,
            "info": get_checkpoint_info(best_checkpoint)
        }
    
    return summary 