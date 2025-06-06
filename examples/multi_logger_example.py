#!/usr/bin/env python3
"""
多文件日志系统使用示例

这个示例展示了如何使用新的多文件日志系统，将不同类型的日志分别保存到不同的文件中。
"""

import os
import sys
import torch
import numpy as np

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.multi_logger import MultiFileLogger, create_multi_logger

def main():
    # 创建一个假的配置字典
    config = {
        'logging': {
            'console_level': 'INFO',
            'file_level': 'DEBUG'
        }
    }
    
    # 创建实验目录
    exp_dir = 'experiments/example_run'
    os.makedirs(exp_dir, exist_ok=True)
    
    # 创建多文件日志管理器
    multi_logger = create_multi_logger(config, exp_dir)
    
    # 获取不同类型的logger
    train_logger = multi_logger.get_logger('train')
    loss_logger = multi_logger.get_logger('loss')
    metrics_logger = multi_logger.get_logger('metrics')
    model_logger = multi_logger.get_logger('model')
    data_logger = multi_logger.get_logger('data')
    error_logger = multi_logger.get_logger('error')
    debug_logger = multi_logger.get_logger('debug')
    
    # 记录训练开始
    multi_logger.log_training_start(config)
    
    # 模拟训练过程
    for epoch in range(3):
        # 记录epoch开始
        multi_logger.log_epoch_start(epoch, 3)
        
        # 模拟数据加载
        data_logger.info(f"Loading data for epoch {epoch+1}")
        data_logger.debug(f"Batch size: 32, Number of workers: 4")
        
        # 模拟模型更新
        model_logger.info(f"Model parameters updated at epoch {epoch+1}")
        model_logger.debug(f"Learning rate: {0.001 * (0.9 ** epoch):.6f}")
        
        # 模拟训练步骤
        for step in range(10):
            # 记录损失
            losses = {
                'total': np.random.uniform(0.5, 1.0),
                'loss_l1': np.random.uniform(0.1, 0.3),
                'loss_ssim': np.random.uniform(0.2, 0.4),
                'loss_depth': np.random.uniform(0.1, 0.2)
            }
            multi_logger.log_loss(losses, epoch * 10 + step, 'train')
            
            # 每5步记录一次详细信息
            if step % 5 == 0:
                debug_logger.debug(f"Step {step}: Memory usage = {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
        
        # 模拟验证
        val_metrics = {
            'psnr': np.random.uniform(25, 30),
            'ssim': np.random.uniform(0.8, 0.95),
            'mae': np.random.uniform(0.01, 0.05)
        }
        multi_logger.log_metrics(val_metrics, epoch, 'val')
        
        # 记录epoch结束
        multi_logger.log_epoch_end(epoch, {
            'train_loss': losses['total'],
            'val_psnr': val_metrics['psnr'],
            'lr': 0.001 * (0.9 ** epoch)
        })
        
        # 模拟错误处理
        if epoch == 1:
            try:
                # 模拟一个错误
                raise ValueError("模拟的错误用于演示")
            except Exception as e:
                multi_logger.log_error("训练过程中捕获到错误", exc_info=True)
                multi_logger.log_warning("继续训练，但可能存在问题")
    
    # 记录最终总结
    train_logger.info("Training completed successfully!")
    train_logger.info(f"Final metrics - PSNR: {val_metrics['psnr']:.2f}, SSIM: {val_metrics['ssim']:.4f}")
    
    # 关闭日志系统
    multi_logger.close()
    
    # 展示生成的日志文件
    log_dir = os.path.join(exp_dir, 'logs')
    print("\n生成的日志文件：")
    for filename in os.listdir(log_dir):
        filepath = os.path.join(log_dir, filename)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath) / 1024  # KB
            print(f"  - {filename}: {size:.2f} KB")

if __name__ == '__main__':
    main() 