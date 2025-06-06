import os
import logging
import logging.handlers
from datetime import datetime
from typing import Dict, Optional, Union
import sys
import warnings

class MultiFileLogger:
    """
    多文件日志管理器，将不同类型的日志分别保存到不同的文件中
    """
    
    def __init__(self, log_dir: str, console_level: str = "INFO", file_level: str = "DEBUG"):
        """
        初始化多文件日志管理器
        
        Args:
            log_dir: 日志目录路径
            console_level: 控制台日志级别
            file_level: 文件日志级别
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 记录当前epoch，用于按epoch切分日志文件
        self.current_epoch = None

        # 日志级别映射
        self.console_level = getattr(logging, console_level.upper(), logging.INFO)
        self.file_level = getattr(logging, file_level.upper(), logging.DEBUG)
        
        # 格式化器
        self.file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [%(name)s] %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # 分类名称
        # 新增 'optimizer' 和 'checkpoint' 两个类别，用于记录优化器和检查点相关信息
        self.categories = [
            'train', 'loss', 'metrics', 'model', 'data',
            'optimizer', 'checkpoint', 'error', 'debug'
        ]

        # 创建日志目录结构
        for cat in self.categories + ['general']:
            os.makedirs(os.path.join(self.log_dir, cat), exist_ok=True)

        # 各logger实例及其当前文件handler
        self.loggers = {}
        self.file_handlers = {}

        for cat in self.categories:
            self.loggers[cat] = self._create_logger(cat)
            self.file_handlers[cat] = None

        # 设置根logger，捕获未分类的日志
        self._setup_root_logger()
        self.file_handlers['general'] = None
        
        # 记录日志系统初始化
        self.loggers['train'].info(f"Multi-file logging system initialized. Logs directory: {log_dir}")
        
    def _create_logger(self, name: str, description: str = "") -> logging.Logger:
        """
        创建一个专门的logger
        
        Args:
            name: logger名称
            description: logger描述
            
        Returns:
            配置好的logger实例
        """
        logger = logging.getLogger(f'TRINUE.{name}')
        logger.setLevel(logging.DEBUG)  # Logger本身接收所有级别
        logger.propagate = False  # 不传播到父logger
        
        # 清除已有的handlers
        logger.handlers = []

        # 控制台handler - 根据logger类型决定是否添加
        if name in ['train', 'error'] or self.console_level <= logging.DEBUG:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.console_level)
            console_handler.setFormatter(self.console_formatter)
            logger.addHandler(console_handler)
        
        # 记录logger创建
        logger.debug(f"Logger '{name}' created - {description}")
        
        return logger
    
    def _setup_root_logger(self):
        """设置根logger，捕获未分类的日志"""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # 清除已有的handlers
        root_logger.handlers = []

        # 控制台handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.console_level)
        console_handler.setFormatter(self.console_formatter)
        root_logger.addHandler(console_handler)
        
        # 设置警告捕获
        def custom_showwarning(message, category, filename, lineno, file=None, line=None):
            warning_msg = warnings.formatwarning(message, category, filename, lineno, line)
            self.loggers['error'].warning(warning_msg.strip())
        
        warnings.showwarning = custom_showwarning

    def _attach_epoch_file_handlers(self, epoch: int):
        """为所有logger创建按epoch划分的文件handler"""
        if self.current_epoch == epoch:
            return

        self.current_epoch = epoch

        target_categories = self.categories + ['general']

        for cat in target_categories:
            logger = logging.getLogger() if cat == 'general' else self.loggers[cat]

            # 移除旧handler
            old_handler = self.file_handlers.get(cat)
            if old_handler:
                logger.removeHandler(old_handler)
                old_handler.close()

            file_path = os.path.join(self.log_dir, cat, f'epoch_{epoch+1}.log')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            handler = logging.FileHandler(file_path, encoding='utf-8')
            handler.setLevel(self.file_level)
            handler.setFormatter(self.file_formatter)
            logger.addHandler(handler)

            self.file_handlers[cat] = handler
    
    def get_logger(self, logger_type: str) -> logging.Logger:
        """
        获取指定类型的logger
        
        Args:
            logger_type: logger类型 ('train', 'loss', 'metrics', 'model',
                                    'data', 'optimizer', 'checkpoint',
                                    'error', 'debug')
            
        Returns:
            对应的logger实例
        """
        return self.loggers.get(logger_type, self.loggers['debug'])
    
    def log_training_start(self, config: dict):
        """记录训练开始的信息"""
        train_logger = self.loggers['train']
        train_logger.info("="*80)
        train_logger.info("TRAINING STARTED")
        train_logger.info("="*80)
        train_logger.info(f"Experiment: {config.get('experiment', {}).get('name', 'unnamed')}")
        train_logger.info(f"Epochs: {config.get('train', {}).get('epochs', 'N/A')}")
        train_logger.info(f"Batch size: {config.get('data', {}).get('batch_size', 'N/A')}")
        train_logger.info(f"Learning rate: {config.get('optimizer', {}).get('lr', 'N/A')}")
        train_logger.info(f"Mixed precision: {config.get('gpu', {}).get('mixed_precision', False)}")
        train_logger.info("="*80)

        # 初始化第一轮日志文件
        self._attach_epoch_file_handlers(0)
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """记录epoch开始"""
        self._attach_epoch_file_handlers(epoch)
        self.loggers['train'].info(f"\n{'='*60}")
        self.loggers['train'].info(f"Epoch {epoch+1}/{total_epochs} started")
        self.loggers['train'].info(f"{'='*60}")
    
    def log_epoch_end(self, epoch: int, metrics: dict):
        """记录epoch结束和总结"""
        train_logger = self.loggers['train']
        train_logger.info(f"Epoch {epoch+1} completed")
        
        # 记录主要指标
        if 'train_loss' in metrics:
            train_logger.info(f"  Train Loss: {metrics['train_loss']:.4f}")
        if 'val_loss' in metrics:
            train_logger.info(f"  Val Loss: {metrics['val_loss']:.4f}")
        if 'val_psnr' in metrics:
            train_logger.info(f"  Val PSNR: {metrics['val_psnr']:.2f}")
        if 'lr' in metrics:
            train_logger.info(f"  Learning Rate: {metrics['lr']:.6f}")
    
    def log_loss(self, losses: dict, step: int, prefix: str = "train"):
        """记录损失值"""
        loss_logger = self.loggers['loss']
        
        # 记录总损失
        if 'total' in losses or 'loss' in losses:
            total_loss = losses.get('total', losses.get('loss', 0))
            loss_logger.info(f"[{prefix}] Step {step}: Total Loss = {total_loss:.6f}")
        
        # 记录各个损失组件
        components = []
        for k, v in losses.items():
            if k not in ['total', 'loss'] and 'loss' in k.lower():
                components.append(f"{k}={v:.6f}")
        
        if components:
            loss_logger.debug(f"[{prefix}] Step {step}: Components - {', '.join(components)}")
    
    def log_metrics(self, metrics: dict, step: int, prefix: str = "val"):
        """记录评估指标"""
        metrics_logger = self.loggers['metrics']
        
        # 格式化指标
        metric_strs = []
        for k, v in metrics.items():
            if isinstance(v, float):
                metric_strs.append(f"{k}={v:.4f}")
            else:
                metric_strs.append(f"{k}={v}")
        
        if metric_strs:
            metrics_logger.info(f"[{prefix}] Step {step}: {', '.join(metric_strs)}")
    
    def log_model_info(self, message: str, level: str = "info"):
        """记录模型相关信息"""
        model_logger = self.loggers['model']
        log_func = getattr(model_logger, level.lower(), model_logger.info)
        log_func(message)
    
    def log_data_info(self, message: str, level: str = "info"):
        """记录数据相关信息"""
        data_logger = self.loggers['data']
        log_func = getattr(data_logger, level.lower(), data_logger.info)
        log_func(message)

    def log_optimizer(self, message: str, level: str = "info"):
        """记录优化器相关信息"""
        opt_logger = self.loggers['optimizer']
        log_func = getattr(opt_logger, level.lower(), opt_logger.info)
        log_func(message)

    def log_checkpoint(self, message: str, level: str = "info"):
        """记录检查点保存和加载信息"""
        ckpt_logger = self.loggers['checkpoint']
        log_func = getattr(ckpt_logger, level.lower(), ckpt_logger.info)
        log_func(message)
    
    def log_error(self, message: str, exc_info=None):
        """记录错误信息"""
        self.loggers['error'].error(message, exc_info=exc_info)
    
    def log_warning(self, message: str):
        """记录警告信息"""
        self.loggers['error'].warning(message)
    
    def log_debug(self, message: str):
        """记录调试信息"""
        self.loggers['debug'].debug(message)
    
    def close(self):
        """关闭所有logger的handlers"""
        for logger in list(self.loggers.values()) + [logging.getLogger()]:
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)


def create_multi_logger(config: dict, exp_dir: str) -> MultiFileLogger:
    """
    创建多文件日志管理器的便捷函数
    
    Args:
        config: 配置字典
        exp_dir: 实验目录
        
    Returns:
        MultiFileLogger实例
    """
    log_dir = os.path.join(exp_dir, 'logs')
    
    # 从配置中获取日志级别
    logging_config = config.get('logging', {})
    console_level = logging_config.get('console_level', 'INFO')
    file_level = logging_config.get('file_level', 'DEBUG')
    
    return MultiFileLogger(log_dir, console_level, file_level) 
