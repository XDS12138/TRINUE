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
        
        # 创建各种专门的logger
        self.loggers = {
            'train': self._create_logger('train', 'train.log', 'Training progress and milestones'),
            'loss': self._create_logger('loss', 'loss.log', 'Loss values and components'),
            'metrics': self._create_logger('metrics', 'metrics.log', 'Validation metrics and evaluations'),
            'model': self._create_logger('model', 'model.log', 'Model architecture and parameters'),
            'data': self._create_logger('data', 'data.log', 'Data loading and preprocessing'),
            'error': self._create_logger('error', 'error.log', 'Errors and warnings'),
            'debug': self._create_logger('debug', 'debug.log', 'Detailed debugging information'),
        }
        
        # 设置根logger，捕获未分类的日志
        self._setup_root_logger()
        
        # 记录日志系统初始化
        self.loggers['train'].info(f"Multi-file logging system initialized. Logs directory: {log_dir}")
        
    def _create_logger(self, name: str, filename: str, description: str) -> logging.Logger:
        """
        创建一个专门的logger
        
        Args:
            name: logger名称
            filename: 日志文件名
            description: logger描述
            
        Returns:
            配置好的logger实例
        """
        logger = logging.getLogger(f'TRINUE.{name}')
        logger.setLevel(logging.DEBUG)  # Logger本身接收所有级别
        logger.propagate = False  # 不传播到父logger
        
        # 清除已有的handlers
        logger.handlers = []
        
        # 文件handler - 使用基于时间的TimedRotatingFileHandler而非基于大小的RotatingFileHandler
        file_path = os.path.join(self.log_dir, filename)
        
        # 创建日志目录的父目录（如果不存在）
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 使用基于日期的轮转，保留所有历史日志
        # 每天在午夜轮转一次日志文件
        file_handler = logging.handlers.TimedRotatingFileHandler(
            file_path,
            when='midnight',
            interval=1,
            backupCount=0,  # 设置为0表示保留所有日志文件
            encoding='utf-8',
            atTime=None
        )
        # 设置日志文件命名格式为: filename.log.YYYY-MM-DD
        file_handler.suffix = "%Y-%m-%d"
        file_handler.setLevel(self.file_level)
        file_handler.setFormatter(self.file_formatter)
        logger.addHandler(file_handler)
        
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
        
        # 添加一个handler将未分类的日志写入general.log
        general_path = os.path.join(self.log_dir, 'general.log')
        
        # 确保日志目录存在
        os.makedirs(os.path.dirname(general_path), exist_ok=True)
        
        # 使用基于时间的日志轮转，保留所有历史日志
        general_handler = logging.handlers.TimedRotatingFileHandler(
            general_path,
            when='midnight',
            interval=1,
            backupCount=0,  # 保留所有日志
            encoding='utf-8',
            atTime=None
        )
        general_handler.suffix = "%Y-%m-%d"  # 设置文件名后缀格式
        general_handler.setLevel(self.file_level)
        general_handler.setFormatter(self.file_formatter)
        root_logger.addHandler(general_handler)
        
        # 设置警告捕获
        def custom_showwarning(message, category, filename, lineno, file=None, line=None):
            warning_msg = warnings.formatwarning(message, category, filename, lineno, line)
            self.loggers['error'].warning(warning_msg.strip())
        
        warnings.showwarning = custom_showwarning
    
    def get_logger(self, logger_type: str) -> logging.Logger:
        """
        获取指定类型的logger
        
        Args:
            logger_type: logger类型 ('train', 'loss', 'metrics', 'model', 'data', 'error', 'debug')
            
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
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """记录epoch开始"""
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
        for logger in self.loggers.values():
            for handler in logger.handlers:
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