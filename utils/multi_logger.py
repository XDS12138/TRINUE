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
        
        # 更新分类名称和描述
        self.categories = {
            # 训练核心类别（高频使用）
            'train': '核心训练过程（进度、批次处理、学习率等）',
            'metrics': '所有指标（PSNR、SSIM、损失等）',
            'validation': '验证过程（多验证集结果、计算指标等）',
            'depth': '深度预测相关（深度估计、误差计算、尺度对齐）',
            
            # 系统和错误类别（必要的）
            'error': '错误和异常（模型、数据、计算等各种错误）',
            'warning': '警告信息（潜在问题、配置冲突等）',
            'checkpoint': '模型保存、加载、恢复训练（权重、优化器状态等）',
            
            # 其他实用类别
            'system': '系统资源（GPU、内存、磁盘使用）',
            'experiment': '实验配置和环境（超参、架构、设备等）',
            'visualization': '可视化相关（图像保存、TensorBoard等）'
        }

        # 控制台显示的日志类别（其他只写入文件）
        self.console_categories = ['train', 'validation', 'error', 'warning']
        
        # 是否在控制台输出所有类别的日志（如果console_level <= DEBUG）
        if self.console_level <= logging.DEBUG:
            self.console_categories = list(self.categories.keys())

        # 创建日志目录结构
        for cat in list(self.categories.keys()) + ['general']:
            os.makedirs(os.path.join(self.log_dir, cat), exist_ok=True)

        # 各logger实例及其当前文件handler
        self.loggers = {}
        self.file_handlers = {}

        for cat in self.categories:
            self.loggers[cat] = self._create_logger(cat, self.categories[cat])
            self.file_handlers[cat] = None

        # 设置根logger，捕获未分类的日志
        self._setup_root_logger()
        self.file_handlers['general'] = None
        
        # 记录日志系统初始化
        self.loggers['train'].info(f"多文件日志系统初始化完成. 日志目录: {log_dir}")
        self.loggers['experiment'].info(f"已配置 {len(self.categories)} 个日志类别")
        self.loggers['experiment'].info(f"控制台日志级别: {console_level}, 文件日志级别: {file_level}")
        
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
        if name in self.console_categories:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.console_level)
            console_handler.setFormatter(self.console_formatter)
            logger.addHandler(console_handler)
        
        # 记录logger创建
        logger.debug(f"Logger '{name}' 创建完成 - {description}")
        
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
            self.loggers['warning'].warning(warning_msg.strip())
        
        warnings.showwarning = custom_showwarning

    def _attach_epoch_file_handlers(self, epoch: int):
        """为所有logger创建按epoch划分的文件handler"""
        if self.current_epoch == epoch:
            return

        self.current_epoch = epoch

        # 更新为新定义的日志类别
        target_categories = list(self.categories.keys()) + ['general']

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
            logger_type: logger类型名称
            
        Returns:
            对应的logger实例
        """
        if logger_type not in self.categories:
            self.loggers['warning'].warning(f"请求了未知类型的logger: '{logger_type}'，使用debug logger代替")
            return self.loggers['debug']
        return self.loggers[logger_type]
    
    def log_training_start(self, config: dict):
        """记录训练开始的信息"""
        train_logger = self.loggers['train']
        exp_logger = self.loggers['experiment']
        
        train_logger.info("="*80)
        train_logger.info("训练开始")
        train_logger.info("="*80)
        
        # 记录基本配置到experiment日志
        exp_logger.info("="*80)
        exp_logger.info("实验配置信息")
        exp_logger.info("="*80)
        exp_logger.info(f"实验名称: {config.get('experiment', {}).get('name', 'unnamed')}")
        exp_logger.info(f"总训练轮次: {config.get('train', {}).get('epochs', 'N/A')}")
        exp_logger.info(f"批量大小: {config.get('data', {}).get('batch_size', 'N/A')}")
        exp_logger.info(f"学习率: {config.get('optimizer', {}).get('lr', 'N/A')}")
        exp_logger.info(f"优化器: {config.get('optimizer', {}).get('name', 'N/A')}")
        exp_logger.info(f"调度器: {config.get('scheduler', {}).get('name', 'N/A')}")
        exp_logger.info(f"混合精度: {config.get('gpu', {}).get('mixed_precision', False)}")
        exp_logger.info(f"分布式训练: {config.get('gpu', {}).get('distributed', False)}")
        exp_logger.info("="*80)

        # 初始化第一轮日志文件
        self._attach_epoch_file_handlers(0)
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """记录epoch开始"""
        self._attach_epoch_file_handlers(epoch)
        self.loggers['train'].info(f"\n{'='*60}")
        self.loggers['train'].info(f"Epoch {epoch+1}/{total_epochs} 开始")
        self.loggers['train'].info(f"{'='*60}")
    
    def log_epoch_end(self, epoch: int, metrics: dict):
        """记录epoch结束和总结"""
        train_logger = self.loggers['train']
        metrics_logger = self.loggers['metrics']
        
        train_logger.info(f"Epoch {epoch+1} 完成")
        
        # 记录主要指标
        metrics_list = []
        if 'train_loss' in metrics:
            train_loss = metrics['train_loss']
            if isinstance(train_loss, (tuple, list)):
                train_loss = train_loss[0]  # 取第一个元素
            metrics_list.append(f"Train Loss: {train_loss:.4f}")
        if 'val_loss' in metrics:
            val_loss = metrics['val_loss']
            if isinstance(val_loss, (tuple, list)):
                val_loss = val_loss[0]  # 取第一个元素
            metrics_list.append(f"Val Loss: {val_loss:.4f}")
        if 'val_psnr' in metrics:
            val_psnr = metrics['val_psnr']
            if isinstance(val_psnr, (tuple, list)):
                val_psnr = val_psnr[0]  # 取第一个元素
            metrics_list.append(f"Val PSNR: {val_psnr:.2f}")
        if 'lr' in metrics:
            lr = metrics['lr']
            if isinstance(lr, (tuple, list)):
                lr = lr[0]  # 取第一个元素
            metrics_list.append(f"LR: {lr:.6f}")
            
        train_logger.info(f"  指标汇总: {' | '.join(metrics_list)}")
        
        # 详细记录到metrics日志
        metrics_logger.info(f"Epoch {epoch+1} 指标详情:")
        for k, v in metrics.items():
            if isinstance(v, float):
                metrics_logger.info(f"  {k}: {v:.6f}")
            else:
                metrics_logger.info(f"  {k}: {v}")
    
    def log_loss(self, losses: dict, step: int, prefix: str = "train"):
        """记录损失值"""
        # 将loss日志整合到metrics日志中
        metrics_logger = self.loggers['metrics']
        
        # 记录总损失
        if 'total' in losses or 'loss' in losses:
            total_loss = losses.get('total', losses.get('loss', 0))
            metrics_logger.info(f"[{prefix}] Step {step}: Total Loss = {total_loss:.6f}")
        
        # 记录各个损失组件
        components = []
        for k, v in losses.items():
            if k not in ['total', 'loss'] and 'loss' in k.lower():
                components.append(f"{k}={v:.6f}")
        
        if components:
            metrics_logger.info(f"[{prefix}] Step {step}: Components - {', '.join(components)}")
    
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
        # 重定向到系统日志
        self.loggers['system'].info(f"[MODEL] {message}")
    
    def log_architecture(self, message: str, level: str = "info"):
        """记录模型架构相关信息"""
        # 重定向到系统日志
        self.loggers['system'].info(f"[ARCH] {message}")
    
    def log_data_info(self, message: str, level: str = "info"):
        """记录数据相关信息"""
        # 重定向到训练日志
        self.loggers['train'].info(f"[DATA] {message}")
    
    def log_dataloader(self, message: str, level: str = "info"):
        """记录数据加载器信息"""
        # 重定向到训练日志
        self.loggers['train'].info(f"[LOADER] {message}")

    def log_optimizer(self, message: str, level: str = "info"):
        """记录优化器相关信息"""
        # 重定向到训练日志
        self.loggers['train'].info(f"[OPTIM] {message}")
    
    def log_scheduler(self, message: str, level: str = "info"):
        """记录学习率调度器信息"""
        # 重定向到训练日志
        self.loggers['train'].info(f"[SCHED] {message}")
    
    def log_gradient(self, message: str, level: str = "info"):
        """记录梯度相关信息"""
        # 重定向到训练日志
        self.loggers['train'].info(f"[GRAD] {message}")

    def log_checkpoint(self, message: str, level: str = "info"):
        """记录检查点保存和加载信息"""
        ckpt_logger = self.loggers['checkpoint']
        log_func = getattr(ckpt_logger, level.lower(), ckpt_logger.info)
        log_func(message)
    
    def log_memory(self, message: str, level: str = "info"):
        """记录内存使用情况"""
        # 重定向到系统日志
        self.loggers['system'].info(f"[MEM] {message}")
    
    def log_gpu(self, message: str, level: str = "info"):
        """记录GPU使用情况"""
        # 重定向到系统日志
        self.loggers['system'].info(f"[GPU] {message}")
    
    def log_visualization(self, message: str, level: str = "info"):
        """记录可视化相关信息"""
        viz_logger = self.loggers['visualization']
        log_func = getattr(viz_logger, level.lower(), viz_logger.info)
        log_func(message)
    
    def log_error(self, message: str, exc_info=None):
        """记录错误信息"""
        self.loggers['error'].error(message, exc_info=exc_info)
    
    def log_warning(self, message: str):
        """记录警告信息"""
        self.loggers['warning'].warning(message)
    
    def log_debug(self, message: str):
        """记录调试信息"""
        # 重定向到训练日志
        self.loggers['train'].debug(message)
    
    def log_experiment(self, message: str, level: str = "info"):
        """记录实验配置和环境信息"""
        exp_logger = self.loggers['experiment']
        log_func = getattr(exp_logger, level.lower(), exp_logger.info)
        log_func(message)
    
    def log_depth(self, message: str, level: str = "info"):
        """记录深度预测相关信息"""
        depth_logger = self.loggers['depth']
        log_func = getattr(depth_logger, level.lower(), depth_logger.info)
        log_func(message)
    
    def log_physics(self, message: str, level: str = "info"):
        """记录物理模型相关信息"""
        # 重定向到深度日志
        self.loggers['depth'].info(f"[PHYSICS] {message}")
    
    def log_attention(self, message: str, level: str = "info"):
        """记录注意力机制相关信息"""
        # 重定向到系统日志
        self.loggers['system'].info(f"[ATTN] {message}")
    
    def log_uncertainty(self, message: str, level: str = "info"):
        """记录不确定性权重相关信息"""
        # 重定向到metrics日志
        self.loggers['metrics'].info(f"[UNCERT] {message}")
    
    def log_validation(self, message: str, level: str = "info"):
        """记录验证相关日志"""
        validation_logger = self.loggers['validation']
        log_func = getattr(validation_logger, level.lower(), validation_logger.info)
        log_func(message)

    def log_system(self, message: str, level: str = "info"):
        """记录系统资源相关信息"""
        system_logger = self.loggers['system']
        log_func = getattr(system_logger, level.lower(), system_logger.info)
        log_func(message)
    
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
