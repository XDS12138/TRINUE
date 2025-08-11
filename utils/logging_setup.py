#!/usr/bin/env python3
"""
日志系统设置模块

负责设置完整的多文件日志系统，包括TensorBoard
"""

import os
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utils.multi_logger import create_multi_logger
from utils.logger import MetricLogger


def setup_logging_system(exp_dir, config):
    """设置完整的多文件日志系统，包括TensorBoard"""
    
    # 分布式rank检测（兼容torchrun）
    try:
        rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
    except Exception:
        rank = 0
    is_rank0 = (rank == 0)
    
    # 创建多文件日志记录器
    multi_logger = create_multi_logger(config, exp_dir)
    
    # 获取主要的日志记录器
    main_logger = multi_logger.get_logger('train')
    
    # 仅在rank0创建TensorBoard，避免DDP多进程写同一事件文件
    tb_log_dir = os.path.join(exp_dir, 'tensorboard', datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(tb_log_dir, exist_ok=True)
    
    tb_writer = None
    if is_rank0:
        try:
            tb_writer = SummaryWriter(log_dir=tb_log_dir)
            # 测试写入确保目录可写
            tb_writer.add_scalar('setup/tensorboard_test', 1, 0)
            tb_writer.flush()
            main_logger.info(f"TensorBoard日志将写入目录: {tb_log_dir}")
            main_logger.info("TensorBoard初始化测试写入成功")
        except ImportError:
            main_logger.warning("TensorBoard未安装，部分可视化功能将不可用。请运行 `pip install tensorboard` 安装。")
            tb_writer = None
        except Exception as e:
            main_logger.error(f"TensorBoard初始化失败: {str(e)}. 请检查路径权限和磁盘空间。")
            tb_writer = None
    else:
        main_logger.info(f"[rank{rank}] 跳过TensorBoard初始化（仅rank0写事件文件）")

    # 创建MetricLogger实例（使用真正的功能完整版本）
    # 非rank0写入独立CSV，避免并发写同一文件
    csv_filename = 'metrics.csv' if is_rank0 else f'metrics_rank{rank}.csv'
    csv_path = os.path.join(exp_dir, csv_filename)
    metrics_csv_cfg = config.get('logging', {}).get('metrics_csv', {})
    # 仅rank0启用紧凑版CSV
    if not is_rank0:
        metrics_csv_cfg = {**metrics_csv_cfg, 'compact_enabled': False}
    metric_logger = MetricLogger(main_logger, tb_writer, csv_path, metrics_csv_cfg)
    
    return multi_logger, metric_logger, tb_writer 