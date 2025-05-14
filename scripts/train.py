#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py - UnderwaterEnhanceNet 训练脚本
----------------------------------------
用途：训练水下图像增强模型，支持多种可配置参数和详细的可视化

关键特点：
- 详细的TensorBoard可视化，包括特征图、自适应深度融合权重等
- 断点续训支持
- 丰富的训练参数配置
- 多种损失函数、评估指标支持

使用示例：
python scripts/train.py --config configs/train.yaml
"""

import os
import sys
import time
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
from datetime import datetime
import torch.multiprocessing as mp

# 添加项目根目录到PATH
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)

# 导入项目模块
from modules.model import UnderwaterEnhanceNet
from modules.loss_fn import TotalLoss
from modules.teacher_student import TeacherStudentLoss
from utils.logger import setup_logger, MetricLogger
from utils.checkpoint import save_checkpoint
from utils.lr_scheduler import get_scheduler
# 假设有一个数据集模块，根据实际情况导入
# from data.dataset import UnderwaterDataset


def parse_args():
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
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_experiment_dir(config):
    """设置实验目录并保存配置，支持自动添加时间戳或序号避免覆盖"""
    exp_name = config['experiment']['name']
    
    # 如果启用自动命名，添加时间戳
    if config['experiment'].get('auto_naming', True):
        timestamp_format = config['experiment'].get('timestamp_format', "%Y%m%d_%H%M%S")
        timestamp = datetime.now().strftime(timestamp_format)
        exp_name = f"{exp_name}_{timestamp}"
    else:
        # 检查是否已存在相同名称的实验目录，如果存在则添加序号
        base_dir = os.path.join(config['experiment']['output_dir'], exp_name)
        if os.path.exists(base_dir):
            i = 1
            while os.path.exists(os.path.join(config['experiment']['output_dir'], f"{exp_name}_{i}")):
                i += 1
            exp_name = f"{exp_name}_{i}"
    
    exp_dir = os.path.join(config['experiment']['output_dir'], exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 保存配置
    config_save_path = os.path.join(exp_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return exp_dir


def setup_training(args, config, local_rank):
    """设置训练环境: 设备、模型、优化器、数据加载器等"""
    # 1. 基本设置
    gpu_config = config.get('gpu', {})
    use_gpu = gpu_config.get('use_gpu', True) and torch.cuda.is_available() and not args.no_cuda
    mixed_precision = gpu_config.get('mixed_precision', False) and use_gpu
    
    # 2. 分布式训练设置
    distributed = args.distributed or gpu_config.get('distributed', False)
    backend = gpu_config.get('backend', 'nccl')
    find_unused_parameters = gpu_config.get('find_unused_parameters', False)
    
    # 处理分布式训练初始化
    if use_gpu and distributed:
        # 命令行指定的分布式训练 (通过 torch.distributed.launch)
        if local_rank != -1:
            device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(local_rank)
            
            # 初始化进程组（如果尚未初始化）
            if not dist.is_initialized():
                dist.init_process_group(backend=backend)
            
            world_size = dist.get_world_size()
            local_world_size = 1  # 单个节点内的进程数
        
        # 单节点多GPU DDP模式 (不通过 torch.distributed.launch)
        elif torch.cuda.device_count() > 1:
            world_size = torch.cuda.device_count()
            local_world_size = world_size
            
            # 设置临时端口进行初始化
            import random
            port = random.randint(10000, 20000)
            
            # 设置环境变量
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = str(port)
            os.environ['WORLD_SIZE'] = str(world_size)
            
            # 启动多个进程
            mp.spawn(
                distributed_worker,
                args=(world_size, config, args),
                nprocs=world_size,
                join=True
            )
            
            # 进程已分叉，当前进程退出
            return None
        
        # 单GPU模式
        else:
            device = torch.device('cuda:0')
            world_size = 1
            local_world_size = 1
            distributed = False
    
    # 非分布式模式
    elif use_gpu:
        device = torch.device('cuda:0')
        # 如果配置了特定GPU设备，则设置可见设备
        if 'device_ids' in gpu_config and gpu_config['device_ids']:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_config['device_ids']))
        world_size = 1
        local_world_size = 1
        distributed = False
    else:
        device = torch.device('cpu')
        world_size = 1
        local_world_size = 1
        distributed = False
    
    # 3. 模型创建
    model = UnderwaterEnhanceNet(
        base_channels=config['model']['base_channels'],
        levels=config['model']['levels'],
        heads=config['model']['heads'],
        bottleneck_blocks=config['model']['bottleneck_blocks']
    )
    model = model.to(device)
    
    # 3.1 SyncBatchNorm (如果使用多GPU)
    if use_gpu and gpu_config.get('sync_bn', False) and world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # 4. 损失函数
    teacher_student_loss = TeacherStudentLoss(
        feat_weight=config['loss']['teacher_student']['feat_weight'],
        attn_weight=config['loss']['teacher_student']['attn_weight']
    )
    criterion = TotalLoss(
        lambda_img=config['loss']['lambda_img'],
        lambda_ssim=config['loss']['lambda_ssim'],
        lambda_perc=config['loss']['lambda_perc'],
        lambda_fft=config['loss']['lambda_fft'],
        lambda_grad=config['loss']['lambda_grad'],
        lambda_depth=config['loss']['lambda_depth'],
        lambda_smooth=config['loss']['lambda_smooth'],
        teacher_student_loss=teacher_student_loss
    )
    criterion = criterion.to(device)
    
    # 5. 优化器
    optimizer_name = config['optimizer'].get('name', 'adamw').lower()
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['optimizer']['lr'],
            betas=(config['optimizer'].get('beta1', 0.9), config['optimizer'].get('beta2', 0.999)),
            weight_decay=config['optimizer'].get('weight_decay', 0)
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer'].get('weight_decay', 0.01),
            betas=(config['optimizer'].get('beta1', 0.9), config['optimizer'].get('beta2', 0.999))
        )
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['optimizer']['lr'],
            momentum=config['optimizer'].get('momentum', 0.9),
            weight_decay=config['optimizer'].get('weight_decay', 0.0001),
            nesterov=config['optimizer'].get('nesterov', False)
        )
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    # 6. 学习率调度器
    scheduler = get_scheduler(
        config['scheduler']['name'],
        optimizer=optimizer,
        num_epochs=config['train']['epochs'],
        warmup_epochs=config['scheduler'].get('warmup_epochs', 0),
        min_lr=config['scheduler'].get('min_lr', 0),
        milestones=config['scheduler'].get('milestones', []),
        gamma=config['scheduler'].get('gamma', 0.1),
        patience=config['scheduler'].get('patience', 5),
        factor=config['scheduler'].get('factor', 0.5)
    )
    
    # 7. 混合精度设置
    scaler = None
    if mixed_precision:
        try:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
        except ImportError:
            print("警告：混合精度训练需要PyTorch 1.6+，已禁用混合精度训练。")
            mixed_precision = False
    
    # 8. 分布式封装
    if distributed and (local_rank != -1 or world_size > 1):
        # 使用DDP包装模型
        model = DDP(
            model, 
            device_ids=[local_rank if local_rank != -1 else 0],
            output_device=local_rank if local_rank != -1 else 0,
            find_unused_parameters=find_unused_parameters
        )
    elif use_gpu and torch.cuda.device_count() > 1 and not distributed:
        # 如果无法使用DDP但有多个GPU，使用DataParallel (不推荐)
        print("警告: 使用DataParallel进行多GPU训练，这比DDP效率低。建议使用 --distributed 参数启用DDP。")
        device_ids = gpu_config.get('device_ids', None)
        model = nn.DataParallel(model, device_ids=device_ids)
    
    # 9. 创建实验目录
    exp_dir = setup_experiment_dir(config)
    
    # 10. 设置日志
    logger, tb_writer, csv_path, debug_logger = setup_logger(
        exp_dir,
        log_file="train.log",
        metrics_file="metrics.csv"
    )
    metric_logger = MetricLogger(logger, tb_writer, csv_path)
    
    # 11. 记录训练配置
    config_text = yaml.dump(config, default_flow_style=False)
    metric_logger.log_text('config', config_text)
    
    # 记录GPU信息
    if use_gpu:
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_info.append(f"GPU {i}: {gpu_name}")
        gpu_info_str = "\n".join(gpu_info)
        metric_logger.log_text('gpu_info', gpu_info_str)
        logger.info(f"使用GPU: {gpu_info_str}")
        
        if distributed:
            logger.info(f"分布式训练已启用，使用 {backend} 后端, 世界大小: {world_size}")
            if gpu_config.get('sync_bn', False):
                logger.info("SyncBatchNorm 已启用")
    else:
        logger.info("使用CPU进行训练")
    
    if mixed_precision:
        logger.info("启用混合精度训练")
    
    return {
        'model': model,
        'criterion': criterion,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'device': device,
        'exp_dir': exp_dir,
        'logger': logger,
        'metric_logger': metric_logger,
        'scaler': scaler,
        'mixed_precision': mixed_precision,
        'world_size': world_size,
        'local_world_size': local_world_size,
        'distributed': distributed,
        'local_rank': local_rank,
        'debug_logger': debug_logger
    }


def prepare_data(config, args):
    """
    准备训练和验证数据加载器，支持文件夹和LMDB两种格式
    现在统一使用多退化数据集逻辑（可用于单退化或多退化情况）
    
    Args:
        config: 配置字典
        args: 命令行参数
        
    Returns:
        dict: 包含训练和验证数据加载器的字典
    """
    # 获取配置
    data_format = config['data'].get('format', 'folder')
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    resolution = config['data']['resolution']
    
    # 处理分辨率配置
    if isinstance(resolution, list):
        height, width = resolution
    else:
        height = width = resolution
        
    # 设置数据增强
    augment_prob = config['data'].get('augment_prob', 0.5)
    augmentations = config['data'].get('augmentations', {})
    
    # 获取数据集类型 (现在只作为标志使用，实际都用MultiDegradationDataset)
    dataset_type = config['data'].get('dataset_type', 'standard')
    
    # 创建数据集实例
    if data_format.lower() == 'folder':
        # 使用文件夹格式
        train_root = config['data']['train_root']
        val_root = config['data']['val_root']
        
        # 获取子文件夹名称
        folder_structure = config['data'].get('folder_structure', {})
        gt_folder = folder_structure.get('gt', 'gt')
        depth_folder = folder_structure.get('depth', 'depth')
        
        # 导入多退化数据集类
        from data.multi_deg_dataset import MultiDegradationDataset
        
        # 获取退化文件夹列表
        degradation_folders = config['data'].get('degradation_folders', ['raw'])
        
        # 确保至少有一个退化文件夹
        if not degradation_folders:
            degradation_folders = ['raw']  # 默认使用"raw"作为退化文件夹
        
        # 打印信息
        print(f"数据集类型: {dataset_type}")
        print(f"使用的退化文件夹: {degradation_folders}")
            
        # 实例化训练集和验证集 (都使用MultiDegradationDataset)
        train_dataset = MultiDegradationDataset(
            raw_folders=[os.path.join(train_root, folder) for folder in degradation_folders],
            gt_folder=os.path.join(train_root, gt_folder),
            depth_folder=os.path.join(train_root, depth_folder),
            patch_size=(height, width),
            augment=augment_prob > 0
        )
        
        val_dataset = MultiDegradationDataset(
            raw_folders=[os.path.join(val_root, folder) for folder in degradation_folders],
            gt_folder=os.path.join(val_root, gt_folder),
            depth_folder=os.path.join(val_root, depth_folder),
            patch_size=(height, width),
            augment=False  # 验证时不使用增强
        )
        
    elif data_format.lower() == 'lmdb':
        # LMDB格式暂时只支持标准的单输入方式
        # 未来可考虑实现MultiDegradationLMDBDataset
        
        train_lmdb = config['data']['lmdb_paths']['train']
        val_lmdb = config['data']['lmdb_paths']['val']
        
        # 实例化训练集和验证集
        from data.lmdb_dataset import UnderwaterLMDBDataset
        train_dataset = UnderwaterLMDBDataset(
            lmdb_path=train_lmdb,
            resolution=(height, width),
            augment_prob=augment_prob,
            augmentations=augmentations,
            mode='train'
        )
        
        val_dataset = UnderwaterLMDBDataset(
            lmdb_path=val_lmdb,
            resolution=(height, width),
            augment_prob=0.0,  # 验证时不使用增强
            mode='val'
        )
        
        print("警告: LMDB格式目前仅支持单退化输入")
    else:
        raise ValueError(f"不支持的数据格式: {data_format}, 应为 'folder' 或 'lmdb'")
    
    # 分布式采样器
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=val_sampler
    )
    
    return {
        'train_loader': train_loader, 
        'val_loader': val_loader, 
        'train_sampler': train_sampler, 
        'val_sampler': val_sampler
    }


def extract_visualization_data(model_outputs, depth_feats, batch_size=4):
    """
    从模型输出提取可视化所需的数据
    
    Returns:
        dict: 包含各种可视化数据的字典
    """
    enhanced, pred_gate, student_feats, teacher_feats = model_outputs
    
    # 限制可视化样本数量
    n_samples = min(batch_size, enhanced.size(0))
    
    # 1. 基本输出：增强图、深度门控
    data = {
        'enhanced': enhanced[:n_samples].detach().cpu(),
        'depth_gate': pred_gate[:n_samples].detach().cpu(),
    }
    
    # 2. 编码器特征
    if student_feats:
        data['student_feats'] = [feat[:n_samples].detach().cpu() 
                                 for feat in student_feats]
    
    if teacher_feats:
        data['teacher_feats'] = [feat[:n_samples].detach().cpu() if feat is not None else None
                                 for feat in teacher_feats]
        
    # 3. 深度特征和融合权重
    if hasattr(model, 'module'):
        decoder = model.module.decoder
    else:
        decoder = model.decoder
        
    # 提取自适应深度融合权重 (从decoder获取，如果实现了hook或属性)
    # 这里暂时是占位，具体获取方式取决于你的模型实现
    # 在实际训练循环中，可能需要通过hook或修改模型来获取这些中间数据
    
    return data


def train_epoch(train_loader, model, criterion, optimizer, device, metric_logger, 
               epoch, config, scaler=None, mixed_precision=False):
    """
    训练一个epoch的函数
    使用统一的多输入处理逻辑，同时支持单输入和多输入场景
    
    Args:
        train_loader: 训练数据加载器
        model: 模型
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        metric_logger: 指标记录器
        epoch: 当前epoch
        config: 配置字典
        scaler: 混合精度训练的梯度缩放器
        mixed_precision: 是否使用混合精度训练
        
    Returns:
        epoch_loss: 本轮epoch的平均损失
    """
    model.train()
    metric_logger.reset()
    
    epoch_loss = 0
    epoch_metrics = defaultdict(float)
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}")
    
    # 可视化频率设置
    vis_interval = config['train'].get('vis_interval', 100)
    param_vis_interval = config['train'].get('param_vis_interval', 500)
    
    for i, batch in enumerate(progress_bar):
        # 解包数据 (统一使用多输入字典格式)
        if isinstance(batch, dict):
            # 获取数据
            raw_imgs = batch['raw_imgs'].to(device)  # [B,N,3,H,W] 或者 [B,3,H,W]
            depth_gt = batch['depth'].to(device) if 'depth' in batch else None
            gt = batch['gt'].to(device) if 'gt' in batch else None
            
            # 确保raw_imgs始终是[B,N,3,H,W]格式
            if len(raw_imgs.shape) == 4:  # [B,3,H,W] -> [B,1,3,H,W]
                raw_imgs = raw_imgs.unsqueeze(1)
                
            # 获取批次大小和退化图像数量
            B, N = raw_imgs.shape[:2]
        else:
            # 兼容旧的元组返回格式 (raw, depth_gt, gt)
            raw, depth_gt, gt = batch[:3]
            raw_imgs = raw.unsqueeze(1).to(device)  # [B,3,H,W] -> [B,1,3,H,W]
            if depth_gt is not None:
                depth_gt = depth_gt.to(device)
            if gt is not None:
                gt = gt.to(device)
            
            B, N = raw_imgs.shape[:2]
        
        # 零梯度
        optimizer.zero_grad()
        
        # 混合精度训练
        if mixed_precision and scaler is not None:
            with torch.cuda.amp.autocast():
                # 使用多输入前向传播
                all_losses = []
                
                # 按批次逐个处理
                for b in range(B):
                    # 提取当前批次的所有退化图像 [N,3,H,W]
                    raw_batch_b = raw_imgs[b]
                    depth_gt_b = depth_gt[b:b+1] if depth_gt is not None else None
                    gt_b = gt[b:b+1] if gt is not None else None
                    
                    # 多退化前向传播
                    outputs = model.multi_forward(raw_batch_b, depth_gt_b, gt_b)
                    
                    # 计算损失 (对每个退化级别单独计算再取平均)
                    enhanced = outputs['outputs']  # [N,3,H,W]
                    pred_gate = outputs['pred_gates']  # [N,1,H,W]
                    student_feats = outputs['student_feats']  # N个特征列表
                    teacher_feats = outputs['teacher_feats']  # 1个教师特征列表
                    
                    # 对每个退化图像计算损失
                    for n in range(N):
                        # 计算单个退化图像的损失
                        loss_n = criterion(
                            enhanced[n:n+1], 
                            gt_b, 
                            pred_gate[n:n+1], 
                            depth_gt_b,
                            student_feats[n], 
                            teacher_feats,
                            None, None
                        )
                        all_losses.append(loss_n)
                
                # 计算所有损失的平均值
                loss = torch.stack(all_losses).mean()
            
            # 缩放损失并反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 常规训练
            # 使用多输入前向传播
            all_losses = []
            
            # 按批次逐个处理
            for b in range(B):
                # 提取当前批次的所有退化图像 [N,3,H,W]
                raw_batch_b = raw_imgs[b]
                depth_gt_b = depth_gt[b:b+1] if depth_gt is not None else None
                gt_b = gt[b:b+1] if gt is not None else None
                
                # 多退化前向传播
                outputs = model.multi_forward(raw_batch_b, depth_gt_b, gt_b)
                
                # 计算损失 (对每个退化级别单独计算再取平均)
                enhanced = outputs['outputs']  # [N,3,H,W]
                pred_gate = outputs['pred_gates']  # [N,1,H,W]
                student_feats = outputs['student_feats']  # N个特征列表
                teacher_feats = outputs['teacher_feats']  # 1个教师特征列表
                
                # 对每个退化图像计算损失
                for n in range(N):
                    # 计算单个退化图像的损失
                    loss_n = criterion(
                        enhanced[n:n+1], 
                        gt_b, 
                        pred_gate[n:n+1], 
                        depth_gt_b,
                        student_feats[n], 
                        teacher_feats,
                        None, None
                    )
                    all_losses.append(loss_n)
            
            # 计算所有损失的平均值
            loss = torch.stack(all_losses).mean()
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
        
        # 记录损失
        current_loss = loss.item()
        epoch_loss += current_loss
        
        # 更新进度条
        progress_bar.set_postfix({"Loss": f"{current_loss:.4f}"})
        
        # 记录到Logger
        metrics = {"loss": current_loss}
        
        # 从损失函数获取详细的损失组件
        # 如果criterion有get_latest_losses方法，则获取详细损失
        if hasattr(criterion, 'get_latest_losses'):
            loss_components = criterion.get_latest_losses()
            
            # 将所有损失组件添加到metrics中，重命名确保分类显示
            for loss_name, loss_value in loss_components.items():
                if loss_name != 'total_loss':  # 避免重复记录总损失
                    # 给每个损失名称加上前缀，确保它们会被分组显示
                    if not (loss_name.startswith('loss_') or loss_name.endswith('_loss')):
                        metrics[f"loss_{loss_name}"] = loss_value
                    else:
                        metrics[loss_name] = loss_value
                    
            # 特别突出深度相关损失，方便监控深度模块训练情况
            if 'depth_total_loss' in loss_components:
                progress_bar.set_postfix({
                    "Loss": f"{current_loss:.4f}", 
                    "Depth": f"{loss_components['depth_total_loss']:.4f}"
                })
        
        metric_logger.log_metrics(metrics, prefix="train")
        
        # 周期性地可视化各种数据
        if i % vis_interval == 0:
            # 多退化可视化
            # 选择第一个批次的数据进行可视化
            # 确保变量定义和局部变量可用
            try:
                metric_logger.logger.info(f"开始记录可视化数据，步骤: {i}")
                if 'outputs' in locals():
                    # 已经有outputs变量
                    vis_data = {
                        'enhanced': enhanced[:min(4, N)].detach().cpu(),
                        'depth_gate': pred_gate[:min(4, N)].detach().cpu(),
                    }
                    
                    # 记录学生特征
                    if student_feats and isinstance(student_feats, list):
                        vis_data['student_feats'] = []
                        for s_idx, s_feat_list in enumerate(student_feats[:min(4, len(student_feats))]):
                            # 只取每个特征列表的第一个(最低级)特征
                            if s_feat_list and len(s_feat_list) > 0:
                                vis_data['student_feats'].append(s_feat_list[0].detach().cpu())
                
                else:
                    # 还没有outputs变量，使用临时前向传播获取可视化数据
                    with torch.no_grad():
                        temp_raw = raw_imgs[0]
                        temp_depth = depth_gt[0:1] if depth_gt is not None else None
                        temp_gt = gt[0:1] if gt is not None else None
                        
                        temp_outputs = model.multi_forward(temp_raw, temp_depth, temp_gt)
                        
                        vis_data = {
                            'enhanced': temp_outputs['outputs'][:min(4, N)].detach().cpu(),
                            'depth_gate': temp_outputs['pred_gates'][:min(4, N)].detach().cpu(),
                        }
                
                # 记录图像
                if 'enhanced' in vis_data:
                    metric_logger.logger.info(f"记录增强图像，形状: {vis_data['enhanced'].shape}")
                    metric_logger.log_images_grid("train/enhanced", vis_data['enhanced'])
                    
                if 'depth_gate' in vis_data:
                    metric_logger.logger.info(f"记录深度门控图，形状: {vis_data['depth_gate'].shape}")
                    metric_logger.log_image("train/depth_gate", vis_data['depth_gate'])
                
                # 记录特征图
                if 'student_feats' in vis_data:
                    if isinstance(vis_data['student_feats'], list):
                        for j, feat in enumerate(vis_data['student_feats']):
                            if feat is not None:
                                metric_logger.logger.info(f"记录学生特征图 {j}，形状: {feat.shape}")
                                metric_logger.log_feature_maps(f"train/student_feat_level{j}", feat)
                
                # 如果可能，记录自适应深度融合权重
                # 这部分需要根据实际实现调整
                if hasattr(model, 'module'):
                    module = model.module
                else:
                    module = model
                    
                # 尝试从模型获取深度融合权重 (假设模型有相应属性或方法)
                if hasattr(module, 'get_depth_fusion_weights'):
                    weights = module.get_depth_fusion_weights()
                    if weights is not None:
                        metric_logger.logger.info(f"记录深度融合权重，形状: {weights.shape}")
                        metric_logger.log_depth_fusion_weights(
                            "train/depth_fusion_weights", weights)
                
                metric_logger.logger.info("可视化数据记录完成")
            except Exception as e:
                metric_logger.logger.error(f"记录可视化数据时发生错误: {str(e)}")
                import traceback
                metric_logger.logger.error(traceback.format_exc())
        
        # 周期性记录模型参数直方图
        if i % param_vis_interval == 0:
            metric_logger.log_model_parameters(model)
        
    # 计算平均损失
    epoch_loss /= len(train_loader)
    
    return epoch_loss


def validate(val_loader, model, criterion, device, metric_logger, epoch, config, mixed_precision=False):
    """
    验证函数 - 统一使用多输入处理逻辑
    
    Args:
        val_loader: 验证数据加载器
        model: 模型
        criterion: 损失函数
        device: 计算设备
        metric_logger: 指标记录器
        epoch: 当前epoch
        config: 配置字典
        mixed_precision: 是否使用混合精度
        
    Returns:
        val_loss: 验证集平均损失
        val_psnr: 验证集平均PSNR (主要衡量指标)
    """
    model.eval()
    val_loss = 0
    val_psnr = 0
    val_ssim = 0
    
    # 获取可视化配置
    vis_config = config.get('visualization', {})
    val_imgs_config = vis_config.get('val_images', {})
    save_val_imgs = val_imgs_config.get('save', True)
    max_samples = val_imgs_config.get('max_samples', 8)
    save_comparison = val_imgs_config.get('save_comparison', True)
    save_metrics = val_imgs_config.get('save_metrics', True)
    
    # 准备保存验证图像的目录
    if save_val_imgs:
        val_img_dir = os.path.join(metric_logger.tb_writer.log_dir, f'val_images_epoch{epoch+1}')
        os.makedirs(val_img_dir, exist_ok=True)
    
    # 用于计算平均指标的计数器
    total_samples = 0
    all_metrics = defaultdict(float)
    
    # 多退化数据集指标跟踪
    metrics_per_level = None
    
    # 存储样本结果用于可视化
    results_for_vis = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Validating Epoch {epoch+1}")
        
        for i, batch in enumerate(progress_bar):
            # 解包数据 (统一使用多输入字典格式)
            if isinstance(batch, dict):
                # 获取数据
                raw_imgs = batch['raw_imgs'].to(device)  # [B,N,3,H,W] 或者 [B,3,H,W]
                depth_gt = batch['depth'].to(device) if 'depth' in batch else None
                gt = batch['gt'].to(device) if 'gt' in batch else None
                
                # 确保raw_imgs始终是[B,N,3,H,W]格式
                if len(raw_imgs.shape) == 4:  # [B,3,H,W] -> [B,1,3,H,W]
                    raw_imgs = raw_imgs.unsqueeze(1)
                    
                # 获取批次大小和退化图像数量
                B, N = raw_imgs.shape[:2]
            else:
                # 兼容旧的元组返回格式 (raw, depth_gt, gt)
                raw, depth_gt, gt = batch[:3]
                raw_imgs = raw.unsqueeze(1).to(device)  # [B,3,H,W] -> [B,1,3,H,W]
                if depth_gt is not None:
                    depth_gt = depth_gt.to(device)
                if gt is not None:
                    gt = gt.to(device)
                
                B, N = raw_imgs.shape[:2]
                
            # 初始化每个退化级别的指标跟踪
            if metrics_per_level is None and N > 1:
                metrics_per_level = [{
                    'loss': 0,
                    'psnr': 0,
                    'ssim': 0,
                    'count': 0
                } for _ in range(N)]
            
            # 多退化数据集验证
            batch_loss = 0
            batch_psnrs = [0] * N
            batch_ssims = [0] * N
            
            for b in range(B):
                # 提取单个样本的所有退化图像
                raw_batch_b = raw_imgs[b]  # [N,3,H,W]
                depth_gt_b = depth_gt[b:b+1] if depth_gt is not None else None  # [1,1,H,W]
                gt_b = gt[b:b+1] if gt is not None else None  # [1,3,H,W]
                
                # 多退化前向传播
                if mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = model.multi_forward(raw_batch_b, depth_gt_b, gt_b)
                else:
                    outputs = model.multi_forward(raw_batch_b, depth_gt_b, gt_b)
                
                # 获取输出
                enhanced = outputs['outputs']  # [N,3,H,W]
                pred_gate = outputs['pred_gates']  # [N,1,H,W]
                student_feats = outputs['student_feats']
                teacher_feats = outputs['teacher_feats']
                
                # 计算每个退化级别的损失和指标
                sample_losses = []
                
                for n in range(N):
                    # 计算损失
                    loss_n = criterion(
                        enhanced[n:n+1],  # [1,3,H,W]
                        gt_b,  # [1,3,H,W]
                        pred_gate[n:n+1],  # [1,1,H,W]
                        depth_gt_b,  # [1,1,H,W]
                        student_feats[n],
                        teacher_feats,
                        None, None
                    )
                    sample_losses.append(loss_n.item())
                    
                    # 获取详细损失组件
                    if hasattr(criterion, 'get_latest_losses'):
                        loss_components = criterion.get_latest_losses()
                        if 'depth_total_loss' in loss_components and metrics_per_level is not None:
                            # 记录深度损失到对应级别的指标
                            if 'depth_loss' not in metrics_per_level[n]:
                                metrics_per_level[n]['depth_loss'] = 0.0
                            metrics_per_level[n]['depth_loss'] += loss_components['depth_total_loss']
                    
                    # 计算PSNR和SSIM
                    if gt_b is not None:
                        from utils.metrics import calculate_psnr, calculate_ssim
                        
                        # 确保图像范围正确
                        if enhanced[n:n+1].min() < 0:
                            enhanced_eval = (enhanced[n:n+1] + 1) / 2
                            gt_eval = (gt_b + 1) / 2
                        else:
                            enhanced_eval = enhanced[n:n+1]
                            gt_eval = gt_b
                        
                        # 计算指标
                        psnr_n = calculate_psnr(enhanced_eval, gt_eval)
                        ssim_n = calculate_ssim(enhanced_eval, gt_eval)
                        
                        # 累加到对应退化级别的指标
                        if metrics_per_level is not None:
                            metrics_per_level[n]['psnr'] += psnr_n
                            metrics_per_level[n]['ssim'] += ssim_n
                            metrics_per_level[n]['loss'] += sample_losses[n]
                            metrics_per_level[n]['count'] += 1
                        
                        # 同时累加到批次平均
                        batch_psnrs[n] += psnr_n
                        batch_ssims[n] += ssim_n
                        
                        # 收集用于可视化的结果
                        if len(results_for_vis) < max_samples:
                            if b == 0:  # 只收集第一个批次的样本
                                for level in range(N):
                                    if len(results_for_vis) < max_samples:
                                        results_for_vis.append({
                                            'input': raw_batch_b[level:level+1].cpu(),
                                            'output': enhanced[level:level+1].cpu(),
                                            'gt': gt_b.cpu() if gt_b is not None else None,
                                            'depth': depth_gt_b.cpu() if depth_gt_b is not None else None,
                                            'depth_pred': pred_gate[level:level+1].cpu() if pred_gate is not None else None,
                                            'psnr': psnr_n if level == n else 0,
                                            'ssim': ssim_n if level == n else 0,
                                            'level': level
                                        })
                
                # 计算样本的平均损失
                batch_loss += sum(sample_losses) / len(sample_losses)
            
            # 计算批次平均
            batch_loss /= B
            batch_psnrs = [p / B for p in batch_psnrs]
            batch_ssims = [s / B for s in batch_ssims]
            
            # 更新进度条 (显示第一个退化级别的指标)
            progress_bar.set_postfix({
                "Loss": f"{batch_loss:.4f}",
                "PSNR_0": f"{batch_psnrs[0]:.2f}",
                "SSIM_0": f"{batch_ssims[0]:.4f}"
            })
            
            # 累加到整体验证
            val_loss += batch_loss * B
            total_samples += B
    
    # 计算平均指标
    val_loss /= total_samples
    
    # 计算所有级别的平均PSNR和SSIM
    if metrics_per_level:
        all_psnrs = []
        all_ssims = []
        
        # 处理每个退化级别的指标 
        for level_idx, level_metrics in enumerate(metrics_per_level):
            if level_metrics['count'] > 0:
                # 计算该级别的平均指标
                avg_psnr = level_metrics['psnr'] / level_metrics['count']
                avg_ssim = level_metrics['ssim'] / level_metrics['count']
                avg_loss = level_metrics['loss'] / level_metrics['count']
                
                # 添加到全局平均
                all_psnrs.append(avg_psnr)
                all_ssims.append(avg_ssim)
                
                # 为该级别记录详细指标
                level_details = {
                    "loss": avg_loss,
                    "psnr": avg_psnr,
                    "ssim": avg_ssim
                }
                
                # 添加深度损失(如果有)
                if 'depth_loss' in level_metrics:
                    avg_depth_loss = level_metrics['depth_loss'] / level_metrics['count']
                    level_details["depth_loss"] = avg_depth_loss
                
                # 记录该级别的指标
                metric_logger.log_metrics(level_details, prefix=f"val/level{level_idx}")
        
        # 计算所有级别的平均值作为全局指标
        if all_psnrs:
            val_psnr = sum(all_psnrs) / len(all_psnrs)
        if all_ssims:
            val_ssim = sum(all_ssims) / len(all_ssims)
            
        # 计算平均深度损失(如果有)
        val_depth_loss = 0.0
        depth_loss_count = 0
        for m in metrics_per_level:
            if m['count'] > 0 and 'depth_loss' in m:
                val_depth_loss += m['depth_loss'] / m['count']
                depth_loss_count += 1
                
        if depth_loss_count > 0:
            val_depth_loss /= depth_loss_count
    
    # 将总体验证指标记录到TensorBoard
    metrics = {
        "loss": val_loss,
        "metrics_psnr": val_psnr,  # 添加前缀确保分组
        "metrics_ssim": val_ssim,  # 添加前缀确保分组
    }
    
    # 添加深度损失(如果有)
    if 'val_depth_loss' in locals() and depth_loss_count > 0:
        metrics["loss_depth"] = val_depth_loss  # 添加前缀确保被归类为损失
    
    metric_logger.log_metrics(metrics, prefix="val")
    
    # 可视化验证结果
    if save_val_imgs and results_for_vis:
        # 1. 保存到TensorBoard
        for i, result in enumerate(results_for_vis):
            # 创建对比图
            images = [result['input'][0], result['output'][0]]
            titles = ['Input', 'Enhanced']
            
            if result['gt'] is not None:
                images.append(result['gt'][0])
                titles.append('Ground Truth')
                
            if save_metrics:
                title_suffix = f"PSNR: {result['psnr']:.2f}, SSIM: {result['ssim']:.4f}"
                if 'level' in result:
                    titles[1] = f"Enhanced (Level {result['level']}, {title_suffix})"
                else:
                    titles[1] = f"Enhanced ({title_suffix})"
                
            metric_logger.log_image_comparison(
                f"val/comparison_sample{i}", 
                images, 
                titles
            )
            
            # 如果有深度预测，也可视化
            if result.get('depth_pred') is not None:
                metric_logger.log_image(
                    f"val/depth_pred_sample{i}",
                    result['depth_pred'][0]
                )
                
            # 如果有真实深度，对比预测和真实深度
            if result.get('depth') is not None and result.get('depth_pred') is not None:
                depth_images = [result['depth'][0], result['depth_pred'][0]]
                depth_titles = ['GT Depth', 'Predicted Depth']
                metric_logger.log_image_comparison(
                    f"val/depth_comparison_sample{i}",
                    depth_images,
                    depth_titles
                )
        
        # 2. 保存到文件系统用于后续查看
        if save_comparison:
            import torchvision.utils as vutils
            import matplotlib.pyplot as plt
            
            for i, result in enumerate(results_for_vis):
                # 创建拼接图
                fig, axs = plt.subplots(1, 3 if result['gt'] is not None else 2, 
                                      figsize=(18 if result['gt'] is not None else 12, 6))
                
                # 输入图像
                input_img = result['input'][0].cpu().numpy()
                if input_img.shape[0] == 3:  # RGB
                    input_img = np.transpose(input_img, (1, 2, 0))
                    if input_img.max() > 1 or input_img.min() < 0:
                        input_img = np.clip((input_img + 1) / 2, 0, 1)
                    axs[0].imshow(input_img)
                else:  # 灰度
                    axs[0].imshow(input_img[0], cmap='gray')
                axs[0].set_title('Input')
                axs[0].axis('off')
                
                # 增强图像
                output_img = result['output'][0].cpu().numpy()
                if output_img.shape[0] == 3:  # RGB
                    output_img = np.transpose(output_img, (1, 2, 0))
                    if output_img.max() > 1 or output_img.min() < 0:
                        output_img = np.clip((output_img + 1) / 2, 0, 1)
                    axs[1].imshow(output_img)
                else:  # 灰度
                    axs[1].imshow(output_img[0], cmap='gray')
                    
                title = 'Enhanced'
                if save_metrics:
                    metric_text = f"PSNR: {result['psnr']:.2f}, SSIM: {result['ssim']:.4f}"
                    if 'level' in result:
                        title += f" (Level {result['level']})\n{metric_text}"
                    else:
                        title += f"\n{metric_text}"
                axs[1].set_title(title)
                axs[1].axis('off')
                
                # 真实图像 (如果有)
                if result['gt'] is not None:
                    gt_img = result['gt'][0].cpu().numpy()
                    if gt_img.shape[0] == 3:  # RGB
                        gt_img = np.transpose(gt_img, (1, 2, 0))
                        if gt_img.max() > 1 or gt_img.min() < 0:
                            gt_img = np.clip((gt_img + 1) / 2, 0, 1)
                        axs[2].imshow(gt_img)
                    else:  # 灰度
                        axs[2].imshow(gt_img[0], cmap='gray')
                    axs[2].set_title('Ground Truth')
                    axs[2].axis('off')
                
                plt.tight_layout()
                sample_filename = f'sample_{i}'
                if 'level' in result:
                    sample_filename += f'_level{result["level"]}'
                plt.savefig(os.path.join(val_img_dir, f'{sample_filename}.png'), dpi=150)
                plt.close(fig)
                
                # 保存深度预测 (如果有)
                if result.get('depth_pred') is not None:
                    plt.figure(figsize=(6, 6))
                    depth_pred = result['depth_pred'][0].cpu().numpy()
                    plt.imshow(depth_pred[0], cmap='viridis')
                    plt.title('Predicted Depth')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(val_img_dir, f'depth_pred_{sample_filename}.png'), dpi=150)
                    plt.close()
    
    return val_loss, val_psnr


def distributed_worker(rank, world_size, config, args):
    """
    用于torch.multiprocessing.spawn的工作进程函数，
    负责设置DDP环境并启动训练
    
    Args:
        rank (int): 当前进程的排名
        world_size (int): 总进程数
        config (dict): 配置字典
        args (argparse.Namespace): 命令行参数
    """
    try:
        # 将进程排名传递给原有的参数
        args.local_rank = rank
        args.distributed = True
        
        # 初始化进程组
        dist.init_process_group(
            backend=config['gpu'].get('backend', 'nccl'),
            world_size=world_size,
            rank=rank
        )
        
        # 执行训练主循环
        main_worker(config, args)
    except Exception as e:
        print(f"进程 {rank} 出错: {str(e)}")
        # 尝试获取详细的异常堆栈
        import traceback
        traceback.print_exc()
        # 确保所有进程终止
        dist.destroy_process_group()
        raise e


def main_worker(config, args):
    """
    主要训练工作函数
    """
    # ----- 初始化 -----
    # 本地排名 (用于分布式训练)
    local_rank = args.local_rank
    
    # 设置实验目录并保存配置
    exp_dir = setup_experiment_dir(config)
    
    # 准备训练/验证数据
    data_loaders = prepare_data(config, args)
    train_loader = data_loaders['train_loader']
    val_loader = data_loaders['val_loader']
    train_sampler = data_loaders.get('train_sampler')
    val_sampler = data_loaders.get('val_sampler')
    
    # 设置设备、模型、优化器等
    training_setup = setup_training(args, config, local_rank)
    device = training_setup['device']
    model = training_setup['model']
    criterion = training_setup['criterion']
    optimizer = training_setup['optimizer']
    scheduler = training_setup['scheduler']
    scaler = training_setup['scaler']
    mixed_precision = training_setup['mixed_precision']
    exp_dir = training_setup['exp_dir']
    logger = training_setup['logger']
    metric_logger = training_setup['metric_logger']
    debug_logger = training_setup['debug_logger']
    
    # 记录设备信息
    if torch.cuda.is_available():
        logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
    if mixed_precision:
        logger.info("启用混合精度训练")
    
    # 日志记录模型架构
    if config['visualization'].get('save_model_graph', False):
        try:
            # 使用一个示例输入记录模型图
            dummy_input = torch.zeros(1, 3, 64, 64).to(device)
            dummy_depth = torch.zeros(1, 1, 64, 64).to(device)
            dummy_gt = None  # 推理阶段不使用GT
            
            metric_logger.log_model_graph(model, (dummy_input, dummy_depth, dummy_gt))
            logger.info("模型架构已保存到TensorBoard")
        except Exception as e:
            logger.warning(f"记录模型图结构失败: {str(e)}")
    
    # ----- 断点续训处理 -----
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    start_epoch = 0
    best_psnr = 0.0
    best_loss = float('inf')
    
    # 检查是否从检查点继续训练
    if args.resume and os.path.exists(checkpoint_dir):
        start_epoch, best_metric = resume_from_checkpoint(
            checkpoint_dir, model, optimizer, scheduler, device, scaler
        )
        best_psnr = best_metric
        logger.info(f"从epoch {start_epoch}继续训练，当前最佳PSNR: {best_psnr:.4f}")

    # 输出调试信息到debug.log而非控制台
    debug_logger.info("=== 模型架构信息 ===")
    debug_logger.info(f"基础通道数: {config['model']['base_channels']}")
    debug_logger.info(f"编码器层级数: {config['model']['levels']}")
    debug_logger.info(f"注意力头数: {config['model']['heads']}")
    debug_logger.info(f"瓶颈Transformer块数: {config['model']['bottleneck_blocks']}")
    
    # 添加UnderwaterEnhanceNet层级输出到debug文件中
    debug_logger.info("\n=== UnderwaterEnhanceNet特征层级 ===")
    debug_logger.info(f"SFE: 输入3通道 -> 输出{config['model']['base_channels']}通道")
    debug_logger.info(f"编码器: {config['model']['levels']}级下采样，每级{config['model']['base_channels']}通道")
    debug_logger.info(f"Bottleneck: {config['model']['bottleneck_blocks']}个Transformer块")
    debug_logger.info(f"解码器: {config['model']['levels']-1}级上采样，PixelShuffle和自适应深度融合")
    
    # 仅评估模式（如果启用）
    if args.eval_only and val_loader is not None:
        logger.info("仅评估模式")
        validate(
            val_loader, model, criterion, device, metric_logger, 
            0, config, mixed_precision
        )
        return

    # ----- 训练循环 -----
    logger.info(f"开始训练... 总epoch数: {config['train']['epochs']}, "
               f"批次大小: {config['data']['batch_size']}, "
               f"有效批次大小: {config['data']['batch_size']}")
    
    # 训练参数
    epochs = config['train']['epochs']
    save_interval = config.get('train', {}).get('save_interval', 10)
    save_best = config.get('train', {}).get('save_best', True)
    val_interval = config.get('train', {}).get('val_interval', 1)
    
    # 开始训练循环
    for epoch in range(start_epoch, epochs):
        # 设置分布式采样器的epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # 训练一个epoch
        train_loss = train_epoch(
            train_loader, model, criterion, optimizer, device, 
            metric_logger, epoch, config, scaler, mixed_precision
        )
        
        # 验证
        if val_loader and (epoch + 1) % val_interval == 0:
            val_loss, val_psnr = validate(
                val_loader, model, criterion, device, metric_logger, 
                epoch, config, mixed_precision
            )
            logger.info(
                f"Epoch {epoch+1}/{config['train']['epochs']} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val PSNR: {val_psnr:.4f}"
            )
            
            # 检查是否最佳模型
            is_best = val_psnr > best_psnr
            if is_best:
                best_psnr = val_psnr
                logger.info(f"发现新的最佳模型，PSNR: {best_psnr:.4f}")
        else:
            # 如果没有验证集，使用训练损失作为标准
            is_best = train_loss < best_loss
            if is_best:
                best_loss = train_loss
            
            logger.info(
                f"Epoch {epoch+1}/{config['train']['epochs']} - "
                f"Train Loss: {train_loss:.4f}"
            )
        
        # 更新学习率
        if config['scheduler']['name'] == 'plateau':
            # ReduceLROnPlateau需要验证指标
            if val_loader:
                scheduler.step(val_psnr)  # 使用PSNR作为指标
            else:
                scheduler.step(train_loss)
        else:
            scheduler.step()
            
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        metric_logger.log_metrics({'lr': current_lr}, prefix='train')
        
        # 保存检查点 (仅主进程)
        if (local_rank <= 0 or local_rank == -1) and ((epoch + 1) % save_interval == 0 or is_best and save_best):
            checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 准备保存的状态
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_metric': best_psnr,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            
            # 添加混合精度状态 (如果使用)
            if mixed_precision and scaler is not None:
                checkpoint['scaler'] = scaler.state_dict()
                
            # 保存检查点
            save_checkpoint(checkpoint, is_best, checkpoint_dir, epoch=(epoch+1))
            
            # 如果是最佳模型，额外保存一个独立的模型权重文件，便于部署
            if is_best and save_best:
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), 
                         os.path.join(checkpoint_dir, 'best_model_weights.pth'))
    
    # 训练结束
    logger.info(f"训练完成! 最佳 PSNR: {best_psnr:.4f}")
    
    # 保存最终模型
    if local_rank <= 0 or local_rank == -1:
        checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), 
                 os.path.join(checkpoint_dir, 'final_model_weights.pth'))
        logger.info(f"最终模型已保存到 {os.path.join(checkpoint_dir, 'final_model_weights.pth')}")
    
    metric_logger.close()


def resume_from_checkpoint(checkpoint_dir: str,
                         model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer = None,
                         scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                         device = None,
                         scaler = None) -> tuple:
    """
    从检查点目录恢复训练，支持混合精度训练
    
    Args:
        checkpoint_dir (str): 包含检查点文件的目录
        model (nn.Module): 要加载状态的模型
        optimizer (Optimizer, optional): 要加载状态的优化器
        scheduler (_LRScheduler, optional): 要加载状态的调度器
        device: 加载设备
        scaler: 混合精度训练的梯度缩放器
        
    Returns:
        epoch (int): 恢复的起始epoch
        best_metric (float): 目前为止的最佳验证指标
    """
    # 查找最新的检查点
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    if not files:
        return 0, 0.0  # 没有找到检查点文件
        
    # 按修改时间排序，找到最新的检查点
    latest = max(files, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    if 'best' in latest:
        # 优先使用最佳模型而不是最新模型
        best_files = [f for f in files if 'best' in f]
        if best_files:
            latest = best_files[0]
    
    checkpoint_path = os.path.join(checkpoint_dir, latest)
    print(f"从检查点恢复: {checkpoint_path}")
    
    # 加载检查点
    map_location = device if device is not None else torch.device('cpu')
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # 加载模型权重
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    # 处理DataParallel/DDP前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.') and not hasattr(model, 'module'):
            # 移除'module.'前缀（如果模型不是DDP或DataParallel实例）
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            # 将优化器状态移动到正确的设备
            if device is not None:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
        except Exception as e:
            print(f"警告: 无法加载优化器状态: {e}")
    
    # 加载学习率调度器状态
    if scheduler is not None and 'scheduler' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler'])
        except Exception as e:
            print(f"警告: 无法加载学习率调度器状态: {e}")
            
    # 加载AMP梯度缩放器状态（用于混合精度训练）
    if scaler is not None and 'scaler' in checkpoint:
        try:
            scaler.load_state_dict(checkpoint['scaler'])
        except Exception as e:
            print(f"警告: 无法加载混合精度缩放器状态: {e}")
    
    epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', 0.0)
    
    return epoch, best_metric


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置本地排名（用于分布式训练）
    local_rank = args.local_rank
    
    # 如果是单GPU或CPU训练，直接调用worker函数
    # 如果是DDP训练（通过torch.distributed.launch），也直接调用
    if local_rank != -1 or not torch.cuda.is_available() or torch.cuda.device_count() <= 1 or args.no_cuda:
        main_worker(config, args)
    else:
        # 对于多GPU、非显式DDP训练，将在setup_training内部使用spawn创建多进程
        setup_training(args, config, local_rank)


if __name__ == "__main__":
    main()
