#!/usr/bin/env python3
"""
训练环境设置模块

负责设置训练环境：设备、模型、优化器、损失函数等
"""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from modules.model import UnderwaterEnhanceNet
from modules.loss_fn import TotalLoss
from utils.lr_scheduler import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP


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
    device, world_size, local_world_size = _setup_device_and_distributed(
        use_gpu, distributed, local_rank, backend, config, args
    )
    
    if device is None:  # 表示需要启动多进程
        return None
    
    # 3. 模型创建
    model = _create_model(config, device)
    
    # 4. 损失函数
    criterion = _create_criterion(config, device)
    
    # 5. 优化器
    optimizer, update_optimizer_fn = _create_optimizer(model, criterion, config)
    
    # 6. 学习率调度器
    scheduler = _create_scheduler(optimizer, config)
    
    # 7. 混合精度设置
    scaler = _setup_mixed_precision(mixed_precision, use_gpu)
    
    # 8. 分布式封装
    model = _setup_distributed_model(model, distributed, local_rank, world_size, 
                                    find_unused_parameters, use_gpu, gpu_config)
    
    return {
        'model': model,
        'criterion': criterion,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'device': device,
        'scaler': scaler,
        'mixed_precision': mixed_precision,
        'world_size': world_size,
        'local_world_size': local_world_size,
        'distributed': distributed,
        'local_rank': local_rank,
        'update_optimizer_fn': update_optimizer_fn
    }


def _setup_device_and_distributed(use_gpu, distributed, local_rank, backend, config, args):
    """设置设备和分布式训练"""
    gpu_config = config.get('gpu', {})  # 🔥 添加gpu_config定义
    
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
            
            # 启动多个进程
            from utils.distributed_utils import launch_distributed_training
            launch_distributed_training(world_size, config, args)
            
            # 进程已分叉，当前进程退出
            return None, None, None
        
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
    
    return device, world_size, local_world_size


def _create_model(config, device):
    """创建模型"""
    model_params = config.get('model', {})
    
    # 获取深度预处理器配置
    loss_config = config.get('loss', {})
    depth_config_from_loss = loss_config.get('depth_processing', {})
    depth_config_from_model = model_params.get('depth_processor_config', {})

    # 合并：以后者为优先级，覆盖前者的同名字段
    depth_config_combined = {**depth_config_from_loss, **depth_config_from_model}
    
    model = UnderwaterEnhanceNet(
        base_channels=model_params.get('base_channels', 48),
        levels=model_params.get('levels', 4),
        heads=model_params.get('heads', 8),
        bottleneck_blocks=model_params.get('bottleneck_blocks', 4),
        encoder_window_size=model_params.get('encoder_window_size', 8),
        bottleneck_window_size=model_params.get('bottleneck_window_size', 0),
        decoder_block_window_size=model_params.get('decoder_block_window_size', 4),
        depth_processor_config=depth_config_combined,
        save_attention_maps=config.get('visualization', {}).get('save_attention_maps', False)
    )
    
    model = model.to(device)
    
    # 🔧 应用交叉注意力配置
    cross_attention_config = config.get('loss', {}).get('cross_attention', {})
    if cross_attention_config.get('enable_saving', False):
        model.enable_attention_saving(True)
        print(f"✓ 启用注意力图保存功能")
    
    return model


def _create_criterion(config, device):
    """创建损失函数"""
    criterion = TotalLoss(
        lambda_smooth=config['loss']['lambda_smooth'],
        # fine-grained lambdas (manual weighting)
        lambda_img_l1=config['loss'].get('lambda_img_l1', 1.0),
        lambda_img_ssim=config['loss'].get('lambda_img_ssim', 1.0),
        lambda_img_perc=config['loss'].get('lambda_img_perc', 1.0),
        lambda_img_fft=config['loss'].get('lambda_img_fft', 1.0),
        lambda_img_grad=config['loss'].get('lambda_img_grad', 1.0),
        lambda_depth_decoder=config['loss'].get('lambda_depth_decoder', 1.0),
        lambda_depth_smooth=config['loss'].get('lambda_depth_smooth', 1.0),
        lambda_depth_rec=config['loss'].get('lambda_depth_rec', 1.0),
        lambda_cons=config['loss'].get('lambda_attncons', 1.0),
        
        # 🔥 CMCL相关参数
        lambda_cmcl=config['loss'].get('lambda_cmcl', 0.1),
        lambda_cmcl_var=config['loss'].get('lambda_cmcl_var', 1.0),
        lambda_cmcl_rgb=config['loss'].get('lambda_cmcl_rgb', 1.0),
        lambda_cmcl_depth=config['loss'].get('lambda_cmcl_depth', 1.0),
        cmcl_k_decay=config['loss'].get('cmcl_k_decay', 1.0),

        use_uncertainty_weighting=config['loss'].get('use_uncertainty_weighting', True),
        sigma_init=config['loss'].get('sigma_init', None),
        min_depth=config['loss']['depth_processing'].get('min_depth', 2000.0),
        max_depth=config['loss']['depth_processing'].get('max_depth', 65535.0)
    )
    
    return criterion.to(device)


def _create_optimizer(model, criterion, config):
    """创建优化器"""
    optimizer_name = config['optimizer'].get('name', 'adamw').lower()
    
    # 参数分组，将交叉注意力参数、物理参数和其他参数分开，应用不同的学习率
    base_params, attn_params, physics_params = [], [], []
    for name, p in model.named_parameters():
        if 'depth2rgb_attn' in name or 'rgb2depth_attn' in name:
            attn_params.append(p)
        elif 'physics_head' in name:
            physics_params.append(p)
        else:
            base_params.append(p)
    
    # 获取注意力模块和物理模块学习率缩放因子
    attn_lr_scale = config['optimizer'].get('attn_lr_scale', 0.1)
    physics_lr_scale = config['optimizer'].get('physics_lr_scale', 1.0)
    
    # 添加损失函数中的不确定性权重参数到优化器中
    uncertainty_params = []
    if hasattr(criterion, 'use_uncertainty_weighting') and criterion.use_uncertainty_weighting:
        # 收集所有不确定性权重参数
        for name, p in criterion.named_parameters():
            if 'log_var' in name:
                uncertainty_params.append(p)
        
        if uncertainty_params:
            print(f"将 {len(uncertainty_params)} 个不确定性权重参数添加到优化器中")
    
    param_groups = [
        {'params': base_params, 'lr': config['optimizer']['lr']},
        {'params': attn_params, 'lr': config['optimizer']['lr'] * attn_lr_scale}
    ]
    
    # 添加物理参数组（如果有）
    if physics_params:
        param_groups.append({'params': physics_params, 'lr': config['optimizer']['lr'] * physics_lr_scale})
    
    # 添加不确定性权重参数组（如果有）
    if uncertainty_params:
        param_groups.append({'params': uncertainty_params, 'lr': config['optimizer']['lr']})
    
    # 创建优化器
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            param_groups,
            betas=(config['optimizer'].get('beta1', 0.9), config['optimizer'].get('beta2', 0.999)),
            weight_decay=config['optimizer'].get('weight_decay', 0)
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=config['optimizer'].get('weight_decay', 0.01),
            betas=(config['optimizer'].get('beta1', 0.9), config['optimizer'].get('beta2', 0.999))
        )
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            momentum=config['optimizer'].get('momentum', 0.9),
            weight_decay=config['optimizer'].get('weight_decay', 0.0001),
            nesterov=config['optimizer'].get('nesterov', False)
        )
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    # 动态参数管理功能
    def update_optimizer_with_new_params(model, optimizer):
        """检查模型中是否有新参数需要添加到优化器"""
        # 获取当前优化器管理的所有参数ID
        current_param_ids = set()
        for group in optimizer.param_groups:
            for param in group['params']:
                current_param_ids.add(id(param))
        
        # 检查模型中的所有参数
        new_params = []
        for name, param in model.named_parameters():
            if id(param) not in current_param_ids:
                new_params.append(param)
                print(f"发现新参数: {name}")
        
        # 如果有新参数，添加到优化器的第一个参数组中
        if new_params:
            optimizer.param_groups[0]['params'].extend(new_params)
            print(f"向优化器添加了 {len(new_params)} 个新参数")
            return True
        return False
    
    return optimizer, update_optimizer_with_new_params


def _create_scheduler(optimizer, config):
    """创建学习率调度器"""
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
    
    print(f"使用学习率调度器: {config['scheduler']['name']}")
    
    return scheduler


def _setup_mixed_precision(mixed_precision, use_gpu):
    """设置混合精度训练"""
    scaler = None
    if mixed_precision and use_gpu:
        try:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
            print("启用混合精度训练 (CUDA AMP)")
        except ImportError:
            try:
                from torch.amp import GradScaler
                scaler = GradScaler()
                print("启用混合精度训练 (Generic AMP)")
            except ImportError:
                print("混合精度训练需要PyTorch 1.6+，已禁用混合精度训练。")
                mixed_precision = False
    
    return scaler


def _setup_distributed_model(model, distributed, local_rank, world_size, 
                            find_unused_parameters, use_gpu, gpu_config):
    """设置分布式模型"""
    if distributed and (local_rank != -1 or world_size > 1):
        # 使用DDP包装模型
        model = DDP(
            model, 
            device_ids=[local_rank if local_rank != -1 else 0],
            output_device=local_rank if local_rank != -1 else 0,
            find_unused_parameters=find_unused_parameters
        )
        print(f"使用DistributedDataParallel进行分布式训练, local_rank={local_rank}, world_size={world_size}")
    elif use_gpu and torch.cuda.device_count() > 1 and not distributed:
        # 如果无法使用DDP但有多个GPU，使用DataParallel (不推荐)
        print("使用DataParallel进行多GPU训练，这比DDP效率低。建议使用 --distributed 参数启用DDP。")
        device_ids = gpu_config.get('device_ids', None)
        model = nn.DataParallel(model, device_ids=device_ids)
        print(f"使用DataParallel进行多GPU训练, device_ids={device_ids}")
    
    return model 