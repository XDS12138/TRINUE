#!/usr/bin/env python3
"""
数据加载模块

负责准备训练和验证数据加载器
"""

import os
import logging
from torch.utils.data import DataLoader
import torch

logger = logging.getLogger(__name__)


def multi_degradation_collate_fn_global(batch):
    """Top-level collate_fn to support multiprocessing pickling (DDP + spawn)."""
    raw_imgs = []
    depths = []
    gts = []
    num_degradations = []
    basenames = []
    for sample in batch:
        raw_imgs.append(sample['raw_imgs'])  # [N, C, H, W]
        depths.append(sample['depth'])       # [1, H, W]
        gts.append(sample['gt'])             # [C, H, W]
        num_degradations.append(sample['num_degradations'])
        basenames.append(sample['basename'])
    
    raw_imgs_batch = torch.stack(raw_imgs, dim=0)  # [B, N, C, H, W]
    depths_batch = torch.stack(depths, dim=0)      # [B, 1, H, W]
    gts_batch = torch.stack(gts, dim=0)            # [B, C, H, W]
    
    return {
        'raw_imgs': raw_imgs_batch,
        'depth': depths_batch,
        'gt': gts_batch,
        'num_degradations': torch.tensor(num_degradations),
        'basename': basenames
    }


def prepare_data(config, args):
    """
    准备训练和验证数据加载器，支持文件夹和LMDB两种格式
    现在统一使用多退化数据集逻辑（可用于单退化或多退化情况）
    支持多验证集配置的向后兼容性
    
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
        train_dataset, val_dataset = _create_folder_datasets(
            config, height, width, augment_prob, dataset_type
        )
        
    elif data_format.lower() == 'lmdb':
        train_dataset, val_dataset = _create_lmdb_datasets(
            config, height, width, augment_prob, augmentations
        )
        
    else:
        raise ValueError(f"不支持的数据格式: {data_format}, 应为 'folder' 或 'lmdb'")
    
    # 分布式采样器
    train_sampler, val_sampler = _create_samplers(train_dataset, val_dataset, args)
    
    # 获取额外的DataLoader配置
    persistent_workers = config['data'].get('persistent_workers', False)
    prefetch_factor = config['data'].get('prefetch_factor', 2)
    pin_memory = config['data'].get('pin_memory', True)
    
    # 🔥 创建自定义的collate函数，正确处理多退化数据
    def multi_degradation_collate_fn(batch):
        """处理多退化数据的collate函数"""
        import torch
        raw_imgs = []
        depths = []
        gts = []
        num_degradations = []
        basenames = []
        
        for sample in batch:
            raw_imgs.append(sample['raw_imgs'])  # [N, C, H, W]
            depths.append(sample['depth'])       # [1, H, W] 
            gts.append(sample['gt'])             # [C, H, W]
            num_degradations.append(sample['num_degradations'])
            basenames.append(sample['basename'])
        
        # 堆叠为batch格式
        raw_imgs_batch = torch.stack(raw_imgs, dim=0)  # [B, N, C, H, W]
        depths_batch = torch.stack(depths, dim=0)      # [B, 1, H, W]
        gts_batch = torch.stack(gts, dim=0)            # [B, C, H, W] 
        
        return {
            'raw_imgs': raw_imgs_batch,
            'depth': depths_batch,
            'gt': gts_batch,
            'num_degradations': torch.tensor(num_degradations),
            'basename': basenames
        }
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=train_sampler,
        drop_last=True,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=multi_degradation_collate_fn_global  # 🔥 使用顶层可picklable的collate函数
    )
    
    # 只有当val_dataset不为None时才创建val_loader
    if val_dataset is not None:
            val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=val_sampler,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
    else:
        val_loader = None
        logger.info("⚠️  传统验证集未配置(val_root=None)，val_loader设置为None，将使用多验证集功能")
    
    # 🔥 向后兼容性检查：如果没有配置多验证集但有传统val_root，确保val_loader可用
    validation_sets_config = config.get('validation_sets', {})
    if not validation_sets_config and val_loader is None:
        # 没有配置多验证集且传统验证也失败，记录警告
        logger.warning("⚠️  既没有配置多验证集(validation_sets)，传统验证集(val_root)也不可用")
        logger.warning("   建议在配置文件中添加validation_sets配置以使用多验证集功能")
    
    return {
        'train_loader': train_loader, 
        'val_loader': val_loader, 
        'train_sampler': train_sampler, 
        'val_sampler': val_sampler
    }


def _create_folder_datasets(config, height, width, augment_prob, dataset_type):
    """创建文件夹格式的数据集"""
    # 使用文件夹格式
    train_root = config['data']['train_root']
    val_root = config['data']['val_root']
    
    # 获取子文件夹名称
    folder_structure = config['data'].get('folder_structure', {})
    gt_folder = folder_structure.get('gt', 'gt')
    depth_folder = folder_structure.get('depth', 'depth')
    
    # 导入多退化数据集类
    from modules.datasets import MultiDegradationDataset
    
    # 获取退化文件夹列表
    degradation_folders = config['data'].get('degradation_folders', ['raw'])
    
    # 确保至少有一个退化文件夹
    if not degradation_folders:
        degradation_folders = ['raw']  # 默认使用"raw"作为退化文件夹
    
    # 打印信息
    logger.info(f"数据集类型: {dataset_type}")
    logger.info(f"使用的退化文件夹: {degradation_folders}")
        
    # 实例化训练集和验证集 (都使用MultiDegradationDataset)
    train_dataset = MultiDegradationDataset(
        raw_folders=[os.path.join(train_root, folder) for folder in degradation_folders],
        gt_folder=os.path.join(train_root, gt_folder),
        depth_folder=os.path.join(train_root, depth_folder),
        patch_size=(height, width),
        augment=augment_prob > 0,
        resolution_strategy="min"  # 🔧 使用最小分辨率策略，确保对齐且节省内存
    )
    
    # 如果没有配置val_root，则不创建传统验证集（使用多验证集配置）
    if val_root is not None:
        val_dataset = MultiDegradationDataset(
            raw_folders=[os.path.join(val_root, folder) for folder in degradation_folders],
            gt_folder=os.path.join(val_root, gt_folder),
            depth_folder=os.path.join(val_root, depth_folder),
            patch_size=(height, width),
            augment=False,  # 验证时不使用增强
            resolution_strategy="min"  # 🔧 验证集也使用相同的分辨率策略
        )
    else:
        val_dataset = None  # 使用多验证集配置，不需要传统验证集
    
    return train_dataset, val_dataset


def _create_lmdb_datasets(config, height, width, augment_prob, augmentations):
    """创建LMDB格式的数据集"""
    
    train_lmdb = config['data']['lmdb_paths']['train']
    val_lmdb = config['data']['lmdb_paths'].get('val', None)
    
    # 导入多退化LMDB数据集类
    from modules.datasets import MultiDegradationLMDBDataset
    
    # 实例化训练集
    train_dataset = MultiDegradationLMDBDataset(
        lmdb_path=train_lmdb,
        patch_size=(height, width),
        augment=augment_prob > 0
    )
    
    # 实例化验证集（如果有）
    val_dataset = None
    if val_lmdb and os.path.exists(val_lmdb):
        val_dataset = MultiDegradationLMDBDataset(
            lmdb_path=val_lmdb,
            patch_size=(height, width),
            augment=False  # 验证时不使用增强
        )
    
    logger.info(f"LMDB数据集加载完成: 训练{len(train_dataset)}样本")
    
    return train_dataset, val_dataset


def _create_samplers(train_dataset, val_dataset, args):
    """创建数据采样器"""
    import torch.utils.data
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        # 只有当val_dataset不为None时才创建val_sampler
        if val_dataset is not None:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        else:
            val_sampler = None
    else:
        train_sampler = None
        val_sampler = None
    
    return train_sampler, val_sampler 