#!/usr/bin/env python3
"""
分布式训练工具模块

负责分布式训练的启动和管理
"""

import os
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def launch_distributed_training(world_size, config, args):
    """启动分布式训练"""
    # 设置临时端口进行初始化
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


def distributed_worker(rank, world_size, config, args):
    """分布式训练的工作进程"""
    # 设置当前进程的排名
    args.local_rank = rank
    
    # 设置进程组
    dist.init_process_group(
        backend='nccl',
        init_method=f"env://",
        world_size=world_size,
        rank=rank
    )
    
    # 确保每个进程有不同的随机种子
    from utils.arg_parser import set_seed
    set_seed(args.seed + rank, config.get('gpu', {}))
    
    # 每个进程都运行主工作函数
    from scripts.train import main_worker
    main_worker(config, args)


def setup_for_distributed_launch(config, args):
    """为分布式训练启动做准备，兼容torchrun和mp.spawn两种方式"""
    
    # 检查是否在torchrun环境中（torchrun会设置这些环境变量）
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # torchrun启动模式，直接使用环境变量
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        return False  # 继续执行main_worker
    
    # 传统mp.spawn启动模式
    if args.distributed and torch.cuda.device_count() > 1:
        # 使用mp.spawn启动分布式训练
        world_size = torch.cuda.device_count()
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(random.randint(10000, 20000))
        os.environ['WORLD_SIZE'] = str(world_size)
        
        mp.spawn(distributed_worker,
                 args=(world_size, config, args),
                 nprocs=world_size,
                 join=True)
        return True  # 表示已启动分布式，主进程应退出
    
    return False  # 表示单GPU或CPU训练，继续执行 