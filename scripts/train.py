import os
import sys

# 先添加项目根目录到Python路径，确保可以导入自定义模块
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import traceback  # 添加traceback模块导入

import warnings 
import logging 
import io
from PIL import Image
import torchvision
import logging
import warnings

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

for logger_name in ['PIL.PngImagePlugin', 'PIL.Image', 'PIL.TiffImagePlugin']:
    pil_logger = logging.getLogger(logger_name)
    pil_logger.setLevel(logging.CRITICAL)  # 只保留严重级别的日志
    pil_logger.propagate = False            # 避免将日志向上冒泡到根记录器

from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
from utils.multi_logger import MultiFileLogger, create_multi_logger
from utils.logger import setup_logger, MetricLogger

def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    """自定义的警告处理函数，将 Python 警告转为 logger 日志输出。"""
    # 格式化警告信息，生成类似 “'filename:lineno: category: message'" 的字符串
    log_message = warnings.formatwarning(message, category, filename, lineno, line)
    # 使用名为 'py.warnings' 的 logger 记录 WARNING 级别的日志
    # strip() 用于去除末尾可能多余的换行符
    logging.getLogger('py.warnings').warning(log_message.strip())
# 将 warnings.showwarning 指向自定义函数
warnings.showwarning = custom_showwarning

# ------------------------------------------------------------
# 设置基本日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s')
logger = logging.getLogger(__name__) # Main logger for this script
# Early debug logs after main logger is available
logger.debug(f"Calculated project root path: {root_path}")
logger.debug(f"sys.path before 'from modules.depth_utils': {sys.path}")


from modules.model import UnderwaterEnhanceNet
from modules.loss_fn import TotalLoss, DepthEdgeColorLoss
from modules.depth import get_depth_config_params, ensure_normalized_depth
from utils.checkpoint import save_checkpoint
from utils.lr_scheduler import get_scheduler
from utils.metrics import calculate_psnr as compute_psnr, calculate_ssim as compute_ssim
from utils import metrics as metrics_module  # 如果需要使用所有指标

def setup_logging_system(exp_dir, config):
    """设置完整的多文件日志系统，包括TensorBoard"""
    
    # 创建多文件日志记录器
    multi_logger = create_multi_logger(config, exp_dir)
    
    # 获取主要的日志记录器
    main_logger = multi_logger.get_logger('train')
    
    # 设置TensorBoard
    tb_log_dir = os.path.join(exp_dir, 'tensorboard', datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(tb_log_dir, exist_ok=True)
    
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

    # 创建MetricLogger实例（使用真正的功能完整版本）
    csv_path = os.path.join(exp_dir, 'metrics.csv')
    metric_logger = MetricLogger(main_logger, tb_writer, csv_path)
    
    return multi_logger, metric_logger, tb_writer

# --- End Logger Setup ---

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
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='设置控制台日志级别')
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
    model_params = config.get('model', {}) # Corrected: 'model_params' -> 'model'
    # 获取深度预处理器配置
    # Assuming depth_processor config is nested under 'model' in YAML, or 'loss' as seen in YAML.
    # The YAML shows depth_processing under 'loss', so let's get it from there or make it flexible.
    # For now, let's assume UnderwaterEnhanceNet expects depth_processor_config directly.
    # If 'depth_processor' is part of the 'model' block in YAML, this is fine:
    depth_config_from_yaml = model_params.get('depth_processor', {}) 
    # However, the YAML shows 'depth_processing' under 'loss'.
    # Let's get it from the correct location as per the YAML structure provided.
# —— 先从 loss.block 里读 depth_processing —— 
    loss_config = config.get('loss', {})
    depth_config_from_loss = loss_config.get('depth_processing', {})
    depth_config_from_model = model_params.get('depth_processor_config', {})

# —— 合并：以后者为优先级，覆盖前者的同名字段 —— 
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
    
    # 3.1 SyncBatchNorm (如果使用多GPU)
    if use_gpu and gpu_config.get('sync_bn', False) and world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # 4. 损失函数
    criterion = TotalLoss(
        lambda_img=config['loss']['lambda_img'],
        lambda_ssim=config['loss']['lambda_ssim'],
        lambda_perc=config['loss']['lambda_perc'],
        lambda_fft=config['loss']['lambda_fft'],
        lambda_grad=config['loss']['lambda_grad'],
        lambda_depth=config['loss']['lambda_depth'],
        lambda_smooth=config['loss']['lambda_smooth'],
        lambda_decl=config['loss'].get('lambda_decl', 0.1),  # 深度边缘颜色损失权重
        lambda_cons=config['loss'].get('lambda_cons', 0.1),  # 注意力一致性损失权重
        lambda_phy_A=config['loss'].get('lambda_phy_A', 0.1),   # 物理一致性A损失权重
        lambda_phy_D=config['loss'].get('lambda_phy_D', 0.1),   # 物理一致性D损失权重
        beta_c=model.beta_c if hasattr(model, 'beta_c') else None,  # 从模型获取Beer-Lambert衰减系数
        B_c=model.B_c if hasattr(model, 'B_c') else None,           # 从模型获取全局背景光
        use_uncertainty_weighting=config['loss'].get('use_uncertainty_weighting', True)  # 从配置中读取是否使用自动调参
    )
    criterion = criterion.to(device)
    
    # 5. 优化器
    optimizer_name = config['optimizer'].get('name', 'adamw').lower()
    
    # 9. 创建实验目录
    exp_dir = setup_experiment_dir(config)
    
    # 10. 设置日志
    # 从配置文件中获取日志配置
    logging_config = config.get('logging', {})
    console_log_level_str = logging_config.get('console_level', args.log_level.upper())
    console_log_level = getattr(logging, console_log_level_str, logging.INFO)
    
    # 如果设置为CRITICAL或"CRITICAL"，则不在控制台输出日志
    if console_log_level_str == "CRITICAL" or console_log_level >= logging.CRITICAL:
        console_log_level = logging.CRITICAL
    
    # 获取文件日志级别
    file_log_level_str = logging_config.get('file_level', 'DEBUG')
    file_log_level = getattr(logging, file_log_level_str, logging.DEBUG)
    
    # 使用多文件日志系统
    multi_logger, metric_logger, tb_writer = setup_logging_system(exp_dir, config)
    
    # 获取不同分类的日志记录器
    logger_instance = multi_logger.get_logger('train')  # 主训练日志
    debug_logger_instance = multi_logger.get_logger('debug')  # 调试日志
    model_logger = multi_logger.get_logger('model')  # 模型相关日志
    arch_logger = multi_logger.get_logger('architecture')  # 模型架构日志
    data_logger = multi_logger.get_logger('data')  # 数据相关日志
    metrics_logger = multi_logger.get_logger('metrics')  # 性能指标日志
    optimizer_logger = multi_logger.get_logger('optimizer')  # 优化器日志
    scheduler_logger = multi_logger.get_logger('scheduler')  # 学习率调度器日志
    checkpoint_logger = multi_logger.get_logger('checkpoint')  # 检查点日志
    experiment_logger = multi_logger.get_logger('experiment')  # 实验配置日志
    gpu_logger = multi_logger.get_logger('gpu')  # GPU使用日志
    
    # 参数分组，将交叉注意力参数和其他参数分开，应用不同的学习率
    base_params, attn_params = [], []
    for name, p in model.named_parameters():
        if 'depth2rgb_attn' in name or 'rgb2depth_attn' in name:
            attn_params.append(p)
        else:
            base_params.append(p)
    
    # 获取注意力模块学习率缩放因子
    attn_lr_scale = config['optimizer'].get('attn_lr_scale', 0.1)
    msg = (
        f"使用差异化学习率: 主干 {config['optimizer']['lr']}, "
        f"注意力模块 {config['optimizer']['lr'] * attn_lr_scale}"
    )
    optimizer_logger.info(msg)
    
    # 添加损失函数中的不确定性权重参数到优化器中
    uncertainty_params = []
    if hasattr(criterion, 'use_uncertainty_weighting') and criterion.use_uncertainty_weighting:
        # 收集所有不确定性权重参数
        for name, p in criterion.named_parameters():
            if 'log_var' in name:
                uncertainty_params.append(p)
        
        if uncertainty_params:
            msg = f"将 {len(uncertainty_params)} 个不确定性权重参数添加到优化器中"
            optimizer_logger.info(msg)
            # 输出每个不确定性权重参数的名称
            param_names = [name for name, p in criterion.named_parameters() if 'log_var' in name]
            optimizer_logger.info(f"不确定性权重参数列表: {param_names}")
    
    param_groups = [
        {'params': base_params, 'lr': config['optimizer']['lr']},
        {'params': attn_params, 'lr': config['optimizer']['lr'] * attn_lr_scale}
    ]
    
    # 添加不确定性权重参数组（如果有）
    if uncertainty_params:
        param_groups.append({'params': uncertainty_params, 'lr': config['optimizer']['lr']})
        
        # 统计优化器中不同参数组的参数数量
        base_param_count = sum(p.numel() for p in base_params)
        attn_param_count = sum(p.numel() for p in attn_params)
        uncertainty_param_count = sum(p.numel() for p in uncertainty_params)
        total_param_count = base_param_count + attn_param_count + uncertainty_param_count
        
        optimizer_logger.info("优化器参数统计:")
        msg_main = f"  主干参数: {base_param_count:,} ({base_param_count/total_param_count*100:.2f}%)"
        optimizer_logger.info(msg_main)
        msg_attn = f"  注意力参数: {attn_param_count:,} ({attn_param_count/total_param_count*100:.2f}%)"
        optimizer_logger.info(msg_attn)
        msg_unc = f"  不确定性权重参数: {uncertainty_param_count:,} ({uncertainty_param_count/total_param_count*100:.2f}%)"
        optimizer_logger.info(msg_unc)
        msg_total = f"  总参数数量: {total_param_count:,}"
        optimizer_logger.info(msg_total)
    
    # 记录模型架构信息
    arch_logger.info(f"模型类型: {type(model).__name__}")
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'channels'):
        arch_logger.info(f"编码器通道数: {model.encoder.channels}")
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            param_groups,
            betas=(config['optimizer'].get('beta1', 0.9), config['optimizer'].get('beta2', 0.999)),
            weight_decay=config['optimizer'].get('weight_decay', 0)
        )
        optimizer_logger.info(f"使用Adam优化器，权重衰减: {config['optimizer'].get('weight_decay', 0)}")
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=config['optimizer'].get('weight_decay', 0.01),
            betas=(config['optimizer'].get('beta1', 0.9), config['optimizer'].get('beta2', 0.999))
        )
        optimizer_logger.info(f"使用AdamW优化器，权重衰减: {config['optimizer'].get('weight_decay', 0.01)}")
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            momentum=config['optimizer'].get('momentum', 0.9),
            weight_decay=config['optimizer'].get('weight_decay', 0.0001),
            nesterov=config['optimizer'].get('nesterov', False)
        )
        optimizer_logger.info(f"使用SGD优化器，动量: {config['optimizer'].get('momentum', 0.9)}, "
                             f"权重衰减: {config['optimizer'].get('weight_decay', 0.0001)}, "
                             f"Nesterov: {config['optimizer'].get('nesterov', False)}")
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
    
    scheduler_logger.info(f"使用学习率调度器: {config['scheduler']['name']}")
    scheduler_logger.info(f"  总训练轮次: {config['train']['epochs']}")
    if config['scheduler'].get('warmup_epochs', 0) > 0:
        scheduler_logger.info(f"  预热轮次: {config['scheduler'].get('warmup_epochs', 0)}")
    scheduler_logger.info(f"  最小学习率: {config['scheduler'].get('min_lr', 0)}")
    
    # 7. 混合精度设置
    scaler = None
    if mixed_precision:
        try:
            from torch.amp import GradScaler
            scaler = GradScaler()
            experiment_logger.info("启用混合精度训练")
            # 注意：不要将整个模型转换为半精度，让autocast自动处理
        except ImportError:
            multi_logger.log_warning("混合精度训练需要PyTorch 1.6+，已禁用混合精度训练。")
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
        experiment_logger.info(f"使用DistributedDataParallel进行分布式训练, local_rank={local_rank}, world_size={world_size}")
    elif use_gpu and torch.cuda.device_count() > 1 and not distributed:
        # 如果无法使用DDP但有多个GPU，使用DataParallel (不推荐)
        multi_logger.log_warning("使用DataParallel进行多GPU训练，这比DDP效率低。建议使用 --distributed 参数启用DDP。")
        device_ids = gpu_config.get('device_ids', None)
        model = nn.DataParallel(model, device_ids=device_ids)
        experiment_logger.info(f"使用DataParallel进行多GPU训练, device_ids={device_ids}")
    
    # 设置TensorBoard
    tb_log_dir = os.path.join(exp_dir, 'tensorboard', datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(tb_log_dir, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        # 测试写入以确保目录可写
        tb_writer.add_scalar('setup/tensorboard_test', 1, 0)
        tb_writer.flush()
        experiment_logger.info(f"TensorBoard日志将写入目录: {tb_log_dir}")
        experiment_logger.info("TensorBoard初始化测试写入成功")
    except ImportError:
        multi_logger.log_warning("TensorBoard未安装，部分可视化功能将不可用。请运行 `pip install tensorboard` 安装。")
        tb_writer = None
    except Exception as e:
        multi_logger.log_error(f"TensorBoard初始化失败: {str(e)}. 请检查路径权限和磁盘空间。", exc_info=True)
        tb_writer = None
    
    # metric_logger已经在setup_logging_system中创建了，无需重复创建
    
    # 记录训练配置
    config_text = yaml.dump(config, default_flow_style=False)
    metric_logger.log_text('config', config_text, step=0)
    
    # 记录GPU信息
    if use_gpu:
        gpu_info = [f"GPU {i}: {torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())]
        metric_logger.log_text('gpu_info', "\n".join(gpu_info), step=0)
        gpu_logger.info(f"使用GPU: {', '.join(gpu_info)}")
        
        if distributed:
            experiment_logger.info(f"分布式训练已启用，使用 {backend} 后端, 世界大小: {world_size}")
            if gpu_config.get('sync_bn', False):
                model_logger.info("SyncBatchNorm 已启用")
    else:
        experiment_logger.info("使用CPU进行训练")
    
    if mixed_precision:
        experiment_logger.info("启用混合精度训练")
    
    return {
        'model': model,
        'criterion': criterion,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'device': device,
        'exp_dir': exp_dir,
        'logger': logger_instance,
        'metric_logger': metric_logger,
        'scaler': scaler,
        'mixed_precision': mixed_precision,
        'world_size': world_size,
        'local_world_size': local_world_size,
        'distributed': distributed,
        'local_rank': local_rank,
        'debug_logger': debug_logger_instance,
        'multi_logger': multi_logger
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
        
        logger.warning("LMDB格式目前仅支持单退化输入")
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
    enhanced, pred_gate, student_feats = None, None, None
    
    # 支持多种输出格式
    if isinstance(model_outputs, dict):
        enhanced = model_outputs.get('enhanced')
        pred_gate = model_outputs.get('pred_gate')
        student_feats = model_outputs.get('student_feats')
    elif hasattr(model_outputs, 'to_dict') and callable(getattr(model_outputs, 'to_dict')):
        # 处理ModelOutput类型
        enhanced = model_outputs.enhanced
        pred_gate = model_outputs.pred_gate
        student_feats = model_outputs.student_feats
    else:
        # 假设是元组形式
        if isinstance(model_outputs, (list, tuple)) and len(model_outputs) >= 3:
            enhanced, pred_gate, student_feats = model_outputs[:3]
    
    # 限制可视化样本数量
    n_samples = min(batch_size, enhanced.size(0))
    
    # 1. 基本输出：增强图、深度门控
    data = {
        'enhanced': enhanced[:n_samples].detach().cpu(),
        'depth_gate': pred_gate[:n_samples].detach().cpu(),
    }
    
    # 2. 编码器特征
    if student_feats:
        data['student_feats'] = [feat[:n_samples].detach().cpu() if feat is not None else None 
                              for feat in student_feats]
    
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
               epoch, config, scaler=None, mixed_precision=False, multi_logger=None):
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
    epoch_loss = 0.0 # Initialize epoch_loss
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}")
    vis_interval = config['train'].get('vis_interval', 100)
    param_vis_interval = config['train'].get('param_vis_interval', 500)
    
    # 获取常用的logger实例，避免重复调用
    if multi_logger:
        train_logger = multi_logger.get_logger('train')
        vis_logger = multi_logger.get_logger('visualization')
        warning_logger = multi_logger.get_logger('warning')
        error_logger = multi_logger.get_logger('error')
        depth_logger = multi_logger.get_logger('depth')
        metrics_logger = multi_logger.get_logger('metrics')
        physics_logger = multi_logger.get_logger('physics')
        attention_logger = multi_logger.get_logger('attention')
        optimizer_logger = multi_logger.get_logger('optimizer')
        data_logger = multi_logger.get_logger('data') # 新增data logger
        gpu_logger = multi_logger.get_logger('gpu') # 新增gpu logger
    else:
        # 如果没有multi_logger，则回退到默认logger
        train_logger = vis_logger = warning_logger = error_logger = depth_logger = metrics_logger = physics_logger = attention_logger = optimizer_logger = data_logger = gpu_logger = metric_logger.logger
    
    # 在训练前检查并记录关键配置
    train_logger.info(f"======== [Epoch {epoch+1}/{config['train']['epochs']}] 开始 ========")
    
    # 记录DECL损失的状态
    lambda_decl = config.get('loss', {}).get('lambda_decl', 0)
    if lambda_decl > 0:
        warning_logger.warning(f"Epoch {epoch+1}: 配置文件中DECL损失权重 (lambda_decl) 为 {lambda_decl}，但代码已禁用此损失。将忽略此权重。")

    # 定义性能记录器
    epoch_loss = 0.0
    step_count = 0
    
    # 可视化设置
    vis_interval = config.get('visualization', {}).get('interval', 100)
    
    for i, batch in enumerate(progress_bar):
        # ===== 数据输入检查与记录 =====
        def log_input_data(tensor, name, current_step):
            if tensor is None:
                data_logger.debug(f"Step {current_step}: 输入张量 '{name}' 为 None。")
                return
            
            # 使用.detach()避免记录计算图
            tensor_data = tensor.detach()
            
            stats = {
                'min': tensor_data.min().item(),
                'max': tensor_data.max().item(),
                'mean': tensor_data.mean().item(),
                'std': tensor_data.std().item(),
                'has_nan': torch.isnan(tensor_data).any().item(),
                'has_inf': torch.isinf(tensor_data).any().item(),
                'shape': list(tensor_data.shape),
                'dtype': str(tensor_data.dtype)
            }
            
            log_msg = f"Step {current_step} [数据输入] | {name:<10} | 形状: {stats['shape']} | 类型: {stats['dtype']} | 范围: [{stats['min']:.4f}, {stats['max']:.4f}] | 均值: {stats['mean']:.4f} | 标准差: {stats['std']:.4f}"
            
            if stats['has_nan'] or stats['has_inf']:
                error_logger.error(f"{log_msg} | 包含异常值!")
                # 如果有异常，记录更详细的信息用于调试
                if stats['has_nan']:
                    nan_indices = torch.nonzero(torch.isnan(tensor_data))
                    error_logger.error(f"  - {name} 的NaN位置 (前5个): {nan_indices[:5].tolist()}")
            else:
                # 只在每个epoch的第一个batch或定期记录常规信息，避免日志泛滥
                if i == 0 or i % 100 == 0:
                    data_logger.info(log_msg)

        current_step = epoch * len(train_loader) + i
        
        if isinstance(batch, dict):
            raw_imgs = batch['raw_imgs'].to(device)
            depth_gt = batch['depth'].to(device) if 'depth' in batch else None
            gt = batch['gt'].to(device) if 'gt' in batch else None
            if len(raw_imgs.shape) == 4: raw_imgs = raw_imgs.unsqueeze(1)
            B, N = raw_imgs.shape[:2]
        else:
            raw, depth_gt_tuple, gt_tuple = batch[:3] # Renamed to avoid conflict
            raw_imgs = raw.unsqueeze(1).to(device) 
            depth_gt = depth_gt_tuple.to(device) if depth_gt_tuple is not None else None
            gt = gt_tuple.to(device) if gt_tuple is not None else None
            B, N = raw_imgs.shape[:2]
        
        # 记录输入数据
        log_input_data(raw_imgs, "raw_imgs", current_step)
        log_input_data(depth_gt, "depth_gt", current_step)  
        log_input_data(gt, "gt_imgs", current_step)
        
        optimizer.zero_grad()
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        optimizer_logger.info(f"Step {current_step}: 当前学习率: {current_lr:.6f}")
        metric_logger.log_metrics({"lr": current_lr}, prefix="optimizer", step=current_step)
        
        if mixed_precision and scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                all_losses = []
                # 按批次逐个处理
                for b_loop_idx in range(B): # Renamed loop variable to avoid potential confusion if 'b' is used outside
                    raw_batch_b = raw_imgs[b_loop_idx]
                    depth_gt_b = depth_gt[b_loop_idx:b_loop_idx+1] if depth_gt is not None else None
                    gt_b = gt[b_loop_idx:b_loop_idx+1] if gt is not None else None
                    
                    outputs = model.multi_forward(raw_batch_b, depth_gt_b, gt_b)
                    
                    enhanced = outputs.enhanced
                    pred_gate = outputs.pred_gate
                    student_feats = outputs.student_feats
                    depth_conf_map = outputs.depth_conf_map
                    attention_maps = outputs.attention_maps
                    depth_pred = outputs.depth_pred  # 获取深度预测
                    
                    for n_loop_idx in range(N): # Renamed loop variable
                        loss_n = criterion(
                            enhanced[n_loop_idx:n_loop_idx+1], 
                            gt_b, 
                            pred_gate[n_loop_idx:n_loop_idx+1], 
                            depth_gt_b,
                            student_feats[n_loop_idx], 
                            attention_maps,  # 传递注意力图
                            depth_pred[n_loop_idx:n_loop_idx+1],  # 传递 depth_pred 给 criterion
                            depth_conf_map,  # 传递depth_conf_map给criterion
                            outputs.J_D[n_loop_idx:n_loop_idx+1] if hasattr(outputs, 'J_D') and outputs.J_D is not None else None,  # 传递J_D用于物理损失
                            outputs.I_A[n_loop_idx:n_loop_idx+1] if hasattr(outputs, 'I_A') and outputs.I_A is not None else None,  # 传递I_A用于物理损失
                            raw_batch_b[n_loop_idx:n_loop_idx+1]  # 传递raw用于物理损失
                        )
                        
                        # 注意：DECL损失已禁用（深度边缘颜色损失）
                            
                        all_losses.append(loss_n)
                
                loss = torch.stack(all_losses).mean()
            
            scaler.scale(loss).backward()
            
            # 梯度裁剪 (如果配置)
            if config['optimizer'].get('clip_grad_norm', 0) > 0:
                # 在 unscale 之前裁剪
                scaler.unscale_(optimizer)
                clip_norm_val = torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['clip_grad_norm'])
                gpu_logger.info(f"Step {current_step}: 梯度范数 (裁剪前): {clip_norm_val:.4f}")
                metric_logger.log_metrics({"grad_norm": clip_norm_val.item()}, prefix="gpu", step=current_step)

            scaler.step(optimizer)
            scaler.update()
        else:
            # 常规训练
            all_losses = []
            # 按批次逐个处理
            for b_loop_idx in range(B): # Renamed loop variable
                raw_batch_b = raw_imgs[b_loop_idx]
                depth_gt_b = depth_gt[b_loop_idx:b_loop_idx+1] if depth_gt is not None else None
                gt_b = gt[b_loop_idx:b_loop_idx+1] if gt is not None else None
                
                outputs = model.multi_forward(raw_batch_b, depth_gt_b, gt_b)
                
                enhanced = outputs.enhanced
                pred_gate = outputs.pred_gate
                student_feats = outputs.student_feats
                depth_conf_map = outputs.depth_conf_map
                attention_maps = outputs.attention_maps
                depth_pred = outputs.depth_pred  # 获取深度预测
                
                for n_loop_idx in range(N): # Renamed loop variable
                    loss_n = criterion(
                        enhanced[n_loop_idx:n_loop_idx+1], 
                        gt_b, 
                        pred_gate[n_loop_idx:n_loop_idx+1], 
                        depth_gt_b,
                        student_feats[n_loop_idx], 
                        attention_maps,  # 传递注意力图
                        depth_pred[n_loop_idx:n_loop_idx+1],  # 传递 depth_pred 给 criterion
                        depth_conf_map,  # 传递depth_conf_map给criterion
                        outputs.J_D[n_loop_idx:n_loop_idx+1] if hasattr(outputs, 'J_D') and outputs.J_D is not None else None,  # 传递J_D用于物理损失
                        outputs.I_A[n_loop_idx:n_loop_idx+1] if hasattr(outputs, 'I_A') and outputs.I_A is not None else None,  # 传递I_A用于物理损失
                        raw_batch_b[n_loop_idx:n_loop_idx+1]  # 传递raw用于物理损失
                    )
                    
                    # 注意：DECL损失已禁用（深度边缘颜色损失）
                        
                    all_losses.append(loss_n)
            
            loss = torch.stack(all_losses).mean()
            loss.backward()

            # 梯度裁剪 (如果配置)
            if config['optimizer'].get('clip_grad_norm', 0) > 0:
                clip_norm_val = torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['clip_grad_norm'])
                gpu_logger.info(f"Step {current_step}: 梯度范数 (裁剪前): {clip_norm_val:.4f}")
                metric_logger.log_metrics({"grad_norm": clip_norm_val.item()}, prefix="gpu", step=current_step)

            optimizer.step()
        
        current_loss = loss.item()
        epoch_loss += current_loss
        
        progress_bar.set_postfix({"Loss": f"{current_loss:.4f}"})
        
        metrics = {"loss": current_loss}
        
        if hasattr(criterion, 'get_latest_losses'):
            loss_components = criterion.get_latest_losses()
            for loss_name, loss_value in loss_components.items():
                if loss_name != 'total_loss':
                    if not (loss_name.startswith('loss_') or loss_name.endswith('_loss')):
                        metrics[f"loss_{loss_name}"] = loss_value
                    else:
                        metrics[loss_name] = loss_value
            if 'depth_total_loss' in loss_components:
                progress_bar.set_postfix({
                    "Loss": f"{current_loss:.4f}",
                    "Depth Loss": f"{loss_components['depth_total_loss']:.4f}",
                    "LR": f"{current_lr:.6f}"
                })
        
        # 注意：DECL损失已禁用
        
        metric_logger.log_metrics(metrics, prefix="train", step=current_step) # Added step to log_metrics
        
        # 使用多文件日志系统记录详细分类的损失信息
        if multi_logger:
            # 基本损失记录
            multi_logger.log_loss(metrics, current_step, prefix="train")
        
            # 将损失按类别分组记录
            if hasattr(criterion, 'get_latest_losses'):
                loss_components = criterion.get_latest_losses()
                
                # 图像质量相关损失
                image_metrics = {}
                for k in ['l1_loss', 'ssim_loss', 'perc_loss', 'fft_loss', 'grad_loss', 'img_total_loss']:
                    if k in loss_components:
                        image_metrics[k] = loss_components[k]
                if image_metrics:
                    metrics_logger.info(f"[训练] Step {current_step} 图像质量损失: " + 
                                      ", ".join([f"{k}={v:.6f}" for k, v in image_metrics.items()]))
                
                # 深度相关损失
                depth_metrics = {}
                for k in ['depth_pred_loss', 'depth_smooth_loss', 'depth_decoder_loss', 'depth_rec_loss', 'depth_total_loss']:
                    if k in loss_components:
                        depth_metrics[k] = loss_components[k]
                if depth_metrics:
                    depth_logger.info(f"[训练] Step {current_step} 深度损失: " + 
                                    ", ".join([f"{k}={v:.6f}" for k, v in depth_metrics.items()]))
                
                # 物理模型相关损失
                physics_metrics = {}
                for k in ['phy_A_L1', 'phy_A_SSIM', 'L_phy_A', 'phy_D_L1', 'phy_D_SSIM', 'L_phy_D']:
                    if k in loss_components:
                        physics_metrics[k] = loss_components[k]
                if physics_metrics:
                    physics_logger.info(f"[训练] Step {current_step} 物理模型损失: " + 
                                      ", ".join([f"{k}={v:.6f}" for k, v in physics_metrics.items()]))
                
                # 注意力相关损失
                attention_metrics = {}
                for k in ['attn_cons_loss']:
                    if k in loss_components:
                        attention_metrics[k] = loss_components[k]
                if attention_metrics:
                    attention_logger.info(f"[训练] Step {current_step} 注意力损失: " + 
                                        ", ".join([f"{k}={v:.6f}" for k, v in attention_metrics.items()]))
                
                # 不确定性权重
                if hasattr(criterion, 'use_uncertainty_weighting') and criterion.use_uncertainty_weighting:
                    uncertainty_metrics = {}
                    for key, value in loss_components.items():
                        if key.startswith('uncertainty_'):
                            uncertainty_metrics[key] = value
                        elif key.startswith('log_var_'):
                            uncertainty_metrics[key] = value
                    
                    if uncertainty_metrics:
                        optimizer_logger.info(f"[训练] Step {current_step} 不确定性权重: " + 
                                           ", ".join([f"{k}={v:.6f}" for k, v in uncertainty_metrics.items()]))
                        metric_logger.log_metrics(uncertainty_metrics, prefix="uncertainty", step=current_step)
        
        if i % vis_interval == 0:
            try:
                vis_logger.info(f"======== 开始记录可视化数据，步骤: {current_step} ========")
                vis_raw_first_item = raw_imgs[0, 0].unsqueeze(0)
                vis_depth_first_item = depth_gt[0].unsqueeze(0) if depth_gt is not None and depth_gt.nelement() > 0 and B > 0 else None
                vis_gt_first_item = gt[0].unsqueeze(0) if gt is not None and gt.nelement() > 0 and B > 0 else None

                # 跟踪深度特征 - 添加详细的深度可视化
                if vis_depth_first_item is not None:
                    vis_logger.info(f"Step {current_step}: 正在处理深度图可视化...")
                    # 对深度图执行对数变换和归一化处理，确保显示正确
                    log_depth = torch.log(vis_depth_first_item + 1.0)
                    # 从配置中获取深度范围
                    depth_config = config['loss'].get('depth_processing', {})
                    min_depth = depth_config.get('min_depth_log', 5000.0)
                    max_depth = depth_config.get('max_depth_log', 65000.0)
                    log_min = torch.log(torch.tensor(min_depth, device=vis_depth_first_item.device) + 1.0)
                    log_max = torch.log(torch.tensor(max_depth, device=vis_depth_first_item.device) + 1.0)
                    norm_depth = (log_depth - log_min) / (log_max - log_min + 1e-6)
                    norm_depth = torch.clamp(norm_depth, 0, 1)
                    
                    # 记录各种深度相关可视化
                    metric_logger.log_image("train/depth_gt_normalized", norm_depth, step=current_step)
                    vis_logger.info(f"  - 已记录 'train/depth_gt_normalized'")
                    
                    # 对原始深度使用另一种可视化方式（热力图风格）
                    try:
                        import matplotlib.pyplot as plt
                        import matplotlib.cm as cm
                        norm_depth_np = norm_depth.squeeze().cpu().numpy()
                        colored_depth = torch.from_numpy(cm.viridis(norm_depth_np)[:, :, :3]).permute(2, 0, 1)
                        metric_logger.log_image("train/depth_gt_colored", colored_depth, step=current_step)
                        vis_logger.info(f"  - 已记录 'train/depth_gt_colored'")
                    except ImportError:
                        warning_logger.warning("matplotlib 未安装, 跳过彩色深度图可视化。 `pip install matplotlib`")
                        metric_logger.log_image("train/depth_gt_colored", norm_depth.repeat(1, 3, 1, 1), step=current_step)
                
                with torch.no_grad():
                    # 模型前向传播，获取所有输出
                    outputs_dict = model.multi_forward(
                        vis_raw_first_item, vis_depth_first_item, vis_gt_first_item
                    )
                    
                    # 解析输出结果 - 修复ModelOutput对象访问方式
                    vis_outputs = outputs_dict.enhanced
                    vis_pred_gate = outputs_dict.pred_gate
                    vis_student_feats = outputs_dict.student_feats
                    vis_depth_pred = outputs_dict.depth_pred  # 获取连续深度预测
                    attention_maps = outputs_dict.attention_maps  # 获取注意力图
                    
                    # 可视化注意力图（如果有）
                    if attention_maps is not None and config.get('visualization', {}).get('save_attention_maps', False):
                        vis_logger.info(f"Step {current_step}: 正在处理注意力图可视化...")
                        depth2rgb_attn, rgb2depth_attn = attention_maps
                        
                        if depth2rgb_attn is not None:
                            # 为可视化选择第一个头的注意力图
                            depth2rgb_viz = depth2rgb_attn[0, 0].unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
                            depth2rgb_viz = (depth2rgb_viz - depth2rgb_viz.min()) / (depth2rgb_viz.max() - depth2rgb_viz.min() + 1e-8)
                            metric_logger.log_image("train/depth2rgb_attention", depth2rgb_viz, step=current_step)
                            attention_logger.info(f"  - 已记录 'train/depth2rgb_attention', shape={depth2rgb_viz.shape}")
                        
                        if rgb2depth_attn is not None:
                            # 为可视化选择第一个头的注意力图
                            rgb2depth_viz = rgb2depth_attn[0, 0].unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
                            rgb2depth_viz = (rgb2depth_viz - rgb2depth_viz.min()) / (rgb2depth_viz.max() - rgb2depth_viz.min() + 1e-8)
                            metric_logger.log_image("train/rgb2depth_attention", rgb2depth_viz, step=current_step)
                            attention_logger.info(f"  - 已记录 'train/rgb2depth_attention', shape={rgb2depth_viz.shape}")
                    
                    # 记录调试信息，确认是否获取了连续深度预测
                    if vis_depth_pred is not None:
                        depth_logger.info(f"Train step {current_step}: 获取到连续深度预测 shape={vis_depth_pred.shape}, range=[{vis_depth_pred.min().item():.4f}, {vis_depth_pred.max().item():.4f}]")
                    else:
                        warning_logger.warning(f"Train step {current_step}: 连续深度预测为None")
                    
                    # 1. 记录单独的图像
                    vis_logger.info(f"Step {current_step}: 正在处理RGB图像可视化...")
                    # 输入图像归一化（假设输入已经在[0,1]或[-1,1]）
                    if vis_raw_first_item.min() < 0:
                        vis_input_normalized = (vis_raw_first_item + 1.0) / 2.0
                    else:
                        vis_input_normalized = vis_raw_first_item
                    metric_logger.log_image("train/input", vis_input_normalized, step=current_step)
                    vis_logger.info(f"  - 已记录 'train/input'")
                    
                    # 增强图归一化处理：从[-1,1]映射到[0,1]
                    if vis_outputs is not None:
                        if vis_outputs.min() < 0:
                            vis_outputs_normalized = (vis_outputs + 1.0) / 2.0
                        else:
                            vis_outputs_normalized = vis_outputs
                        metric_logger.log_image("train/enhanced", vis_outputs_normalized, step=current_step)
                        vis_logger.info(f"  - 已记录 'train/enhanced'")
                    else:
                        warning_logger.warning(f"训练可视化: 增强图为None，无法记录")
                    
                    # GT图像
                    if vis_gt_first_item is not None:
                        if vis_gt_first_item.min() < 0:
                            vis_gt_normalized = (vis_gt_first_item + 1.0) / 2.0
                        else:
                            vis_gt_normalized = vis_gt_first_item
                        metric_logger.log_image("train/gt", vis_gt_normalized, step=current_step)
                        vis_logger.info(f"  - 已记录 'train/gt'")
                        
                        # 2. 创建RGB对比图（输入/增强/GT并排）
                        if vis_outputs is not None:
                            vis_logger.info(f"Step {current_step}: 正在创建RGB对比图...")
                            # 确保所有图像尺寸相同
                            comparison_list = []
                            
                            # 确保所有张量都有4维 (B, C, H, W)
                            def ensure_4d(tensor):
                                if tensor is None:
                                    return None
                                if tensor.ndim == 3:  # C, H, W -> B, C, H, W
                                    return tensor.unsqueeze(0)
                                elif tensor.ndim == 2:  # H, W -> B, C, H, W (假设是灰度图)
                                    return tensor.unsqueeze(0).unsqueeze(0)
                                return tensor
                            
                            # 添加原始输入图
                            vis_input_4d = ensure_4d(vis_input_normalized)
                            if vis_input_4d is not None:
                                comparison_list.append(vis_input_4d)
                            
                            # 添加增强图
                            vis_outputs_4d = ensure_4d(vis_outputs_normalized)
                            if vis_outputs_4d is not None:
                                comparison_list.append(vis_outputs_4d)
                            
                            # 添加GT图
                            vis_gt_4d = ensure_4d(vis_gt_normalized)
                            if vis_gt_4d is not None:
                                comparison_list.append(vis_gt_4d)
                            
                            # 水平拼接（只有当所有图像都存在时才拼接）
                            if len(comparison_list) >= 2:  # 至少需要2张图片才能比较
                                comparison_rgb = torch.cat(comparison_list, dim=-1)
                                metric_logger.log_image("train/comparison_rgb", comparison_rgb, step=current_step)
                                vis_logger.info(f"  - 已记录 'train/comparison_rgb'")
                            else:
                                warning_logger.warning(f"训练可视化: 没有足够的图像进行对比（仅有{len(comparison_list)}张），跳过对比图")
                            
                            # 3. 计算并显示误差图
                            vis_logger.info(f"Step {current_step}: 正在创建误差图...")
                            error_map = torch.abs(vis_outputs_normalized - vis_gt_normalized)
                            metric_logger.log_image("train/error_map", error_map, step=current_step)
                            vis_logger.info(f"  - 已记录 'train/error_map'")
                            
                            # 4. 添加热图形式的误差可视化（更易观察）
                            try:
                                import matplotlib.pyplot as plt
                                import matplotlib.cm as cm
                                error_map_gray = error_map.squeeze().cpu().numpy()
                                
                                # 确保error_map_gray是2D数组
                                if error_map_gray.ndim > 2:
                                    error_map_gray = error_map_gray.mean(axis=0)  # 如果是多维，取平均
                                
                                # 应用viridis色彩映射，只取RGB三个通道
                                colored_array = cm.viridis(error_map_gray)[:, :, :3]  # 只取RGB，去掉alpha
                                colored_error = torch.from_numpy(colored_array).permute(2, 0, 1).unsqueeze(0)  # 添加batch维度
                                metric_logger.log_image("train/error_heatmap", colored_error, step=current_step)
                                vis_logger.info(f"  - 已记录 'train/error_heatmap'")
                            except ImportError:
                                warning_logger.warning("matplotlib 未安装, 跳过彩色误差图可视化。 `pip install matplotlib`")
                                metric_logger.log_image("train/error_heatmap", error_map.repeat(1, 3, 1, 1) if error_map.shape[1] == 1 else error_map, step=current_step)
                        else:
                            warning_logger.warning(f"训练可视化: 增强图为None，跳过对比图和误差图")
                    
                    # 深度相关可视化
                    vis_logger.info(f"Step {current_step}: 正在处理深度相关可视化...")
                    if vis_pred_gate is not None:
                        # 添加调试信息，输出深度门控的值范围和统计信息
                        pred_gate_min = vis_pred_gate.min().item()
                        pred_gate_max = vis_pred_gate.max().item()
                        pred_gate_mean = vis_pred_gate.mean().item()
                        pred_gate_std = vis_pred_gate.std().item()
                        depth_logger.info(f"深度门控(pred_gate)统计: Step {current_step} | 范围: [{pred_gate_min:.6f}, {pred_gate_max:.6f}] | 均值: {pred_gate_mean:.6f} | 标准差: {pred_gate_std:.6f}")
                        
                        # 改进深度门控可视化 - 使用更强的对比度增强
                        vis_pred_gate_enhanced = vis_pred_gate.clone()
                        
                        # 如果值域过小（导致看起来是空白），增强对比度
                        if pred_gate_max - pred_gate_min < 0.1:  # 如果范围很小
                            depth_logger.info(f"  - 深度门控值域过小 ({pred_gate_max - pred_gate_min:.6f})，应用对比度增强")
                            # 应用标准化增强对比度
                            if pred_gate_std > 0:  # 避免除以零
                                # 使用Z-score标准化增强对比度（扩大差异）
                                vis_pred_gate_enhanced = (vis_pred_gate - pred_gate_mean) / (pred_gate_std + 1e-8)
                                # 将增强后的值裁剪到合理范围并重新缩放到[0,1]
                                vis_pred_gate_enhanced = torch.clamp(vis_pred_gate_enhanced, -3, 3)  # 限制在±3个标准差内
                                vis_pred_gate_enhanced = (vis_pred_gate_enhanced + 3) / 6  # 从[-3,3]映射到[0,1]
                            else:
                                warning_logger.warning(f"  - 深度门控标准差为零，无法增强对比度")
                                # 为避免全黑图像，手动设置一个渐变
                                h, w = vis_pred_gate.shape[-2:]
                                vis_pred_gate_enhanced = torch.linspace(0, 1, w).view(1, 1, 1, w).repeat(1, 1, h, 1)
                        
                        # 记录原始深度门控
                        metric_logger.log_image("train/depth_gate_original", vis_pred_gate, step=current_step)
                        # 记录增强后的深度门控
                        metric_logger.log_image("train/depth_gate_enhanced", vis_pred_gate_enhanced, step=current_step)
                        vis_logger.info(f"  - 已记录 'depth_gate' (原始图与增强图)")
                    
                    # 连续深度预测可视化
                    if vis_depth_pred is not None:
                        # 简单归一化以便可视化
                        depth_pred_norm = vis_depth_pred.clone().detach()
                        if depth_pred_norm.min() != depth_pred_norm.max():
                            depth_pred_norm = (depth_pred_norm - depth_pred_norm.min()) / (depth_pred_norm.max() - depth_pred_norm.min())
                        metric_logger.log_image("train/depth_pred_continuous", depth_pred_norm, step=current_step)
                        vis_logger.info(f"  - 已记录 'depth_pred_continuous'")
                    
                    if depth_conf_map is not None:
                        metric_logger.log_image("train/depth_conf_map", depth_conf_map, step=current_step)
                        vis_logger.info(f"  - 已记录 'depth_conf_map'")
                    
                    if vis_depth_first_item is not None:
                        # 确保深度GT可视化在全部范围内可见
                        norm_depth_vis = norm_depth.clone()
                        
                        # 显示深度GT
                        metric_logger.log_image("train/depth_gt_comparison", norm_depth_vis, step=current_step)
                        vis_logger.info(f"  - 已记录 'depth_gt_comparison'")
                        
                        # 创建深度对比图（如果有连续深度预测）
                        if vis_depth_pred is not None and 'depth_pred_norm' in locals():
                            # 确保两个深度图有相同的维度
                            def ensure_same_dims(tensor1, tensor2):
                                # 确保都是4维 (B, C, H, W)
                                if tensor1.ndim == 3:  # H, W, C -> B, C, H, W
                                    if tensor1.shape[-1] in [1, 3]:  # HWC format
                                        tensor1 = tensor1.permute(2, 0, 1).unsqueeze(0)
                                    else:  # C, H, W -> B, C, H, W
                                        tensor1 = tensor1.unsqueeze(0)
                                elif tensor1.ndim == 2:  # H, W -> B, C, H, W
                                    tensor1 = tensor1.unsqueeze(0).unsqueeze(0)
                                    
                                if tensor2.ndim == 3:  # H, W, C or C, H, W -> B, C, H, W
                                    if tensor2.shape[-1] in [1, 3]:  # HWC format
                                        tensor2 = tensor2.permute(2, 0, 1).unsqueeze(0)
                                    else:  # C, H, W -> B, C, H, W
                                        tensor2 = tensor2.unsqueeze(0)
                                elif tensor2.ndim == 2:  # H, W -> B, C, H, W
                                    tensor2 = tensor2.unsqueeze(0).unsqueeze(0)
                                
                                # 确保通道数相同
                                if tensor1.shape[1] != tensor2.shape[1]:
                                    if tensor1.shape[1] == 1 and tensor2.shape[1] == 3:
                                        tensor1 = tensor1.repeat(1, 3, 1, 1)
                                    elif tensor1.shape[1] == 3 and tensor2.shape[1] == 1:
                                        tensor2 = tensor2.repeat(1, 3, 1, 1)
                                
                                return tensor1, tensor2
                            
                            norm_depth_vis_4d, depth_pred_norm_4d = ensure_same_dims(norm_depth_vis, depth_pred_norm)
                            depth_comparison = torch.cat([norm_depth_vis_4d, depth_pred_norm_4d], dim=-1)
                            metric_logger.log_image("train/depth_comparison", depth_comparison, step=current_step)
                            vis_logger.info(f"  - 已记录 'depth_comparison'")
                
                # 特征图可视化
                if vis_student_feats:
                    vis_logger.info(f"Step {current_step}: 正在处理特征图可视化...")
                    for j, feat in enumerate(vis_student_feats):
                        if feat is not None:
                            # 添加类型检查，确保是张量而不是列表
                            if isinstance(feat, list):
                                warning_logger.warning(f"特征level{j}是列表类型而不是张量，跳过可视化")
                                continue
                                
                            # 取特征图的前几个通道进行可视化
                            feat_vis = feat[0:1, 0:min(3, feat.shape[1])].mean(1, keepdim=True)  # 平均前3个通道
                            feat_vis = (feat_vis - feat_vis.min()) / (feat_vis.max() - feat_vis.min() + 1e-6)
                            metric_logger.log_image(f"train/student_feat_level{j}", feat_vis, step=current_step)
                            vis_logger.info(f"  - 已记录 'train/student_feat_level{j}'")
                
                # 融合权重可视化
                vis_logger.info(f"Step {current_step}: 正在处理融合权重可视化...")
                current_model_for_weights = model.module if hasattr(model, 'module') else model
                if hasattr(current_model_for_weights.decoder, 'last_fusion_weights') and current_model_for_weights.decoder.last_fusion_weights is not None:
                    fusion_weights = current_model_for_weights.decoder.last_fusion_weights
                    vis_logger.info(f"  - 找到融合权重，形状: {fusion_weights.shape}")
                    # 可视化每个尺度的权重
                    for scale_idx in range(min(4, fusion_weights.shape[2])):  # 最多显示4个尺度
                        weight_map = fusion_weights[0, 0, scale_idx:scale_idx+1]
                        metric_logger.log_image(f"train/depth_fusion_weight_scale{scale_idx}", weight_map, step=current_step)
                        vis_logger.info(f"    - 已记录 'train/depth_fusion_weight_scale{scale_idx}'")
                else:
                    vis_logger.info("  - 未找到融合权重，跳过可视化")
                
                vis_logger.info(f"======== 可视化数据记录完成，步骤: {current_step} ========")
            except Exception as e:
                error_logger.error(f"记录可视化数据时发生错误: {str(e)}，步骤: {current_step}")
                import traceback
                error_logger.error(traceback.format_exc())
        
        if i % param_vis_interval == 0:
            metric_logger.log_model_parameters(model, step=current_step)
        
    epoch_loss /= len(train_loader)
    train_logger.info(f"======== [Epoch {epoch+1}] 结束 | 平均损失: {epoch_loss:.4f} ========")
    return epoch_loss


def validate(val_loader, model, criterion, device, metric_logger, epoch, config, mixed_precision=False, multi_logger=None):
    model.eval()
    progress_bar = tqdm(val_loader, desc=f"Validation [{epoch+1}]")
    
    # 获取常用的logger实例，避免重复调用
    if multi_logger:
        validation_logger = multi_logger.get_logger('validation')
        vis_logger = multi_logger.get_logger('visualization')
        warning_logger = multi_logger.get_logger('warning')
        error_logger = multi_logger.get_logger('error')
        depth_logger = multi_logger.get_logger('depth')
        metrics_logger = multi_logger.get_logger('metrics')
        debug_logger = multi_logger.get_logger('debug')
        data_logger = multi_logger.get_logger('data') # 新增
    else:
        # 如果没有multi_logger，则回退到默认logger
        validation_logger = vis_logger = warning_logger = error_logger = depth_logger = metrics_logger = debug_logger = data_logger = metric_logger.logger
    
    validation_logger.info(f"======== [Epoch {epoch+1}] 开始验证 ========")

    # 验证中也禁用DECL损失
    lambda_decl = config.get('loss', {}).get('lambda_decl', 0)
    if lambda_decl > 0:
        warning_logger.warning(f"Epoch {epoch+1} [验证]: 配置文件中DECL损失权重 (lambda_decl) 为 {lambda_decl}，但代码已禁用此损失。")
    
    # Visualization and metric config from YAML
    vis_config = config.get('visualization', {})
    val_imgs_config = vis_config.get('val_images', {})
    save_val_imgs_to_disk = val_imgs_config.get('save', True)
    max_samples_for_processing = val_imgs_config.get('max_samples', 8) # For disk saving and detailed metrics
    # save_comparison_to_disk = val_imgs_config.get('save_comparison', True) # Implicitly handled by save_val_imgs_to_disk
    # save_metrics_text_on_disk_image = val_imgs_config.get('save_metrics', True) # Will be handled during plotting

    val_img_output_dir = None
    if save_val_imgs_to_disk and hasattr(metric_logger, 'tb_writer') and metric_logger.tb_writer is not None:
        val_img_output_dir = os.path.join(metric_logger.tb_writer.log_dir, f'val_images_epoch{epoch+1}')
        os.makedirs(val_img_output_dir, exist_ok=True)
        vis_logger.info(f"Validation images will be saved to: {val_img_output_dir}")
    elif save_val_imgs_to_disk:
        warning_logger.warning("Cannot save validation images to disk: TensorBoard writer not initialized.")

    collected_samples_for_processing = [] 
    epoch_total_batch_losses = [] 
    
    with torch.no_grad():
        for i_batch_loop, current_batch_data in enumerate(progress_bar):
            raw_imgs_b_n_c_h_w, depth_gt_b_or_1, gt_imgs_b_or_1 = None, None, None
            current_batch_size_B, num_degradations_N = 0, 0

            if isinstance(current_batch_data, dict):
                raw_imgs_b_n_c_h_w = current_batch_data['raw_imgs'].to(device)
                depth_gt_b_or_1 = current_batch_data['depth'].to(device) if 'depth' in current_batch_data else None
                gt_imgs_b_or_1 = current_batch_data['gt'].to(device) if 'gt' in current_batch_data else None
                if raw_imgs_b_n_c_h_w.ndim == 4: # B,C,H,W -> B,1,C,H,W
                    raw_imgs_b_n_c_h_w = raw_imgs_b_n_c_h_w.unsqueeze(1)
                current_batch_size_B = raw_imgs_b_n_c_h_w.shape[0]
                num_degradations_N = raw_imgs_b_n_c_h_w.shape[1]
            else: # Old tuple format
                raw_tuple, depth_gt_tuple, gt_tuple = current_batch_data[:3]
                raw_imgs_b_n_c_h_w = raw_tuple.unsqueeze(1).to(device) # Add N dim for consistency
                depth_gt_b_or_1 = depth_gt_tuple.to(device) if depth_gt_tuple is not None else None
                gt_imgs_b_or_1 = gt_tuple.to(device) if gt_tuple is not None else None
                current_batch_size_B = raw_imgs_b_n_c_h_w.shape[0]
                num_degradations_N = raw_imgs_b_n_c_h_w.shape[1]

            for b_idx_in_batch in range(current_batch_size_B):
                # Prepare inputs for model.multi_forward (expects N,C,H,W for raw)
                raw_n_degradations = raw_imgs_b_n_c_h_w[b_idx_in_batch] # Shape: N,C,H,W
                # Depth GT and GT images might be per batch item B, or shared (shape [1,C,H,W])
                depth_gt_for_model = depth_gt_b_or_1[b_idx_in_batch:b_idx_in_batch+1] if depth_gt_b_or_1 is not None and depth_gt_b_or_1.shape[0] == current_batch_size_B else depth_gt_b_or_1
                gt_for_model = gt_imgs_b_or_1[b_idx_in_batch:b_idx_in_batch+1] if gt_imgs_b_or_1 is not None and gt_imgs_b_or_1.shape[0] == current_batch_size_B else gt_imgs_b_or_1

                # Model forward pass
                model_outputs_dict = {}
                if mixed_precision:
                    with torch.amp.autocast(device_type='cuda'):
                        model_outputs_dict = model.multi_forward(raw_n_degradations, depth_gt_for_model, gt_for_model)
                else:
                    model_outputs_dict = model.multi_forward(raw_n_degradations, depth_gt_for_model, gt_for_model)

                # Extract outputs (all are expected to be N,C,H,W or list for feats)
                enhanced_n_items = model_outputs_dict.enhanced
                pred_gate_n_items = model_outputs_dict.pred_gate
                continuous_depth_pred_n_items = model_outputs_dict.depth_pred  # 获取连续深度输出
                student_feats_n_lists = model_outputs_dict.student_feats
                depth_conf_map_for_b_item = model_outputs_dict.depth_conf_map
                attention_maps_for_b_item = model_outputs_dict.attention_maps  # 获取注意力图

                for n_idx_in_degradations in range(num_degradations_N):
                    # Get individual item from the N degradations
                    current_enhanced_item = enhanced_n_items[n_idx_in_degradations:n_idx_in_degradations+1]
                    current_pred_gate_item = pred_gate_n_items[n_idx_in_degradations:n_idx_in_degradations+1]
                    current_student_feats = student_feats_n_lists[n_idx_in_degradations] if student_feats_n_lists and n_idx_in_degradations < len(student_feats_n_lists) else None
                    current_continuous_depth_pred = continuous_depth_pred_n_items[n_idx_in_degradations:n_idx_in_degradations+1] if continuous_depth_pred_n_items is not None else None
                    
                    # Calculate loss if GT is available
                    if gt_for_model is not None:
                        loss_value = criterion(
                            current_enhanced_item, gt_for_model, current_pred_gate_item, depth_gt_for_model,
                            current_student_feats, 
                            attention_maps_for_b_item,  # 传递注意力图
                            current_continuous_depth_pred,  # 传递连续深度预测
                            depth_conf_map_for_b_item,  # 传递depth_conf_map
                            model_outputs_dict.J_D[n_idx_in_degradations:n_idx_in_degradations+1] if hasattr(model_outputs_dict, 'J_D') and model_outputs_dict.J_D is not None else None,  # 传递J_D用于物理损失
                            model_outputs_dict.I_A[n_idx_in_degradations:n_idx_in_degradations+1] if hasattr(model_outputs_dict, 'I_A') and model_outputs_dict.I_A is not None else None,  # 传递I_A用于物理损失
                            raw_n_degradations[n_idx_in_degradations:n_idx_in_degradations+1]  # 传递raw用于物理损失
                        )
                        # 注意：DECL损失已禁用（深度边缘颜色损失）
                        epoch_total_batch_losses.append(loss_value.item())

                    # Collect samples for detailed metric calculation and disk saving (only from first b_item for simplicity)
                    if len(collected_samples_for_processing) < max_samples_for_processing and b_idx_in_batch == 0:
                        collected_samples_for_processing.append({
                            'input': raw_n_degradations[n_idx_in_degradations:n_idx_in_degradations+1].cpu(),
                            'output': current_enhanced_item.cpu(),
                            'gt': gt_for_model.cpu() if gt_for_model is not None else None,
                            'depth_gt': depth_gt_for_model.cpu() if depth_gt_for_model is not None else None,
                            'depth_pred_continuous': current_continuous_depth_pred.cpu() if current_continuous_depth_pred is not None else None,
                            'depth_pred_gate': current_pred_gate_item.cpu(), # Store gate separately if needed for viz
                            'filename_suffix': f"epoch{epoch+1}_bidx{i_batch_loop}_b{b_idx_in_batch}_n{n_idx_in_degradations}"
                        })
            
            # 改进的验证过程TensorBoard可视化
            val_vis_interval = vis_config.get('val_vis_interval', 50)  # 可以单独设置验证可视化间隔
            max_val_vis_samples = vis_config.get('max_val_vis_samples', 10)  # 最多可视化几个样本
            if i_batch_loop % val_vis_interval == 0 and (i_batch_loop // val_vis_interval) < max_val_vis_samples and current_batch_size_B > 0 and num_degradations_N > 0:
                # Log the first degradation (n=0) of the first item in the batch (b=0)
                tb_log_raw_img = raw_imgs_b_n_c_h_w[0,0].unsqueeze(0) 
                tb_log_enh_img = enhanced_n_items[0:1] if enhanced_n_items is not None else None
                tb_log_gt_img  = gt_imgs_b_or_1[0:1] if gt_imgs_b_or_1 is not None and gt_imgs_b_or_1.shape[0] == current_batch_size_B else gt_imgs_b_or_1
                tb_log_depth_gt_img = depth_gt_b_or_1[0:1] if depth_gt_b_or_1 is not None and depth_gt_b_or_1.shape[0] == current_batch_size_B else depth_gt_b_or_1
                tb_log_depth_gate_pred = pred_gate_n_items[0:1] if pred_gate_n_items is not None else None
                tb_log_depth_continuous_pred = continuous_depth_pred_n_items[0:1] if continuous_depth_pred_n_items is not None else None
                tb_log_depth_conf_map = depth_conf_map_for_b_item # This is from b_idx=0

                if tb_log_enh_img is not None:
                    sample_idx = i_batch_loop // val_vis_interval
                    
                    # 归一化函数（改进版）
                    def normalize_image(img):
                        if img is None: return None
                        img = img.clone().detach().cpu()
                        if img.min() < -0.5:  # 假设是[-1, 1]范围
                            img = (img + 1.0) / 2.0
                        return img.clamp(0, 1)
                    
                    def normalize_depth(depth):
                        if depth is None: return None
                        depth = depth.clone().detach().cpu()
                        # 对数变换（如果深度值很大）
                        if depth.max() > 100:  # 假设是原始深度值
                            depth_config = config['loss'].get('depth_processing', {})
                            min_depth = depth_config.get('min_depth_log', 5000.0)
                            max_depth = depth_config.get('max_depth_log', 65000.0)
                            depth = torch.log(depth + 1.0)
                            log_min = torch.log(torch.tensor(min_depth) + 1.0)
                            log_max = torch.log(torch.tensor(max_depth) + 1.0)
                            depth = (depth - log_min) / (log_max - log_min + 1e-6)
                        return depth.clamp(0, 1)

                    # 1. 单独的图像 - 添加更详细的日志
                    if tb_log_raw_img is not None:
                        normalized_input = normalize_image(tb_log_raw_img)
                        metric_logger.log_image(f"val/sample{sample_idx}/input", normalized_input, step=epoch)
                        vis_logger.info(f"记录验证图像到TensorBoard: val/sample{sample_idx}/input")
                    else:
                        warning_logger.warning(f"验证样本{sample_idx}的输入图像为None，跳过TB记录")
                        
                    if tb_log_enh_img is not None:
                        normalized_enhanced = normalize_image(tb_log_enh_img)
                        metric_logger.log_image(f"val/sample{sample_idx}/enhanced", normalized_enhanced, step=epoch)
                        # 添加彩色增强图（更易观察）
                        metric_logger.log_image(f"val/sample{sample_idx}/enhanced_color", normalized_enhanced, step=epoch)
                        vis_logger.info(f"记录验证图像到TensorBoard: val/sample{sample_idx}/enhanced")
                    else:
                        warning_logger.warning(f"验证样本{sample_idx}的增强图像为None，跳过TB记录")
                        
                    if tb_log_gt_img is not None: 
                        normalized_gt = normalize_image(tb_log_gt_img)
                        metric_logger.log_image(f"val/sample{sample_idx}/gt", normalized_gt, step=epoch)
                        # 添加彩色GT图（更易观察）
                        metric_logger.log_image(f"val/sample{sample_idx}/gt_color", normalized_gt, step=epoch)
                        vis_logger.info(f"记录验证图像到TensorBoard: val/sample{sample_idx}/gt")
                        
                        # 2. RGB对比图（输入/增强/GT并排）
                        if normalized_enhanced is not None and normalized_input is not None:
                            # 确保所有图像尺寸相同
                            comparison_list = []
                            # 添加原始输入图
                            comparison_list.append(normalized_input)
                            # 添加增强图
                            comparison_list.append(normalized_enhanced)
                            # 添加GT图
                            comparison_list.append(normalized_gt)
                            
                            # 水平拼接
                            comparison_rgb = torch.cat(comparison_list, dim=-1)
                            metric_logger.log_image(f"val/sample{sample_idx}/comparison_rgb", comparison_rgb, step=epoch)
                            vis_logger.info(f"记录验证对比图到TensorBoard: val/sample{sample_idx}/comparison_rgb")
                            
                            # 3. 误差图
                            error_map = torch.abs(normalized_enhanced - normalized_gt)
                            error_map_colored = error_map.repeat(1, 3, 1, 1) if error_map.shape[1] == 1 else error_map
                            metric_logger.log_image(f"val/sample{sample_idx}/error_map", error_map_colored, step=epoch)
                            metric_logger.log_image(f"val/sample{sample_idx}/error_heatmap", error_map_colored, step=epoch)
                            vis_logger.info(f"记录验证图像到TensorBoard: val/sample{sample_idx}/error_map")
                    else:
                        warning_logger.warning(f"验证样本{sample_idx}的GT图像为None，跳过对比图和误差图")
                    
                    # 4. 深度相关可视化
                    if tb_log_depth_gt_img is not None: 
                        metric_logger.log_image(f"val/sample{sample_idx}/depth_gt", normalize_depth(tb_log_depth_gt_img), step=epoch)
                    if tb_log_depth_gate_pred is not None: 
                        # 添加调试信息，输出深度门控的值范围和统计信息
                        gate_min = tb_log_depth_gate_pred.min().item()
                        gate_max = tb_log_depth_gate_pred.max().item()
                        gate_mean = tb_log_depth_gate_pred.mean().item()
                        gate_std = tb_log_depth_gate_pred.std().item()
                        depth_logger.info(f"验证深度门控(depth_pred_gate)统计: 最小值={gate_min:.6f}, 最大值={gate_max:.6f}, 平均值={gate_mean:.6f}, 标准差={gate_std:.6f}")
                        
                        # 改进深度门控可视化 - 使用更强的对比度增强
                        gate_enhanced = tb_log_depth_gate_pred.clone()
                        
                        # 如果值域过小（导致看起来是空白），增强对比度
                        if gate_max - gate_min < 0.1:  # 如果范围很小
                            depth_logger.info(f"验证深度门控值域过小 ({gate_max - gate_min:.6f})，应用对比度增强")
                            # 应用标准化增强对比度
                            if gate_std > 0:  # 避免除以零
                                # 使用Z-score标准化增强对比度（扩大差异）
                                gate_enhanced = (tb_log_depth_gate_pred - gate_mean) / (gate_std + 1e-8)
                                # 将增强后的值裁剪到合理范围并重新缩放到[0,1]
                                gate_enhanced = torch.clamp(gate_enhanced, -3, 3)  # 限制在±3个标准差内
                                gate_enhanced = (gate_enhanced + 3) / 6  # 从[-3,3]映射到[0,1]
                            else:
                                warning_logger.warning(f"验证深度门控标准差为零，无法增强对比度")
                                # 为避免全黑图像，手动设置一个渐变
                                h, w = tb_log_depth_gate_pred.shape[-2:]
                                gate_enhanced = torch.linspace(0, 1, w).view(1, 1, 1, w).repeat(1, 1, h, 1)
                        
                        # 记录原始深度门控
                        metric_logger.log_image(f"val/sample{sample_idx}/depth_pred_gate_original", tb_log_depth_gate_pred, step=epoch)
                        # 记录增强后的深度门控
                        metric_logger.log_image(f"val/sample{sample_idx}/depth_pred_gate", gate_enhanced, step=epoch)
                    if tb_log_depth_continuous_pred is not None: 
                        # 使用简单的归一化方式确保深度可视化正确
                        depth_continuous_norm = tb_log_depth_continuous_pred.clone().detach()
                        if depth_continuous_norm.min() != depth_continuous_norm.max():
                            depth_continuous_norm = (depth_continuous_norm - depth_continuous_norm.min()) / (depth_continuous_norm.max() - depth_continuous_norm.min())
                        metric_logger.log_image(f"val/sample{sample_idx}/depth_continuous", depth_continuous_norm, step=epoch)
                        
                        # 确保记录一条日志，帮助诊断是否正确获取了连续深度预测
                        depth_logger.info(f"Val sample {sample_idx}: 记录连续深度预测 shape={tb_log_depth_continuous_pred.shape}, range=[{tb_log_depth_continuous_pred.min().item():.4f}, {tb_log_depth_continuous_pred.max().item():.4f}]")
                    if tb_log_depth_conf_map is not None: 
                        metric_logger.log_image(f"val/sample{sample_idx}/depth_conf_map", tb_log_depth_conf_map, step=epoch)
                    
                    # 5. 深度对比图
                    if tb_log_depth_gt_img is not None:
                        depth_comparison_list = [normalize_depth(tb_log_depth_gt_img).cpu()]
                        if tb_log_depth_gate_pred is not None:
                            depth_comparison_list.append(tb_log_depth_gate_pred.cpu())
                        if tb_log_depth_continuous_pred is not None:
                            depth_comparison_list.append(normalize_depth(tb_log_depth_continuous_pred).cpu())
                        
                        if len(depth_comparison_list) > 1:
                            depth_comparison = torch.cat(depth_comparison_list, dim=-1)
                            metric_logger.log_image(f"val/sample{sample_idx}/depth_comparison", depth_comparison, step=epoch)
                    
        avg_epoch_loss = sum(epoch_total_batch_losses) / len(epoch_total_batch_losses) if epoch_total_batch_losses else 0.0
        final_metrics_summary = {"loss": avg_epoch_loss} # Start with loss
        
        metrics_config_from_yaml = config.get('validation', {}).get('metrics', {})
        aggregated_metric_values = defaultdict(list) # To store lists of metric values for averaging

        if collected_samples_for_processing:
            metrics_logger.info(f"Calculating detailed metrics for {len(collected_samples_for_processing)} validation samples...")
            # Loop through the collected samples to calculate detailed metrics
            for sample_data in tqdm(collected_samples_for_processing, desc="Calculating Detailed Val Metrics"):
                pred_img_tensor = sample_data['output'] # Expected CPU tensor
                gt_img_tensor = sample_data.get('gt')
                depth_pred_tensor = sample_data.get('depth_pred_continuous')
                depth_gt_tensor = sample_data.get('depth_gt')

                for metric_name, metric_func_handle in metrics_module.ALL_METRICS.items():
                    if metrics_config_from_yaml.get(metric_name, False):
                        try:
                            calculated_value = None # For single return value metrics
                            if metric_name in metrics_module.DEPTH_METRICS:
                                if depth_pred_tensor is not None and depth_gt_tensor is not None:
                                    if metric_name == "depth_delta": # This metric returns a dictionary
                                        delta_results_dict = metric_func_handle(depth_pred_tensor, depth_gt_tensor)
                                        for d_key, d_val in delta_results_dict.items():
                                            aggregated_metric_values[d_key].append(d_val) # Store each delta threshold result
                                        # No single `calculated_value` for depth_delta, it's handled above
                                    else: # Other depth metrics (MAE, RMSE)
                                        calculated_value = metric_func_handle(depth_pred_tensor, depth_gt_tensor)
                                else:
                                    debug_logger.debug(f"Skipping depth metric {metric_name} for {sample_data['filename_suffix']}: depth_pred or depth_gt is None.")
                            elif metric_name in metrics_module.FULL_REFERENCE_METRICS: # PSNR, SSIM, LPIPS, etc.
                                if gt_img_tensor is not None:
                                    calculated_value = metric_func_handle(pred_img_tensor, gt_img_tensor)
                                else:
                                    debug_logger.debug(f"Skipping FR metric {metric_name} for {sample_data['filename_suffix']}: gt_img is None.")
                            elif metric_name in metrics_module.NO_REFERENCE_METRICS: # UCIQE, NIQE, etc.
                                calculated_value = metric_func_handle(pred_img_tensor)
                            else:
                                warning_logger.warning(f"Metric {metric_name} not categorized as DEPTH, FR, or NR. Skipping.")

                            if calculated_value is not None: # If the metric returned a single value
                                if isinstance(calculated_value, (float, int)) and not (np.isnan(calculated_value) or np.isinf(calculated_value)):
                                    aggregated_metric_values[metric_name].append(calculated_value)
                                elif isinstance(calculated_value, (float, int)) and (np.isnan(calculated_value) or np.isinf(calculated_value)):
                                     warning_logger.warning(f"Metric {metric_name} for {sample_data['filename_suffix']} resulted in NaN/Inf, not included in average.")

                        except Exception as e_metric_calculation:
                            error_logger.error(f"Error calculating metric '{metric_name}' for sample {sample_data['filename_suffix']}: {e_metric_calculation}")
            
            # Average all collected metric values
            for name, values_list in aggregated_metric_values.items():
                if values_list:
                    final_metrics_summary[name] = sum(values_list) / len(values_list)
                else:
                    final_metrics_summary[name] = 0.0 # Default if no valid calculations (e.g., all samples skipped)
        else: # No samples collected for detailed metrics
            warning_logger.warning("No samples were collected for detailed metric calculations during validation.")

        # Log all averaged metrics to TensorBoard and console
        metric_logger.log_metrics(final_metrics_summary, prefix="val", step=epoch)
        
        # 记录不确定性权重（如果使用自动加权）
        if hasattr(criterion, 'use_uncertainty_weighting') and criterion.use_uncertainty_weighting:
            uncertainty_metrics = {}
            # 从损失对象中提取不确定性权重
            latest_losses = criterion.get_latest_losses()
            for key, value in latest_losses.items():
                if key.startswith('uncertainty_'):
                    uncertainty_metrics[key] = value
            
            if uncertainty_metrics:
                metric_logger.log_metrics(uncertainty_metrics, prefix="val_uncertainty", step=epoch)
                
        summary_print_string = ", ".join([f"{k}: {v:.4f}" for k, v in final_metrics_summary.items() if isinstance(v, (float, int))])
        print(f"Validation Epoch {epoch+1} Summary: {summary_print_string}")

        # --- Saving images to disk (using `collected_samples_for_processing`) ---
        save_metrics_text_on_disk_image = vis_config.get('val_images', {}).get('save_metrics', True)
        if save_val_imgs_to_disk and val_img_output_dir and collected_samples_for_processing:
            import matplotlib.pyplot as plt # Keep import local
            vis_logger.info(f"Saving {len(collected_samples_for_processing)} validation image sets to {val_img_output_dir}")
            
            for vis_item_data in tqdm(collected_samples_for_processing, desc="Saving Validation Images"):
                img_filename_suffix = vis_item_data.get('filename_suffix', f'sample_e{epoch}_unknownidx')
                
                # RGB Image comparison plot
                if vis_item_data.get('output') is not None:
                    num_img_subplots = 0
                    if vis_item_data.get('input') is not None: num_img_subplots += 1
                    if vis_item_data.get('output') is not None: num_img_subplots += 1
                    if vis_item_data.get('gt') is not None: num_img_subplots += 1
                    
                    if num_img_subplots > 0:
                        fig_rgb, axs_rgb = plt.subplots(1, num_img_subplots, figsize=(6 * num_img_subplots, 6), squeeze=False)
                        current_subplot_idx = 0
                        if vis_item_data.get('input') is not None:
                            img_in_np_rgb = metrics_module._prepare_image_for_metric(vis_item_data['input'], target_range_0_1=True, target_hwc=True)
                            axs_rgb[0, current_subplot_idx].imshow(img_in_np_rgb.squeeze()); axs_rgb[0, current_subplot_idx].set_title('Input'); axs_rgb[0, current_subplot_idx].axis('off'); current_subplot_idx+=1
                        
                        img_out_np_rgb = metrics_module._prepare_image_for_metric(vis_item_data['output'], target_range_0_1=True, target_hwc=True)
                        title_enhanced_img = 'Enhanced'
                        if save_metrics_text_on_disk_image:
                            psnr_val_str = f"{final_metrics_summary.get('psnr', 0.0):.2f}" 
                            ssim_val_str = f"{final_metrics_summary.get('ssim', 0.0):.4f}" 
                            title_enhanced_img += f"\nPSNR:{psnr_val_str}(avg) SSIM:{ssim_val_str}(avg)"
                        axs_rgb[0, current_subplot_idx].imshow(img_out_np_rgb.squeeze()); axs_rgb[0, current_subplot_idx].set_title(title_enhanced_img); axs_rgb[0, current_subplot_idx].axis('off'); current_subplot_idx+=1
                        
                        if vis_item_data.get('gt') is not None:
                            img_gt_np_rgb = metrics_module._prepare_image_for_metric(vis_item_data['gt'], target_range_0_1=True, target_hwc=True)
                            axs_rgb[0, current_subplot_idx].imshow(img_gt_np_rgb.squeeze()); axs_rgb[0, current_subplot_idx].set_title('Ground Truth'); axs_rgb[0, current_subplot_idx].axis('off')
                        
                        plt.tight_layout(); plt.savefig(os.path.join(val_img_output_dir, f'rgb_comparison_{img_filename_suffix}.png'), dpi=150); plt.close(fig_rgb)

                # Depth plots (continuous prediction vs GT)
                depth_pred_continuous_to_vis = vis_item_data.get('depth_pred_continuous')
                depth_gt_to_vis = vis_item_data.get('depth_gt')

                if depth_pred_continuous_to_vis is not None:
                    plt.figure(figsize=(7,6))
                    depth_pred_np_vis = metrics_module._prepare_image_for_metric(depth_pred_continuous_to_vis, target_range_0_1=False).squeeze()
                    if depth_pred_np_vis.max() > 1.0 or depth_pred_np_vis.min() < 0.0 or depth_pred_np_vis.max() == depth_pred_np_vis.min():
                        depth_pred_np_vis_norm = (depth_pred_np_vis - depth_pred_np_vis.min()) / (depth_pred_np_vis.max() - depth_pred_np_vis.min() + 1e-6)
                    else:
                        depth_pred_np_vis_norm = depth_pred_np_vis
                    plt.imshow(depth_pred_np_vis_norm.clip(0,1), cmap='viridis'); plt.colorbar(label='Predicted Depth (Normalized)'); plt.title('Predicted Continuous Depth'); plt.axis('off'); plt.tight_layout()
                    plt.savefig(os.path.join(val_img_output_dir, f'depth_continuous_pred_{img_filename_suffix}.png'), dpi=150); plt.close()
                
                if depth_gt_to_vis is not None and depth_pred_continuous_to_vis is not None:
                    fig_depth_comp, axs_depth_comp = plt.subplots(1, 3, figsize=(18,6))
                    depth_gt_np_vis = metrics_module._prepare_image_for_metric(depth_gt_to_vis, target_range_0_1=False).squeeze()
                    depth_pred_np_comp_vis = metrics_module._prepare_image_for_metric(depth_pred_continuous_to_vis, target_range_0_1=False).squeeze()
                    
                    def robust_norm(d_map):
                        if d_map.size == 0 or d_map.max() == d_map.min(): return np.zeros_like(d_map) if d_map.size > 0 else np.array([0.0])
                        return (d_map - d_map.min()) / (d_map.max() - d_map.min() + 1e-6)
                        
                    depth_gt_norm_for_plot = robust_norm(depth_gt_np_vis).clip(0,1)
                    depth_pred_norm_for_plot = robust_norm(depth_pred_np_comp_vis).clip(0,1)
                    
                    im0 = axs_depth_comp[0].imshow(depth_gt_norm_for_plot, cmap='viridis'); axs_depth_comp[0].set_title('GT Depth (Norm)'); axs_depth_comp[0].axis('off'); fig_depth_comp.colorbar(im0, ax=axs_depth_comp[0])
                    im1 = axs_depth_comp[1].imshow(depth_pred_norm_for_plot, cmap='viridis'); axs_depth_comp[1].set_title('Pred. Depth (Norm)'); axs_depth_comp[1].axis('off'); fig_depth_comp.colorbar(im1, ax=axs_depth_comp[1])
                    diff_map_norm = np.abs(depth_gt_norm_for_plot - depth_pred_norm_for_plot)
                    im2 = axs_depth_comp[2].imshow(diff_map_norm, cmap='hot'); axs_depth_comp[2].set_title('Abs Difference (Norm)'); axs_depth_comp[2].axis('off'); fig_depth_comp.colorbar(im2, ax=axs_depth_comp[2])
                    plt.tight_layout(); plt.savefig(os.path.join(val_img_output_dir, f'depth_comparison_continuous_{img_filename_suffix}.png'), dpi=150); plt.close(fig_depth_comp)

    # Return main loss and a primary metric (e.g. PSNR) for scheduler and best model tracking
    primary_metric_for_scheduler = final_metrics_summary.get('psnr', 0.0) 
    if not metrics_config_from_yaml.get('psnr', True) and epoch_total_batch_losses: 
        primary_metric_for_scheduler = -avg_epoch_loss 
    
    # 使用多文件日志系统记录详细分类的验证指标
    if multi_logger:
        metrics_dict = {
            'val_loss': avg_epoch_loss,
            'val_psnr': primary_metric_for_scheduler
        }
        # 添加其他指标
        for k, v in final_metrics_summary.items():
            if k != 'psnr':  # 已经添加过了
                metrics_dict[f'val_{k}'] = v
        
        # 基本指标记录
        multi_logger.log_metrics(metrics_dict, epoch, prefix="val")
        
        # 将验证指标按类别分组记录
        # 图像质量相关指标
        image_metrics = {}
        for k in ['psnr', 'ssim', 'lpips']:
            if k in final_metrics_summary:
                image_metrics[k] = final_metrics_summary[k]
        if image_metrics:
            multi_logger.log_validation(f"[验证] Epoch {epoch+1} 图像质量指标: " + 
                                       ", ".join([f"{k}={v:.6f}" for k, v in image_metrics.items()]))
        
        # 深度相关指标
        depth_metrics = {}
        for k in final_metrics_summary:
            if 'depth' in k.lower():
                depth_metrics[k] = final_metrics_summary[k]
        if depth_metrics:
            multi_logger.log_depth(f"[验证] Epoch {epoch+1} 深度评估指标: " + 
                                 ", ".join([f"{k}={v:.6f}" for k, v in depth_metrics.items()]))
        
        # 其他评估指标
        other_metrics = {}
        for k in final_metrics_summary:
            if not any(term in k.lower() for term in ['psnr', 'ssim', 'lpips', 'depth']):
                other_metrics[k] = final_metrics_summary[k]
        if other_metrics:
            multi_logger.log_validation(f"[验证] Epoch {epoch+1} 其他评估指标: " + 
                                       ", ".join([f"{k}={v:.6f}" for k, v in other_metrics.items()]))
        
    return avg_epoch_loss, primary_metric_for_scheduler


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
    set_seed(args.seed + rank)
    
    # 每个进程都运行主工作函数
    main_worker(config, args)


def resume_from_checkpoint(checkpoint_dir: str,
                         model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer = None,
                         scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                         device = None,
                         scaler = None) -> tuple:
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
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        
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


def main_worker(config, args):
    """主工作函数，负责训练和验证"""
    
    # 1. 设置训练环境
    setup_result = setup_training(args, config, args.local_rank)
    
    # 在DDP模式下，mp.spawn会为每个进程调用此函数。
    # setup_training会处理DDP进程的启动，完成后主进程可以退出
    if setup_result is None:
        return
        
    model = setup_result['model']
    criterion = setup_result['criterion']
    optimizer = setup_result['optimizer']
    scheduler = setup_result['scheduler']
    device = setup_result['device']
    exp_dir = setup_result['exp_dir']
    logger = setup_result['logger']
    metric_logger = setup_result['metric_logger']
    scaler = setup_result['scaler']
    mixed_precision = setup_result['mixed_precision']
    world_size = setup_result['world_size']
    local_rank = setup_result['local_rank']
    multi_logger = setup_result['multi_logger'] # 获取multi_logger
    
    # 2. 准备数据
    data_loaders = prepare_data(config, args)
    train_loader = data_loaders['train_loader']
    val_loader = data_loaders['val_loader']
    train_sampler = data_loaders['train_sampler']

    # 3. 恢复训练 (如果需要)
    start_epoch = 0
    best_metric = 0.0
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    
    # 打印日志，确认是否尝试恢复训练
    if args.resume:
        logger.info(f"正在尝试从检查点目录 '{checkpoint_dir}' 恢复训练...")
        model, optimizer, scheduler, scaler, start_epoch, best_metric = resume_from_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler
        )
    else:
        logger.info("不从检查点恢复，将从头开始训练。")
        
    # 如果只进行评估，直接进入验证流程
    if args.eval_only:
        logger.info("进入仅验证模式...")
        validate(val_loader, model, criterion, device, metric_logger, start_epoch, config, mixed_precision, multi_logger)
        logger.info("验证完成。")
        return
        
    # 记录模型图 (在开始训练前)
    # 确保只在主进程上执行
    if local_rank in [-1, 0]:
        try:
            # 由于模型包含动态操作，暂时跳过模型图记录
            # 未来可以考虑使用torch.onnx.export或手动创建静态图
            logger.info("跳过模型图记录 - 模型包含动态操作，无法被PyTorch JIT trace")
            
            # 可选：记录模型的基本信息
            if hasattr(model, 'module'):
                actual_model = model.module
            else:
                actual_model = model
                
            # 统计模型参数
            total_params = sum(p.numel() for p in actual_model.parameters())
            trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
            
            metric_logger.log_text('model_info', 
                f"模型参数统计:\n"
                f"- 总参数数量: {total_params:,}\n"
                f"- 可训练参数: {trainable_params:,}\n"
                f"- 模型类型: {type(actual_model).__name__}", 
                step=0)
            
        except Exception as e:
            logger.error(f"记录模型信息失败: {e}", exc_info=True)
            
    # 4. 训练循环
    logger.info("="*20 + " 开始训练 " + "="*20)
    for epoch in range(start_epoch, config['train']['epochs']):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            
        # 训练
        train_loss = train_epoch(
            train_loader, model, criterion, optimizer, device,
            metric_logger, epoch, config, scaler, mixed_precision, multi_logger
        )
        
        # 验证
        val_results = validate(
            val_loader, model, criterion, device, metric_logger,
            epoch, config, mixed_precision, multi_logger
        )
        
        # 从验证结果中提取损失和主要指标
        val_loss, val_metric = val_results

        # 更新学习率
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metric) # 基于验证指标更新
            else:
                scheduler.step() # 基于epoch更新
        
        # 保存检查点
        if local_rank in [-1, 0]:
            is_best = val_metric > best_metric
            if is_best:
                best_metric = val_metric
            
            save_checkpoint(
                state={
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_metric': best_metric,
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                },
                is_best=is_best,
                checkpoint_dir=checkpoint_dir,
                filename=f"checkpoint_epoch_{epoch+1}.pth"
            )
            logger.info(f"Epoch {epoch+1}: 已保存检查点，当前PSNR: {val_metric:.4f}, 最佳PSNR: {best_metric:.4f}")

    logger.info("="*20 + " 训练完成 " + "="*20)
    metric_logger.close()


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置文件
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 设置随机种子
    set_seed(args.seed)

    # 分布式训练处理
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
    else:
        # 单GPU或CPU训练
        args.local_rank = -1
        main_worker(config, args)

if __name__ == '__main__':
    main()


