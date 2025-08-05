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
    # 格式化警告信息，生成类似 "'filename:lineno: category: message'" 的字符串
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
from modules.loss_fn import TotalLoss
from modules.depth import get_depth_config_params, ensure_normalized_depth
from utils.checkpoint import save_checkpoint
from utils.lr_scheduler import get_scheduler
from utils.metrics import calculate_psnr as compute_psnr, calculate_ssim as compute_ssim, calculate_depth_statistics
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


def find_latest_experiment_with_checkpoints(output_dir, base_name):
    """
    在输出目录中查找最新的包含检查点的实验目录
    
    Args:
        output_dir: 实验输出根目录
        base_name: 实验基础名称 (如 'underwater_enhance_run')
        
    Returns:
        str or None: 找到的最新实验目录路径，如果没有找到返回None
    """
    if not os.path.exists(output_dir):
        return None
    
    # 查找所有匹配的实验目录
    matching_dirs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith(base_name):
            # 检查是否有检查点目录且不为空
            checkpoint_dir = os.path.join(item_path, 'checkpoints')
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(('.pth', '.pth.tar'))]
                if checkpoints:
                    # 记录目录修改时间用于排序
                    mtime = os.path.getmtime(item_path)
                    matching_dirs.append((item_path, mtime))
    
    # 按修改时间排序，返回最新的
    if matching_dirs:
        matching_dirs.sort(key=lambda x: x[1], reverse=True)
        latest_dir = matching_dirs[0][0]
        return latest_dir
    
    return None


def setup_experiment_dir(config, resume_mode=False):
    """
    设置实验目录并保存配置，支持自动添加时间戳或序号避免覆盖
    
    Args:
        config: 配置字典
        resume_mode: 是否为恢复训练模式
        
    Returns:
        str: 实验目录路径
    """
    exp_name = config['experiment']['name']
    output_dir = config['experiment']['output_dir']
    
    # 🔥 智能恢复逻辑：如果是恢复模式，优先查找现有实验
    if resume_mode:
        latest_exp_dir = find_latest_experiment_with_checkpoints(output_dir, exp_name)
        if latest_exp_dir:
            print(f"[智能恢复] 找到包含检查点的最新实验目录: {latest_exp_dir}")
            print(f"[智能恢复] 将恢复训练而不是创建新目录")
            return latest_exp_dir
        else:
            print(f"[智能恢复] 未找到包含检查点的实验目录，将创建新实验")
    
    # 原有逻辑：创建新实验目录
    if config['experiment'].get('auto_naming', True):
        timestamp_format = config['experiment'].get('timestamp_format', "%Y%m%d_%H%M%S")
        timestamp = datetime.now().strftime(timestamp_format)
        exp_name = f"{exp_name}_{timestamp}"
    else:
        # 检查是否已存在相同名称的实验目录，如果存在则添加序号
        base_dir = os.path.join(output_dir, exp_name)
        if os.path.exists(base_dir):
            i = 1
            while os.path.exists(os.path.join(output_dir, f"{exp_name}_{i}")):
                i += 1
            exp_name = f"{exp_name}_{i}"
    
    exp_dir = os.path.join(output_dir, exp_name)
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
    
    # 🔧 应用交叉注意力配置
    cross_attention_config = config.get('loss', {}).get('cross_attention', {})
    if cross_attention_config.get('enable_saving', False):
        model.enable_attention_saving(True)
        logger.info(f"✓ 启用注意力图保存功能")
    
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
        lambda_cons=config['loss'].get('lambda_cons', 0.05),  # 🔧 取消注释，添加注意力一致性损失权重
        
        # 🔥 CMCL相关参数
        lambda_cmcl=config['loss'].get('lambda_cmcl', 0.1),
        lambda_cmcl_var=config['loss'].get('lambda_cmcl_var', 1.0),
        lambda_cmcl_rgb=config['loss'].get('lambda_cmcl_rgb', 1.0),
        lambda_cmcl_depth=config['loss'].get('lambda_cmcl_depth', 1.0),
        cmcl_k_decay=config['loss'].get('cmcl_k_decay', 1.0),
        
        # 🔧 添加交叉注意力配置传递
        cross_attention_config=config['loss'].get('cross_attention', None),

        use_uncertainty_weighting=config['loss'].get('use_uncertainty_weighting', True),  # 从配置中读取是否使用自动调参
        sigma_init=config['loss'].get('sigma_init', None),  # 从配置中读取不确定性初始值
        min_depth=config['loss']['depth_processing'].get('min_depth', 5000.0),  # 从配置中读取深度范围
        max_depth=config['loss']['depth_processing'].get('max_depth', 65000.0)  # 从配置中读取深度范围
    )
    criterion = criterion.to(device)
    
    # 5. 优化器
    optimizer_name = config['optimizer'].get('name', 'adamw').lower()
    
    # 9. 实验目录将在main_worker中创建，日志系统也将在那里初始化
    
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
    physics_lr_scale = config['optimizer'].get('physics_lr_scale', 1.0)  # 物理参数默认使用相同学习率
    
    # 差异化学习率设置（详细日志将在main_worker中记录）
    print(f"使用差异化学习率: 主干 {config['optimizer']['lr']}, "
          f"注意力模块 {config['optimizer']['lr'] * attn_lr_scale}, "
          f"物理模块 {config['optimizer']['lr'] * physics_lr_scale}")
    
    # 添加损失函数中的不确定性权重参数到优化器中
    uncertainty_params = []
    if hasattr(criterion, 'use_uncertainty_weighting') and criterion.use_uncertainty_weighting:
        # 收集所有不确定性权重参数
        for name, p in criterion.named_parameters():
            if 'log_var' in name:
                uncertainty_params.append(p)
        
        if uncertainty_params:
            print(f"将 {len(uncertainty_params)} 个不确定性权重参数添加到优化器中")
            # 输出每个不确定性权重参数的名称
            param_names = [name for name, p in criterion.named_parameters() if 'log_var' in name]
            print(f"不确定性权重参数列表: {param_names}")
    
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
        
        # 统计优化器中不同参数组的参数数量
        base_param_count = sum(p.numel() for p in base_params)
        attn_param_count = sum(p.numel() for p in attn_params)
    physics_param_count = sum(p.numel() for p in physics_params) if physics_params else 0
    uncertainty_param_count = sum(p.numel() for p in uncertainty_params) if uncertainty_params else 0
    total_param_count = base_param_count + attn_param_count + physics_param_count + uncertainty_param_count
        
    print("优化器参数统计:")
    print(f"  主干参数: {base_param_count:,} ({base_param_count/total_param_count*100:.2f}%)")
    print(f"  注意力参数: {attn_param_count:,} ({attn_param_count/total_param_count*100:.2f}%)")
    if physics_param_count > 0:
        print(f"  物理参数: {physics_param_count:,} ({physics_param_count/total_param_count*100:.2f}%)")
    if uncertainty_param_count > 0:
        print(f"  不确定性权重参数: {uncertainty_param_count:,} ({uncertainty_param_count/total_param_count*100:.2f}%)")
    print(f"  总参数数量: {total_param_count:,}")
    
    # 记录模型架构信息
    print(f"模型类型: {type(model).__name__}")
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'channels'):
        print(f"编码器通道数: {model.encoder.channels}")
    
    # 添加动态参数管理功能
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
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            param_groups,
            betas=(config['optimizer'].get('beta1', 0.9), config['optimizer'].get('beta2', 0.999)),
            weight_decay=config['optimizer'].get('weight_decay', 0)
        )
        print(f"使用Adam优化器，权重衰减: {config['optimizer'].get('weight_decay', 0)}")
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=config['optimizer'].get('weight_decay', 0.01),
            betas=(config['optimizer'].get('beta1', 0.9), config['optimizer'].get('beta2', 0.999))
        )
        print(f"使用AdamW优化器，权重衰减: {config['optimizer'].get('weight_decay', 0.01)}")
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            momentum=config['optimizer'].get('momentum', 0.9),
            weight_decay=config['optimizer'].get('weight_decay', 0.0001),
            nesterov=config['optimizer'].get('nesterov', False)
        )
        print(f"使用SGD优化器，动量: {config['optimizer'].get('momentum', 0.9)}, "
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
    
    print(f"使用学习率调度器: {config['scheduler']['name']}")
    print(f"  总训练轮次: {config['train']['epochs']}")
    if config['scheduler'].get('warmup_epochs', 0) > 0:
        print(f"  预热轮次: {config['scheduler'].get('warmup_epochs', 0)}")
    print(f"  最小学习率: {config['scheduler'].get('min_lr', 0)}")
    
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
        print("使用DataParallel进行多GPU训练，这比DDP效率低。建议使用 --distributed 参数启用DDP。")
        device_ids = gpu_config.get('device_ids', None)
        model = nn.DataParallel(model, device_ids=device_ids)
        print(f"使用DataParallel进行多GPU训练, device_ids={device_ids}")
    
    # 日志记录将在main_worker中处理，setup_training只负责基础组件初始化
    
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
        'update_optimizer_fn': update_optimizer_with_new_params
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
               epoch, config, scaler=None, mixed_precision=False, multi_logger=None, 
               update_optimizer_fn=None):
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
    
    # 从配置文件获取TensorBoard记录频率控制
    tb_config = config.get('visualization', {}).get('tensorboard', {})
    train_metrics_freq = tb_config.get('train_metrics_freq', 1)
    physics_config = tb_config.get('physics_metrics', {})
    physics_enable = physics_config.get('enable', True)
    physics_freq = physics_config.get('freq', 1)
    params_config = tb_config.get('model_params', {})
    params_enable = params_config.get('enable', False)
    params_freq = params_config.get('freq', 500)
    grads_config = tb_config.get('gradients', {})
    grads_enable = grads_config.get('enable', False)
    grads_freq = grads_config.get('freq', 500)
    uncertainty_config = tb_config.get('uncertainty_weights', {})
    uncertainty_enable = uncertainty_config.get('enable', True)
    uncertainty_freq = uncertainty_config.get('freq', 10)
    
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
        
        # 新增：检查是否启用多输入一致性学习
        enable_multi_input_consistency = config.get('multi_input_consistency', {}).get('enable', False)
        
        if isinstance(batch, dict):
            raw_imgs = batch['raw_imgs'].to(device)  # Shape: [B, N, C, H, W]
            depth_gt = batch['depth'].to(device) if 'depth' in batch else None
            gt = batch['gt'].to(device) if 'gt' in batch else None
            
                    # 🔥 多输入处理逻辑更新
        is_multi_input = raw_imgs.dim() == 5 and raw_imgs.shape[1] > 1
        
        if enable_multi_input_consistency and is_multi_input:
            # 🔥 多输入一致性学习模式：保持5D张量，传递给模型
            if data_logger and i % 100 == 0:
                data_logger.info(f"多输入一致性学习模式: raw_imgs.shape={raw_imgs.shape}")
            # 不修改raw_imgs，保持[B, N, C, H, W]格式
            # depth_gt和gt也保持原有维度或适当调整
        elif not enable_multi_input_consistency and is_multi_input:
            # 传统模式：取第一个退化
            raw_imgs = raw_imgs[:, 0]  # [B, N, C, H, W] -> [B, C, H, W]
            if depth_gt is not None and depth_gt.dim() == 5:
                depth_gt = depth_gt[:, 0]
            if gt is not None and gt.dim() == 5:
                gt = gt[:, 0]
        elif raw_imgs.dim() == 5 and raw_imgs.shape[1] == 1:
            # 单退化情况
            raw_imgs = raw_imgs.squeeze(1)
        else:
            raw, depth_gt_tuple, gt_tuple = batch[:3] # Renamed to avoid conflict
            raw_imgs = raw.to(device)
            depth_gt = depth_gt_tuple.to(device) if depth_gt_tuple is not None else None
            gt = gt_tuple.to(device) if gt_tuple is not None else None
            # Handle potential 5D tensor from dataset
            if raw_imgs.dim() == 5 and raw_imgs.shape[1] == 1:
                raw_imgs = raw_imgs.squeeze(1)  # [B, N, C, H, W] -> [B, C, H, W]
            elif raw_imgs.dim() == 5:
                raw_imgs = raw_imgs[:, 0]  # Use first degradation
        
        # 🔥 确保B变量在所有分支中都被定义
        B = raw_imgs.shape[0]
        
        # 记录输入数据
        log_input_data(raw_imgs, "raw_imgs", current_step)
        log_input_data(depth_gt, "depth_gt", current_step)  
        log_input_data(gt, "gt_imgs", current_step)
        
        optimizer.zero_grad()
        
        # 在第一个step后检查是否有新的动态参数需要添加到优化器中
        if current_step == 1 and update_optimizer_fn is not None:
            model_to_check = model.module if hasattr(model, 'module') else model
            update_optimizer_fn(model_to_check, optimizer)
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        optimizer_logger.info(f"Step {current_step}: 当前学习率: {current_lr:.6f}")
        
        # 根据配置控制训练指标记录频率
        if current_step % train_metrics_freq == 0:
            metric_logger.log_metrics({"lr": current_lr}, prefix="optimizer", step=current_step)
        
        if mixed_precision and scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                # 🔥 使用新的前向传播，支持多输入一致性
                if enable_multi_input_consistency and is_multi_input:
                    outputs = model(raw_imgs, depth_gt, gt, enable_multi_input_consistency=True)
                    consistency_features = getattr(outputs, 'consistency_features', None)
                    # 对于损失计算，如果是多输入模式，需要处理GT的维度
                    if gt is not None and gt.dim() == 4:  # [B, C, H, W]
                        # GT需要与多输入的第一个退化对应
                        gt_for_loss = gt  # 使用原始GT
                    elif gt is not None and gt.dim() == 5:  # [B, N, C, H, W]
                        gt_for_loss = gt[:, 0]  # 取第一个作为主要GT
                    else:
                        gt_for_loss = gt
                else:
                    outputs = model.multi_forward(raw_imgs, depth_gt, gt)
                    consistency_features = None
                    gt_for_loss = gt
                    
                enhanced = outputs.enhanced
                pred_gate = outputs.pred_gate
                student_feats = outputs.student_feats
                depth_conf_map = outputs.depth_conf_map
                attention_maps = outputs.attention_maps
                depth_pred = outputs.depth_pred
                
                # 🔥 计算损失（处理多输入一致性）
                loss = criterion(
                    outputs.enhanced, gt_for_loss, 
                    depth_gt=depth_gt if not is_multi_input or not enable_multi_input_consistency else (depth_gt[:, 0] if depth_gt is not None and depth_gt.dim() == 5 else depth_gt),
                    student_feats=student_feats,
                    attention_maps=attention_maps,
                    depth_pred=depth_pred,
                    depth_conf_map=depth_conf_map,
                    raw=raw_imgs if not is_multi_input or not enable_multi_input_consistency else raw_imgs[:, 0],

                )
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度裁剪 (如果配置)
            if config['optimizer'].get('clip_grad_norm', 0) > 0:
                # 在 unscale 之前裁剪
                scaler.unscale_(optimizer)
                clip_norm_val = torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['clip_grad_norm'])
                gpu_logger.info(f"Step {current_step}: 梯度范数 (裁剪前): {clip_norm_val:.4f}")
                if current_step % train_metrics_freq == 0:
                    metric_logger.log_metrics({"grad_norm": clip_norm_val.item()}, prefix="gpu", step=current_step)

            scaler.step(optimizer)
            scaler.update()
            
            # 清理梯度和计算图
            optimizer.zero_grad()
        else:
            # 🔥 常规训练，支持多输入一致性
            if enable_multi_input_consistency and is_multi_input:
                outputs = model(raw_imgs, depth_gt, gt, enable_multi_input_consistency=True)
                consistency_features = getattr(outputs, 'consistency_features', None)
            else:
                outputs = model.multi_forward(raw_imgs, depth_gt, gt)
                consistency_features = None
                
            enhanced = outputs.enhanced
            pred_gate = outputs.pred_gate
            student_feats = outputs.student_feats
            depth_conf_map = outputs.depth_conf_map
            attention_maps = outputs.attention_maps
            depth_pred = outputs.depth_pred
            
            # 计算损失（对整个批次）
            # 🔥 支持残差级别一致性损失
            multi_enhanced = getattr(outputs, 'multi_enhanced', None)
            multi_res_d = getattr(outputs, 'multi_res_d', None)
            multi_res_c = getattr(outputs, 'multi_res_c', None)
            
            loss = criterion(
                outputs.enhanced, gt, 
                depth_gt=depth_gt,
                student_feats=student_feats,
                attention_maps=attention_maps,
                depth_pred=depth_pred,
                depth_conf_map=depth_conf_map,
                raw=raw_imgs,

                multi_enhanced=multi_enhanced,              # 🔥 多输入增强结果
                multi_res_d=multi_res_d,                    # 🔥 多输入去模糊残差
                multi_res_c=multi_res_c                     # 🔥 多输入颜色校正残差
            )
            
            # 反向传播
            loss.backward()

            # 梯度裁剪 (如果配置)
            if config['optimizer'].get('clip_grad_norm', 0) > 0:
                clip_norm_val = torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['clip_grad_norm'])
                gpu_logger.info(f"Step {current_step}: 梯度范数 (裁剪前): {clip_norm_val:.4f}")
                if current_step % train_metrics_freq == 0:
                    metric_logger.log_metrics({"grad_norm": clip_norm_val.item()}, prefix="gpu", step=current_step)

            optimizer.step()
            
            # 清理梯度和计算图
            optimizer.zero_grad()
        
        current_loss = loss.item()
        epoch_loss += current_loss
        step_count += 1
        
        # 清理计算图引用，防止重复反向传播错误
        del loss
        
        # 安全地清理模型状态，只清理自定义状态，不触碰PyTorch内部状态
        def clear_model_state_safe(model):
            """安全地清理模型中的自定义状态，避免破坏PyTorch内部结构"""
            for name, module in model.named_modules():
                # 清理注意力图
                if hasattr(module, 'last_attn'):
                    module.last_attn = None
                # 清理自定义缓存（但避免触碰PyTorch内部属性）
                if hasattr(module, '_cached_features'):
                    module._cached_features = None
                # 清理RNN/LSTM状态
                if hasattr(module, 'hidden'):
                    module.hidden = None
        
        clear_model_state_safe(model)
        
        if mixed_precision:
            torch.cuda.empty_cache()
        
        progress_bar.set_postfix({"Loss": f"{current_loss:.4f}"})
        
        metrics = {"loss": current_loss}
        
        # 记录物理参数到TensorBoard (支持固定参数和动态预测参数)
        # 根据配置控制物理参数记录频率
        if physics_enable and (current_step % physics_freq == 0):
            # 获取实际模型（处理DataParallel）
            if hasattr(model, 'module'):
                actual_model = model.module
            else:
                actual_model = model
            
            # 🔥 记录物理参数
            try:
                # 获取物理参数
                if hasattr(actual_model, 'physics_head'):
                    # 创建一个dummy输入来获取物理参数
                    with torch.no_grad():
                        # 使用当前batch的瓶颈特征
                        if hasattr(actual_model, 'bottleneck') and len(student_feats) > 0:
                            # 使用最后一层特征作为瓶颈输入
                            bottleneck_feat = actual_model.bottleneck(student_feats[-1])
                            
                            # 检查输入特征是否包含NaN或Inf
                            if torch.isnan(bottleneck_feat).any() or torch.isinf(bottleneck_feat).any():
                                physics_logger.warning(f"Step {current_step}: bottleneck_feat包含NaN或Inf，跳过物理参数记录")
                                continue
                            
                            beta_c, B_c, blur_scale = actual_model.physics_head(bottleneck_feat)
                            
                            # 检查物理参数是否包含NaN或Inf
                            if (torch.isnan(beta_c).any() or torch.isinf(beta_c).any() or
                                torch.isnan(B_c).any() or torch.isinf(B_c).any() or
                                torch.isnan(blur_scale).any() or torch.isinf(blur_scale).any()):
                                physics_logger.warning(f"Step {current_step}: 物理参数包含NaN或Inf")
                                physics_logger.warning(f"  beta_c: NaN={torch.isnan(beta_c).any()}, Inf={torch.isinf(beta_c).any()}")
                                physics_logger.warning(f"  B_c: NaN={torch.isnan(B_c).any()}, Inf={torch.isinf(B_c).any()}")
                                physics_logger.warning(f"  blur_scale: NaN={torch.isnan(blur_scale).any()}, Inf={torch.isinf(blur_scale).any()}")
                                continue
                            
                            # 安全计算统计信息
                            def safe_stat(tensor, stat_name):
                                try:
                                    if stat_name == 'mean':
                                        return tensor.mean().item()
                                    elif stat_name == 'std':
                                        # 处理单元素张量的情况
                                        if tensor.numel() <= 1:
                                            return 0.0  # 单元素的标准差定义为0
                                        std_val = tensor.std().item()
                                        return 0.0 if torch.isnan(torch.tensor(std_val)) else std_val
                                    elif stat_name == 'min':
                                        return tensor.min().item()
                                    elif stat_name == 'max':
                                        return tensor.max().item()
                                except:
                                    physics_logger.warning(f"计算{stat_name}失败，返回0")
                                    return 0.0
                            
                            # 记录物理参数统计信息
                            physics_metrics = {
                                'beta_c_mean': safe_stat(beta_c, 'mean'),
                                'beta_c_std': safe_stat(beta_c, 'std'),
                                'beta_c_min': safe_stat(beta_c, 'min'),
                                'beta_c_max': safe_stat(beta_c, 'max'),
                                'B_c_mean': safe_stat(B_c, 'mean'),
                                'B_c_std': safe_stat(B_c, 'std'),
                                'B_c_min': safe_stat(B_c, 'min'),
                                'B_c_max': safe_stat(B_c, 'max'),
                                'blur_scale_mean': safe_stat(blur_scale, 'mean'),
                                'blur_scale_std': safe_stat(blur_scale, 'std'),
                                'blur_scale_min': safe_stat(blur_scale, 'min'),
                                'blur_scale_max': safe_stat(blur_scale, 'max'),
                            }
                            
                            # 记录到TensorBoard
                            metric_logger.log_metrics(physics_metrics, prefix="physics", step=current_step)
                            
                            # 记录到物理日志
                            physics_logger.info(f"[训练] Step {current_step} 物理参数: " + 
                                              f"beta_c=[{beta_c.min().item():.3f}, {beta_c.max().item():.3f}], " +
                                              f"B_c=[{B_c.min().item():.3f}, {B_c.max().item():.3f}], " +
                                              f"blur_scale=[{blur_scale.min().item():.3f}, {blur_scale.max().item():.3f}]")
                            
                            # 记录物理参数的分布直方图（可选）
                            physics_save_histograms = physics_config.get('save_histograms', True)
                            if (physics_save_histograms and 
                                hasattr(metric_logger, 'tb_writer') and 
                                metric_logger.tb_writer is not None):
                                metric_logger.tb_writer.add_histogram('physics/beta_c_distribution', beta_c, current_step)
                                metric_logger.tb_writer.add_histogram('physics/B_c_distribution', B_c, current_step)
                                metric_logger.tb_writer.add_histogram('physics/blur_scale_distribution', blur_scale, current_step)
                        
                else:
                    physics_logger.warning(f"模型没有physics_head属性，跳过物理参数记录")
                    
            except Exception as e:
                physics_logger.error(f"记录物理参数时出错: {e}")
                import traceback
                physics_logger.debug(traceback.format_exc())
        
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
        
        
        # 根据配置控制训练指标记录频率
        if current_step % train_metrics_freq == 0:
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
                for k in ['depth_pred_loss', 'depth_smooth_loss', 'depth_decoder_loss', 'depth_rec_loss', 'multi_candidate_depth_rec_loss', 'depth_total_loss']:
                    if k in loss_components:
                        depth_metrics[k] = loss_components[k]
                if depth_metrics:
                    depth_logger.info(f"[训练] Step {current_step} 深度损失: " + 
                                    ", ".join([f"{k}={v:.6f}" for k, v in depth_metrics.items()]))
                

                
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
                        if uncertainty_enable and (current_step % uncertainty_freq == 0):
                            metric_logger.log_metrics(uncertainty_metrics, prefix="uncertainty", step=current_step)
        
        if i % vis_interval == 0:
            try:
                vis_logger.info(f"======== 开始记录可视化数据，步骤: {current_step} ========")
                vis_raw_first_item = raw_imgs[0].unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
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
        
        # 根据配置控制参数分布记录频率
        if params_enable and (current_step % params_freq == 0):
            metric_logger.log_model_parameters(model, step=current_step, log_gradients=grads_enable)
        
        # 记录各种指标和参数
        
        # 记录损失
        metrics = criterion.get_latest_losses()  # 获取损失函数记录的最新损失值
        
        # 深度相关指标记录
        if outputs.depth_pred is not None and depth_gt is not None:
            try:
                depth_stats = calculate_depth_statistics(outputs.depth_pred, depth_gt)
                for stat_key, stat_value in depth_stats.items():
                    metrics[f"depth/{stat_key}"] = stat_value
                
                if current_step % 100 == 0:
                    multi_logger.get_logger('depth').info(f"Depth stats: {depth_stats}")
            except Exception as e:
                multi_logger.get_logger('warning').warning(f"Error computing depth statistics: {e}")
        elif outputs.depth_pred is not None:
            # 如果只有深度预测但没有GT，记录深度预测的基本统计信息
            try:
                depth_pred_stats = {
                    'pred_min': outputs.depth_pred.min().item(),
                    'pred_max': outputs.depth_pred.max().item(),
                    'pred_mean': outputs.depth_pred.mean().item(),
                    'pred_std': outputs.depth_pred.std().item()
                }
                for stat_key, stat_value in depth_pred_stats.items():
                    metrics[f"depth/{stat_key}"] = stat_value
                    
                if current_step % 100 == 0:
                    multi_logger.get_logger('depth').info(f"Depth prediction stats: {depth_pred_stats}")
            except Exception as e:
                multi_logger.get_logger('warning').warning(f"Error computing depth prediction statistics: {e}")
        
        # 物理参数记录已删除
        
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
        
        
        # 根据配置控制训练指标记录频率
        if current_step % train_metrics_freq == 0:
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
                for k in ['depth_pred_loss', 'depth_smooth_loss', 'depth_decoder_loss', 'depth_rec_loss', 'multi_candidate_depth_rec_loss', 'depth_total_loss']:
                    if k in loss_components:
                        depth_metrics[k] = loss_components[k]
                if depth_metrics:
                    depth_logger.info(f"[验证] Step {current_step} 深度损失: " + 
                                    ", ".join([f"{k}={v:.6f}" for k, v in depth_metrics.items()]))
                
                # 物理模型相关损失已删除
                
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
        
    # 计算平均损失
    avg_epoch_loss = epoch_loss / step_count if step_count > 0 else 0.0
    
    # 主要指标用于学习率调度器（通常使用平均损失）
    primary_metric_for_scheduler = avg_epoch_loss
    
    return avg_epoch_loss, primary_metric_for_scheduler


def validate(val_loader, model, criterion, device, metric_logger, epoch, config, 
             mixed_precision=False, multi_logger=None):
    """
    验证函数，计算RGB和深度的各种指标并记录到TensorBoard
    
    Returns:
        tuple: (avg_val_loss, primary_metric)
    """
    model.eval()
    
    # 获取日志记录器
    if multi_logger:
        val_logger = multi_logger.get_logger('validation')
        metrics_logger = multi_logger.get_logger('metrics')
        depth_logger = multi_logger.get_logger('depth')
    else:
        val_logger = metrics_logger = depth_logger = logging.getLogger(__name__)
    
    # 获取可视化配置
    vis_config = config.get('visualization', {})
    val_vis_interval = vis_config.get('val_vis_interval', 50)
    max_val_vis_samples = vis_config.get('max_val_vis_samples', 10)
    val_metrics_freq = vis_config.get('tensorboard', {}).get('val_metrics_freq', 1)
    
    # 获取验证图像保存配置
    val_images_config = vis_config.get('val_images', {})
    save_val_images = val_images_config.get('save', False)
    max_val_samples = val_images_config.get('max_samples', 8)
    save_comparison = val_images_config.get('save_comparison', True)
    
    # 创建验证图像保存目录
    val_images_dir = None
    if save_val_images:
        exp_dir = config.get('experiment', {}).get('output_dir', 'experiments/train')
        exp_name = config.get('experiment', {}).get('name', 'underwater_enhance_run')
        # 找到当前实验目录
        import glob
        exp_pattern = os.path.join(exp_dir, f"{exp_name}_*")
        exp_dirs = glob.glob(exp_pattern)
        if exp_dirs:
            current_exp_dir = max(exp_dirs, key=os.path.getctime)  # 最新的实验目录
            val_images_dir = os.path.join(current_exp_dir, 'val_images', f'epoch_{epoch+1:03d}')
            os.makedirs(val_images_dir, exist_ok=True)
            val_logger.info(f"验证图像将保存到: {val_images_dir}")
    
    val_logger.info(f"开始验证 Epoch {epoch+1}...")
    
    total_loss = 0.0
    num_batches = 0
    
    # RGB指标累积
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    
    # 深度指标累积
    total_depth_mae = 0.0
    total_depth_rmse = 0.0
    total_depth_abs_rel = 0.0
    total_depth_sq_rel = 0.0
    
    # 损失组件累积
    loss_components_sum = defaultdict(float)
    
    vis_count = 0  # 可视化计数器
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"验证 Epoch {epoch+1}")
        
        for i, batch in enumerate(progress_bar):
            # 解析批次数据 - 处理字典格式的batch
            if isinstance(batch, dict):
                # MultiDegradationDataset 返回字典格式
                raw_imgs = batch['raw_imgs'].to(device)  # [B, N, C, H, W]
                depth_gt = batch['depth'].to(device) if batch['depth'] is not None else None
                gt = batch['gt'].to(device) if batch['gt'] is not None else None
                
                # 🔥 处理5D张量 - 验证时暂时还是选择第一个退化级别
                # 未来可以扩展为验证所有退化级别
                if raw_imgs.dim() == 5:
                    raw_imgs = raw_imgs[:, 0]  # [B, C, H, W]
                    
            elif len(batch) >= 4:  # 其他数据集格式
                raw_imgs, depth_gt_tuple, gt_tuple, _ = batch[:4]
                raw_imgs = raw_imgs.to(device)
                depth_gt = depth_gt_tuple.to(device) if depth_gt_tuple is not None else None
                gt = gt_tuple.to(device) if gt_tuple is not None else None
                
                # 处理5D张量
                if raw_imgs.dim() == 5 and raw_imgs.shape[1] == 1:
                    raw_imgs = raw_imgs.squeeze(1)
                elif raw_imgs.dim() == 5:
                    raw_imgs = raw_imgs[:, 0]
            else:
                raw, depth_gt_tuple, gt_tuple = batch[:3]
                raw_imgs = raw.to(device)
                depth_gt = depth_gt_tuple.to(device) if depth_gt_tuple is not None else None
                gt = gt_tuple.to(device) if gt_tuple is not None else None
                
                if raw_imgs.dim() == 5 and raw_imgs.shape[1] == 1:
                    raw_imgs = raw_imgs.squeeze(1)
                elif raw_imgs.dim() == 5:
                    raw_imgs = raw_imgs[:, 0]
            
            B = raw_imgs.shape[0]
            
            # 前向传播
            if mixed_precision:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model.multi_forward(raw_imgs, depth_gt, gt)
                    
                    # 🔥 支持残差级别一致性损失
                    multi_enhanced = getattr(outputs, 'multi_enhanced', None)
                    multi_res_d = getattr(outputs, 'multi_res_d', None)
                    multi_res_c = getattr(outputs, 'multi_res_c', None)
                    
                    loss = criterion(
                        outputs.enhanced, gt,
                        depth_gt=depth_gt,
                        student_feats=outputs.student_feats,
                        attention_maps=outputs.attention_maps,
                        depth_pred=outputs.depth_pred,
                        depth_conf_map=outputs.depth_conf_map,
                        raw=raw_imgs,
                        multi_enhanced=multi_enhanced,              # 🔥 多输入增强结果
                        multi_res_d=multi_res_d,                    # 🔥 多输入去模糊残差
                        multi_res_c=multi_res_c                     # 🔥 多输入颜色校正残差
                    )
            else:
                outputs = model.multi_forward(raw_imgs, depth_gt, gt)
                
                # 🔥 支持残差级别一致性损失
                multi_enhanced = getattr(outputs, 'multi_enhanced', None)
                multi_res_d = getattr(outputs, 'multi_res_d', None)
                multi_res_c = getattr(outputs, 'multi_res_c', None)
                
                loss = criterion(
                    outputs.enhanced, gt,
                    depth_gt=depth_gt,
                    student_feats=outputs.student_feats,
                    attention_maps=outputs.attention_maps,
                    depth_pred=outputs.depth_pred,
                    depth_conf_map=outputs.depth_conf_map,
                    raw=raw_imgs,
                    multi_enhanced=multi_enhanced,              # 🔥 多输入增强结果
                    multi_res_d=multi_res_d,                    # 🔥 多输入去模糊残差
                    multi_res_c=multi_res_c                     # 🔥 多输入颜色校正残差
                )
            
            total_loss += loss.item()
            num_batches += 1
            
            # 累积损失组件
            if hasattr(criterion, 'get_latest_losses'):
                loss_components = criterion.get_latest_losses()
                for k, v in loss_components.items():
                    loss_components_sum[k] += v
            
            # 计算RGB指标
            if outputs.enhanced is not None and gt is not None:
                # 确保数据范围正确
                enhanced_norm = outputs.enhanced
                gt_norm = gt
                
                # 如果数据在[-1,1]范围，转换到[0,1]
                if enhanced_norm.min() < 0:
                    enhanced_norm = (enhanced_norm + 1.0) / 2.0
                if gt_norm.min() < 0:
                    gt_norm = (gt_norm + 1.0) / 2.0
                
                # 计算PSNR
                psnr = compute_psnr(enhanced_norm, gt_norm)
                total_psnr += psnr
                
                # 计算SSIM
                ssim = compute_ssim(enhanced_norm, gt_norm)
                total_ssim += ssim
                
                # 计算LPIPS (如果可用)
                try:
                    import lpips
                    lpips_fn = lpips.LPIPS(net='alex').to(device)
                    lpips_val = lpips_fn(enhanced_norm * 2 - 1, gt_norm * 2 - 1).mean().item()
                    total_lpips += lpips_val
                except:
                    lpips_val = 0.0
                    total_lpips += lpips_val
            
            # 计算深度指标
            if outputs.depth_pred is not None and depth_gt is not None:
                depth_stats = calculate_depth_statistics(outputs.depth_pred, depth_gt)
                total_depth_mae += depth_stats.get('mae', 0.0)
                total_depth_rmse += depth_stats.get('rmse', 0.0)
                total_depth_abs_rel += depth_stats.get('abs_rel', 0.0)
                total_depth_sq_rel += depth_stats.get('sq_rel', 0.0)
            
            # 可视化记录
            if (vis_count < max_val_vis_samples and 
                i % val_vis_interval == 0 and 
                outputs.enhanced is not None):
                
                try:
                    # 记录第一个样本的可视化
                    sample_raw = raw_imgs[0:1]
                    sample_enhanced = outputs.enhanced[0:1]
                    sample_gt = gt[0:1] if gt is not None else None
                    sample_depth_pred = outputs.depth_pred[0:1] if outputs.depth_pred is not None else None
                    sample_depth_gt = depth_gt[0:1] if depth_gt is not None else None
                    
                    # 归一化到[0,1]
                    if sample_raw.min() < 0:
                        sample_raw = (sample_raw + 1.0) / 2.0
                    if sample_enhanced.min() < 0:
                        sample_enhanced = (sample_enhanced + 1.0) / 2.0
                    if sample_gt is not None and sample_gt.min() < 0:
                        sample_gt = (sample_gt + 1.0) / 2.0
                    
                    # 记录RGB图像
                    metric_logger.log_image(f"val/input_{vis_count}", sample_raw, step=epoch)
                    metric_logger.log_image(f"val/enhanced_{vis_count}", sample_enhanced, step=epoch)
                    if sample_gt is not None:
                        metric_logger.log_image(f"val/gt_{vis_count}", sample_gt, step=epoch)
                    
                    # 保存图像到本地文件
                    if save_val_images and val_images_dir is not None and vis_count < max_val_samples:
                        # 保存单独的图像
                        save_image(sample_raw, os.path.join(val_images_dir, f'input_{vis_count:03d}.png'))
                        save_image(sample_enhanced, os.path.join(val_images_dir, f'enhanced_{vis_count:03d}.png'))
                        if sample_gt is not None:
                            save_image(sample_gt, os.path.join(val_images_dir, f'gt_{vis_count:03d}.png'))
                        
                        # 保存对比图
                        if save_comparison:
                            if sample_gt is not None:
                                comparison = torch.cat([sample_raw, sample_enhanced, sample_gt], dim=3)  # 水平拼接
                                save_image(comparison, os.path.join(val_images_dir, f'comparison_{vis_count:03d}.png'))
                            else:
                                comparison = torch.cat([sample_raw, sample_enhanced], dim=3)
                                save_image(comparison, os.path.join(val_images_dir, f'comparison_{vis_count:03d}.png'))
                    
                    # 记录深度图像
                    if sample_depth_pred is not None:
                        # 归一化深度图
                        depth_norm = (sample_depth_pred - sample_depth_pred.min()) / (sample_depth_pred.max() - sample_depth_pred.min() + 1e-8)
                        metric_logger.log_image(f"val/depth_pred_{vis_count}", depth_norm, step=epoch)
                        
                        # 保存深度图到本地
                        if save_val_images and val_images_dir is not None and vis_count < max_val_samples:
                            save_image(depth_norm, os.path.join(val_images_dir, f'depth_pred_{vis_count:03d}.png'))
                    
                    if sample_depth_gt is not None:
                        depth_gt_norm = (sample_depth_gt - sample_depth_gt.min()) / (sample_depth_gt.max() - sample_depth_gt.min() + 1e-8)
                        metric_logger.log_image(f"val/depth_gt_{vis_count}", depth_gt_norm, step=epoch)
                        
                        # 保存GT深度图到本地
                        if save_val_images and val_images_dir is not None and vis_count < max_val_samples:
                            save_image(depth_gt_norm, os.path.join(val_images_dir, f'depth_gt_{vis_count:03d}.png'))
                    
                    vis_count += 1
                    
                except Exception as e:
                    val_logger.warning(f"验证可视化记录失败: {e}")
            
            # 更新进度条
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "PSNR": f"{psnr:.2f}" if 'psnr' in locals() else "N/A"
            })
    
    # 检查是否有验证数据
    if num_batches == 0:
        val_logger.warning(f"⚠️  验证数据加载器为空！跳过验证阶段 Epoch {epoch+1}")
        # 返回默认值，避免除零错误
        return 0.0, 0.0
    
    # 计算平均指标
    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    avg_lpips = total_lpips / num_batches
    
    avg_depth_mae = total_depth_mae / num_batches
    avg_depth_rmse = total_depth_rmse / num_batches
    avg_depth_abs_rel = total_depth_abs_rel / num_batches
    avg_depth_sq_rel = total_depth_sq_rel / num_batches
    
    # 计算平均损失组件
    avg_loss_components = {k: v / num_batches for k, v in loss_components_sum.items()}
    
    # 记录到TensorBoard
    val_metrics = {
        'loss': avg_loss,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'lpips': avg_lpips,
    }
    
    depth_metrics = {
        'depth_mae': avg_depth_mae,
        'depth_rmse': avg_depth_rmse,
        'depth_abs_rel': avg_depth_abs_rel,
        'depth_sq_rel': avg_depth_sq_rel,
    }
    
    # 记录RGB指标
    if epoch % val_metrics_freq == 0:
        metric_logger.log_metrics(val_metrics, prefix="val", step=epoch)
        metric_logger.log_metrics(depth_metrics, prefix="val", step=epoch)
        metric_logger.log_metrics(avg_loss_components, prefix="val_loss", step=epoch)
    
    # 记录到日志文件
    val_logger.info(f"[验证] Epoch {epoch+1} RGB指标: " + 
                   f"Loss={avg_loss:.6f}, PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")
    
    depth_logger.info(f"[验证] Epoch {epoch+1} 深度指标: " + 
                     f"MAE={avg_depth_mae:.4f}, RMSE={avg_depth_rmse:.4f}, " +
                     f"AbsRel={avg_depth_abs_rel:.4f}, SqRel={avg_depth_sq_rel:.4f}")
    
    if multi_logger:
        # 分类记录损失组件
        rgb_loss_metrics = {}
        depth_loss_metrics = {}
        
        for k, v in avg_loss_components.items():
            if any(term in k.lower() for term in ['l1', 'ssim', 'perc', 'fft', 'grad', 'img']):
                rgb_loss_metrics[k] = v
            elif any(term in k.lower() for term in ['depth']):
                depth_loss_metrics[k] = v
        
        if rgb_loss_metrics:
            metrics_logger.info(f"[验证] Epoch {epoch+1} RGB损失: " + 
                              ", ".join([f"{k}={v:.6f}" for k, v in rgb_loss_metrics.items()]))
        
        if depth_loss_metrics:
            depth_logger.info(f"[验证] Epoch {epoch+1} 深度损失: " + 
                            ", ".join([f"{k}={v:.6f}" for k, v in depth_loss_metrics.items()]))
    
    val_logger.info(f"验证完成 Epoch {epoch+1}, 平均损失: {avg_loss:.6f}, 主要指标(PSNR): {avg_psnr:.2f}")
    
    return avg_loss, avg_psnr


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
        checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
        
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
    
    # 🔥 智能恢复逻辑：在设置训练环境之前处理实验目录
    if args.resume:
        print(f"[智能恢复] 检测到 --resume 参数，正在搜索现有实验...")
        exp_dir = setup_experiment_dir(config, resume_mode=True)
    else:
        print(f"[新训练] 创建新的实验目录...")
        exp_dir = setup_experiment_dir(config, resume_mode=False)
    
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
    # 使用我们智能选择的实验目录
    scaler = setup_result['scaler']
    mixed_precision = setup_result['mixed_precision']
    world_size = setup_result['world_size']
    local_rank = setup_result['local_rank']
    update_optimizer_fn = setup_result['update_optimizer_fn'] # 获取动态参数更新函数
    
    # 🔥 现在初始化日志系统，使用正确的exp_dir
    multi_logger, metric_logger, tb_writer = setup_logging_system(exp_dir, config)
    
    # 获取不同分类的日志记录器
    logger = multi_logger.get_logger('train')  # 主训练日志
    
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
    
    # 初始化多文件日志系统，确保文件处理器被正确创建
    if multi_logger:
        multi_logger.log_training_start(config)
    
    for epoch in range(start_epoch, config['train']['epochs']):
        # 🔧 记录epoch开始，确保日志文件正确切换
        if multi_logger:
            multi_logger.log_epoch_start(epoch, config['train']['epochs'])
            
        if args.distributed:
            train_sampler.set_epoch(epoch)
            
        # 训练
        train_loss = train_epoch(
            train_loader, model, criterion, optimizer, device,
            metric_logger, epoch, config, scaler, mixed_precision, multi_logger,
            update_optimizer_fn
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
        
        # 🔧 记录epoch结束和指标汇总
        if multi_logger:
            epoch_metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_psnr': val_metric,
                'lr': optimizer.param_groups[0]['lr']
            }
            multi_logger.log_epoch_end(epoch, epoch_metrics)
        
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


