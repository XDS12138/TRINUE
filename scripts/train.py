#!/usr/bin/env python3
"""
重构后的训练脚本

保留核心的训练和验证逻辑，集成多验证集管理器
"""

import os
import sys
import yaml
import logging
import warnings
import torch
import torch.optim as optim
from collections import defaultdict
from tqdm import tqdm
from torchvision.utils import save_image
from torch.cuda.amp import autocast

# 添加项目根目录到Python路径
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)

# 设置CUDA内存配置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 抑制部分警告和日志
warnings.filterwarnings('ignore')
for logger_name in ['PIL.PngImagePlugin', 'PIL.Image', 'PIL.TiffImagePlugin']:
    pil_logger = logging.getLogger(logger_name)
    pil_logger.setLevel(logging.CRITICAL)
    pil_logger.propagate = False

# 导入我们的工具模块
from utils.arg_parser import parse_args, set_seed
from utils.experiment_manager import setup_experiment_dir
from utils.logging_setup import setup_logging_system
from utils.training_setup import setup_training
from utils.data_loader import prepare_data
from utils.checkpoint_manager import resume_from_checkpoint, save_checkpoint_extended
from utils.visualization_utils import log_training_visualization, save_validation_images
from utils.distributed_utils import setup_for_distributed_launch
from utils.multi_validation_manager import MultiValidationManager
from utils.metrics import calculate_psnr as compute_psnr, calculate_ssim as compute_ssim

# 设置基本日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s')
logger = logging.getLogger(__name__)


def train_epoch(train_loader, model, criterion, optimizer, device, metric_logger, 
               epoch, config, scaler=None, mixed_precision=False, multi_logger=None, 
               update_optimizer_fn=None):
    """
    训练一个epoch的核心函数
    """
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}")
    vis_interval = config.get('visualization', {}).get('interval', 100)
    
    # 获取日志记录器
    if multi_logger:
        train_logger = multi_logger.get_logger('train')
        error_logger = multi_logger.get_logger('error')
    else:
        train_logger = error_logger = logger
    
    train_logger.info(f"======== [Epoch {epoch+1}/{config['train']['epochs']}] 训练开始 ========")
    
    for i, batch in enumerate(progress_bar):
        current_step = epoch * len(train_loader) + i
        
        # 解析批次数据
        raw_imgs, depth_gt, gt = _parse_batch_data(batch, device, config)
        B = raw_imgs.shape[0]
        
        optimizer.zero_grad()
        
        # 动态参数检查
        if current_step == 1 and update_optimizer_fn is not None:
            model_to_check = model.module if hasattr(model, 'module') else model
            update_optimizer_fn(model_to_check, optimizer)
        
        # 前向传播
        try:
            if mixed_precision and scaler is not None:
                with autocast():
                    # 启用多输入一致性学习（如果配置中启用）
                    enable_consistency = config.get('multi_input_consistency', {}).get('enable', False)
                    
                    # 🔥 添加调试信息
                    if step < 3:
                        print(f"[TRAIN DEBUG] Step {step}: hasattr(model, 'forward'): {hasattr(model, 'forward')}")
                        print(f"[TRAIN DEBUG] Step {step}: raw_imgs.dim(): {raw_imgs.dim()}")
                        print(f"[TRAIN DEBUG] Step {step}: enable_consistency: {enable_consistency}")
                        print(f"[TRAIN DEBUG] Step {step}: condition: {hasattr(model, 'forward') and raw_imgs.dim() == 5}")
                    
                    if hasattr(model, 'forward') and raw_imgs.dim() == 5:
                        if step < 3:
                            print(f"[TRAIN DEBUG] Step {step}: 调用 model.forward() 方法")
                        outputs = model.forward(raw_imgs, depth_gt, gt, enable_multi_input_consistency=enable_consistency)
                    else:
                        if step < 3:
                            print(f"[TRAIN DEBUG] Step {step}: 调用 model.multi_forward() 方法")
                        outputs = model.multi_forward(raw_imgs, depth_gt, gt)
                    loss = criterion(
                        outputs.enhanced, gt,
                        depth_gt=depth_gt,
                        student_feats=outputs.student_feats,
                        attention_maps=outputs.attention_maps,
                        depth_pred=outputs.depth_pred,
                        raw=raw_imgs,
                        multi_enhanced=getattr(outputs, 'multi_enhanced', None),
                        multi_depth_pred=getattr(outputs, 'multi_depth_pred', None)
                    )
                
                # 反向传播（混合精度）
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                if config['optimizer'].get('clip_grad_norm', 0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['clip_grad_norm'])
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # 启用多输入一致性学习（如果配置中启用）
                enable_consistency = config.get('multi_input_consistency', {}).get('enable', False)
                if hasattr(model, 'forward') and raw_imgs.dim() == 5:
                    outputs = model.forward(raw_imgs, depth_gt, gt, enable_multi_input_consistency=enable_consistency)
                else:
                    outputs = model.multi_forward(raw_imgs, depth_gt, gt)
                loss = criterion(
                    outputs.enhanced, gt,
                    depth_gt=depth_gt,
                    student_feats=outputs.student_feats,
                    attention_maps=outputs.attention_maps,
                    depth_pred=outputs.depth_pred,
                    raw=raw_imgs,
                    multi_enhanced=getattr(outputs, 'multi_enhanced', None),
                    multi_depth_pred=getattr(outputs, 'multi_depth_pred', None)
                )
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                if config['optimizer'].get('clip_grad_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['clip_grad_norm'])
                
                optimizer.step()
            
            optimizer.zero_grad()
            
        except Exception as e:
            error_logger.error(f"训练步骤 {current_step} 出错: {e}")
            # 🔧 添加详细的错误跟踪
            if "unsupported operand type(s) for +: 'Tensor' and 'str'" in str(e):
                import traceback
                error_logger.error("🎯 捕获到Tensor+str错误的完整堆栈跟踪:")
                error_logger.error(traceback.format_exc())
            continue
        
        current_loss = loss.item()
        epoch_loss += current_loss
        
        progress_bar.set_postfix({"Loss": f"{current_loss:.4f}"})
        
        # 记录指标
        _log_training_metrics(criterion, current_step, config, metric_logger, multi_logger)
        
        # 可视化记录
        if i % vis_interval == 0:
            log_training_visualization(raw_imgs, outputs, depth_gt, gt, 
                                     metric_logger, current_step, config, multi_logger)
        
        # 清理内存
        del loss
        if mixed_precision:
            torch.cuda.empty_cache()
    
    avg_epoch_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    train_logger.info(f"======== [Epoch {epoch+1}] 训练完成，平均损失: {avg_epoch_loss:.6f} ========")
    
    return avg_epoch_loss, avg_epoch_loss


def validate_legacy(val_loader, model, criterion, device, metric_logger, epoch, config, 
                   mixed_precision=False, multi_logger=None):
    """
    传统单验证集验证函数（向后兼容）
    """
    model.eval()
    
    # 获取日志记录器
    if multi_logger:
        val_logger = multi_logger.get_logger('validation')
    else:
        val_logger = logger
    
    val_logger.info(f"开始传统验证 Epoch {epoch+1}...")
    
    # 初始化累积指标
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0
    
    # 设置验证图像保存
    val_images_dir = _setup_val_images_dir(config, epoch)
    vis_count = 0
    max_val_samples = config.get('visualization', {}).get('val_images', {}).get('max_samples', 8)
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"验证 Epoch {epoch+1}")
        
        for i, batch in enumerate(progress_bar):
            # 解析批次数据
            raw_imgs, depth_gt, gt = _parse_batch_data(batch, device, config)
            
            # 前向传播
            if mixed_precision:
                with autocast():
                    outputs = model.multi_forward(raw_imgs, depth_gt, gt)
                    loss = criterion(
                        outputs.enhanced, gt,
                        depth_gt=depth_gt,
                        student_feats=outputs.student_feats,
                        attention_maps=outputs.attention_maps,
                        depth_pred=outputs.depth_pred,
                        raw=raw_imgs
                    )
            else:
                outputs = model.multi_forward(raw_imgs, depth_gt, gt)
                loss = criterion(
                    outputs.enhanced, gt,
                    depth_gt=depth_gt,
                    student_feats=outputs.student_feats,
                    attention_maps=outputs.attention_maps,
                    depth_pred=outputs.depth_pred,
                    raw=raw_imgs
                )
            
            total_loss += loss.item()
            num_batches += 1
            
            # 计算RGB指标
            if outputs.enhanced is not None and gt is not None:
                enhanced_norm, gt_norm = _normalize_images(outputs.enhanced, gt)
                
                psnr = compute_psnr(enhanced_norm, gt_norm)
                ssim = compute_ssim(enhanced_norm, gt_norm)
                
                total_psnr += psnr
                total_ssim += ssim
            
            # 保存验证图像
            if vis_count < max_val_samples and val_images_dir is not None:
                save_validation_images(
                    raw_imgs[0:1], outputs.enhanced[0:1], gt[0:1] if gt is not None else None,
                    outputs.depth_pred[0:1] if outputs.depth_pred is not None else None,
                    depth_gt[0:1] if depth_gt is not None else None,
                    val_images_dir, vis_count, config
                )
                vis_count += 1
            
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "PSNR": f"{psnr:.2f}" if 'psnr' in locals() else "N/A"
            })
    
    # 计算平均指标
    if num_batches == 0:
        val_logger.warning(f"⚠️  验证数据加载器为空！跳过验证阶段 Epoch {epoch+1}")
        return 0.0, 0.0
    
    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    
    # 记录指标
    val_metrics = {
        'loss': avg_loss,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
    }
    
    metric_logger.log_metrics(val_metrics, prefix="val", step=epoch)
    
    val_logger.info(f"[传统验证] Epoch {epoch+1} - Loss: {avg_loss:.6f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
    
    return avg_loss, avg_psnr


def main_worker(config, args):
    """主工作函数"""
    # 1. 设置实验目录
    if args.resume:
        exp_dir = setup_experiment_dir(config, resume_mode=True)
    else:
        exp_dir = setup_experiment_dir(config, resume_mode=False)
    
    # 2. 设置日志系统
    multi_logger, metric_logger, tb_writer = setup_logging_system(exp_dir, config)
    logger = multi_logger.get_logger('train')
    
    # 3. 设置训练环境
    setup_result = setup_training(args, config, args.local_rank)
    if setup_result is None:
        return
    
    model = setup_result['model']
    criterion = setup_result['criterion']
    optimizer = setup_result['optimizer']
    scheduler = setup_result['scheduler']
    device = setup_result['device']
    scaler = setup_result['scaler']
    mixed_precision = setup_result['mixed_precision']
    local_rank = setup_result['local_rank']
    update_optimizer_fn = setup_result['update_optimizer_fn']
    
    # 4. 准备数据
    data_loaders = prepare_data(config, args)
    train_loader = data_loaders['train_loader']
    val_loader = data_loaders['val_loader']
    train_sampler = data_loaders['train_sampler']
    
    # 5. 初始化多验证集管理器
    multi_val_manager = MultiValidationManager(config, device, multi_logger, metric_logger)
    
    # 6. 恢复训练
    start_epoch = 0
    best_metric = 0.0
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    
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
    
    # 7. 仅验证模式
    if args.eval_only:
        logger.info("进入仅验证模式...")
        
        # 运行传统验证
        validate_legacy(val_loader, model, criterion, device, metric_logger, start_epoch, config, mixed_precision, multi_logger)
        
        # 运行多验证集验证
        multi_val_results = multi_val_manager.validate_all_sets(model, criterion, start_epoch, exp_dir)
        
        # 打印结果摘要
        summary = multi_val_manager.get_validation_summary(multi_val_results)
        _print_validation_summary(summary, logger)
        
        logger.info("验证完成。")
        return
    
    # 8. 训练循环
    logger.info("="*20 + " 开始训练 " + "="*20)
    
    for epoch in range(start_epoch, config['train']['epochs']):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # 训练
        train_loss, _ = train_epoch(
            train_loader, model, criterion, optimizer, device,
            metric_logger, epoch, config, scaler, mixed_precision, multi_logger,
            update_optimizer_fn
        )
        
        # 传统验证（向后兼容）
        if val_loader is not None:
            val_loss, val_metric = validate_legacy(
                val_loader, model, criterion, device, metric_logger,
                epoch, config, mixed_precision, multi_logger
            )
        else:
            val_loss, val_metric = 0.0, 0.0
        
        # 🔥 多验证集验证 - 从第20个epoch开始
        multi_val_results = {}
        validation_start_epoch = config.get('train', {}).get('validation_start_epoch', 20)
        
        if epoch + 1 >= validation_start_epoch:  # epoch从0开始，所以+1比较
            logger.info(f"开始多验证集验证 (Epoch {epoch+1} >= {validation_start_epoch})")
            multi_val_results = multi_val_manager.validate_all_sets(model, criterion, epoch, exp_dir)
        else:
            logger.info(f"跳过多验证集验证 (Epoch {epoch+1} < {validation_start_epoch})")
            multi_val_results = {}
        
        # 从多验证集结果中选择最佳指标作为主要指标
        main_metric = _get_main_metric_from_multi_validation(multi_val_results, val_metric)
        
        # 打印验证结果摘要
        if multi_val_results:
            summary = multi_val_manager.get_validation_summary(multi_val_results)
            _print_validation_summary(summary, logger)
        
        # 更新学习率
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(main_metric)
            else:
                scheduler.step()
        
        # 保存检查点
        if local_rank in [-1, 0]:
            is_best = main_metric > best_metric
            if is_best:
                best_metric = main_metric
            
            save_checkpoint_extended(
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
            logger.info(f"Epoch {epoch+1}: 已保存检查点，当前主要指标: {main_metric:.4f}, 最佳指标: {best_metric:.4f}")
    
    logger.info("="*20 + " 训练完成 " + "="*20)
    metric_logger.close()


def _parse_batch_data(batch, device, config):
    """解析批次数据"""
    if isinstance(batch, dict):
        raw_imgs = batch['raw_imgs'].to(device)
        depth_gt = batch['depth'].to(device) if 'depth' in batch and batch['depth'] is not None else None
        gt = batch['gt'].to(device) if 'gt' in batch and batch['gt'] is not None else None
        
        # 🔥 添加调试信息
        print(f"[DEBUG] _parse_batch_data: raw_imgs.shape={raw_imgs.shape}, dim={raw_imgs.dim()}")
        print(f"[DEBUG] _parse_batch_data: depth_gt.shape={depth_gt.shape if depth_gt is not None else None}")
        print(f"[DEBUG] _parse_batch_data: gt.shape={gt.shape if gt is not None else None}")
        
        # 保持5D张量用于多输入处理
        # raw_imgs形状应该是 [B, N, C, H, W] 其中N是退化类型数量
    else:
        raw_imgs, depth_gt_tuple, gt_tuple = batch[:3]
        raw_imgs = raw_imgs.to(device)
        depth_gt = depth_gt_tuple.to(device) if depth_gt_tuple is not None else None
        gt = gt_tuple.to(device) if gt_tuple is not None else None
        
        # 保持5D张量用于多输入处理
    
    return raw_imgs, depth_gt, gt


def _normalize_images(enhanced, gt):
    """归一化图像到[0,1]范围"""
    enhanced_norm = enhanced
    gt_norm = gt
    
    if enhanced_norm.min() < 0:
        enhanced_norm = (enhanced_norm + 1.0) / 2.0
    if gt_norm.min() < 0:
        gt_norm = (gt_norm + 1.0) / 2.0
    
    return enhanced_norm, gt_norm


def _log_training_metrics(criterion, step, config, metric_logger, multi_logger):
    """记录训练指标"""
    # 获取损失组件
    if hasattr(criterion, 'get_latest_losses'):
        loss_components = criterion.get_latest_losses()
        metrics = {}
        
        for loss_name, loss_value in loss_components.items():
            if loss_name != 'total_loss':
                # 🔧 确保所有值都是Python数值，避免Tensor+str错误
                if hasattr(loss_value, 'item'):
                    loss_value = loss_value.item()
                elif isinstance(loss_value, torch.Tensor):
                    loss_value = float(loss_value.cpu().detach())
                
                if not (loss_name.startswith('loss_') or loss_name.endswith('_loss')):
                    metrics[f"loss_{loss_name}"] = loss_value
                else:
                    metrics[loss_name] = loss_value
        
        # 记录频率控制
        train_metrics_freq = config.get('visualization', {}).get('tensorboard', {}).get('train_metrics_freq', 1)
        if step % train_metrics_freq == 0:
            metric_logger.log_metrics(metrics, prefix="train", step=step)


def _setup_val_images_dir(config, epoch):
    """设置验证图像保存目录"""
    val_images_config = config.get('visualization', {}).get('val_images', {})
    if not val_images_config.get('save', False):
        return None
    
    try:
        exp_dir = config.get('experiment', {}).get('output_dir', 'experiments/train')
        exp_name = config.get('experiment', {}).get('name', 'underwater_enhance_run')
        
        import glob
        exp_pattern = os.path.join(exp_dir, f"{exp_name}_*")
        exp_dirs = glob.glob(exp_pattern)
        if exp_dirs:
            current_exp_dir = max(exp_dirs, key=os.path.getctime)
            val_images_dir = os.path.join(current_exp_dir, 'val_images', f'epoch_{epoch+1:03d}')
            os.makedirs(val_images_dir, exist_ok=True)
            return val_images_dir
    except Exception as e:
        logger.warning(f"创建验证图像目录失败: {e}")
    
    return None


def _get_main_metric_from_multi_validation(multi_val_results, fallback_metric):
    """从多验证集结果中选择主要指标"""
    # 优先选择全参考验证集的PSNR作为主要指标
    for set_id, metrics in multi_val_results.items():
        if 'psnr' in metrics:
            return metrics['psnr']
    
    # 其次选择无参考验证集的UCIQE
    for set_id, metrics in multi_val_results.items():
        if 'uciqe' in metrics:
            return metrics['uciqe']
    
    # 最后使用传统验证的指标
    return fallback_metric


def _print_validation_summary(summary, logger):
    """打印验证结果摘要"""
    logger.info("======== 验证结果摘要 ========")
    
    for set_name, result_info in summary.items():
        set_type = result_info['type']
        main_metrics = result_info['main_metrics']
        
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in main_metrics.items()])
        logger.info(f"📊 {set_name} ({set_type}): {metrics_str}")
    
    logger.info("=" * 30)


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 分布式训练处理
    if setup_for_distributed_launch(config, args):
        return  # 分布式训练已启动，主进程退出
    
    # 单GPU或CPU训练
    args.local_rank = -1
    main_worker(config, args)


if __name__ == '__main__':
    main() 