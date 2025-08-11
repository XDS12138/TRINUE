#!/usr/bin/env python3
"""
可视化工具模块

负责训练过程中的图像和特征可视化
"""

import os
import torch
import logging
from torchvision.utils import save_image

logger = logging.getLogger(__name__)


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
    
    return data


def log_training_visualization(raw_imgs, outputs, depth_gt, gt, metric_logger, 
                             step, config, multi_logger=None):
    """
    记录训练过程中的可视化数据
    
    Args:
        raw_imgs: 输入图像
        outputs: 模型输出（支持dict或对象）
        depth_gt: 深度真值
        gt: 图像真值
        metric_logger: 指标记录器
        step: 当前步数
        config: 配置
        multi_logger: 多文件日志记录器
    """
    if multi_logger:
        vis_logger = multi_logger.get_logger('visualization')
        error_logger = multi_logger.get_logger('error')
    else:
        vis_logger = error_logger = logger
    
    try:
        vis_logger.info(f"======== 开始记录可视化数据，步骤: {step} ========")
        
        # 取第一个样本进行可视化
        vis_raw_first_item = raw_imgs[0].unsqueeze(0)
        vis_depth_first_item = depth_gt[0].unsqueeze(0) if depth_gt is not None else None
        vis_gt_first_item = gt[0].unsqueeze(0) if gt is not None else None

        # 解析输出（兼容 dict / 对象）
        if isinstance(outputs, dict):
            out_enh = outputs.get('enhanced')
            out_pred_gate = outputs.get('pred_gate')
            out_depth = outputs.get('depth_pred')
            out_feats = outputs.get('student_feats')
            out_attn = outputs.get('attention_maps')
        else:
            out_enh = getattr(outputs, 'enhanced', None)
            out_pred_gate = getattr(outputs, 'pred_gate', None)
            out_depth = getattr(outputs, 'depth_pred', None)
            out_feats = getattr(outputs, 'student_feats', None)
            out_attn = getattr(outputs, 'attention_maps', None)

        # 记录深度相关可视化
        if vis_depth_first_item is not None:
            _log_depth_visualization(vis_depth_first_item, metric_logger, step, config, vis_logger)

        with torch.no_grad():
            # 模型输出可视化
            _log_model_outputs(
                vis_raw_first_item,
                vis_depth_first_item,
                vis_gt_first_item,
                out_enh,
                out_pred_gate,
                out_depth,
                out_attn,
                metric_logger,
                step,
                vis_logger
            )
            
            # 记录RGB图像对比
            _log_rgb_comparison(
                vis_raw_first_item,
                out_enh[0:1] if out_enh is not None else None,
                vis_gt_first_item,
                metric_logger,
                step,
                vis_logger
            )
            
            # 记录特征图
            if out_feats:
                _log_feature_maps(out_feats, metric_logger, step, vis_logger)

        vis_logger.info(f"======== 可视化数据记录完成，步骤: {step} ========")
        
    except Exception as e:
        error_logger.error(f"记录可视化数据时发生错误: {str(e)}，步骤: {step}")
        import traceback
        error_logger.error(traceback.format_exc())


def _log_depth_visualization(depth_gt, metric_logger, step, config, vis_logger):
    """记录深度相关可视化"""
    vis_logger.info(f"Step {step}: 正在处理深度图可视化...")
    
    # 对深度图执行对数变换和归一化处理
    log_depth = torch.log(depth_gt + 1.0)
    
    # 从配置中获取深度范围
    depth_config = config['loss'].get('depth_processing', {})
    min_depth = depth_config.get('min_depth_log', 5000.0)
    max_depth = depth_config.get('max_depth_log', 65000.0)
    log_min = torch.log(torch.tensor(min_depth, device=depth_gt.device) + 1.0)
    log_max = torch.log(torch.tensor(max_depth, device=depth_gt.device) + 1.0)
    norm_depth = (log_depth - log_min) / (log_max - log_min + 1e-6)
    norm_depth = torch.clamp(norm_depth, 0, 1)
    
    # 记录各种深度相关可视化
    metric_logger.log_image("train/depth_gt_normalized", norm_depth, step=step)
    vis_logger.info(f"  - 已记录 'train/depth_gt_normalized'")


def _log_model_outputs(vis_raw, vis_depth, vis_gt, enhanced, pred_gate, depth_pred, attention_maps, metric_logger, step, vis_logger):
    """记录模型输出可视化（兼容dict提取后的张量）"""
    # 可视化注意力图（如果有）
    if attention_maps is not None:
        _log_attention_maps(attention_maps, metric_logger, step, vis_logger)
    
    # 连续深度预测可视化
    if depth_pred is not None:
        vis_depth_pred = depth_pred[0:1]
        depth_pred_norm = vis_depth_pred.clone().detach()
        if depth_pred_norm.min() != depth_pred_norm.max():
            depth_pred_norm = (depth_pred_norm - depth_pred_norm.min()) / (depth_pred_norm.max() - depth_pred_norm.min())
        metric_logger.log_image("train/depth_pred_continuous", depth_pred_norm, step=step)
        vis_logger.info(f"  - 已记录 'depth_pred_continuous'")
    
    # 深度门控可视化
    if pred_gate is not None:
        _log_depth_gate_visualization(pred_gate[0:1], metric_logger, step, vis_logger)


def _log_attention_maps(attention_maps, metric_logger, step, vis_logger):
    """记录注意力图可视化"""
    vis_logger.info(f"Step {step}: 正在处理注意力图可视化...")
    depth2rgb_attn, rgb2depth_attn = attention_maps
    
    if depth2rgb_attn is not None:
        # 为可视化选择第一个头的注意力图
        depth2rgb_viz = depth2rgb_attn[0, 0].unsqueeze(0).unsqueeze(0)
        depth2rgb_viz = (depth2rgb_viz - depth2rgb_viz.min()) / (depth2rgb_viz.max() - depth2rgb_viz.min() + 1e-8)
        metric_logger.log_image("train/depth2rgb_attention", depth2rgb_viz, step=step)
        vis_logger.info(f"  - 已记录 'train/depth2rgb_attention'")
    
    if rgb2depth_attn is not None:
        rgb2depth_viz = rgb2depth_attn[0, 0].unsqueeze(0).unsqueeze(0)
        rgb2depth_viz = (rgb2depth_viz - rgb2depth_viz.min()) / (rgb2depth_viz.max() - rgb2depth_viz.min() + 1e-8)
        metric_logger.log_image("train/rgb2depth_attention", rgb2depth_viz, step=step)
        vis_logger.info(f"  - 已记录 'train/rgb2depth_attention'")


def _log_depth_gate_visualization(pred_gate, metric_logger, step, vis_logger):
    """记录深度门控可视化"""
    pred_gate_min = pred_gate.min().item()
    pred_gate_max = pred_gate.max().item()
    pred_gate_mean = pred_gate.mean().item()
    pred_gate_std = pred_gate.std().item()
    
    vis_logger.info(f"深度门控统计: Step {step} | 范围: [{pred_gate_min:.6f}, {pred_gate_max:.6f}] | "
                   f"均值: {pred_gate_mean:.6f} | 标准差: {pred_gate_std:.6f}")
    
    # 改进深度门控可视化 - 使用更强的对比度增强
    vis_pred_gate_enhanced = pred_gate.clone()
    
    # 如果值域过小，增强对比度
    if pred_gate_max - pred_gate_min < 0.1:
        if pred_gate_std > 0:
            # 使用Z-score标准化增强对比度
            vis_pred_gate_enhanced = (pred_gate - pred_gate_mean) / (pred_gate_std + 1e-8)
            vis_pred_gate_enhanced = torch.clamp(vis_pred_gate_enhanced, -3, 3)
            vis_pred_gate_enhanced = (vis_pred_gate_enhanced + 3) / 6
        else:
            # 为避免全黑图像，手动设置一个渐变
            h, w = pred_gate.shape[-2:]
            vis_pred_gate_enhanced = torch.linspace(0, 1, w).view(1, 1, 1, w).repeat(1, 1, h, 1)
    
    # 记录原始和增强后的深度门控
    metric_logger.log_image("train/depth_gate_original", pred_gate, step=step)
    metric_logger.log_image("train/depth_gate_enhanced", vis_pred_gate_enhanced, step=step)
    vis_logger.info(f"  - 已记录 'depth_gate' (原始图与增强图)")


def _log_rgb_comparison(vis_input, vis_output, vis_gt, metric_logger, step, vis_logger):
    """记录RGB图像对比"""
    vis_logger.info(f"Step {step}: 正在处理RGB图像可视化...")
    
    # 输入图像归一化
    if vis_input.min() < 0:
        vis_input_normalized = (vis_input + 1.0) / 2.0
    else:
        vis_input_normalized = vis_input
    metric_logger.log_image("train/input", vis_input_normalized, step=step)
    
    # 增强图归一化处理
    if vis_output is not None:
        if vis_output.min() < 0:
            vis_outputs_normalized = (vis_output + 1.0) / 2.0
        else:
            vis_outputs_normalized = vis_output
        metric_logger.log_image("train/enhanced", vis_outputs_normalized, step=step)
        
        # GT图像
        if vis_gt is not None:
            if vis_gt.min() < 0:
                vis_gt_normalized = (vis_gt + 1.0) / 2.0
            else:
                vis_gt_normalized = vis_gt
            metric_logger.log_image("train/gt", vis_gt_normalized, step=step)
            
            # 创建对比图 - 处理维度不匹配问题
            # 确保所有张量都是4维的用于比较
            if vis_input_normalized.dim() == 5:
                # 多退化输入：取第一个退化级别进行比较 [B, N, C, H, W] -> [B, C, H, W]
                vis_input_for_comparison = vis_input_normalized[:, 0, :, :, :]
            else:
                vis_input_for_comparison = vis_input_normalized
                
            comparison_rgb = torch.cat([vis_input_for_comparison, vis_outputs_normalized, vis_gt_normalized], dim=-1)
            metric_logger.log_image("train/comparison_rgb", comparison_rgb, step=step)
            
            # 误差图
            error_map = torch.abs(vis_outputs_normalized - vis_gt_normalized)
            metric_logger.log_image("train/error_map", error_map, step=step)


def _log_feature_maps(student_feats, metric_logger, step, vis_logger):
    """记录特征图可视化"""
    vis_logger.info(f"Step {step}: 正在处理特征图可视化...")
    
    for j, feat in enumerate(student_feats):
        if feat is not None and not isinstance(feat, list):
            # 取特征图的前几个通道进行可视化
            feat_vis = feat[0:1, 0:min(3, feat.shape[1])].mean(1, keepdim=True)
            feat_vis = (feat_vis - feat_vis.min()) / (feat_vis.max() - feat_vis.min() + 1e-6)
            metric_logger.log_image(f"train/student_feat_level{j}", feat_vis, step=step)
            vis_logger.info(f"  - 已记录 'train/student_feat_level{j}'")


def save_validation_images(raw_imgs, enhanced, gt, depth_pred, depth_gt, 
                         val_images_dir, sample_idx, config):
    """
    保存验证图像到本地文件
    
    Args:
        raw_imgs: 输入图像
        enhanced: 增强图像
        gt: 真值图像
        depth_pred: 预测深度
        depth_gt: 真值深度
        val_images_dir: 保存目录
        sample_idx: 样本索引
        config: 配置
    """
    if val_images_dir is None:
        return
    
    try:
        # 归一化到[0,1]
        if raw_imgs.min() < 0:
            raw_imgs = (raw_imgs + 1.0) / 2.0
        if enhanced.min() < 0:
            enhanced = (enhanced + 1.0) / 2.0
        if gt is not None and gt.min() < 0:
            gt = (gt + 1.0) / 2.0
        
        # 保存RGB图像
        save_image(raw_imgs, os.path.join(val_images_dir, f'input_{sample_idx:03d}.png'))
        save_image(enhanced, os.path.join(val_images_dir, f'enhanced_{sample_idx:03d}.png'))
        if gt is not None:
            save_image(gt, os.path.join(val_images_dir, f'gt_{sample_idx:03d}.png'))
        
        # 保存深度图
        if depth_pred is not None:
            depth_norm = (depth_pred - depth_pred.min()) / (depth_pred.max() - depth_pred.min() + 1e-8)
            save_image(depth_norm, os.path.join(val_images_dir, f'depth_pred_{sample_idx:03d}.png'))
        
        if depth_gt is not None:
            depth_gt_norm = (depth_gt - depth_gt.min()) / (depth_gt.max() - depth_gt.min() + 1e-8)
            save_image(depth_gt_norm, os.path.join(val_images_dir, f'depth_gt_{sample_idx:03d}.png'))
            
        # 保存对比图
        val_images_config = config.get('visualization', {}).get('val_images', {})
        if val_images_config.get('save_comparison', True):
            if gt is not None:
                comparison = torch.cat([raw_imgs, enhanced, gt], dim=3)
                save_image(comparison, os.path.join(val_images_dir, f'comparison_{sample_idx:03d}.png'))
            else:
                comparison = torch.cat([raw_imgs, enhanced], dim=3)
                save_image(comparison, os.path.join(val_images_dir, f'comparison_{sample_idx:03d}.png'))
        
    except Exception as e:
        logger.error(f"保存验证图像失败 (sample {sample_idx}): {e}") 