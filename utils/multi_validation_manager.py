#!/usr/bin/env python3
"""
多验证集管理器

负责管理和执行多个不同类型的验证集验证，包括：
- 全参考图像增强验证集
- 无参考图像增强验证集  
- 深度预测验证集
"""

import os
import csv
import logging
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image, ImageFont, ImageDraw

from modules.datasets import MultiDegradationDataset
from torch.utils.data import DataLoader, Dataset
from utils.metrics import (
    evaluate_with_reference, evaluate_no_reference, evaluate_depth_estimation,
    calculate_psnr, calculate_ssim, calculate_ciede2000, calculate_lpips,
    calculate_uciqe, calculate_uiqm, calculate_niqe
)

logger = logging.getLogger(__name__)


class DepthValidationDataset(Dataset):
    """
    专门用于深度验证的数据集类
    处理RGB图像和numpy格式的深度文件
    """
    
    def __init__(self, rgb_folder, depth_folder, depth_format='npy'):
        """
        初始化深度验证数据集
        
        Args:
            rgb_folder: RGB图像文件夹路径
            depth_folder: 深度文件文件夹路径  
            depth_format: 深度文件格式 ('npy' 或 'png')
        """
        self.rgb_folder = rgb_folder
        self.depth_folder = depth_folder
        self.depth_format = depth_format
        
        # 获取RGB文件列表
        self.rgb_files = []
        for ext in ['.png', '.jpg', '.jpeg']:
            self.rgb_files.extend([f for f in os.listdir(rgb_folder) if f.lower().endswith(ext)])
        
        self.rgb_files.sort()
        
        # 验证对应的深度文件是否存在
        self.valid_pairs = []
        for rgb_file in self.rgb_files:
            base_name = os.path.splitext(rgb_file)[0]
            
            if depth_format == 'npy':
                depth_file = f"{base_name}.npy"
            else:
                depth_file = f"{base_name}.png"
            
            depth_path = os.path.join(depth_folder, depth_file)
            if os.path.exists(depth_path):
                self.valid_pairs.append((rgb_file, depth_file))
        
        logger.info(f"深度验证数据集: 找到 {len(self.valid_pairs)} 对有效的RGB-深度文件")
        
        if len(self.valid_pairs) == 0:
            raise ValueError(f"未找到有效的RGB-深度文件对，RGB文件夹: {rgb_folder}, 深度文件夹: {depth_folder}")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        rgb_file, depth_file = self.valid_pairs[idx]
        
        # 加载RGB图像
        rgb_path = os.path.join(self.rgb_folder, rgb_file)
        rgb_image = Image.open(rgb_path).convert('RGB')
        rgb_tensor = torch.from_numpy(np.array(rgb_image)).permute(2, 0, 1).float() / 255.0
        
        # 加载深度数据
        depth_path = os.path.join(self.depth_folder, depth_file)
        if self.depth_format == 'npy':
            depth_data = np.load(depth_path)
            depth_tensor = torch.from_numpy(depth_data).float()
        else:
            depth_image = Image.open(depth_path)
            depth_tensor = torch.from_numpy(np.array(depth_image)).float()
        
        # 确保深度张量有正确的维度
        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0)  # 添加通道维度
        
        # 创建批次维度并返回字典格式（兼容MultiDegradationDataset格式）
        return {
            'raw_imgs': rgb_tensor.unsqueeze(0),  # [1, C, H, W]
            'depth': depth_tensor.unsqueeze(0),   # [1, C, H, W]  
            'gt': rgb_tensor.unsqueeze(0)         # 使用RGB作为GT（用于深度预测任务）
        }


class MultiValidationManager:
    """多验证集管理器"""
    
    def __init__(self, config, device, multi_logger=None, metric_logger=None):
        """
        初始化多验证集管理器
        
        Args:
            config: 配置字典
            device: 计算设备
            multi_logger: 多文件日志记录器
            metric_logger: 指标记录器
        """
        self.config = config
        self.device = device
        self.multi_logger = multi_logger
        self.metric_logger = metric_logger
        
        # 获取验证集配置
        self.validation_config = config.get('validation_sets', {})
        self.vis_config = config.get('visualization', {}).get('multi_validation', {})
        
        # 日志记录器
        if multi_logger:
            self.val_logger = multi_logger.get_logger('validation')
            self.metrics_logger = multi_logger.get_logger('metrics')
        else:
            self.val_logger = self.metrics_logger = logger
        
        # 初始化验证集（在日志记录器设置之后）
        self.validation_sets = self._initialize_validation_sets()
        
        self.val_logger.info(f"初始化多验证集管理器，共 {len(self.validation_sets)} 个验证集")
    
    def _initialize_validation_sets(self):
        """初始化所有验证集"""
        validation_sets = {}
        
        for set_id, set_config in self.validation_config.items():
            if not set_config.get('enabled', True):
                continue
                
            try:
                val_set = self._create_validation_set(set_id, set_config)
                if val_set is not None:
                    validation_sets[set_id] = {
                        'config': set_config,
                        'dataset': val_set['dataset'],
                        'dataloader': val_set['dataloader'],
                        'type': set_config['type'],
                        'metrics': set_config.get('metrics', []),
                        'name': set_config.get('name', set_id)
                    }
                    self.val_logger.info(f"成功加载验证集: {set_config.get('name', set_id)} "
                                       f"({set_config['type']}, {len(val_set['dataset'])} 样本)")
            except Exception as e:
                self.val_logger.error(f"加载验证集 {set_id} 失败: {e}")
        
        return validation_sets
    
    def _create_validation_set(self, set_id, set_config):
        """创建单个验证集"""
        data_root = set_config['data_root']
        if not os.path.exists(data_root):
            self.val_logger.warning(f"验证集路径不存在: {data_root}")
            return None
        
        set_type = set_config['type']
        folder_structure = set_config.get('folder_structure', {})
        
        try:
            if set_type in ['enhancement_with_reference', 'enhancement_no_reference']:
                # 🔧 图像增强验证集 - 支持多输入和单输入
                if set_config.get('multi_input', False):
                    # 🔥 多输入验证集处理
                    input_folders = set_config.get('input_folders', [])
                    if not input_folders:
                        self.val_logger.error(f"多输入验证集未指定input_folders: {set_id}")
                        return None
                    
                    # 构建所有输入文件夹路径
                    raw_folders = []
                    for folder_name in input_folders:
                        folder_path = os.path.join(data_root, folder_name)
                        if os.path.exists(folder_path):
                            raw_folders.append(folder_path)
                        else:
                            self.val_logger.warning(f"输入文件夹不存在，跳过: {folder_path}")
                    
                    if not raw_folders:
                        self.val_logger.error(f"没有找到有效的输入文件夹: {set_id}")
                        return None
                    
                    self.val_logger.info(f"多输入验证集 {set_config['name']}: 找到 {len(raw_folders)} 个输入文件夹")
                    
                else:
                    # 🔧 单输入验证集处理（原逻辑）
                        input_folder_name = folder_structure.get('input', 'input')
                        
                        # 支持将图像直接放在data_root中
                        if input_folder_name == ".":
                            raw_folders = [data_root]
                        else:
                            raw_folders = [os.path.join(data_root, input_folder_name)]
    
                        # 验证输入文件夹是否存在
                        if not os.path.exists(raw_folders[0]):
                            self.val_logger.error(f"输入文件夹不存在: {raw_folders[0]}")
                            return None
                
                # GT文件夹（仅全参考验证集需要）
                gt_folder_name = folder_structure.get('gt', 'gt')
                gt_folder = None
                if set_type == 'enhancement_with_reference':
                    gt_folder = os.path.join(data_root, gt_folder_name)
                    if not os.path.exists(gt_folder):
                        self.val_logger.error(f"GT文件夹不存在: {gt_folder}")
                        return None
                
                dataset = MultiDegradationDataset(
                    raw_folders=raw_folders,
                    gt_folder=gt_folder,
                    depth_folder=None,  # 图像增强验证集不需要深度
                    patch_size=None,  # 验证时不裁剪
                    augment=False
                )
                
            elif set_type == 'depth_prediction':
                # 🔧 深度预测验证集 - 支持多RGB输入和单RGB输入
                depth_folder_name = folder_structure.get('depth', 'depth')
                depth_folder = os.path.join(data_root, depth_folder_name)
                
                # 验证深度文件夹是否存在
                if not os.path.exists(depth_folder):
                    self.val_logger.error(f"深度文件夹不存在: {depth_folder}")
                    return None
                
                if set_config.get('multi_input', False):
                    # 🔥 多RGB输入深度验证集处理
                    rgb_folders = set_config.get('rgb_folders', [])
                    if not rgb_folders:
                        self.val_logger.error(f"多输入深度验证集未指定rgb_folders: {set_id}")
                        return None
                    
                    # 找到第一个存在的RGB文件夹作为代表
                    rgb_folder = None
                    for folder_name in rgb_folders:
                        folder_path = os.path.join(data_root, folder_name)
                        if os.path.exists(folder_path):
                            rgb_folder = folder_path
                            break
                    
                    if rgb_folder is None:
                        self.val_logger.error(f"没有找到有效的RGB文件夹: {set_id}")
                        return None
                    
                    self.val_logger.info(f"多RGB输入深度验证集 {set_config['name']}: 使用 {os.path.basename(rgb_folder)} 作为代表输入")
                    
                else:
                    # 🔧 单RGB输入深度验证集处理（原逻辑）
                    rgb_folder_name = folder_structure.get('rgb', 'rgb')
                    rgb_folder = os.path.join(data_root, rgb_folder_name)
                    
                    # 验证RGB文件夹是否存在
                    if not os.path.exists(rgb_folder):
                        self.val_logger.error(f"RGB文件夹不存在: {rgb_folder}")
                        return None
                
                # 🔧 创建特殊的深度数据集
                dataset = self._create_depth_dataset(rgb_folder, depth_folder, set_config)
                
            else:
                self.val_logger.error(f"不支持的验证集类型: {set_type}")
                return None
            
            if dataset is None:
                return None
            
            # 创建数据加载器
            dataloader = DataLoader(
                dataset,
                batch_size=self.config['data']['batch_size'],
                shuffle=False,
                num_workers=min(8, self.config['data']['num_workers']),  # 验证时使用较少的worker
                pin_memory=True
            )
            
            return {'dataset': dataset, 'dataloader': dataloader}
            
        except Exception as e:
            self.val_logger.error(f"创建验证集 {set_id} 失败: {e}")
            import traceback
            self.val_logger.error(traceback.format_exc())
            return None
    
    def validate_all_sets(self, model, criterion, epoch, exp_dir):
        """
        对所有验证集进行验证
        
        Args:
            model: 模型
            criterion: 损失函数
            epoch: 当前epoch
            exp_dir: 实验目录
            
        Returns:
            dict: 包含所有验证集结果的字典
        """
        if not self.vis_config.get('enabled', True):
            self.val_logger.info("多验证集验证已禁用，跳过")
            return {}
        
        model.eval()
        all_results = {}
        
        self.val_logger.info(f"======== Epoch {epoch+1} 多验证集验证开始 ========")
        
        for set_id, val_set in self.validation_sets.items():
            self.val_logger.info(f"开始验证集: {val_set['name']} ({val_set['type']})")
            
            try:
                results = self._validate_single_set(
                    model, criterion, val_set, epoch, exp_dir, set_id
                )
                all_results[set_id] = results
                
                # 记录主要指标
                main_metrics = self._get_main_metrics(results, val_set['type'])
                metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in main_metrics.items()])
                self.val_logger.info(f"验证集 {val_set['name']} 完成: {metrics_str}")
                
            except Exception as e:
                self.val_logger.error(f"验证集 {val_set['name']} 验证失败: {e}")
                import traceback
                self.val_logger.error(traceback.format_exc())
        
        # 保存综合结果
        self._save_comprehensive_results(all_results, epoch, exp_dir)
        
        self.val_logger.info(f"======== Epoch {epoch+1} 多验证集验证完成 ========")
        
        return all_results
    
    def _validate_single_set(self, model, criterion, val_set, epoch, exp_dir, set_id):
        """验证单个验证集"""
        val_config = val_set['config']
        val_type = val_set['type']
        dataloader = val_set['dataloader']
        
        # 创建保存目录
        save_dir = os.path.join(exp_dir, 'validation_results', f'epoch_{epoch+1:03d}', val_set['name'])
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化结果累积
        accumulated_metrics = defaultdict(list)
        saved_images = 0
        max_save_images = val_config.get('save_images', 5)
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc=f"验证 {val_set['name']}")
            
            for i, batch in enumerate(progress_bar):
                # 解析批次数据
                raw_imgs, depth_gt, gt = self._parse_batch_data(batch, val_type)
                
                # 前向传播（统一入口，兼容DDP返回dict）
                outputs = model(raw_imgs, depth_gt, gt, enable_multi_input_consistency=False)
                if isinstance(outputs, dict):
                    out_enhanced = outputs.get('enhanced')
                    out_depth = outputs.get('depth_pred')
                else:
                    out_enhanced = outputs.enhanced
                    out_depth = outputs.depth_pred
                
                # 计算指标（函数内部也将兼容dict/对象）
                batch_metrics = self._compute_batch_metrics(
                    outputs, gt, depth_gt, val_set['metrics'], val_type, val_set
                )
                
                # 累积指标
                for metric_name, metric_value in batch_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        accumulated_metrics[metric_name].append(metric_value)
                
                # 保存图像
                if saved_images < max_save_images:
                    self._save_comparison_images(
                        raw_imgs, out_enhanced, gt, depth_gt, out_depth,
                        save_dir, saved_images, val_type, batch_metrics
                    )
                    saved_images += 1
                
                # 更新进度条
                if batch_metrics:
                    main_metric = list(batch_metrics.keys())[0]
                    progress_bar.set_postfix({main_metric: f"{batch_metrics[main_metric]:.4f}"})
        
        # 计算平均指标
        final_metrics = {}
        for metric_name, values in accumulated_metrics.items():
            if values:
                final_metrics[metric_name] = np.mean(values)
        
        # 保存验证集结果
        self._save_validation_set_results(final_metrics, val_set, epoch, save_dir)
        
        # 记录到TensorBoard
        if self.vis_config.get('tensorboard_logging', True) and self.metric_logger:
            for metric_name, metric_value in final_metrics.items():
                self.metric_logger.log_metrics(
                    {metric_name: metric_value}, 
                    prefix=f"val_{val_set['name']}", 
                    step=epoch
                )
        
        return final_metrics
    
    def _parse_batch_data(self, batch, val_type):
        """解析批次数据"""
        if isinstance(batch, dict):
            raw_imgs = batch['raw_imgs'].to(self.device)
            depth_gt = batch.get('depth')
            gt = batch.get('gt')
            
            # 处理5D张量（取第一个退化级别）
            if raw_imgs.dim() == 5:
                raw_imgs = raw_imgs[:, 0]
                
        else:
            raw_imgs, depth_gt, gt = batch[:3]
            raw_imgs = raw_imgs.to(self.device)
            
            if raw_imgs.dim() == 5:
                raw_imgs = raw_imgs[:, 0]
        
        # 根据验证集类型处理GT数据
        if val_type == 'enhancement_no_reference':
            gt = None  # 无参考验证集不需要GT
        elif gt is not None:
            gt = gt.to(self.device)
            
        if depth_gt is not None:
            depth_gt = depth_gt.to(self.device)
            
        return raw_imgs, depth_gt, gt
    
    def _compute_batch_metrics(self, outputs, gt, depth_gt, metrics_list, val_type, val_set):
        """计算批次指标"""
        batch_metrics = {}
        
        try:
            if val_type in ['enhancement_with_reference', 'enhancement_no_reference']:
                # 图像增强指标
                enhanced = outputs['enhanced'] if isinstance(outputs, dict) else outputs.enhanced
                if enhanced is None:
                    return batch_metrics
                
                # 归一化到[0,1]范围
                enhanced_norm = self._normalize_images(enhanced)
                
                # 转换一次为numpy，避免重复
                enhanced_np = self._tensor_to_numpy_image(enhanced_norm)
                gt_np = None
                if gt is not None:
                    gt_norm = self._normalize_images(gt)
                    gt_np = self._tensor_to_numpy_image(gt_norm)
                    
                if val_type == 'enhancement_with_reference' and gt_np is not None:
                    # 同时支持全参考与无参考指标（按metrics_list请求）
                    for metric_name in metrics_list:
                        try:
                            if metric_name == 'psnr':
                                batch_metrics['psnr'] = calculate_psnr(enhanced_np, gt_np)
                            elif metric_name == 'ssim':
                                ssim_val = calculate_ssim(enhanced_np, gt_np)
                                if ssim_val is not None and ssim_val > 0:
                                    batch_metrics['ssim'] = ssim_val
                                else:
                                    self.val_logger.warning(f"SSIM计算返回无效值: {ssim_val}")
                            elif metric_name == 'ciede2000':
                                ciede_val = calculate_ciede2000(enhanced_np, gt_np)
                                if ciede_val is not None and ciede_val > 0:
                                    batch_metrics['ciede2000'] = ciede_val
                                else:
                                    self.val_logger.warning(f"CIEDE2000计算返回无效值: {ciede_val}")
                            elif metric_name == 'lpips':
                                lpips_val = calculate_lpips(enhanced_np, gt_np)
                                if lpips_val is not None:
                                    batch_metrics['lpips'] = lpips_val
                                else:
                                    self.val_logger.warning(f"LPIPS计算返回无效值: {lpips_val}")
                            elif metric_name == 'uciqe':
                                batch_metrics['uciqe'] = calculate_uciqe(enhanced_np)
                            elif metric_name == 'uiqm':
                                batch_metrics['uiqm'] = calculate_uiqm(enhanced_np)
                            elif metric_name == 'niqe':
                                niqe_val = calculate_niqe(enhanced_np)
                                if niqe_val is not None and niqe_val > 0:
                                    batch_metrics['niqe'] = niqe_val
                                else:
                                    self.val_logger.warning(f"NIQE计算返回无效值: {niqe_val}")
                        except Exception as e:
                            self.val_logger.warning(f"{metric_name}计算失败: {e}")
                
                elif val_type == 'enhancement_no_reference':
                    # 无参考指标
                    for metric_name in metrics_list:
                        if metric_name == 'uciqe':
                            batch_metrics['uciqe'] = calculate_uciqe(enhanced_np)
                        elif metric_name == 'uiqm':
                            batch_metrics['uiqm'] = calculate_uiqm(enhanced_np)
                        elif metric_name == 'niqe':
                            try:
                                batch_metrics['niqe'] = calculate_niqe(enhanced_np)
                            except Exception as e:
                                self.val_logger.warning(f"NIQE计算失败: {e}")
                        # 其他未实现的指标跳过
            
            elif val_type == 'depth_prediction':
                # 深度预测指标
                out_depth = outputs['depth_pred'] if isinstance(outputs, dict) else outputs.depth_pred
                if out_depth is not None and depth_gt is not None:
                    # 从配置中获取深度评估参数
                    val_config = val_set['config']
                    depth_params = {
                        'gt_units': val_config.get('depth_gt_units', 'm'),
                        'valid_min': val_config.get('depth_valid_min', 0.1),
                        'valid_max': val_config.get('depth_valid_max', 100.0),
                        'pred_clamp_val': val_config.get('pred_clamp_val', 80.0)
                    }
                    
                    depth_metrics = evaluate_depth_estimation(
                        out_depth, depth_gt, **depth_params
                    )
                    
                    for metric_name in metrics_list:
                        if metric_name in depth_metrics:
                            batch_metrics[metric_name] = depth_metrics[metric_name]
                            
        except Exception as e:
            self.val_logger.warning(f"计算指标时出错: {e}")
        
        return batch_metrics
    
    def _normalize_images(self, images):
        """归一化图像到[0,1]范围"""
        if images.min() < 0:
            return (images + 1.0) / 2.0
        return images
    
    def _save_comparison_images(self, raw_imgs, enhanced, gt, depth_gt, depth_pred, 
                               save_dir, image_idx, val_type, metrics):
        """保存对比图像"""
        try:
            # 取第一个样本
            raw_img = raw_imgs[0:1]
            enhanced_img = enhanced[0:1] if enhanced is not None else None
            
            if enhanced_img is None:
                return
            
            # 归一化
            raw_norm = self._normalize_images(raw_img)
            enhanced_norm = self._normalize_images(enhanced_img)
            
            # 创建对比图
            comparison_images = [raw_norm, enhanced_norm]
            titles = ['Input', 'Enhanced']
            
            # 添加GT（如果有）
            if gt is not None:
                gt_norm = self._normalize_images(gt[0:1])
                comparison_images.append(gt_norm)
                titles.append('Ground Truth')
            
            # 水平拼接
            if self.vis_config.get('comparison_layout', 'horizontal') == 'horizontal':
                comparison = torch.cat(comparison_images, dim=3)  # 水平拼接
            else:
                comparison = torch.cat(comparison_images, dim=2)  # 垂直拼接
            
            # 保存对比图
            save_path = os.path.join(save_dir, f'comparison_{image_idx:03d}.png')
            save_image(comparison, save_path)
            
            # 保存单独的图像
            save_image(raw_norm, os.path.join(save_dir, f'input_{image_idx:03d}.png'))
            save_image(enhanced_norm, os.path.join(save_dir, f'enhanced_{image_idx:03d}.png'))
            if gt is not None:
                save_image(gt_norm, os.path.join(save_dir, f'gt_{image_idx:03d}.png'))
            
            # 保存深度图（如果有）
            if val_type == 'depth_prediction':
                if depth_pred is not None:
                    depth_norm = (depth_pred[0:1] - depth_pred[0:1].min()) / (depth_pred[0:1].max() - depth_pred[0:1].min() + 1e-8)
                    save_image(depth_norm, os.path.join(save_dir, f'depth_pred_{image_idx:03d}.png'))
                
                if depth_gt is not None:
                    depth_gt_norm = (depth_gt[0:1] - depth_gt[0:1].min()) / (depth_gt[0:1].max() - depth_gt[0:1].min() + 1e-8)
                    save_image(depth_gt_norm, os.path.join(save_dir, f'depth_gt_{image_idx:03d}.png'))
            
            # 记录到TensorBoard
            if self.metric_logger and self.vis_config.get('tensorboard_logging', True):
                self.metric_logger.log_image(f"val_comparison", comparison, step=image_idx)
                
        except Exception as e:
            self.val_logger.warning(f"保存对比图像失败: {e}")
    
    def _save_validation_set_results(self, metrics, val_set, epoch, save_dir):
        """保存验证集结果到CSV文件"""
        if not self.vis_config.get('save_detailed_results', True):
            return
        
        try:
            # 保存到验证集专用的CSV文件
            csv_path = os.path.join(save_dir, 'metrics.csv')
            
            # 检查文件是否存在以决定是否写入表头
            file_exists = os.path.exists(csv_path)
            
            with open(csv_path, 'a', newline='') as csvfile:
                # 使用配置定义的指标顺序，缺失用空值占位，确保列稳定
                requested = val_set.get('metrics', [])
                fieldnames = ['epoch'] + requested
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                row = {'epoch': epoch + 1}
                for m in requested:
                    row[m] = metrics.get(m, '')
                writer.writerow(row)
                
            self.val_logger.info(f"验证集 {val_set['name']} 结果已保存到: {csv_path}")
            
        except Exception as e:
            self.val_logger.error(f"保存验证集结果失败: {e}")
    
    def _save_comprehensive_results(self, all_results, epoch, exp_dir):
        """保存综合结果"""
        if not self.vis_config.get('save_detailed_results', True):
            return
        
        try:
            # 保存综合结果到主CSV文件
            comprehensive_csv = os.path.join(exp_dir, 'comprehensive_validation_results.csv')
            
            # 准备数据行
            row_data = {'epoch': epoch + 1}
            # 遍历所有验证集，按配置顺序稳定写列
            for set_id, set_cfg in self.validation_sets.items():
                set_name = set_cfg['name']
                requested = set_cfg.get('metrics', [])
                results = all_results.get(set_id, {}) or {}
                for metric_name in requested:
                    key = f"{set_name}_{metric_name}"
                    row_data[key] = results.get(metric_name, '')
            
            # 检查文件是否存在
            file_exists = os.path.exists(comprehensive_csv)
            
            with open(comprehensive_csv, 'a', newline='') as csvfile:
                    fieldnames = list(row_data.keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(row_data)
            
            self.val_logger.info(f"综合验证结果已保存到: {comprehensive_csv}")
            
        except Exception as e:
            self.val_logger.error(f"保存综合结果失败: {e}")
    
    def _get_main_metrics(self, results, val_type):
        """获取主要指标用于日志显示"""
        main_metrics = {}
        
        if val_type == 'enhancement_with_reference':
            # 全参考增强：PSNR和SSIM
            if 'psnr' in results:
                main_metrics['PSNR'] = results['psnr']
            if 'ssim' in results:
                main_metrics['SSIM'] = results['ssim']
                
        elif val_type == 'enhancement_no_reference':
            # 无参考增强：UCIQE和UIQM
            if 'uciqe' in results:
                main_metrics['UCIQE'] = results['uciqe']
            if 'uiqm' in results:
                main_metrics['UIQM'] = results['uiqm']
                
        elif val_type == 'depth_prediction':
            # 深度预测：MAE和RMSE
            if 'depth_mae' in results:
                main_metrics['MAE'] = results['depth_mae']
            if 'depth_rmse' in results:
                main_metrics['RMSE'] = results['depth_rmse']
        
        return main_metrics
    
    def get_validation_summary(self, all_results):
        """获取验证结果摘要"""
        summary = {}
        
        for set_id, metrics in all_results.items():
            val_set = self.validation_sets[set_id]
            main_metrics = self._get_main_metrics(metrics, val_set['type'])
            
            summary[val_set['name']] = {
                'type': val_set['type'],
                'main_metrics': main_metrics,
                'all_metrics': metrics
            }
        
        return summary
    
    def _create_depth_dataset(self, rgb_folder, depth_folder, set_config):
        """
        创建深度预测数据集
        
        Args:
            rgb_folder: RGB图像文件夹路径
            depth_folder: 深度文件夹路径
            set_config: 验证集配置
            
        Returns:
            DepthValidationDataset: 深度验证数据集实例
        """
        try:
            depth_format = set_config.get('depth_format', 'npy')
            
            dataset = DepthValidationDataset(
                rgb_folder=rgb_folder,
                depth_folder=depth_folder,
                depth_format=depth_format
            )
            
            self.val_logger.info(f"成功创建深度数据集: {set_config['name']}, "
                               f"格式: {depth_format}, 样本数: {len(dataset)}")
            
            return dataset
            
        except Exception as e:
            self.val_logger.error(f"创建深度数据集失败: {e}")
            return None
    
    def _tensor_to_numpy_image(self, tensor):
        """
        将PyTorch张量转换为numpy图像数组
        
        Args:
            tensor: PyTorch张量，格式为[B, C, H, W]或[C, H, W]
            
        Returns:
            numpy array: 格式为[H, W, C]的numpy数组
        """
        # 转换为numpy
        if hasattr(tensor, 'cpu'):
            numpy_array = tensor.cpu().numpy()
        else:
            numpy_array = tensor
        
        # 处理batch维度
        if len(numpy_array.shape) == 4:
            numpy_array = numpy_array[0]  # 取第一个样本 [C, H, W]
        
        # 从[C, H, W]转换为[H, W, C]
        if numpy_array.shape[0] == 3 or numpy_array.shape[0] == 1:  # 通道在第一维
            numpy_array = numpy_array.transpose(1, 2, 0)
        
        return numpy_array 