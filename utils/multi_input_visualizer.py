"""
多输入一致性学习可视化工具
支持显示多个退化级别的输入、输出和一致性特征图
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import os
from typing import Dict, List, Optional, Tuple, Any
import logging


class MultiInputVisualizer:
    """多输入一致性学习的可视化器"""
    
    def __init__(self, save_dir: str, max_degradations: int = 3, figsize: Tuple[int, int] = (15, 10)):
        """
        Args:
            save_dir: 图像保存目录
            max_degradations: 最多显示的退化级别数
            figsize: matplotlib图像尺寸
        """
        self.save_dir = save_dir
        self.max_degradations = max_degradations
        self.figsize = figsize
        self.logger = logging.getLogger('visualization')
        
        os.makedirs(save_dir, exist_ok=True)
    
    def visualize_multi_input_results(self, 
                                    raw_imgs: torch.Tensor,
                                    outputs: Any,
                                    gt: Optional[torch.Tensor] = None,
                                    depth_gt: Optional[torch.Tensor] = None,
                                    step: int = 0,
                                    prefix: str = "train") -> Dict[str, np.ndarray]:
        """
        可视化多输入一致性学习的结果
        
        Args:
            raw_imgs: [B, N, C, H, W] 或 [B, C, H, W] 多退化输入
            outputs: 模型输出，包含 multi_enhanced, consistency_features 等
            gt: [B, C, H, W] 或 [B, N, C, H, W] 真值图像
            depth_gt: [B, 1, H, W] 或 [B, N, 1, H, W] 深度真值
            step: 当前步数
            prefix: 保存前缀
            
        Returns:
            visualization_results: 包含各种可视化结果的字典
        """
        results = {}
        
        # 检查是否为多输入模式
        is_multi_input = hasattr(outputs, 'is_multi_input') and outputs.is_multi_input
        
        if not is_multi_input:
            # 单输入模式的可视化
            return self._visualize_single_input(raw_imgs, outputs, gt, depth_gt, step, prefix)
        
        # 多输入模式的可视化
        B = raw_imgs.shape[0] if raw_imgs.dim() == 5 else raw_imgs.shape[0]
        N = raw_imgs.shape[1] if raw_imgs.dim() == 5 else 1
        
        # 限制显示的退化级别数
        N_show = min(N, self.max_degradations)
        
        self.logger.info(f"可视化多输入结果: B={B}, N={N}, 显示N_show={N_show}")
        
        # 选择第一个样本进行可视化
        sample_idx = 0
        
        try:
            # 🔥 多退化输入可视化
            if raw_imgs.dim() == 5:  # [B, N, C, H, W]
                multi_raw = raw_imgs[sample_idx, :N_show]  # [N_show, C, H, W]
                results['multi_input'] = self._tensor_to_image_grid(multi_raw, title="Multi Degradation Inputs")
            
            # 🔥 多输出增强结果可视化
            if hasattr(outputs, 'multi_enhanced') and outputs.multi_enhanced is not None:
                multi_enhanced = outputs.multi_enhanced[sample_idx, :N_show]  # [N_show, C, H, W]
                results['multi_enhanced'] = self._tensor_to_image_grid(multi_enhanced, title="Multi Enhanced Outputs")
            
            # 🔥 一致性特征可视化
            if hasattr(outputs, 'consistency_features') and outputs.consistency_features is not None:
                consistency_feat = outputs.consistency_features[sample_idx, :N_show]  # [N_show, C_feat, H_feat, W_feat]
                # 对特征进行可视化处理
                consistency_vis = self._visualize_features(consistency_feat)
                results['consistency_features'] = consistency_vis
            
            # 🔥 GT对比（如果有的话）
            if gt is not None:
                if gt.dim() == 5:  # [B, N, C, H, W]
                    gt_show = gt[sample_idx, :N_show]
                elif gt.dim() == 4:  # [B, C, H, W] - 广播显示
                    gt_single = gt[sample_idx:sample_idx+1]  # [1, C, H, W]
                    gt_show = gt_single.repeat(N_show, 1, 1, 1)  # [N_show, C, H, W]
                else:
                    gt_show = None
                
                if gt_show is not None:
                    results['ground_truth'] = self._tensor_to_image_grid(gt_show, title="Ground Truth")
            
            # 🔥 深度预测可视化
            if hasattr(outputs, 'multi_depth_pred') and outputs.multi_depth_pred is not None:
                multi_depth = outputs.multi_depth_pred[sample_idx, :N_show]  # [N_show, 1, H, W]
                depth_vis = self._visualize_depth_maps(multi_depth)
                results['multi_depth'] = depth_vis
            
            # 🔥 创建综合对比图
            comparison_fig = self._create_comprehensive_comparison(
                raw_imgs[sample_idx:sample_idx+1] if raw_imgs.dim() == 5 else raw_imgs[sample_idx:sample_idx+1].unsqueeze(1),
                outputs,
                gt[sample_idx:sample_idx+1] if gt is not None else None,
                N_show
            )
            results['comprehensive_comparison'] = comparison_fig
            
            # 保存结果
            self._save_visualization_results(results, step, prefix)
            
        except Exception as e:
            self.logger.error(f"多输入可视化失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return results
    
    def _visualize_single_input(self, raw_imgs, outputs, gt, depth_gt, step, prefix):
        """单输入模式的可视化（回退方案）"""
        results = {}
        
        try:
            sample_idx = 0
            
            # 输入图像
            if raw_imgs.dim() == 4:  # [B, C, H, W]
                raw_vis = raw_imgs[sample_idx:sample_idx+1]
                results['input'] = self._tensor_to_image_grid(raw_vis, title="Input")
            
            # 增强输出
            if hasattr(outputs, 'enhanced') and outputs.enhanced is not None:
                enhanced_vis = outputs.enhanced[sample_idx:sample_idx+1]
                results['enhanced'] = self._tensor_to_image_grid(enhanced_vis, title="Enhanced")
            
            # GT对比
            if gt is not None:
                gt_vis = gt[sample_idx:sample_idx+1]
                results['ground_truth'] = self._tensor_to_image_grid(gt_vis, title="Ground Truth")
            
            self._save_visualization_results(results, step, prefix)
            
        except Exception as e:
            self.logger.error(f"单输入可视化失败: {e}")
        
        return results
    
    def _tensor_to_image_grid(self, tensor: torch.Tensor, title: str = "", nrow: int = None) -> np.ndarray:
        """将张量转换为图像网格"""
        if tensor.dim() == 4:  # [N, C, H, W]
            if nrow is None:
                nrow = min(tensor.shape[0], 4)  # 每行最多4张图
            
            # 确保值在[0,1]范围内
            if tensor.min() < 0:
                tensor = (tensor + 1) / 2  # [-1,1] -> [0,1]
            tensor = torch.clamp(tensor, 0, 1)
            
            grid = make_grid(tensor, nrow=nrow, padding=2, normalize=False)
            grid_np = grid.permute(1, 2, 0).cpu().numpy()
            
            return grid_np
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
    
    def _visualize_features(self, features: torch.Tensor) -> np.ndarray:
        """可视化特征图"""
        # features: [N, C, H, W]
        N, C, H, W = features.shape
        
        # 选择前几个通道进行可视化
        max_channels = 3
        if C >= 3:
            # 选择前3个通道作为RGB
            feat_vis = features[:, :3]  # [N, 3, H, W]
        else:
            # 重复通道到3个
            feat_vis = features[:, :1].repeat(1, 3, 1, 1)  # [N, 3, H, W]
        
        # 归一化到[0,1]
        feat_vis = feat_vis - feat_vis.min()
        feat_vis = feat_vis / (feat_vis.max() + 1e-8)
        
        return self._tensor_to_image_grid(feat_vis, title="Consistency Features")
    
    def _visualize_depth_maps(self, depth_maps: torch.Tensor) -> np.ndarray:
        """可视化深度图"""
        # depth_maps: [N, 1, H, W]
        
        # 归一化深度图
        depth_norm = depth_maps - depth_maps.min()
        depth_norm = depth_norm / (depth_norm.max() + 1e-8)
        
        # 转换为3通道以便可视化
        depth_rgb = depth_norm.repeat(1, 3, 1, 1)  # [N, 3, H, W]
        
        return self._tensor_to_image_grid(depth_rgb, title="Depth Predictions")
    
    def _create_comprehensive_comparison(self, raw_imgs, outputs, gt, N_show):
        """创建综合对比图"""
        fig, axes = plt.subplots(4, N_show, figsize=self.figsize)
        
        if N_show == 1:
            axes = axes.reshape(-1, 1)
        
        for n in range(N_show):
            # 第1行：输入图像
            if raw_imgs.dim() == 5:  # [1, N, C, H, W]
                raw_img = raw_imgs[0, n].permute(1, 2, 0).cpu().numpy()
            else:  # [1, C, H, W]
                raw_img = raw_imgs[0].permute(1, 2, 0).cpu().numpy()
            
            raw_img = np.clip((raw_img + 1) / 2, 0, 1) if raw_img.min() < 0 else np.clip(raw_img, 0, 1)
            axes[0, n].imshow(raw_img)
            axes[0, n].set_title(f"Input {n+1}")
            axes[0, n].axis('off')
            
            # 第2行：增强输出
            if hasattr(outputs, 'multi_enhanced') and outputs.multi_enhanced is not None:
                enhanced_img = outputs.multi_enhanced[0, n].permute(1, 2, 0).cpu().numpy()
                enhanced_img = np.clip((enhanced_img + 1) / 2, 0, 1) if enhanced_img.min() < 0 else np.clip(enhanced_img, 0, 1)
                axes[1, n].imshow(enhanced_img)
                axes[1, n].set_title(f"Enhanced {n+1}")
            else:
                axes[1, n].text(0.5, 0.5, 'No Enhanced', ha='center', va='center', transform=axes[1, n].transAxes)
            axes[1, n].axis('off')
            
            # 第3行：GT
            if gt is not None:
                if gt.dim() == 5:  # [1, N, C, H, W]
                    gt_img = gt[0, n].permute(1, 2, 0).cpu().numpy()
                else:  # [1, C, H, W]
                    gt_img = gt[0].permute(1, 2, 0).cpu().numpy()
                gt_img = np.clip((gt_img + 1) / 2, 0, 1) if gt_img.min() < 0 else np.clip(gt_img, 0, 1)
                axes[2, n].imshow(gt_img)
                axes[2, n].set_title(f"GT {n+1}")
            else:
                axes[2, n].text(0.5, 0.5, 'No GT', ha='center', va='center', transform=axes[2, n].transAxes)
            axes[2, n].axis('off')
            
            # 第4行：一致性特征
            if hasattr(outputs, 'consistency_features') and outputs.consistency_features is not None:
                feat = outputs.consistency_features[0, n]  # [C, H, W]
                if feat.shape[0] >= 3:
                    feat_img = feat[:3].permute(1, 2, 0).cpu().numpy()
                else:
                    feat_img = feat[0:1].repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy()
                
                # 归一化特征
                feat_img = feat_img - feat_img.min()
                feat_img = feat_img / (feat_img.max() + 1e-8)
                axes[3, n].imshow(feat_img)
                axes[3, n].set_title(f"Feature {n+1}")
            else:
                axes[3, n].text(0.5, 0.5, 'No Features', ha='center', va='center', transform=axes[3, n].transAxes)
            axes[3, n].axis('off')
        
        plt.tight_layout()
        
        # 转换为numpy数组
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return img
    
    def _save_visualization_results(self, results: Dict[str, np.ndarray], step: int, prefix: str):
        """保存可视化结果"""
        for name, img in results.items():
            if isinstance(img, np.ndarray):
                save_path = os.path.join(self.save_dir, f"{prefix}_step_{step:06d}_{name}.png")
                plt.imsave(save_path, img)
                self.logger.debug(f"保存可视化结果: {save_path}")


def calculate_consistency_loss_stats(consistency_features: torch.Tensor) -> Dict[str, float]:
    """
    计算一致性特征的统计信息，用于分析
    
    Args:
        consistency_features: [B, N, C, H, W] 一致性特征
        
    Returns:
        stats: 包含各种统计信息的字典
    """
    if consistency_features is None or consistency_features.dim() != 5:
        return {}
    
    B, N, C, H, W = consistency_features.shape
    stats = {}
    
    # 计算各退化级别之间的相似性
    similarities = []
    for i in range(N):
        for j in range(i + 1, N):
            feat_i = consistency_features[:, i].flatten(1)  # [B, C*H*W]
            feat_j = consistency_features[:, j].flatten(1)  # [B, C*H*W]
            
            # 余弦相似度
            cos_sim = F.cosine_similarity(feat_i, feat_j, dim=1).mean().item()
            similarities.append(cos_sim)
    
    stats['avg_cosine_similarity'] = np.mean(similarities) if similarities else 0.0
    stats['std_cosine_similarity'] = np.std(similarities) if similarities else 0.0
    
    # 计算特征的多样性（方差）
    feat_flat = consistency_features.reshape(B * N, -1)  # [B*N, C*H*W]
    feature_variance = torch.var(feat_flat, dim=0).mean().item()
    stats['feature_variance'] = feature_variance
    
    # 计算跨退化的特征标准差
    cross_degradation_std = torch.std(consistency_features, dim=1).mean().item()  # 在N维度上计算标准差
    stats['cross_degradation_std'] = cross_degradation_std
    
    return stats 