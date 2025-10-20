import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import sys
import logging

# 确保所有损失类都被导出
__all__ = [
    'TotalLoss',
    'ImageLoss',
    'DepthLoss',
    'L1Loss',
    'SSIMLoss', 
    'PerceptualLoss',
    'FFTLoss',
    'GradientLoss',
    'EdgeAwareDepthLoss',
    'DepthSmoothLoss',
    'CrossAttentionConsistencyLoss',
    'DepthReconstructionLoss',
    'CrossDegradationConsistencyLoss',

]

# SSIM loss implementation (simplified)
def gaussian_window(window_size: int, sigma: float, channel: int):
    coords = torch.arange(window_size).to(dtype=torch.float)
    coords -= window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    g = g[:, None] * g[None, :]
    window = g.unsqueeze(0).unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    return window

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.C1 = C1
        self.C2 = C2
        self.register_buffer('window', None)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        # assume img range [-1,1] -> [0,1]
        img1 = torch.clamp((img1 + 1) / 2, 0, 1)
        img2 = torch.clamp((img2 + 1) / 2, 0, 1)
        
        # 确保输入为float32以避免半精度问题
        original_dtype = img1.dtype
        if img1.dtype == torch.float16:
            img1 = img1.float()
            img2 = img2.float()
            
        B, C, H, W = img1.size()
        window = gaussian_window(self.window_size, self.sigma, C).to(img1.device, dtype=img1.dtype)
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=C)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=C)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=C) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=C) - mu1_mu2
        
        # 添加小的epsilon防止分母为0
        eps = 1e-12
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2) + eps)
        loss = torch.clamp((1 - ssim_map) / 2, 0, 1).mean()
        
        # 恢复到原始数据类型
        if original_dtype == torch.float16:
            loss = loss.half()
            
        return loss

class PerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[4,9,18,27], use_gpu=True):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        if use_gpu:
            vgg = vgg.cuda()
        self.layers = layer_ids
        self.vgg_layers = nn.ModuleList([vgg[i] for i in range(max(self.layers)+1)])
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        self.criterion = nn.MSELoss()

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        # expect img normalized to ImageNet stats
        x = img1
        y = img2
        loss = 0.0
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            y = layer(y)
            if i in self.layers:
                loss += self.criterion(x, y)
        return loss

class FFTLoss(nn.Module):
    """频域 L1 损失"""
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        # img in [-1,1]
        # 深度诊断NaN产生的原因
        
        # 1. 检查输入数据的基本信息
        img1_stats = {
            'min': img1.min().item(),
            'max': img1.max().item(), 
            'mean': img1.mean().item(),
            'std': img1.std().item(),
            'has_nan': torch.isnan(img1).any().item(),
            'has_inf': torch.isinf(img1).any().item()
        }
        
        img2_stats = {
            'min': img2.min().item(),
            'max': img2.max().item(),
            'mean': img2.mean().item(), 
            'std': img2.std().item(),
            'has_nan': torch.isnan(img2).any().item(),
            'has_inf': torch.isinf(img2).any().item()
        }
        
        # 2. 如果发现异常，详细报告
        if img1_stats['has_nan'] or img2_stats['has_nan']:
            logging.getLogger('error').error(f"[FFT-LOSS] 致命错误: 输入包含NaN! img1_stats: {img1_stats}, img2_stats: {img2_stats}")
            # 不要返回0，让错误传播
            
        if img1_stats['has_inf'] or img2_stats['has_inf']:
            logging.getLogger('error').error(f"[FFT-LOSS] 致命错误: 输入包含无穷大值! img1_stats: {img1_stats}, img2_stats: {img2_stats}")
            
        # 3. 检查数值范围是否异常
        if abs(img1_stats['max']) > 100 or abs(img1_stats['min']) > 100:
            logging.getLogger('warning').warning(f"[FFT-LOSS] 警告: img1值域异常 [{img1_stats['min']:.4f}, {img1_stats['max']:.4f}]")
            
        if abs(img2_stats['max']) > 100 or abs(img2_stats['min']) > 100:
            logging.getLogger('warning').warning(f"[FFT-LOSS] 警告: img2值域异常 [{img2_stats['min']:.4f}, {img2_stats['max']:.4f}]")
        
        # 4. 执行FFT，使用更稳定的实现
        try:
            # 确保输入在float32精度下进行FFT计算，避免半精度问题
            img1_fp32 = img1.float()
            img2_fp32 = img2.float()
            
            # 添加小的噪声避免某些FFT的数值问题
            eps = 1e-8
            img1_fp32 = img1_fp32 + eps * torch.randn_like(img1_fp32)
            img2_fp32 = img2_fp32 + eps * torch.randn_like(img2_fp32)
            
            # 使用更稳定的FFT实现
            f1 = torch.fft.rfft2(img1_fp32, norm='ortho')
            f2 = torch.fft.rfft2(img2_fp32, norm='ortho')
            
            # 检查FFT结果
            if torch.isnan(f1).any():
                logging.getLogger('warning').warning(f"[FFT-LOSS] FFT(img1)仍然产生NaN，回退到空间域损失")
                return F.l1_loss(img1, img2)
            if torch.isnan(f2).any():
                logging.getLogger('warning').warning(f"[FFT-LOSS] FFT(img2)仍然产生NaN，回退到空间域损失")
                return F.l1_loss(img1, img2)
                
            f1_abs = torch.abs(f1)
            f2_abs = torch.abs(f2)
            
            # 检查绝对值计算
            if torch.isnan(f1_abs).any() or torch.isnan(f2_abs).any():
                logging.getLogger('warning').warning(f"[FFT-LOSS] FFT幅度计算产生NaN，回退到空间域损失")
                return F.l1_loss(img1, img2)
                
            loss = self.l1(f1_abs, f2_abs)
            
            # 检查最终损失
            if torch.isnan(loss):
                logging.getLogger('warning').warning(f"[FFT-LOSS] L1损失产生NaN，回退到空间域损失")
                return F.l1_loss(img1, img2)
                
            # 转换回原始数据类型
            if img1.dtype != torch.float32:
                loss = loss.to(img1.dtype)
                
            return loss
            
        except Exception as e:
            logging.getLogger('error').error(f"[FFT-LOSS] FFT计算异常: {e}，回退到空间域损失", exc_info=True)
            return F.l1_loss(img1, img2)

class GradientLoss(nn.Module):
    """高频梯度 L1 损失"""
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        def gradient(x):
            gx = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:])
            gy = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1])
            return gx, gy
        gx1, gy1 = gradient(img1)
        gx2, gy2 = gradient(img2)
        return self.l1(gx1, gx2) + self.l1(gy1, gy2)

class DepthSmoothLoss(nn.Module):
    """门控图平滑 L1 损失"""
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, G: torch.Tensor):
        # G: [B,1,H,W]
        dx = torch.abs(G[:,:,1:,:] - G[:,:,:-1,:])
        dy = torch.abs(G[:,:,:,1:] - G[:,:,:,:-1])
        return dx.mean() + dy.mean()

class EdgeAwareDepthLoss(nn.Module):
    """边缘感知深度损失

    结合了L1损失和边缘加权梯度损失，能够更好地保留深度边界
    """
    def __init__(self, edge_weight=0.5, min_depth=2000.0, max_depth=65535.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.edge_weight = edge_weight
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.register_buffer('eps', torch.tensor(1e-8), persistent=False)
        
        # Sobel算子用于检测深度边缘
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 初始化Sobel算子权重
        sobel_x_kernel = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        sobel_y_kernel = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
        self.sobel_x.weight.data = sobel_x_kernel
        self.sobel_y.weight.data = sobel_y_kernel
        
        # 冻结参数，确保不会在反向传播中更新
        for param in self.parameters():
            param.requires_grad = False
        
        # 设备管理标志
        self._device_set = False
        
        # 添加统计计数器和调试标志
        self.debug_mode = True  # 设置为True以输出额外的调试信息
        self.calls_count = 0
        self.pred_min_max_history = []
        self.pred_stats_history = []

    def _normalize_depths(self, depth):
        """对深度图进行预处理，统一到[0,1]范围"""
        return (depth - self.min_depth) / (self.max_depth - self.min_depth + self.eps)
        
    def _compute_depth_edges(self, depth):
        """计算深度图的梯度图"""
        # 确保Sobel算子在正确的设备上
        if not self._device_set or self.sobel_x.weight.device != depth.device:
            self.sobel_x = self.sobel_x.to(depth.device)
            self.sobel_y = self.sobel_y.to(depth.device)
            self._device_set = True
        
        # 首先归一化到[0,1]
        depth_norm = self._normalize_depths(depth)
        
        # 计算水平和垂直梯度
        grad_x = self.sobel_x(depth_norm)
        grad_y = self.sobel_y(depth_norm)
        
        # 梯度幅度
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.eps)
        return grad_mag

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Args:
            pred: 预测深度 [B,1,H,W]
            target: 目标深度 [B,1,H,W]
        """
        # 递增调用计数
        self.calls_count += 1
        
        # 获取预测深度统计信息
        pred_min = pred.min().item()
        pred_max = pred.max().item()
        pred_mean = pred.mean().item()
        pred_std = pred.std().item()
        self.pred_min_max_history.append((pred_min, pred_max))
        self.pred_stats_history.append((pred_mean, pred_std))
        
        # 只有在debug_mode开启时才输出调试信息，并减少输出频率
        if self.debug_mode and self.calls_count % 100 == 0:
            # 计算最近的统计信息平均值
            recent_history = self.pred_min_max_history[-20:]  # 最近20次调用的记录
            if recent_history:
                avg_min = sum(mm[0] for mm in recent_history) / len(recent_history)
                avg_max = sum(mm[1] for mm in recent_history) / len(recent_history)
                avg_range = avg_max - avg_min
                
                recent_stats = self.pred_stats_history[-20:]
                avg_mean = sum(ms[0] for ms in recent_stats) / len(recent_stats)
                avg_std = sum(ms[1] for ms in recent_stats) / len(recent_stats)
                
                # Use logger instead of print
                logging.getLogger('depth').info(f"[DEPTH-LOSS] 统计: 范围[{avg_min:.4f}, {avg_max:.4f}], 均值{avg_mean:.4f}, 标准差{avg_std:.4f}")
        
        # 检测目标深度是否已经归一化
        target_min = target.min().item()
        target_max = target.max().item()
        
        # 增强的已处理深度图检测 
        is_normalized = False
        target_normalized = False
        pred_normalized = False
        
        # 1. 检查目标深度是否已归一化
        if hasattr(target, '_depth_processed') and target._depth_processed:
            target_normalized = True
        # 2. 通过值域检测
        elif target_max < self.min_depth or (target_max <= 1.0 and target_min >= 0.0):
            target_normalized = True
        
        # 3. 检查预测深度是否已归一化
        if hasattr(pred, '_depth_processed') and pred._depth_processed:
            pred_normalized = True
        # 4. 通过值域检测
        elif pred.max().item() <= 1.0 and pred.min().item() >= 0.0:
            pred_normalized = True
            
        # 归一化处理
        if target_normalized and pred_normalized:
            # 两者都已归一化，直接计算损失
            is_normalized = True
            valid_mask = torch.ones_like(target, dtype=torch.bool)
            # 获取目标深度的边缘
            gt_edges = self._compute_normalized_edges(target)
        elif target_normalized and not pred_normalized:
            # 目标已归一化，预测未归一化，归一化预测深度
            is_normalized = True
            valid_mask = torch.ones_like(target, dtype=torch.bool)
            # 归一化预测深度
            pred = self._normalize_depths(pred)
            # 获取目标深度的边缘
            gt_edges = self._compute_normalized_edges(target)
        elif not target_normalized and pred_normalized:
            # 目标未归一化，预测已归一化，归一化目标深度
            is_normalized = True
            # 创建有效深度掩码
            valid_mask = (target > self.min_depth) & (target < self.max_depth)
            valid_ratio = valid_mask.float().mean().item()
            
            # 归一化目标深度
            target_norm = torch.zeros_like(target)
            target_norm[valid_mask] = self._normalize_depths(target[valid_mask])
            target = target_norm
            
            # 获取归一化后的目标深度边缘
            gt_edges = self._compute_depth_edges(target)
        else:
            # 两者都未归一化，创建有效深度掩码并归一化两者
            # 如果是原始深度，创建有效深度掩码并进行归一化
            valid_mask = (target > self.min_depth) & (target < self.max_depth)
            valid_ratio = valid_mask.float().mean().item()
            
            # 如果没有有效像素，返回零损失
            if not valid_mask.any():
                if self.debug_mode:
                    logging.getLogger('warning').warning(f"[DEPTH-LOSS] 警告: 没有有效像素在 [{self.min_depth}, {self.max_depth}] 范围内")
                return {'total': torch.tensor(0.0, device=pred.device)}
            
            # 归一化目标深度和预测深度
            target_norm = torch.zeros_like(target)
            pred_norm = torch.zeros_like(pred)
            
            target_norm[valid_mask] = self._normalize_depths(target[valid_mask])
            pred_norm[valid_mask] = self._normalize_depths(pred[valid_mask])
            
            target = target_norm
            pred = pred_norm
            
            # 提取边缘信息
            gt_edges = self._compute_depth_edges(target)
        
        # 计算基础L1损失
        l1_loss = torch.abs(pred - target) * valid_mask.float()
        
        # 基于边缘的权重 - 在边缘处有更高权重
        edge_weights = 1.0 + self.edge_weight * gt_edges
        
        # 应用边缘权重到损失上
        weighted_loss = l1_loss * edge_weights
        
        # 计算平均损失
        num_valid = valid_mask.sum().float() + self.eps
        
        # 如果是原始深度图且有效像素比例低，调整权重
        if not is_normalized and 'valid_ratio' in locals() and valid_ratio < 0.3:
            # 根据有效像素比例调整损失
            adjust_factor = max(0.1, valid_ratio / 0.3)  # 至少保留10%的损失权重
            loss = weighted_loss.sum() / num_valid * adjust_factor
        else:
            loss = weighted_loss.sum() / num_valid
        
        # 返回字典
        return {'total': loss, 'l1': l1_loss.mean(), 'edge_weighted': weighted_loss.mean()}
        
    def _compute_normalized_edges(self, depth):
        """计算已归一化深度图的梯度图"""
        # 确保Sobel算子在正确的设备上
        if not self._device_set or self.sobel_x.weight.device != depth.device:
            self.sobel_x = self.sobel_x.to(depth.device)
            self.sobel_y = self.sobel_y.to(depth.device)
            self._device_set = True
        
        # 直接计算水平和垂直梯度
        grad_x = self.sobel_x(depth)
        grad_y = self.sobel_y(depth)
        
        # 梯度幅度
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.eps)
        return grad_mag

class CrossAttentionConsistencyLoss(nn.Module):
     """
     交叉注意力对偶一致性损失 (软约束版)
     -----------------------
     使用对称 KL 散度(= KL(P‖Q)+KL(Q‖P)) 将 Depth→RGB 和 RGB→Depth 注意力拉向对称，而非硬性 L1/L2 差分。
     输入张量 shape: [B, heads, N, N]
     """
     def __init__(self, reduction='mean', eps: float = 1e-8):
         super().__init__()
         self.reduction = reduction
         self.eps = eps
 
     def _symmetric_kl(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
         """计算对称 KL 散度: KL(p‖q)+KL(q‖p)"""
         kl_pq = (p * (torch.log(p + self.eps) - torch.log(q + self.eps))).sum(dim=(-2, -1))
         kl_qp = (q * (torch.log(q + self.eps) - torch.log(p + self.eps))).sum(dim=(-2, -1))
         return kl_pq + kl_qp
 
     def forward(self, attn_d2r: torch.Tensor, attn_r2d: torch.Tensor) -> torch.Tensor:
         """计算软对称一致性损失"""
         if attn_d2r is None or attn_r2d is None:
             # 任一注意力缺失则返回 0 损失
             device = attn_d2r.device if attn_d2r is not None else attn_r2d.device
             return torch.tensor(0.0, device=device, requires_grad=False)
 
         # 保证形状匹配
         if attn_d2r.shape != attn_r2d.shape:
             raise ValueError(f"Attention shape mismatch: d2r {attn_d2r.shape}, r2d {attn_r2d.shape}")
 
         # 转置 r2d 以与 d2r 对齐
         attn_r2d_T = attn_r2d.transpose(-2, -1)
 
         # 归一化到概率分布 (确保行和为1)，使用 softmax 提升数值稳定
         p = torch.softmax(attn_d2r, dim=-1)
         q = torch.softmax(attn_r2d_T, dim=-1)
 
         # 计算对称 KL
         loss = self._symmetric_kl(p, q)
 
         # 根据 reduction 处理 batch/head 维
         if self.reduction == 'mean':
             return loss.mean()
         elif self.reduction == 'sum':
             return loss.sum()
         else:
             # 无 reduction, 保持 [B, heads] 形状
             return loss

class ImageLoss(nn.Module):
    """
    图像重建损失组合
    --------------
    组合多种图像重建损失，包括L1、SSIM、感知损失、频域损失和梯度损失
    
    Args:
        lambda_l1: L1损失权重
        lambda_ssim: SSIM损失权重
        lambda_perceptual: 感知损失权重（如果可用）
        lambda_fft: 频域损失权重
        lambda_grad: 梯度损失权重
    """
    def __init__(self, 
                 lambda_l1=1.0, 
                 lambda_ssim=0.5, 
                 lambda_perceptual=0.1,
                 lambda_fft=0.1,
                 lambda_grad=0.1):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_perceptual = lambda_perceptual
        self.lambda_fft = lambda_fft
        self.lambda_grad = lambda_grad
        
        # 实例化各个损失函数
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.fft_loss = FFTLoss()
        self.grad_loss = GradientLoss()
        
        # 尝试实例化感知损失（如果可用）
        try:
            import lpips
            self.perceptual_loss = lpips.LPIPS(net='vgg').eval()
            self.has_perceptual = True
            self._perceptual_device_set = False
        except ImportError:
            self.has_perceptual = False
            self.perceptual_loss = None
            print("LPIPS库未安装，将不使用感知损失。可使用'pip install lpips'安装")
        
        # 存储最近的损失值
        self.last_losses = {}
        
    def forward(self, pred, target):
        """
        计算图像重建损失
        
        Args:
            pred: 预测图像 [B,3,H,W]
            target: 目标图像 [B,3,H,W]
            
        Returns:
            total_loss: 总图像重建损失
        """
        self.last_losses = {}
        
        # 1. L1 损失
        l1_loss = self.l1_loss(pred, target)
        total_loss = self.lambda_l1 * l1_loss
        self.last_losses['l1_loss'] = l1_loss.item()
        
        # 2. SSIM 损失
        ssim_loss = self.ssim_loss(pred, target)
        total_loss += self.lambda_ssim * ssim_loss
        self.last_losses['ssim_loss'] = ssim_loss.item()
        
        # 3. 感知损失（如果可用）
        if self.has_perceptual:
            # 确保LPIPS在正确的设备上
            if not self._perceptual_device_set or next(self.perceptual_loss.parameters()).device != pred.device:
                self.perceptual_loss = self.perceptual_loss.to(pred.device)
                self._perceptual_device_set = True
            
            # LPIPS期望输入在[-1,1]范围内
            perceptual_loss = self.perceptual_loss(pred, target).mean()
            total_loss += self.lambda_perceptual * perceptual_loss
            self.last_losses['perceptual_loss'] = perceptual_loss.item()
        else:
            perceptual_loss = torch.tensor(0.0, device=pred.device)
        
        # 4. 频域损失
        fft_loss = self.fft_loss(pred, target)
        total_loss += self.lambda_fft * fft_loss
        self.last_losses['fft_loss'] = fft_loss.item()
        
        # 5. 梯度损失
        grad_loss = self.grad_loss(pred, target)
        total_loss += self.lambda_grad * grad_loss
        self.last_losses['grad_loss'] = grad_loss.item()
        
        self.last_losses['img_loss_total'] = total_loss.item()
        
        # 修改返回值：返回字典而不是总损失值
        losses_dict = {
            'l1': l1_loss,
            'ssim': ssim_loss,
            'perceptual': perceptual_loss if self.has_perceptual else torch.tensor(0.0, device=pred.device),
            'fft': fft_loss,
            'gradient': grad_loss,
            'total': total_loss
        }
        
        return losses_dict

class DepthLoss(nn.Module):
    """
    深度重建损失组合
    --------------
    组合深度预测损失和深度平滑损失
    
    Args:
        lambda_depth: 深度预测损失权重
        lambda_smooth: 深度平滑损失权重
    """
    def __init__(self, lambda_depth=1.0, lambda_smooth=0.01, min_depth=2000.0, max_depth=65535.0):
        super().__init__()
        self.lambda_depth = lambda_depth
        self.lambda_smooth = lambda_smooth
        
        # 实例化深度损失函数，传递深度范围参数
        self.depth_loss = EdgeAwareDepthLoss(
            edge_weight=0.5,
            min_depth=min_depth,
            max_depth=max_depth
        )
        self.smooth_loss = DepthSmoothLoss()
        
        # 存储最近的损失值
        self.last_losses = {}
        
    def forward(self, pred, target):
        """
        计算深度重建损失
        
        Args:
            pred: 预测深度图 [B,1,H,W]
            target: 目标深度图 [B,1,H,W]
            
        Returns:
            total_loss: 总深度重建损失
        """
        self.last_losses = {}
        
        # 1. 深度预测损失
        depth_loss = self.depth_loss(pred, target)
        total_loss = self.lambda_depth * depth_loss['total']
        self.last_losses['depth_pred_loss'] = depth_loss['total'].item()
        
        # 2. 深度平滑损失
        smooth_loss = self.smooth_loss(pred)
        total_loss += self.lambda_smooth * smooth_loss
        self.last_losses['depth_smooth_loss'] = smooth_loss.item()
        
        self.last_losses['depth_loss_total'] = total_loss.item()
        
        # 修改返回值：返回字典而不是总损失值
        losses_dict = {
            'depth_pred': depth_loss['total'],
            'depth_smooth': smooth_loss,
            'total': total_loss
        }
        
        return losses_dict

class DepthReconstructionLoss(nn.Module):
    """多尺度深度重建损失，结合L1损失和SSIM损失"""
    def forward(self, pred, gt):
        """
        深度重建损失
        
        Args:
            pred: 预测的深度图 [B,1,H,W] 或深度引导的RGB图像 [B,3,H,W]
            gt: 目标RGB图像 [B,3,H,W] 或深度图 [B,1,H,W]
        
        Returns:
            losses_dict: 包含各损失组件的字典
        """
        # 确保输入数据类型一致，避免mixed precision训练中的类型不匹配
        if pred.dtype != gt.dtype:
            # 统一转换为float32以确保兼容性
            pred = pred.float()
            gt = gt.float()
        
        # 确保输入尺寸一致
        if pred.shape[-2:] != gt.shape[-2:]:
            # 调整预测图的大小以匹配目标图
            pred = F.interpolate(pred, size=gt.shape[-2:], mode='bilinear', align_corners=False)
        
        # 计算L1损失
        l1 = torch.abs(pred - gt).mean()
        
        # 计算SSIM损失
        ssim_l = (1 - ssim(pred, gt)).mean()
        
        # 计算总损失
        total_loss = l1 + 0.4 * ssim_l
        
        # 返回字典而不是单一值
        losses_dict = {
            'l1': l1,
            'ssim': ssim_l,
            'total': total_loss
        }
        
        return losses_dict

class TotalLoss(nn.Module):
    """
    总损失函数类，组合多种损失
    
    组件:
    1. 图像损失组
        - L1 损失: lambda_img * L1(pred, target)
        - SSIM 损失: lambda_img * lambda_ssim * SSIM(pred, target)
        - 感知损失: lambda_img * lambda_perc * Perceptual(pred, target)
        - FFT 损失: lambda_img * lambda_fft * FFT(pred, target)
        - 梯度损失: lambda_img * lambda_grad * Gradient(pred, target)
    
    2. 深度损失组
        - 深度预测损失: lambda_depth * L1(depth_pred, depth_gt)
        - 深度平滑损失: lambda_depth * lambda_smooth * Smooth(depth_pred)
        - 深度重建损失: lambda_depth * SSIM(pred_from_depth, target)
        
    3. 注意力一致性损失: lambda_cons * ConsistencyLoss(depth2rgb_attn, rgb2depth_attn)
    """
    def __init__(self,
                 # group-level lambdas removed; rely on uncertainty or fine-grained manual lambdas only
                 lambda_smooth=0.01,
                  # fine-grained lambdas for manual weighting (when uncertainty weighting is off)
                  lambda_img_l1: float = 1.0,
                  lambda_img_ssim: float = 1.0,
                  lambda_img_perc: float = 1.0,
                  lambda_img_fft: float = 1.0,
                  lambda_img_grad: float = 1.0,
                  lambda_depth_decoder: float = 1.0,
                  lambda_depth_smooth: float = 1.0,
                  lambda_depth_rec: float = 1.0,
                  lambda_cons: float = 1.0,
                  # 🔥 CMCL相关参数
                  lambda_cmcl=0.1,         # CMCL总权重
                  lambda_cmcl_var=1.0,     # 方差损失权重
                  lambda_cmcl_rgb=1.0,     # RGB一致性权重  
                  lambda_cmcl_depth=1.0,   # 深度一致性权重
                 cmcl_k_decay=1.0,        # 深度衰减系数
                 use_uncertainty_weighting=True,
                 sigma_init=None,
                 min_depth=2000.0,
                 max_depth=65535.0):
        super().__init__()
        
        # 获取日志记录器
        self.metrics_logger = logging.getLogger('metrics')
        self.depth_logger = logging.getLogger('depth')
        self.physics_logger = logging.getLogger('physics')
        self.attention_logger = logging.getLogger('attention')
        self.optimizer_logger = logging.getLogger('optimizer')
        self.warning_logger = logging.getLogger('warning')
        
        # 权重参数
        self.lambda_smooth = lambda_smooth
        # fine-grained lambdas (manual weighting)
        self.lambda_img_l1 = lambda_img_l1
        self.lambda_img_ssim = lambda_img_ssim
        self.lambda_img_perc = lambda_img_perc
        self.lambda_img_fft = lambda_img_fft
        self.lambda_img_grad = lambda_img_grad
        self.lambda_depth_decoder = lambda_depth_decoder
        self.lambda_depth_smooth = lambda_depth_smooth
        self.lambda_depth_rec = lambda_depth_rec
        self.lambda_cons = lambda_cons
        # 特征一致性损失已删除

        # 🔥 CMCL权重参数
        self.lambda_cmcl = lambda_cmcl
        self.lambda_cmcl_var = lambda_cmcl_var
        self.lambda_cmcl_rgb = lambda_cmcl_rgb
        self.lambda_cmcl_depth = lambda_cmcl_depth
        self.cmcl_k_decay = cmcl_k_decay
        
        # 组件损失
        self.image_loss = ImageLoss(
            lambda_l1=1.0,
            lambda_ssim=1.0,
            lambda_perceptual=1.0,
            lambda_fft=1.0,
            lambda_grad=1.0
        )
        
        self.depth_loss = DepthLoss(
            lambda_depth=1.0,
            lambda_smooth=lambda_smooth,
            min_depth=min_depth,
            max_depth=max_depth
        )
        
        
        self.attention_consistency_loss = CrossAttentionConsistencyLoss()
        
        # 删除：重投影一致性损失（已简化为仅使用CMCL）
        
        # 🔥 新增：跨退化一致性损失
        self.cmcl_loss = CrossDegradationConsistencyLoss(
            lambda_var=lambda_cmcl_var,
            lambda_rgb=lambda_cmcl_rgb,
            lambda_depth=lambda_cmcl_depth,
            k_decay=cmcl_k_decay
        )
        
        # 不确定性加权
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        if use_uncertainty_weighting:
            # 🔥 使用配置文件中的sigma_init参数（细粒度覆盖）
            if sigma_init is not None:
                # 基础组别默认
                rgb_sigma = sigma_init.get('rgb', 0.2)
                depth_sigma = sigma_init.get('depth', 1.0)
                                 # depth_rec removed
                # 细粒度覆盖（若未提供则回退到所属组别默认）
                l1_sigma   = sigma_init.get('rgb_l1',   rgb_sigma)
                ssim_sigma = sigma_init.get('rgb_ssim', rgb_sigma)
                perc_sigma = sigma_init.get('rgb_perc', rgb_sigma)
                fft_sigma  = sigma_init.get('rgb_fft',  rgb_sigma)
                grad_sigma = sigma_init.get('rgb_grad', rgb_sigma)

                depth_decoder_sigma = sigma_init.get('depth_decoder', depth_sigma)
                depth_smooth_sigma  = sigma_init.get('depth_smooth',  depth_sigma)

                attncons_sigma  = sigma_init.get('attncons',  depth_sigma)
                cmcl_sigma  = sigma_init.get('cmcl',  rgb_sigma)
                # CMCL 细粒度：如未提供各自sigma，回退到cmcl_sigma
                cmcl_var_sigma   = sigma_init.get('cmcl_var',   cmcl_sigma)
                cmcl_rgb_sigma   = sigma_init.get('cmcl_rgb',   cmcl_sigma)
                cmcl_depth_sigma = sigma_init.get('cmcl_depth', cmcl_sigma)

                # 转换为log_var: log(σ²) = 2*log(σ)
                l1_log   = 2 * math.log(l1_sigma)
                ssim_log = 2 * math.log(ssim_sigma)
                perc_log = 2 * math.log(perc_sigma)
                fft_log  = 2 * math.log(fft_sigma)
                grad_log = 2 * math.log(grad_sigma)

                depth_dec_log   = 2 * math.log(depth_decoder_sigma)
                depth_smooth_log= 2 * math.log(depth_smooth_sigma)
                # depth_rec_log removed

                attncons_log = 2 * math.log(attncons_sigma)
                cmcl_var_log   = 2 * math.log(cmcl_var_sigma)
                cmcl_rgb_log   = 2 * math.log(cmcl_rgb_sigma)
                cmcl_depth_log = 2 * math.log(cmcl_depth_sigma)

                self.depth_logger.info(
                    "使用自定义sigma_init: "
                    f"rgb(l1={l1_sigma}, ssim={ssim_sigma}, perc={perc_sigma}, fft={fft_sigma}, grad={grad_sigma}); "
                    f"depth(dec={depth_decoder_sigma}, smooth={depth_smooth_sigma}); "
                    f"attncons={attncons_sigma}, cmcl_var={cmcl_var_sigma}, cmcl_rgb={cmcl_rgb_sigma}, cmcl_depth={cmcl_depth_sigma}"
                )
            else:
                # 默认值：更保守的初始化策略
                # RGB组默认
                l1_log = ssim_log = perc_log = fft_log = grad_log = -0.5   # exp(0.5)≈1.65
                # 深度组默认
                depth_dec_log = depth_smooth_log = -1.0                    # exp(1)≈2.7
                # depth_rec_log removed
                # 其他
                attncons_log = depth_dec_log
                cmcl_var_log = cmcl_rgb_log = cmcl_depth_log = -0.5
                self.depth_logger.info(
                    f"使用默认sigma_init: rgb_log={-0.5:.4f}, depth_log={-1.0:.4f}, attncons_log={attncons_log:.4f}, "
                    f"cmcl_var_log={cmcl_var_log:.4f}, cmcl_rgb_log={cmcl_rgb_log:.4f}, cmcl_depth_log={cmcl_depth_log:.4f}")
            
            # 依据上面计算结果初始化log_var参数
            self.log_var_l1   = nn.Parameter(torch.ones(1) * l1_log)
            self.log_var_ssim = nn.Parameter(torch.ones(1) * ssim_log)
            self.log_var_perc = nn.Parameter(torch.ones(1) * perc_log)
            self.log_var_fft  = nn.Parameter(torch.ones(1) * fft_log)
            self.log_var_grad = nn.Parameter(torch.ones(1) * grad_log)
            
            self.log_var_depth_decoder = nn.Parameter(torch.ones(1) * depth_dec_log)
            self.log_var_depth_smooth  = nn.Parameter(torch.ones(1) * depth_smooth_log)
            # self.log_var_depth_rec removed
            
            self.log_var_attncons = nn.Parameter(torch.ones(1) * attncons_log)
            # CMCL 拆分为三个独立不确定性参数
            self.log_var_cmcl_var   = nn.Parameter(torch.ones(1) * cmcl_var_log)
            self.log_var_cmcl_rgb   = nn.Parameter(torch.ones(1) * cmcl_rgb_log)
            self.log_var_cmcl_depth = nn.Parameter(torch.ones(1) * cmcl_depth_log)
            
            # 🔥 添加参数约束，防止log_var过度变化
            self.log_var_min = -3.0  # 最小值，防止权重过大
            self.log_var_max = 2.0   # 最大值，防止权重过小
        
        # 记录损失值
        self.losses = {}
        self._forward_count = 0 # 内部计数器，用于控制日志频率
    
    def get_latest_losses(self):
        return self.losses.copy()
    
    def forward(self, pred, target, depth_gt=None, 
                student_feats=None, attention_maps=None, depth_pred=None,
                raw=None, multi_enhanced=None, multi_res_d=None, multi_res_c=None,
                multi_depth_pred=None):
        """
        计算总损失 - 支持多候选输出的全面损失计算
        
        Args:
            pred: 主预测的RGB图像 [B,3,H,W] - 通常是多候选的集成结果
            target: 目标RGB图像 [B,3,H,W]
            depth_gt: 深度GT [B,1,H,W] or None
            student_feats: 学生特征列表 or None
            attention_maps: 注意力图元组 (depth2rgb, rgb2depth) or None
            depth_pred: 主深度预测 [B,1,H,W] or None - 通常是多候选的集成结果
            raw: 原始退化水下图 [B,3,H,W]
            multi_enhanced: 多候选增强结果 [B,N,3,H,W] or None - 所有N个候选都参与损失计算
            multi_depth_pred: 多候选深度预测 [B,N,1,H,W] or None - 所有N个候选都参与损失计算
            multi_res_d: 多候选去模糊残差 [B,N,3,H,W] or None
            multi_res_c: 多候选颜色校正残差 [B,N,3,H,W] or None
        
        Returns:
            total_loss: 标量总损失 (包含主输出损失 + 多候选平均损失 + 一致性损失)
        """
        self._forward_count += 1
        log_this_step = (self._forward_count % 100 == 1) # 每100步记录一次详细日志
        
        if log_this_step:
            self.metrics_logger.info(f"--- [Loss Calculation Step {self._forward_count}] ---")

        device = pred.device
        
        # 🔧 重构：仅对每一路输出分别监督（多候选或单候选回退），不再对融合主输出单独计算损失
        img_l1 = torch.tensor(0.0, device=device)
        img_ssim = torch.tensor(0.0, device=device)
        img_perc = torch.tensor(0.0, device=device)
        img_fft = torch.tensor(0.0, device=device)
        img_grad = torch.tensor(0.0, device=device)
        num_img = 0

        if multi_enhanced is not None and multi_enhanced.dim() == 5:
            # multi_enhanced: [B, N, C, H, W]
            B, N, C, H, W = multi_enhanced.shape
            if log_this_step:
                self.metrics_logger.info(f"  [Multi-Candidate] Processing {N} image candidates (no fused primary loss)")
            # 对每个候选计算图像损失并累计
            for i in range(N):
                candidate_enhanced = multi_enhanced[:, i]  # [B, C, H, W]
                candidate_img_losses = self.image_loss(candidate_enhanced, target)
                img_l1 += candidate_img_losses['l1']
                img_ssim += candidate_img_losses['ssim']
                img_perc += candidate_img_losses['perceptual']
                img_fft += candidate_img_losses['fft']
                img_grad += candidate_img_losses['gradient']
            num_img = N
        elif pred is not None and target is not None:
            # 单候选回退：按一路输出监督
            candidate_img_losses = self.image_loss(pred, target)
            img_l1 += candidate_img_losses['l1']
            img_ssim += candidate_img_losses['ssim']
            img_perc += candidate_img_losses['perceptual']
            img_fft += candidate_img_losses['fft']
            img_grad += candidate_img_losses['gradient']
            num_img = 1

        if num_img > 0:
            l1_loss   = img_l1 / num_img
            ssim_loss = img_ssim / num_img
            perc_loss = img_perc / num_img
            fft_loss  = img_fft / num_img
            grad_loss = img_grad / num_img
        else:
            l1_loss = ssim_loss = perc_loss = fft_loss = grad_loss = torch.tensor(0.0, device=device)

        img_total_loss = (l1_loss + ssim_loss + perc_loss + fft_loss + grad_loss)
        
        if log_this_step:
            self.metrics_logger.info(f"  [Image] avg_per_path: l1={l1_loss:.4f}, ssim={ssim_loss:.4f}, perceptual={perc_loss:.4f}, fft={fft_loss:.4f}, grad={grad_loss:.4f}")

        # 深度损失组：对每一路深度输出分别监督（多候选或单候选回退）

        depth_decoder_loss = torch.tensor(0.0, device=device)
        depth_smooth_loss  = torch.tensor(0.0, device=device)
        num_depth = 0

        if multi_depth_pred is not None and depth_gt is not None and multi_depth_pred.dim() == 5:
            B, N, C, H, W = multi_depth_pred.shape
            if log_this_step:
                self.depth_logger.info(f"  [Multi-Candidate Depth] Processing {N} depth candidates (no fused primary loss)")
            for i in range(N):
                candidate_depth = multi_depth_pred[:, i]
                candidate_losses = self.depth_loss(candidate_depth, depth_gt)
                depth_decoder_loss += candidate_losses['depth_pred']
                depth_smooth_loss  += candidate_losses['depth_smooth']
            num_depth = N
        elif depth_pred is not None and depth_gt is not None:
            # 单候选回退
            if log_this_step:
                self.depth_logger.info(f"  [Depth] Single candidate supervision")
            candidate_losses = self.depth_loss(depth_pred, depth_gt)
            depth_decoder_loss += candidate_losses['depth_pred']
            depth_smooth_loss  += candidate_losses['depth_smooth']
            num_depth = 1

        if num_depth > 0:
            depth_decoder_loss = depth_decoder_loss / num_depth
            depth_smooth_loss  = depth_smooth_loss  / num_depth
        depth_total_loss = depth_decoder_loss + depth_smooth_loss
        
        # 深度重建损失（如果有深度预测和目标图像）
        # 移除深度重建损失
        depth_rec_loss = torch.tensor(0.0, device=device)
        # 保持 total 仅由 decoder + smooth 组成
        depth_total_loss = depth_total_loss
        
        # 注意力一致性损失 - 🔥 支持多退化输入的简化计算
        attn_cons_loss = torch.tensor(0.0, device=device)
        if attention_maps is not None:
            depth2rgb_attn, rgb2depth_attn = attention_maps
            if depth2rgb_attn is not None and rgb2depth_attn is not None:
                # 🔥 简化多退化注意力一致性计算：取batch和head维度的均值
                if depth2rgb_attn.dim() >= 4:  # [B, heads, N, N] 或更高维
                    # 对batch和head维度取平均，简化为单个注意力图的一致性计算
                    d2r_mean = depth2rgb_attn.mean(dim=(0, 1), keepdim=True)  # [1, 1, N, N]
                    r2d_mean = rgb2depth_attn.mean(dim=(0, 1), keepdim=True)  # [1, 1, N, N]
                    attn_cons_loss = self.attention_consistency_loss(d2r_mean, r2d_mean)
                else:
                    # 原始计算方式（向后兼容）
                    attn_cons_loss = self.attention_consistency_loss(depth2rgb_attn, rgb2depth_attn)
                
                if log_this_step:
                    self.attention_logger.info(f"  [Attention] consistency_loss={attn_cons_loss:.4f} (shape: d2r={depth2rgb_attn.shape}, r2d={rgb2depth_attn.shape})")
            else:
                if log_this_step:
                    self.warning_logger.warning(f"  [Attention] Missing attention maps for consistency loss: d2r={depth2rgb_attn is not None}, r2d={rgb2depth_attn is not None}")
        else:
            if log_this_step:
                self.warning_logger.warning("  [Attention] attention_maps is None, skipping consistency loss")

        # 特征一致性损失已删除，只保留输出层面的CMCL损失
        
        # 🔥 跨退化一致性损失 (CMCL)
        cmcl_loss = torch.tensor(0.0, device=device)
        cmcl_var_loss = torch.tensor(0.0, device=device)
        cmcl_rgb_loss = torch.tensor(0.0, device=device)
        cmcl_depth_loss = torch.tensor(0.0, device=device)
        
        if multi_enhanced is not None and multi_depth_pred is not None:
            cmcl_total_loss, cmcl_dict = self.cmcl_loss(multi_enhanced, multi_depth_pred)
            cmcl_loss = cmcl_total_loss
            cmcl_var_loss = cmcl_dict['cmcl_var']
            cmcl_rgb_loss = cmcl_dict['cmcl_rgb']
            cmcl_depth_loss = cmcl_dict['cmcl_depth']
            
            if log_this_step:
                self.metrics_logger.info(f"  [CMCL] total={cmcl_loss:.4f}, var={cmcl_var_loss:.4f}, rgb={cmcl_rgb_loss:.4f}, depth={cmcl_depth_loss:.4f}")

        # 使用不确定性加权或手动加权计算总损失
        if self.use_uncertainty_weighting:
            # 不确定性加权辅助函数
            def weight_loss(loss_value, log_var):
                """根据不确定性对损失进行加权
                loss = precision * loss_value + log_var/2
                """
                if not isinstance(loss_value, torch.Tensor):
                    loss_value = torch.tensor(loss_value, device=log_var.device)
                
                # 🔥 限制log_var范围，防止数值不稳定
                log_var_clamped = torch.clamp(log_var, min=self.log_var_min, max=self.log_var_max)
                
                # 如果log_var超出范围，记录警告
                if abs(log_var.item() - log_var_clamped.item()) > 1e-6:
                    logging.getLogger('warning').warning(f"log_var被裁剪: {log_var.item():.4f} -> {log_var_clamped.item():.4f}")
                
                precision = torch.exp(-log_var_clamped) + 1e-8
                weighted_loss = precision * loss_value + log_var_clamped / 2
                
                # 🔥 确保损失非负，避免零梯度
                return torch.clamp(weighted_loss, min=1e-8)
            
            # 图像损失组
            weighted_l1_loss   = weight_loss(l1_loss,   self.log_var_l1)
            weighted_ssim_loss = weight_loss(ssim_loss, self.log_var_ssim)
            weighted_perc_loss = weight_loss(perc_loss, self.log_var_perc)
            weighted_fft_loss  = weight_loss(fft_loss,  self.log_var_fft)
            weighted_grad_loss = weight_loss(grad_loss, self.log_var_grad)
            
            # 深度损失组

            weighted_depth_smooth_loss = weight_loss(depth_smooth_loss, self.log_var_depth_smooth)
            weighted_depth_decoder_loss = weight_loss(depth_decoder_loss, self.log_var_depth_decoder)  # 深度解码器损失
            # depth_rec removed
             
            # 其他损失
            weighted_attn_cons_loss = weight_loss(attn_cons_loss, self.log_var_attncons)
            # 特征一致性损失已删除

            weighted_cmcl_var_loss   = weight_loss(self.lambda_cmcl_var * cmcl_var_loss,   self.log_var_cmcl_var)
            weighted_cmcl_rgb_loss   = weight_loss(self.lambda_cmcl_rgb * cmcl_rgb_loss,   self.log_var_cmcl_rgb)
            weighted_cmcl_depth_loss = weight_loss(self.lambda_cmcl_depth * cmcl_depth_loss, self.log_var_cmcl_depth)
            
            # 计算总损失
            loss = (weighted_l1_loss + weighted_ssim_loss + weighted_perc_loss
                    + weighted_fft_loss + weighted_grad_loss
                    + weighted_depth_smooth_loss + weighted_depth_decoder_loss
                    + weighted_attn_cons_loss
                    + weighted_cmcl_var_loss + weighted_cmcl_rgb_loss + weighted_cmcl_depth_loss)
            
            if log_this_step:
                self.optimizer_logger.info(f"  [Uncertainty Weights]")
                self.optimizer_logger.info(f"    - Img: l1={torch.exp(-self.log_var_l1).item():.4f}, ssim={torch.exp(-self.log_var_ssim).item():.4f}, perc={torch.exp(-self.log_var_perc).item():.4f}, fft={torch.exp(-self.log_var_fft).item():.4f}, grad={torch.exp(-self.log_var_grad).item():.4f}")
                self.optimizer_logger.info(f"    - Dep: decoder={torch.exp(-self.log_var_depth_decoder).item():.4f}, smooth={torch.exp(-self.log_var_depth_smooth).item():.4f}")
                self.optimizer_logger.info(f"    - Oth: attncons={torch.exp(-self.log_var_attncons).item():.4f}")
                self.optimizer_logger.info(f"    - CMCL: var={torch.exp(-self.log_var_cmcl_var).item():.4f}, rgb={torch.exp(-self.log_var_cmcl_rgb).item():.4f}, depth={torch.exp(-self.log_var_cmcl_depth).item():.4f}")
                self.metrics_logger.info(f"  [Total Loss] Weighted Sum: {loss.item():.4f}")

            # 记录各个不确定性权重
            # 图像损失组 - 修复：记录的应该是权重值，不是exp(-log_var)
            self.losses['uncertainty_l1'] = torch.exp(-self.log_var_l1).item()
            self.losses['uncertainty_ssim'] = torch.exp(-self.log_var_ssim).item()
            self.losses['uncertainty_perc'] = torch.exp(-self.log_var_perc).item()
            self.losses['uncertainty_fft'] = torch.exp(-self.log_var_fft).item()
            self.losses['uncertainty_grad'] = torch.exp(-self.log_var_grad).item()
            
            # 深度损失组
            self.losses['uncertainty_depth_decoder'] = torch.exp(-self.log_var_depth_decoder).item()
            self.losses['uncertainty_depth_smooth'] = torch.exp(-self.log_var_depth_smooth).item()
            # self.losses['uncertainty_depth_rec'] removed

            # 其他损失的不确定性权重
            self.losses['uncertainty_attncons'] = torch.exp(-self.log_var_attncons).item()

            self.losses['uncertainty_cmcl_var'] = torch.exp(-self.log_var_cmcl_var).item()
            self.losses['uncertainty_cmcl_rgb'] = torch.exp(-self.log_var_cmcl_rgb).item()
            self.losses['uncertainty_cmcl_depth'] = torch.exp(-self.log_var_cmcl_depth).item()
            
            # 记录原始log_var值，便于调试
            self.losses['log_var_l1'] = self.log_var_l1.item()
            self.losses['log_var_ssim'] = self.log_var_ssim.item()
            self.losses['log_var_perc'] = self.log_var_perc.item()
            self.losses['log_var_fft'] = self.log_var_fft.item()
            self.losses['log_var_grad'] = self.log_var_grad.item()
            self.losses['log_var_depth_decoder'] = self.log_var_depth_decoder.item()
            self.losses['log_var_depth_smooth'] = self.log_var_depth_smooth.item()
            # self.losses['log_var_depth_rec'] removed
            self.losses['log_var_attncons'] = self.log_var_attncons.item()

            self.losses['log_var_cmcl_var'] = self.log_var_cmcl_var.item()
            self.losses['log_var_cmcl_rgb'] = self.log_var_cmcl_rgb.item()
            self.losses['log_var_cmcl_depth'] = self.log_var_cmcl_depth.item()
            
            # 🔥 添加调试信息，确保数值正确
            if log_this_step:
                self.optimizer_logger.info(f"  [Debug] Raw log_var values:")
                self.optimizer_logger.info(f"    - log_var_l1={self.log_var_l1.item():.4f}, weight={torch.exp(-self.log_var_l1).item():.4f}")
                self.optimizer_logger.info(f"    - log_var_attncons={self.log_var_attncons.item():.4f}, weight={torch.exp(-self.log_var_attncons).item():.4f}")
                self.optimizer_logger.info(f"    - log_var_depth_decoder={self.log_var_depth_decoder.item():.4f}, weight={torch.exp(-self.log_var_depth_decoder).item():.4f}")
                self.optimizer_logger.info(f"    - log_var_cmcl_var={self.log_var_cmcl_var.item():.4f}, weight={torch.exp(-self.log_var_cmcl_var).item():.4f}")
                self.optimizer_logger.info(f"    - log_var_cmcl_rgb={self.log_var_cmcl_rgb.item():.4f}, weight={torch.exp(-self.log_var_cmcl_rgb).item():.4f}")
                self.optimizer_logger.info(f"    - log_var_cmcl_depth={self.log_var_cmcl_depth.item():.4f}, weight={torch.exp(-self.log_var_cmcl_depth).item():.4f}")
        else:
            # 手动加权：按项加权（不再有组级lambda_img/lambda_depth）
            img_manual = (
                self.lambda_img_l1   * l1_loss +
                self.lambda_img_ssim * ssim_loss +
                self.lambda_img_perc * perc_loss +
                self.lambda_img_fft  * fft_loss +
                self.lambda_img_grad * grad_loss
            )
            depth_manual = (
                self.lambda_depth_decoder * depth_decoder_loss +
                self.lambda_depth_smooth  * depth_smooth_loss
            )

            loss = img_manual + depth_manual + self.lambda_cons * attn_cons_loss + self.lambda_cmcl * cmcl_loss
            
            if log_this_step:
                self.metrics_logger.info(f"  [Manual Weights] Total Loss: {loss.item():.4f}")
        
        # 物理一致性损失计算已删除
        
        # 记录总损失和组件损失
        self.losses['l1_loss'] = l1_loss.item()
        self.losses['ssim_loss'] = ssim_loss.item()
        self.losses['perc_loss'] = perc_loss.item()
        self.losses['fft_loss'] = fft_loss.item()
        self.losses['grad_loss'] = grad_loss.item()
        self.losses['img_total_loss'] = img_total_loss.item()
        
        # 重构后：不再单独记录multi_candidate_*，img/depth相关损失已按候选平均
         
 
        self.losses['depth_smooth_loss'] = depth_smooth_loss.item()
        self.losses['depth_decoder_loss'] = depth_decoder_loss.item()  # DepthDecoder 深度预测损失
        # self.losses['depth_rec_loss'] removed
        self.losses['depth_total_loss'] = depth_total_loss.item()
        
        self.losses['attn_cons_loss'] = attn_cons_loss.item()

        # 🔥 CMCL损失记录
        self.losses['cmcl_loss'] = cmcl_loss.item()
        self.losses['cmcl_var_loss'] = cmcl_var_loss.item()
        self.losses['cmcl_rgb_loss'] = cmcl_rgb_loss.item()
        self.losses['cmcl_depth_loss'] = cmcl_depth_loss.item()
        
        self.losses['total_loss'] = loss.item()
        
        # 🔥 验证所有损失值都是有效的数值（log_var可以为负数）
        for loss_name, loss_value in self.losses.items():
            # log_var参数可以为负数，所以不检查它们的符号
            is_log_var = loss_name.startswith('log_var_')
            
            if not isinstance(loss_value, (int, float)) or not torch.isfinite(torch.tensor(loss_value)):
                self.warning_logger.warning(f"Invalid loss value detected: {loss_name}={loss_value}")
                # 将无效值设为0（对于log_var，设为合理的默认值）
                if is_log_var:
                    self.losses[loss_name] = -1.0  # log_var的合理默认值
                else:
                    self.losses[loss_name] = 0.0
            elif not is_log_var and loss_value < 0:
                # 只对非log_var的损失检查负值
                self.warning_logger.warning(f"Negative loss value detected: {loss_name}={loss_value}")
                self.losses[loss_name] = 0.0
        
        if log_this_step:
            self.metrics_logger.info(f"--- [End Loss Calculation Step {self._forward_count}] ---")
            
        # 🔥 约束不确定性权重参数，防止数值不稳定
        if self.use_uncertainty_weighting:
            with torch.no_grad():
                self.log_var_l1.data.clamp_(self.log_var_min, self.log_var_max)
                self.log_var_ssim.data.clamp_(self.log_var_min, self.log_var_max)
                self.log_var_perc.data.clamp_(self.log_var_min, self.log_var_max)
                self.log_var_fft.data.clamp_(self.log_var_min, self.log_var_max)
                self.log_var_grad.data.clamp_(self.log_var_min, self.log_var_max)
                self.log_var_depth_decoder.data.clamp_(self.log_var_min, self.log_var_max)
                self.log_var_depth_smooth.data.clamp_(self.log_var_min, self.log_var_max)
                # self.log_var_depth_rec removed
                self.log_var_attncons.data.clamp_(self.log_var_min, self.log_var_max)

                # CMCL 三项参数约束
                self.log_var_cmcl_var.data.clamp_(self.log_var_min, self.log_var_max)
                self.log_var_cmcl_rgb.data.clamp_(self.log_var_min, self.log_var_max)
                self.log_var_cmcl_depth.data.clamp_(self.log_var_min, self.log_var_max)
        
        return loss
    

class CrossDegradationConsistencyLoss(nn.Module):
    """
    🔥 跨退化一致性损失 (Cross Multi-degradation Consistency Loss, CMCL)
    
    包含三个组件：
    1. L_var: 基于深度置信度的像素方差约束
    2. L_rgb: 像素级L1损失对齐不同退化下的RGB增强结果  
    3. L_depth: L2损失约束多退化下的深度预测结果一致性
    
    数学公式：
    L_var = (1/BHW) * Σ_b Σ_xy w_d^(b)(x,y) * Var({E^(i)_b(x,y)})
    w_d^(b)(x,y) = exp(-k * D̄_b(x,y))
    D̄_b(x,y) = (1/N) * Σ_i D^(i)_b(x,y)
    
    L_rgb = Σ_{i<j} ||E^(i) - E^(j)||_1
    L_depth = Σ_{i<j} ||D^(i) - D^(j)||_2^2
    
    L_CMCL = λ_var * L_var + λ_rgb * L_rgb + λ_depth * L_depth
    """
    
    def __init__(self, 
                 lambda_var=1.0,
                 lambda_rgb=1.0, 
                 lambda_depth=1.0,
                 k_decay=1.0):
        """
        Args:
            lambda_var: 像素方差损失权重
            lambda_rgb: RGB一致性损失权重  
            lambda_depth: 深度一致性损失权重
            k_decay: 深度置信权重衰减系数
        """
        super().__init__()
        self.lambda_var = lambda_var
        self.lambda_rgb = lambda_rgb
        self.lambda_depth = lambda_depth
        self.k_decay = k_decay
        
        # 日志记录器
        self.logger = logging.getLogger('cmcl_loss')
        
    def compute_variance_loss(self, multi_enhanced, multi_depth_pred):
        """
        计算基于深度置信度的像素方差约束
        
        L_var = (1/BHW) * Σ_b Σ_xy w_d^(b)(x,y) * Var({E^(i)_b(x,y)})
        w_d^(b)(x,y) = exp(-k * D̄_b(x,y))
        D̄_b(x,y) = (1/N) * Σ_i D^(i)_b(x,y)
        
        Args:
            multi_enhanced: [B, N, 3, H, W] - 多退化增强结果
            multi_depth_pred: [B, N, 1, H, W] - 多退化深度预测
        Returns:
            variance_loss: 标量
        """
        if multi_enhanced is None or multi_depth_pred is None:
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
            
        B, N, C, H, W = multi_enhanced.shape
        if N < 2:
            return torch.tensor(0.0, device=multi_enhanced.device)
        
        # 计算平均深度: D̄_b(x,y) = (1/N) * Σ_i D^(i)_b(x,y)
        mean_depth = torch.mean(multi_depth_pred, dim=1)  # [B, 1, H, W]
        
        # 计算深度置信权重: w_d^(b)(x,y) = exp(-k * D̄_b(x,y))
        depth_confidence_weights = torch.exp(-self.k_decay * mean_depth)  # [B, 1, H, W]
        
        # 计算每个像素位置的方差: Var({E^(i)_b(x,y)})
        # 首先计算均值
        mean_enhanced = torch.mean(multi_enhanced, dim=1)  # [B, 3, H, W]
        
        # 计算方差：Var = E[(X - μ)²]
        variance = torch.mean((multi_enhanced - mean_enhanced.unsqueeze(1))**2, dim=1)  # [B, 3, H, W]
        
        # 扩展深度置信权重到3通道
        depth_weights_3ch = depth_confidence_weights.expand(-1, C, -1, -1)  # [B, 3, H, W]
        
        # 加权方差损失：w_d * Var
        weighted_variance = depth_weights_3ch * variance  # [B, 3, H, W]
        
        # 平均到所有批次、高度、宽度：(1/BHW) * Σ
        variance_loss = torch.mean(weighted_variance)
        
        return variance_loss
    
    def compute_rgb_consistency_loss(self, multi_enhanced):
        """
        计算像素级L1损失对齐不同退化下的RGB增强结果
        
        🔥 优化版：使用方差方法替代配对计算，从O(N²)优化为O(N)
        原方法: L_rgb = Σ_{i<j} ||E^(i) - E^(j)||_1  [105次计算]
        方差方法: L_rgb = Σ_i ||E^(i) - mean(E)||_1  [15次计算，~7倍加速]
        
        Args:
            multi_enhanced: [B, N, 3, H, W]
        Returns:
            rgb_loss: 标量
        """
        if multi_enhanced is None:
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
            
        B, N, C, H, W = multi_enhanced.shape
        if N < 2:
            return torch.tensor(0.0, device=multi_enhanced.device)
            
        # 🚀 方差方法：计算所有候选的均值
        mean_enhanced = torch.mean(multi_enhanced, dim=1, keepdim=True)  # [B, 1, C, H, W]
        
        # 计算每个候选与均值的L1偏差
        deviations = multi_enhanced - mean_enhanced  # [B, N, C, H, W]
        
        # L1损失：平均绝对偏差
        rgb_loss = torch.mean(torch.abs(deviations))
        
        return rgb_loss
    
    def compute_depth_consistency_loss(self, multi_depth_pred):
        """
        计算L2损失约束多退化下的深度预测结果一致性
        
        🔥 优化版：使用方差方法替代配对计算，从O(N²)优化为O(N)
        原方法: L_depth = Σ_{i<j} ||D^(i) - D^(j)||_2^2  [105次计算]
        方差方法: L_depth = Σ_i ||D^(i) - mean(D)||_2^2  [15次计算，~7倍加速]
        
        数学等价性：对于L2损失，方差方法与配对平均在数学上完全等价
        
        Args:
            multi_depth_pred: [B, N, 1, H, W]
        Returns:
            depth_loss: 标量
        """
        if multi_depth_pred is None:
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
            
        B, N, C, H, W = multi_depth_pred.shape
        if N < 2:
            return torch.tensor(0.0, device=multi_depth_pred.device)
            
        # 🚀 方差方法：计算所有候选的均值
        mean_depth = torch.mean(multi_depth_pred, dim=1, keepdim=True)  # [B, 1, 1, H, W]
        
        # 计算每个候选与均值的L2偏差
        deviations = multi_depth_pred - mean_depth  # [B, N, 1, H, W]
        
        # L2损失：均方偏差（方差）
        depth_loss = torch.mean(deviations ** 2)
        
        return depth_loss
    
    def forward(self, multi_enhanced=None, multi_depth_pred=None):
        """
        计算完整的跨退化一致性损失
        
        L_CMCL = λ_var * L_var + λ_rgb * L_rgb + λ_depth * L_depth
        
        Args:
            multi_enhanced: [B, N, 3, H, W] - 多退化增强结果
            multi_depth_pred: [B, N, 1, H, W] - 多退化深度预测
        Returns:
            total_loss: 标量
            loss_dict: 各项损失的字典
        """
        # 计算三个损失组件
        loss_var = self.compute_variance_loss(multi_enhanced, multi_depth_pred)
        loss_rgb = self.compute_rgb_consistency_loss(multi_enhanced)  
        loss_depth = self.compute_depth_consistency_loss(multi_depth_pred)
        
        # 线性组合
        total_loss = (self.lambda_var * loss_var + 
                     self.lambda_rgb * loss_rgb + 
                     self.lambda_depth * loss_depth)
        
        # 返回总损失和各项损失
        loss_dict = {
            'cmcl_var': loss_var,
            'cmcl_rgb': loss_rgb, 
            'cmcl_depth': loss_depth,
            'cmcl_total': total_loss
        }
        
        return total_loss, loss_dict


# 实现SSIM函数，用于DepthReconstructionLoss
def ssim(img1, img2, window_size=11, size_average=True):
    """计算结构相似度 (SSIM) 指标"""
    # 使用高斯窗口
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # 确保两个输入张量的数据类型一致
    if img1.dtype != img2.dtype:
        # 如果类型不一致，将两者都转换为float32
        img1 = img1.float()
        img2 = img2.float()
    
    # 确保两个输入的通道数一致
    if img1.size(1) != img2.size(1):
        # 如果一个是深度图(1通道)，一个是RGB(3通道)，则将深度图复制到3通道
        if img1.size(1) == 1:
            img1 = img1.expand(-1, img2.size(1), -1, -1)
        elif img2.size(1) == 1:
            img2 = img2.expand(-1, img1.size(1), -1, -1)
    
    # 获取通道数用于分组卷积
    channels = img1.size(1)
    
    # 创建高斯窗口，确保与输入张量相同的数据类型
    window = torch.ones(window_size, window_size, device=img1.device, dtype=img1.dtype) / (window_size * window_size)
    window = window.view(1, 1, window_size, window_size).expand(channels, 1, window_size, window_size)
    
    # 均值平滑
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # 方差平滑
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channels) - mu1_mu2
    
    # SSIM 计算
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# ReprojectionConsistencyLoss已删除，只保留输出层面的CMCL一致性约束
