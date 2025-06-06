import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import sys

# 确保所有损失类都被导出
__all__ = [
    'TotalLoss',
    'ImageLoss',
    'DepthLoss',
    'SSIMLoss',
    'PerceptualLoss',
    'FFTLoss',
    'GradientLoss',
    'EdgeAwareDepthLoss',
    'DepthSmoothLoss',
    'DepthEdgeColorLoss',
    'TeacherStudentLoss',
    'CrossAttentionConsistencyLoss',
    'DepthReconstructionLoss'
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
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2
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
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        return torch.clamp((1 - ssim_map) / 2, 0, 1).mean()

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
        # to complex by view as real
        f1 = torch.fft.rfft2(img1, norm='ortho')
        f2 = torch.fft.rfft2(img2, norm='ortho')
        return self.l1(torch.abs(f1), torch.abs(f2))

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
    def __init__(self, edge_weight=0.5, min_depth=5000.0, max_depth=65000.0):
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
                
                print(f"[DEPTH-LOSS] 统计: 范围[{avg_min:.4f}, {avg_max:.4f}], 均值{avg_mean:.4f}, 标准差{avg_std:.4f}")
        
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
        elif target_max < 5000.0 or (target_max <= 1.0 and target_min >= 0.0):
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
            gt_edges = self._compute_normalized_edges(target)
        else:
            # 两者都未归一化，创建有效深度掩码并归一化两者
            # 如果是原始深度，创建有效深度掩码并进行归一化
            valid_mask = (target > self.min_depth) & (target < self.max_depth)
            valid_ratio = valid_mask.float().mean().item()
            
            # 如果没有有效像素，返回零损失
            if not valid_mask.any():
                if self.debug_mode:
                    print(f"[DEPTH-LOSS] 警告: 没有有效像素在 [{self.min_depth}, {self.max_depth}] 范围内")
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
        # 直接计算水平和垂直梯度
        grad_x = self.sobel_x(depth)
        grad_y = self.sobel_y(depth)
        
        # 梯度幅度
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.eps)
        return grad_mag

class DepthEdgeColorLoss(nn.Module):
    """深度边缘颜色损失
    
    结合深度加权的颜色差异和边缘保持损失，使颜色校正在深度边界处更加精确
    """
    def __init__(self, w_edge=1.0, w_depth=0.5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.w_edge = w_edge
        self.w_depth = w_depth
        
        # Sobel算子用于检测边缘
        self._sobel_op = None # 改名为私有属性
        
    def rgb2lab(self, rgb):
        """简化的RGB到Lab转换,作为颜色差异度量"""
        # 这里使用简化的公式,返回便于计算颜色差异的特征
        r, g, b = rgb.split(1, dim=1)
        L = 0.299 * r + 0.587 * g + 0.114 * b
        a = 0.5 * (r - L)
        b = 0.5 * (b - L)
        return torch.cat([L, a, b], dim=1)
        
    def _get_sobel_op(self):
        """获取Sobel算子（懒加载）"""
        if self._sobel_op is None:
            # 懒加载初始化Sobel算子
            import kornia.filters as K
            self._sobel_op = K.Sobel()
            
            # 此处不再使用next(self.parameters())因为可能没有参数
            # 而是在forward时根据输入设备设置
        
        return self._sobel_op
    
    def forward(self, pred_rgb, tgt_rgb, depth_conf, mask=None):
        """
        Args:
            pred_rgb: 预测RGB图像 [B,3,H,W]
            tgt_rgb: 目标RGB图像 [B,3,H,W]
            depth_conf: 深度置信度 [B,1,H,W]
            mask: 可选掩码,防止极端曝光区域参与计算 [B,1,H,W]
        """
        # 确保Sobel算子在正确的设备上
        device = pred_rgb.device
        if self._sobel_op is not None and next(self._sobel_op.parameters(), None) is not None:
            if next(self._sobel_op.parameters()).device != device:
                self._sobel_op = self._sobel_op.to(device)
        else:
            import kornia.filters as K
            self._sobel_op = K.Sobel().to(device)
                
        # 确保输入形状匹配
        if depth_conf.shape[-2:] != pred_rgb.shape[-2:]:
            depth_conf = F.interpolate(depth_conf, size=pred_rgb.shape[-2:], 
                                      mode='bilinear', align_corners=False)
        
        # 计算颜色差异(基于L1)
        loss_color = torch.abs(pred_rgb - tgt_rgb)
        
        # 计算深度权重: 对中等深度区域(0.5左右)赋予最高权重
        depth_w = torch.exp(-((depth_conf - 0.5) ** 2) / (0.12 ** 2))
        
        # 深度加权颜色差异
        weighted_color_loss = (depth_w * loss_color).mean()
        
        # 计算边缘保持损失
        sobel_op = self._get_sobel_op()
        grad_p = sobel_op(pred_rgb)
        grad_t = sobel_op(tgt_rgb) 
        loss_edge = torch.abs(grad_p - grad_t).mean()
        
        # 组合损失
        total_loss = weighted_color_loss + self.w_edge * loss_edge
        
        # 若有掩码,则应用掩码
        if mask is not None:
            if mask.shape[-2:] != pred_rgb.shape[-2:]:
                mask = F.interpolate(mask, size=pred_rgb.shape[-2:],
                                    mode='nearest')
            safe = mask.float()
            total_loss = total_loss * safe.mean()
        
        # 返回字典而不是单一值
        losses_dict = {
            'color': weighted_color_loss,
            'edge': loss_edge,
            'total': total_loss
        }
        
        return losses_dict

class CrossAttentionConsistencyLoss(nn.Module):
    """
    交叉注意力对偶一致性损失
    -----------------------
    计算 Depth→RGB 和 RGB→Depth 两个方向的注意力图之间的一致性。
    理论上，两者应该满足对称性关系，即 A_{d2r} ≈ A_{r2d}^T。
    这有助于两个方向的注意力机制学习互补且一致的表示。
    
    输入:
      - attn_d2r : Tensor[B, heads, N, N] - Depth→RGB 注意力图
      - attn_r2d : Tensor[B, heads, N, N] - RGB→Depth 注意力图
    输出:
      - loss : 对偶一致性损失值
    """
    def __init__(self, reduction='mean', norm_type=1):
        super().__init__()
        self.reduction = reduction
        self.norm_type = norm_type
        
    def forward(self, attn_d2r, attn_r2d):
        """计算两个方向注意力图之间的对偶一致性损失"""
        if attn_d2r is None or attn_r2d is None:
            # 如果任一注意力图为空，返回零损失
            device = attn_d2r.device if attn_d2r is not None else attn_r2d.device
            print(f"[注意力一致性损失] 注意力图缺失: d2r={attn_d2r is not None}, r2d={attn_r2d is not None}")
            return {'total': torch.tensor(0.0, device=device)}
        
        # 获取批次大小和头数
        batch_size, num_heads = attn_d2r.shape[:2]
        
        # 计算 attn_r2d 的转置（在最后两个维度上）
        attn_r2d_t = attn_r2d.transpose(-1, -2)
        
        # 计算注意力图之间的差异，使用指定的范数
        if self.norm_type == 1:
            # L1 范数（曼哈顿距离）
            diff = torch.abs(attn_d2r - attn_r2d_t)
        elif self.norm_type == 2:
            # L2 范数（欧几里得距离）
            diff = (attn_d2r - attn_r2d_t) ** 2
        else:
            # 默认使用 L1 范数
            diff = torch.abs(attn_d2r - attn_r2d_t)
        
        # 根据 reduction 方式计算最终损失
        if self.reduction == 'mean':
            loss = diff.mean()
        elif self.reduction == 'sum':
            loss = diff.sum()
        elif self.reduction == 'none':
            loss = diff
        else:
            # 默认使用 mean
            loss = diff.mean()
            
        # 返回字典
        return {'total': loss}

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
        except ImportError:
            self.has_perceptual = False
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
            # LPIPS期望输入在[-1,1]范围内
            perceptual_loss = self.perceptual_loss(pred, target).mean()
            total_loss += self.lambda_perceptual * perceptual_loss
            self.last_losses['perceptual_loss'] = perceptual_loss.item()
        
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
    def __init__(self, lambda_depth=1.0, lambda_smooth=0.01):
        super().__init__()
        self.lambda_depth = lambda_depth
        self.lambda_smooth = lambda_smooth
        
        # 实例化深度损失函数
        self.depth_loss = EdgeAwareDepthLoss()
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
        - 深度边缘颜色损失: lambda_decl * DepthEdgeColor(pred, target, depth_conf)
        
    3. 注意力一致性损失: lambda_cons * ConsistencyLoss(depth2rgb_attn, rgb2depth_attn)
    """
    def __init__(self,
                 lambda_img=10.0, 
                 lambda_ssim=1.0, 
                 lambda_perc=1.0,
                 lambda_fft=0.1,
                 lambda_grad=0.1,
                 lambda_depth=0.1,
                 lambda_smooth=0.01,
                 lambda_decl=0.1,
                 lambda_cons=0.05,
                 lambda_phy_A=0.1,
                 lambda_phy_D=0.1,
                 beta_c=None,
                 B_c=None,
                 use_uncertainty_weighting=True):
        super().__init__()
        
        # 权重参数
        self.lambda_img = lambda_img
        self.lambda_ssim = lambda_ssim
        self.lambda_perc = lambda_perc
        self.lambda_fft = lambda_fft
        self.lambda_grad = lambda_grad
        self.lambda_depth = lambda_depth
        self.lambda_smooth = lambda_smooth
        self.lambda_decl = lambda_decl
        self.lambda_cons = lambda_cons
        self.lambda_phy_A = lambda_phy_A  # 物理一致性损失A权重
        self.lambda_phy_D = lambda_phy_D  # 物理一致性损失D权重
        
        # 物理模型参数（从模型传入）
        self.beta_c = beta_c  # Beer-Lambert 衰减系数
        self.B_c = B_c        # 全局背景光
        
        # 组件损失
        self.image_loss = ImageLoss(
            lambda_l1=1.0,
            lambda_ssim=lambda_ssim,
            lambda_perceptual=lambda_perc,
            lambda_fft=lambda_fft,
            lambda_grad=lambda_grad
        )
        
        self.depth_loss = DepthLoss(
            lambda_depth=1.0,
            lambda_smooth=lambda_smooth
        )
        
        self.depth_rec_loss = DepthReconstructionLoss()
        
        self.depth_edge_color_loss = DepthEdgeColorLoss()
        
        self.attention_consistency_loss = CrossAttentionConsistencyLoss()
        
        # 不确定性加权
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        if use_uncertainty_weighting:
            # log(σ²) 参数初始化为0，相当于初始不确定度为1
            self.log_var_l1 = nn.Parameter(torch.zeros(1))
            self.log_var_ssim = nn.Parameter(torch.zeros(1))
            self.log_var_perc = nn.Parameter(torch.zeros(1))
            self.log_var_fft = nn.Parameter(torch.zeros(1))
            self.log_var_grad = nn.Parameter(torch.zeros(1))
            
            self.log_var_depth_pred = nn.Parameter(torch.zeros(1))
            self.log_var_depth_smooth = nn.Parameter(torch.zeros(1))
            self.log_var_depth_rec = nn.Parameter(torch.zeros(1))
            
            self.log_var_cons = nn.Parameter(torch.zeros(1))
            self.log_var_phy_A = nn.Parameter(torch.zeros(1))
            self.log_var_phy_D = nn.Parameter(torch.zeros(1))
        
        # 记录损失值
        self.losses = {}
    
    def get_latest_losses(self):
        return self.losses
    
    def forward(self, pred, target, depth_gate=None, depth_gt=None, 
                student_feats=None, attention_maps=None, depth_pred=None, depth_conf_map=None,
                J_D=None, I_A=None, raw=None):
        """
        计算总损失
        
        Args:
            pred: 预测的RGB图像 [B,3,H,W]
            target: 目标RGB图像 [B,3,H,W]
            depth_gate: 预测的深度门控图 [B,1,H,W] or None
            depth_gt: 深度GT [B,1,H,W] or None
            student_feats: 学生特征列表 or None
            attention_maps: 注意力图元组 (depth2rgb, rgb2depth) or None
            depth_pred: DepthDecoder 输出的连续深度 [B,1,H,W] or None
            depth_conf_map: 深度置信度图 [B,1,H,W] or None (用于DECL损失)
            J_D: D式模糊后的去模糊图 [B,3,H,W]
            I_A: A式Beer–Lambert合成输出 [B,3,H,W]
            raw: 原始退化水下图 [B,3,H,W]
        
        Returns:
            total_loss: 标量总损失
        """
        device = pred.device
        
        # 图像损失组：L1 + SSIM + 感知 + FFT + 梯度
        img_losses = self.image_loss(pred, target)
        l1_loss = img_losses['l1']
        ssim_loss = img_losses['ssim']
        perc_loss = img_losses['perceptual']
        fft_loss = img_losses['fft']
        grad_loss = img_losses['gradient']
        
        img_total_loss = (l1_loss + ssim_loss + perc_loss + fft_loss + grad_loss)
        
        # 深度损失组：深度预测 + 深度平滑
        depth_pred_loss = torch.tensor(0.0, device=device)
        depth_smooth_loss = torch.tensor(0.0, device=device)
        depth_total_loss = torch.tensor(0.0, device=device)
        
        # 连续深度预测损失（DepthDecoder 输出）
        depth_decoder_loss = torch.tensor(0.0, device=device)
        
        # 使用 DepthDecoder 输出的深度预测计算深度回归损失
        if depth_pred is not None and depth_gt is not None:
            depth_losses = self.depth_loss(depth_pred, depth_gt)
            depth_decoder_loss = depth_losses['depth_pred']
            depth_total_loss += depth_decoder_loss
            
            # 记录额外的深度损失组件
            if 'l1' in depth_losses:
                self.losses['depth_decoder_l1'] = depth_losses['l1'].item()
            if 'edge_weighted' in depth_losses:
                self.losses['depth_decoder_edge'] = depth_losses['edge_weighted'].item()
        
        # 深度边缘颜色损失
        decl_loss = torch.tensor(0.0, device=device)
        decl_color_loss = torch.tensor(0.0, device=device)
        decl_edge_loss = torch.tensor(0.0, device=device)
        if pred is not None and target is not None and depth_gate is not None:
            # 使用depth_gate替代depth_conf_map
            decl_losses = self.depth_edge_color_loss(pred, target, depth_gate)
            decl_loss = decl_losses['total']
            decl_color_loss = decl_losses['color']
            decl_edge_loss = decl_losses['edge']
        
        # 深度重建损失（如果有深度预测和目标图像）
        depth_rec_loss = torch.tensor(0.0, device=device)
        if depth_pred is not None and target is not None:
            # 确保深度预测和RGB目标的尺寸匹配
            if depth_pred.shape[-2:] != target.shape[-2:]:
                depth_pred_resized = F.interpolate(depth_pred, size=target.shape[-2:], 
                                                  mode='bilinear', align_corners=False)
            else:
                depth_pred_resized = depth_pred
            
            # 计算深度重建损失
            depth_rec_losses = self.depth_rec_loss(depth_pred_resized, target)
            depth_rec_loss = depth_rec_losses['total']
            # 将深度重建损失添加到总深度损失中
            depth_total_loss += depth_rec_loss
            
            # 记录深度重建损失的各个组件
            self.losses['depth_rec_l1'] = depth_rec_losses['l1'].item()
            self.losses['depth_rec_ssim'] = depth_rec_losses['ssim'].item()
        
        # 注意力一致性损失
        attn_cons_loss = torch.tensor(0.0, device=device)
        if attention_maps is not None:
            depth2rgb_attn, rgb2depth_attn = attention_maps
            if depth2rgb_attn is not None and rgb2depth_attn is not None:
                attn_cons_losses = self.attention_consistency_loss(depth2rgb_attn, rgb2depth_attn)
                attn_cons_loss = attn_cons_losses['total']
        
        # 使用不确定性加权或手动加权计算总损失
        if self.use_uncertainty_weighting:
            # 不确定性加权辅助函数
            def weight_loss(loss_value, log_var):
                """根据不确定性对损失进行加权
                loss = loss / (2*exp(log_var)) + log_var/2
                """
                if not isinstance(loss_value, torch.Tensor):
                    loss_value = torch.tensor(loss_value, device=log_var.device)
                # 加上小的epsilon防止数值问题
                precision = torch.exp(-log_var) + 1e-8
                return precision * loss_value + log_var / 2
            
            # 图像损失组
            weighted_l1_loss   = weight_loss(l1_loss,   self.log_var_l1)
            weighted_ssim_loss = weight_loss(ssim_loss, self.log_var_ssim)
            weighted_perc_loss = weight_loss(perc_loss, self.log_var_perc)
            weighted_fft_loss  = weight_loss(fft_loss,  self.log_var_fft)
            weighted_grad_loss = weight_loss(grad_loss, self.log_var_grad)
            
            # 深度损失组
            weighted_depth_pred_loss = weight_loss(depth_pred_loss, self.log_var_depth_pred)
            weighted_depth_smooth_loss = weight_loss(depth_smooth_loss, self.log_var_depth_smooth)
            weighted_depth_decoder_loss = weight_loss(depth_decoder_loss, self.log_var_depth_pred)  # 使用相同的权重
            weighted_depth_rec_loss = weight_loss(depth_rec_loss, self.log_var_depth_rec)
            
            # 其他损失
            weighted_attn_cons_loss = weight_loss(attn_cons_loss, self.log_var_cons)
            
            # 物理一致性损失
            L_phy_A = torch.tensor(0.0, device=device)
            L_phy_D = torch.tensor(0.0, device=device)
            weighted_phy_A_loss = torch.tensor(0.0, device=device)
            weighted_phy_D_loss = torch.tensor(0.0, device=device)
            
            if I_A is not None and J_D is not None and raw is not None and depth_pred is not None and self.beta_c is not None and self.B_c is not None:
                # 4.1 计算 t_c(exp) 和 B_c，这是用来重建 ˆI_D 时的 Beer–Lambert 衰减
                depth_norm = torch.clamp(depth_pred, 0.0, 1.0)        # [B,1,H,W]
                depth_3ch = depth_norm.repeat(1, 3, 1, 1)            # [B,3,H,W]
                
                # 4.2 L_phy_A: 直接让 I_A 与 raw 对齐
                L1_A = F.l1_loss(I_A, raw)
                Lssim_A = 1.0 - ssim(I_A, raw)  # SSIM越大越好，所以这里用1-SSIM
                L_phy_A = L1_A + Lssim_A
                
                # 4.3 L_phy_D: 先让 J_D 经 Beer–Lambert，再跟 raw 对齐
                t = torch.exp(- self.beta_c * depth_3ch)              # [B,3,H,W]
                I_hat_D = J_D * t + self.B_c * (1.0 - t)              # [B,3,H,W]
                I_hat_D = torch.clamp(I_hat_D, 0.0, 1.0)
                L1_D = F.l1_loss(I_hat_D, raw)
                Lssim_D = 1.0 - ssim(I_hat_D, raw)
                L_phy_D = L1_D + Lssim_D
                
                # 记录物理一致性损失
                self.losses['phy_A_L1'] = L1_A.item()
                self.losses['phy_A_SSIM'] = Lssim_A.item()
                self.losses['L_phy_A'] = L_phy_A.item()
                self.losses['phy_D_L1'] = L1_D.item()
                self.losses['phy_D_SSIM'] = Lssim_D.item()
                self.losses['L_phy_D'] = L_phy_D.item()
                
                # 4.4 处理物理一致性损失
                if self.use_uncertainty_weighting:
                    weighted_phy_A_loss = weight_loss(L_phy_A, self.log_var_phy_A)
                    weighted_phy_D_loss = weight_loss(L_phy_D, self.log_var_phy_D)
                else:
                    weighted_phy_A_loss = self.lambda_phy_A * L_phy_A
                    weighted_phy_D_loss = self.lambda_phy_D * L_phy_D
            
            # 计算总损失
            if self.use_uncertainty_weighting:
                loss = (weighted_l1_loss + weighted_ssim_loss + weighted_perc_loss
                        + weighted_fft_loss + weighted_grad_loss
                        + weighted_depth_pred_loss + weighted_depth_smooth_loss + weighted_depth_decoder_loss + weighted_depth_rec_loss
                        + 0.1 * decl_loss + weighted_attn_cons_loss
                        + weighted_phy_A_loss + weighted_phy_D_loss)
                
                # 记录各个不确定性权重
                # 图像损失组
                self.losses['uncertainty_l1'] = torch.exp(-self.log_var_l1).item()
                self.losses['uncertainty_ssim'] = torch.exp(-self.log_var_ssim).item()
                self.losses['uncertainty_perc'] = torch.exp(-self.log_var_perc).item()
                self.losses['uncertainty_fft'] = torch.exp(-self.log_var_fft).item()
                self.losses['uncertainty_grad'] = torch.exp(-self.log_var_grad).item()
                
                # 深度损失组
                self.losses['uncertainty_depth_pred'] = torch.exp(-self.log_var_depth_pred).item()
                self.losses['uncertainty_depth_smooth'] = torch.exp(-self.log_var_depth_smooth).item()
                self.losses['uncertainty_depth_rec'] = torch.exp(-self.log_var_depth_rec).item()
                
                # 其他损失的不确定性权重
                self.losses['uncertainty_cons'] = torch.exp(-self.log_var_cons).item()
                
                # 物理一致性损失的不确定性权重
                if I_A is not None and J_D is not None:
                    self.losses['uncertainty_phy_A'] = torch.exp(-self.log_var_phy_A).item()
                    self.losses['uncertainty_phy_D'] = torch.exp(-self.log_var_phy_D).item()
                
                # 记录原始log_var值，便于调试
                self.losses['log_var_l1'] = self.log_var_l1.item()
                self.losses['log_var_ssim'] = self.log_var_ssim.item()
                self.losses['log_var_perc'] = self.log_var_perc.item()
                self.losses['log_var_fft'] = self.log_var_fft.item()
                self.losses['log_var_grad'] = self.log_var_grad.item()
                self.losses['log_var_depth_pred'] = self.log_var_depth_pred.item()
                self.losses['log_var_depth_smooth'] = self.log_var_depth_smooth.item()
                self.losses['log_var_depth_rec'] = self.log_var_depth_rec.item()
                self.losses['log_var_phy_A'] = self.log_var_phy_A.item()
                self.losses['log_var_phy_D'] = self.log_var_phy_D.item()
                self.losses['log_var_cons'] = self.log_var_cons.item()
            else:
                # 手动加权
                loss = (self.lambda_img * img_total_loss + 
                        self.lambda_depth * depth_total_loss + 
                        self.lambda_decl * decl_loss)
                
                # 加上注意力一致性损失和深度重建损失
                loss += (self.lambda_cons * attn_cons_loss +
                        self.lambda_depth * depth_rec_loss + 
                        weighted_phy_A_loss + weighted_phy_D_loss)
        else:
            # 手动加权
            loss = (self.lambda_img * img_total_loss + 
                    self.lambda_depth * depth_total_loss + 
                    self.lambda_decl * decl_loss)
            
            # 加上注意力一致性损失和深度重建损失
            loss += (self.lambda_cons * attn_cons_loss +
                    self.lambda_depth * depth_rec_loss)
        
        # 记录总损失和组件损失
        self.losses['l1_loss'] = l1_loss.item()
        self.losses['ssim_loss'] = ssim_loss.item()
        self.losses['perc_loss'] = perc_loss.item()
        self.losses['fft_loss'] = fft_loss.item()
        self.losses['grad_loss'] = grad_loss.item()
        self.losses['img_total_loss'] = img_total_loss.item()
        
        self.losses['depth_pred_loss'] = depth_pred_loss.item()
        self.losses['depth_smooth_loss'] = depth_smooth_loss.item()
        self.losses['depth_decoder_loss'] = depth_decoder_loss.item()  # 添加对 depth_pred 的监督损失
        self.losses['depth_rec_loss'] = depth_rec_loss.item()
        self.losses['depth_total_loss'] = depth_total_loss.item()
        
        self.losses['decl_loss'] = decl_loss.item()
        self.losses['decl_color_loss'] = decl_color_loss.item()  # 记录边缘颜色损失的颜色组件
        self.losses['decl_edge_loss'] = decl_edge_loss.item()    # 记录边缘颜色损失的边缘组件
        self.losses['attn_cons_loss'] = attn_cons_loss.item()
        
        self.losses['total_loss'] = loss.item()
        return loss
    
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
