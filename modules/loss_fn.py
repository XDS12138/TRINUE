import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

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
        window = gaussian_window(self.window_size, self.sigma, C).to(img1.device)
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

class TotalLoss(nn.Module):
    def __init__(self,
                 lambda_img=1.0,
                 lambda_ssim=0.5,
                 lambda_perc=0.1,
                 lambda_fft=0.05,
                 lambda_grad=0.1,
                 lambda_depth=1.0,
                 lambda_smooth=0.01,
                 teacher_student_loss=None):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.perc = PerceptualLoss()
        self.fft = FFTLoss()
        self.grad_loss = GradientLoss()
        self.depth_loss = nn.L1Loss()
        self.smooth = DepthSmoothLoss()
        self.teacher_student = teacher_student_loss

        self.lam_img = lambda_img
        self.lam_ssim = lambda_ssim
        self.lam_perc = lambda_perc
        self.lam_fft = lambda_fft
        self.lam_grad = lambda_grad
        self.lam_depth = lambda_depth
        self.lam_smooth = lambda_smooth
        
        # 用于存储最近计算的各个损失组件
        self.latest_losses = {}

    def forward(self, enhanced, gt, depth_pred, depth_gt,
                student_feats=None, teacher_feats=None,
                student_attns=None, teacher_attns=None):
        loss = 0.0
        self.latest_losses = {}  # 重置损失组件记录
        
        # 基础像素与结构
        img_loss = self.lam_img * self.l1(enhanced, gt)
        ssim_loss = self.lam_ssim * self.ssim(enhanced, gt)
        loss += img_loss
        loss += ssim_loss
        self.latest_losses['img_loss'] = img_loss.item()
        self.latest_losses['ssim_loss'] = ssim_loss.item()
        
        # 感知
        perc_loss = self.lam_perc * self.perc(enhanced, gt)
        loss += perc_loss
        self.latest_losses['perc_loss'] = perc_loss.item()
        
        # 频域
        fft_loss = self.lam_fft * self.fft(enhanced, gt)
        loss += fft_loss
        self.latest_losses['fft_loss'] = fft_loss.item()
        
        # 高频梯度
        grad_loss = self.lam_grad * self.grad_loss(enhanced, gt)
        loss += grad_loss
        self.latest_losses['grad_loss'] = grad_loss.item()
        
        # 深度蒸馏
        depth_loss = torch.tensor(0.0, device=enhanced.device)
        if depth_pred is not None and depth_gt is not None:
            depth_loss = self.lam_depth * self.depth_loss(depth_pred, depth_gt)
            loss += depth_loss
            self.latest_losses['depth_pred_loss'] = depth_loss.item()
        
        # 深度平滑
        smooth_loss = self.lam_smooth * self.smooth(depth_pred)
        loss += smooth_loss
        self.latest_losses['depth_smooth_loss'] = smooth_loss.item()
        
        # 总深度损失 = 深度预测 + 平滑
        total_depth_loss = depth_loss + smooth_loss
        self.latest_losses['depth_total_loss'] = total_depth_loss.item()
        
        # 对齐损失
        if self.teacher_student is not None and student_feats is not None:
            ts_loss = self.teacher_student(student_feats, teacher_feats, student_attns, teacher_attns)
            loss += ts_loss
            self.latest_losses['teacher_student_loss'] = ts_loss.item()
            
        # 记录总损失
        self.latest_losses['total_loss'] = loss.item()
        
        return loss
    
    def get_depth_loss(self, depth_pred, depth_gt=None):
        """
        单独计算深度相关损失
        
        Args:
            depth_pred: 预测的深度/门控图 [B,1,H,W]
            depth_gt: 真实深度图 [B,1,H,W] 或 None
        
        Returns:
            total_loss: 总深度损失
            components: 深度损失的各个组件字典
        """
        losses = {}
        total_loss = 0.0
        
        # 深度预测损失
        depth_pred_loss = torch.tensor(0.0, device=depth_pred.device)
        if depth_gt is not None:
            depth_pred_loss = self.lam_depth * self.depth_loss(depth_pred, depth_gt)
            total_loss += depth_pred_loss
            losses['depth_pred_loss'] = depth_pred_loss.item()
        
        # 深度平滑损失
        smooth_loss = self.lam_smooth * self.smooth(depth_pred)
        total_loss += smooth_loss
        losses['depth_smooth_loss'] = smooth_loss.item()
        
        return total_loss, losses
    
    def get_latest_losses(self):
        """返回最近一次前向传播计算的所有损失组件"""
        return self.latest_losses.copy()

# 示例:
# ts_loss = TeacherStudentLoss(feat_weight=0.5, attn_weight=0.2)
# criterion = TotalLoss(teacher_student_loss=ts_loss)
# loss = criterion(enhanced, gt, depth_pred, depth_gt, stu_feats, tea_feats, stu_atts, tea_atts)
