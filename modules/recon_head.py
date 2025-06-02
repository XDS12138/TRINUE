import torch
import torch.nn as nn

def default_activation(x: torch.Tensor) -> torch.Tensor:
    """
    默认激活函数，将输入张量控制在 [-1,1] 范围，模拟 tanh 行为
    """
    return torch.tanh(x)

class ReconHead(nn.Module):
    """
    Reconstruction Head
    -------------------
    将原图与去模糊和颜色残差合成，并执行激活。

    输入:
      - raw:   Tensor[B,3,H,W] 原始图 (归一化到 [-1,1])
      - res_d: Tensor[B,3,H,W] 高频去模糊残差
      - res_c: Tensor[B,3,H,W] 低频颜色残差
    输出:
      - out:   Tensor[B,3,H,W] 最终增强图 ([-1,1])

    默认操作: out = tanh(raw + res_d + res_c)
    可定制 activation
    """
    def __init__(self, activation=default_activation):
        super().__init__()
        self.activation = activation

    def forward(self,
                raw: torch.Tensor,
                res_d: torch.Tensor,
                res_c: torch.Tensor) -> torch.Tensor:
        # 合成 residuals
        x = raw + res_d + res_c
        # 激活到 [-1,1]
        return self.activation(x)

# 示例用法:
# recon = ReconHead()
# enhanced = recon(raw, res_d, res_c)  # Tensor[B,3,H,W]

class MultiTaskHead(nn.Module):
    """
    Multi-Task Reconstruction Head
    -----------------------------
    同时输出增强图像和深度预测，支持多任务学习。
    
    输入:
      - raw:   Tensor[B,3,H,W] 原始图 (归一化到 [-1,1])
      - res_d: Tensor[B,3,H,W] 高频去模糊残差  
      - res_c: Tensor[B,3,H,W] 低频颜色残差
      - feat:  Tensor[B,C,H,W] 解码器特征图 (用于深度预测)
    输出:
      - enhanced:   Tensor[B,3,H,W] 最终增强图 ([-1,1])
      - depth_pred: Tensor[B,1,H,W] 预测深度图 ([0,1])
    """
    def __init__(self, feat_channels=48, activation=default_activation):
        super().__init__()
        self.activation = activation
        
        # 深度预测分支
        self.depth_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(feat_channels // 2, 1, kernel_size=1, bias=True),
            nn.Sigmoid()  # 输出 [0,1] 范围的深度图
        )

    def forward(self,
                raw: torch.Tensor,
                res_d: torch.Tensor,
                res_c: torch.Tensor,
                feat: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        # 增强图像重建（保持原有逻辑）
        enhanced = raw + res_d + res_c
        enhanced = self.activation(enhanced)
        
        # 深度预测
        if feat is not None:
            depth_pred = self.depth_head(feat)
        else:
            # 如果没有特征图，返回零深度图
            depth_pred = torch.zeros(raw.shape[0], 1, raw.shape[2], raw.shape[3], 
                                   device=raw.device, dtype=raw.dtype)
        
        return enhanced, depth_pred
