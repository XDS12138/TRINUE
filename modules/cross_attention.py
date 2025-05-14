import torch
import torch.nn as nn
import math

class CrossAttention(nn.Module):
    """
    Cross-Attention 模块 (RGB←Depth fusion)
    ---------------------------------------
    将深度特征作为 Key/Value，引导 RGB 特征的注意力分布。

    输入:
      - rgb_feat : Tensor[B, C, H, W]
      - dep_feat : Tensor[B, C, H, W]
    输出:
      - fused_feat : Tensor[B, C, H, W]

    实现细节:
    1. 使用 1×1 卷积分别生成 Q (从 rgb)、K、V (从 depth)
    2. 按头切分，展开空间维度 N=H×W
    3. 计算注意力 weights = softmax(Q·Kᵀ / sqrt(d_k))
    4. 输出 = weights · V, reshape 回 (B,C,H,W)
    5. 1×1 卷积融合通道
    """
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        assert dim % heads == 0, 'dim must be divisible by heads'
        self.dim = dim
        self.heads = heads
        self.d_k = dim // heads
        # 生成 Q, K, V
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        # 输出映射
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, rgb_feat: torch.Tensor, dep_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = rgb_feat.shape
        # 1×1 映射
        q = self.to_q(rgb_feat)
        k = self.to_k(dep_feat)
        v = self.to_v(dep_feat)
        # 分头，展开 N
        q = q.view(B, self.heads, self.d_k, H*W)          # [B, heads, d_k, N]
        k = k.view(B, self.heads, self.d_k, H*W)          # [B, heads, d_k, N]
        v = v.view(B, self.heads, self.d_k, H*W)          # [B, heads, d_k, N]
        # 计算注意力: qᵀ·k → [B, heads, N, N]
        attn = torch.einsum('bhcN,bhcM->bhNM', q, k) / math.sqrt(self.d_k)
        attn = torch.softmax(attn, dim=-1)
        # 加权 v: attn @ vᵀ → [B, heads, N, d_k]
        out = torch.einsum('bhNM,bhcM->bhcN', attn, v)
        # 恢复形状: [B, C, H, W]
        out = out.contiguous().view(B, C, H, W)
        # 通道融合
        fused = self.proj(out)
        return fused

# 示例:
# ca = CrossAttention(dim=48, heads=8)
# fused_feat = ca(rgb_feat, depth_feat)  # Tensor[B,48,H,W]
