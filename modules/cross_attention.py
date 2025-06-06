import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, dim: int, heads: int = 8, window_size: int = 0):
        super().__init__()
        assert dim % heads == 0, 'dim must be divisible by heads'
        self.dim = dim
        self.heads = heads
        self.d_k = dim // heads
        self.window_size = window_size                # 0 = 全局
        # 生成 Q, K, V
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        # 输出映射
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        # 保存最后的注意力图
        self.last_attn = None
        # 是否保存注意力图
        self.save_attention = False

    # ---------- 辅助：窗口切分 / 还原 ----------
    def _window_partition(self, x):
        """x:[B,C,H,W] → windows:[nW*B,C,ws,ws], 返回 pad 后尺寸"""
        if self.window_size == 0:        # 不打窗
            return x, None
        B, C, H, W = x.shape
        ws = self.window_size
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        x = F.pad(x, (0, pad_r, 0, pad_b))            # 右、下补零
        B, C, Hp, Wp = x.shape
        x = x.view(B, C, Hp//ws, ws, Wp//ws, ws)       # B,C,Nh,ws,Nw,ws
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()   # B,Nh,Nw,C,ws,ws
        windows = x.view(-1, C, ws, ws)                # nW*B,C,ws,ws
        return windows, (Hp, Wp)

    def _window_reverse(self, windows, pad_hw, B):
        """windows:[nW*B,C,ws,ws] → [B,C,H,W] (去掉补零)"""
        if self.window_size == 0:
            return windows
        Hp, Wp = pad_hw
        ws = self.window_size
        C = windows.size(1)
        x = windows.view(B, Hp//ws, Wp//ws, C, ws, ws) \
                .permute(0, 3, 1, 4, 2, 5).contiguous() \
                .view(B, C, Hp, Wp)
        # 去掉 padding
        return x[..., :Hp - (ws - Hp % ws) % ws,
                 :Wp - (ws - Wp % ws) % ws]
                 
    def enable_attention_saving(self, enable=True):
        """启用或禁用注意力图保存功能"""
        self.save_attention = enable
        print(f"[CrossAttention] 设置 save_attention = {enable}, dim={self.dim}, heads={self.heads}")
        
    def get_last_attn(self):
        """获取最近一次计算的注意力图"""
        return self.last_attn

    def forward(self, rgb_feat: torch.Tensor, dep_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = rgb_feat.shape

        # -- 1×1 conv 先做，再按需打窗 --
        q = self.to_q(rgb_feat)
        k = self.to_k(dep_feat)
        v = self.to_v(dep_feat)

        # 窗口切分（若 ws>0 且分辨率足够大）
        if self.window_size > 0 and max(H, W) > self.window_size:
            q, pad_hw = self._window_partition(q)
            k, _      = self._window_partition(k)
            v, _      = self._window_partition(v)
            B_ = q.size(0)                             # = nW*B
            H_ = W_ = self.window_size
        else:
            pad_hw = None
            B_ = B
            H_, W_ = H, W

        # 分头，展开 N
        q = q.view(B_, self.heads, self.d_k, H_*W_)
        k = k.view(B_, self.heads, self.d_k, H_*W_)
        v = v.view(B_, self.heads, self.d_k, H_*W_)
        # 计算注意力: qᵀ·k → [B, heads, N, N]
        attn = torch.einsum('bhcN,bhcM->bhNM', q, k) / math.sqrt(self.d_k)
        attn = torch.softmax(attn, dim=-1)
        
        # 保存注意力图（如果启用）
        if self.save_attention:
            # 如果使用窗口，需要将窗口注意力合并回原始图像尺寸
            if pad_hw is not None:
                # 这里实现窗口注意力的合并比较复杂，简化为仅保存第一个窗口的注意力
                # 在实际实现中可能需要更复杂的合并逻辑
                self.last_attn = attn[:B, :, :, :].detach()
                print(f"[CrossAttention] 保存窗口注意力图: shape={self.last_attn.shape}")
            else:
                self.last_attn = attn.detach()
                print(f"[CrossAttention] 保存全局注意力图: shape={self.last_attn.shape}")
        
        # 加权 v: attn @ vᵀ → [B, heads, N, d_k]
        out = torch.einsum('bhNM,bhcM->bhcN', attn, v)
        # 恢复形状: [B, C, H, W]
        out = out.contiguous().view(B_, C, H_, W_)

        # 还原窗口
        if pad_hw is not None:
            out = self._window_reverse(out, pad_hw, B)

        # 通道融合
        fused = self.proj(out)
        return fused

# 示例:
# ca = CrossAttention(dim=48, heads=8, window_size=8)
# fused_feat = ca(rgb_feat, depth_feat)  # Tensor[B,48,H,W]
# ca.enable_attention_saving(True)  # 启用注意力图保存
# attn_map = ca.get_last_attn()  # 获取注意力图
