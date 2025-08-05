import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    """
    Cross-Attention 模块 (RGB←Depth fusion) with Memory Optimization
    ----------------------------------------------------------------
    将深度特征作为 Key/Value，引导 RGB 特征的注意力分布。

    输入:
      - rgb_feat : Tensor[B, C, H, W]
      - dep_feat : Tensor[B, C, H, W]
    输出:
      - fused_feat : Tensor[B, C, H, W]

    实现细节:
    1. 使用 1×1 卷积分别生成 Q (从 rgb)、K、V (从 depth)
    2. 按头切分，展开空间维度 N=H×W
    3. **内存优化**: 使用分块计算避免显存爆炸
    4. 计算注意力 weights = softmax(Q·Kᵀ / sqrt(d_k))
    5. 输出 = weights · V, reshape 回 (B,C,H,W)
    6. 1×1 卷积融合通道
    """
    def __init__(self, dim: int, heads: int = 8, window_size: int = 0, chunk_size: int = 1024):
        super().__init__()
        assert dim % heads == 0, 'dim must be divisible by heads'
        self.dim = dim
        self.heads = heads
        self.d_k = dim // heads
        self.window_size = window_size                # 0 = 全局
        self.chunk_size = chunk_size                  # 分块大小，用于内存优化
        
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
        # print(f"[CrossAttention] 设置 save_attention = {enable}, dim={self.dim}, heads={self.heads}")
        
    def get_last_attn(self):
        """获取最近一次计算的注意力图"""
        return self.last_attn

    def _chunked_attention(self, q, k, v):
        """分块计算注意力，减少内存使用"""
        B_, heads, d_k, N = q.shape
        chunk_size = min(self.chunk_size, N)
        
        # 输出张量
        out = torch.zeros_like(q)
        attn_weights = None
        
        # 保存注意力权重用于可视化（仅保存第一个chunk）
        if self.save_attention:
            first_chunk_size = min(chunk_size, N)
            attn_weights = torch.zeros(B_, heads, first_chunk_size, N, 
                                     device=q.device, dtype=q.dtype)
        
        # 分块处理
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            q_chunk = q[:, :, :, i:end_i]  # [B_, heads, d_k, chunk_size]
            
            # 计算当前chunk的注意力
            attn_chunk = torch.einsum('bhcI,bhcJ->bhIJ', q_chunk, k) / math.sqrt(d_k)
            attn_chunk = torch.softmax(attn_chunk, dim=-1)
            
            # 保存第一个chunk的注意力权重
            if self.save_attention and i == 0:
                attn_weights[:, :, :, :] = attn_chunk
            
            # 计算输出
            out_chunk = torch.einsum('bhIJ,bhcJ->bhcI', attn_chunk, v)
            out[:, :, :, i:end_i] = out_chunk
            
            # 清理中间变量
            del attn_chunk, out_chunk
        
        return out, attn_weights

    def forward(self, rgb_feat: torch.Tensor, dep_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = rgb_feat.shape

        # 清理之前的注意力图引用
        if hasattr(self, 'last_attn'):
            self.last_attn = None

        # -- 检查输入尺寸，如果太大使用窗口注意力 --
        total_pixels = H * W
        memory_threshold = 64 * 64  # 超过64x64使用窗口注意力
        
        if total_pixels > memory_threshold and self.window_size == 0:
            # 自动启用窗口注意力
            adaptive_window_size = min(32, max(8, int(math.sqrt(memory_threshold))))
            print(f"[CrossAttention] 自动启用窗口注意力: window_size={adaptive_window_size}, input_size={H}x{W}")
            original_window_size = self.window_size
            self.window_size = adaptive_window_size
        else:
            original_window_size = None

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
        
        # 使用分块计算注意力
        out, attn_weights = self._chunked_attention(q, k, v)
        
        # 保存注意力图（如果启用）
        if self.save_attention and attn_weights is not None:
            if pad_hw is not None:
                # 窗口注意力情况
                self.last_attn = attn_weights[:B, :, :, :].detach().clone()
                # print(f"[CrossAttention] 保存窗口注意力图: shape={self.last_attn.shape}")
            else:
                self.last_attn = attn_weights.detach().clone()
                # print(f"[CrossAttention] 保存全局注意力图: shape={self.last_attn.shape}")
        
            # 立即断开与原始计算图的所有连接
            self.last_attn.requires_grad_(False)
            
        # 恢复形状: [B, C, H, W]
        out = out.contiguous().view(B_, C, H_, W_)

        # 还原窗口
        if pad_hw is not None:
            out = self._window_reverse(out, pad_hw, B)

        # 恢复原始窗口大小设置
        if original_window_size is not None:
            self.window_size = original_window_size

        # 通道融合
        fused = self.proj(out)
        return fused

# 示例:
# ca = CrossAttention(dim=48, heads=8, window_size=8, chunk_size=512)
# fused_feat = ca(rgb_feat, depth_feat)  # Tensor[B,48,H,W]
# ca.enable_attention_saving(True)  # 启用注意力图保存
# attn_map = ca.get_last_attn()  # 获取注意力图
