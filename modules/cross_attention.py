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
        return windows, (Hp, Wp, H, W)  # 保存填充后和原始尺寸

    def _window_reverse(self, windows, size_info, B):
        """windows:[nW*B,C,ws,ws] → [B,C,H,W] (去掉补零)"""
        if self.window_size == 0:
            return windows
        Hp, Wp, H, W = size_info  # 解包填充后和原始尺寸
        ws = self.window_size
        C = windows.size(1)
        x = windows.view(B, Hp//ws, Wp//ws, C, ws, ws) \
                .permute(0, 3, 1, 4, 2, 5).contiguous() \
                .view(B, C, Hp, Wp)
        # 直接切片到原始尺寸，避免复杂的填充计算
        return x[..., :H, :W]
                 
    def enable_attention_saving(self, enable=True):
        """启用或禁用注意力图保存功能"""
        self.save_attention = enable
        # print(f"[CrossAttention] 设置 save_attention = {enable}, dim={self.dim}, heads={self.heads}")
        
    def get_last_attn(self):
        """获取最近一次计算的注意力图"""
        return self.last_attn

    def _chunked_attention(self, q, k, v, save_max_batch: int = 1):
        """分块计算注意力，减少内存使用
        Args:
            q, k, v: [B_, heads, d_k, N]
            save_max_batch: 保存注意力图时最多保留的 batch 数（避免为所有窗口分配内存）
        """
        B_, heads, d_k, N = q.shape
        chunk_size = min(self.chunk_size, N)
        
        # 输出张量
        out = torch.zeros_like(q)
        attn_weights = None
        
        # 仅保存最多 save_max_batch 个样本的第一个chunk的注意力（避免 B_*nW 的巨量内存）
        if self.save_attention:
            first_chunk_size = min(chunk_size, N)
            B_save = min(save_max_batch, B_)
            attn_weights = torch.zeros(B_save, heads, first_chunk_size, N,
                                       device=q.device, dtype=q.dtype)
        
        scale = 1.0 / math.sqrt(self.d_k)
        
        # 进一步在窗口批维度 B_ 上分块，限制每次分配的注意力张量大小
        # 目标显存预算（单次 attn_chunk），可按需调整（单位：字节）
        target_bytes = 256 * 1024 * 1024  # 256 MB
        elem_size = q.element_size()
        # 单块 attn_chunk 的大小 ~ Bb * heads * chunk_size * N * elem_size
        # 计算 B_ 分块大小以满足内存预算
        denom = max(1, heads * chunk_size * N * elem_size)
        b_chunk = max(1, min(B_, target_bytes // denom))
        
                # 分块处理
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            # 在 B_ 维度上进一步分块
            for b0 in range(0, B_, b_chunk):
                b1 = min(b0 + b_chunk, B_)
                q_chunk = q[b0:b1, :, :, i:end_i]  # [Bb, heads, d_k, chunk_size]
                k_chunk = k[b0:b1]
                v_chunk = v[b0:b1]
                
                # 计算当前chunk的注意力
                attn_chunk = torch.einsum('bhcI,bhcJ->bhIJ', q_chunk, k_chunk) * scale
                attn_chunk = torch.softmax(attn_chunk, dim=-1)
                
                # 保存第一个i-chunk且第一个b-chunk的小批次注意力权重
                if self.save_attention and i == 0 and b0 == 0 and attn_weights is not None:
                    B_save = attn_weights.size(0)
                    attn_weights[:, :, :, :] = attn_chunk[:B_save]
                
                # 计算输出
                out_chunk = torch.einsum('bhIJ,bhcJ->bhcI', attn_chunk, v_chunk)
                out[b0:b1, :, :, i:end_i] = out_chunk
                
                # 清理中间变量
                del attn_chunk, out_chunk, q_chunk, k_chunk, v_chunk
        
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
            q, size_info = self._window_partition(q)
            k, _         = self._window_partition(k)
            v, _         = self._window_partition(v)
            B_ = q.size(0)                             # = nW*B
            H_ = W_ = self.window_size
        else:
            size_info = None
            B_ = B
            H_, W_ = H, W

        # 分头，展开 N
        q = q.view(B_, self.heads, self.d_k, H_*W_)
        k = k.view(B_, self.heads, self.d_k, H_*W_)
        v = v.view(B_, self.heads, self.d_k, H_*W_)
        
        # 使用分块计算注意力
        out, attn_weights = self._chunked_attention(q, k, v, save_max_batch=B)
        
        # 保存注意力图（如果启用）
        if self.save_attention and attn_weights is not None:
            # 直接保存已裁剪的小批次注意力图
            self.last_attn = attn_weights.detach().clone()
            
            # 立即断开与原始计算图的所有连接
            self.last_attn.requires_grad_(False)
            
        # 恢复形状: [B, C, H, W]
        out = out.contiguous().view(B_, C, H_, W_)

        # 还原窗口
        if size_info is not None:
            out = self._window_reverse(out, size_info, B)

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
