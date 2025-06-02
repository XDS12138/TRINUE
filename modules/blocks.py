import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class DACBlock(nn.Module):
    """
    Depth-Adaptive Color Block for color branch.
    输入:
      - x: Tensor[B,C,H,W]
      - gate_map: Tensor[B,1,H,W]
    """
    def __init__(self, C):
        super().__init__()
        # 空洞深度可分离卷积
        self.depth_dw = nn.Conv2d(C, C, kernel_size=3, padding=2, dilation=2, groups=C)
        # 通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, C//4, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(C//4, C, 1, bias=False),
            nn.Sigmoid()
        )
        # 逐点卷积混合
        self.point_pw = nn.Conv2d(C, C, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x, gate_map):
        # 1) 深度 DW + 膨胀
        y = self.depth_dw(x)
        # 2) 通道注意力
        ca = self.channel_attn(y)
        
        # 修复gate_map的通道维度
        # 确保gate_map的通道数与ca匹配
        if gate_map.shape[1] != ca.shape[1]:
            # 如果gate_map是单通道的，扩展它以匹配ca通道数
            if gate_map.shape[1] == 1:
                gate_map = gate_map.expand(-1, ca.shape[1], -1, -1)
            else:
                # 否则，应用通道投影
                gate_map = torch.mean(gate_map, dim=1, keepdim=True).expand(-1, ca.shape[1], -1, -1)
        
        # 应用通道注意力和gate_map
        y = y * ca * gate_map
        
        # 3) 逐点卷积 + 激活
        y = self.point_pw(y)
        y = self.act(y)
        # 4) 残差融合
        return y + x 

class MDTA(nn.Module):
    """
    Multi-DConv Head Transposed Attention (Axial Attention for Restormer)
    ---------------------------------------------------------------
    Efficient self-attention by decomposing over spatial axes using depthwise conv.

    Args:
      dim (int): number of input channels
      heads (int): number of attention heads
    """
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by number of heads"
        self.dim = dim
        self.heads = heads
        self.d_k = dim // heads
        # project to Q, K, V
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        # depthwise conv for spatial context
        self.dw_conv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=False)
        # output projection
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # generate Q, K, V
        qkv = self.dw_conv(self.qkv(x))  # [B, 3C, H, W]
        q, k, v = torch.chunk(qkv, 3, dim=1)
        # reshape for multi-head
        q = q.view(B, self.heads, self.d_k, H * W)
        k = k.view(B, self.heads, self.d_k, H * W)
        v = v.view(B, self.heads, self.d_k, H * W)
        # attention: Q^T K
        attn = torch.einsum('bhcN,bhcM->bhNM', q, k) / math.sqrt(self.d_k)
        attn = torch.softmax(attn, dim=-1)
        # apply attention to V
        out = torch.einsum('bhNM,bhcM->bhcN', attn, v)
        out = out.contiguous().view(B, C, H, W)
        return self.proj(out)

class GDFN(nn.Module):
    """
    Gated Depthwise Feed-Forward Network
    -------------------------------------
    Combines pointwise expand, depthwise spatial mixing, and gating.

    Args:
      dim (int): input/output channels
      expand_ratio (int): expansion factor for hidden channels
    """
    def __init__(self, dim: int, expand_ratio: int = 2):
        super().__init__()
        hidden = dim * expand_ratio
        # projection in
        self.project_in = nn.Conv2d(dim, hidden, kernel_size=1, bias=False)
        # spatial mixing
        self.dw_conv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False)
        # gating
        self.gate_conv = nn.Conv2d(hidden, hidden, kernel_size=1, bias=False)
        # activation
        self.act = nn.GELU()
        # projection out
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.project_in(x)
        x_dw = self.dw_conv(x_in)
        x_gate = self.gate_conv(x_in)
        x = x_gate * x_dw
        x = self.act(x)
        return self.project_out(x)

class LayerNorm2d(nn.Module):
    """
    LayerNorm for channels of 2D spatial BCHW tensors
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # [B,C,H,W] -> [B,H,W,C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # [B,H,W,C] -> [B,C,H,W]
        x = x.permute(0, 3, 1, 2)
        return x

class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention.
    Args:
        dim (int)        : channel dim
        heads (int)      : #heads
        window_size (int): one side of the square window (e.g. 8)
    """
    def __init__(self, dim, heads=8, window_size=8):
        super().__init__()
        assert dim % heads == 0, "Channel dim must be divisible by heads"
        self.heads       = heads
        self.window_size = window_size
        self.d_k         = dim // heads

        # projections
        self.qkv   = nn.Linear(dim, dim * 3, bias=False)
        self.proj  = nn.Linear(dim, dim, bias=False)

    def _window_partition(self, x):
        """
        x: [B, C, H, W]  →  [num_windows*B, window^2, C]
        """
        B, C, H, W = x.shape
        ws = self.window_size
        # pad if needed
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        x = F.pad(x, (0, pad_r, 0, pad_b))
        B, C, Hp, Wp = x.shape
        x = x.view(B, C, Hp // ws, ws, Wp // ws, ws)          # B,C,Nh,ws,Nw,ws
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()          # B,Nh,Nw,ws,ws,C
        windows = x.view(-1, ws * ws, C)                      # B*Nh*Nw , ws² , C
        return windows, (Hp, Wp)

    def _window_reverse(self, windows, pad_hw, B):
        Hp, Wp      = pad_hw
        ws          = self.window_size
        C           = windows.size(-1)
        x = windows.view(B, Hp // ws, Wp // ws, ws, ws, C)    # B,Nh,Nw,ws,ws,C
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()          # B,C,Nh,ws,Nw,ws
        x = x.view(B, C, Hp, Wp)
        return x[:, :, :Hp - (ws - Hp % ws) % ws, :Wp - (ws - Wp % ws) % ws]

    def forward(self, x):
        """
        x  : [B, C, H, W]
        out: [B, C, H, W]
        """
        B, C, H, W = x.shape
        windows, pad_hw = self._window_partition(x)           # nW*B, ws², C
        qkv = self.qkv(windows).reshape(-1, windows.size(1), 3, self.heads, self.d_k)
        q, k, v = qkv.unbind(dim=2)                           # each: [nW*B, ws², heads, dk]
        q = q.transpose(1, 2)                                 # [nW*B, heads, ws², dk]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_k)
        attn = attn.softmax(dim=-1)
        out  = (attn @ v)                                     # [nW*B, heads, ws², dk]
        out  = out.transpose(1, 2).reshape(windows.shape)     # [nW*B, ws², C]
        out  = self.proj(out)
        out  = self._window_reverse(out, pad_hw, B)           # [B, C, H, W]
        return out

class RestormerBlock(nn.Module):
    """
    Restormer Transformer Block
    ---------------------------
    Combines axial self-attention (MDTA) and gated FFN (GDFN) in residual fashion.

    Args:
      dim (int): channel dimension
      heads (int): number of attention heads
      window_size (int): window size for attention (0 for global)
    """
    def __init__(self, dim: int, heads: int = 8, window_size: int = 0):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn  = WindowAttention(dim, heads, window_size) \
                     if window_size > 0 else MDTA(dim, heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn   = GDFN(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # axial attention with residual
        x = x + self.attn(self.norm1(x))
        # gated feed-forward with residual
        x = x + self.ffn(self.norm2(x))
        return x

class TransformerBlock(nn.Module):
    """
    Transformer Block with downsampling and dimension change capabilities
    ---------------------------------------------------------------------
    This block optionally performs downsampling and changes the number of channels.
    It uses MDTA for attention and GDFN for feed-forward network, similar to RestormerBlock.

    Args:
        dim (int): Input channel dimension
        output_dim (int): Output channel dimension
        num_heads (int): Number of attention heads
        window_size (int): Window size for attention (0 for global). MDTA in this file does not use this yet.
        downsample (bool): Whether to perform downsampling
    """
    def __init__(self, dim: int, output_dim: int, num_heads: int = 8, window_size: int = 0, downsample: bool = False):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.downsample = downsample
        
        current_dim = dim
        # Optional downsampling layer (stride=2 conv)
        if downsample:
            self.downsample_layer = nn.Conv2d(
                current_dim, current_dim, kernel_size=3, stride=2, padding=1, bias=False
            )
        else:
            self.downsample_layer = nn.Identity()
            
        # Transformer core components (MDTA, GDFN operate on current_dim)
        self.norm1 = LayerNorm2d(current_dim)
        # Note: The current MDTA in this file doesn't accept window_size. 
        # If windowed attention is desired for this TransformerBlock, MDTA needs to be updated or a different attention mechanism used.
        self.attn = MDTA(dim=current_dim, heads=num_heads) 
        self.norm2 = LayerNorm2d(current_dim)
        self.ffn = GDFN(dim=current_dim)
        
        # Output projection if dimensions don't match after transformer operations
        if current_dim != output_dim:
            self.output_proj = nn.Conv2d(current_dim, output_dim, kernel_size=1, bias=False)
        else:
            self.output_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply downsampling if enabled
        x = self.downsample_layer(x)
        
        # Apply transformer core (attention + ffn with residuals)
        # The residual connection should be with the state after downsampling.
        x_res = x 
        x = x_res + self.attn(self.norm1(x)) # MDTA operates on x, which is (B, current_dim, H_new, W_new)
        x = x + self.ffn(self.norm2(x))      # GDFN also operates on current_dim
        
        # Project to output dimension if needed
        x = self.output_proj(x) # Projects from current_dim to output_dim
        
        return x 
    

    # 添加缺失的ConvBlock类定义
class ConvBlock(nn.Module):
    """基本卷积块，包含卷积、归一化和激活函数"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.GroupNorm(1, out_channels)  # 使用GroupNorm代替BatchNorm
        self.act = nn.GELU()  # 使用GELU激活函数
        
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# 添加缺失的LocalTransformerBlock类定义
class LocalTransformerBlock(nn.Module):
    """局部注意力Transformer块"""
    def __init__(self, channels, num_heads=8, ffn_expansion=2):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        # 规范化层
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        
        # 多头自注意力
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        
        # 前馈网络
        ffn_dim = channels * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, ffn_dim, 1),
            nn.GELU(),
            nn.Conv2d(ffn_dim, channels, 1)
        )
        
    def forward(self, x):
        # 输入: [B, C, H, W]
        b, c, h, w = x.shape
        
        # 自注意力
        x_norm = self.norm1(x)
        # 将空间维度展平作为序列
        x_flat = x_norm.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = attn_out.permute(0, 2, 1).reshape(b, c, h, w)  # 恢复形状
        x = x + attn_out  # 残差连接
        
        # 前馈网络
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)  # 残差连接
        
        return x

