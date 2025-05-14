import torch
import torch.nn as nn
import math

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

class RestormerBlock(nn.Module):
    """
    Restormer Transformer Block
    ---------------------------
    Combines axial self-attention (MDTA) and gated FFN (GDFN) in residual fashion.

    Args:
      dim (int): channel dimension
    """
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn  = MDTA(dim)
        self.norm2 = LayerNorm2d(dim)
        self.ffn   = GDFN(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # axial attention with residual
        x = x + self.attn(self.norm1(x))
        # gated feed-forward with residual
        x = x + self.ffn(self.norm2(x))
        return x
