import torch
import torch.nn as nn

class ShallowFeatureExtractor(nn.Module):
    """
    Shallow Feature Extractor (SFE)
    -------------------------------
    两层 3x3 卷积 + GroupNorm + GELU
    输入 3×H×W, 输出 C×H×W
    用于提取初级纹理与色彩特征
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 48):
        super().__init__()
        # 第一层卷积
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.act1 = nn.GELU()
        # 第二层卷积
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 确保输入和权重的数据类型匹配，解决混合精度训练问题
        weight_dtype = self.conv1.weight.dtype
        if x.dtype != weight_dtype:
            x = x.to(dtype=weight_dtype)
            
        # 第一层：Conv3x3 -> GN -> GELU
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        # 第二层：Conv3x3 -> GN -> GELU
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x
