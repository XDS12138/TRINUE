import torch
import torch.nn as nn

class DepthFeatureExtractor(nn.Module):
    """
    Depth Feature Extractor
    -----------------------
    从高精度深度图 D_gt (1×H×W) 提取多尺度深度特征:
      - 每级分辨率依次下采 (H/2, H/4, H/8, H/16)
      - 每级包含: Conv3×3(s=2) -> ReLU -> Conv3×3(s=1) -> ReLU

    输入:
      x: Tensor[B, 1, H, W]
    输出:
      feats: List[Tensor[B, C, H/2^i, W/2^i]]  i=1..levels
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 48, levels: int = 4):
        super().__init__()
        self.levels = levels
        self.base_channels = base_channels  # Store base_channels as an instance variable
        self.blocks = nn.ModuleList()
        for i in range(levels):
            # 第一层接收原始深度通道，后续层输入恒为 base_channels
            in_c = in_channels if i == 0 else base_channels
            self.blocks.append(nn.Sequential(
                # 下采 (stride=2)
                nn.Conv2d(in_c, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.ReLU(inplace=True),
                # 分辨率保持 (stride=1)
                nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))

    def forward(self, x: torch.Tensor) -> list:
        feats = []
        out = x
        for block in self.blocks:
            out = block(out)
            feats.append(out)
        return feats

# Example:
# dfe = DepthFeatureExtractor(in_channels=1, base_channels=48, levels=4)
# depth_feats = dfe(depth_gt)  # List of 4 feature maps at H/2, H/4, H/8, H/16
