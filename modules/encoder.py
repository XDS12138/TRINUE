import torch
import torch.nn as nn
from .sfe import ShallowFeatureExtractor
from .depth_feature_extractor import DepthFeatureExtractor
from .cross_attention import CrossAttention
from .transformer_block import RestormerBlock


class GTEncoder(nn.Module):
    """
    Teacher (GT) Encoder - Frozen CNN
    ----------------------------------
    从清晰 GT 图像提取多尺度特征，用于对齐监督。
    每级包含两次下采卷积。
    """
    def __init__(self, in_channels=3, base_channels=48, levels=4):
        super().__init__()
        self.levels = levels
        self.blocks = nn.ModuleList()
        for i in range(levels):
            inc = in_channels if i == 0 else base_channels
            self.blocks.append(nn.Sequential(
                nn.Conv2d(inc, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GELU(),
                nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GELU()
            ))
        # Freeze weights
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> list:
        feats = []
        out = x
        for block in self.blocks:
            out = block(out)
            feats.append(out)
        return feats


class RawEncoder(nn.Module):
    """
    Raw Encoder with Cross-Modal Fusion
    -----------------------------------
    同时融合 RGB 特征与多尺度深度特征，输出多级学生特征。

    输入:
      - raw: Tensor[B,3,H,W]
      - depth: Tensor[B,1,H,W]
    输出:
      - student_feats: list of Tensor[B,C,H/2^i,W/2^i], i=1..levels
      - teacher_feats: list of Tensor[B,C,...],  同级 GT 特征
    """
    def __init__(self,
                 in_channels=3,
                 depth_channels=1,
                 base_channels=48,
                 levels=4,
                 heads=8):
        super().__init__()
        self.levels = levels
        # 浅层特征提取
        self.sfe = ShallowFeatureExtractor(in_channels, base_channels)
        # 多尺度深度特征
        self.depth_extractor = DepthFeatureExtractor(depth_channels, base_channels, levels)
        # GT 编码器（冻结）
        self.gt_encoder = GTEncoder(in_channels, base_channels, levels)
        # 每级下采 conv + 融合 + Transformer
        self.down_convs = nn.ModuleList()
        self.cross_attns = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for _ in range(levels):
            # 下采块
            self.down_convs.append(nn.Sequential(
                nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(1, base_channels),
                nn.GELU()
            ))
            # 跨模态注意力
            self.cross_attns.append(CrossAttention(base_channels, heads))
            # 两个 Restormer Block
            self.transformers.append(nn.Sequential(
                RestormerBlock(base_channels),
                RestormerBlock(base_channels)
            ))

    def forward(self,
                raw: torch.Tensor,
                depth: torch.Tensor,
                gt: torch.Tensor) -> tuple:
        # 1. 浅层编码
        f0 = self.sfe(raw)                          # BxCxHxW
        # 2. 多尺度深度特征
        depth_feats = self.depth_extractor(depth)   # list of BxCxH/2^i
        # 3. GT 多尺度特征
        teacher_feats = self.gt_encoder(gt)         # list of BxCxH/2^i
        # 4. 多级融合编码
        student_feats = []
        x = f0
        for i in range(self.levels):
            # 下采
            x = self.down_convs[i](x)
            # 跨模态融合 (RGB ← Depth)
            x = self.cross_attns[i](x, depth_feats[i])
            # Transformer 编码
            x = self.transformers[i](x)
            # 收集学生特征
            student_feats.append(x)
        return student_feats, teacher_feats
