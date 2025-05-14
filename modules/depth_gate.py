import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthGate(nn.Module):
    """
    DepthGate 模块
    ---------------
    输入:
      - skip_feat: Tensor[B, C, H, W]   (来自 RGB 编码器的跳跃特征)
      - depth_feat: Tensor[B, C_d, H, W] (来自 Depth 特征或预测深度头的门控图)
    输出:
      - gated_feat: Tensor[B, C, H, W]

    实现细节:
    1. 使用 1x1 卷积把 depth_feat 映射为单通道 gate_map (B×1×H×W)
    2. Sigmoid 激活得到 [0,1] 空间权重
    3. skip_feat * gate_map 广播相乘，实现空间选择性过滤
    """
    def __init__(self, in_channels_depth: int, reduction: int = 1):
        super().__init__()
        # depth_feat 通道映射到 1
        self.gate_conv = nn.Conv2d(in_channels_depth, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, skip_feat: torch.Tensor, depth_feat: torch.Tensor) -> torch.Tensor:
        # depth_feat: [B, C_d, H, W]
        gate_map = self.gate_conv(depth_feat)  # [B,1,H,W]
        gate_map = self.sigmoid(gate_map)
        
        # 处理空间尺寸不匹配
        if gate_map.shape[2:] != skip_feat.shape[2:]:
            gate_map = F.interpolate(gate_map, size=skip_feat.shape[2:], 
                                     mode='bilinear', align_corners=False)
        
        # 广播乘法
        return skip_feat * gate_map

# 示例用法:
# dg = DepthGate(in_channels_depth=48)
# gated = dg(skip_feat, depth_feat)  # Tensor[B,48,H,W]
