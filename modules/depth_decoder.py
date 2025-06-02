# depth_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import RestormerBlock      # 复用现有块
import logging

logger = logging.getLogger(__name__)

class DepthDecoder(nn.Module):
    """
    对称 U-shape Depth Decoder
    --------------------------
    Inputs:
      - bottleneck : Tensor[B, C_b, H/16, W/16]
      - skip_feats : list[Tensor]  len = levels   (RGB encoder 的 shallow→deep)
    Outputs:
      - depth_pred : Tensor[B, 1, H, W]    连续深度
      - depth_feats: list[Tensor]           多尺度深度特征 (用于 RGB 引导)
    """
    def __init__(self, base_c=48, levels=4, window=4):
        super().__init__()
        self.levels = levels
        ch = [base_c * 2**i for i in range(levels)][::-1]   # H/16→H
        self.ups = nn.ModuleList()
        self.skip_proj = nn.ModuleList()
        self.blocks = nn.ModuleList()

        in_c = ch[0]
        for i in range(levels-1):            # 共 levels-1 级上采
            out_c = ch[i+1]
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(in_c, out_c*4, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            self.skip_proj.append(nn.Conv2d(out_c, out_c, 1, bias=False))
            self.blocks.append(RestormerBlock(out_c, heads=8, window_size=window))
            in_c = out_c

        self.pred = nn.Conv2d(ch[-1], 1, 3, padding=1)

    def forward(self, bottleneck, skip_feats):
        # 检查skip_feats长度
        if len(skip_feats) != self.levels:
            logger.warning(f"Expected {self.levels} skip features, got {len(skip_feats)}. Attempting to adapt.")
            # 尝试调整skip_feats列表长度
            if len(skip_feats) < self.levels:
                # 不足则复制最后一个
                while len(skip_feats) < self.levels:
                    skip_feats.append(skip_feats[-1])
            else:
                # 过多则截断
                skip_feats = skip_feats[:self.levels]
        
        x = bottleneck
        depth_feats = []
        # 记录每层特征的形状，便于调试
        logger.debug(f"DepthDecoder bottleneck shape: {bottleneck.shape}")
        
        for i in range(self.levels-1):
            x = self.ups[i](x)
            logger.debug(f"DepthDecoder after upsampling[{i}]: shape={x.shape}")
            
            # 确保skip特征尺寸匹配
            skip = F.interpolate(skip_feats[-(i+2)], size=x.shape[-2:], mode='bilinear', align_corners=False)
            logger.debug(f"DepthDecoder skip[{i}] shape after resize: {skip.shape}")
            
            # 投影并拼接
            projected_skip = self.skip_proj[i](skip)
            logger.debug(f"DepthDecoder projected skip[{i}] shape: {projected_skip.shape}")
            
            x = torch.cat([x, projected_skip], dim=1)
            logger.debug(f"DepthDecoder after concat[{i}]: shape={x.shape}")
            
            # 通过块处理
            x = self.blocks[i](x)
            logger.debug(f"DepthDecoder after block[{i}]: shape={x.shape}")
            
            # 插入到输出特征列表的前面（从深到浅的顺序）
            depth_feats.insert(0, x)
        
        # 生成最终深度预测并添加到特征列表开头
        depth_pred = torch.sigmoid(self.pred(x))
        logger.debug(f"DepthDecoder final prediction shape: {depth_pred.shape}")
        depth_feats.insert(0, depth_pred)
        
        # 打印所有特征形状以便调试
        logger.debug(f"DepthDecoder output feature shapes: {[f.shape for f in depth_feats]}")
        
        return depth_pred, depth_feats 