# depth_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import RestormerBlock      # 复用现有块
from .cross_attention import CrossAttention  # 引入交叉注意力模块
import logging

logger = logging.getLogger(__name__)

class DepthDecoder(nn.Module):
    """
    对称 U-shape Depth Decoder with RGB→Depth Cross-Attention
    ---------------------------------------------------------
    Inputs:
      - bottleneck : Tensor[B, C_b, H/16, W/16]
      - skip_feats : list[Tensor]  len = levels   (RGB encoder 的 shallow→deep)
    Outputs:
      - depth_pred : Tensor[B, 1, H, W]    连续深度
      - depth_feats: list[Tensor]           多尺度深度特征 (用于 RGB 引导)
    
    新增功能：在解码过程中，每一层的深度特征都与对应的RGB特征进行RGB→Depth交叉注意力融合
    """
    def __init__(self, base_c=48, levels=4, window=4, enable_cross_attention=True, 
                 min_depth=2000.0, max_depth=65535.0):
        super().__init__()
        self.levels = levels
        self.base_c = base_c  # 保存base_c用于后续计算
        self.enable_cross_attention = enable_cross_attention
        
        # 🔧 新增：深度范围配置
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        ch = [base_c * 2**i for i in range(levels)][::-1]   # H/16→H
        self.ups = nn.ModuleList()
        self.skip_proj = nn.ModuleList()
        self.fusion_proj = nn.ModuleList()  # 新增：特征融合后的通道投影层
        self.blocks = nn.ModuleList()
        
        # 新增：RGB→Depth交叉注意力模块
        if self.enable_cross_attention:
            self.rgb2depth_attn_blocks = nn.ModuleList()
            self.r2d_gamma = nn.ParameterList()  # 可学习门控参数
        else:
            self.rgb2depth_attn_blocks = nn.ModuleList([nn.Identity() for _ in range(levels)])
            self.r2d_gamma = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(levels)])

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
            # 新增：融合投影层，将拼接后的2*out_c通道降为out_c通道
            self.fusion_proj.append(nn.Conv2d(out_c * 2, out_c, 1, bias=False))
            self.blocks.append(RestormerBlock(out_c, heads=8, window_size=window))
            
            # 添加RGB→Depth交叉注意力
            if self.enable_cross_attention:
                self.rgb2depth_attn_blocks.append(CrossAttention(out_c, heads=8, window_size=8, chunk_size=512))
                self.r2d_gamma.append(nn.Parameter(torch.zeros(1), requires_grad=True))
            
            in_c = out_c
        
        # 为最深层（bottleneck级别）也添加交叉注意力
        if self.enable_cross_attention:
            self.rgb2depth_attn_blocks.append(CrossAttention(ch[0], heads=8, window_size=8, chunk_size=512))  # bottleneck层
            self.r2d_gamma.append(nn.Parameter(torch.zeros(1), requires_grad=True))

        self.pred = nn.Conv2d(ch[-1], 1, 3, padding=1)
        
        # 移除冗余的channel_adapters，由RawEncoder统一管理通道适配
        # DepthDecoder现在直接输出与自身架构一致的通道数
        logger.info(f"DepthDecoder initialized with levels={levels}, channels={ch}, cross_attention={enable_cross_attention}")

    def forward(self, bottleneck, skip_feats, apply_cross_attention=True):
        """
        Args:
            bottleneck: 瓶颈层特征
            skip_feats: RGB编码器的跳跃连接特征
            apply_cross_attention: 是否应用RGB→Depth交叉注意力（Pass-1时为True）
        """
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
        
        # 确保输入和权重的数据类型匹配，解决混合精度训练问题
        if len(self.ups) > 0:
            ups_dtype = next(self.ups[0].parameters()).dtype
            if bottleneck.dtype != ups_dtype:
                bottleneck = bottleneck.to(dtype=ups_dtype)
            
            # 同样检查skip_feats的数据类型
            for i in range(len(skip_feats)):
                if skip_feats[i].dtype != ups_dtype:
                    skip_feats[i] = skip_feats[i].to(dtype=ups_dtype)
        
        x = bottleneck
        depth_feats = []
        # 记录每层特征的形状，便于调试
        logger.debug(f"DepthDecoder bottleneck shape: {bottleneck.shape}")
        
        # 保存每个层次的特征，用于与投影层期望的通道数匹配
        level_features = []
        # 保存原始bottleneck用于最深层特征
        original_bottleneck = bottleneck
        
        # 首先对bottleneck应用RGB→Depth交叉注意力（如果启用）
        if apply_cross_attention and self.enable_cross_attention:
            # 使用最深层的RGB特征（skip_feats[-1]）与bottleneck进行交叉注意力
            rgb_feat_deep = skip_feats[-1]  # 最深层RGB特征
            
            # 确保空间尺寸匹配
            if rgb_feat_deep.shape[2:] != x.shape[2:]:
                rgb_feat_deep = F.interpolate(rgb_feat_deep, size=x.shape[2:], mode='bilinear', align_corners=False)
            
            # RGB→Depth交叉注意力
            fused_depth = self.rgb2depth_attn_blocks[-1](x, rgb_feat_deep)  # 最后一个注意力块用于bottleneck
            gamma_r2d = self.r2d_gamma[-1]
            gated_depth = gamma_r2d * fused_depth
            x = x + gated_depth
            logger.debug(f"DepthDecoder: Bottleneck RGB→Depth attention applied, γ_r2d={gamma_r2d.item():.4f}")
        
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
            
            # 使用融合投影层将通道数从2*out_c降到out_c
            x = self.fusion_proj[i](x)
            logger.debug(f"DepthDecoder after fusion projection[{i}]: shape={x.shape}")
            
            # 通过块处理
            x = self.blocks[i](x)
            logger.debug(f"DepthDecoder after block[{i}]: shape={x.shape}")
            
            # 应用RGB→Depth交叉注意力（如果启用）
            if apply_cross_attention and self.enable_cross_attention and i < len(self.rgb2depth_attn_blocks) - 1:
                # 使用对应层级的RGB特征
                rgb_feat_level = skip_feats[-(i+2)]  # 对应的RGB特征层
                
                # 确保空间尺寸匹配
                if rgb_feat_level.shape[2:] != x.shape[2:]:
                    rgb_feat_level = F.interpolate(rgb_feat_level, size=x.shape[2:], mode='bilinear', align_corners=False)
                
                # RGB→Depth交叉注意力
                fused_depth = self.rgb2depth_attn_blocks[i](x, rgb_feat_level)
                gamma_r2d = self.r2d_gamma[i]
                gated_depth = gamma_r2d * fused_depth
                x = x + gated_depth
                logger.debug(f"DepthDecoder: Level {i} RGB→Depth attention applied, γ_r2d={gamma_r2d.item():.4f}")
            
            # 保存每一层的特征
            level_features.append(x)
        
        # 生成最终深度预测
        # 🔧 修复: 移除sigmoid限制，直接输出原始深度范围
        raw_pred = self.pred(x)
        
        # 🔧 新增: 将原始预测映射到实际深度范围[5000, 65000]
        # 使用tanh激活确保输出在合理范围内，然后映射到深度范围
        normalized_pred = torch.tanh(raw_pred)  # 输出范围 [-1, 1]
        
        # 映射到 [0, 1] 然后再映射到深度范围
        normalized_pred = (normalized_pred + 1.0) / 2.0  # 映射到 [0, 1]
        
        # 映射到实际深度范围
        depth_pred = normalized_pred * (self.max_depth - self.min_depth) + self.min_depth
        
        logger.debug(f"DepthDecoder final prediction shape: {depth_pred.shape}, range: [{depth_pred.min().item():.1f}, {depth_pred.max().item():.1f}]")
        
        # 构建depth_feats列表，按照从浅到深的顺序（与encoder一致）
        # 期望的通道数：[48, 96, 192, 384]（对应级别 0, 1, 2, 3）
        
        # 基于解码器架构重新构建正确的depth_feats
        # ch = [384, 192, 96, 48] 是解码器的通道配置（从深到浅）
        ch = [self.base_c * 2**i for i in range(self.levels)][::-1]
        depth_feats = []
        
        # 第0级：48通道（最浅层）
        depth_feats.append(x)  # 48 channels @ full resolution
        
        # 第1级：96通道 
        if len(level_features) >= 1:
            depth_feats.append(level_features[-1])  # 最后处理的特征（96通道）
        else:
            # fallback: 下采样并调整通道
            feat_96 = F.avg_pool2d(x, kernel_size=2, stride=2)
            depth_feats.append(torch.cat([feat_96, feat_96], dim=1))  # 48->96
            
        # 第2级：192通道
        if len(level_features) >= 2:
            depth_feats.append(level_features[-2])  # 倒数第二个特征（192通道）
        else:
            # fallback: 下采样并调整通道
            feat_192 = F.avg_pool2d(x, kernel_size=4, stride=4)
            depth_feats.append(feat_192.repeat(1, 4, 1, 1))  # 48->192
            
        # 第3级：384通道（最深层）
        # 使用增强后的bottleneck或原始bottleneck
        if apply_cross_attention and self.enable_cross_attention:
            # 如果应用了交叉注意力，x可能已经被修改，但应该使用正确维度的特征
            depth_feats.append(original_bottleneck)  # 始终使用384通道的原始bottleneck
        else:
            depth_feats.append(original_bottleneck)  # 384 channels
            
        # 确保depth_feats长度正确
        while len(depth_feats) < self.levels:
            depth_feats.append(depth_feats[-1])  # 用最后一个填充
        depth_feats = depth_feats[:self.levels]  # 截断到正确长度
        
        # 打印所有特征形状以便调试
        logger.debug(f"DepthDecoder output feature shapes: {[f.shape for f in depth_feats]}")
        
        return depth_pred, depth_feats 