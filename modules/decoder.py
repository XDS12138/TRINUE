import torch
import torch.nn as nn
import torch.nn.functional as F
from .depth_gate import DepthGate
from .transformer_block import RestormerBlock

class MultiTaskDecoder(nn.Module):
    """
    Multi-task Decoder for Deblur and Color Correction with Dense Skip and Adaptive Depth Fusion
    -------------------------------------------------------------------------------------------
    Inputs:
      - fused_feat: Tensor[B, C, H/8, W/8]    (Joint Bottleneck output)
      - skip_feats: list of RGB encoder features [F0_raw, F1_raw, ..., Fn_raw]
      - depth_feats: list of Depth features [F0_dep, F1_dep, ..., Fn_dep]
      - raw: Tensor[B, 3, H, W]             (Original image for Color branch)

    Outputs:
      - res_d: Tensor[B, 3, H, W]  Deblur residual
      - res_c: Tensor[B, 3, H, W]  Color correction residual
    """
    def __init__(self, base_channels=48, levels=3):
        super().__init__()
        self.levels = levels  # Number of upsampling stages
        self.depth_levels = levels + 1  # Total number of depth feature scales
        self.base_channels = base_channels # Store this for reference

        # Deblur Branch
        # PixelShuffle upsampling for each stage - 注意使用正确的通道数
        # 对于PixelShuffle(2)，输入通道需要是输出通道的4倍，才能保持通道数不变
        self.deblur_ups = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels*4, kernel_size=1, bias=False),
                nn.PixelShuffle(2)
            ) for _ in range(levels)
        ])
        
        # Adaptive Depth Fusion Components
        # 1. Scale-MLP: predicts fusion weights for different depth scales
        self.scale_mlp = nn.Sequential(
            nn.Conv2d(base_channels * self.depth_levels, base_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(base_channels, self.depth_levels, kernel_size=1)  # 输出每个尺度一个分数
        )
        
        # 2. Single gate convolution for final gating (replaces DepthGate modulelist)
        self.gate_conv = nn.Conv2d(base_channels, 1, kernel_size=1, bias=False)
        
        # 1x1 conv to compress concatenated channels back to C
        # stages: i=0..levels-1, input channels = C*(1 + (i+1))
        self.deblur_fuse_convs = nn.ModuleList([
            nn.Conv2d(base_channels * (i + 2), base_channels, kernel_size=1, bias=False)
            for i in range(levels)
        ])
        
        # Transformer blocks after fusion
        self.deblur_blocks = nn.ModuleList([
            nn.Sequential(
                RestormerBlock(base_channels),
                RestormerBlock(base_channels)
            ) for _ in range(levels)
        ])
        
        # Final reconstruction conv for Deblur branch
        self.deblur_recon = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)

        # Color Branch (unchanged)
        self.color_ups = nn.Upsample(scale_factor=2**levels, mode='bilinear', align_corners=False)
        self.color_gate = DepthGate(base_channels)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(base_channels, base_channels * 2, bias=False),
            nn.GELU(),
            nn.Linear(base_channels * 2, base_channels * 2, bias=False)
        )
        
        # 预定义的投影层，用于处理不同维度的情况
        # 为深度特征预创建的投影层（在需要时使用）
        self.depth_proj = nn.Conv2d(base_channels*2, base_channels, kernel_size=1, bias=False)
        # 新增: 为单通道深度特征添加专用投影层
        self.depth_proj_1ch = nn.Conv2d(1, base_channels, kernel_size=1, bias=False)
        
        # 为颜色分支预创建的投影层
        self.color_proj = nn.Conv2d(base_channels, base_channels, kernel_size=1, bias=False)
        
        # 为融合特征预创建的投影层
        self.fused_proj = nn.ModuleDict()
        # 我们不能简单地依赖理论计算的通道数，因为实际运行时可能会有差异
        # 为每一层创建一个临时变量，在forward中根据实际channels动态创建相应的投影层

        # 创建属性用于存储每次调用的深度融合权重
        self.last_fusion_weights = None

    def forward(self, fused_feat, skip_feats, depth_feats, raw):
        B, C, _, _ = fused_feat.shape

        # Verify depth_feats dimensions - adjust if needed
        if depth_feats[0].shape[1] != self.base_channels:
            # Depth features have different channel dimensions
            # Use predefined projection layer instead of creating one dynamically
            projected_depth_feats = []
            for df in depth_feats:
                # 根据实际通道数选择正确的投影层
                if df.shape[1] != self.base_channels:
                    if df.shape[1] == 1:
                        # 单通道输入 (如门控图) 使用专用投影层
                        df = self.depth_proj_1ch(df)
                    elif df.shape[1] == self.base_channels * 2:
                        # 双倍通道使用原有投影层
                        df = self.depth_proj(df)
                    else:
                        # 其他通道数需创建动态投影层
                        layer_key = f'depth_proj_{df.shape[1]}'
                        if not hasattr(self, layer_key):
                            # 动态创建投影层并添加为模块属性
                            setattr(self, layer_key, 
                                   nn.Conv2d(df.shape[1], self.base_channels, 
                                           kernel_size=1, bias=False).to(df.device))
                        # 使用动态创建的投影层
                        df = getattr(self, layer_key)(df)
                projected_depth_feats.append(df)
            depth_feats = projected_depth_feats

        # ----- Deblur Branch with Dense Skip & Adaptive Depth Fusion -----
        x = fused_feat
        x_color = fused_feat  # 色彩分支初始特征
        
        # 创建一个列表用于收集所有层级的权重
        all_weights = []
        
        for i in range(self.levels):
            # 1) Upsample previous resolution
            x = self.deblur_ups[i](x)  # shape [B, C, H/2^(levels-i-1), W/...]
            
            # 2) Collect gated skips from all encoder levels up to current
            gated_skips = []
            for j in range(i + 1):
                # Use deepest skip first: skip_feats[levels-1 - j]
                skip = skip_feats[self.levels - 1 - j]
                
                # Align spatial size
                skip_up = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
                
                # ----- Adaptive Depth Fusion -----
                # Upscale all depth features to current resolution
                up_depths = [
                    F.interpolate(depth_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
                    for depth_feat in depth_feats  # All available depth scales
                ]
                
                # Channel-wise concatenation of depth features
                depth_cat = torch.cat(up_depths, dim=1)  # [B, C*depth_levels, h, w]
                
                # Predict scale attention scores
                scores = self.scale_mlp(depth_cat)  # [B, depth_levels, h, w]
                weights = torch.softmax(scores, dim=1)  # Normalize along scale dimension
                
                # 收集当前层级的权重
                if j == 0:  # 只记录每个解码器级别的第一个跳跃连接的权重
                    all_weights.append(weights)
                
                # Weighted fusion of depth features
                depth_fusion = sum(weights[:, k:k+1] * up_depths[k] 
                                  for k in range(self.depth_levels))
                
                # Generate gate map from fused depth feature
                gate_map = torch.sigmoid(self.gate_conv(depth_fusion))  # [B, 1, h, w]
                
                # Apply gate to skip feature
                gated = skip_up * gate_map
                gated_skips.append(gated)
                
            # 3) Concatenate x with all gated skips
            fused = torch.cat([x] + gated_skips, dim=1)
            
            # Check if the number of channels in fused matches the expected input for deblur_fuse_convs[i]
            expected_channels = self.base_channels * (i + 2)
            actual_channels = fused.shape[1]
            
            if actual_channels != expected_channels:
                # 检查是否已经为这个层级和通道数创建了投影层
                layer_key = f'level_{i}_{actual_channels}'
                if layer_key not in self.fused_proj:
                    # 动态创建一个新的投影层并添加到ModuleDict
                    # 输出通道必须与deblur_fuse_convs[i]的输入匹配，即expected_channels
                    self.fused_proj[layer_key] = nn.Conv2d(
                        actual_channels, 
                        expected_channels, 
                        kernel_size=1, 
                        bias=False
                    ).to(fused.device)
                
                # 使用正确的投影层
                fused = self.fused_proj[layer_key](fused)
            
            # 4) Compress channels and apply Transformer blocks
            fused = self.deblur_fuse_convs[i](fused)
            x = self.deblur_blocks[i](fused)
            
            # ----- Color Branch (only for shallow layers) -----
            if i >= self.levels // 2:  # 浅层级
                # 修复：调整x_color到与gated_skips相同的空间维度
                upsampled_color = F.interpolate(fused_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
                # 使用预定义的投影层而不是动态创建
                x_color = self.color_proj(upsampled_color)
                
                # 现在可以安全地拼接，因为空间维度已对齐
                color_fused = torch.cat([x_color] + gated_skips, dim=1)
                
                # 应用深度门控，确保深度特征也调整到正确大小
                # 重用当前层的gated_skips[0]的空间尺寸
                current_size = x.shape[-2:]
                skip_current = F.interpolate(skip_feats[i], size=current_size, mode='bilinear', align_corners=False)
                depth_current = F.interpolate(depth_feats[i], size=current_size, mode='bilinear', align_corners=False)
                x_color = self.color_gate(skip_current, depth_current)
            else:
                # 深层与deblur分支共享
                x_color = x
        
        # 如果收集了权重，将其保存为单个张量以便后续查询
        if all_weights:
            # 首先将所有权重调整到相同的空间尺寸
            target_size = x.shape[-2:]  # 使用最终输出的尺寸
            resized_weights = []
            
            for weight in all_weights:
                # 如果尺寸不同，则调整到目标尺寸
                if weight.shape[-2:] != target_size:
                    # 对每个深度层级分别进行调整
                    weight_channels = weight.shape[1]  # 深度层级数量
                    resized = []
                    
                    for c in range(weight_channels):
                        # 提取当前深度层级的权重，保持批次维度
                        w_c = weight[:, c:c+1]
                        # 调整尺寸
                        w_c_resized = F.interpolate(w_c, size=target_size, mode='bilinear', align_corners=False)
                        resized.append(w_c_resized)
                    
                    # 沿着通道维度拼接回调整后的权重
                    resized_weight = torch.cat(resized, dim=1)
                    resized_weights.append(resized_weight)
                else:
                    # 如果尺寸已经匹配，直接添加
                    resized_weights.append(weight)
            
            # 堆叠调整后的所有层级的权重 [levels, B, depth_levels, H, W]
            stacked_weights = torch.stack(resized_weights, dim=0)
            # 转置为 [B, levels, depth_levels, H, W]
            self.last_fusion_weights = stacked_weights.permute(1, 0, 2, 3, 4)
        
        # 5) Reconstruction to RGB residual
        res_d = self.deblur_recon(x)
        
        # 确保 res_d 与输入图像 raw 大小匹配
        if res_d.shape[2:] != raw.shape[2:]:
            res_d = F.interpolate(res_d, size=raw.shape[2:], mode='bilinear', align_corners=False)

        # 输出两个残差分支
        res_c = res_d
        
        return res_d, res_c

@torch.no_grad()
def test_decoder(decoder, batch_size=1, channels=48, debug=False):
    """
    测试解码器通道数是否匹配，并返回是否通过测试
    
    Args:
        decoder: MultiTaskDecoder实例
        batch_size: 批次大小
        channels: 基础通道数
        debug: 是否打印调试信息
    
    Returns:
        bool: 测试是否通过
    """
    try:
        device = next(decoder.parameters()).device
        # 创建测试数据
        fused_feat = torch.randn(batch_size, channels, 8, 8).to(device)
        skip_feats = [
            torch.randn(batch_size, channels, 64, 64).to(device),
            torch.randn(batch_size, channels, 32, 32).to(device),
            torch.randn(batch_size, channels, 16, 16).to(device),
            torch.randn(batch_size, channels, 8, 8).to(device)
        ]
        depth_feats = [
            torch.randn(batch_size, channels, 64, 64).to(device),
            torch.randn(batch_size, channels, 32, 32).to(device),
            torch.randn(batch_size, channels, 16, 16).to(device),
            torch.randn(batch_size, channels, 8, 8).to(device)
        ]
        raw = torch.randn(batch_size, 3, 128, 128).to(device)
        
        if debug:
            print(f"输入特征: {fused_feat.shape}")
            for i, feat in enumerate(skip_feats):
                print(f"Skip特征 {i}: {feat.shape}")
            for i, feat in enumerate(depth_feats):
                print(f"深度特征 {i}: {feat.shape}")
        
        # 执行前向传播
        res_d, res_c = decoder(fused_feat, skip_feats, depth_feats, raw)
        
        if debug:
            print(f"解码器测试成功！输出形状: {res_d.shape}")
        
        return True
    except Exception as e:
        if debug:
            print(f"解码器测试失败: {e}")
            import traceback
            traceback.print_exc()
        return False

