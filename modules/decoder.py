import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import RestormerBlock, DACBlock
from .blocks import BeerLambertPML, PSFPML
import logging
# import kornia.filters as K  # 用于高斯卷积（当前未使用）

logger = logging.getLogger(__name__)

class MultiTaskDecoder(nn.Module):
    """
    Multi-task Decoder for Deblur and Color Correction with Dense Skip and Adaptive Depth Fusion
    -------------------------------------------------------------------------------------------
    Inputs:
      - fused_feat: Tensor[B, C_bottleneck, H_bottleneck, W_bottleneck] (Joint Bottleneck output)
      - skip_feats: list of RGB encoder features [F0_raw, F1_raw, ..., Fn_raw]
                    Expected to have self.levels + 1 features, with varying channel sizes.
      - depth_feats: list of Depth features [F0_dep, F1_dep, ..., Fn_dep]
                     Expected to have self.levels + 1 features, with varying channel sizes.
      - raw: Tensor[B, 3, H, W]             (Original image for Color branch)

    Outputs:
      - res_d: Tensor[B, 3, H, W]  Deblur residual
      - res_c: Tensor[B, 3, H, W]  Color correction residual
    """
    def __init__(self, base_channels=48, input_channels_bottleneck=384, levels=3, decoder_block_window_size=8, num_encoder_feature_levels=None,
                 depth_raw_min: float = 2000.0, depth_raw_max: float = 65535.0,
                 depth_meter_min: float = 0.1, depth_meter_max: float = 30.0):
        super().__init__()
        self.levels = levels
        self.base_channels = base_channels
        self.input_channels_bottleneck = input_channels_bottleneck

        if num_encoder_feature_levels is None:
            self.num_encoder_feature_levels = levels + 1 
        else:
            self.num_encoder_feature_levels = num_encoder_feature_levels

        logger.info(f"MultiTaskDecoder initialized with {levels} decoder levels, {self.num_encoder_feature_levels} expected encoder feature levels (for depth fusion weights).")
        
        # 深度标定参数（像素→米）
        self.depth_raw_min = float(depth_raw_min)
        self.depth_raw_max = float(depth_raw_max)
        self.depth_meter_min = float(depth_meter_min)
        self.depth_meter_max = float(depth_meter_max)

        # Deblur Branch
        self.deblur_ups = nn.ModuleList()#上采样模块
        self.deblur_blocks = nn.ModuleList() # Initialize as empty list first，transformerblock finally
        self.deblur_skips = nn.ModuleList() #跳跃连接
        self.deblur_fuse_convs = nn.ModuleList() # Moved initialization up#融合卷积

        current_channels_deblur = input_channels_bottleneck
        for i in range(levels):
            # 计算当前层输出通道数
            out_c_deblur = self.base_channels * (2**(levels - 1 - i)) if levels > 1 else self.base_channels
            # Upsampling for deblur branch ，1*1卷积从当前通道数到4倍，然后pixelshuffle上采样
            self.deblur_ups.append(
                nn.Sequential(
                    nn.Conv2d(current_channels_deblur, out_c_deblur * 4, kernel_size=1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            # Skip connection conv for deblur branch，将编码器和解码器对应层特征与上采样特征concatenate后再用1*1卷积
            encoder_skip_channels_deblur = self.base_channels * (2**(self.num_encoder_feature_levels - 1 - i)) if self.num_encoder_feature_levels > 1 else self.base_channels
            self.deblur_skips.append(
                 nn.Conv2d(encoder_skip_channels_deblur + out_c_deblur, out_c_deblur, kernel_size=1, bias=True)
            )
            # Fuse conv for deblur branch，正确计算输入通道数：out_c_deblur + (i+1)*base_channels
            # i+1 表示当前层会使用前 i+1 个skip特征
            expected_skip_channels = (i + 1) * self.base_channels
            expected_fuse_input_channels = out_c_deblur + expected_skip_channels
            self.deblur_fuse_convs.append(
                 nn.Conv2d(expected_fuse_input_channels, out_c_deblur, kernel_size=1, bias=False)
            )
            # RestormerBlock for deblur branch，输入通道数为out_c_deblur，输出通道数为out_c_deblur，head数为8，窗口大小为decoder_block_window_size
            self.deblur_blocks.append(RestormerBlock(out_c_deblur, heads=8, window_size=decoder_block_window_size)) # Corrected call
            current_channels_deblur = out_c_deblur#更新当前通道数
        
        self.deblur_recon = nn.Conv2d(current_channels_deblur, 3, kernel_size=3, padding=1)#最后用1*3卷积输出3通道，将顶层特征映射到3通道RGB残差

        # 深度融合已被禁用，删除未使用的组件
        
        # 添加DAC块到颜色分支
        self.dac_blocks = nn.ModuleList([
            DACBlock(self.base_channels * (2**(levels - 1 - i)) if levels > 1 else self.base_channels)
            for i in range(levels)
        ])
        
        # Color Branch
        self.color_ups = nn.ModuleList() # Initialize as empty list
        self.color_blocks = nn.ModuleList() # Initialize as empty list
        self.color_skips = nn.ModuleList()
        self.color_fuse_convs = nn.ModuleList() # Added fuse_convs for color branch consistency

        current_channels_color = input_channels_bottleneck
        for i in range(levels):
            out_c_color = self.base_channels * (2**(levels - 1 - i)) if levels > 1 else self.base_channels
            # Upsampling for color branch (example from a more complete version)
            self.color_ups.append(
            nn.Sequential(
                    nn.Conv2d(current_channels_color, out_c_color * 4, kernel_size=1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            # Skip connection conv for color branch
            encoder_skip_channels_color = self.base_channels * (2**(self.num_encoder_feature_levels - 1 - i)) if self.num_encoder_feature_levels > 1 else self.base_channels
            self.color_skips.append(
                 nn.Conv2d(encoder_skip_channels_color + out_c_color, out_c_color, kernel_size=1, bias=True)
            )
            # Fuse conv for color branch，正确计算输入通道数：out_c_color + (i+1)*base_channels
            expected_skip_channels_color = (i + 1) * self.base_channels
            expected_fuse_input_channels_color = out_c_color + expected_skip_channels_color
            self.color_fuse_convs.append(
                 nn.Conv2d(expected_fuse_input_channels_color, out_c_color, kernel_size=1, bias=False)
            )
            # RestormerBlock for color branch，无dac时才使用这个
            self.color_blocks.append(RestormerBlock(out_c_color, heads=8, window_size=decoder_block_window_size)) # Corrected call
            current_channels_color = out_c_color
            
        self.color_recon = nn.Conv2d(current_channels_color, 3, kernel_size=3, padding=1) # Recon for color branch
        
        # 添加融合模块：将J_D和I_A融合为最终输出
        self.fuse_conv = nn.Conv2d(6, 3, kernel_size=1)  # 1x1卷积融合两个分支的结果
        
        # Global components for color correction (if needed, from a different design pattern)，全局颜色校正辅助，投影字典初始化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.base_channels, self.base_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.base_channels, self.base_channels * 2)
        )
        self.color_proj_bottleneck = nn.Conv2d(self.input_channels_bottleneck, self.base_channels, kernel_size=1, bias=False) # Define color_proj_bottleneck
        
        # 🔥 修复显存跳变：在__init__中预先创建所有可能的投影层
        # 扩展的通道配置，覆盖更多可能的组合
        common_channel_configs = []
        
        # 基于base_channels动态生成常见配置
        base_variants = [base_channels, base_channels*2, base_channels*4, base_channels*8]
        for in_ch in base_variants:
            for out_ch in base_variants:
                common_channel_configs.append((in_ch, out_ch))
        
        # 添加默认配置以保证兼容性
        default_configs = [
            # 基础配置
            (48, 48), (96, 48), (192, 48), (384, 48),
            # 额外的常见配置
            (32, 48), (64, 48), (128, 48), (256, 48), (512, 48),
            # 反向配置（如果需要）
            (48, 32), (48, 64), (48, 96), (48, 128), (48, 192), (48, 256),
            # 常见的2的幂次组合
            (16, 16), (16, 32), (16, 64), (16, 128),
            (32, 16), (32, 32), (32, 64), (32, 128),
            (64, 16), (64, 32), (64, 64), (64, 128),
            (128, 16), (128, 32), (128, 64), (128, 128),
            # 其他可能的组合
            (24, 48), (72, 48), (144, 48), (288, 48), (576, 48),
            # 2的幂次配置
            (16, 48), (1024, 48),
            # 特殊情况
            (1, 48), (3, 48), (6, 48), (12, 48),
        ]
        
        common_channel_configs.extend(default_configs)
        
        self.skip_feature_projections = nn.ModuleDict()
        self.depth_channel_projections = nn.ModuleDict()
        
        # 预先创建常用的投影层
        for in_ch, out_ch in common_channel_configs:
            layer_key = f"{in_ch}_to_{out_ch}"
            # Skip feature projections
            if layer_key not in self.skip_feature_projections:
                self.skip_feature_projections[layer_key] = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            # Depth channel projections  
            if layer_key not in self.depth_channel_projections:
                self.depth_channel_projections[layer_key] = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        
        # 🔥 添加物理调制层
        # Color branch: BeerLambertPML for each level
        self.color_pmls = nn.ModuleList()
        # Deblur branch: PSFPML for each level  
        self.deblur_pmls = nn.ModuleList()
        
        for i in range(levels):
            # 计算每层的通道数
            out_c_color = self.base_channels * (2**(levels - 1 - i)) if levels > 1 else self.base_channels
            out_c_deblur = self.base_channels * (2**(levels - 1 - i)) if levels > 1 else self.base_channels
            
            # 为每层添加对应的PML
            self.color_pmls.append(BeerLambertPML(out_c_color))
            self.deblur_pmls.append(PSFPML(out_c_deblur))
        
        logger.info(f"Initialized MultiTaskDecoder. Pre-created projection layers for common channel configurations.")
        logger.info(f"Added {levels} BeerLambertPML and {levels} PSFPML layers for physics modulation.")
        self.last_fusion_weights = None

    def _get_projection_layer(self, proj_dict: nn.ModuleDict, in_channels: int, out_channels: int, device: torch.device):
        """Helper to get projection layer. If not exists, raise error instead of creating dynamically."""
        layer_key = f"{in_channels}_to_{out_channels}"
        if layer_key not in proj_dict:
            # 🔥 修复显存跳变：不再动态创建层！
            # 记录错误并使用1x1恒等映射作为应急方案
            logger.error(f"❌ Missing projection layer for {in_channels} -> {out_channels} channels! "
                        f"This indicates a channel configuration not covered in __init__. "
                        f"Available keys: {list(proj_dict.keys())}")
            
            # 应急处理：如果输入输出通道数相同，返回恒等映射
            if in_channels == out_channels:
                logger.warning(f"⚠️ Using identity mapping for {in_channels} == {out_channels} channels")
                if not hasattr(self, '_identity_layer'):
                    self._identity_layer = nn.Identity()
                return self._identity_layer
            else:
                # 如果通道数不同，这是一个严重错误，应该停止训练
                raise RuntimeError(f"🚨 Channel mismatch: {in_channels} -> {out_channels}. "
                                 f"Please add this configuration to __init__ common_channel_configs. "
                                 f"Dynamic layer creation has been disabled to fix memory jumps.")
        return proj_dict[layer_key]

    def forward(self, fused_feat, skip_feats, depth_feats, raw, 
                depth_pred=None, beta_c=None, B_c=None, blur_scale=None, enhanced_depth_feats=None):
        """
        前向传播
        
        Args:
            fused_feat: 瓶颈层融合特征 [B, input_channels_bottleneck, H_bot, W_bot]
            skip_feats: 编码器跳连特征列表，从高分辨率到低分辨率
            depth_feats: 深度分支特征列表 (原始depth特征)
            raw: 原始输入图像 [B, 3, H, W]（用于计算残差）
            depth_pred: 预测深度图 [B, 1, H, W]（用于物理调制）
            beta_c: 消光系数 [B, 3, 1, 1]（用于Beer-Lambert调制）
            B_c: 背景光 [B, 3, 1, 1]（用于Beer-Lambert调制） 
            blur_scale: 模糊尺度 [B, 1, 1, 1]（用于PSF调制）
            enhanced_depth_feats: 增强深度特征列表 (可选，优先使用)
            
        Returns:
            res_d: 去模糊残差 [B, 3, H, W]
            res_c: 颜色校正残差 [B, 3, H, W]
        """
        logger = logging.getLogger('decoder')
        current_device = fused_feat.device
        
        # 🔥 决定使用哪种深度特征：优先使用增强深度特征
        actual_depth_feats = enhanced_depth_feats if enhanced_depth_feats is not None else depth_feats
        if enhanced_depth_feats is not None:
            logger.debug(f"Decoder: Using enhanced depth features ({len(enhanced_depth_feats)} levels)")
        elif depth_feats is not None:
            logger.debug(f"Decoder: Using original depth features ({len(depth_feats)} levels)")
        else:
            logger.debug("Decoder: No depth features available")
        
        # 跳连数量自动适配
        if len(skip_feats) != self.num_encoder_feature_levels:
            # 自动调整跳连特征数量以匹配期望
            if len(skip_feats) > self.num_encoder_feature_levels:
                # 截断多余的特征
                skip_feats = skip_feats[:self.num_encoder_feature_levels]
            else:
                # 补足缺少的特征（复制最后一个特征）
                while len(skip_feats) < self.num_encoder_feature_levels:
                    skip_feats.append(skip_feats[-1])
        
        # 深度特征数量自动适配（如果提供）
        if actual_depth_feats is not None and len(actual_depth_feats) != len(skip_feats):
            # 自动调整深度特征数量以匹配跳连特征
            if len(actual_depth_feats) > len(skip_feats):
                actual_depth_feats = actual_depth_feats[:len(skip_feats)]
            else:
                while len(actual_depth_feats) < len(skip_feats):
                    actual_depth_feats.append(actual_depth_feats[-1])
            
        # ----- 去模糊分支 -----
        x = fused_feat  # 从瓶颈层开始
        
        for i in range(self.levels):
            x = self.deblur_ups[i](x)
            gated_skips = []
            
            num_skips_to_use_at_level_i = i + 1
            for k_skip_idx in range(num_skips_to_use_at_level_i):
                if k_skip_idx >= len(skip_feats):
                    logger.warning(f"尝试访问索引为 {k_skip_idx} 的skip_feat，但解码层 {i} 只有 {len(skip_feats)} 个可用。")
                    continue 
                
                skip_raw = skip_feats[k_skip_idx]
                if skip_raw is None:
                    logger.warning(f"在索引 {k_skip_idx} 处发现空的skip_feat，解码层 {i}。")
                    continue

                skip_up = F.interpolate(skip_raw, size=x.shape[-2:], mode='bilinear', align_corners=False)
                
                # 确保所有skip特征都被投影到base_channels，避免通道数不匹配
                if skip_up.shape[1] != self.base_channels:
                    proj_layer = self._get_projection_layer(self.skip_feature_projections, skip_up.shape[1], self.base_channels, current_device)
                    try:
                        projected_skip = proj_layer(skip_up)
                    except Exception as e:
                        logger.error(f"投影skip_feat时出错 (idx {k_skip_idx}): {e}。使用原始特征。")
                        projected_skip = skip_up
                else:
                    projected_skip = skip_up
                
                # 由于深度融合已被禁用，直接使用skip特征
                gated = projected_skip
                gated_skips.append(gated)
                
            fused = torch.cat([x] + gated_skips, dim=1)
            
            # 由于我们已经正确计算了fusion conv的输入通道数，不再需要应急投影
            if fused.shape[1] != self.deblur_fuse_convs[i].in_channels:
                logger.error(
                    f"解码层 {i}: 意外的通道不匹配！融合特征: {fused.shape[1]}, 预期: {self.deblur_fuse_convs[i].in_channels}。"
                    f"提供的跳连数: {len(gated_skips)} (预期 {i+1})。这在新的通道计算方式下不应该发生。"
                )
                raise RuntimeError(f"通道匹配失败：级别 {i}，融合特征通道数 {fused.shape[1]} != 预期通道数 {self.deblur_fuse_convs[i].in_channels}")
            
            logger.debug(f"Decoder level {i}: About to call deblur_fuse_convs[{i}] (expected in: {self.deblur_fuse_convs[i].in_channels}) with fused shape {fused.shape}")
            fused = self.deblur_fuse_convs[i](fused)#降通道
            
            # 🔥 插入PSF物理调制层 (PSF-PML)
            if depth_pred is not None and blur_scale is not None:
                # 计算归一化深度 
                d_min = torch.amin(depth_pred, dim=(2, 3), keepdim=True)  # [B, 1, 1, 1]
                d_max = torch.amax(depth_pred, dim=(2, 3), keepdim=True)  # [B, 1, 1, 1]
                depth_norm = (depth_pred - d_min) / (d_max - d_min + 1e-6)  # [B, 1, H, W]
                depth_norm = torch.clamp(depth_norm, 0.0, 1.0)
                
                # 上采样深度到当前特征尺寸
                depth_norm_resized = F.interpolate(depth_norm, size=fused.shape[-2:], 
                                                 mode='bilinear', align_corners=False)
                
                # 应用PSF物理调制
                fused = self.deblur_pmls[i](fused, depth_norm_resized, blur_scale)
                
            x = self.deblur_blocks[i](fused)#restormer
            
        res_d = self.deblur_recon(x)
        
        # ----- 颜色分支 -----
        x_color = fused_feat
        
        for i in range(self.levels):
            x_color = self.color_ups[i](x_color)
            gated_skips_color = []
            
            num_skips_to_use_at_level_i = i + 1
            for k_skip_idx in range(num_skips_to_use_at_level_i):
                if k_skip_idx >= len(skip_feats):
                    logger.warning(f"颜色分支: 尝试访问索引为 {k_skip_idx} 的skip_feat，但解码层 {i} 只有 {len(skip_feats)} 个可用。")
                    continue 
                
                skip_raw = skip_feats[k_skip_idx]
                if skip_raw is None:
                    logger.warning(f"颜色分支: 在索引 {k_skip_idx} 处发现空的skip_feat，解码层 {i}。")
                    continue

                skip_up = F.interpolate(skip_raw, size=x_color.shape[-2:], mode='bilinear', align_corners=False)
                
                # 确保所有skip特征都被投影到base_channels，避免通道数不匹配
                if skip_up.shape[1] != self.base_channels:
                    proj_layer = self._get_projection_layer(self.skip_feature_projections, skip_up.shape[1], self.base_channels, current_device)
                    try:
                        projected_skip = proj_layer(skip_up)
                    except Exception as e:
                        logger.error(f"颜色分支: 投影skip_feat时出错 (idx {k_skip_idx}): {e}。使用原始特征。")
                        projected_skip = skip_up
                else:
                    projected_skip = skip_up
                
                # 由于深度融合已被禁用，直接使用skip特征
                gated = projected_skip
                gated_skips_color.append(gated)
                
            fused = torch.cat([x_color] + gated_skips_color, dim=1)
            
            # 由于我们已经正确计算了fusion conv的输入通道数，不再需要应急投影
            if fused.shape[1] != self.color_fuse_convs[i].in_channels:
                logger.error(
                    f"颜色分支: 解码层 {i}: 意外的通道不匹配！融合特征: {fused.shape[1]}, 预期: {self.color_fuse_convs[i].in_channels}。"
                    f"提供的跳连数: {len(gated_skips_color)} (预期 {i+1})。这在新的通道计算方式下不应该发生。"
                )
                raise RuntimeError(f"颜色分支通道匹配失败：级别 {i}，融合特征通道数 {fused.shape[1]} != 预期通道数 {self.color_fuse_convs[i].in_channels}")
            
            logger.debug(f"颜色分支: 解码层 {i}: 调用color_fuse_convs[{i}] (预期输入: {self.color_fuse_convs[i].in_channels}) 融合特征形状 {fused.shape}")
            fused = self.color_fuse_convs[i](fused)
            
            # 🔥 插入Beer-Lambert物理调制层 (BL-PML)
            if depth_pred is not None and beta_c is not None and B_c is not None:
                # 上采样深度和原始图像到当前特征尺寸
                depth_resized = F.interpolate(depth_pred, size=fused.shape[-2:], 
                                            mode='bilinear', align_corners=False)
                raw_resized = F.interpolate(raw, size=fused.shape[-2:], 
                                          mode='bilinear', align_corners=False)
                
                # 3️⃣ 深度标定（像素→米），再传递给Beer-Lambert PML
                # 将原始像素值范围 [depth_raw_min, depth_raw_max] 映射到物理米制 [depth_meter_min, depth_meter_max]
                depth_meters = (depth_resized - self.depth_raw_min) / (self.depth_raw_max - self.depth_raw_min + 1e-6)
                depth_meters = torch.clamp(depth_meters, 0.0, 1.0)
                depth_meters = depth_meters * (self.depth_meter_max - self.depth_meter_min) + self.depth_meter_min

                # 原图从 [-1,1] → [0,1]，以匹配 BL-PML 内部的 I_tilde = 2*I_raw - 1
                raw_unit = (raw_resized + 1.0) / 2.0

                fused = self.color_pmls[i](fused, raw_unit, depth_meters, beta_c, B_c)
            
            # 通过DACBlock，不再应用深度加权
            if i < len(self.dac_blocks):
                # 不再需要深度权重计算
                x_color = self.dac_blocks[i](fused)
            else:
                # 如果没有对应的DACBlock，则使用RestormerBlock
                x_color = self.color_blocks[i](fused)
                
        # 使用颜色重建头生成残差
        res_c = self.color_recon(x_color)

        # 深度融合已禁用，不再记录权重
        self.last_fusion_weights = None
            
        # 物理模型计算已删除
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
        # 创建测试数据 - fused_feat应该使用input_channels_bottleneck
        fused_feat = torch.randn(batch_size, decoder.input_channels_bottleneck, 8, 8).to(device)
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
        raw = torch.randn(batch_size, 3, 64, 64).to(device)  # 匹配解码器输出尺寸
        
        if debug:
            print(f"输入特征: {fused_feat.shape}")
            for i, feat in enumerate(skip_feats):
                print(f"Skip特征 {i}: {feat.shape}")
            for i, feat in enumerate(depth_feats):
                print(f"深度特征 {i}: {feat.shape}")
        
        # 执行前向传播
        result = decoder(
            fused_feat, 
            skip_feats, 
            depth_feats, 
            raw
        )
        
        if len(result) == 2:
            res_d, res_c = result
        else:
            res_d, res_c = result
        
        if debug:
            print(f"解码器测试成功！输出形状: {res_d.shape}")
        
        return True
    except Exception as e:
        if debug:
            print(f"解码器测试失败: {e}")
            import traceback
            traceback.print_exc()
        return False

