import torch
import torch.nn as nn
import torch.nn.functional as F
from .depth import DepthGate
from .blocks import RestormerBlock, DACBlock
import logging
import kornia.filters as K  # 用于高斯卷积

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
    def __init__(self, base_channels=48, input_channels_bottleneck=384, levels=3, decoder_block_window_size=8, num_encoder_feature_levels=None):
        super().__init__()
        self.levels = levels
        self.base_channels = base_channels
        self.input_channels_bottleneck = input_channels_bottleneck

        if num_encoder_feature_levels is None:
            self.num_encoder_feature_levels = levels + 1 
        else:
            self.num_encoder_feature_levels = num_encoder_feature_levels

        logger.info(f"MultiTaskDecoder initialized with {levels} decoder levels, {self.num_encoder_feature_levels} expected encoder feature levels (for depth fusion weights).")

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
            # Fuse conv for deblur branch，将跳跃连接和上采样特征concatenate后再用1*1卷积，再降回 out_c_deblur
            self.deblur_fuse_convs.append(
                 nn.Conv2d(out_c_deblur * 2, out_c_deblur, kernel_size=1, bias=False) # Assuming skip and upsample result in out_c_deblur each
            )
            # RestormerBlock for deblur branch，输入通道数为out_c_deblur，输出通道数为out_c_deblur，head数为8，窗口大小为decoder_block_window_size
            self.deblur_blocks.append(RestormerBlock(out_c_deblur, heads=8, window_size=decoder_block_window_size)) # Corrected call
            current_channels_deblur = out_c_deblur#更新当前通道数
        
        self.deblur_recon = nn.Conv2d(current_channels_deblur, 3, kernel_size=3, padding=1)#最后用1*3卷积输出3通道，将顶层特征映射到3通道RGB残差

        # Adaptive Depth Fusion Components (ensure these are correctly placed and initialized)
        self.scale_mlp = nn.Sequential(
            nn.Conv2d(base_channels * self.num_encoder_feature_levels, base_channels, kernel_size=1, bias=False),#将编码器特征通道数*num_encoder_feature_levels，然后1*1卷积到base_channels
            nn.GELU(),#激活函数
            nn.Conv2d(base_channels, self.num_encoder_feature_levels, kernel_size=1)#将base_channels卷积到num_encoder_feature_levels
        )
        self.gate_conv = nn.Conv2d(base_channels, 1, kernel_size=1, bias=False)#将base_channels卷积到1通道，用于深度融合权重
        #相当于对每个像素点产生一个长度为 num_encoder_feature_levels 的向量，代表"对不同尺度深度特征"的加权系数（score）。后续会对这组 score 做 softmax，得到权重分布。#全连接
        # 为颜色分支添加深度加权组件
        self.scale_mlp_color = nn.Sequential(
            nn.Conv2d(base_channels * self.num_encoder_feature_levels, base_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(base_channels, self.num_encoder_feature_levels, kernel_size=1)
        )
        self.gate_conv_color = nn.Conv2d(base_channels, 1, kernel_size=1, bias=False)
        
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
            # Fuse conv for color branch
            self.color_fuse_convs.append(
                 nn.Conv2d(out_c_color * 2, out_c_color, kernel_size=1, bias=False) # Assuming skip and upsample result in out_c_color each
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
        
        self.skip_feature_projections = nn.ModuleDict()
        self.depth_channel_projections = nn.ModuleDict()
        logger.info(f"Initialized MultiTaskDecoder. Projection layers for skip/depth will be lazily created.")
        self.last_fusion_weights = None
    def _get_projection_layer(self, proj_dict: nn.ModuleDict, in_channels: int, out_channels: int, device: torch.device):
        """Helper to get or create a LazyConv2d projection layer."""#跳连时深度通道与解码器预期不一致，需要1*1卷积投影
        layer_key = f"{in_channels}_to_{out_channels}" # Key based on in_channels and out_channels，无需指定输入通道数，知道第一次传入真实tensor时次啊根据输入通道完成权重初始化
        if layer_key not in proj_dict:
            # We specify out_channels, kernel_size, and bias.
            lazy_layer = nn.LazyConv2d(out_channels, kernel_size=1, bias=False)
            lazy_layer.to(device)  # Move to the target device immediately after creation
            # Add to ModuleDict. It will be properly registered and moved to device with the parent module.
            proj_dict[layer_key] = lazy_layer
            logger.info(f"Created lazy projection layer for {in_channels} -> {out_channels} channels (key: {layer_key}) in {proj_dict._get_name()}")
        return proj_dict[layer_key]

    def forward(self, fused_feat, skip_feats, depth_feats, raw, depth_pred=None, beta_c=None, B_c=None, blur_scale=None):
        """
        MultiTaskDecoder forward with A/D physical models
        
        Args:
            fused_feat: Tensor[B, C_bottleneck, H_bottleneck, W_bottleneck] (Joint Bottleneck output)
            skip_feats: list of RGB encoder features [F0_raw, F1_raw, ..., Fn_raw]
            depth_feats: list of Depth features [F0_dep, F1_dep, ..., Fn_dep]
            raw: Tensor[B, 3, H, W] (Original image) 
            depth_pred: Tensor[B, 1, H, W] (Predicted depth map)
            beta_c: Parameter for Beer-Lambert attenuation coefficient [1, 3, 1, 1]
            B_c: Global background light [1, 3, 1, 1]
            blur_scale: Scalar for depth PSF
        
        Returns:
            res_d: Tensor[B, 3, H, W] (Deblur residual)
            res_c: Tensor[B, 3, H, W] (Color correction residual)
        """
        B, C_bottleneck, H_bottleneck, W_bottleneck = fused_feat.shape #最小分辨率瓶颈特征
        current_device = fused_feat.device

        # --- Project depth_feats to self.base_channels if necessary (using LazyConv2d) ---
        processed_depth_feats = []#深度特征预处理
        if depth_feats is not None:
            # Pad/truncate if adaptive depth fusion relies on exact number.
            # The scale_mlp expects input from self.num_encoder_feature_levels features.
            if len(depth_feats) != self.num_encoder_feature_levels:
                logger.warning(f"MultiTaskDecoder: Expected {self.num_encoder_feature_levels} depth_feats, got {len(depth_feats)}. Adjusting for scale_mlp.")
                temp_depth_feats = list(depth_feats)
                if len(temp_depth_feats) > self.num_encoder_feature_levels:
                    depth_feats_for_processing = temp_depth_feats[:self.num_encoder_feature_levels]
                else:
                    depth_feats_for_processing = temp_depth_feats
                    if temp_depth_feats: # Check if not empty before trying to access last element
                        while len(depth_feats_for_processing) < self.num_encoder_feature_levels:
                            depth_feats_for_processing.append(temp_depth_feats[-1]) # Pad with last valid feature
                    else: # If temp_depth_feats was empty, pad with Nones
                        while len(depth_feats_for_processing) < self.num_encoder_feature_levels:
                            depth_feats_for_processing.append(None)
            else:
                depth_feats_for_processing = depth_feats#对深度特征进行尺寸裁剪

            for df_idx, df in enumerate(depth_feats_for_processing):#将深度特征投影
                if df is None:
                    processed_depth_feats.append(None)
                    # logger.warning(f"Encountered None in depth_feats at index {df_idx} during projection.")
                    continue
                
                if df.shape[1] != self.base_channels:
                    proj_layer = self._get_projection_layer(self.depth_channel_projections, df.shape[1], self.base_channels, current_device)
                    try:
                        df_float = df.float() # Ensure df is float32 before potentially disabling autocast
                        # Temporarily disable autocast for projection operation (numerical stability)
                        with torch.amp.autocast(device_type='cuda', enabled=False):
                            projected_df = proj_layer(df_float) # Pass float32 df_float to proj_layer
                        processed_depth_feats.append(projected_df)
                    except Exception as e:
                        logger.error(f"Error projecting depth_feat (idx {df_idx}) from {df.shape[1]} to {self.base_channels} chans: {e}. Appending original.")
                        processed_depth_feats.append(df.to(current_device))
                else:
                    processed_depth_feats.append(df)
        else: # depth_feats is None
            # Fill processed_depth_feats with Nones to match num_encoder_feature_levels for scale_mlp padding logic later
            processed_depth_feats = [None] * self.num_encoder_feature_levels

        # ----- Deblur Branch with Dense Skip & Adaptive Depth Fusion -----
        x = fused_feat
        all_weights_for_logging = []
        
        for i in range(self.levels):
            x = self.deblur_ups[i](x)#上采
            gated_skips = []
            
            num_skips_to_use_at_level_i = i + 1#跳连个数
            for k_skip_idx in range(num_skips_to_use_at_level_i):
                if k_skip_idx >= len(skip_feats):
                    logger.warning(f"Attempting to access skip_feat index {k_skip_idx} but only {len(skip_feats)} available at decoder level {i}.")
                    continue 
                
                skip_raw = skip_feats[k_skip_idx]
                if skip_raw is None:
                    logger.warning(f"Encountered None skip_feat at index {k_skip_idx} for decoder level {i}.")
                    continue
                #跳跃特征插值

                skip_up = F.interpolate(skip_raw, size=x.shape[-2:], mode='bilinear', align_corners=False)
                projected_skip = skip_up
                if skip_up.shape[1] != self.base_channels:
                    proj_layer = self._get_projection_layer(self.skip_feature_projections, skip_up.shape[1], self.base_channels, current_device)
                    try:
                        projected_skip = proj_layer(skip_up)
                    except Exception as e:
                        logger.error(f"Error projecting skip_feat (idx {k_skip_idx}) from {skip_up.shape[1]} to {self.base_channels} chans: {e}. Using original.")
                
                # Adaptive Depth Fusion
                # Use processed_depth_feats which should now have self.num_encoder_feature_levels elements (some might be None)
                #将所有尺度的depth_feats都插值到当前大小，再cat送入sacle_mlp生成weights
                up_depths_for_fusion_step = []
                valid_depth_count_for_fusion_step = 0
                for df_processed in processed_depth_feats:
                    if df_processed is not None:
                        up_depths_for_fusion_step.append(F.interpolate(df_processed, size=x.shape[-2:], mode='bilinear', align_corners=False))
                        valid_depth_count_for_fusion_step += 1
                    else: # df_processed is None
                        # Pad with zeros if a None is encountered, to maintain channel count for cat
                        dummy_zeros = torch.zeros(B, self.base_channels, x.shape[2], x.shape[3], device=current_device)
                        up_depths_for_fusion_step.append(dummy_zeros)
                
                if valid_depth_count_for_fusion_step > 0:
                    # Ensure up_depths_for_fusion_step has self.num_encoder_feature_levels items for cat
                    # This should be guaranteed by the initial padding/truncation of depth_feats_for_processing
                    # and the subsequent padding of Nones with zeros here.
                    if len(up_depths_for_fusion_step) == self.num_encoder_feature_levels:#如果所有udffs列表长度等于nefl即可拼接
                        depth_cat_for_mlp = torch.cat(up_depths_for_fusion_step, dim=1)#拼接channel维度cat->[B, base_channels * L, H, W]
                        scores = self.scale_mlp(depth_cat_for_mlp)#MLP 得到分数： [B, num_encoder_feature_levels, H, W]
                        weights = torch.softmax(scores, dim=1)#weights = torch.softmax(scores, dim=1)# 记录第一层 (k_skip_idx == 0) 的 weights 用于可视化
                        if k_skip_idx == 0: all_weights_for_logging.append(weights)
                        
                        # 已禁用深度门控，不再计算深度特征权重和gate_map
                        # depth_fusion_sum = sum(weights[:,j:j+1] * up_depths_for_fusion_step[j] for j in range(self.num_encoder_feature_levels))
                        # gate_map = torch.sigmoid(self.gate_conv(depth_fusion_sum))
                        gated = projected_skip  # 直接使用原始skip特征，不应用gate_map
                    else:
                        logger.error(f"Decoder level {i}, skip {k_skip_idx}: Mismatch in depth features for MLP after processing. Expected {self.num_encoder_feature_levels}, got {len(up_depths_for_fusion_step)}. Using skip directly.")
                        gated = projected_skip
                else:
                    gated = projected_skip #对不齐就不用深度门控，没有深度也不深度门控
                    if k_skip_idx == 0 and i == 0 : logger.info("MultiTaskDecoder: No valid depth features provided for fusion for first skip.")

                gated_skips.append(gated)
                
            fused = torch.cat([x] + gated_skips, dim=1)
            #进行通道拼接
            if fused.shape[1] != self.deblur_fuse_convs[i].in_channels:
                logger.warning(
                    f"Decoder level {i}: Channel mismatch for deblur_fuse_convs. Fused: {fused.shape[1]}, Expected: {self.deblur_fuse_convs[i].in_channels}. "
                    f"Skips provided: {len(gated_skips)} (expected {i+1}). Attempting emergency projection."
                )
                proj_layer_fused = self._get_projection_layer(
                    self.skip_feature_projections, 
                    fused.shape[1], 
                    self.deblur_fuse_convs[i].in_channels, 
                    current_device
                )
                try:
                    fused = proj_layer_fused(fused)
                except Exception as e:
                    logger.error(f"Error in emergency projection of fused features at level {i}: {e}. Passing as is.")
            
            logger.debug(f"Decoder level {i}: About to call deblur_fuse_convs[{i}] (expected in: {self.deblur_fuse_convs[i].in_channels if hasattr(self.deblur_fuse_convs[i], 'in_channels') else 'N/A'}) with fused shape {fused.shape}")
            fused = self.deblur_fuse_convs[i](fused)#降通道
            x = self.deblur_blocks[i](fused)#restormer
            
        res_d = self.deblur_recon(x)
        
        # ----- 颜色分支 - 深度加权实现 -----
        x_color = fused_feat
        all_weights_for_logging_color = []
        
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
                projected_skip = skip_up
                if skip_up.shape[1] != self.base_channels:
                    proj_layer = self._get_projection_layer(self.skip_feature_projections, skip_up.shape[1], self.base_channels, current_device)
                    try:
                        projected_skip = proj_layer(skip_up)
                    except Exception as e:
                        logger.error(f"颜色分支: 投影skip_feat时出错 (idx {k_skip_idx}): {e}。使用原始特征。")
                
                # 自适应深度融合
                up_depths_for_fusion_step = []
                valid_depth_count_for_fusion_step = 0
                for df_processed in processed_depth_feats:
                    if df_processed is not None:
                        up_depths_for_fusion_step.append(F.interpolate(df_processed, size=x_color.shape[-2:], mode='bilinear', align_corners=False))
                        valid_depth_count_for_fusion_step += 1
                    else: # df_processed is None
                        # 如果遇到None则用零填充，以保持通道数一致
                        dummy_zeros = torch.zeros(B, self.base_channels, x_color.shape[2], x_color.shape[3], device=current_device)
                        up_depths_for_fusion_step.append(dummy_zeros)
                
                if valid_depth_count_for_fusion_step > 0:
                    if len(up_depths_for_fusion_step) == self.num_encoder_feature_levels:
                        depth_cat_for_mlp = torch.cat(up_depths_for_fusion_step, dim=1)
                        scores = self.scale_mlp_color(depth_cat_for_mlp)
                        weights = torch.softmax(scores, dim=1)
                        if k_skip_idx == 0: all_weights_for_logging_color.append(weights)
                        
                        # 已禁用深度门控，不再计算深度特征权重和gate_map
                        # depth_fusion_sum = sum(weights[:,j:j+1] * up_depths_for_fusion_step[j] for j in range(self.num_encoder_feature_levels))
                        # gate_map = torch.sigmoid(self.gate_conv_color(depth_fusion_sum))
                        gated = projected_skip  # 直接使用原始skip特征，不应用gate_map
                    else:
                        logger.error(f"颜色分支: 解码层 {i}, skip {k_skip_idx}: MLP处理后深度特征不匹配。预期 {self.num_encoder_feature_levels}, 得到 {len(up_depths_for_fusion_step)}。直接使用skip。")
                        gated = projected_skip
                else:
                    gated = projected_skip 
                    if k_skip_idx == 0 and i == 0: 
                        logger.info("颜色分支: 首个skip没有提供有效深度特征进行融合。")

                gated_skips_color.append(gated)
                
            fused = torch.cat([x_color] + gated_skips_color, dim=1)
            
            if fused.shape[1] != self.color_fuse_convs[i].in_channels:
                logger.warning(
                    f"颜色分支: 解码层 {i}: 颜色融合卷积通道不匹配。融合特征: {fused.shape[1]}, 预期: {self.color_fuse_convs[i].in_channels}。"
                    f"提供的跳连数: {len(gated_skips_color)} (预期 {i+1})。尝试应急投影。"
                )
                proj_layer_fused = self._get_projection_layer(
                    self.skip_feature_projections, 
                    fused.shape[1], 
                    self.color_fuse_convs[i].in_channels, 
                    current_device
                )
                try:
                    fused = proj_layer_fused(fused)
                except Exception as e:
                    logger.error(f"颜色分支: 解码层 {i} 特征融合应急投影出错: {e}。原样传递。")
            
            logger.debug(f"颜色分支: 解码层 {i}: 调用color_fuse_convs[{i}] (预期输入: {self.color_fuse_convs[i].in_channels if hasattr(self.color_fuse_convs[i], 'in_channels') else 'N/A'}) 融合特征形状 {fused.shape}")
            fused = self.color_fuse_convs[i](fused)
            
            # 通过DACBlock，不再应用深度加权
            if i < len(self.dac_blocks):
                # 不再需要深度权重计算
                x_color = self.dac_blocks[i](fused)
            else:
                # 如果没有对应的DACBlock，则使用RestormerBlock
                x_color = self.color_blocks[i](fused)
                
        # 使用颜色重建头生成残差
        res_c = self.color_recon(x_color)

        if all_weights_for_logging:
            target_size = x.shape[-2:]
            resized_weights_list = []
            for w_log in all_weights_for_logging:
                if w_log.shape[-2:] != target_size:
                    num_depth_scales_in_w = w_log.shape[1]
                    per_scale_resized = []
                    for s_idx in range(num_depth_scales_in_w):
                        per_scale_resized.append(F.interpolate(w_log[:,s_idx:s_idx+1], size=target_size, mode='bilinear', align_corners=False))
                    resized_weights_list.append(torch.cat(per_scale_resized, dim=1))
                else:
                    resized_weights_list.append(w_log)
            if resized_weights_list:
                try:
                    self.last_fusion_weights = torch.stack(resized_weights_list, dim=1) 
                except Exception as e:
                    logger.error(f"Error stacking fusion weights: {e}. Shapes: {[rw.shape for rw in resized_weights_list]}")
                    self.last_fusion_weights = None
            else:
                self.last_fusion_weights = None
        else:
            self.last_fusion_weights = None
            
        # 如果提供了深度图和物理模型参数，应用A/D物理模型公式
        if depth_pred is not None and beta_c is not None and B_c is not None and blur_scale is not None:
            # --- 3.3 深度条件 PSF 卷积 (D 式) ---
            # 3.3.1 线性拼接得到"清晰图"
            j_clear = raw + res_d                  # [B,3,H,W]
            # 3.3.2 计算 σ(x)
            # 确保 depth_pred 归一到 [0,1]
            depth_norm = torch.clamp(depth_pred, min=0.0, max=1.0)  # [B,1,H,W]
            sigma = blur_scale * depth_norm    # [B,1,H,W]

            # 3.3.3 调用 Kornia Gaussian blur
            kernel_size = 5  # 高斯核大小，可微调
            
            # 计算每个样本的平均sigma值作为高斯模糊的标准差
            batch_size = sigma.shape[0]
            # 为每个样本创建一个[2]形状的sigma值 (x和y方向相同)
            sigma_mean = sigma.view(batch_size, -1).mean(dim=1)  # [B]
            sigma_for_blur = torch.stack([sigma_mean, sigma_mean], dim=1)  # [B,2]
            
            # 确保 j_clear 在 [0,1] 范围内，避免模糊失真
            j_clear_clamped = torch.clamp(j_clear, 0.0, 1.0)

            # 执行 Gaussian blur
            J_D = K.gaussian_blur2d(j_clear_clamped,
                                 (kernel_size, kernel_size),
                                 sigma=sigma_for_blur,
                                 border_type='reflect')  # [B,3,H,W]

            # --- 3.4 Beer–Lambert (A 式) ---
            j_color = raw + res_c              # [B,3,H,W]
            j_color_clamped = torch.clamp(j_color, 0.0, 1.0)

            # 3.4.1 归一化深度
            depth_norm = torch.clamp(depth_pred, min=0.0, max=1.0)  # [B,1,H,W]
            # 3.4.2 计算 t_c(x)：先把 depth 复制成 3 通道
            depth_3ch = depth_norm.repeat(1, 3, 1, 1)              # [B,3,H,W]
            t = torch.exp(- beta_c * depth_3ch)                   # [B,3,H,W]

            # 3.4.3 Beer–Lambert 合成
            I_A = j_color_clamped * t + B_c * (1.0 - t)           # [B,3,H,W]
            I_A = torch.clamp(I_A, 0.0, 1.0)

            # 更新 res_d 和 res_c 以反映物理模型处理
            # 从 J_D 和 I_A 中恢复残差
            res_d = J_D - raw
            res_c = I_A - raw
            
            # 双支路融合
            merged = torch.cat([J_D, I_A], dim=1)   # [B,6,H,W]
            merged = self.fuse_conv(merged)         # [B,3,H,W]
            final_out = torch.sigmoid(merged)       # 保证在 [0,1]
            
            # 返回融合输出以及两个分支的结果，便于计算物理一致性损失
            return final_out, J_D, I_A
        else:
            # 如果没有物理模型参数，返回原始的残差
            return res_d, None, None

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
        depth_pred = torch.rand(batch_size, 1, 128, 128).to(device)  # 模拟深度预测
        beta_c = torch.ones(1, 3, 1, 1).to(device) * 0.7  # 模拟Beer-Lambert系数
        B_c = torch.zeros(1, 3, 1, 1).to(device)  # 模拟背景光
        blur_scale = torch.tensor(0.005).to(device)  # 模拟模糊比例参数
        
        res_d, res_c = decoder(
            fused_feat, 
            skip_feats, 
            depth_feats, 
            raw, 
            depth_pred=depth_pred,
            beta_c=beta_c,
            B_c=B_c,
            blur_scale=blur_scale
        )
        
        if debug:
            print(f"解码器测试成功！输出形状: {res_d.shape}")
        
        return True
    except Exception as e:
        if debug:
            print(f"解码器测试失败: {e}")
            import traceback
            traceback.print_exc()
        return False

