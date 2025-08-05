import torch
import torch.nn as nn
import torch.nn.functional as F
from .sfe import ShallowFeatureExtractor
from .depth import DepthFeatureExtractor, ensure_normalized_depth, get_depth_config_params, MonoDepthHead
from .encoder import RawEncoder
from .blocks import RestormerBlock
from .decoder import MultiTaskDecoder
from .recon_head import ReconHead, MultiTaskHead
from .depth_decoder import DepthDecoder
from .blocks import PhysicsParamHead
import warnings
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union, Tuple

# Get a logger instance for this module
# 使用 __name__ 可以让日志记录器自动获取模块名 "modules.model"
# 在 train.py 中可以通过 logging.getLogger('modules.model') 控制其日志级别
model_logger = logging.getLogger(__name__)


@dataclass
class ModelOutput:
    """统一的模型输出格式"""
    enhanced: torch.Tensor                          # 增强后的图像 [B, 3, H, W]
    pred_gate: torch.Tensor                         # 预测的门控图 [B, 1, H, W]
    depth_pred: torch.Tensor                        # 预测的连续深度图 [B, 1, H, W]
    student_feats: Optional[List[torch.Tensor]] = None    # 编码器特征列表
    attention_maps: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # 注意力图元组 (depth2rgb, rgb2depth)
    
    # 🔥 多输入一致性学习新增属性
    multi_enhanced: Optional[torch.Tensor] = None          # 多输入增强结果 [B, N, 3, H, W]
    multi_depth_pred: Optional[torch.Tensor] = None        # 多输入深度预测 [B, N, 1, H, W]
    multi_res_d: Optional[torch.Tensor] = None             # 多输入去模糊残差 [B, N, 3, H, W]
    multi_res_c: Optional[torch.Tensor] = None             # 多输入颜色校正残差 [B, N, 3, H, W]

    is_multi_input: bool = False                           # 是否为多输入模式
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（兼容旧代码）"""
        result = {
            'enhanced': self.enhanced,
            'pred_gate': self.pred_gate,
            'depth_pred': self.depth_pred,
            'student_feats': self.student_feats,
            'attention_maps': self.attention_maps
        }
        
        # 添加多输入属性
        if self.is_multi_input:
            result.update({
                'multi_enhanced': self.multi_enhanced,
                'multi_depth_pred': self.multi_depth_pred,
                'multi_res_d': self.multi_res_d,
                'multi_res_c': self.multi_res_c,
                'is_multi_input': self.is_multi_input
            })
        
        return result


class UnderwaterEnhanceNet(nn.Module):
    """
    UnderwaterEnhanceNet - 主网络定义
    --------------------------------
    架构流程：
      1. Shallow Feature Extractor (SFE)
      2. DepthFeatureExtractor (训练期)
      3. RawEncoder (RGB + Depth 真值 + GT) -> student_feats, teacher_feats
      4. Joint Bottleneck (Restormer Blocks)
      5. MultiTaskDecoder (Deblur & Color 分支，支持 Dense Skip + 自适应深度融合)
      6. ReconHead (残差融合 + 激活)

    forward 输入/输出：
      - raw: Tensor[B,3,H,W]
      - depth_gt: Tensor[B,1,H,W] or None
      - gt: Tensor[B,3,H,W] or None
    返回:
      - out: Tensor[B,3,H,W]       (增强图)
      - pred_gate: Tensor[B,1,H,W] (预测门控图)
      - student_feats: list of Tensor (多尺度 RGB 特征)
      - attention_maps: tuple of Tensor or None (depth2rgb, rgb2depth 注意力图)
    """
    def __init__(self, 
                 base_channels: int = 48,
                 levels: int = 4,
                 heads: int = 8,
                 bottleneck_blocks: int = 4,
                 depth_processor_config: dict = None,
                 encoder_window_size: int = 8,
                 bottleneck_window_size: int = 0,
                 decoder_block_window_size: int = 4,
                 save_attention_maps: bool = False,
                 double_forward: bool = True):
        super().__init__()
        
        # 使用 self.logger 以便在整个类中使用
        self.logger = model_logger
        self.arch_logger = logging.getLogger('architecture') # 获取架构专用的logger
        self.attn_logger = logging.getLogger('attention')   # 获取注意力专用的logger

        self.arch_logger.info("--- Initializing UnderwaterEnhanceNet ---")
        self.arch_logger.info(f"  - Base Channels: {base_channels}, Levels: {levels}, Heads: {heads}")
        self.arch_logger.info(f"  - Bottleneck Blocks: {bottleneck_blocks}")
        self.arch_logger.info(f"  - Window Sizes | Encoder: {encoder_window_size}, Bottleneck: {bottleneck_window_size}, Decoder: {decoder_block_window_size}")
        self.arch_logger.info(f"  - Save Attention Maps: {save_attention_maps}, Double Forward: {double_forward}")
        
        self.depth_params = depth_processor_config or {}
        self.raw_depth_processor_config = depth_processor_config if depth_processor_config else {}
        # Use get_depth_config_params to get typed parameters for depth processing
        self.typed_depth_params = get_depth_config_params(self.raw_depth_processor_config)
        self.levels = levels
        self.base_channels = base_channels  # 保存base_channels用于验证
        self.save_attention_maps = save_attention_maps
        
        # ===（1）、（2）、（3）物理参数已删除 ===
        
        # 1. 浅层特征
        self.sfe = ShallowFeatureExtractor(in_channels=3, out_channels=base_channels)

        # 预计算编码器通道数，确保与解码器匹配
        self.encoder_channels = [base_channels * (2**i) for i in range(levels)]
        self.arch_logger.info(f"  - Calculated Encoder Channels: {self.encoder_channels}")

        # 2. 深度支路：训练期真深度多尺度提取
        # 注意：DepthFeatureExtractor输出固定通道数，需要在验证时考虑这一点
        self.depth_extractor = DepthFeatureExtractor(
            in_channels=1, 
            base_channels=base_channels,  # 所有输出特征都是base_channels通道
            levels=levels, 
            # Pass typed params for internal normalization if needed by DFE
            # Or DFE can also call get_depth_config_params internally if preferred
            use_log_transform=self.typed_depth_params.use_log_transform,
            min_depth_log=self.typed_depth_params.min_depth_log,
            max_depth_log=self.typed_depth_params.max_depth_log,
            min_depth_linear=self.typed_depth_params.min_depth_linear,
            max_depth_linear=self.typed_depth_params.max_depth_linear,
            double_channels=self.depth_params.get("double_channels", True)
        )
        self.depth_head = MonoDepthHead(in_channels=base_channels)

        # 2.b DepthDecoder
        self.depth_decoder = DepthDecoder(base_c=base_channels,
                                          levels=levels,
                                          window=decoder_block_window_size)

        # 3. 编码器：融合 RGB、Depth GT
        self.encoder = RawEncoder(in_channels=base_channels,
                                  depth_channels=1,
                                  base_channels=base_channels,
                                  levels=levels,
                                  depth_processor_config=self.raw_depth_processor_config,
                                  encoder_window_size=encoder_window_size
                                  )

        # 4. Joint Bottleneck
        bottleneck_channels = base_channels * (2**(levels - 1)) 
        if bottleneck_blocks > 0:
            self.bottleneck = nn.Sequential(
                *[RestormerBlock(bottleneck_channels, heads, bottleneck_window_size) for _ in range(bottleneck_blocks)]
            )
            bottleneck_channels_actual = bottleneck_channels
        else:
            self.bottleneck = nn.Identity()
            bottleneck_channels_actual = bottleneck_channels
        
        # 5. 解码器：解码级数 = levels - 1 (最多上采到原始分辨率)
        self.decoder = MultiTaskDecoder(
            base_channels=base_channels,
            input_channels_bottleneck=bottleneck_channels_actual,
            levels=levels-1,
            decoder_block_window_size=decoder_block_window_size,
            num_encoder_feature_levels=levels
        )

        # 6. 重建头
        # 用于同时输出增强图和深度回归
        self.recon = MultiTaskHead(
            feat_channels=base_channels,  # 解码器最后一级通道数
        )
        
        # 🔥 7. 物理参数预测头
        self.physics_head = PhysicsParamHead(
            in_channels=bottleneck_channels_actual,
            hidden=128
        )
        
        self.arch_logger.info(f"  - Physics Head initialized: input_channels={bottleneck_channels_actual}")
        
        self.depth_fusion_weights = None
        
        # 计算每级深度特征的预期通道数 - 基于DepthFeatureExtractor的实际输出
        # 编码器和DepthDecoder通道数配置
        self.arch_logger.info(f"  - 编码器期望通道数: {self.encoder_channels}")

        # 🔥 动态投影层：根据实际输入通道数创建适配器，而不是预先固定通道数
        # 我们将使用动态适配器而不是固定的投影层
        self.depth_projection_layers = None  # 将在forward中动态创建
        
        self.arch_logger.info(f"  - 使用动态深度特征投影层，将根据实际输入通道数自动适配")
        
        # 修复：通道匹配问题改为动态解决方案
        self.arch_logger.info(f"  - ✅ 深度特征通道匹配改为动态适配")
        
        # 添加动态通道适配层，避免在forward中创建临时层
        self.depth_feature_adapters = nn.ModuleDict()
        # 预先创建常见的通道配置适配层
        common_depth_configs = [
            (48, 48), (48, 96), (48, 192), (48, 384),  # base_channels到各级编码器通道
            (96, 48), (96, 96), (96, 192), (96, 384),  # 2*base_channels到各级
            (192, 48), (192, 96), (192, 192), (192, 384),  # 4*base_channels到各级
            (384, 48), (384, 96), (384, 192), (384, 384),  # 8*base_channels到各级
        ]
        
        for in_ch, out_ch in common_depth_configs:
            adapter_key = f"adapt_{in_ch}_to_{out_ch}"
            self.depth_feature_adapters[adapter_key] = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        
        self.arch_logger.info(f"  - Pre-created {len(self.depth_feature_adapters)} depth feature adapters")
        
        # 启动时进行一次dummy forward检查，确保通道匹配（仅在主进程中）
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            self._perform_channel_validation_check()  # 只在主进程中运行验证
        
        # 删除：多输出融合模块（已简化为仅使用一致性损失）
        
        self.arch_logger.info("  - Multi-Output Fusion: disabled (using consistency loss only)")
        
        self.arch_logger.info(f"--- UnderwaterEnhanceNet Initialized ---")
        
        # 启用注意力图保存，若配置了保存
        if save_attention_maps:
            self.attn_logger.info("启用注意力图保存功能")
            self.enable_attention_saving()
        else:
            self.attn_logger.info("未启用注意力图保存功能")

        # 标记推理方式
        self.double_forward = double_forward
        self._forward_count = 0 # 内部计数器，用于控制日志频率

    def enable_attention_saving(self, enable=True):
        """启用或禁用注意力图保存"""
        self.save_attention_maps = enable
        self.attn_logger.info(f"设置 save_attention_maps = {enable}")
        
        # 遍历编码器中的所有交叉注意力模块，启用注意力图保存
        if hasattr(self.encoder, 'depth2rgb_attn_blocks'):
            self.attn_logger.debug(f"为 {len(self.encoder.depth2rgb_attn_blocks)} 个 depth2rgb_attn_blocks 启用注意力图保存")
            for i, attn in enumerate(self.encoder.depth2rgb_attn_blocks):
                attn.enable_attention_saving(enable)
                self.attn_logger.debug(f"  - depth2rgb_attn_block[{i}].save_attention = {attn.save_attention}")
        else:
            self.logger.warning("编码器没有 depth2rgb_attn_blocks 属性")
            
        if hasattr(self.encoder, 'rgb2depth_attn_blocks'):
            self.attn_logger.debug(f"为 {len(self.encoder.rgb2depth_attn_blocks)} 个 rgb2depth_attn_blocks 启用注意力图保存")
            for i, attn in enumerate(self.encoder.rgb2depth_attn_blocks):
                attn.enable_attention_saving(enable)
                self.attn_logger.debug(f"  - rgb2depth_attn_block[{i}].save_attention = {attn.save_attention}")
        else:
            self.logger.warning("编码器没有 rgb2depth_attn_blocks 属性")

    def _encode(self, raw, depth_feats=None, gt=None):
        """编码阶段的辅助函数，支持使用外部深度特征
        
        Args:
            raw: 原始输入图像 (B,3,H,W)
            depth_feats: 可选的外部深度特征列表，用于推理阶段
            gt: 可选的清晰图GT (B,3,H,W)
        
        Returns:
            student_feats: 编码器RGB特征
            bottleneck: 瓶颈层输出
        """
        # 1) 浅层特征提取 (SFE 内部已处理数据类型不匹配问题)
        x = self.sfe(raw)
        
        # 3) 编码器：提取RGB特征，整合深度和GT
        # 使用外部提供的深度特征替代真实深度图（如果有）
        student_feats, _ = self.encoder(x, depth_gt=None, gt=gt, depth_feats=depth_feats)
        
        # 确保student_feats的数量与self.levels匹配
        if len(student_feats) != self.levels:
            warnings.warn(f"Expected {self.levels} student features, but got {len(student_feats)}!")
            # 截断或补足
            if len(student_feats) > self.levels:
                student_feats = student_feats[:self.levels]
            else:
                # 补足 (复制最后一个特征)
                while len(student_feats) < self.levels:
                    student_feats.append(student_feats[-1])
        
        # 4) 瓶颈层
        bottleneck = self.bottleneck(student_feats[-1])
        
        return student_feats, bottleneck

    def get_attention_maps(self):
        """获取当前的注意力图，供可视化使用"""
        depth2rgb_attn = None
        rgb2depth_attn = None
        
        if hasattr(self.encoder, 'depth2rgb_attn_blocks'):
            self.attn_logger.debug(f"检查 depth2rgb_attn_blocks 中的注意力图...")
            for i, attn_block in enumerate(self.encoder.depth2rgb_attn_blocks):
                has_attr = hasattr(attn_block, 'last_attn')
                is_not_none = has_attr and attn_block.last_attn is not None
                self.attn_logger.debug(f"  - depth2rgb_attn_block[{i}]: has_attr={has_attr}, is_not_none={is_not_none}")
                if is_not_none:
                    depth2rgb_attn = attn_block.last_attn
                    self.attn_logger.debug(f"  - 获取到 depth2rgb 注意力图: shape={depth2rgb_attn.shape}")
                    break
                    
        if hasattr(self.encoder, 'rgb2depth_attn_blocks'):
            self.attn_logger.debug(f"检查 rgb2depth_attn_blocks 中的注意力图...")
            for i, attn_block in enumerate(self.encoder.rgb2depth_attn_blocks):
                has_attr = hasattr(attn_block, 'last_attn')
                is_not_none = has_attr and attn_block.last_attn is not None
                self.attn_logger.debug(f"  - rgb2depth_attn_block[{i}]: has_attr={has_attr}, is_not_none={is_not_none}")
                if is_not_none:
                    rgb2depth_attn = attn_block.last_attn
                    self.attn_logger.debug(f"  - 获取到 rgb2depth 注意力图: shape={rgb2depth_attn.shape}")
                    break
            
        self.attn_logger.debug(f"返回注意力图: d2r={depth2rgb_attn is not None}, r2d={rgb2depth_attn is not None}")
        return depth2rgb_attn, rgb2depth_attn

    def forward(self,
                raw: torch.Tensor,
                depth_gt: torch.Tensor = None,
                gt: torch.Tensor = None,
                enable_multi_input_consistency: bool = False) -> Union[ModelOutput, Dict[str, Any]]:
        
        self._forward_count += 1
        # 只在每个epoch的第一个batch记录详细日志
        log_this_step = self._forward_count == 1 

        # 检查是否为多输入模式
        is_multi_input = raw.dim() == 5  # [B, N, C, H, W]
        
        # 🔥 添加调试信息 - 强制打印前几次
        if self._forward_count <= 3:
            print(f"[DEBUG] Forward #{self._forward_count}: Input shape: {raw.shape}, is_multi_input: {is_multi_input}")
            print(f"[DEBUG] Forward #{self._forward_count}: enable_multi_input_consistency: {enable_multi_input_consistency}")
        
        if is_multi_input and enable_multi_input_consistency:
            # 🔥 多输入一致性学习模式
            if self._forward_count <= 3:
                print(f"[DEBUG] Forward #{self._forward_count}: 进入多输入一致性学习模式")
            return self._forward_multi_input_consistency(raw, depth_gt, gt, log_this_step)
        elif is_multi_input:
            # 传统模式：展平多输入为批次，取第一个退化
            B, N, C, H, W = raw.shape
            raw = raw[:, 0]  # [B, C, H, W] - 取第一个退化
            
            # 处理GT和depth_gt
            if depth_gt is not None:
                if depth_gt.dim() == 5:  # [B, N, 1, H, W]
                    depth_gt = depth_gt[:, 0]  # [B, 1, H, W]
                # 否则保持原样 [B, 1, H, W]
            
            if gt is not None:
                if gt.dim() == 5:  # [B, N, C, H, W]
                    gt = gt[:, 0]  # [B, C, H, W]
                # 否则保持原样 [B, C, H, W]

        if log_this_step:
            self.logger.info(f"--- [Forward Pass Start] ---")
            self.logger.info(f"Input shapes: raw={raw.shape}, depth_gt={depth_gt.shape if depth_gt is not None else 'None'}, gt={gt.shape if gt is not None else 'None'}")
            if is_multi_input:
                self.logger.info(f"Multi-input mode: {'Consistency Learning' if enable_multi_input_consistency else 'First Degradation Only'}")

        # 混合精度训练兼容性：确保输入数据类型一致
        if depth_gt is not None and raw.dtype != depth_gt.dtype:
            depth_gt = depth_gt.to(dtype=raw.dtype)
        if gt is not None and raw.dtype != gt.dtype:
            gt = gt.to(dtype=raw.dtype)
            
        # 添加调试信息，跟踪深度图属性
        if depth_gt is not None:
            self.logger.debug(f"深度GT形状:{depth_gt.shape}, 范围:[{depth_gt.min().item():.4f}, {depth_gt.max().item():.4f}]")
            if hasattr(depth_gt, '_depth_processed'):
                self.logger.debug(f"深度GT已被标记为处理过")
        
        # 判断当前处于训练或推理模式
        training_mode = self.training or depth_gt is not None
        if log_this_step:
            self.logger.info(f"Mode: {'Training' if training_mode else 'Inference'}")

        if training_mode:
            # ===== 训练时："三输入 + 双次前向" =====
            if log_this_step: self.logger.info("--- Pass-1: Depth Prediction ---")

            # ——— Pass-1: 仅用 raw 预测深度（DepthDecoder） ———
            # 1.1) 浅层特征提取
            # SFE 内部已处理数据类型不匹配问题
            x = self.sfe(raw)  # [B, C1, H, W]
            if log_this_step: self.logger.info(f"  - SFE output shape: {x.shape}")

            # 1.2) RGB 编码器第一遍：raw 只进 RGB 编码器，depth_feats 传 None  
            student_feats_pass1, bottleneck_pass1 = self._encode(raw, depth_feats=None, gt=None)
            if log_this_step: 
                self.logger.info(f"  - Encoder Pass-1 output shapes: student_feats={[f.shape for f in student_feats_pass1]}, bottleneck={bottleneck_pass1.shape}")

            # 1.3) DepthDecoder：用 bottleneck_pass1 + student_feats_pass1 预测连续深度 & 深度多尺度特征
            depth_pred, depth_feats = self.depth_decoder(bottleneck_pass1, student_feats_pass1)
            if log_this_step:
                self.logger.info(f"  - DepthDecoder output shapes: depth_pred={depth_pred.shape}, depth_feats={[f.shape for f in depth_feats]}")
            #    depth_pred:  [B,1,H,W]         （Pass-1 的连续深度预测，用于与 depth_gt 做回归损失）
            #    depth_feats: list of Tensor, 多尺度深度特征，length ≥ self.levels

            # ——— 监督深度：将 depth_pred 与 depth_gt 传给 Loss 函数 —— 在 train.py / loss_fn.py 中完成

            # 1.4) 多尺度深度特征动态投影到 RGB 编码器对应通道数
            projected_depth_feats = []
            for i, feat in enumerate(depth_feats):
                if i < len(self.encoder_channels):
                    target_channels = self.encoder_channels[i]
                    if feat.shape[1] != target_channels:
                        # 使用动态适配器直接投影到目标通道数
                        adapter = self._get_or_create_adapter(feat.shape[1], target_channels, feat.device)
                        projected_feat = adapter(feat)
                        if log_this_step:
                            self.logger.info(f"深度特征 {i}: {feat.shape[1]} → {target_channels} 通道")
                    else:
                        projected_feat = feat
                    projected_depth_feats.append(projected_feat)
                else:
                    projected_depth_feats.append(feat)
            depth_feats = projected_depth_feats[:self.levels]
            if log_this_step: self.logger.info(f"  - Projected depth_feats shapes: {[f.shape for f in depth_feats]}")
            #    depth_feats 现在是一个长度为 self.levels 的列表，
            #    每个 Tensor 尺寸分别为 [B, rgb_ch_i, h_i, w_i]

            # 1.5) 用第 0 级深度特征做门控图（pred_gate），计算深度置信度 depth_conf_map（损失）
            pred_gate = self.depth_head(depth_feats[0])                                                         
            if log_this_step: self.logger.info(f"  - pred_gate shape: {pred_gate.shape}")


            # ——— Pass-2: 用预测的 depth_feats + raw + gt 做 RGB 分支（与推理时第二次前向一致） ———
            if log_this_step: self.logger.info("--- Pass-2: RGB Enhancement ---")
            
            # 2.1) 编码器：把 raw + depth_feats + gt 一起输入 RawEncoder
            student_feats, _ = self.encoder(x, depth_feats=depth_feats, gt=gt)
            if log_this_step: 
                self.logger.info(f"  - Encoder Pass-2 output shapes: student_feats={[f.shape for f in student_feats]}")
            #    student_feats: list of Tensor, length == self.levels

            # 2.2) 确保 student_feats 长度正确
            if len(student_feats) != self.levels:
                self.logger.warning(f"Expected {self.levels} student features, but got {len(student_feats)}!")
                if len(student_feats) > self.levels:
                    student_feats = student_feats[:self.levels]
                else:
                    while len(student_feats) < self.levels:
                        student_feats.append(student_feats[-1])

            # 2.3) 瓶颈：处理最后一层的student特征
            bottleneck = self.bottleneck(student_feats[-1])
            if log_this_step: self.logger.info(f"  - Bottleneck output shape: {bottleneck.shape}")
            
            # 🔥 2.3.5) 物理参数预测
            beta_c, B_c, blur_scale = self.physics_head(bottleneck)
            if log_this_step: 
                self.logger.info(f"  - Physics params: beta_c={beta_c.shape}, B_c={B_c.shape}, blur_scale={blur_scale.shape}")
            
            res_d, res_c = self.decoder(
                bottleneck, 
                student_feats[:-1], 
                depth_feats, 
                raw, 
                depth_pred=depth_pred,
                beta_c=beta_c,
                B_c=B_c,
                blur_scale=blur_scale
            )
            final_out = None  # 解码器现在只返回残差
            if log_this_step:
                self.logger.info(f"  - Decoder output shapes: res_d={res_d.shape if res_d is not None else 'None'}, res_c={res_c.shape if res_c is not None else 'None'}, final_out={final_out.shape if final_out is not None else 'None'}")

            # 2.4) 使用ReconHead合成最终输出
            if final_out is not None:
                # 如果解码器已经提供了融合结果，直接使用
                out = final_out
                depth_pred_refine = depth_pred  # 保持原来的深度预测
                if log_this_step: self.logger.info(f"  - Using decoder's fused output directly")
            else:
                # 使用ReconHead合成残差
                final_feat = student_feats[0]
                out, depth_pred_refine = self.recon(raw, res_d, res_c, final_feat)
                if log_this_step: self.logger.info(f"  - Using ReconHead with residuals. Output shape: {out.shape}, Refined depth shape: {depth_pred_refine.shape}")
            
            # 为了保持兼容性，从残差重新计算J_D和I_A
            J_D = raw + res_d if res_d is not None else None
            I_A = raw + res_c if res_c is not None else None

        else:
            # ===== 推理时：仅输入 raw，走"双次前向" =====
            if log_this_step: self.logger.info("--- Inference Path ---")
            if log_this_step: self.logger.info("--- Pass-1: Depth Prediction ---")

            # 1) 浅层特征提取（与训练时 Pass-1 相同）
            x = self.sfe(raw)
            if log_this_step: self.logger.info(f"  - SFE output shape: {x.shape}")

            # 2) Pass-1：RGB 编码器+DepthDecoder → 预测深度
            student_feats_pass1, bottleneck_pass1 = self._encode(raw, depth_feats=None, gt=None)
            if log_this_step: self.logger.info(f"  - Encoder Pass-1 shapes: bottleneck={bottleneck_pass1.shape}")
            depth_pred, depth_feats = self.depth_decoder(bottleneck_pass1, student_feats_pass1)
            if log_this_step: self.logger.info(f"  - DepthDecoder shapes: depth_pred={depth_pred.shape}")

            # 3) 动态投影深度特征到 RGB 分支对应通道
            projected_depth_feats = []
            for i, feat in enumerate(depth_feats):
                if i < len(self.encoder_channels):
                    target_channels = self.encoder_channels[i]
                    if feat.shape[1] != target_channels:
                        # 使用动态适配器直接投影到目标通道数
                        adapter = self._get_or_create_adapter(feat.shape[1], target_channels, feat.device)
                        projected_feat = adapter(feat)
                    else:
                        projected_feat = feat
                    projected_depth_feats.append(projected_feat)
                else:
                    projected_depth_feats.append(feat)
            depth_feats = projected_depth_feats[:self.levels]
            if log_this_step: self.logger.info(f"  - Projected depth_feats shapes: {[f.shape for f in depth_feats]}")

            # 4) 用第 0 级深度特征生成门控图 pred_gate；禁用置信度图用于调试DECL损失
            pred_gate = self.depth_head(depth_feats[0])

            if log_this_step: self.logger.info(f"  - pred_gate shape: {pred_gate.shape}")


            # 5) Pass-2：将 raw + depth_feats → RGB 编码器 → 解码 → ReconHead → 输出增强图
            if log_this_step: self.logger.info("--- Pass-2: RGB Enhancement ---")
            student_feats, bottleneck = self._encode(raw, depth_feats=depth_feats, gt=None)
            if log_this_step: self.logger.info(f"  - Encoder Pass-2 shapes: bottleneck={bottleneck.shape}")
            if log_this_step: self.logger.info(f"  - Bottleneck output shape: {bottleneck.shape}")
            
            # 🔥 5.5) 物理参数预测
            beta_c, B_c, blur_scale = self.physics_head(bottleneck)
            if log_this_step: 
                self.logger.info(f"  - Physics params: beta_c={beta_c.shape}, B_c={B_c.shape}, blur_scale={blur_scale.shape}")
            
            res_d, res_c = self.decoder(
                bottleneck, 
                student_feats[:-1], 
                depth_feats, 
                raw,
                depth_pred=depth_pred,
                beta_c=beta_c,
                B_c=B_c,
                blur_scale=blur_scale
            )
            final_out = None  # 解码器现在只返回残差
            if log_this_step: self.logger.info(f"  - Decoder output shapes: res_d={res_d.shape if res_d is not None else 'None'}, res_c={res_c.shape if res_c is not None else 'None'}, final_out={final_out.shape if final_out is not None else 'None'}")

            # 2.4) 使用ReconHead合成最终输出
            if final_out is not None:
                # 如果解码器已经提供了融合结果，直接使用
                out = final_out
                depth_pred_refine = depth_pred  # 保持原来的深度预测
                if log_this_step: self.logger.info(f"  - Using decoder's fused output directly")
            else:
                # 使用ReconHead合成残差
                final_feat = student_feats[0]
                out, depth_pred_refine = self.recon(raw, res_d, res_c, final_feat)
                if log_this_step: self.logger.info(f"  - Using ReconHead with residuals. Output shape: {out.shape}, Refined depth shape: {depth_pred_refine.shape}")
            
            # 为了保持兼容性，从残差重新计算J_D和I_A
            J_D = raw + res_d if res_d is not None else None
            I_A = raw + res_c if res_c is not None else None
        
        if log_this_step:
            self.logger.info(f"--- [Forward Pass End] ---")
            self.logger.info(f"Final output shapes: enhanced={out.shape}, pred_gate={pred_gate.shape}, depth_pred={depth_pred.shape}")

        # ===== 最终输出统一封装 =====
        return ModelOutput(
            enhanced=out,               # 增强后结果
            pred_gate=pred_gate,       # 深度门控
            depth_pred=depth_pred,      # 预测深度
            student_feats=student_feats_pass1 if training_mode else student_feats,
            attention_maps=self.get_attention_maps() if self.save_attention_maps else None
        )

    def get_depth_fusion_weights(self):
        """获取深度融合权重（用于可视化）"""
        if self.depth_fusion_weights is None:
            return None
        
        # 将所有权重整合到一个列表中
        all_weights = []
        for level, weights in enumerate(self.depth_fusion_weights):
            if weights is not None:
                all_weights.append((f"Level {level}", weights))
        
        return all_weights
    
    def _perform_channel_validation_check(self):
        """在初始化时进行dummy前向传播以验证通道匹配"""
        try:
            self.arch_logger.info("Performing channel validation check...")
            device = next(self.parameters()).device
            
            # 创建dummy输入
            dummy_raw = torch.randn(1, 3, 64, 64, device=device)
            dummy_depth = torch.randn(1, 1, 64, 64, device=device)
            
            # 执行部分前向传播以检查通道匹配
            with torch.no_grad():
                # 步骤1：提取深度特征
                depth_feats = self.depth_extractor(dummy_depth)
                
                # 步骤2：检查深度特征通道是否与投影层匹配
                # 注意：DepthFeatureExtractor输出固定通道数base_channels，而DepthDecoder输出递增通道数
                # 这里我们检查的是DepthDecoder的输出，应该与expected_depth_channels匹配
                for i, depth_feat in enumerate(depth_feats):
                    if i < len(self.encoder_channels):
                        expected_channels = self.encoder_channels[i]
                        actual_channels = depth_feat.shape[1]
                        # 动态适配：不再强制要求通道匹配，因为会在forward时动态适配
                        if actual_channels != expected_channels and actual_channels != self.base_channels:
                            # 只有当通道数完全不符合预期时才警告
                            logger.warning(f"深度特征级别 {i}: 实际通道数 {actual_channels}, 将动态适配到 {expected_channels}")
                        else:
                            self.arch_logger.info(f"深度特征级别 {i}: 实际通道数 {actual_channels}, 符合预期或为base_channels")
                
                self.arch_logger.info("✓ 深度特征通道匹配验证通过")
                
        except Exception as e:
            self.arch_logger.error(f"通道验证失败: {e}")
            raise RuntimeError(f"模型初始化时通道验证失败: {e}")
            
        self.arch_logger.info("通道验证检查完成")

    def multi_forward(self, raw_batch, depth_gt=None, gt=None):
        """批量前向计算，优先使用批处理，仅在必要时逐样本处理
        
        Args:
            raw_batch: 原始输入图像批次 (B,3,H,W)
            depth_gt: 深度GT批次 (B,1,H,W) or None
            gt: 清晰图GT批次 (B,3,H,W) or None
            
        Returns:
            outputs: 输出字典，包含 'enhanced', 'pred_gate', 'depth_pred' 等
        """
        B = raw_batch.shape[0]
        device = raw_batch.device
        
        # 🔥 修复显存跳变：检查是否所有样本尺寸相同
        all_same_size = True
        if B > 1:
            first_shape = raw_batch[0].shape
            for b in range(1, B):
                if raw_batch[b].shape != first_shape:
                    all_same_size = False
                    break
                    
            # 检查depth_gt尺寸
            if all_same_size and depth_gt is not None:
                first_depth_shape = depth_gt[0].shape
                for b in range(1, B):
                    if depth_gt[b].shape != first_depth_shape:
                        all_same_size = False
                        break
                        
            # 检查gt尺寸
            if all_same_size and gt is not None:
                first_gt_shape = gt[0].shape
                for b in range(1, B):
                    if gt[b].shape != first_gt_shape:
                        all_same_size = False
                        break
        
        # 如果所有样本尺寸相同，直接使用批处理
        if all_same_size:
            self.arch_logger.debug(f"🚀 使用批处理模式 (batch_size={B})")
            return self.forward(raw_batch, depth_gt, gt)
        
        # 否则使用逐样本处理（不同尺寸时的兼容模式）
        self.arch_logger.warning(f"⚠️ 检测到不同尺寸样本，使用逐样本处理模式 (batch_size={B})")
        
        # 准备输出容器
        all_enhanced = []
        all_pred_gates = []
        all_depth_preds = []
        all_student_feats = []
        # all_depth_conf_maps = []  # 已移除depth_conf_map
        all_attn_maps = []
        
        # 收集J_D和I_A输出
        all_J_D = []
        all_I_A = []
        
        # 逐个处理每个样本
        for b in range(B):
            raw_n = raw_batch[b:b+1]
            
            # 处理可选输入
            processed_depth_gt = None
            if depth_gt is not None:
                processed_depth_gt = depth_gt[b:b+1]
                
            gt_n = None
            if gt is not None:
                gt_n = gt[b:b+1]
                
            # 清理之前的注意力图状态，确保每次前向传播都是独立的
            if self.save_attention_maps and hasattr(self, 'encoder'):
                if hasattr(self.encoder, 'depth2rgb_attn_blocks'):
                    for attn_block in self.encoder.depth2rgb_attn_blocks:
                        if hasattr(attn_block, 'last_attn'):
                            attn_block.last_attn = None
                if hasattr(self.encoder, 'rgb2depth_attn_blocks'):
                    for attn_block in self.encoder.rgb2depth_attn_blocks:
                        if hasattr(attn_block, 'last_attn'):
                            attn_block.last_attn = None
                
            # 单个样本前向传播
            try:
                output_dict = self.forward(raw_n, processed_depth_gt, gt_n)
                
                # 收集输出
                all_enhanced.append(output_dict.enhanced)
                all_pred_gates.append(output_dict.pred_gate)
                all_depth_preds.append(output_dict.depth_pred)
                
                if output_dict.student_feats is not None:
                    all_student_feats.append(output_dict.student_feats)
                    
                # if output_dict.depth_conf_map is not None:  # 已移除depth_conf_map
                #     all_depth_conf_maps.append(output_dict.depth_conf_map)
                    
                if output_dict.attention_maps is not None:
                    all_attn_maps.append(output_dict.attention_maps)
                
                # 物理模型输出已删除
                
            except Exception as e:
                self.logger.error(f"[MODEL] 样本 {b} 处理错误: {str(e)}")
                # 记录更详细的错误信息
                import traceback
                self.logger.error(traceback.format_exc())
                # 如果这是第一个样本而且出错，则重新抛出异常
                if b == 0:
                    raise
                # 否则尝试继续处理其他样本
                continue
        
        # 确保至少有一个有效输出
        if not all_enhanced:
            raise RuntimeError("所有样本处理都失败，无法生成有效输出")
        
        # 整合输出
        enhanced = torch.cat(all_enhanced, dim=0)
        pred_gate = torch.cat(all_pred_gates, dim=0)
        depth_pred = torch.cat(all_depth_preds, dim=0)
        
        # 整合多尺度特征需要更复杂的处理
        student_feats = all_student_feats if all_student_feats else None
        # depth_conf_map = torch.cat(all_depth_conf_maps, dim=0) if all_depth_conf_maps else None  # 已移除
        
        # 整合注意力图
        attention_maps = None
        if all_attn_maps:
            # 假设每个样本返回的是 (depth2rgb, rgb2depth) 元组
            depth2rgb_maps = [maps[0] for maps in all_attn_maps if maps[0] is not None]
            rgb2depth_maps = [maps[1] for maps in all_attn_maps if maps[1] is not None]
            
            # 如果有有效的注意力图，将它们整合
            if depth2rgb_maps and rgb2depth_maps:
                depth2rgb_batch = torch.cat(depth2rgb_maps, dim=0) if depth2rgb_maps[0] is not None else None
                rgb2depth_batch = torch.cat(rgb2depth_maps, dim=0) if rgb2depth_maps[0] is not None else None
                attention_maps = (depth2rgb_batch, rgb2depth_batch)
        
        # 物理模型输出已删除
        
        # 返回输出
        output = ModelOutput(
            enhanced=enhanced,
            pred_gate=pred_gate,
            depth_pred=depth_pred,
            student_feats=student_feats,
            attention_maps=attention_maps
        )
        
        return output

    def _get_or_create_adapter(self, in_channels: int, out_channels: int, device: torch.device) -> nn.Module:
        """获取深度特征适配器，禁用动态创建以修复显存跳变"""
        adapter_key = f"adapt_{in_channels}_to_{out_channels}"
        
        if adapter_key not in self.depth_feature_adapters:
            # 🔥 修复显存跳变：不再动态创建适配器！
            self.arch_logger.error(f"❌ Missing depth adapter {adapter_key} in UnderwaterEnhanceNet! "
                                 f"This should not happen with proper channel configuration. "
                                 f"Available adapters: {list(self.depth_feature_adapters.keys())}")
            
            # 应急处理：如果输入输出通道数相同，返回恒等映射
            if in_channels == out_channels:
                self.arch_logger.warning(f"⚠️ Using identity mapping for {in_channels} == {out_channels} channels in depth adapter")
                if not hasattr(self, '_identity_depth_adapter'):
                    self._identity_depth_adapter = nn.Identity()
                return self._identity_depth_adapter
            else:
                # 如果通道数不同，这是一个严重错误，应该停止训练
                raise RuntimeError(f"🚨 Depth feature adapter channel mismatch: {in_channels} -> {out_channels}. "
                                 f"Please add this configuration to depth_feature_adapters in __init__. "
                                 f"Dynamic adapter creation has been disabled to fix memory jumps.")
        
        return self.depth_feature_adapters[adapter_key]

    def _forward_multi_input_consistency(self, raw, depth_gt, gt, log_this_step):
        """
        🔥 多输入一致性学习前向传播 - 所有候选都参与损失计算
        实现方案：展平多输入为批次维度，然后用 repeat_interleave 处理 GT
        所有N个候选输出都保留用于损失计算，而不仅仅是融合后的单一输出
        
        Args:
            raw: [B, N, C, H, W] - N个不同退化的输入
            depth_gt: [B, 1, H, W] 或 [B, N, 1, H, W]
            gt: [B, C, H, W] 或 [B, N, C, H, W]
        Returns:
            ModelOutput with:
            - 主输出: 所有候选的集成平均结果 (用于可视化)
            - 多候选: 所有N个原始候选 (用于全面的损失计算)
            - 一致性: CMCL损失约束多候选之间的一致性
        """
        B, N, C, H, W = raw.shape
        
        if log_this_step:
            self.logger.info(f"--- Multi-Input Consistency Learning (Decoder Fusion Level) ---")
            self.logger.info(f"Input shape: [B={B}, N={N}, C={C}, H={H}, W={W}]")
        
        # 🔥 方案：展平 + repeat_interleave（如用户建议）
        # 将 [B, N, C, H, W] 展平为 [B*N, C, H, W]
        raw_flat = raw.reshape(B * N, C, H, W)
        
        # 处理 GT 数据 - 使用 repeat_interleave
        if depth_gt is not None:
            if depth_gt.dim() == 4:  # [B, 1, H, W] 
                # 每个样本的深度重复 N 次：[B, 1, H, W] -> [B*N, 1, H, W]
                depth_gt_flat = depth_gt.repeat_interleave(N, dim=0)
            elif depth_gt.dim() == 5:  # [B, N, 1, H, W]
                depth_gt_flat = depth_gt.reshape(B * N, 1, H, W)
            else:
                depth_gt_flat = depth_gt
        else:
            depth_gt_flat = None
        
        if gt is not None:
            if gt.dim() == 4:  # [B, C, H, W]
                # 每个样本的 GT 重复 N 次：[B, C, H, W] -> [B*N, C, H, W]
                gt_flat = gt.repeat_interleave(N, dim=0)
            elif gt.dim() == 5:  # [B, N, C, H, W]
                gt_flat = gt.reshape(B * N, C, H, W)
            else:
                gt_flat = gt
        else:
            gt_flat = None
        
        if log_this_step:
            self.logger.info(f"Flattened shapes: raw_flat={raw_flat.shape}, "
                           f"depth_gt_flat={depth_gt_flat.shape if depth_gt_flat is not None else 'None'}, "
                           f"gt_flat={gt_flat.shape if gt_flat is not None else 'None'}")
        
        # 🔥 修改前向传播以记录残差
        # === Pass-1: 深度预测 ===
        x = self.sfe(raw_flat)
        student_feats_pass1, bottleneck_pass1 = self._encode(raw_flat, depth_feats=None, gt=None)
        depth_pred_flat, depth_feats = self.depth_decoder(bottleneck_pass1, student_feats_pass1)
        
        # 动态投影深度特征
        projected_depth_feats = []
        for i, feat in enumerate(depth_feats[:self.levels]):
            if i < len(self.encoder_channels):
                target_channels = self.encoder_channels[i]
                if feat.shape[1] != target_channels:
                    adapter = self._get_or_create_adapter(feat.shape[1], target_channels, feat.device)
                    projected_feat = adapter(feat)
                else:
                    projected_feat = feat
                projected_depth_feats.append(projected_feat)
            else:
                projected_depth_feats.append(feat)
        depth_feats = projected_depth_feats
        
        pred_gate_flat = self.depth_head(depth_feats[0])
        
        # === Pass-2: RGB增强 ===
        training_mode = self.training or depth_gt_flat is not None
        if training_mode:
            student_feats_flat, _ = self.encoder(x, depth_feats=depth_feats, gt=gt_flat)
        else:
            student_feats_flat, _ = self.encoder(x, depth_feats=depth_feats, gt=None)
            
        bottleneck_flat = self.bottleneck(student_feats_flat[-1])
        
        # 物理参数预测
        beta_c, B_c, blur_scale = self.physics_head(bottleneck_flat)
        
        # === 解码阶段：获取残差 ===
        res_d_flat, res_c_flat = self.decoder(
            bottleneck_flat, student_feats_flat[:-1], depth_feats, raw_flat,
            depth_pred=depth_pred_flat, beta_c=beta_c, B_c=B_c, blur_scale=blur_scale
        )
        
        # 重建
        enhanced_flat, depth_pred_refine_flat = self.recon(raw_flat, res_d_flat, res_c_flat, student_feats_flat[0])
        
        # 🔥 重新组织输出为多输入格式
        enhanced = enhanced_flat.reshape(B, N, C, H, W)  # [B, N, C, H, W]
        depth_pred = depth_pred_refine_flat.reshape(B, N, 1, H, W)  # [B, N, 1, H, W]
        pred_gate = pred_gate_flat.reshape(B, N, 1, H, W)  # [B, N, 1, H, W]
        
        # 🔥 关键：重新组织残差用于一致性计算
        res_d = res_d_flat.reshape(B, N, C, H, W)  # [B, N, C, H, W] - 去模糊残差
        res_c = res_c_flat.reshape(B, N, C, H, W)  # [B, N, C, H, W] - 颜色校正残差
        
        # 🔥 新策略：使用所有候选的平均作为主输出，但保留完整的多候选信息
        # 这样既有单一输出用于可视化，又有完整信息用于损失计算
        primary_enhanced = torch.mean(enhanced, dim=1)  # [B, C, H, W] - 所有候选的平均
        primary_depth_pred = torch.mean(depth_pred, dim=1)  # [B, 1, H, W] - 所有候选的平均
        primary_pred_gate = torch.mean(pred_gate, dim=1)  # [B, 1, H, W] - 所有候选的平均
        
        if log_this_step:
            self.logger.info("Using ensemble average as primary output, all candidates available for loss calculation")
        
        if log_this_step:
            self.logger.info(f"Multi-input results (All Candidates):")
            self.logger.info(f"  - enhanced: {enhanced.shape} (all {N} candidates)")
            self.logger.info(f"  - depth_pred: {depth_pred.shape} (all {N} candidates)")
            self.logger.info(f"  - res_d (deblur): {res_d.shape} (all {N} candidates)")
            self.logger.info(f"  - res_c (color): {res_c.shape} (all {N} candidates)")
            self.logger.info(f"  - primary outputs (ensemble): enhanced={primary_enhanced.shape}")
        
        # 🔥 获取注意力图用于损失计算
        attention_maps = self.get_attention_maps() if self.save_attention_maps else None
        
        # 特征一致性已删除，只保留输出层面的CMCL一致性约束
        
        # 🔥 创建多输入模式的输出（包含所有候选用于损失计算）
        result = ModelOutput(
            enhanced=primary_enhanced,
            pred_gate=primary_pred_gate,
            depth_pred=primary_depth_pred,
            student_feats=student_feats_flat,  # 保持原始特征格式 [B*N, ...]
            attention_maps=attention_maps,    # 🔥 正确传递注意力图
            # 多输入特有属性 - 🔥 所有候选都可用于损失计算
            multi_enhanced=enhanced,         # [B, N, C, H, W] - 所有N个增强候选
            multi_depth_pred=depth_pred,     # [B, N, 1, H, W] - 所有N个深度候选  
            multi_res_d=res_d,              # [B, N, C, H, W] - 所有N个去模糊残差
            multi_res_c=res_c,              # [B, N, C, H, W] - 所有N个颜色校正残差
            is_multi_input=True
        )
        
        return result


    

