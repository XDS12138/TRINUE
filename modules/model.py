import torch
import torch.nn as nn
import torch.nn.functional as F
from .sfe import ShallowFeatureExtractor
from .depth import DepthFeatureExtractor, MonoDepthHead, ensure_normalized_depth, get_depth_config_params
from .encoder import RawEncoder
from .blocks import RestormerBlock
from .decoder import MultiTaskDecoder
from .recon_head import ReconHead, MultiTaskHead
from .depth_decoder import DepthDecoder
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
    depth_conf_map: Optional[torch.Tensor] = None         # 深度置信度图
    attention_maps: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # 注意力图元组 (depth2rgb, rgb2depth)
    J_D: Optional[torch.Tensor] = None              # D式模糊后的去模糊图 [B, 3, H, W]
    I_A: Optional[torch.Tensor] = None              # A式Beer–Lambert合成输出 [B, 3, H, W]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（兼容旧代码）"""
        return {
            'enhanced': self.enhanced,
            'pred_gate': self.pred_gate,
            'depth_pred': self.depth_pred,
            'student_feats': self.student_feats,
            'depth_conf_map': self.depth_conf_map,
            'attention_maps': self.attention_maps,
            'J_D': self.J_D,
            'I_A': self.I_A
        }


class UnderwaterEnhanceNet(nn.Module):
    """
    UnderwaterEnhanceNet - 主网络定义
    --------------------------------
    架构流程：
      1. Shallow Feature Extractor (SFE)
      2. DepthFeatureExtractor (训练期) + MonoDepthHead (蒸馏 & 推理)
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
      - teacher_feats: list of Tensor or None (多尺度 GT 特征)
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
        self.save_attention_maps = save_attention_maps
        
        # ===（1）CB通道 Beer–Lambert 衰减系数 β_c，初值设为 0.7，可调===
        # 形状：[1, 3, 1, 1]，对 R/G/B 三个通道分别学习一个系数
        self.beta_c = nn.Parameter(torch.ones(1, 3, 1, 1) * 0.7)

        # ===（2）全局背景光 B_c，初始设为零，也可改用 MLP 从特征中预测===
        # 形状：[1, 3, 1, 1]
        self.B_c = nn.Parameter(torch.zeros(1, 3, 1, 1))

        # ===（3）深度调节模糊比例 k（Depth PSF 中 σ = k * depth）===
        # 标量可学习，初值设为 0.005，也可自行调
        self.blur_scale = nn.Parameter(torch.tensor(0.005))
        
        # 1. 浅层特征
        self.sfe = ShallowFeatureExtractor(in_channels=3, out_channels=base_channels)

        # 预计算编码器通道数，确保与解码器匹配
        self.encoder_channels = [base_channels * (2**i) for i in range(levels)]
        self.arch_logger.info(f"  - Calculated Encoder Channels: {self.encoder_channels}")

        # 2. 深度支路：训练期真深度多尺度提取
        self.depth_extractor = DepthFeatureExtractor(
            in_channels=1, 
            base_channels=base_channels, 
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
        
        self.depth_fusion_weights = None
        
        # 计算每级深度特征的预期通道数 - 修复通道数计算
        expected_depth_channels = []
        for i in range(levels):
            if i == 0:
                # 第0级是预测的深度图，用base_channels表示
                expected_depth_channels.append(base_channels)
            else:
                # 根据深度解码器的特性，特征通道数可能会增加
                if self.depth_params.get('double_channels', True):
                    # 如果启用了通道翻倍，与编码器保持一致
                    expected_depth_channels.append(base_channels * (2**i))
                else:
                    # 否则维持基础通道数
                    expected_depth_channels.append(base_channels)

        # 打印调试信息
        self.arch_logger.info(f"  - Initializing depth projection layers with expected channels: {expected_depth_channels}")

        self.depth_projection_layers = nn.ModuleList([
            nn.Conv2d(expected_depth_channels[i], self.encoder_channels[i], kernel_size=1, bias=False)
            for i in range(levels)
        ])
        
        self.arch_logger.info(f"  - Depth Projection Layers: Input channels: {[layer.in_channels for layer in self.depth_projection_layers]}, Output channels: {[layer.out_channels for layer in self.depth_projection_layers]}")
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
                gt: torch.Tensor = None) -> Union[ModelOutput, Dict[str, Any]]:
        
        self._forward_count += 1
        # 只在每个epoch的第一个batch记录详细日志
        log_this_step = self._forward_count == 1 

        if log_this_step:
            self.logger.info(f"--- [Forward Pass Start] ---")
            self.logger.info(f"Input shapes: raw={raw.shape}, depth_gt={depth_gt.shape if depth_gt is not None else 'None'}, gt={gt.shape if gt is not None else 'None'}")

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

            # 1.4) 多尺度深度特征投影到 RGB 编码器对应通道数
            projected_depth_feats = []
            for i, feat in enumerate(depth_feats):
                if i < len(self.depth_projection_layers):
                    # 检查通道数不匹配的情况
                    if feat.shape[1] != self.depth_projection_layers[i].in_channels:
                        self.logger.warning(f"通道不匹配：深度特征 {i} 有 {feat.shape[1]} 通道，但投影层期望 {self.depth_projection_layers[i].in_channels} 通道")
                        # 对于任何深度级别，创建适配层处理通道数不匹配
                        temp_adapter = nn.Conv2d(
                            feat.shape[1], 
                            self.depth_projection_layers[i].in_channels, 
                            kernel_size=1, 
                            bias=False
                        ).to(feat.device)
                        # 初始化权重以保留信息
                        nn.init.ones_(temp_adapter.weight[:, :min(feat.shape[1], self.depth_projection_layers[i].in_channels), :, :])
                        # 应用适配层
                        feat = temp_adapter(feat)
                        self.logger.info(f"已将深度特征 {i} 的通道从 {feat.shape[1]} 适配到 {self.depth_projection_layers[i].in_channels}")
                    
                    # 投影：1x1 卷积将深度特征通道数转换成与 RGB 编码器第 i 级相同
                    projected_feat = self.depth_projection_layers[i](feat)
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

            if depth_gt is None:
                # 推理模式下，使用pred_gate作为置信度图（已经是0-1范围）
                depth_conf_map = pred_gate.detach()
                if log_this_step: self.logger.info(f"  - Generated depth_conf_map from pred_gate (Inference mode logic)")
            elif hasattr(depth_gt, '_depth_processed'):
                # 如果深度已处理，使用pred_gate作为置信度
                depth_conf_map = pred_gate.detach()
                if log_this_step: self.logger.info(f"  - Generated depth_conf_map from pred_gate (depth already processed)")
            else:
                # 基于深度梯度和pred_gate的组合生成置信度图
                depth_dx, depth_dy = torch.gradient(depth_gt, dim=(-2, -1))
                depth_grad_mag = torch.sqrt(depth_dx**2 + depth_dy**2)
                # 归一化梯度幅度到[0,1]
                grad_normalized = (depth_grad_mag - depth_grad_mag.min()) / (depth_grad_mag.max() - depth_grad_mag.min() + 1e-8)
                # 反转梯度（边缘处置信度低）
                edge_conf = 1.0 - grad_normalized
                # 结合pred_gate和边缘置信度
                depth_conf_map = 0.5 * pred_gate.detach() + 0.5 * edge_conf
                if log_this_step: self.logger.info(f"  - Generated depth_conf_map from gradient and pred_gate")
            if log_this_step: self.logger.info(f"  - depth_conf_map shape: {depth_conf_map.shape}")

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
            final_out, J_D, I_A = self.decoder(
                bottleneck, 
                student_feats[:-1], 
                depth_feats, 
                raw, 
                depth_pred=depth_pred,
                beta_c=self.beta_c,
                B_c=self.B_c,
                blur_scale=self.blur_scale
            )
            if log_this_step:
                self.logger.info(f"  - Decoder output shapes: final_out={final_out.shape}, J_D={J_D.shape if J_D is not None else 'None'}, I_A={I_A.shape if I_A is not None else 'None'}")

            # 2.4) 如果物理模型已融合，直接使用融合结果；否则使用ReconHead
            if J_D is not None and I_A is not None:
                # 使用融合后的物理模型结果，跳过ReconHead
                out = final_out
                depth_pred_refine = depth_pred  # 保持原来的深度预测
                if log_this_step: self.logger.info(f"  - Using physical model output directly (J_D and I_A are not None)")
            else:
                # 传统方式：使用ReconHead合成
                final_feat = student_feats[0]
                # 假设res_c也是由decoder返回的，或者在这里计算
                # 这里假设res_c是final_out的一部分或可以通过其他方式获得
                # 如果ReconHead的API改变了，需要调整这里
                res_c = torch.zeros_like(final_out) # Placeholder if res_c is not available
                out, depth_pred_refine = self.recon(raw, final_out, res_c, final_feat)
                if log_this_step: self.logger.info(f"  - Using ReconHead. Output shape: {out.shape}, Refined depth shape: {depth_pred_refine.shape}")

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

            # 3) 投影深度特征到 RGB 分支对应通道
            projected_depth_feats = []
            for i, feat in enumerate(depth_feats):
                if i < len(self.depth_projection_layers):
                    # 检查通道数不匹配的情况
                    if feat.shape[1] != self.depth_projection_layers[i].in_channels:
                        self.logger.warning(f"通道不匹配：深度特征 {i} 有 {feat.shape[1]} 通道，但投影层期望 {self.depth_projection_layers[i].in_channels} 通道")
                        # 对于任何深度级别，创建适配层处理通道数不匹配
                        temp_adapter = nn.Conv2d(
                            feat.shape[1], 
                            self.depth_projection_layers[i].in_channels, 
                            kernel_size=1, 
                            bias=False
                        ).to(feat.device)
                        # 初始化权重以保留信息
                        nn.init.ones_(temp_adapter.weight[:, :min(feat.shape[1], self.depth_projection_layers[i].in_channels), :, :])
                        # 应用适配层
                        feat = temp_adapter(feat)
                        self.logger.info(f"已将深度特征 {i} 的通道从 {feat.shape[1]} 适配到 {self.depth_projection_layers[i].in_channels}")
                    
                    projected_feat = self.depth_projection_layers[i](feat)
                    projected_depth_feats.append(projected_feat)
                else:
                    projected_depth_feats.append(feat)
            depth_feats = projected_depth_feats[:self.levels]
            if log_this_step: self.logger.info(f"  - Projected depth_feats shapes: {[f.shape for f in depth_feats]}")

            # 4) 用第 0 级深度特征生成门控图 pred_gate；推理时使用pred_gate作为置信度
            pred_gate = self.depth_head(depth_feats[0])
            depth_conf_map = pred_gate.detach()  # 使用预测的门控图作为置信度
            if log_this_step: self.logger.info(f"  - pred_gate shape: {pred_gate.shape}")

            # 5) Pass-2：将 raw + depth_feats → RGB 编码器 → 解码 → ReconHead → 输出增强图
            if log_this_step: self.logger.info("--- Pass-2: RGB Enhancement ---")
            student_feats, bottleneck = self._encode(raw, depth_feats=depth_feats, gt=None)
            if log_this_step: self.logger.info(f"  - Encoder Pass-2 shapes: bottleneck={bottleneck.shape}")
            if log_this_step: self.logger.info(f"  - Bottleneck output shape: {bottleneck.shape}")
            final_out, J_D, I_A = self.decoder(
                bottleneck, 
                student_feats[:-1], 
                depth_feats, 
                raw, 
                depth_pred=depth_pred,
                beta_c=self.beta_c,
                B_c=self.B_c,
                blur_scale=self.blur_scale
            )
            if log_this_step: self.logger.info(f"  - Decoder output shapes: final_out={final_out.shape}, J_D={J_D.shape if J_D is not None else 'None'}, I_A={I_A.shape if I_A is not None else 'None'}")

            # 2.4) 如果物理模型已融合，直接使用融合结果；否则使用ReconHead
            if J_D is not None and I_A is not None:
                # 使用融合后的物理模型结果，跳过ReconHead
                out = final_out
                depth_pred_refine = depth_pred  # 保持原来的深度预测
                if log_this_step: self.logger.info(f"  - Using physical model output directly")
            else:
                # 传统方式：使用ReconHead合成
                final_feat = student_feats[0]
                res_c = torch.zeros_like(final_out) # Placeholder
                out, depth_pred_refine = self.recon(raw, final_out, res_c, final_feat)
                if log_this_step: self.logger.info(f"  - Using ReconHead. Output shape: {out.shape}, Refined depth shape: {depth_pred_refine.shape}")
        
        if log_this_step:
            self.logger.info(f"--- [Forward Pass End] ---")
            self.logger.info(f"Final output shapes: enhanced={out.shape}, pred_gate={pred_gate.shape}, depth_pred={depth_pred.shape}")

        # ===== 最终输出统一封装 =====
        return ModelOutput(
            enhanced=out,               # 增强后结果
            pred_gate=pred_gate,       # 深度门控
            depth_pred=depth_pred,      # 预测深度
            student_feats=student_feats_pass1 if training_mode else student_feats,
            depth_conf_map=depth_conf_map,           # 置信度图
            attention_maps=self.get_attention_maps() if self.save_attention_maps else None,
            J_D=J_D,
            I_A=I_A
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

    def multi_forward(self, raw_batch, depth_gt=None, gt=None):
        """批量前向计算，处理可能不同尺寸的图像
        
        Args:
            raw_batch: 原始输入图像批次 (B,3,H,W)
            depth_gt: 深度GT批次 (B,1,H,W) or None
            gt: 清晰图GT批次 (B,3,H,W) or None
            
        Returns:
            outputs: 输出字典，包含 'enhanced', 'pred_gate', 'depth_pred' 等
        """
        B = raw_batch.shape[0]
        device = raw_batch.device
        
        # 准备输出容器
        all_enhanced = []
        all_pred_gates = []
        all_depth_preds = []
        all_student_feats = []
        all_depth_conf_maps = []
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
                
            # 单个样本前向传播
            try:
                output_dict = self.forward(raw_n, processed_depth_gt, gt_n)
                
                # 收集输出
                all_enhanced.append(output_dict.enhanced)
                all_pred_gates.append(output_dict.pred_gate)
                all_depth_preds.append(output_dict.depth_pred)
                
                if output_dict.student_feats is not None:
                    all_student_feats.append(output_dict.student_feats)
                    
                if output_dict.depth_conf_map is not None:
                    all_depth_conf_maps.append(output_dict.depth_conf_map)
                    
                if output_dict.attention_maps is not None:
                    all_attn_maps.append(output_dict.attention_maps)
                
                # 收集物理模型输出
                if output_dict.J_D is not None:
                    all_J_D.append(output_dict.J_D)
                
                if output_dict.I_A is not None:
                    all_I_A.append(output_dict.I_A)
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
        depth_conf_map = torch.cat(all_depth_conf_maps, dim=0) if all_depth_conf_maps else None
        
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
        
        # 整合J_D和I_A (如果有)
        J_D_batch = torch.cat(all_J_D, dim=0) if all_J_D else None
        I_A_batch = torch.cat(all_I_A, dim=0) if all_I_A else None
        
        # 返回输出
        output = ModelOutput(
            enhanced=enhanced,
            pred_gate=pred_gate,
            depth_pred=depth_pred,
            student_feats=student_feats,
            depth_conf_map=depth_conf_map,
            attention_maps=attention_maps,
            J_D=J_D_batch,
            I_A=I_A_batch
        )
        
        return output
