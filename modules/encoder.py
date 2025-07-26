# modules/encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging  # 用于日志输出

from .sfe import ShallowFeatureExtractor
from .depth import DepthFeatureExtractor, get_depth_config_params, ensure_normalized_depth
from .cross_attention import CrossAttention
from .blocks import RestormerBlock, ConvBlock

# 获取该模块的 logger
logger = logging.getLogger(__name__)


class RawEncoder(nn.Module):
    """
    RawEncoder - 原始图编码器
    ---------------------
    多尺度特征提取，每一级都包含:
    1. 卷积下采样 (s=2，除第0级)
    2. RestormerBlock
    3. 双向深度交叉注意力 (仅在有深度图时，并可选是否门控)

    每级下采样通道数翻倍:
    - 级别 0 (原尺寸): C
    - 级别 1 (1/2): 2C
    - 级别 2 (1/4): 4C
    - 级别 3 (1/8): 8C
    - ...
    """

    def __init__(
        self,
        in_channels: int,
        depth_channels: int,
        base_channels: int,
        levels: int,
        depth_processor_config: dict = None,
        encoder_window_size: int = 8
    ):
        super().__init__()

        self.in_channels = in_channels
        self.depth_channels = depth_channels
        self.base_channels = base_channels
        self.levels = levels

        # 计算每一级的输出通道数：base * 2^i
        self.channels = [base_channels * (2 ** i) for i in range(levels)]
        logger.info(f"RawEncoder: {levels} levels with channels: {self.channels}")

        # === 一、准备 Depth 投影层，用于把深度特征映射到和当前层级一致的通道数 ===
        self.depth_projections = nn.ModuleList()
        # 计算每级 depth 输入通道数：与DepthDecoder的输出一致，都是递增的
        expected_depth_channels = [base_channels * (2**i) for i in range(levels)]

        for i in range(levels):
            in_ch = expected_depth_channels[i]
            out_ch = self.channels[i]
            conv1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            self.depth_projections.append(conv1x1)
            logger.info(f"RawEncoder: Depth projection level {i}: Input {in_ch} -> Output {out_ch}")

        # === 二、构造 RGB 分支 & Depth 分支 的编码块 ===
        # 1) student_blocks 用于提取 RGB（或浅层特征 x）每级的特征：ConvBlock + RestormerBlock
        self.student_blocks = nn.ModuleList()
        curr_ch = in_channels
        # 是否启用 cross-attention，默认为 True
        self.use_cross_attn = depth_processor_config.get('use_cross_attn', True) if depth_processor_config else True

        # 2) 构造双向 CrossAttention 模块列表
        if self.use_cross_attn:
            # Depth -> RGB
            self.depth2rgb_attn_blocks = nn.ModuleList([
                CrossAttention(self.channels[i], heads=8, window_size=encoder_window_size)
                for i in range(levels)
            ])
            # RGB -> Depth
            self.rgb2depth_attn_blocks = nn.ModuleList([
                CrossAttention(self.channels[i], heads=8, window_size=encoder_window_size)
                for i in range(levels)
            ])
        else:
            # 不启用时，用 Identity 占位
            self.depth2rgb_attn_blocks = nn.ModuleList([nn.Identity() for _ in range(levels)])
            self.rgb2depth_attn_blocks = nn.ModuleList([nn.Identity() for _ in range(levels)])

        # 3) 构造每一级的 Conv+RestormerBlock
        for i in range(levels):
            out_ch = self.channels[i]
            stride = 2 if i > 0 else 1  # 级别0不下采样，其他级别下采样2倍
            block = nn.Sequential(
                ConvBlock(curr_ch, out_ch, stride=stride),
                RestormerBlock(out_ch, heads=8, window_size=encoder_window_size)
            )
            self.student_blocks.append(block)
            curr_ch = out_ch

        # === 三、深度预处理相关：DepthPreprocessor 配置 & DepthFeatureExtractor ===
        if depth_processor_config:
            self.depth_params = get_depth_config_params(depth_processor_config)
        else:
            self.depth_params = None

        # 用一个简单卷积把原始 depth_gt (1 通道) 映射到 base_channels
        self.depth_feature_extractor = nn.Conv2d(1, base_channels, kernel_size=3, padding=1, bias=False)

        # === 四、新增：双向可学习门控 γ ===
        # 如果启用 cross-attn，就为每一级分别加上两个可学习标量：γ_d2r 和 γ_r2d
        if self.use_cross_attn:
            self.d2r_gamma = nn.ParameterList([
                nn.Parameter(torch.zeros(1), requires_grad=True) for _ in range(levels)
            ])
            self.r2d_gamma = nn.ParameterList([
                nn.Parameter(torch.zeros(1), requires_grad=True) for _ in range(levels)
            ])
        else:
            # 不启用时，为保持属性的一致性，也创建相同结构，但不会真正使用
            self.d2r_gamma = nn.ParameterList([
                nn.Parameter(torch.zeros(1), requires_grad=True) for _ in range(levels)
            ])
            self.r2d_gamma = nn.ParameterList([
                nn.Parameter(torch.zeros(1), requires_grad=True) for _ in range(levels)
            ])

        # === 五、新增：增强深度特征返回选项 ===
        # 是否返回增强后的深度特征 (默认False保持向后兼容)
        self.return_enhanced_depth = depth_processor_config.get('return_enhanced_depth', False) if depth_processor_config else False
        logger.info(f"RawEncoder: return_enhanced_depth = {self.return_enhanced_depth}")


    def forward(
        self,
        x: torch.Tensor,
        depth_gt: torch.Tensor = None,
        gt: torch.Tensor = None,
        depth_feats: list = None,
        return_enhanced_depth: bool = None,
        attention_mode: str = "both"  # 新增：控制注意力方向 "both", "d2r_only", "r2d_only"
    ) -> tuple:
        """
        前向传播，获取编码器特征
        Args:
            x: [B, C, H, W]  原始图特征 (通常是 SFE 输出)
            depth_gt: [B, 1, H, W] 原始深度图 (未使用)
            gt: [B, 3, H, W]  清晰图 ground truth (未使用，仅 API 兼容)
            depth_feats: list 多尺度深度特征（推理时可能用双次前向提供）
            return_enhanced_depth: bool 是否返回增强深度特征 (None时使用初始化配置)
            attention_mode: str 注意力方向控制
                - "both": 双向注意力 (默认，保持向后兼容)
                - "d2r_only": 仅Depth→RGB注意力 (Pass-2模式)
                - "r2d_only": 仅RGB→Depth注意力 (Pass-1模式，需要在DepthDecoder后调用)
        Returns:
            student_feats: List[Tensor[B, C_i, H/2^i, W/2^i]] - RGB / 图像各级特征
            enhanced_depth_feats: List[Tensor] 或 None - 增强后的深度特征 (可选)
        """

        # 确定是否返回增强深度特征
        should_return_enhanced_depth = (
            return_enhanced_depth if return_enhanced_depth is not None 
            else self.return_enhanced_depth
        )

        batch_size, _, height, width = x.shape
        device = x.device

        student_feats = []
        enhanced_depth_feats = [] if should_return_enhanced_depth else None
        current_x = x
        current_depth = None

        logger.debug(f"RawEncoder: attention_mode = {attention_mode}")

        # === 处理 depth_feats 或 depth_gt ===
        if depth_feats is not None:
            # 推理时双次前向：使用外部提供的 depth_feats[0] 作为第0级深度
            current_depth = depth_feats[0]
            logger.debug(f"RawEncoder: Using external depth features, level 0 shape: {current_depth.shape}")
        # elif depth_gt is not None:
        #     # 1) 先归一化/对数空间变换
        #     normalized_depth = ensure_normalized_depth(
        #         depth_gt,
        #         min_depth=self.depth_params.min_depth_log if self.depth_params else None,
        #         max_depth=self.depth_params.max_depth_log if self.depth_params else None,
        #         use_log_transform=self.depth_params.use_log_transform if self.depth_params else None,
        #         eps=self.depth_params.eps if self.depth_params else None,
        #         source_tag="RawEncoder"
        #     )

        #     # 2) 如果与 x 分辨率不一致，则插值到 x 的分辨率
        #     if normalized_depth.shape[2:] != x.shape[2:]:
        #         normalized_depth = F.interpolate(
        #             normalized_depth,
        #             size=(height, width),
        #             mode='bilinear',
        #             align_corners=False
        #         )

        #     # 3) 提取深度特征：从 1 通道映射到 base_channels
        #     current_depth = self.depth_feature_extractor(normalized_depth)

        # === 主循环：依次经过每一层 student_block，然后做定向跨模态注意力（若 depth 可用） ===
        for i, block in enumerate(self.student_blocks):
            # —— 1）通过 ConvBlock + RestormerBlock ——  
            current_x = block(current_x)  # [B, C_i, H_i, W_i]

            # —— 2）如果 depth 特征可用，则进行定向 Cross-Attention ——  
            if current_depth is not None:
                # 如果 depth_feats 存在且提供了多级特征，使用 depth_feats[i]
                if depth_feats is not None and i < len(depth_feats):
                    current_depth = depth_feats[i]

                # 2.1) 确保 current_depth 与 current_x 空间形状对齐
                if current_depth.shape[2:] != current_x.shape[2:]:
                    current_depth = F.interpolate(
                        current_depth,
                        size=current_x.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )

                # 2.2) 投影深度特征到当前通道数
                if i < len(self.depth_projections):
                    # 记录调试信息
                    if i == 0:
                        logger.debug(f"RawEncoder: Level {i} depth_feat shape before projection: {current_depth.shape}")
                        logger.debug(f"RawEncoder: Level {i} RGB feat shape: {current_x.shape}")
                    
                    # 检查通道数是否匹配投影层的输入通道 - 现在应该不会出现不匹配
                    if current_depth.shape[1] != self.depth_projections[i].weight.shape[1]:
                        logger.error(f"RawEncoder: Level {i} depth_feat channel mismatch. "
                                     f"Expected: {self.depth_projections[i].weight.shape[1]}, "
                                     f"Got: {current_depth.shape[1]}. "
                                     f"这不应该发生，请检查depth_feats的生成逻辑。")
                        raise ValueError(f"深度特征通道数不匹配：{current_depth.shape[1]} vs {self.depth_projections[i].weight.shape[1]}")
                    
                    current_depth_projected = self.depth_projections[i](current_depth)
                else:
                    # 万一 projection 数量不够，用 interpolate 保持不变
                    current_depth_projected = current_depth

                # —— 2.3) 根据attention_mode进行定向注意力 ——

                # Depth -> RGB 交叉注意力 (Pass-2或both模式)
                if attention_mode in ["both", "d2r_only"]:
                    fused_rgb = self.depth2rgb_attn_blocks[i](current_x, current_depth_projected)
                    gamma_d2r = self.d2r_gamma[i]       # 可学习标量
                    gated_rgb = gamma_d2r * fused_rgb
                    current_x = current_x + gated_rgb
                    logger.debug(f"RawEncoder: Level {i} Applied D2R attention, γ_d2r={gamma_d2r.item():.4f}")

                # RGB -> Depth 交叉注意力 (Pass-1后处理或both模式)
                if attention_mode in ["both", "r2d_only"]:
                    fused_depth = self.rgb2depth_attn_blocks[i](current_depth_projected, current_x)
                    gamma_r2d = self.r2d_gamma[i]
                    gated_depth = gamma_r2d * fused_depth
                    current_depth = current_depth_projected + gated_depth
                    logger.debug(f"RawEncoder: Level {i} Applied R2D attention, γ_r2d={gamma_r2d.item():.4f}")
                else:
                    # 如果不应用R2D注意力，保持原始投影深度特征
                    current_depth = current_depth_projected

                # —— 2.4) 保存增强后的深度特征 (如果需要) ——
                if should_return_enhanced_depth:
                    enhanced_depth_feats.append(current_depth.clone())
                    logger.debug(f"RawEncoder: Level {i} saved enhanced depth feat shape: {current_depth.shape}")

                # —— 2.5) 如果不是最后一层、且不是外部 depth_feats 模式，则下采样 depth 传递给下一层 ——  
                if i < self.levels - 1 and depth_feats is None:
                    # 对当前深度特征下采样
                    next_depth = F.avg_pool2d(current_depth, kernel_size=2, stride=2)
                    if i == 0:
                        logger.debug(f"RawEncoder: Depth downsampled for next level {i+1}, shape: {next_depth.shape}")
                    current_depth = next_depth

            else:
                # 如果没有深度特征但需要返回增强深度特征列表，添加None占位
                if should_return_enhanced_depth:
                    enhanced_depth_feats.append(None)

            # —— 3）存储当前层的 RGB 特征 ——  
            student_feats.append(current_x)

            # —— 4）检查通道数是否正确 ——  
            exp_ch = self.channels[i]
            if current_x.shape[1] != exp_ch:
                logger.warning(
                    f"RawEncoder: Level {i} student_feat channel mismatch. "
                    f"Expected: {exp_ch}, Got: {current_x.shape[1]}"
                )

        # 打印增强深度特征信息 (如果启用)
        if should_return_enhanced_depth and enhanced_depth_feats is not None:
            valid_feats = [f for f in enhanced_depth_feats if f is not None]
            logger.debug(f"RawEncoder: Returning {len(valid_feats)} enhanced depth features")

        # 返回所有级别的 RGB 特征和增强深度特征（可选）
        return student_feats, enhanced_depth_feats
