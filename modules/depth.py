import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import types

# 获取一个logger实例
logger = logging.getLogger(__name__)

# 默认的深度处理参数 (可以从config中覆盖)
# 🌊 水下图像深度范围：0.1-30米 (深色距离近，浅色距离远)
DEFAULT_MIN_DEPTH = 0.1  # 最近距离 0.1米
DEFAULT_MAX_DEPTH = 30.0  # 最远距离 30米
DEFAULT_EPS = 1e-6
DEFAULT_USE_LOG_TRANSFORM = True  # 使用对数变换增强近距离细节

#######################
# Depth Utilities
#######################

def ensure_normalized_depth(
    depth_tensor: torch.Tensor,
    min_depth: float = None,
    max_depth: float = None,
    use_log_transform: bool = None,
    eps: float = None,
    source_tag: str = "unknown"
) -> torch.Tensor:
    """
    Ensures that the depth tensor is normalized, typically to the [0, 1] range.
    Applies log transformation if specified. Marks the tensor with _depth_processed = True.

    Args:
        depth_tensor (torch.Tensor): The input depth tensor. Can be raw or already processed.
        min_depth (float, optional): Minimum depth value for normalization.
                                     Defaults to DEFAULT_MIN_DEPTH.
        max_depth (float, optional): Maximum depth value for normalization.
                                     Defaults to DEFAULT_MAX_DEPTH.
        use_log_transform (bool, optional): Whether to apply log transform before normalization.
                                            Defaults to DEFAULT_USE_LOG_TRANSFORM.
        eps (float, optional): Epsilon value to prevent division by zero.
                               Defaults to DEFAULT_EPS.
        source_tag (str, optional): A tag to indicate where this function is called from, for logging.

    Returns:
        torch.Tensor: The normalized depth tensor.
    """
    if depth_tensor is None:
        logger.debug(f"[{source_tag}] Depth tensor is None, returning None.")
        return None

    # Populate defaults if not provided
    min_depth = min_depth if min_depth is not None else DEFAULT_MIN_DEPTH
    max_depth = max_depth if max_depth is not None else DEFAULT_MAX_DEPTH
    use_log_transform = use_log_transform if use_log_transform is not None else DEFAULT_USE_LOG_TRANSFORM
    eps = eps if eps is not None else DEFAULT_EPS

    # 1. Check if already processed by attribute
    if hasattr(depth_tensor, '_depth_processed') and depth_tensor._depth_processed:
        logger.debug(f"[{source_tag}] Depth tensor already marked as processed. Shape: {depth_tensor.shape}, Range: [{depth_tensor.min().item():.4f}, {depth_tensor.max().item():.4f}]")
        return depth_tensor

    # 2. Heuristic check by value range (if not marked by attribute)
    #    A common heuristic for normalized depth is [0, C] where C is small (e.g., 1.0 or similar).
    #    Raw depth (e.g., from Kinect or in millimeters) is usually much larger.
    current_min_val = depth_tensor.min().item()
    current_max_val = depth_tensor.max().item()

    # If max value is very small (e.g. < 2.0) and min is non-negative, it's likely normalized.
    # This threshold (2.0) is a heuristic and might need adjustment based on typical raw depth scales.
    # Consider min_depth_config from YAML which is e.g. 5000.0 for raw depth.
    if current_max_val < min_depth and current_max_val <= 2.0 and current_min_val >= 0: # Heuristic
        logger.debug(f"[{source_tag}] Depth tensor heuristically determined as ALREADY normalized. Shape: {depth_tensor.shape}, Range: [{current_min_val:.4f}, {current_max_val:.4f}]")
        depth_tensor._depth_processed = True
        return depth_tensor
    
    logger.debug(f"[{source_tag}] Normalizing depth tensor. Input range: [{current_min_val:.4f}, {current_max_val:.4f}], Shape: {depth_tensor.shape}")
    logger.debug(f"[{source_tag}] Params: min_depth={min_depth}, max_depth={max_depth}, use_log={use_log_transform}, eps={eps}")

    processed_depth = depth_tensor.clone() # Work on a copy

    if use_log_transform:
        # Ensure min_depth and max_depth are positive for log transform
        log_min_val = torch.log(torch.tensor(min_depth, device=processed_depth.device) + eps)
        log_max_val = torch.log(torch.tensor(max_depth, device=processed_depth.device) + eps)
        
        # Apply log transform to the input depth
        processed_depth = torch.log(processed_depth + eps)
        
        # Normalize in log space
        normalized_depth = (processed_depth - log_min_val) / (log_max_val - log_min_val + eps)
    else:
        # Linear normalization
        normalized_depth = (processed_depth - min_depth) / (max_depth - min_depth + eps)

    # Clamp to [0, 1] for safety, though ideally normalization should achieve this.
    normalized_depth = torch.clamp(normalized_depth, 0.0, 1.0)
    
    # 💡 深度语义说明：
    # - 0.0 对应 0.1米（最近距离，深色，高置信度）
    # - 1.0 对应 30米（最远距离，浅色，低置信度）
    # - 水下成像：距离越近受散射影响越小，增强效果越可靠
    
    # Mark as processed
    normalized_depth._depth_processed = True
    
    logger.debug(f"[{source_tag}] Normalized depth tensor. Output range: [{normalized_depth.min().item():.4f}, {normalized_depth.max().item():.4f}]")
    
    return normalized_depth

def get_depth_config_params(config_dict: dict) -> types.SimpleNamespace:
    """
    Extracts depth processing parameters from a configuration dictionary.
    Uses defaults if specific keys are missing.
    Returns a SimpleNamespace object for attribute-style access.
    """
    if config_dict is None:
        config_dict = {}
        
    # Get base min/max depth first, as they might be defaults for log/linear versions
    base_min_depth = float(config_dict.get('min_depth', DEFAULT_MIN_DEPTH))
    base_max_depth = float(config_dict.get('max_depth', DEFAULT_MAX_DEPTH))

    params_dict = {
        'min_depth': base_min_depth,
        'max_depth': base_max_depth,
        'use_log_transform': bool(config_dict.get('use_log_transform', DEFAULT_USE_LOG_TRANSFORM)),
        'eps': float(config_dict.get('eps', DEFAULT_EPS)),

        # Add specific log and linear depth parameters
        # They default to the base min/max_depth if not specified in config_dict
        'min_depth_log': float(config_dict.get('min_depth_log', base_min_depth)),
        'max_depth_log': float(config_dict.get('max_depth_log', base_max_depth)),
        'min_depth_linear': float(config_dict.get('min_depth_linear', base_min_depth)),
        'max_depth_linear': float(config_dict.get('max_depth_linear', base_max_depth)),
    } 
    return types.SimpleNamespace(**params_dict)

#######################
# Depth Preprocessor
#######################

class DepthPreprocessor(nn.Module):
    """
    深度图预处理模块
    
    针对16位深度图的特性
    """
    def __init__(self, input_channels=1, **kwargs):
        super().__init__()
        self.input_channels = input_channels
        # Store kwargs to pass them to ensure_normalized_depth
        self.config_params = get_depth_config_params(kwargs) # Extract and type-cast
        logger.info(f"DepthPreprocessor initialized with config: {self.config_params}")
        
        # 固定深度统计范围（避免不同批次间的不一致性）
        self.register_buffer('depth_min', torch.tensor(self.config_params.min_depth))  # Changed to attribute access
        self.register_buffer('depth_max', torch.tensor(self.config_params.max_depth))  # Changed to attribute access
        
        # 预处理卷积层 - 学习边缘特征
        self.preprocess = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, input_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x is None:
            logger.debug("[DPP] Input tensor is None, returning None.")
            return None
        
        # Ensure the input is at least 3D [C, H, W] or 4D [B, C, H, W]
        if x.dim() == 2: # H, W -> 1, 1, H, W (assuming batch 1, channel 1)
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3: # C, H, W -> B, C, H, W (assuming batch 1)
            if x.shape[0] == self.input_channels: # Correct channel size
                x = x.unsqueeze(0)
            else: # Potentially H, W, C, needs reshape or error
                # This case should be handled by data loader or raise error
                logger.warning(f"[DPP] Input tensor has 3 dims but first dim ({x.shape[0]}) != input_channels ({self.input_channels}). Unexpected shape.")
                # Attempting to treat as B,H,W -> B,1,H,W if channels=1
                if self.input_channels == 1:
                    x = x.unsqueeze(1)
                else:
                    # Cannot safely infer, return as is or raise error
                    logger.error("[DPP] Cannot infer tensor shape for 3D input with multiple channels.")
                    return x # Or raise error
        elif x.dim() == 4:
            if x.shape[1] != self.input_channels:
                logger.warning(f"[DPP] Input tensor has 4 dims but channel dim ({x.shape[1]}) != input_channels ({self.input_channels}).")
                # If input_channels is 1 and tensor has 3 channels (e.g. RGB depth), take first channel
                if self.input_channels == 1 and x.shape[1] == 3:
                    logger.info("[DPP] Taking the first channel of a 3-channel input for single-channel depth.")
                    x = x[:, 0:1, :, :] 
                # else: error or pass through
        
        # Select appropriate min/max depth based on log_transform flag
        if self.config_params.use_log_transform:
            current_min_depth = self.config_params.min_depth_log
            current_max_depth = self.config_params.max_depth_log
        else:
            current_min_depth = self.config_params.min_depth_linear
            current_max_depth = self.config_params.max_depth_linear

        # Use the centralized normalization function
        # Pass only the parameters expected by ensure_normalized_depth
        normalized_x = ensure_normalized_depth(
            x,
            min_depth=current_min_depth,
            max_depth=current_max_depth,
            use_log_transform=self.config_params.use_log_transform,
            eps=self.config_params.eps,
            source_tag="DepthPreprocessor"
        )
        return normalized_x

#######################
# DepthFeatureExtractor
#######################

class DepthFeatureExtractor(nn.Module):
    """
    从深度图提取多尺度特征
    
    输入深度图，输出多尺度特征列表，从细到粗
    每个尺度的特征都具有固定通道数base_channels
    
    Args:
        in_channels: 输入通道数，通常为1（灰度深度图）
        base_channels: 基础通道数，所有尺度的特征都使用这个通道数
        levels: 特征层级数，决定输出列表长度
        **processor_kwargs: 传递给DepthPreprocessor的参数
    """
    def __init__(self, in_channels=1, base_channels=64, levels=4, **processor_kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.levels = levels
        
        # 深度预处理器
        self.preprocessor = DepthPreprocessor(input_channels=in_channels, **processor_kwargs)
        # 添加别名以兼容 model.multi_forward 方法
        self.depth_preprocessor = self.preprocessor
        
        # 初始特征提取（保持通道数为base_channels）
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # 下采样层
        self.downsample = nn.ModuleList()
        for i in range(levels-1):
            self.downsample.append(
                nn.Sequential(
                    nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
        
        # 确保所有特征都是base_channels通道数的投影层
        self.projection = nn.ModuleList()
        for i in range(levels):
            self.projection.append(
                nn.Conv2d(base_channels, base_channels, kernel_size=1, bias=False)
            )
        
        # 添加通道适配器，避免在forward中动态创建
        self.channel_adapters = nn.ModuleDict()
        # 预先创建常见的通道配置
        common_channels = [48, 96, 192, 384]  # 常见的特征通道数
        for in_ch in common_channels:
            if in_ch != base_channels:
                adapter_key = f"adapt_{in_ch}_to_{base_channels}"
                self.channel_adapters[adapter_key] = nn.Conv2d(in_ch, base_channels, kernel_size=1, bias=False)
        
        logger.info(f"DepthFeatureExtractor initialized with {levels} levels, "
                   f"fixed base_channels={base_channels}")

    def _get_or_create_adapter(self, in_channels: int, out_channels: int, device: torch.device) -> nn.Module:
        """获取通道适配器，禁用动态创建以修复显存跳变"""
        adapter_key = f"adapt_{in_channels}_to_{out_channels}"
        
        if adapter_key not in self.channel_adapters:
            # 🔥 修复显存跳变：不再动态创建适配器！
            logger.error(f"❌ Missing channel adapter {adapter_key} in DepthFeatureExtractor! "
                        f"This should not happen with proper preprocessing. "
                        f"Available adapters: {list(self.channel_adapters.keys())}")
            
            # 应急处理：如果输入输出通道数相同，返回恒等映射
            if in_channels == out_channels:
                logger.warning(f"⚠️ Using identity mapping for {in_channels} == {out_channels} channels in DepthFeatureExtractor")
                if not hasattr(self, '_identity_adapter'):
                    self._identity_adapter = nn.Identity()
                return self._identity_adapter
            else:
                # 如果通道数不同，这是一个严重错误，应该停止训练
                raise RuntimeError(f"🚨 DepthFeatureExtractor channel mismatch: {in_channels} -> {out_channels}. "
                                 f"All depth features should have {self.base_channels} channels. "
                                 f"Dynamic adapter creation has been disabled to fix memory jumps.")
        
        return self.channel_adapters[adapter_key]

    def forward(self, x: torch.Tensor) -> list:
        """
        提取多尺度深度特征
        
        Args:
            x: [B,1,H,W] 深度图
        
        Returns:
            list of [B,base_channels,H/(2^i),W/(2^i)]: 多尺度特征，从细到粗
        """
        if x is None:
            logger.warning("DepthFeatureExtractor received None input, returning empty list")
            return []
        
        # 预处理
        x = self.preprocessor(x)
        if x is None:
            logger.warning("DepthPreprocessor returned None, returning empty list")
            return []
        
        # 初始特征
        features = []
        x = self.init_conv(x)
        
        # 添加第一级特征（原始分辨率）
        features.append(self.projection[0](x))
        logger.debug(f"DepthFeatureExtractor: Level 0 feature shape: {features[0].shape}")
        
        # 下采样并添加后续特征
        for i in range(self.levels-1):
            if i < len(self.downsample):
                x = self.downsample[i](x)
                # 确保每个特征都有固定的通道数 base_channels
                projected_feature = self.projection[i+1](x)
                features.append(projected_feature)
                logger.debug(f"DepthFeatureExtractor: Level {i+1} feature shape: {projected_feature.shape}")
        
        # 确保输出exactly self.levels个特征
        if len(features) < self.levels:
            logger.warning(f"DepthFeatureExtractor: Expected {self.levels} features, but got {len(features)}. Padding with last feature.")
            last_feature = features[-1]
            while len(features) < self.levels:
                features.append(last_feature)
        elif len(features) > self.levels:
            logger.warning(f"DepthFeatureExtractor: Generated {len(features)} features, but only {self.levels} expected. Truncating.")
            features = features[:self.levels]
        
        # 检查通道数是否符合预期
        for i, feat in enumerate(features):
            expected_channels = self.base_channels
            if feat.shape[1] != expected_channels:
                logger.error(f"DepthFeatureExtractor: Feature level {i} has {feat.shape[1]} channels instead of expected {expected_channels}!")
                # 如果通道数不匹配，尝试修复
                if feat.shape[1] < expected_channels:
                    logger.warning(f"DepthFeatureExtractor: Attempting to adapt channel count from {feat.shape[1]} to {expected_channels}")
                    temp_adapter = self._get_or_create_adapter(feat.shape[1], expected_channels, feat.device)
                    features[i] = temp_adapter(feat)
                else:
                    # 通道数过多，截断
                    logger.warning(f"DepthFeatureExtractor: Truncating channels from {feat.shape[1]} to {expected_channels}")
                    features[i] = feat[:, :expected_channels]
        
        # 记录最终的特征形状
        logger.debug(f"DepthFeatureExtractor returning {len(features)} features with shapes: {[f.shape for f in features]}")
        return features

#######################
# DepthGate (已废弃 - 深度融合功能已被禁用)
#######################

# class DepthGate(nn.Module):
#     """
#     DepthGate 模块 - 已废弃
#     ---------------
#     原用于深度门控的跳跃连接融合，现已被禁用
#     """
#     def __init__(self, in_channels_depth: int, reduction: int = 1):
#         super().__init__()
#         # depth_feat 通道映射到 1
#         self.gate_conv = nn.Conv2d(in_channels_depth, 1, kernel_size=1, bias=False)
#         self.sigmoid = nn.Sigmoid()
# 
#     def forward(self, skip_feat: torch.Tensor, depth_feat: torch.Tensor) -> torch.Tensor:
#         # depth_feat: [B, C_d, H, W]
#         gate_map = self.gate_conv(depth_feat)  # [B,1,H,W]
#         gate_map = self.sigmoid(gate_map)
#         
#         # 处理空间尺寸不匹配
#         if gate_map.shape[2:] != skip_feat.shape[2:]:
#             gate_map = F.interpolate(gate_map, size=skip_feat.shape[2:], 
#                                      mode='bilinear', align_corners=False)
#         
#         # 广播乘法
#         return skip_feat * gate_map

#######################
# MonoDepthHead
#######################

class MonoDepthHead(nn.Module):
    """
    MonoDepthHead (深度门控蒸馏 & 单输入推理版)
    -----------------------------------------
    输入浅层特征 F0 (B×C×H×W)，输出深度门控图 G_depth (B×1×H×W)
    • 训练期：用真深度 D_gt 蒸馏 G_depth
    • 推理期：只需 I_raw，自动预测门控图

    结构:
      - 下采阶段：Conv3×3(s=2)→ReLU → Conv3×3(s=2)→ReLU
      - 上采阶段：ConvTranspose2×2→GELU → ConvTranspose2×2→GELU
      - 输出层：Conv1×1 → Sigmoid
    """
    def __init__(self, in_channels: int, mid_channels: int = None, enhance_contrast: bool = True):
        super().__init__()
        mid = mid_channels or in_channels
        # 下采阶段: H×W -> H/2×W/2 -> H/4×W/4
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        # 上采阶段: H/4×W/4 -> H/2×W/2 -> H×W
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(mid * 2, mid, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(mid, mid, kernel_size=2, stride=2),
            nn.GELU(),
        )
        # 输出门控图
        self.proj = nn.Conv2d(mid, 1, kernel_size=1)
        self.act = nn.Sigmoid()
        
        # 使用直方图均衡化增强对比度的标志
        self.enhance_contrast = enhance_contrast
        self.register_buffer('running_min', torch.tensor(0.0))
        self.register_buffer('running_max', torch.tensor(1.0))
        self.momentum = 0.9  # 运行统计更新动量

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 确保输入和权重的数据类型匹配，解决混合精度训练问题
        encoder_dtype = next(self.encoder.parameters()).dtype
        if x.dtype != encoder_dtype:
            x = x.to(dtype=encoder_dtype)
            
        # 下采阶段，获取粗粒度深度特征
        x = self.encoder(x)
        # 上采阶段，回复至原分辨率
        x = self.decoder(x)
        # 生成门控 logits → Sigmoid to [0,1]
        x = self.proj(x)
        g_depth = self.act(x)
        
        # 增强对比度（如果启用），帮助可视化
        if self.enhance_contrast and self.training:
            # 在训练时更新运行统计
            with torch.no_grad():
                batch_min = g_depth.min().item()
                batch_max = g_depth.max().item()
                # 更新运行最小/最大值
                self.running_min = self.momentum * self.running_min + (1 - self.momentum) * batch_min
                self.running_max = self.momentum * self.running_max + (1 - self.momentum) * batch_max
        
        if self.enhance_contrast:
            # 基于运行统计信息应用对比度增强
            # 创建一个增强版本，不影响原始输出（避免干扰训练）
            g_depth_enhanced = g_depth.clone()
            
            # 标准化增强对比度
            min_val = self.running_min if self.training else g_depth.min()
            max_val = self.running_max if self.training else g_depth.max()
            
            # 如果范围太小，进行更强的对比度增强
            if max_val - min_val < 0.1:
                # 使用通用的对比度增强
                # 减去均值，除以标准差，再缩放回[0,1]范围
                mean_val = g_depth_enhanced.mean()
                std_val = g_depth_enhanced.std()
                
                if std_val > 0:  # 避免除以零
                    g_depth_enhanced = (g_depth_enhanced - mean_val) / (std_val + 1e-8)
                    g_depth_enhanced = torch.clamp(g_depth_enhanced, -3, 3)  # 限制在±3个标准差内
                    g_depth_enhanced = (g_depth_enhanced + 3) / 6  # 从[-3,3]映射到[0,1]
            else:
                # 线性拉伸对比度
                g_depth_enhanced = (g_depth_enhanced - min_val) / (max_val - min_val + 1e-8)
            
            # 为了调试和可视化目的，将增强后的版本存储为原始门控的属性
            g_depth.enhanced = g_depth_enhanced
        
        return g_depth 