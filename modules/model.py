import torch
import torch.nn as nn
from .sfe import ShallowFeatureExtractor
from .depth_feature_extractor import DepthFeatureExtractor
from .depth_head import MonoDepthHead
from .encoder import RawEncoder
from .transformer_block import RestormerBlock
from .decoder import MultiTaskDecoder
from .recon_head import ReconHead

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
    """
    def __init__(self, 
                 base_channels: int = 48,
                 levels: int = 4,
                 heads: int = 8,
                 bottleneck_blocks: int = 4):
        super().__init__()
        # 1. 浅层特征
        self.sfe = ShallowFeatureExtractor(in_channels=3, out_channels=base_channels)

        # 2. 深度支路：训练期真深度多尺度提取 + 蒸馏/推理门控头
        self.depth_extractor = DepthFeatureExtractor(in_channels=1,
                                                     base_channels=base_channels,
                                                     levels=levels)
        self.depth_head = MonoDepthHead(in_channels=base_channels)

        # 3. 编码器：融合 RGB、Depth GT、GT 清晰图
        self.encoder = RawEncoder(in_channels=3,
                                  depth_channels=1,
                                  base_channels=base_channels,
                                  levels=levels,
                                  heads=heads)

        # 4. Joint Bottleneck
        self.bottleneck = nn.Sequential(
            *[RestormerBlock(base_channels) for _ in range(bottleneck_blocks)]
        )

        # 5. 解码器：解码级数 = levels - 1 (最多上采到原始分辨率)
        self.decoder = MultiTaskDecoder(base_channels=base_channels,
                                        levels=levels-1)

        # 6. 重建头
        self.recon = ReconHead()
        
        # 用于存储深度融合权重的缓存
        self.depth_fusion_weights = None

    def forward(self,
                raw: torch.Tensor,
                depth_gt: torch.Tensor = None,
                gt: torch.Tensor = None):
        # 1) 浅层特征
        f0 = self.sfe(raw)  # [B, C, H, W]

        # 2) 深度分支
        depth_feats = None
        if depth_gt is not None:
            # 有真实深度图时，都使用深度特征提取器获取多尺度特征
            # 不再仅限于训练模式
            depth_feats = self.depth_extractor(depth_gt)
        # 蒸馏 & 推理统一使用门控头预测
        pred_gate = self.depth_head(f0)  # [B,1,H,W]

        # 3) 编码器融合
        student_feats, teacher_feats = self.encoder(raw, depth_gt, gt)
        # student_feats: list of [B,C,H/2^i,W/2^i]
        # teacher_feats: 同维度或 None

        # 4) Bottleneck
        x = student_feats[-1]
        x = self.bottleneck(x)

        # 5) 解码
        if depth_feats is None:
            # 推理期没有真深度时，用预测门控图填充多尺度列表
            depth_feats = [pred_gate] * (self.decoder.levels + 1)
        res_d, res_c = self.decoder(fused_feat=x,
                                    skip_feats=student_feats,
                                    depth_feats=depth_feats,
                                    raw=raw)

        # 6) 重建融合
        out = self.recon(raw, res_d, res_c)
        return out, pred_gate, student_feats, teacher_feats
    
    def get_depth_fusion_weights(self):
        """
        返回最后一次计算的自适应深度融合权重
        
        Returns:
            torch.Tensor | None: 形状为 (B, L, D, H, W) 的权重张量
            B: 批次大小
            L: 层级数量
            D: 深度尺度数量
            H, W: 权重的空间尺寸
        """
        if hasattr(self.decoder, 'last_fusion_weights'):
            return self.decoder.last_fusion_weights
        return None
    
    def multi_forward(self, raw_batch, depth_gt=None, gt=None):
        """
        处理多个输入图像的批次处理
        
        Args:
            raw_batch: [N,3,H,W] 原始退化图像
            depth_gt: [1,1,H,W] 深度图（可选，训练时提供）
            gt: [1,3,H,W] 真实图像（可选，训练时提供）
            
        Returns:
            dict: {
                'outputs': [N,3,H,W], 
                'pred_gates': [N,1,H,W], 
                'student_feats': list of N lists, 
                'teacher_feats': list or None
            }
        """
        N = raw_batch.shape[0]
        enhanced_list = []
        pred_gate_list = []
        student_feats_list = []
        teacher_feats = None  # 公用一套GT特征
        
        # 逐个处理输入
        for n in range(N):
            raw_n = raw_batch[n:n+1]  # [1,3,H,W]
            enhanced, pred_gate, student_feats, teacher_feats_n = self.forward(
                raw_n, depth_gt, gt
            )
            enhanced_list.append(enhanced)
            pred_gate_list.append(pred_gate)
            student_feats_list.append(student_feats)
            
            # 仅记录一次教师特征
            if teacher_feats is None and teacher_feats_n is not None:
                teacher_feats = teacher_feats_n
        
        # 堆叠结果
        outputs = torch.cat(enhanced_list, dim=0)  # [N,3,H,W]
        pred_gates = torch.cat(pred_gate_list, dim=0)  # [N,1,H,W]
        
        return {
            'outputs': outputs,
            'pred_gates': pred_gates,
            'student_feats': student_feats_list,
            'teacher_feats': teacher_feats
        }
