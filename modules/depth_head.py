import torch
import torch.nn as nn

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
    def __init__(self, in_channels: int, mid_channels: int = None):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 下采阶段，获取粗粒度深度特征
        x = self.encoder(x)
        # 上采阶段，回复至原分辨率
        x = self.decoder(x)
        # 生成门控 logits → Sigmoid to [0,1]
        x = self.proj(x)
        g_depth = self.act(x)
        return g_depth

# 示例用法:
# head = MonoDepthHead(in_channels=48)
# G_depth = head(F0)  # Tensor[B,1,H,W]
