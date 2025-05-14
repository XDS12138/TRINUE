import torch
import torch.nn as nn

def default_activation(x: torch.Tensor) -> torch.Tensor:
    """
    默认激活函数，将输入张量控制在 [-1,1] 范围，模拟 tanh 行为
    """
    return torch.tanh(x)

class ReconHead(nn.Module):
    """
    Reconstruction Head
    -------------------
    将原图与去模糊和颜色残差合成，并执行激活。

    输入:
      - raw:   Tensor[B,3,H,W] 原始图 (归一化到 [-1,1])
      - res_d: Tensor[B,3,H,W] 高频去模糊残差
      - res_c: Tensor[B,3,H,W] 低频颜色残差
    输出:
      - out:   Tensor[B,3,H,W] 最终增强图 ([-1,1])

    默认操作: out = tanh(raw + res_d + res_c)
    可定制 activation
    """
    def __init__(self, activation=default_activation):
        super().__init__()
        self.activation = activation

    def forward(self,
                raw: torch.Tensor,
                res_d: torch.Tensor,
                res_c: torch.Tensor) -> torch.Tensor:
        # 合成 residuals
        x = raw + res_d + res_c
        # 激活到 [-1,1]
        return self.activation(x)

# 示例用法:
# recon = ReconHead()
# enhanced = recon(raw, res_d, res_c)  # Tensor[B,3,H,W]
