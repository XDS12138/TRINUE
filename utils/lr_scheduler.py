# utils/lr_scheduler.py
"""
Learning‑rate schedulers in one place.

Usage
-----
from utils.lr_scheduler import get_scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
scheduler = get_scheduler(
    name="cosine_warmup",
    optimizer=optimizer,
    num_epochs=cfg.epochs,
    warmup_epochs=cfg.warmup_epochs,
    min_lr=1e‑6
)
for epoch in range(cfg.epochs):
    train(...)
    validate(...)
    scheduler.step()          # 一行即可
"""

import math
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, MultiStepLR, StepLR, ReduceLROnPlateau

# -----------------------------------------------------------------------------#
# Warm‑up wrapper
# -----------------------------------------------------------------------------#
class WarmupWrapper(_LRScheduler):
    """
    Wrap any scheduler with a linear warm‑up phase.

    Args
    ----
    base_scheduler : _LRScheduler
        the scheduler to wrap after warm‑up.
    warmup_epochs : int
        number of epochs to warm‑up linearly from 0 → base_lr.
    last_epoch : int
        same meaning as in PyTorch schedulers.
    """

    def __init__(self,
                 optimizer,
                 base_scheduler: _LRScheduler,
                 warmup_epochs: int,
                 last_epoch: int = -1):
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = float(self.last_epoch + 1) / float(self.warmup_epochs)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        # Delegate to the wrapped scheduler
        return self.base_scheduler.get_last_lr()

    def step(self, epoch=None):
        # Step our own epoch counter
        super().step(epoch)
        # Make base_scheduler "think" it is at (epoch‑warmup_epochs)
        if self.last_epoch >= self.warmup_epochs:
            self.base_scheduler.last_epoch = self.last_epoch - self.warmup_epochs
            self.base_scheduler.step(None)


# -----------------------------------------------------------------------------#
# Factory
# -----------------------------------------------------------------------------#
def get_scheduler(name: str,
                  optimizer,
                  num_epochs: int,
                  warmup_epochs: int = 0,
                  min_lr: float = 0.0,
                  milestones: tuple[int] | None = None,
                  gamma: float = 0.1,
                  patience: int = 5,  
                  factor: float = 0.5):
    """
    Build a scheduler by name.

    Parameters
    ----------
    name : {"cosine", "cosine_warmup", "multistep", "step", "plateau"}
    optimizer : torch.optim.Optimizer
    num_epochs : total training epochs
    warmup_epochs : linear warm‑up epochs (only *_warmup)
    min_lr : final minimal lr for cosine
    milestones : tuple, epoch indices for multistep
    gamma : decay factor for multistep / step
    patience : patience epochs for plateau scheduler
    factor : lr reduction factor for plateau scheduler
    """
    name = name.lower()
    if name == "cosine":
        sched = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)
    elif name == "cosine_warmup":
        base = CosineAnnealingLR(optimizer,
                                 T_max=num_epochs - warmup_epochs,
                                 eta_min=min_lr)
        sched = WarmupWrapper(optimizer, base_scheduler=base, warmup_epochs=warmup_epochs)
    elif name == "multistep":
        if milestones is None:
            milestones = (int(num_epochs * 0.6), int(num_epochs * 0.8))
        sched = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif name == "step":
        step_size = int(num_epochs * 0.3)
        sched = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == "plateau":
        sched = ReduceLROnPlateau(
            optimizer,
            mode='max',  # 假设使用验证指标，值越大越好（如PSNR）
            factor=factor,  # 学习率减小的因子
            patience=patience,  # 在降低学习率之前等待指标无改善的轮次
            min_lr=min_lr  # 最小学习率
        )
    else:
        raise ValueError(f"Unknown scheduler '{name}'")
    return sched
