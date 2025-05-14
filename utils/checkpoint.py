import os
import torch

def save_checkpoint(state: dict,
                    is_best: bool,
                    checkpoint_dir: str,
                    filename: str = "checkpoint.pth.tar",
                    best_filename: str = "model_best.pth.tar",
                    epoch: int = None) -> None:
    """
    Save training checkpoint.

    Args:
        state (dict): Contains model state_dict, optimizer state_dict, scheduler state_dict, epoch, best_metric, etc.
        is_best (bool): True if this checkpoint has the best metric so far.
        checkpoint_dir (str): Directory to save checkpoints.
        filename (str): Filename for the current checkpoint.
        best_filename (str): Filename for the best-performing checkpoint.
        epoch (int, optional): Current epoch number. If provided, will save as checkpoint_{epoch}.pth.tar
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 如果提供了epoch，创建带有epoch编号的文件名
    if epoch is not None:
        filename = f"checkpoint_epoch{epoch}.pth.tar"
        
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(checkpoint_dir, best_filename)
        torch.save(state, best_path)


def load_checkpoint(checkpoint_path: str,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer = None,
                    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                    map_location: str = None) -> dict:
    """
    Load training checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (nn.Module): Model to load state_dict into.
        optimizer (Optimizer, optional): Optimizer to load state_dict into.
        scheduler (_LRScheduler, optional): Scheduler to load state_dict into.
        map_location (str, optional): Device mapping for loading (e.g., "cpu" or "cuda:0").

    Returns:
        dict: The checkpoint dictionary containing epoch, best_metric, etc.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Handle DataParallel/DistributedDataParallel prefixes
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove module. prefix if present
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint


def resume_from_checkpoint(checkpoint_dir: str,
                           model: torch.nn.Module,
                           optimizer: torch.optim.Optimizer = None,
                           scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                           map_location: str = None) -> (int, float):
    """
    Resume training from the latest checkpoint in a directory.

    Args:
        checkpoint_dir (str): Directory containing checkpoint files.
        model (nn.Module): Model to load state into.
        optimizer (Optimizer, optional): Optimizer to load state into.
        scheduler (_LRScheduler, optional): Scheduler to load state into.
        map_location (str, optional): Device mapping for loading.

    Returns:
        epoch (int): Epoch to resume from.
        best_metric (float): Best validation metric so far.
    """
    # Find latest checkpoint by timestamp or filename
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    if not files:
        return 0, float("inf")  # or 0, 0 depending on metric convention

    # Assuming checkpoints named 'checkpoint.pth.tar' and 'model_best.pth.tar'
    latest = max(files, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    checkpoint_path = os.path.join(checkpoint_dir, latest)
    checkpoint = load_checkpoint(checkpoint_path, model, optimizer, scheduler, map_location)

    epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", float("inf"))
    return epoch, best_metric

# Example usage in train.py:
#
# from utils.checkpoint import save_checkpoint, resume_from_checkpoint
#
# start_epoch, best_psnr = resume_from_checkpoint(cfg.checkpoint_dir, model, optimizer, scheduler, map_location=device)
# for epoch in range(start_epoch, cfg.epochs):
#     train_one_epoch(...)
#     val_psnr = validate(...)
#     is_best = val_psnr > best_psnr
#     best_psnr = max(val_psnr, best_psnr)
#     save_checkpoint({
#         'epoch': epoch + 1,
#         'state_dict': model.state_dict(),
#         'best_metric': best_psnr,
#         'optimizer': optimizer.state_dict(),
#         'scheduler': scheduler.state_dict(),
#     }, is_best, cfg.checkpoint_dir)

