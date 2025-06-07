import os
import logging
import csv
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple
from matplotlib import cm
import sys
import warnings
import logging.handlers

DEFAULT_LOG_FORMAT = '%(asctime)s [%(levelname)s] [%(name)s] %(message)s (%(filename)s:%(lineno)d)'
DEFAULT_LOG_FORMAT_CONSOLE = '%(asctime)s [%(levelname)s] [%(name)s] %(message)s'

# Store the original sys.excepthook
original_sys_excepthook = sys.excepthook

def custom_excepthook(exc_type, exc_value, exc_traceback):
    # Log the exception using the root logger
    # Using a logger named 'CRITICAL' or similar for unhandled exceptions
    exception_logger = logging.getLogger('UnhandledException')
    exception_logger.critical("Unhandled exception:", exc_info=(exc_type, exc_value, exc_traceback))
    # Call the original excepthook to ensure Python's default behavior (e.g., printing to stderr)
    original_sys_excepthook(exc_type, exc_value, exc_traceback)

def setup_logger(config, log_dir, main_logger_name=None, include_pid=False):
    # Centralized log configuration
    log_level_str = config.get('log_level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    console_log_level_str = config.get('console_log_level', 'INFO').upper()
    console_log_level_actual = getattr(logging, console_log_level_str, logging.INFO)

    log_format_file = config.get('log_format_file', DEFAULT_LOG_FORMAT)
    log_format_console = config.get('log_format_console', DEFAULT_LOG_FORMAT_CONSOLE)

    # === Aggressive Pre-emptive Silencing and Handler Clearing ===
    # Clear handlers on root logger first
    if main_logger_name is None or main_logger_name == "" or main_logger_name == logging.root.name:
        logging.root.handlers = []

    # Clear handlers and silence specific known noisy loggers before any other setup
    for logger_name_to_clear in ['py.warnings', 'PIL.PngImagePlugin']:
        existing_logger = logging.getLogger(logger_name_to_clear)
        existing_logger.handlers = []      # Remove any pre-existing handlers
        existing_logger.propagate = False  # Stop propagation immediately
        existing_logger.setLevel(logging.CRITICAL + 1) # Silence until we reconfigure
    # =============================================================

    # Fallback basicConfig for the root logger. This will only add a handler if root has none.
    # Since we cleared root's handlers, this will set a default stderr handler.
    # Its level is WARNING, so it won't be too noisy if other things aren't caught.
    logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(name)s: %(message)s')

    # Main/Root Logger (application logger)
    # If main_logger_name is None or empty string, it refers to the root logger.
    if main_logger_name is None or main_logger_name == "":
        main_logger = logging.getLogger() # Get the root logger
    else:
        main_logger = logging.getLogger(main_logger_name)

    main_logger.setLevel(log_level) # Main logger processes messages from this level onwards

    # Prevent main_logger from propagating to root if it's not the root logger itself.
    # If it IS the root logger, propagate is irrelevant for it.
    if main_logger_name is not None and main_logger_name != "":
        main_logger.propagate = False


    # --- Console Handler for Main Logger ---
    # This handler is specifically for the main application logs to the console.
    main_logger_console_handler = logging.StreamHandler(sys.stdout)
    main_logger_console_handler.setFormatter(logging.Formatter(log_format_console))
    main_logger_console_handler.setLevel(console_log_level_actual)
    main_logger.addHandler(main_logger_console_handler)
    if main_logger_name is None or main_logger_name == "": # If root logger
        logging.info(f"Root logger console log level set to: {console_log_level_str}")
    else:
        main_logger.info(f"Logger '{main_logger_name}' console log level set to: {console_log_level_str}")


    # --- File Handlers (common for main_logger, py.warnings, PIL) ---
    # Ensure log_dir exists
    os.makedirs(log_dir, exist_ok=True)

    # Create a unique sub-directory for each run using a timestamp
    # or use a simpler scheme if preferred.
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(log_dir, f"run_{current_time_str}")
    if include_pid:
        run_log_dir += f"_pid{os.getpid()}"
    os.makedirs(run_log_dir, exist_ok=True)

    # Debug log file (all levels from log_level)
    debug_log_file = os.path.join(run_log_dir, config.get('debug_log_filename', 'debug.log'))
    debug_file_handler = logging.handlers.RotatingFileHandler(
        debug_log_file,
        maxBytes=config.get('log_max_bytes', 10*1024*1024), # 10MB
        backupCount=config.get('log_backup_count', 5)
    )
    debug_file_handler.setFormatter(logging.Formatter(log_format_file))
    debug_file_handler.setLevel(log_level) # Capture from main log_level (e.g. DEBUG or INFO)
    main_logger.addHandler(debug_file_handler)

    # Info log file (INFO and above) - or rather, a general "train.log"
    # For this file, let's set its level to INFO, regardless of the main log_level,
    # unless main log_level is higher (e.g. WARNING), in which case it takes precedence.
    info_file_log_level = max(logging.INFO, log_level) # Ensures it's at least INFO
    info_log_file = os.path.join(run_log_dir, config.get('info_log_filename', 'train.log'))
    info_file_handler = logging.handlers.RotatingFileHandler(
        info_log_file,
        maxBytes=config.get('log_max_bytes', 10*1024*1024),
        backupCount=config.get('log_backup_count', 5)
    )
    info_file_handler.setFormatter(logging.Formatter(log_format_file))
    info_file_handler.setLevel(info_file_log_level)
    main_logger.addHandler(info_file_handler)

    main_logger.info(f"Logging initialized. File logs in: {run_log_dir}")


    # --- Setup for 'py.warnings' logger ---
    # This logger is used by the custom warnings.showwarning handler
    warnings_logger = logging.getLogger('py.warnings')
    warnings_logger.handlers = [] # Ensure it's clean before adding our handlers
    warnings_logger.setLevel(logging.DEBUG) # Capture all warnings internally

    # Console handler for py.warnings
    py_warnings_console_handler = logging.StreamHandler(sys.stdout)
    py_warnings_console_handler.setFormatter(logging.Formatter(log_format_console))
    # Use the same console_log_level as the main logger for warnings on console
    py_warnings_console_handler.setLevel(console_log_level_actual)
    warnings_logger.addHandler(py_warnings_console_handler)

    # File handlers for py.warnings
    warnings_logger.addHandler(debug_file_handler) # Already configured with its level
    warnings_logger.addHandler(info_file_handler)  # Already configured with its level

    warnings_logger.propagate = False # Do not propagate to main_logger to avoid duplicate handling / filtering issues


    # --- Setup for 'PIL.PngImagePlugin' logger ---
    pil_logger = logging.getLogger('PIL.PngImagePlugin')
    pil_logger.handlers = [] # Ensure it's clean
    pil_logger.setLevel(logging.WARNING) # Process WARNING+ internally for file logs
    pil_console_handler = logging.StreamHandler(sys.stdout)
    pil_console_handler.setFormatter(logging.Formatter(log_format_console))
    pil_console_handler.setLevel(logging.CRITICAL) # Silence PIL on console
    pil_logger.addHandler(pil_console_handler)

    # File handlers for PIL (to capture its WARNING/ERROR messages)
    # These are the same file handler instances used by main_logger and warnings_logger.
    # This is fine as handlers can be shared.
    pil_logger.addHandler(debug_file_handler)
    pil_logger.addHandler(info_file_handler)

    pil_logger.propagate = False # Do not propagate to main_logger


    # --- Setup for UnhandledException logger ---
    # This logger is used by the custom sys.excepthook
    exception_logger = logging.getLogger('UnhandledException')
    # Set level to CRITICAL as these are unhandled exceptions
    exception_logger.setLevel(logging.CRITICAL)

    # Add all handlers to it, as critical errors should go everywhere.
    # Console handler for exceptions (should always appear on console)
    exception_console_handler = logging.StreamHandler(sys.stderr) # Use stderr for critical errors
    exception_console_handler.setFormatter(logging.Formatter(log_format_console))
    exception_console_handler.setLevel(logging.CRITICAL) # Ensure it outputs critical messages
    exception_logger.addHandler(exception_console_handler)

    exception_logger.addHandler(debug_file_handler)
    exception_logger.addHandler(info_file_handler)
    exception_logger.propagate = False # Self-contained

    # Set the custom excepthook
    # We only want to set this once.
    if sys.excepthook == original_sys_excepthook:
        sys.excepthook = custom_excepthook
        # Log that we've set it, using the main logger if available, or root.
        logging.getLogger(main_logger_name or '').info("Custom sys.excepthook set up.")
    else:
        logging.getLogger(main_logger_name or '').warning("Custom sys.excepthook was already modified. Not overriding.")


    # Return the main logger instance and the specific run_log_dir
    # The logger returned here is the one named main_logger_name, or root if not specified.
    return main_logger, run_log_dir


# Helper function to get a logger instance, typically used by other modules.
# It's better for other modules to just call logging.getLogger(__name__)
# and let the setup_logger configure the handlers for root or named loggers.
def get_logger(name=None, config=None, log_dir=None):
    if name is None:
        # If no name is provided, assume it's the root logger that should be configured by setup_logger.
        # However, setup_logger should ideally be called once in the main script.
        # This function is a bit ambiguous if setup_logger hasn't run.
        # Let's assume if config and log_dir are provided, it's a hint to ensure setup.
        if config and log_dir:
             # This might re-initialize if called multiple times, be careful.
            logger, _ = setup_logger(config, log_dir, main_logger_name=None)
            return logger
        return logging.getLogger() # Get root logger, hoping it's configured
    
    logger = logging.getLogger(name)
    # If this logger has no handlers, it will propagate to the root logger.
    # We assume the root logger (or a relevant parent) has been configured by setup_logger.
    # No specific configuration here, relies on propagation unless logger 'name' is specially handled.
    return logger


class MetricLogger:
    """
    Logs scalar metrics to console, file, TensorBoard, and CSV for external visualization.
    Also supports logging images, histograms, and text to TensorBoard.
    """
    def __init__(self, logger: logging.Logger, tb_writer: SummaryWriter, csv_path: str):
        self.logger = logger
        self.tb_writer = tb_writer
        self.csv_path = csv_path
        self.step = 0
        self.global_step = 0  # 全局步数计数器，不会被reset重置
        self.csv_header_written = False
        
        # 记录TensorBoard日志目录位置（如果有TensorBoard writer）
        if self.tb_writer is not None:
            self.logger.info(f"TensorBoard日志将写入目录: {self.tb_writer.log_dir}")
            
            # 确保TensorBoard目录存在并有写入权限
            if not os.path.exists(self.tb_writer.log_dir):
                try:
                    os.makedirs(self.tb_writer.log_dir, exist_ok=True)
                    self.logger.info(f"已创建TensorBoard目录: {self.tb_writer.log_dir}")
                except Exception as e:
                    self.logger.error(f"创建TensorBoard目录失败: {str(e)}")
            
            # 尝试写入一个测试事件
            try:
                self.tb_writer.add_text("setup", "MetricLogger初始化成功", 0)
                self.tb_writer.flush()
                self.logger.info("TensorBoard初始化测试写入成功")
            except Exception as e:
                self.logger.error(f"TensorBoard写入测试失败: {str(e)}")
        else:
            self.logger.info("MetricLogger初始化成功（无TensorBoard）")

    def reset(self):
        """Reset internal step counter (e.g., at epoch￼￼
art)."""
        self.step = 0
        # 全局步数计数器不重置，确保TensorBoard图表持续向右绘制

    def log_metrics(self, metrics: dict, prefix: str = "train", step: int = None):
        """
        Log a dict of scalar metrics.

        Args:
            metrics (dict): {metric_name: value}
            prefix (str): Tag prefix, e.g. 'train', 'val'.
            step (int): Optional step number for tensorboard logging.
        """
        # 使用传入的step或默认的global_step
        current_step = step if step is not None else self.global_step
        
        # 1) Console/file
        msg = f"[{prefix}] Step {current_step}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.logger.info(msg)

        # 2) TensorBoard scalars and histograms (如果有TensorBoard writer)
        if self.tb_writer is not None:
            for k, v in metrics.items():
                # 根据指标名称分类，确保不同类型的损失分开绘制
                if k.startswith('loss_') or k.endswith('_loss'):
                    # 损失组件放在单独的图表组中
                    tag = f"{prefix}/losses/{k}"
                elif k == 'loss':
                    # 总损失单独一个图表
                    tag = f"{prefix}/{k}"
                elif k in ['psnr', 'ssim']:
                    # 质量评估指标放在一组
                    tag = f"{prefix}/metrics/{k}"
                elif k == 'lr':
                    # 学习率单独一个图表
                    tag = f"{prefix}/{k}"
                else:
                    # 其他指标
                    tag = f"{prefix}/other/{k}"
                    
                # 使用指定的步数记录
                self.tb_writer.add_scalar(tag, v, current_step)
                
                # optional histogram if tensor-like
                if isinstance(v, torch.Tensor) and v.numel() > 1:
                    self.tb_writer.add_histogram(f"{prefix}/{k}_hist", v, current_step)

        # 3) Append to CSV
        fieldnames = ['step', 'global_step', 'prefix'] + list(metrics.keys())
        if not self.csv_header_written:
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            self.csv_header_written = True

        row = {'step': current_step, 'global_step': self.global_step, 'prefix': prefix, **{k: float(v) for k, v in metrics.items()}}
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        self.step += 1
        self.global_step += 1  # 全局步数始终增加，不受epoch重置影响

    def flush(self):
        """强制刷新TensorBoard写入器，确保所有数据都写入磁盘"""
        if self.tb_writer is not None:
            try:
                self.tb_writer.flush()
                self.logger.info("TensorBoard数据已刷新到磁盘")
            except Exception as e:
                self.logger.error(f"刷新TensorBoard数据时出错: {str(e)}")
        else:
            self.logger.debug("跳过TensorBoard刷新（无TensorBoard writer）")

    def log_image(self, tag: str, image: torch.Tensor, step: int = None):
        """
        Log a single image tensor to TensorBoard.

        Args:
            tag (str): e.g. 'train/input'
            image (Tensor): shape (C,H,W) or (B,C,H,W)
            step (int): step for TB; defaults to current internal global_step.
        """
        if self.tb_writer is None:
            self.logger.debug(f"跳过图像记录 '{tag}'（无TensorBoard writer）")
            return
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
        
        # 制作图像副本以避免修改原始数据
        image = image.detach().clone()
        
        # 特殊处理深度图 - 针对16位深度图优化显示
        if ("depth" in tag or "gate" in tag) and image.shape[1] == 1:
            # 创建一个新的图像列表来存储转换后的彩色图像
            colored_images = []
            
            for i in range(image.shape[0]):
                img = image[i]
                min_val, max_val = img.min(), img.max()
                
                # 记录深度图的值范围，便于调试
                self.logger.info(f"Depth image '{tag}' value range: [{min_val:.6f}, {max_val:.6f}], shape: {img.shape}")
                
                # 优化16位深度图的显示
                if max_val > 1000:  # 可能是原始16位深度图
                    # 使用对数变换增强可视化效果
                    img_np = img.cpu().numpy()
                    log_img = np.log(img_np + 1.0)
                    # 再归一化到0-1
                    if log_img.max() > log_img.min():
                        norm_img = (log_img - log_img.min()) / (log_img.max() - log_img.min())
                    else:
                        norm_img = np.zeros_like(log_img)
                else:
                    # 已经是归一化的深度图或门控图，直接使用
                    if max_val > min_val + 1e-5:
                        norm_img = (img - min_val) / (max_val - min_val)
                        norm_img = norm_img.cpu().numpy()
                    else:
                        norm_img = img.cpu().numpy()
                
                # 将单通道图像转为三通道热力图
                colored = cm.turbo(norm_img[0])[:,:,:3]  # 使用turbo colormap，删除alpha通道
                colored_tensor = torch.from_numpy(colored).permute(2, 0, 1)  # (H,W,3) -> (3,H,W)
                colored_images.append(colored_tensor)
            
            # 替换原始图像列表为彩色图像
            colored_batch = torch.stack(colored_images)
            
            # 处理批次维度
            B = colored_batch.shape[0]
            
            if B > 1:
                # 创建网格
                grid = vutils.make_grid(colored_batch, nrow=min(B, 8), normalize=False)
                self.tb_writer.add_image(tag, grid, step or self.global_step)
            else:
                # 单个样本，直接记录
                self.tb_writer.add_image(tag, colored_batch[0], step or self.global_step, dataformats='CHW')
            
            self.flush()  # 强制刷新
            return
            
        # 处理普通图像
        elif image.max() > 1.0 or image.min() < 0:
            image = torch.clamp((image + 1) / 2, 0, 1)  # 假设原范围是 [-1, 1]
        
        # 处理批次维度
        B, C, H, W = image.shape
        
        if B > 1:
            # 如果是批次图像，只记录第一张图像或使用 make_grid 创建网格
            if C == 1 and "depth" in tag:  # 深度图/门控图通常是单通道
                # 对深度图/门控图，我们取第一个样本
                image_to_log = image[0]  # (1,H,W)
                self.tb_writer.add_image(tag, image_to_log, step or self.global_step, dataformats='CHW')
            else:
                # 对于多样本的普通图像，创建网格
                grid = vutils.make_grid(image, nrow=min(B, 8), normalize=False)
                self.tb_writer.add_image(tag, grid, step or self.global_step)
        else:
            # 单个样本，直接记录
            self.tb_writer.add_image(tag, image[0], step or self.global_step, dataformats='CHW')
            
        self.flush()  # 强制刷新

    def log_images_grid(self, tag: str, images: torch.Tensor, nrow: int = 8, step: int = None):
        """
        Log a batch of images as a grid.
        
        Args:
            tag (str): e.g. 'train/outputs'
            images (Tensor): shape (B,C,H,W)
            nrow (int): Number of images per row
            step (int): step for TB; defaults to current internal global_step.
        """
        if images.max() > 1.0 or images.min() < 0:
            images = torch.clamp((images + 1) / 2, 0, 1)
            
        grid = vutils.make_grid(images, nrow=nrow, normalize=False)
        self.tb_writer.add_image(tag, grid, step or self.global_step)
        self.flush()  # 强制刷新

    def log_image_comparison(self, tag: str, 
                            images: List[torch.Tensor], 
                            titles: List[str] = None,
                            step: int = None):
        """
        Log multiple images side by side for comparison.
        
        Args:
            tag (str): e.g. 'val/comparison'
            images (List[Tensor]): List of tensors each (C,H,W)
            titles (List[str]): Optional labels for each image
            step (int): step for TB; defaults to current internal global_step.
        """
        fig, axs = plt.subplots(1, len(images), figsize=(4*len(images), 4))
        if len(images) == 1:
            axs = [axs]
            
        for i, img in enumerate(images):
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
                # 首先确保我们有一个可以直接显示的形状
                if img.ndim == 3:  # (C,H,W)
                    if img.shape[0] == 1:  # 单通道图像
                        img = img[0]  # 从(1,H,W)变为(H,W)
                    else:  # RGB图像
                        img = np.transpose(img, (1, 2, 0))  # 转为(H,W,C)
                elif img.ndim == 4 and img.shape[0] == 1:  # (1,C,H,W)
                    if img.shape[1] == 1:  # 单通道图像
                        img = img[0, 0]  # 从(1,1,H,W)变为(H,W)
                    else:  # RGB图像
                        img = np.transpose(img[0], (1, 2, 0))  # 从(1,C,H,W)变为(H,W,C)
                
                # 特殊处理深度图 - 使用更好的colormap
                if "depth" in tag and img.ndim == 2:
                    img_min, img_max = img.min(), img.max()
                    if img_max > img_min + 1e-5:  # 避免除以零或接近零的值
                        img = (img - img_min) / (img_max - img_min)
                    axs[i].imshow(img, cmap='turbo')  # 使用turbo而不是灰度
                elif img.ndim == 2:  # 其他灰度图
                    axs[i].imshow(img, cmap='gray')
                else:  # RGB图
                    if img.max() > 1.0 or img.min() < 0:
                        img = np.clip((img + 1) / 2, 0, 1)
                    axs[i].imshow(img)
            
            if titles and i < len(titles):
                axs[i].set_title(titles[i])
            axs[i].axis('off')
            
        plt.tight_layout()
        self.tb_writer.add_figure(tag, fig, step or self.global_step)
        plt.close(fig)
        self.flush()  # 强制刷新

    def log_depth_fusion_weights(self, tag: str, 
                               weights: torch.Tensor, 
                               depth_images: List[torch.Tensor] = None,
                               step: int = None):
        """
        可视化自适应深度融合中的权重.
        
        Args:
            tag (str): e.g. 'train/depth_weights'
            weights (Tensor): 形状 (B,L,D,H,W) 的权重张量
                B: 批次大小
                L: 层级数量
                D: 深度尺度数量
                H, W: 空间尺寸
            depth_images (List[Tensor]): 可选，各深度图用于对比
            step (int): 步数，默认使用当前global_step
        """
        # 处理五维权重张量 (B,L,D,H,W)
        B, L, D, H, W = weights.shape
        weights = weights.detach().cpu()
        
        for b in range(min(B, 4)):  # 最多显示4个样本
            # 为每个层级创建权重热力图
            for l in range(L):
                # 创建当前层级下所有深度尺度的权重图
                fig, axs = plt.subplots(1, D, figsize=(4*D, 4))
                if D == 1:
                    axs = [axs]
                    
                for d in range(D):
                    im = axs[d].imshow(weights[b, l, d], vmin=0, vmax=1, cmap='viridis')
                    axs[d].set_title(f'Depth {d}')
                    axs[d].axis('off')
                    
                plt.colorbar(im, ax=axs)
                plt.tight_layout()
                self.tb_writer.add_figure(f"{tag}/level{l}_sample{b}", fig, step or self.global_step)
                plt.close(fig)
                
        # 为每个样本创建层级平均权重可视化
        for b in range(min(B, 4)):
            # 计算每个层级所有深度尺度的平均权重
            avg_weights = weights[b].mean(dim=1)  # (L,H,W)
            
            fig, axs = plt.subplots(1, L, figsize=(4*L, 4))
            if L == 1:
                axs = [axs]
                
            for l in range(L):
                im = axs[l].imshow(avg_weights[l], vmin=0, vmax=1, cmap='viridis')
                axs[l].set_title(f'Level {l} Avg Weight')
                axs[l].axis('off')
                
            plt.colorbar(im, ax=axs)
            plt.tight_layout()
            self.tb_writer.add_figure(f"{tag}/avg_weights_sample{b}", fig, step or self.global_step)
            plt.close(fig)
            
        # 如果提供了深度图，可以绘制加权深度图(这部分保留但需要进一步适配五维权重)
        if depth_images and len(depth_images) == D:
            # 暂时跳过深度图可视化，因为需要重新适配五维权重结构
            pass

    def log_attention_maps(self, tag: str, attention_maps: torch.Tensor, step: int = None):
        """
        可视化注意力图.

        Args:
            tag (str): e.g. 'train/attention'
            attention_maps (Tensor): 注意力权重, 形状为 (B,H,N,N) 或 (H,N,N)
            step (int): 步数，默认使用当前global_step
        """
        if attention_maps.dim() == 3:
            attention_maps = attention_maps.unsqueeze(0)  # 添加批次维度
            
        B, num_heads, seq_len, _ = attention_maps.shape
        attention_maps = attention_maps.detach().cpu()
        
        for b in range(min(B, 4)):  # 最多可视化4个样本
            # 为每个样本绘制多头注意力图
            num_heads_to_plot = min(num_heads, 4)  # 最多显示4个头
            fig, axs = plt.subplots(1, num_heads_to_plot, figsize=(4*num_heads_to_plot, 4))
            if num_heads_to_plot == 1:
                axs = [axs]
                
            for h in range(num_heads_to_plot):
                im = axs[h].matshow(attention_maps[b, h], cmap='viridis')
                axs[h].set_title(f'Head {h}')
                axs[h].axis('off')
                
            plt.colorbar(im, ax=axs)
            plt.tight_layout()
            self.tb_writer.add_figure(f"{tag}/sample{b}", fig, step or self.global_step)
            plt.close(fig)

    def log_feature_maps(self, tag: str, features: Union[torch.Tensor, List[torch.Tensor]], 
                      max_features: int = 16, step: int = None):
        """
        可视化特征图.
        
        Args:
            tag (str): e.g. 'train/features'
            features (Tensor or List[Tensor]): 特征图 (B,C,H,W) 或特征图列表
            max_features (int): 每个特征张量最多显示的通道数
            step (int): 步数，默认使用当前global_step
        """
        if not isinstance(features, list):
            features = [features]
            
        for i, feat in enumerate(features):
            if not isinstance(feat, torch.Tensor):
                continue
                
            feat = feat.detach().cpu()
            if feat.dim() < 4:
                feat = feat.unsqueeze(0)  # 添加批次维度
                
            B, C, H, W = feat.shape
            # 对每个样本，选择几个代表性通道可视化
            for b in range(min(B, 2)):  # 最多2个样本
                # 选择通道
                channels_to_plot = min(C, max_features)
                selected_channels = torch.linspace(0, C-1, channels_to_plot).long()
                
                # 提取所选通道
                channel_maps = feat[b, selected_channels]  # (channels_to_plot, H, W)
                
                # 标准化每个特征图用于可视化
                for c in range(channels_to_plot):
                    c_map = channel_maps[c]
                    c_min, c_max = c_map.min(), c_map.max()
                    if c_max > c_min:
                        channel_maps[c] = (c_map - c_min) / (c_max - c_min)
                
                # 以网格形式可视化
                nrow = int(np.ceil(np.sqrt(channels_to_plot)))
                grid = vutils.make_grid(channel_maps.unsqueeze(1), nrow=nrow)  # 将每个图扩展为1通道
                self.tb_writer.add_image(f"{tag}/level{i}/sample{b}", grid, step or self.global_step)
        
        self.flush()  # 强制刷新到磁盘

    def log_depth_gate_effect(self, tag: str, 
                           original_feature: torch.Tensor, 
                           gate_map: torch.Tensor, 
                           gated_output: torch.Tensor,
                           step: int = None):
        """
        可视化深度门控效果.

        Args:
            tag (str): e.g. 'train/depth_gate'
            original_feature (Tensor): 原始特征 (B,C,H,W)
            gate_map (Tensor): 门控图 (B,1,H,W)
            gated_output (Tensor): 应用门控后的特征 (B,C,H,W)
            step (int): 步数，默认使用当前global_step
        """
        B = original_feature.shape[0]
        
        for b in range(min(B, 4)):  # 最多显示4个样本
            # 准备可视化
            feature_vis = self._prepare_feature_for_vis(original_feature[b])
            gated_vis = self._prepare_feature_for_vis(gated_output[b])
            
            # 特殊处理门控图 - 使用min-max归一化
            gate = gate_map[b].detach().clone()
            min_val, max_val = gate.min(), gate.max()
            if max_val > min_val + 1e-5:
                gate = (gate - min_val) / (max_val - min_val)
            
            # 在第一次处理时打印门控图的值范围，便于调试
            if b == 0:
                self.logger.info(f"Gate map '{tag}' value range: [{min_val:.6f}, {max_val:.6f}]")
                
            # 将门控图从灰度转为彩色热力图以增强可视化效果
            gate_np = gate.cpu().numpy()
            if gate.shape[0] == 1:
                colored = cm.turbo(gate_np[0])[:,:,:3]  # 使用turbo colormap，删除alpha通道
                gate_vis = torch.from_numpy(colored).permute(2, 0, 1)  # (H,W,3) -> (3,H,W)
            else:
                gate_vis = gate[:3]  # 使用前三个通道
            
            # 绘制对比图
            images = [feature_vis, gate_vis, gated_vis]
            titles = ['Original Feature', 'Gate Map', 'Gated Feature']
            
            try:
                self.log_image_comparison(f"{tag}/sample{b}", images, titles, step or self.global_step)
            except Exception as e:
                self.logger.error(f"门控效果可视化失败: {str(e)}")
                # 提供额外的诊断信息
                for i, img in enumerate(images):
                    self.logger.info(f"图像 {i} ({titles[i]}) 形状: {img.shape}, 类型: {img.dtype if isinstance(img, torch.Tensor) else type(img)}")

    def _prepare_feature_for_vis(self, feature: torch.Tensor) -> torch.Tensor:
        """Helper to convert feature tensor to RGB visualization."""
        feature = feature.detach().cpu()
        if feature.dim() == 2:  # (H,W)
            feature = feature.unsqueeze(0)  # 添加通道维度
        
        C = feature.shape[0]
        if C == 1:  # 灰度图
            feature = feature.repeat(3, 1, 1)  # 复制3通道
        elif C > 3:  # 多通道特征图
            # 使用前3个通道或计算通道平均值
            feature = feature[:3]
        
        # 归一化
        feature_min = feature.min()
        feature_max = feature.max()
        if feature_max > feature_min:
            feature = (feature - feature_min) / (feature_max - feature_min)
            
        return feature

    def log_figure(self, tag: str, figure, step: int = None):
        """
        Log a matplotlib figure to TensorBoard.

        Args:
            tag (str): e.g. 'val/comparison'
            figure (matplotlib.figure.Figure)
            step (int): step index; defaults to current global_step.
        """
        self.tb_writer.add_figure(tag, figure, step or self.global_step)

    def log_text(self, tag: str, text: str, step: int = None):
        """
        Log text data to TensorBoard.

        Args:
            tag (str): e.g. 'hyperparams'
            text (str): Arbitrary text content.
            step (int): step index; defaults to current global_step.
        """
        self.tb_writer.add_text(tag, text, step or self.global_step)

    def log_model_graph(self, model, input_tensor):
        """
        Log model architecture graph to TensorBoard.
        
        Args:
            model: PyTorch model
            input_tensor: Example input tensor for tracing the model
        """
        self.tb_writer.add_graph(model, input_tensor)

    def log_model_parameters(self, model, step: int = None):
        """
        Log parameter statistics of model layers.
        
        Args:
            model: PyTorch model
            step: Global step; defaults to current global_step
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                # 参数值直接记录
                if torch.isfinite(param).all() and param.numel() > 0:
                    self.tb_writer.add_histogram(f"params/{name}", param, step or self.global_step)
                
                # 梯度需要检查是否为None和是否包含有效值
                if param.grad is not None:
                    # 确保梯度是有限值且不为空
                    if torch.isfinite(param.grad).all() and param.grad.numel() > 0:
                        self.tb_writer.add_histogram(f"grads/{name}", param.grad, step or self.global_step)

    def close(self):
        """Close the TensorBoard writer."""
        self.tb_writer.close()

    def log_depth_comparison(self, tag: str, 
                           depth_gt: torch.Tensor, 
                           depth_pred: torch.Tensor,
                           step: int = None):
        """
        创建深度预测与真值的对比可视化
        
        Args:
            tag (str): e.g. 'val/depth_comparison'
            depth_gt (Tensor): 真实深度图 [B,1,H,W]
            depth_pred (Tensor): 预测深度图 [B,1,H,W]
            step (int): 步骤索引
        """
        # 确保输入是正确的形状
        if depth_gt.dim() == 3:  # [B,H,W]
            depth_gt = depth_gt.unsqueeze(1)  # [B,1,H,W]
        if depth_pred.dim() == 3:
            depth_pred = depth_pred.unsqueeze(1)
        
        B = min(depth_gt.shape[0], depth_pred.shape[0])
        
        for b in range(min(B, 4)):  # 最多显示4个样本
            # 获取当前批次的深度图
            gt = depth_gt[b:b+1]  # 保持 [1,1,H,W] 形状
            pred = depth_pred[b:b+1]  # 保持 [1,1,H,W] 形状
            
            # 创建深度差异图
            with torch.no_grad():
                # 如果真实深度是16位深度图，需要首先归一化
                if gt.max() > 1000:  # 可能是原始16位深度图
                    # 对数归一化
                    log_gt = torch.log(gt + 1.0)
                    log_min = torch.log(torch.tensor(5000.0, device=gt.device) + 1.0)
                    log_max = torch.log(torch.tensor(65000.0, device=gt.device) + 1.0)
                    norm_gt = (log_gt - log_min) / (log_max - log_min + 1e-6)
                    norm_gt = torch.clamp(norm_gt, 0, 1)
                else:
                    # 已经归一化的深度图
                    norm_gt = gt
                
                # 计算差异 - 保持相同的形状
                diff = torch.abs(norm_gt - pred)
                
                # 记录三种图 - 使用log_image直接处理原始格式
                self.log_image(f"{tag}/gt_{b}", gt, step)
                self.log_image(f"{tag}/pred_{b}", pred, step)
                self.log_image(f"{tag}/diff_{b}", diff, step)
                
                # 创建对比可视化 - 这里不再需要特别处理
                images = [norm_gt, pred, diff]
                titles = ["Ground Truth", "Prediction", "Absolute Difference"]
                
                try:
                    self.log_image_comparison(f"{tag}/sample_{b}", images, titles, step)
                except Exception as e:
                    self.logger.error(f"记录深度对比图像时发生错误: {str(e)}")
                    # 提供详细的调试信息
                    for i, img in enumerate(images):
                        self.logger.info(f"图像 {i} 形状: {img.shape}, 类型: {img.dtype}, 范围: [{img.min().item():.4f}, {img.max().item():.4f}]")

