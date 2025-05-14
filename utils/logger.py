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
import matplotlib.cm as cm

def setup_logger(log_dir: str,
                 log_file: str = "train.log",
                 metrics_file: str = "metrics.csv",
                 level: int = logging.INFO):
    """
    Set up a comprehensive logger that writes to console, a log file, TensorBoard, and a CSV metrics file.

    Args:
        log_dir (str): Directory where logs, TB events, and CSV will be saved.
        log_file (str): Name of the text log file.
        metrics_file (str): Name of the CSV file for scalar metrics.
        level (int): Logging level.

    Returns:
        logger (logging.Logger): Configured Python logger.
        tb_writer (SummaryWriter): TensorBoard writer.
        csv_path (str): Path to the CSV metrics file.
        debug_logger (logging.Logger): 专用于调试信息的logger
    """
    os.makedirs(log_dir, exist_ok=True)

    # 1) Python logger
    logger = logging.getLogger("UnderwaterEnhance")
    logger.setLevel(level)
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(os.path.join(log_dir, log_file))
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
    
    # 创建专用于调试的logger
    debug_logger = logging.getLogger("UnderwaterEnhance.Debug")
    debug_logger.setLevel(logging.DEBUG)
    if not debug_logger.handlers:
        # 仅文件处理，不输出到控制台
        debug_fh = logging.FileHandler(os.path.join(log_dir, "debug.log"))
        debug_fh.setLevel(logging.DEBUG)
        debug_fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S"))
        debug_logger.addHandler(debug_fh)
        # 防止调试信息传递给根logger
        debug_logger.propagate = False

    # 2) TensorBoard writer
    tb_log_dir = os.path.join(log_dir, "tensorboard", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tb_writer = SummaryWriter(tb_log_dir)

    # 3) CSV metrics file setup: create header on first write
    csv_path = os.path.join(log_dir, metrics_file)
    if not os.path.exists(csv_path):
        # ensure exists; header will be written on first log_metrics call
        open(csv_path, 'w').close()

    return logger, tb_writer, csv_path, debug_logger


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
        
        # 记录TensorBoard日志目录位置
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

    def reset(self):
        """Reset internal step counter (e.g., at epoch￼￼
art)."""
        self.step = 0
        # 全局步数计数器不重置，确保TensorBoard图表持续向右绘制

    def log_metrics(self, metrics: dict, prefix: str = "train"):
        """
        Log a dict of scalar metrics.

        Args:
            metrics (dict): {metric_name: value}
            prefix (str): Tag prefix, e.g. 'train', 'val'.
        """
        # 1) Console/file
        msg = f"[{prefix}] Step {self.step}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.logger.info(msg)

        # 2) TensorBoard scalars and histograms
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
                
            # 使用全局步数记录，确保图表持续向右绘制
            self.tb_writer.add_scalar(tag, v, self.global_step)
            
            # optional histogram if tensor-like
            if isinstance(v, torch.Tensor) and v.numel() > 1:
                self.tb_writer.add_histogram(f"{prefix}/{k}_hist", v, self.global_step)

        # 3) Append to CSV
        fieldnames = ['step', 'global_step', 'prefix'] + list(metrics.keys())
        if not self.csv_header_written:
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            self.csv_header_written = True

        row = {'step': self.step, 'global_step': self.global_step, 'prefix': prefix, **{k: float(v) for k, v in metrics.items()}}
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        self.step += 1
        self.global_step += 1  # 全局步数始终增加，不受epoch重置影响

    def flush(self):
        """强制刷新TensorBoard写入器，确保所有数据都写入磁盘"""
        try:
            self.tb_writer.flush()
            self.logger.info("TensorBoard数据已刷新到磁盘")
        except Exception as e:
            self.logger.error(f"刷新TensorBoard数据时出错: {str(e)}")
            
    def log_image(self, tag: str, image: torch.Tensor, step: int = None):
        """
        Log a single image tensor to TensorBoard.

        Args:
            tag (str): e.g. 'train/input'
            image (Tensor): shape (C,H,W) or (B,C,H,W)
            step (int): step for TB; defaults to current internal global_step.
        """
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
        
        # 制作图像副本以避免修改原始数据
        image = image.detach().clone()
        
        # 特殊处理深度图 - 使用min-max归一化，并应用更好的可视化策略
        if "depth" in tag and image.shape[1] == 1:
            # 创建一个新的图像列表来存储转换后的彩色图像
            colored_images = []
            
            for i in range(image.shape[0]):
                img = image[i]
                min_val, max_val = img.min(), img.max()
                # 仅在值的范围足够大时才进行归一化，避免噪声放大
                if max_val > min_val + 1e-5:
                    # 归一化到0-1范围
                    norm_img = (img - min_val) / (max_val - min_val)
                    # 将单通道图像转为三通道热力图来提高可视化效果
                    img_np = norm_img.cpu().numpy()
                    colored = cm.jet(img_np[0])[:,:,:3]  # 删除alpha通道
                    colored_tensor = torch.from_numpy(colored).permute(2, 0, 1)  # (H,W,3) -> (3,H,W)
                    colored_images.append(colored_tensor)
                else:
                    # 如果没有有效范围，只复制为三通道灰度
                    colored_images.append(img.repeat(3, 1, 1))
                
                # 记录深度图的值范围，便于调试
                if i == 0:  # 只记录第一个样本
                    self.logger.info(f"Depth image '{tag}' value range: [{min_val:.6f}, {max_val:.6f}], shape: {img.shape}")
            
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
                if img.shape[0] == 1:  # 灰度图
                    img = img[0]
                    # 特殊处理深度图 - 使用更好的colormap
                    if "depth" in tag:
                        img_min, img_max = img.min(), img.max()
                        if img_max > img_min + 1e-5:  # 避免除以零或接近零的值
                            img = (img - img_min) / (img_max - img_min)
                        axs[i].imshow(img, cmap='turbo')  # 使用turbo而不是灰度
                    else:
                        axs[i].imshow(img, cmap='gray')
                else:  # RGB
                    img = np.transpose(img, (1, 2, 0))
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
            self.log_image_comparison(f"{tag}/sample{b}", images, titles, step or self.global_step)

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

