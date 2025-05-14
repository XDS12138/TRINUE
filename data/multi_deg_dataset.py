import os
import random
from typing import List, Tuple, Dict, Optional, Union
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as T
import numpy as np
import glob

def load_image(path, mode='RGB'):
    """
    加载图像，处理错误并返回PIL图像对象
    """
    try:
        img = Image.open(path).convert(mode)
        return img
    except Exception as e:
        raise IOError(f"无法加载图像 {path}: {e}")

class MultiDegradationDataset(Dataset):
    """
    PyTorch Dataset for underwater enhancement with multiple degradation levels:
      - Multiple I_raw (RGB) images with different degradation levels
      - Single D_gt (grayscale depth)
      - Single I_gt (RGB ground truth)

    Applies synchronized geometric transforms to all images.
    Returns tensors normalized to [-1, 1] for RGB images.
    """
    def __init__(self,
                 raw_folders: List[str],     # 多个退化图像的文件夹列表
                 depth_folder: str,          # 深度图文件夹
                 gt_folder: str,             # 目标清晰图文件夹
                 patch_size: Union[int, Tuple[int, int]] = 256,
                 augment: bool = True,
                 file_ext: List[str] = ['.jpg', '.png', '.tif']):
        super().__init__()
        self.raw_folders = raw_folders
        self.depth_folder = depth_folder
        self.gt_folder = gt_folder
        
        # 处理patch_size为整数或元组的情况
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            self.patch_size = patch_size
            
        self.augment = augment
        self.num_degradations = len(raw_folders)
        
        # 验证至少有一个退化文件夹
        if self.num_degradations == 0:
            raise ValueError("必须至少提供一个退化图像文件夹")
        
        # 所有文件夹中共同的文件名（假设所有文件夹的文件名相同）
        self.file_exts = [ext.lower() if not ext.startswith('.') else ext for ext in file_ext]
        
        # 获取第一个退化文件夹中的所有文件名
        self.names = []
        for f in os.listdir(raw_folders[0]):
            ext = os.path.splitext(f)[1].lower()
            if ext in self.file_exts:
                # 检查其他文件夹中是否存在相同文件名的图像
                base_name = os.path.splitext(f)[0]
                exists_in_all = True
                
                # 检查深度图和GT图
                depth_exists = False
                gt_exists = False
                
                for e in self.file_exts:
                    if os.path.exists(os.path.join(depth_folder, base_name + e)):
                        depth_exists = True
                    if os.path.exists(os.path.join(gt_folder, base_name + e)):
                        gt_exists = True
                
                if not depth_exists or not gt_exists:
                    exists_in_all = False
                
                # 检查所有退化文件夹
                for folder in raw_folders[1:]:
                    folder_exists = False
                    for e in self.file_exts:
                        if os.path.exists(os.path.join(folder, base_name + e)):
                            folder_exists = True
                            break
                    if not folder_exists:
                        exists_in_all = False
                        break
                
                if exists_in_all:
                    self.names.append(base_name)
        
        self.names.sort()
        print(f"找到 {len(self.names)} 个有效样本 (在 {self.num_degradations} 个退化级别中)")
        
        # Photometric jitter for raw
        self.gamma_range = (0.8, 1.2)
        self.color_jitter = T.ColorJitter(brightness=0.1,
                                          contrast=0.1,
                                          saturation=0.1,
                                          hue=0.02)

    def __len__(self):
        return len(self.names)
    
    def _find_file_with_basename(self, folder, basename):
        """找到指定文件夹中与basename匹配的文件"""
        for ext in self.file_exts:
            path = os.path.join(folder, basename + ext)
            if os.path.exists(path):
                return path
        
        # 如果没有找到，使用通配符搜索
        pattern = os.path.join(folder, basename + ".*")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
            
        raise FileNotFoundError(f"在 {folder} 中找不到 {basename} 的任何文件")

    def __getitem__(self, idx):
        basename = self.names[idx]
        
        # 加载多个退化图像
        raw_images = []
        for folder in self.raw_folders:
            try:
                raw_path = self._find_file_with_basename(folder, basename)
                raw_img = load_image(raw_path, mode='RGB')
                raw_images.append(raw_img)
            except FileNotFoundError as e:
                print(f"警告: {e}")
                # 如果无法加载，复制第一个退化图像（如果有的话）
                if raw_images:
                    raw_images.append(raw_images[0].copy())
                else:
                    # 如果连第一个都没有，只能报错
                    raise e
        
        # 加载深度图和GT图
        try:
            depth_path = self._find_file_with_basename(self.depth_folder, basename)
            gt_path = self._find_file_with_basename(self.gt_folder, basename)
            
            D_gt = load_image(depth_path, mode='L')
            I_gt = load_image(gt_path, mode='RGB')
        except FileNotFoundError as e:
            print(f"错误: {e}")
            raise e
            
        # 同步几何变换（裁剪、翻转）- 对所有图像应用相同变换
        # 随机裁剪
        if self.augment:
            # 获取第一个图像的尺寸（假设所有图像尺寸相同）
            w, h = raw_images[0].size
            ps_h, ps_w = self.patch_size  # 使用高度和宽度
            
            if w > ps_w and h > ps_h:
                left = random.randint(0, w-ps_w)
                top  = random.randint(0, h-ps_h)
                box = (left, top, left+ps_w, top+ps_h)
                
                # 对所有图像执行相同裁剪
                for i in range(len(raw_images)):
                    raw_images[i] = raw_images[i].crop(box)
                D_gt = D_gt.crop(box)
                I_gt = I_gt.crop(box)
            else:
                # 如果图像小于补丁大小，则调整大小
                for i in range(len(raw_images)):
                    raw_images[i] = raw_images[i].resize(self.patch_size[::-1], Image.BILINEAR)  # PIL使用(w,h)格式
                D_gt = D_gt.resize(self.patch_size[::-1], Image.BILINEAR)
                I_gt = I_gt.resize(self.patch_size[::-1], Image.BILINEAR)
        else:
            # 非增强模式，统一调整大小
            for i in range(len(raw_images)):
                raw_images[i] = raw_images[i].resize(self.patch_size[::-1], Image.BILINEAR)  # PIL使用(w,h)格式
            D_gt = D_gt.resize(self.patch_size[::-1], Image.BILINEAR)
            I_gt = I_gt.resize(self.patch_size[::-1], Image.BILINEAR)

        # 随机水平/垂直翻转
        if self.augment and random.random() < 0.5:
            for i in range(len(raw_images)):
                raw_images[i] = F.hflip(raw_images[i])
            D_gt = F.hflip(D_gt)
            I_gt = F.hflip(I_gt)
            
        if self.augment and random.random() < 0.5:
            for i in range(len(raw_images)):
                raw_images[i] = F.vflip(raw_images[i])
            D_gt = F.vflip(D_gt)
            I_gt = F.vflip(I_gt)

        # 针对每个退化图像的单独光度增强
        if self.augment:
            for i in range(len(raw_images)):
                # 不同的伽马值
                gamma = random.uniform(*self.gamma_range)
                raw_images[i] = ImageEnhance.Brightness(raw_images[i]).enhance(gamma)
                # 颜色抖动
                raw_images[i] = self.color_jitter(raw_images[i])

        # 转换为张量并归一化
        raw_tensors = []
        for raw_img in raw_images:
            raw_t = F.to_tensor(raw_img)  # [0,1]
            # 归一化到 [-1,1]
            raw_t = raw_t.mul(2.0).sub(1.0)
            raw_tensors.append(raw_t)
            
        gt_t = F.to_tensor(I_gt)
        depth_t = F.to_tensor(D_gt)
        
        # 归一化 RGB 到 [-1,1]
        gt_t = gt_t.mul(2.0).sub(1.0)
        
        # 堆叠多个退化图像成为一个批次 [N,C,H,W]
        raw_batch = torch.stack(raw_tensors, dim=0)

        # 返回字典，包含所有退化图像和对应的GT和深度图
        return {
            'raw_imgs': raw_batch,        # 所有退化图像 [N,3,H,W]
            'depth': depth_t,              # 深度图 [1,H,W]
            'gt': gt_t,                    # 目标图 [3,H,W]
            'num_degradations': len(raw_tensors),  # 退化级别数量
            'basename': basename           # 文件基础名（用于调试）
        } 