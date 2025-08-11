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
    """Load an image as PIL Image with specified mode."""
    try:
        with Image.open(path) as img:
            return img.convert(mode)
    except Exception as e:
        raise IOError(f"Unable to load image {path}: {e}")

def load_depth_image(path):
    """
    Special loader for 16-bit depth images to preserve precision.
    Also detects if the depth map has already been processed.
    """
    try:
        # Load depth image in original mode without conversion
        depth_img = Image.open(path)
        # Check if depth image is 16-bit
        if depth_img.mode in ['I;16', 'I']:
            # Use NumPy for direct conversion to 16-bit array
            depth_array = np.array(depth_img)
            
            # Check depth range to determine if it's already normalized
            min_depth_val = depth_array.min()
            max_depth_val = depth_array.max()
            
            if max_depth_val < 100.0 or (max_depth_val <= 1.0 and min_depth_val >= 0.0):
                # print(f"[DATASET] Detected normalized depth map, range [{min_depth_val:.4f}, {max_depth_val:.4f}]")
                if max_depth_val > 1.0 or min_depth_val < 0.0:
                    # print(f"[DATASET] Re-normalizing depth map to [0,1] range")
                    depth_array = (depth_array - min_depth_val) / (max_depth_val - min_depth_val + 1e-6)
                
            return depth_img, depth_array
        else:
            depth_array = np.array(depth_img)
            min_depth_val = depth_array.min()
            max_depth_val = depth_array.max()
            if max_depth_val < 100.0 or (max_depth_val <= 1.0 and min_depth_val >= 0.0):
                # print(f"[DATASET] Detected non-16bit normalized depth map, range [{min_depth_val:.4f}, {max_depth_val:.4f}]")
                if max_depth_val > 1.0 or min_depth_val < 0.0:
                    # print(f"[DATASET] Re-normalizing depth map to [0,1] range")
                    depth_array = (depth_array - min_depth_val) / (max_depth_val - min_depth_val + 1e-6)
            return depth_img, depth_array
    except Exception as e:
        raise IOError(f"Unable to load depth map {path}: {e}")

class UnderwaterDatasetBase(Dataset):
    def __init__(self, 
                 patch_size: Union[int, Tuple[int, int]] = 256,
                 augment: bool = True,
                 gamma_range: Tuple[float, float] = (0.8, 1.2),
                 color_jitter_params: Dict = None,
                 file_ext: List[str] = None):
        super().__init__()
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            self.patch_size = patch_size
        self.augment = augment
        self.gamma_range = gamma_range
        if color_jitter_params is None:
            color_jitter_params = {'brightness': 0.1, 'contrast': 0.1, 'saturation': 0.1, 'hue': 0.02}
        self.color_jitter = T.ColorJitter(**color_jitter_params)
        if file_ext is None:
            self.file_exts = ['.jpg', '.png', '.tif', '.jpeg']
        else:
            self.file_exts = [ext.lower() if not ext.startswith('.') else ext for ext in file_ext]
    
    def _find_file_with_basename(self, folder, basename):
        # 首先尝试精确匹配
        for ext in self.file_exts:
            path = os.path.join(folder, basename + ext)
            if os.path.exists(path):
                return path
        
        # 如果是depth文件夹，尝试匹配带后缀的文件名（如 _mist_vis_0001）
        if 'depth' in folder.lower():
            for ext in self.file_exts:
                pattern = os.path.join(folder, basename + "_*" + ext)
                matches = glob.glob(pattern)
                if matches:
                    return matches[0]
        
        # 通用通配符匹配
        pattern = os.path.join(folder, basename + ".*")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Could not find any file for {basename} in {folder}")
    
    def apply_random_crop(self, images_pil, depth_np=None):
        if not images_pil: return images_pil, depth_np
        
        # 如果patch_size为None，不进行裁剪（验证模式）
        if self.patch_size is None:
            # 验证模式下统一图像尺寸
            return self.resize_images_to_common_size(images_pil, depth_np)
            
        w, h = images_pil[0].size
        ps_h, ps_w = self.patch_size
        cropped_images = []
        if w > ps_w and h > ps_h:
            left = random.randint(0, w - ps_w)
            top = random.randint(0, h - ps_h)
            box = (left, top, left + ps_w, top + ps_h)
            for img in images_pil: cropped_images.append(img.crop(box))
            if depth_np is not None: depth_np = depth_np[top:top + ps_h, left:left + ps_w]
        else:
            for img in images_pil: cropped_images.append(img.resize(self.patch_size[::-1], Image.BILINEAR))
            if depth_np is not None: 
                # 🔥 修复深度图resize问题
                depth_pil = Image.fromarray(depth_np.astype(np.uint8) if depth_np.dtype != np.uint8 else depth_np)
                depth_resized = depth_pil.resize(self.patch_size[::-1], Image.NEAREST)
                depth_np = np.array(depth_resized).astype(depth_np.dtype)
        return cropped_images, depth_np
    
    def resize_images_to_common_size(self, images_pil, depth_np=None):
        """验证模式下将所有图像调整到统一尺寸，保持长宽比并使用填充"""
        if not images_pil:
            return images_pil, depth_np
        
        # 🔥 智能选择目标尺寸
        target_size = self._determine_optimal_validation_size(images_pil)
        
        # 调整所有图像到目标尺寸（保持长宽比+填充）
        resized_images = []
        for img in images_pil:
            resized_img = self._resize_with_padding(img, target_size)
            resized_images.append(resized_img)
        
        # 调整深度图（如果存在）
        resized_depth = depth_np
        if depth_np is not None:
            depth_pil = Image.fromarray(depth_np)
            depth_resized_pil = self._resize_with_padding(depth_pil, target_size, is_depth=True)
            resized_depth = np.array(depth_resized_pil)
        
        return resized_images, resized_depth
    
    def _determine_optimal_validation_size(self, images_pil):
        """确定验证时的最优尺寸"""
        # 计算所有图像的尺寸统计
        widths = [img.size[0] for img in images_pil]
        heights = [img.size[1] for img in images_pil]
        
        # 使用最常见的尺寸，或者适中的尺寸
        max_width = max(widths)
        max_height = max(heights)
        
        # 🔥 限制最大分辨率避免内存问题
        max_resolution = 1024  # 可以从配置中读取
        
        if max_width > max_resolution or max_height > max_resolution:
            # 按比例缩放
            scale = min(max_resolution / max_width, max_resolution / max_height)
            target_width = int(max_width * scale)
            target_height = int(max_height * scale)
        else:
            target_width = max_width
            target_height = max_height
        
        # 确保尺寸是8的倍数（对于某些网络架构更友好）
        target_width = ((target_width + 7) // 8) * 8
        target_height = ((target_height + 7) // 8) * 8
        
        return (target_width, target_height)
    
    def _resize_with_padding(self, img, target_size, is_depth=False):
        """保持长宽比地调整图像尺寸，并使用填充"""
        target_w, target_h = target_size
        current_w, current_h = img.size
        
        # 计算缩放比例（保持长宽比）
        scale = min(target_w / current_w, target_h / current_h)
        
        # 计算缩放后的尺寸
        new_w = int(current_w * scale)
        new_h = int(current_h * scale)
        
        # 缩放图像
        if is_depth:
            resized_img = img.resize((new_w, new_h), Image.NEAREST)
        else:
            resized_img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # 创建目标尺寸的画布并填充
        if is_depth:
            # 深度图使用0填充
            canvas = Image.new('L', target_size, 0)
        else:
            # RGB图像使用反射填充
            canvas = Image.new('RGB', target_size, (0, 0, 0))
        
        # 计算粘贴位置（居中）
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        
        # 粘贴缩放后的图像
        canvas.paste(resized_img, (paste_x, paste_y))
        
        # 🔥 对RGB图像进行边界填充优化
        if not is_depth and (paste_x > 0 or paste_y > 0):
            canvas = self._apply_reflect_padding(canvas, resized_img, paste_x, paste_y, new_w, new_h)
        
        return canvas
    
    def _apply_reflect_padding(self, canvas, resized_img, paste_x, paste_y, img_w, img_h):
        """对RGB图像应用反射填充"""
        import numpy as np
        
        canvas_np = np.array(canvas)
        resized_np = np.array(resized_img)
        
        # 左右填充
        if paste_x > 0:
            # 左填充
            left_padding = resized_np[:, :paste_x]  # 取左边缘
            left_padding = np.fliplr(left_padding)  # 水平翻转
            canvas_np[paste_y:paste_y+img_h, :paste_x] = left_padding
            
            # 右填充
            right_start = paste_x + img_w
            # 计算实际需要的右填充宽度
            right_width = canvas_np.shape[1] - right_start
            if right_width > 0:
                # 从图像右边缘取相应宽度的像素
                right_source_width = min(right_width, paste_x, img_w)
                right_padding = resized_np[:, -right_source_width:]  # 取右边缘
                right_padding = np.fliplr(right_padding)  # 水平翻转
                # 如果需要更多填充，重复边缘像素
                if right_width > right_source_width:
                    edge_col = right_padding[:, -1:]  # 最右边的列
                    extra_padding = np.repeat(edge_col, right_width - right_source_width, axis=1)
                    right_padding = np.concatenate([right_padding, extra_padding], axis=1)
                canvas_np[paste_y:paste_y+img_h, right_start:right_start+right_width] = right_padding[:, :right_width]
        
        # 上下填充
        if paste_y > 0:
            # 上填充
            top_padding = canvas_np[paste_y:2*paste_y, :]  # 取上边缘
            top_padding = np.flipud(top_padding)  # 垂直翻转
            canvas_np[:paste_y, :] = top_padding
            
            # 下填充
            bottom_start = paste_y + img_h
            # 计算实际需要的下填充高度
            bottom_height = canvas_np.shape[0] - bottom_start
            if bottom_height > 0:
                # 从已填充区域取相应高度的像素
                bottom_source_height = min(bottom_height, paste_y, img_h)
                bottom_padding = canvas_np[bottom_start-bottom_source_height:bottom_start, :]  # 取下边缘
                bottom_padding = np.flipud(bottom_padding)  # 垂直翻转
                # 如果需要更多填充，重复边缘像素
                if bottom_height > bottom_source_height:
                    edge_row = bottom_padding[-1:, :]  # 最下边的行
                    extra_padding = np.repeat(edge_row, bottom_height - bottom_source_height, axis=0)
                    bottom_padding = np.concatenate([bottom_padding, extra_padding], axis=0)
                canvas_np[bottom_start:bottom_start+bottom_height, :] = bottom_padding[:bottom_height, :]
        
        return Image.fromarray(canvas_np)
    
    def apply_flips(self, images_pil, depth_np=None):
        if self.augment and random.random() < 0.5:
            images_pil = [F.hflip(img) for img in images_pil]
            if depth_np is not None: depth_np = np.fliplr(depth_np)
        if self.augment and random.random() < 0.5:
            images_pil = [F.vflip(img) for img in images_pil]
            if depth_np is not None: depth_np = np.flipud(depth_np)
        return images_pil, depth_np
    
    def apply_photometric_augmentation(self, images_pil):
        if not self.augment: return images_pil
        augmented_images = []
        for img in images_pil:
            gamma = random.uniform(*self.gamma_range)
            img = ImageEnhance.Brightness(img).enhance(gamma)
            img = self.color_jitter(img)
            augmented_images.append(img)
        return augmented_images
    
    def normalize_rgb_tensor(self, tensor, to_range=(-1, 1)):
        min_val, max_val = to_range
        return tensor.mul(max_val - min_val).add(min_val)

    def unify_resolutions(self, images_pil, depth_np, strategy="min"):
        """
        统一多个图像的分辨率到同一目标分辨率
        🔧 优化：缓存分辨率决策，避免重复计算
        """
        # 获取所有图像的分辨率
        resolutions = [img.size for img in images_pil]
        
        # 创建分辨率的缓存键
        resolution_key = tuple(sorted(set(resolutions)))
        
        # 检查是否已经缓存了这个分辨率组合的决策
        if not hasattr(self, '_resolution_cache'):
            self._resolution_cache = {}
        
        if resolution_key in self._resolution_cache:
            target_size = self._resolution_cache[resolution_key]
        else:
            # 根据策略确定目标分辨率
            if strategy == "max":
                max_width = max(res[0] for res in resolutions)
                max_height = max(res[1] for res in resolutions)
                target_size = (max_width, max_height)
            elif strategy == "median":
                widths = sorted([res[0] for res in resolutions])
                heights = sorted([res[1] for res in resolutions])
                median_width = widths[len(widths)//2]
                median_height = heights[len(heights)//2]
                target_size = (median_width, median_height)
            elif strategy == "720p":
                target_size = (1280, 720)
            elif strategy == "1080p":
                target_size = (1920, 1080)
            else:
                # 默认使用最小分辨率
                min_width = min(res[0] for res in resolutions)
                min_height = min(res[1] for res in resolutions)
                target_size = (min_width, min_height)
            
            # 缓存决策
            self._resolution_cache[resolution_key] = target_size
            
            # 只在第一次遇到新分辨率组合时打印日志
            print(f"[DATASET] 分辨率统一策略: {strategy}")
            print(f"[DATASET] 原始分辨率: {list(resolution_key)}")
            print(f"[DATASET] 目标分辨率: {target_size}")
        
        # 统一所有图像到目标分辨率
        unified_images = []
        for i, img in enumerate(images_pil):
            if img.size != target_size:
                # 使用LANCZOS进行高质量下采样/上采样
                unified_img = img.resize(target_size, Image.LANCZOS)
                unified_images.append(unified_img)
            else:
                unified_images.append(img)
        
        # 处理深度图（如果存在）
        unified_depth = depth_np
        if depth_np is not None:
            # 深度图也需要调整到相应分辨率
            current_h, current_w = depth_np.shape
            if (current_w, current_h) != target_size:
                # 对深度图使用NEAREST插值保持精度
                depth_pil = Image.fromarray(depth_np)
                depth_resized = depth_pil.resize(target_size, Image.NEAREST)
                unified_depth = np.array(depth_resized)
        
        return unified_images, unified_depth

class UnderwaterPreprocessDataset(UnderwaterDatasetBase):
    def __init__(self, raw_dir, depth_dir, gt_dir, patch_size=256, augment=True, 
                 gamma_range=(0.8, 1.2), color_jitter_params=None, file_ext=None):
        super().__init__(patch_size, augment, gamma_range, color_jitter_params, file_ext)
        self.raw_dir = raw_dir
        self.depth_dir = depth_dir
        self.gt_dir = gt_dir
        self.names = sorted([f for f in os.listdir(raw_dir) if os.path.splitext(f)[1].lower() in self.file_exts])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        I_raw = load_image(os.path.join(self.raw_dir, name), mode='RGB')
        I_gt  = load_image(os.path.join(self.gt_dir, name),  mode='RGB')
        D_gt  = load_image(os.path.join(self.depth_dir, name), mode='L')
        
        images = [I_raw, I_gt, D_gt]
        images, _ = self.apply_random_crop(images)
        I_raw, I_gt, D_gt = images
        images, _ = self.apply_flips([I_raw, I_gt, D_gt])
        I_raw, I_gt, D_gt = images
        
        if self.augment: I_raw = self.apply_photometric_augmentation([I_raw])[0]

        raw_t = F.to_tensor(I_raw)
        gt_t  = F.to_tensor(I_gt)
        depth_t = F.to_tensor(D_gt)

        raw_t = self.normalize_rgb_tensor(raw_t)
        gt_t  = self.normalize_rgb_tensor(gt_t)

        ps_h, ps_w = self.patch_size if isinstance(self.patch_size, tuple) else (self.patch_size, self.patch_size)
        raw_half  = F.resize(raw_t, [ps_h//2, ps_w//2], interpolation=F.InterpolationMode.BILINEAR)
        gt_half   = F.resize(gt_t,  [ps_h//2, ps_w//2], interpolation=F.InterpolationMode.BILINEAR)
        depth_half= F.resize(depth_t, [ps_h//2, ps_w//2], interpolation=F.InterpolationMode.BILINEAR)

        return {'raw': raw_t, 'depth': depth_t, 'gt': gt_t,
                'raw_half': raw_half, 'depth_half': depth_half, 'gt_half': gt_half}

class MultiDegradationDataset(UnderwaterDatasetBase):
    def __init__(self, raw_folders: List[str], depth_folder: str, gt_folder: str,
                 patch_size: Union[int, Tuple[int, int]] = 256, augment: bool = True,
                 gamma_range: Tuple[float, float] = (0.8, 1.2),
                 color_jitter_params: Dict = None, file_ext: List[str] = None,
                 resolution_strategy: str = "min"):
        if file_ext is None: file_ext = ['.jpg', '.png', '.tif']
        super().__init__(patch_size, augment, gamma_range, color_jitter_params, file_ext)
        
        self.raw_folders = raw_folders
        self.depth_folder = depth_folder
        self.gt_folder = gt_folder
        self.resolution_strategy = resolution_strategy
        self.num_degradations = len(raw_folders)
        if self.num_degradations == 0: raise ValueError("Must provide at least one degradation image folder")
        
        self.names = []
        # 改进文件查找逻辑，使其更具鲁棒性
        if not os.path.isdir(raw_folders[0]):
             raise FileNotFoundError(f"Input folder not found: {raw_folders[0]}")

        for f_name in os.listdir(raw_folders[0]):
            if os.path.splitext(f_name)[1].lower() in self.file_exts:
                base_name = os.path.splitext(f_name)[0]
                
                # 只检查存在的文件夹
                check_folders = [f for f in raw_folders[1:] if f]
                if gt_folder: check_folders.append(gt_folder)
                if depth_folder: check_folders.append(depth_folder)

                # 使用改进的文件查找方法来支持depth文件的特殊命名
                all_files_exist = True
                for folder in check_folders:
                    try:
                        self._find_file_with_basename(folder, base_name)
                    except FileNotFoundError:
                        all_files_exist = False
                        break
                
                if all_files_exist:
                    self.names.append(base_name)
        self.names.sort()
        # print(f"Found {len(self.names)} valid samples (across {self.num_degradations} degradation levels)")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        basename = self.names[idx]
        raw_images = []
        for folder in self.raw_folders:
            try:
                raw_images.append(load_image(self._find_file_with_basename(folder, basename), mode='RGB'))
            except FileNotFoundError:
                if raw_images: raw_images.append(raw_images[0].copy())
                else: raise
        
        # 按需加载GT和Depth
        I_gt = None
        if self.gt_folder:
            try:
                I_gt = load_image(self._find_file_with_basename(self.gt_folder, basename), mode='RGB')
            except FileNotFoundError:
                # 对于某些验证集，GT可能是可选的
                pass

        D_gt_array = None
        if self.depth_folder:
            try:
                _, D_gt_array = load_depth_image(self._find_file_with_basename(self.depth_folder, basename))
            except FileNotFoundError:
                pass
        
        # 🔧 先统一所有图像的分辨率，确保空间对齐
        all_images = raw_images
        if I_gt:
            all_images.append(I_gt)

        # 统一分辨率后再进行裁剪
        all_images_unified, D_gt_array_unified = self.unify_resolutions(all_images, D_gt_array, self.resolution_strategy)
        all_images_cropped, D_gt_array_cropped = self.apply_random_crop(all_images_unified, D_gt_array_unified)

        if I_gt:
            raw_images_cropped, I_gt_cropped = all_images_cropped[:-1], all_images_cropped[-1]
        else:
            raw_images_cropped, I_gt_cropped = all_images_cropped, None

        all_images_flipped, D_gt_array_flipped = self.apply_flips(all_images_cropped, D_gt_array_cropped)
        
        if I_gt:
            raw_images_flipped, I_gt_flipped = all_images_flipped[:-1], all_images_flipped[-1]
        else:
            raw_images_flipped, I_gt_flipped = all_images_flipped, None

        # 🔧 修复对齐问题：水下图像增强任务中，raw图像本身已经是退化的模拟
        # 不需要额外的光度增强，保持与GT的视觉一致性
        raw_images_final = raw_images_flipped
        I_gt_final = I_gt_flipped
        D_gt_array_final = D_gt_array_flipped

        raw_tensors = [self.normalize_rgb_tensor(F.to_tensor(img)) for img in raw_images_final]
        
        gt_t = torch.zeros_like(raw_tensors[0])
        if I_gt_final:
            gt_t = self.normalize_rgb_tensor(F.to_tensor(I_gt_final))

        depth_t = torch.zeros_like(raw_tensors[0])[0:1, :, :] # 1 channel depth
        if D_gt_array_final is not None:
            depth_t = torch.from_numpy(D_gt_array_final.copy().astype(np.float32)).unsqueeze(0)
        
        raw_batch = torch.stack(raw_tensors, dim=0)

        return {'raw_imgs': raw_batch, 'depth': depth_t, 'gt': gt_t,
                'num_degradations': len(raw_tensors), 'basename': basename} 