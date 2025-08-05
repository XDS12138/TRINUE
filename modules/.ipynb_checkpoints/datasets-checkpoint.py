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
        for ext in self.file_exts:
            path = os.path.join(folder, basename + ext)
            if os.path.exists(path):
                return path
        pattern = os.path.join(folder, basename + ".*")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Could not find any file for {basename} in {folder}")
    
    def apply_random_crop(self, images_pil, depth_np=None):
        if not images_pil: return images_pil, depth_np
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
            if depth_np is not None: depth_np = np.array(Image.fromarray(depth_np).resize(self.patch_size[::-1], Image.NEAREST))
        return cropped_images, depth_np
    
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
                 color_jitter_params: Dict = None, file_ext: List[str] = None):
        if file_ext is None: file_ext = ['.jpg', '.png', '.tif']
        super().__init__(patch_size, augment, gamma_range, color_jitter_params, file_ext)
        
        self.raw_folders = raw_folders
        self.depth_folder = depth_folder
        self.gt_folder = gt_folder
        self.num_degradations = len(raw_folders)
        if self.num_degradations == 0: raise ValueError("Must provide at least one degradation image folder")
        
        self.names = []
        for f_name in os.listdir(raw_folders[0]):
            if os.path.splitext(f_name)[1].lower() in self.file_exts:
                base_name = os.path.splitext(f_name)[0]
                if all(any(os.path.exists(os.path.join(folder, base_name + e)) for e in self.file_exts) for folder in [depth_folder, gt_folder] + raw_folders[1:]):
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
        
        I_gt = load_image(self._find_file_with_basename(self.gt_folder, basename), mode='RGB')
        _, D_gt_array = load_depth_image(self._find_file_with_basename(self.depth_folder, basename))
        # if idx == 0: print(f"Depth map mode: {D_gt_img.mode}, shape: {D_gt_array.shape}, type: {D_gt_array.dtype}, Range: [{D_gt_array.min()}, {D_gt_array.max()}]")
            
        # 合并所有图像裁剪为一次操作，确保空间对齐
        all_images = raw_images + [I_gt]
        all_images_cropped, D_gt_array_cropped = self.apply_random_crop(all_images, D_gt_array)
        raw_images_cropped, I_gt_cropped = all_images_cropped[:-1], all_images_cropped[-1]

        all_images_flipped, D_gt_array_flipped = self.apply_flips(all_images_cropped, D_gt_array_cropped)
        raw_images_flipped, I_gt_flipped = all_images_flipped[:-1], all_images_flipped[-1]

        if self.augment: raw_images_final = self.apply_photometric_augmentation(raw_images_flipped)
        else: raw_images_final = raw_images_flipped
        I_gt_final = I_gt_flipped
        D_gt_array_final = D_gt_array_flipped

        raw_tensors = [self.normalize_rgb_tensor(F.to_tensor(img)) for img in raw_images_final]
        gt_t = self.normalize_rgb_tensor(F.to_tensor(I_gt_final))
        depth_t = torch.from_numpy(D_gt_array_final.copy().astype(np.float32)).unsqueeze(0)
        
        raw_batch = torch.stack(raw_tensors, dim=0)

        return {'raw_imgs': raw_batch, 'depth': depth_t, 'gt': gt_t,
                'num_degradations': len(raw_tensors), 'basename': basename} 