import os
import random
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as T

def load_image(path, mode='RGB'):
    """Load an image as PIL Image with specified mode."""
    with Image.open(path) as img:
        return img.convert(mode)

class UnderwaterPreprocessDataset(Dataset):
    """
    PyTorch Dataset for three-input underwater enhancement:
      - I_raw (RGB)
      - D_gt  (grayscale depth)
      - I_gt  (RGB ground truth)

    Applies synchronized geometric transforms to all three,
    plus photometric augmentation to raw only.
    Returns full- and half-resolution tensors.
    """
    def __init__(self,
                 raw_dir,
                 depth_dir,
                 gt_dir,
                 patch_size=256,
                 augment=True):
        super().__init__()
        self.raw_dir = raw_dir
        self.depth_dir = depth_dir
        self.gt_dir = gt_dir
        self.patch_size = patch_size
        self.augment = augment

        # List of file basenames (assumes matching names)
        self.names = [f for f in os.listdir(raw_dir) if os.path.splitext(f)[1].lower() in ['.jpg','.png','.tif']]
        self.names.sort()

        # Photometric jitter for raw
        self.gamma_range = (0.8, 1.2)
        self.color_jitter = T.ColorJitter(brightness=0.1,
                                          contrast=0.1,
                                          saturation=0.1,
                                          hue=0.02)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        # Load images
        I_raw = load_image(os.path.join(self.raw_dir, name), mode='RGB')
        I_gt  = load_image(os.path.join(self.gt_dir, name),  mode='RGB')
        D_gt  = load_image(os.path.join(self.depth_dir, name),mode='L')

        # Synchronized geometric transforms
        # Random crop
        if self.augment:
            w, h = I_raw.size
            ps = self.patch_size
            if w > ps and h > ps:
                left = random.randint(0, w-ps)
                top  = random.randint(0, h-ps)
                box = (left, top, left+ps, top+ps)
                I_raw = I_raw.crop(box)
                I_gt  = I_gt.crop(box)
                D_gt  = D_gt.crop(box)
            else:
                I_raw = I_raw.resize((ps, ps), Image.BILINEAR)
                I_gt  = I_gt.resize((ps, ps), Image.BILINEAR)
                D_gt  = D_gt.resize((ps, ps), Image.BILINEAR)
        else:
            I_raw = I_raw.resize((self.patch_size, self.patch_size), Image.BILINEAR)
            I_gt  = I_gt.resize((self.patch_size, self.patch_size), Image.BILINEAR)
            D_gt  = D_gt.resize((self.patch_size, self.patch_size), Image.BILINEAR)

        # Random horizontal/vertical flip
        if self.augment and random.random() < 0.5:
            I_raw = F.hflip(I_raw); I_gt = F.hflip(I_gt); D_gt = F.hflip(D_gt)
        if self.augment and random.random() < 0.5:
            I_raw = F.vflip(I_raw); I_gt = F.vflip(I_gt); D_gt = F.vflip(D_gt)

        # Photometric on raw
        if self.augment:
            # Gamma
            gamma = random.uniform(*self.gamma_range)
            I_raw = ImageEnhance.Brightness(I_raw).enhance(gamma)
            # Color jitter
            I_raw = self.color_jitter(I_raw)

        # To tensor and normalize
        raw_t = F.to_tensor(I_raw)  # [0,1]
        gt_t  = F.to_tensor(I_gt)
        depth_t = F.to_tensor(D_gt)

        # Normalize RGB to [-1,1]
        raw_t = raw_t.mul(2.0).sub(1.0)
        gt_t  = gt_t.mul(2.0).sub(1.0)

        # Multi-scale: half resolution
        raw_half  = F.resize(raw_t, [self.patch_size//2, self.patch_size//2], interpolation=F.InterpolationMode.BILINEAR)
        gt_half   = F.resize(gt_t,  [self.patch_size//2, self.patch_size//2], interpolation=F.InterpolationMode.BILINEAR)
        depth_half= F.resize(depth_t, [self.patch_size//2, self.patch_size//2], interpolation=F.InterpolationMode.BILINEAR)

        return {
            'raw': raw_t,
            'depth': depth_t,
            'gt': gt_t,
            'raw_half': raw_half,
            'depth_half': depth_half,
            'gt_half': gt_half
        }

# Example usage:
# dataset = UnderwaterPreprocessDataset(
#     raw_dir='/path/to/raw',
#     depth_dir='/path/to/depth',
#     gt_dir='/path/to/gt',
#     patch_size=256,
#     augment=True
# )
# loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
