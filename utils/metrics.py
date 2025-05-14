import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple
import cv2

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array"""
    if tensor.is_cuda:
        tensor = tensor.detach().cpu()
    return tensor.detach().numpy()

def calculate_psnr(img1: Union[torch.Tensor, np.ndarray], 
                  img2: Union[torch.Tensor, np.ndarray], 
                  crop_border: int = 0,
                  test_y_channel: bool = False) -> float:
    """Calculate PSNR (Peak Signal-to-Noise Ratio)
    
    Args:
        img1 (Tensor or ndarray): Images with range [0, 1].
        img2 (Tensor or ndarray): Images with range [0, 1].
        crop_border (int): Cropped pixels in each edge of an image. Default: 0.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    
    Returns:
        float: PSNR result.
    """
    # Convert tensor to numpy
    if isinstance(img1, torch.Tensor):
        img1 = tensor_to_numpy(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor_to_numpy(img2)
    
    if img1.ndim == 4:  # batch images
        img1 = img1[0]  # take the first image
    if img2.ndim == 4:
        img2 = img2[0]
        
    # move channel to last dimension
    img1 = np.transpose(img1, (1, 2, 0))
    img2 = np.transpose(img2, (1, 2, 0))
    
    # [-1,1] -> [0,1]
    if img1.min() < 0:
        img1 = (img1 + 1.0) / 2.0
    if img2.min() < 0:
        img2 = (img2 + 1.0) / 2.0
    
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    if test_y_channel and img1.shape[2] == 3:  # RGB to YCbCr, use Y
        from skimage.color import rgb2ycbcr
        img1_y = rgb2ycbcr(img1)[:, :, 0]
        img2_y = rgb2ycbcr(img2)[:, :, 0]
        img1, img2 = img1_y, img2_y
    
    # [0,1] -> [0,255]
    img1 = img1 * 255.
    img2 = img2 * 255.
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    return 20. * np.log10(255. / np.sqrt(mse))

def calculate_ssim(img1: Union[torch.Tensor, np.ndarray], 
                  img2: Union[torch.Tensor, np.ndarray], 
                  crop_border: int = 0,
                  test_y_channel: bool = False) -> float:
    """Calculate SSIM (structural similarity)
    
    Args:
        img1 (Tensor or ndarray): Images with range [0, 1].
        img2 (Tensor or ndarray): Images with range [0, 1].
        crop_border (int): Cropped pixels in each edge of an image. Default: 0.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    
    Returns:
        float: SSIM result.
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        try:
            # 旧版本 scikit-image 的导入方式
            from skimage.measure import compare_ssim as ssim
        except ImportError:
            # 如果以上两种都不可用，使用替代实现
            def ssim(img1, img2, **kwargs):
                """对 ssim 的简单实现"""
                C1 = (0.01 * 255)**2
                C2 = (0.03 * 255)**2
                img1 = img1.astype(np.float64)
                img2 = img2.astype(np.float64)
                kernel = cv2.getGaussianKernel(11, 1.5)
                window = np.outer(kernel, kernel.transpose())
                mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
                mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
                mu1_sq = mu1**2
                mu2_sq = mu2**2
                mu1_mu2 = mu1 * mu2
                sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
                sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
                sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
                ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
                return ssim_map.mean()
    
    # Convert tensor to numpy
    if isinstance(img1, torch.Tensor):
        img1 = tensor_to_numpy(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor_to_numpy(img2)
    
    if img1.ndim == 4:  # batch images
        img1 = img1[0]  # take the first image
    if img2.ndim == 4:
        img2 = img2[0]
        
    # move channel to last dimension
    img1 = np.transpose(img1, (1, 2, 0))
    img2 = np.transpose(img2, (1, 2, 0))
    
    # [-1,1] -> [0,1]
    if img1.min() < 0:
        img1 = (img1 + 1.0) / 2.0
    if img2.min() < 0:
        img2 = (img2 + 1.0) / 2.0
    
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    if test_y_channel and img1.shape[2] == 3:  # RGB to YCbCr, use Y
        try:
            from skimage.color import rgb2ycbcr
            img1_y = rgb2ycbcr(img1)[:, :, 0]
            img2_y = rgb2ycbcr(img2)[:, :, 0]
            img1, img2 = img1_y, img2_y
        except ImportError:
            pass
    
    # Always convert to [0, 255] range for computing SSIM
    img1 = img1 * 255.0
    img2 = img2 * 255.0
    
    # single channel image
    if img1.ndim == 2:
        return ssim(img1, img2, data_range=255.0)
    else:
        # 适配不同版本的 scikit-image
        try:
            # 新版本 scikit-image
            return ssim(img1, img2, channel_axis=2, data_range=255.0)
        except TypeError:
            try:
                # 较旧版本 scikit-image
                return ssim(img1, img2, multichannel=True, data_range=255.0)
            except:
                # 如果都失败，尝试针对每个通道计算然后平均
                ssims = []
                for i in range(img1.shape[2]):
                    ssims.append(ssim(img1[..., i], img2[..., i], data_range=255.0))
                return np.mean(ssims) 