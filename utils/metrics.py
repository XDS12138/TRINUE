import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, Dict, Callable, List
import cv2

# Attempt to import necessary libraries, with placeholders for now
try:
    import lpips
    lpips_available = True
except ImportError:
    lpips_available = False
    print("LPIPS library not found. Please install it using 'pip install lpips'")

try:
    import piq
    piq_available = True
except ImportError:
    piq_available = False
    print("PIQ library not found. Please install it using 'pip install piq'")

try:
    from colour_difference import delta_E_CIEDE2000
    colour_difference_available = True
except ImportError:
    colour_difference_available = False
    print("colour-difference library not found. Please install it using 'pip install colour-difference'")

# For UCIQE, UIQM, NIQE, BRISQUE, PIQE, we might use pyiqa or individual implementations.
# For now, let's assume pyiqa or direct implementations will be handled.
# If using pyiqa, you would import from it:
# from pyiqa import create_metric
# uciqe_metric = create_metric('uciqe', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
# uiqm_metric = create_metric('uiqm', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
# niqe_metric = create_metric('niqe', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) # NIQE usually takes path or grayscale numpy
# brisque_metric = create_metric('brisque', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) # BRISQUE usually takes path or grayscale numpy
# piqe_metric = create_metric('piqe', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) # PIQE usually takes path or grayscale numpy


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array"""
    if tensor.is_cuda:
        tensor = tensor.detach().cpu()
    return tensor.float().numpy() # Ensure float for calculations

def _prepare_image_for_metric(img: Union[torch.Tensor, np.ndarray], 
                              target_range_0_1: bool = True,
                              target_chw: bool = False,
                              target_hwc: bool = False) -> np.ndarray:
    """Helper to convert image to numpy, handle batch, transpose, and normalize range."""
    if isinstance(img, torch.Tensor):
        img_np = tensor_to_numpy(img)
    else:
        img_np = img.astype(np.float32)

    if img_np.ndim == 4:  # Batch: B, C, H, W or B, H, W, C
        img_np = img_np[0] # Take the first image

    # Ensure correct channel order before range normalization if needed
    # Assuming input is likely CHW from PyTorch or HWC from cv2/skimage
    if img_np.shape[0] == 1 or img_np.shape[0] == 3: # Likely CHW
        if target_hwc: # if metric needs HWC
            img_np = np.transpose(img_np, (1, 2, 0))
    elif (img_np.shape[-1] == 1 or img_np.shape[-1] == 3) and target_chw: # Likely HWC but metric needs CHW
         img_np = np.transpose(img_np, (2, 0, 1))


    if target_range_0_1:
        if img_np.min() < 0.0 or img_np.max() > 1.0: # Heuristic: if not in [0,1]
            if img_np.min() >= -1.0 and img_np.max() <= 1.0: # Likely in [-1, 1]
                img_np = (img_np + 1.0) / 2.0
            else: # For other ranges (e.g. 0-255), try to normalize to 0-1
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
        img_np = np.clip(img_np, 0.0, 1.0)
        
    return img_np

def _prepare_rgb_image_for_colour_metric(img_np: np.ndarray) -> np.ndarray:
    """Prepare image for colour-difference: HWC, 0-1 range, then to Lab."""
    if not (img_np.ndim == 3 and img_np.shape[-1] == 3):
        raise ValueError("CIEDE2000 requires RGB images (HWC).")
    # colour-difference expects image in [0,1] for conversion
    img_lab = cv2.cvtColor(img_np.astype(np.float32), cv2.COLOR_RGB2Lab)
    return img_lab

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
    img1_np = _prepare_image_for_metric(img1, target_range_0_1=False, target_hwc=True) # PSNR scales to 255 later
    img2_np = _prepare_image_for_metric(img2, target_range_0_1=False, target_hwc=True)
    
    # [-1,1] -> [0,1] if necessary, then scale to 0-255
    if img1_np.min() < 0: img1_np = (img1_np + 1.0) / 2.0
    if img2_np.min() < 0: img2_np = (img2_np + 1.0) / 2.0
    
    if crop_border > 0:
        img1_np = img1_np[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2_np = img2_np[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    if test_y_channel and img1_np.shape[-1] == 3:
        from skimage.color import rgb2ycbcr
        img1_y = rgb2ycbcr(img1_np)[:, :, 0]
        img2_y = rgb2ycbcr(img2_np)[:, :, 0]
        img1_np, img2_np = img1_y, img2_y
    
    img1_np = img1_np * 255.
    img2_np = img2_np * 255.
    
    mse = np.mean((img1_np - img2_np) ** 2)
    if mse == 0:
        return float('inf')
    
    return 20. * np.log10(255. / np.sqrt(mse))

def calculate_ssim(img1: Union[torch.Tensor, np.ndarray], 
                  img2: Union[torch.Tensor, np.ndarray], 
                  crop_border: int = 0,
                  test_y_channel: bool = False) -> float:
    """Calculate SSIM (structural similarity) - Reuses existing logic, ensures proper input prep"""
    try:
        from skimage.metrics import structural_similarity as ssim_skimage
    except ImportError:
        try:
            from skimage.measure import compare_ssim as ssim_skimage # older skimage
        except ImportError:
            ssim_skimage = None # Fallback handled below

    img1_np = _prepare_image_for_metric(img1, target_range_0_1=False, target_hwc=True) # SSIM scales to 255 later
    img2_np = _prepare_image_for_metric(img2, target_range_0_1=False, target_hwc=True)

    if img1_np.min() < 0: img1_np = (img1_np + 1.0) / 2.0
    if img2_np.min() < 0: img2_np = (img2_np + 1.0) / 2.0

    if crop_border > 0:
        img1_np = img1_np[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2_np = img2_np[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel and img1_np.shape[-1] == 3:
        try:
            from skimage.color import rgb2ycbcr
            img1_y = rgb2ycbcr(img1_np)[:, :, 0]
            img2_y = rgb2ycbcr(img2_np)[:, :, 0]
            img1_np, img2_np = img1_y, img2_y
        except ImportError:
            print("scikit-image not found for Y channel SSIM. Calculating on RGB or grayscale.")
            pass # Continue with original img1_np, img2_np

    img1_np_255 = img1_np * 255.0
    img2_np_255 = img2_np * 255.0

    if ssim_skimage is not None:
        if img1_np_255.ndim == 2: # Grayscale
            return ssim_skimage(img1_np_255, img2_np_255, data_range=255.0)
        elif img1_np_255.ndim == 3 and img1_np_255.shape[-1] == 3: # Color
            try: # New scikit-image
                return ssim_skimage(img1_np_255, img2_np_255, channel_axis=2, data_range=255.0, multichannel=True) # multichannel for older skimage
            except TypeError: # older skimage
                 return ssim_skimage(img1_np_255, img2_np_255, multichannel=True, data_range=255.0)
        else: # Fallback for unexpected shapes with skimage
            print(f"Unexpected image shape for scikit-image SSIM: {img1_np_255.shape}. Trying per-channel average.")
            ssims = []
            for i in range(img1_np_255.shape[-1]):
                ssims.append(ssim_skimage(img1_np_255[..., i], img2_np_255[..., i], data_range=255.0))
            return np.mean(ssims) if ssims else 0.0
    else: # Fallback cv2 implementation (simplified from original, might need full one if skimage is commonly missing)
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        img1_f64 = img1_np_255.astype(np.float64)
        img2_f64 = img2_np_255.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        
        def _ssim_channel(ch1, ch2):
            mu1 = cv2.filter2D(ch1, -1, window)[5:-5, 5:-5]
            mu2 = cv2.filter2D(ch2, -1, window)[5:-5, 5:-5]
            mu1_sq = mu1**2
            mu2_sq = mu2**2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(ch1**2, -1, window)[5:-5, 5:-5] - mu1_sq
            sigma2_sq = cv2.filter2D(ch2**2, -1, window)[5:-5, 5:-5] - mu2_sq
            sigma12 = cv2.filter2D(ch1 * ch2, -1, window)[5:-5, 5:-5] - mu1_mu2
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean()
    
        if img1_f64.ndim == 2: # Grayscale
            return _ssim_channel(img1_f64, img2_f64)
        elif img1_f64.ndim == 3: # Color, average over channels
            ssim_vals = [_ssim_channel(img1_f64[..., i], img2_f64[..., i]) for i in range(img1_f64.shape[-1])]
            return np.mean(ssim_vals) if ssim_vals else 0.0
        return 0.0


# --- New Metric Implementations ---

# No-Reference Metrics
def calculate_uciqe(img: Union[torch.Tensor, np.ndarray]) -> float:
    """Calculate UCIQE (Underwater Color Image Quality Evaluation). Placeholder."""
    img_np = _prepare_image_for_metric(img, target_range_0_1=True, target_hwc=True)
    # UCIQE expects BGR uint8 image usually.
    img_bgr_uint8 = (img_np * 255).astype(np.uint8)
    if img_bgr_uint8.shape[-1] == 3: # If color, assume RGB and convert to BGR
        img_bgr_uint8 = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_RGB2BGR)
    
    # Placeholder: Actual UCIQE implementation is complex.
    # Typically involves chroma, saturation, contrast components.
    # Example using a known standalone implementation structure (conceptual)
    # try:
    #     from third_party_uciqe import get_uciqe_score
    #     return get_uciqe_score(img_bgr_uint8)
    # except ImportError:
    #     print("UCIQE implementation not found.")
    #     return 0.0
    # For now, using pyiqa if available (conceptual)
    try:
        from pyiqa import create_metric
        uciqe_metric_func = create_metric('uciqe', as_loss=False, device=torch.device('cpu')) # Assuming pyiqa uses tensors
        # pyiqa might need B,C,H,W tensor in [0,1]
        img_tensor = torch.from_numpy(np.transpose(img_np, (2,0,1))).unsqueeze(0)
        return uciqe_metric_func(img_tensor).item()
    except ImportError:
        print("pyiqa not found for UCIQE. Returning 0.")
        return 0.0
    except Exception as e:
        print(f"Error calculating UCIQE with pyiqa: {e}. Returning 0.")
        return 0.0

def calculate_uiqm(img: Union[torch.Tensor, np.ndarray]) -> float:
    """Calculate UIQM (Underwater Image Quality Measure). Placeholder."""
    img_np = _prepare_image_for_metric(img, target_range_0_1=True, target_hwc=True)
    img_bgr_uint8 = (img_np * 255).astype(np.uint8)
    if img_bgr_uint8.shape[-1] == 3:
        img_bgr_uint8 = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_RGB2BGR)
        
    # Placeholder: UIQM = c1*UICM + c2*UISM + c3*UIConM
    # try:
    #     from third_party_uiqm import get_uiqm_score
    #     return get_uiqm_score(img_bgr_uint8)
    # except ImportError:
    #     print("UIQM implementation not found.")
    #     return 0.0
    try:
        from pyiqa import create_metric
        uiqm_metric_func = create_metric('uiqm', as_loss=False, device=torch.device('cpu'))
        img_tensor = torch.from_numpy(np.transpose(img_np, (2,0,1))).unsqueeze(0)
        return uiqm_metric_func(img_tensor).item()
    except ImportError:
        print("pyiqa not found for UIQM. Returning 0.")
        return 0.0
    except Exception as e:
        print(f"Error calculating UIQM with pyiqa: {e}. Returning 0.")
        return 0.0

# Full-Reference Metrics
def calculate_ciede2000(img1: Union[torch.Tensor, np.ndarray], img2: Union[torch.Tensor, np.ndarray]) -> float:
    """Calculate CIEDE2000 color difference."""
    if not colour_difference_available:
        print("colour-difference library not available for CIEDE2000. Returning 0.")
        return 0.0
    
    img1_rgb_np = _prepare_image_for_metric(img1, target_range_0_1=True, target_hwc=True)
    img2_rgb_np = _prepare_image_for_metric(img2, target_range_0_1=True, target_hwc=True)

    if not (img1_rgb_np.ndim == 3 and img1_rgb_np.shape[-1] == 3 and \
            img2_rgb_np.ndim == 3 and img2_rgb_np.shape[-1] == 3):
        print("CIEDE2000 requires RGB images (H,W,3). Returning 0 for non-RGB or mismatched shapes.")
        return 0.0

    img1_lab = _prepare_rgb_image_for_colour_metric(img1_rgb_np)
    img2_lab = _prepare_rgb_image_for_colour_metric(img2_rgb_np)
    
    # Calculate delta_E for each pixel and then average
    # delta_E_CIEDE2000 expects two Lab pixels (1D arrays of length 3)
    delta_e_map = np.zeros_like(img1_lab[..., 0])
    for r in range(img1_lab.shape[0]):
        for c in range(img1_lab.shape[1]):
            delta_e_map[r,c] = delta_E_CIEDE2000(img1_lab[r,c,:], img2_lab[r,c,:])
            
    return np.mean(delta_e_map)


lpips_model_alex_rgb = None # Global LPIPS model instance
def calculate_lpips(img1: Union[torch.Tensor, np.ndarray], img2: Union[torch.Tensor, np.ndarray], net_type='alex') -> float:
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity)."""
    global lpips_model_alex_rgb
    if not lpips_available:
        print("LPIPS library not available. Returning 0.")
        return 0.0

    # LPIPS expects torch.Tensor, CHW, range [-1, 1]
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1.astype(np.float32))
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2.astype(np.float32))

    # Ensure CHW
    if img1.ndim == 3 and img1.shape[-1] == 3: # HWC to CHW
        img1 = img1.permute(2, 0, 1)
    if img2.ndim == 3 and img2.shape[-1] == 3: # HWC to CHW
        img2 = img2.permute(2, 0, 1)
    
    if img1.ndim == 2: # Grayscale H,W to 1,H,W
        img1 = img1.unsqueeze(0)
    if img2.ndim == 2: # Grayscale H,W to 1,H,W
        img2 = img2.unsqueeze(0)

    # Add batch dim if missing: C,H,W to B,C,H,W
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
    if img2.ndim == 3:
        img2 = img2.unsqueeze(0)
    
    # Normalize to [-1, 1]
    if img1.max() > 1.0: # Assuming [0, 255] or [0, 1]
        if img1.max() > 2.0: # Likely [0, 255]
            img1 = img1 / 255.0
        img1 = img1 * 2.0 - 1.0
    elif img1.min() >= 0.0 and img1.max() <=1.0: # in [0,1]
         img1 = img1 * 2.0 - 1.0

    if img2.max() > 1.0:
        if img2.max() > 2.0:
            img2 = img2 / 255.0
        img2 = img2 * 2.0 - 1.0
    elif img2.min() >=0.0 and img2.max() <=1.0:
        img2 = img2 * 2.0 - 1.0
    
    img1 = torch.clamp(img1, -1.0, 1.0)
    img2 = torch.clamp(img2, -1.0, 1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if lpips_model_alex_rgb is None:
        try:
            lpips_model_alex_rgb = lpips.LPIPS(net=net_type, version='0.1').to(device) # Common version
            lpips_model_alex_rgb.eval() # Set to eval mode
        except Exception as e:
            print(f"Failed to load LPIPS model: {e}. Returning 0.")
            return 0.0

    try:
        with torch.no_grad():
            distance = lpips_model_alex_rgb(img1.to(device).float(), img2.to(device).float())
        return distance.item()
    except Exception as e:
        print(f"Error during LPIPS calculation: {e}. Returning 0.")
        return 0.0

def calculate_fsim(img1: Union[torch.Tensor, np.ndarray], img2: Union[torch.Tensor, np.ndarray]) -> float:
    """Calculate FSIM (Feature Similarity Index)."""
    if not piq_available:
        print("PIQ library not available for FSIM. Returning 0.")
        return 0.0

    # PIQ's FSIM expects torch.Tensor, BCHW, range [0, 1], RGB or Grayscale
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1.astype(np.float32))
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2.astype(np.float32))

    # Ensure CHW if HWC
    if img1.ndim == 3 and img1.shape[-1] in [1,3]: img1 = img1.permute(2,0,1)
    if img2.ndim == 3 and img2.shape[-1] in [1,3]: img2 = img2.permute(2,0,1)
    
    # Ensure 3 channels for FSIM if grayscale (PIQ might handle it, but good to be explicit)
    # if img1.ndim == 2 or (img1.ndim == 3 and img1.shape[0] == 1): # H,W or 1,H,W
    #     img1 = img1.squeeze().repeat(3,1,1) if img1.ndim > 2 else img1.repeat(3,1,1)
    # if img2.ndim == 2 or (img2.ndim == 3 and img2.shape[0] == 1):
    #     img2 = img2.squeeze().repeat(3,1,1) if img2.ndim > 2 else img2.repeat(3,1,1)
        
    # Add batch dim
    if img1.ndim == 2 : img1 = img1.unsqueeze(0).unsqueeze(0) # H,W -> B,C,H,W
    elif img1.ndim == 3: img1 = img1.unsqueeze(0) # C,H,W -> B,C,H,W

    if img2.ndim == 2 : img2 = img2.unsqueeze(0).unsqueeze(0)
    elif img2.ndim == 3: img2 = img2.unsqueeze(0)


    # Normalize to [0, 1]
    if img1.max() > 1.0: img1 = img1 / 255.0
    if img2.max() > 1.0: img2 = img2 / 255.0
    img1 = torch.clamp(img1, 0.0, 1.0)
    img2 = torch.clamp(img2, 0.0, 1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # FSIM in PIQ defaults to data_range=1.0
        fsim_val: torch.Tensor = piq.fsim(img1.to(device).float(), img2.to(device).float(), data_range=1.)
        return fsim_val.item()
    except Exception as e:
        print(f"Error calculating FSIM with PIQ: {e}. Returning 0.")
        return 0.0

# No-Reference metrics (NIQE, BRISQUE, PIQE)
# These often require specific grayscale uint8 numpy inputs or paths.
# Using pyiqa is a good option here.
def _calculate_nriqa_metric(img: Union[torch.Tensor, np.ndarray], metric_name: str) -> float:
    img_np_hwc = _prepare_image_for_metric(img, target_range_0_1=True, target_hwc=True)
    # Most IQA metrics in pyiqa prefer BGR uint8 or grayscale uint8
    
    # if img_np_hwc.shape[-1] == 3: # RGB
    #     img_gray_uint8 = (cv2.cvtColor(img_np_hwc, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
    # elif img_np_hwc.ndim == 2 or img_np_hwc.shape[-1] == 1: # Grayscale
    #     img_gray_uint8 = (img_np_hwc.squeeze() * 255).astype(np.uint8)
    # else:
    #     print(f"{metric_name} expects grayscale or RGB image. Got shape {img_np_hwc.shape}. Returning 0.")
    #     return 0.0

    try:
        from pyiqa import create_metric
        # For pyiqa, usually input is a tensor B,C,H,W in [0,1]
        if img_np_hwc.ndim == 2: # H,W grayscale
            img_tensor = torch.from_numpy(img_np_hwc).unsqueeze(0).unsqueeze(0).float() # B,1,H,W
        elif img_np_hwc.ndim == 3 and img_np_hwc.shape[-1] == 1: # H,W,1 grayscale
            img_tensor = torch.from_numpy(np.transpose(img_np_hwc, (2,0,1))).unsqueeze(0).float() # B,1,H,W
        elif img_np_hwc.ndim == 3 and img_np_hwc.shape[-1] == 3: # H,W,C color
            img_tensor = torch.from_numpy(np.transpose(img_np_hwc, (2,0,1))).unsqueeze(0).float() # B,3,H,W
        else:
            print(f"Unsupported image shape for pyiqa {metric_name}: {img_np_hwc.shape}")
            return 0.0

        metric_func = create_metric(metric_name.lower(), as_loss=False, device=torch.device("cpu")) # Use CPU to avoid OOM with many metrics
        return metric_func(img_tensor).item()
    except ImportError:
        print(f"pyiqa not found for {metric_name}. Please install it. Returning 0.")
        return 0.0
    except Exception as e:
        print(f"Error calculating {metric_name} with pyiqa: {e}. Returning 0.")
        return 0.0

def calculate_niqe(img: Union[torch.Tensor, np.ndarray]) -> float:
    """Calculate NIQE (Natural Image Quality Evaluator)."""
    return _calculate_nriqa_metric(img, "niqe")

def calculate_brisque(img: Union[torch.Tensor, np.ndarray]) -> float:
    """Calculate BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)."""
    return _calculate_nriqa_metric(img, "brisque")

def calculate_piqe(img: Union[torch.Tensor, np.ndarray]) -> float:
    """Calculate PIQE (Perception based Image Quality Evaluator)."""
    return _calculate_nriqa_metric(img, "piqe")


# Depth Metrics
def calculate_depth_mae(pred: Union[torch.Tensor, np.ndarray], gt: Union[torch.Tensor, np.ndarray]) -> float:
    """Calculate Mean Absolute Error for depth maps."""
    pred_np = _prepare_image_for_metric(pred, target_range_0_1=False) # Keep original scale
    gt_np = _prepare_image_for_metric(gt, target_range_0_1=False)
    if pred_np.shape != gt_np.shape:
        # Try to resize pred to gt if they don't match (simple F.interpolate like)
        # This is a basic resize, more sophisticated alignment might be needed
        pred_tensor = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0) # B,C,H,W
        pred_resized = F.interpolate(pred_tensor, size=gt_np.shape[-2:], mode='bilinear', align_corners=False)
        pred_np = pred_resized.squeeze().numpy()

    if pred_np.shape != gt_np.shape: # Check again after resize
        print(f"Depth MAE: pred shape {pred_np.shape} and gt shape {gt_np.shape} mismatch. Returning 0.")
        return 0.0
    return np.mean(np.abs(pred_np - gt_np))

def calculate_depth_rmse(pred: Union[torch.Tensor, np.ndarray], gt: Union[torch.Tensor, np.ndarray]) -> float:
    """Calculate Root Mean Squared Error for depth maps."""
    pred_np = _prepare_image_for_metric(pred, target_range_0_1=False)
    gt_np = _prepare_image_for_metric(gt, target_range_0_1=False)
    if pred_np.shape != gt_np.shape:
        pred_tensor = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0)
        pred_resized = F.interpolate(pred_tensor, size=gt_np.shape[-2:], mode='bilinear', align_corners=False)
        pred_np = pred_resized.squeeze().numpy()
        
    if pred_np.shape != gt_np.shape:
        print(f"Depth RMSE: pred shape {pred_np.shape} and gt shape {gt_np.shape} mismatch. Returning 0.")
        return 0.0
    return np.sqrt(np.mean((pred_np - gt_np)**2))

def calculate_delta_thresholds(pred: Union[torch.Tensor, np.ndarray], 
                               gt: Union[torch.Tensor, np.ndarray], 
                               thresholds: List[float] = [1.25, 1.25**2, 1.25**3]) -> Dict[str, float]:
    """Calculate percentage of pixels satisfying delta thresholds for depth maps."""
    pred_np = _prepare_image_for_metric(pred, target_range_0_1=False).squeeze()
    gt_np = _prepare_image_for_metric(gt, target_range_0_1=False).squeeze()

    if pred_np.shape != gt_np.shape:
        pred_tensor = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0)
        pred_resized = F.interpolate(pred_tensor, size=gt_np.shape[-2:], mode='bilinear', align_corners=False)
        pred_np = pred_resized.squeeze().numpy()

    if pred_np.shape != gt_np.shape:
        print(f"Delta Thresholds: pred shape {pred_np.shape} and gt shape {gt_np.shape} mismatch. Returning empty dict.")
        return {}

    # Ensure positive depths for ratio calculation, clamp small values
    valid_mask = (gt_np > 1e-3) & (pred_np > 1e-3)
    pred_np_valid = pred_np[valid_mask]
    gt_np_valid = gt_np[valid_mask]

    if gt_np_valid.size == 0:
        print("Delta Thresholds: No valid pixels for comparison (all gt depths are too small or zero).")
        return {f"delta_{t:.2f}": 0.0 for t in thresholds}

    ratio = np.maximum(gt_np_valid / pred_np_valid, pred_np_valid / gt_np_valid)
    
    results = {}
    for t in thresholds:
        results[f"delta_{t:.2f}"] = np.mean((ratio < t).astype(np.float32)) * 100.0 # Percentage
    return results


# --- ALL_METRICS Dictionary ---
ALL_METRICS: Dict[str, Callable[..., Union[float, Dict[str, float]]]] = {
    "psnr": calculate_psnr,
    "ssim": calculate_ssim,
    "uciqe": calculate_uciqe,
    "uiqm": calculate_uiqm,
    "ciede2000": calculate_ciede2000,
    "lpips": calculate_lpips,
    "fsim": calculate_fsim,
    "niqe": calculate_niqe,
    "brisque": calculate_brisque,
    "piqe": calculate_piqe,
    "depth_mae": calculate_depth_mae,
    "depth_rmse": calculate_depth_rmse,
    "depth_delta": calculate_delta_thresholds, # Note: This returns a dict
}

FULL_REFERENCE_METRICS = [
    "psnr", 
    "ssim", 
    "ciede2000", 
    "lpips", 
    "fsim"
]

NO_REFERENCE_METRICS = [
    "uciqe", 
    "uiqm", 
    "niqe", 
    "brisque", 
    "piqe"
]

DEPTH_METRICS = [
    "depth_mae",
    "depth_rmse",
    "depth_delta"
]

def calculate_depth_statistics(pred: Union[torch.Tensor, np.ndarray], 
                              gt: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
    """
    Calculate comprehensive depth statistics including MAE, RMSE, AbsRel, SqRel
    
    Args:
        pred: Predicted depth map
        gt: Ground truth depth map
        
    Returns:
        Dict containing various depth metrics
    """
    pred_np = _prepare_image_for_metric(pred, target_range_0_1=False).squeeze()
    gt_np = _prepare_image_for_metric(gt, target_range_0_1=False).squeeze()
    
    # Resize if shapes don't match
    if pred_np.shape != gt_np.shape:
        pred_tensor = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0)
        pred_resized = F.interpolate(pred_tensor, size=gt_np.shape[-2:], mode='bilinear', align_corners=False)
        pred_np = pred_resized.squeeze().numpy()
    
    if pred_np.shape != gt_np.shape:
        print(f"Depth statistics: pred shape {pred_np.shape} and gt shape {gt_np.shape} mismatch. Returning zeros.")
        return {'mae': 0.0, 'rmse': 0.0, 'abs_rel': 0.0, 'sq_rel': 0.0}
    
    # Create valid mask (avoid division by zero)
    valid_mask = (gt_np > 1e-3) & (pred_np > 1e-3)
    
    if not np.any(valid_mask):
        print("Depth statistics: No valid pixels for comparison.")
        return {'mae': 0.0, 'rmse': 0.0, 'abs_rel': 0.0, 'sq_rel': 0.0}
    
    pred_valid = pred_np[valid_mask]
    gt_valid = gt_np[valid_mask]
    
    # Calculate metrics
    mae = np.mean(np.abs(pred_valid - gt_valid))
    rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))
    abs_rel = np.mean(np.abs(pred_valid - gt_valid) / gt_valid)
    sq_rel = np.mean(((pred_valid - gt_valid) ** 2) / gt_valid)
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'abs_rel': float(abs_rel),
        'sq_rel': float(sq_rel)
    } 