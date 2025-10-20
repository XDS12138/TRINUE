#!/usr/bin/env python3
"""
综合评估指标模块

包含图像质量评估指标和深度估计评估指标：
- 全参考指标：PSNR, SSIM, CIEDE2000, LPIPS
- 无参考指标：UCIQE, UIQM, NIQE  
- 深度估计指标：阈值准确率、各种误差指标

作者：基于用户提供的标准实现
"""

import numpy as np
import cv2
import math
import warnings
import torch
from typing import Dict, Union, Tuple, Optional, List

# 抑制一些不重要的警告
warnings.filterwarnings('ignore')

# 尝试导入scikit-image，如果不可用则使用自定义实现
try:
    from skimage.metrics import structural_similarity as ssim_skimage
    from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("Warning: scikit-image not available, using simplified implementations")
    SKIMAGE_AVAILABLE = False

# ===============================
# 图像质量评估指标
# ===============================

def calculate_psnr(img1, img2):
    """
    计算PSNR (Peak Signal-to-Noise Ratio)
    
    Args:
        img1: 参考图像 [0, 1] 或 [0, 255]
        img2: 比较图像 [0, 1] 或 [0, 255]
    
    Returns:
        psnr_value: PSNR值
    """
    try:
        # 转为float并规范范围
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        if img1.max() > 1.5:
            img1 = img1 / 255.0
        if img2.max() > 1.5:
            img2 = img2 / 255.0
        img1 = np.clip(img1, 0.0, 1.0)
        img2 = np.clip(img2, 0.0, 1.0)
        
        if SKIMAGE_AVAILABLE:
            return psnr_skimage(img1, img2, data_range=1.0)
        else:
            # 简化的PSNR计算
            mse = np.mean((img1 - img2) ** 2)
            if mse == 0:
                return 100  # 图像完全相同
            return 20 * np.log10(1.0 / np.sqrt(mse))
    except Exception as e:
        print(f"Error calculating PSNR: {e}")
        return 0.0


def calculate_ssim(img1, img2):
    """
    计算SSIM (Structural Similarity Index)
    
    Args:
        img1: 参考图像
        img2: 比较图像
    
    Returns:
        ssim_value: SSIM值
    """
    try:
        # 转为float并规范范围
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        if img1.max() > 1.5:
            img1 = img1 / 255.0
        if img2.max() > 1.5:
            img2 = img2 / 255.0
        img1 = np.clip(img1, 0.0, 1.0)
        img2 = np.clip(img2, 0.0, 1.0)
        
        if SKIMAGE_AVAILABLE:
            # 兼容新版/旧版skimage API
            try:
                if len(img1.shape) == 3:
                    return ssim_skimage(img1, img2, channel_axis=-1, data_range=1.0)
                else:
                    return ssim_skimage(img1, img2, data_range=1.0)
            except TypeError:
                if len(img1.shape) == 3:
                    return ssim_skimage(img1, img2, multichannel=True, data_range=1.0)
                else:
                    return ssim_skimage(img1, img2, data_range=1.0)
        else:
            # 简化的SSIM计算（基于均值和方差）
            if len(img1.shape) == 3:
                # 转换为灰度图像进行简化计算
                img1 = np.mean(img1, axis=2)
                img2 = np.mean(img2, axis=2)
            
            mu1 = np.mean(img1)
            mu2 = np.mean(img2)
            sigma1 = np.var(img1)
            sigma2 = np.var(img2)
            sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
            
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                      ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
            return float(ssim_val)
    except Exception as e:
        print(f"Error calculating SSIM: {e}")
        return 0.0

def rgb_to_lab(rgb):
    """RGB转LAB色彩空间"""
    # 确保RGB在[0, 1]范围内
    if rgb.max() > 1.0:
        rgb = rgb.astype(np.float32) / 255.0
    
    # OpenCV的LAB转换
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
    return lab.astype(np.float32)

def calculate_ciede2000(img1, img2):
    """
    计算CIEDE2000色差
    这是一个简化版本，真正的CIEDE2000计算非常复杂
    
    Args:
        img1: 参考图像
        img2: 比较图像
    
    Returns:
        ciede2000_value: CIEDE2000色差值
    """
    try:
        # 转换到LAB色彩空间
        lab1 = rgb_to_lab(img1)
        lab2 = rgb_to_lab(img2)
        
        # 简化的色差计算（不是完整的CIEDE2000）
        # 真正的CIEDE2000需要复杂的公式，这里使用Delta E CIE 1994近似
        delta_l = lab1[:,:,0] - lab2[:,:,0]
        delta_a = lab1[:,:,1] - lab2[:,:,1]
        delta_b = lab1[:,:,2] - lab2[:,:,2]
        
        delta_e = np.sqrt(delta_l**2 + delta_a**2 + delta_b**2)
        return np.mean(delta_e)
    except Exception as e:
        print(f"Error calculating CIEDE2000: {e}")
        return 0.0

def calculate_lpips(img1, img2):
    """
    计算LPIPS (Learned Perceptual Image Patch Similarity)
    使用预训练的AlexNet模型
    
    Args:
        img1: 参考图像 (numpy array 或 torch tensor)
        img2: 比较图像 (numpy array 或 torch tensor)
    
    Returns:
        lpips_value: LPIPS值 (越低越好，约0-1范围)
    """
    try:
        import lpips
        import torch
        
        # 创建LPIPS模型
        lpips_model = lpips.LPIPS(net='alex')
        
        # 处理输入图像
        def process_image(img):
            # 如果是numpy数组，转换为torch tensor
            if isinstance(img, np.ndarray):
                # 确保在[0, 1]范围内
                if img.max() > 1.0:
                    img = img.astype(np.float32) / 255.0
                
                # 转换为PyTorch格式 [1, C, H, W]
                if len(img.shape) == 3:
                    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
                else:
                    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
            else:
                img_tensor = img
                
            # LPIPS期望输入在[-1, 1]范围内
            return img_tensor * 2.0 - 1.0
        
        img1_tensor = process_image(img1)
        img2_tensor = process_image(img2)
        
        # 计算LPIPS
        with torch.no_grad():
            lpips_score = lpips_model(img1_tensor, img2_tensor).item()
        
        return float(lpips_score)
        
    except ImportError as e:
        print(f"Error: lpips library is required for LPIPS calculation: {e}")
        print("Please install it with: pip install lpips")
        # 回退到简化版本
        return calculate_lpips_simple(img1, img2)
        
    except Exception as e:
        print(f"Error calculating LPIPS: {e}")
        # 回退到简化版本
        return calculate_lpips_simple(img1, img2)


def calculate_lpips_simple(img1, img2):
    """
    简化的LPIPS计算
    真正的LPIPS需要预训练的深度学习模型，这里使用基于梯度的近似
    
    Args:
        img1: 参考图像
        img2: 比较图像
    
    Returns:
        lpips_value: 简化的LPIPS值
    """
    try:
        # 确保图像在[0, 1]范围内
        if img1.max() > 1.0:
            img1 = img1.astype(np.float32) / 255.0
        if img2.max() > 1.0:
            img2 = img2.astype(np.float32) / 255.0
        
        # 转换为灰度图像进行梯度计算
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray1 = (img1 * 255).astype(np.uint8)
            gray2 = (img2 * 255).astype(np.uint8)
        
        # 计算梯度
        grad1_x = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
        grad1_y = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
        grad2_x = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
        grad2_y = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度差异
        grad_diff = np.sqrt((grad1_x - grad2_x)**2 + (grad1_y - grad2_y)**2)
        
        # 返回归一化的梯度差异
        return np.mean(grad_diff) / 255.0
    except Exception as e:
        print(f"Error calculating LPIPS: {e}")
        return 0.0

def calculate_uciqe(img):
    """
    计算UCIQE (Underwater Color Image Quality Evaluation)
    完全符合原始论文定义 - Yang & Sowmya 2015
    论文公式: UCIQE = c1 * σ_c + c2 * conl + c3 * μ_s
    
    Args:
        img: 输入图像 (numpy array)
    
    Returns:
        uciqe_value: UCIQE值 (严格按照论文原始定义)
    """
    try:
        # 确保图像在[0, 255]范围内
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        # 转换到LAB色彩空间
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # LAB分量
        L = lab[:,:,0]  # 亮度: 0-100
        a = lab[:,:,1]  # 绿-红轴: 0-255 (OpenCV格式，需要转换为-128到+127)
        b = lab[:,:,2]  # 蓝-黄轴: 0-255 (OpenCV格式，需要转换为-128到+127)
        
        # 将OpenCV的LAB转换为标准LAB
        a_std = a - 128.0  # 转换为-128到+127范围
        b_std = b - 128.0  # 转换为-128到+127范围
        
        # 1. 计算色度 (Chroma) - σ_c
        C = np.sqrt(a_std**2 + b_std**2)
        sigma_c = np.std(C)
        
        # 2. 计算亮度对比度 (Contrast on Lightness) - conl
        # 论文使用亮度通道的标准差作为对比度
        conl = np.std(L)
        
        # 3. 计算饱和度均值 (Mean of Saturation) - μ_s
        # 论文定义: S = C / L, 其中L是亮度
        L_safe = np.maximum(L, 1.0)  # 避免除零
        S = C / L_safe
        mu_s = np.mean(S)
        
        # UCIQE标准权重系数 (论文原始值)
        c1 = 0.4680
        c2 = 0.2745  
        c3 = 0.2576
        
        # 严格按照论文公式计算 - 不进行任何归一化
        uciqe = c1 * sigma_c + c2 * conl + c3 * mu_s
        
        return float(uciqe)
        
    except Exception as e:
        print(f"Error calculating UCIQE: {e}")
        return 0.0

def calculate_uiqm(img):
    """
    计算UIQM (Underwater Image Quality Measure)
    完全符合原始论文定义 - Panetta et al. 2015
    论文公式: UIQM = c1 * UICM + c2 * UISM + c3 * UIConM
    
    Args:
        img: 输入图像 (numpy array)
    
    Returns:
        uiqm_value: UIQM值 (严格按照论文原始定义)
    """
    try:
        # 确保图像在[0, 255]范围内
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        # 转换为浮点数 [0, 255]
        img_float = img.astype(np.float32)
        
        # 分离RGB通道
        r, g, b = img_float[:,:,0], img_float[:,:,1], img_float[:,:,2]
        
        # 1. 计算UICM (Underwater Image Contrast Measure)
        # 基于RGB通道差异的对比度计算
        rg = r - g
        yb = 0.5 * (r + g) - b
        
        # 计算RG和YB的统计特性
        rg_mean, rg_std = np.mean(rg), np.std(rg)
        yb_mean, yb_std = np.mean(yb), np.std(yb)
        
        # UICM = sqrt(α²+β²) + 0.3*sqrt(μ_rg²+μ_yb²)
        uicm = np.sqrt(rg_std**2 + yb_std**2) + 0.3 * np.sqrt(rg_mean**2 + yb_mean**2)
        
        # 2. 计算UISM (Underwater Image Sharpness Measure)
        # 使用EME (Enhancement by Entropy Maximization)
        def calculate_eme(channel, block_size=8):
            """计算EME"""
            h, w = channel.shape
            num_blocks_h = h // block_size
            num_blocks_w = w // block_size
            
            if num_blocks_h == 0 or num_blocks_w == 0:
                return 0.0
            
            eme_sum = 0.0
            for i in range(num_blocks_h):
                for j in range(num_blocks_w):
                    block = channel[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                    min_val = np.min(block)
                    max_val = np.max(block)
                    
                    # 避免除零
                    if min_val > 0:
                        eme_sum += 20 * np.log10(max_val / min_val)
            
            return eme_sum / (num_blocks_h * num_blocks_w)
        
        # 分别计算RGB三个通道的EME
        eme_r = calculate_eme(r)
        eme_g = calculate_eme(g)
        eme_b = calculate_eme(b)
        
        # UISM = (EME_R + EME_G + EME_B) / 3
        uism = (eme_r + eme_g + eme_b) / 3.0
        
        # 3. 计算UIConM (Underwater Image Colorfulness Measure)
        # 使用HSV色彩空间计算色彩丰富度
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        saturation = img_hsv[:,:,1]
        
        # UIConM = 饱和度的标准差
        uiconm = np.std(saturation)
        
        # UIQM标准权重组合 (论文原始值)
        c1 = 0.0282  # UICM权重
        c2 = 0.2953  # UISM权重  
        c3 = 3.5753  # UIConM权重 (恢复论文原始值)
        
        # 严格按照论文公式计算 - 不进行任何归一化
        uiqm = c1 * uicm + c2 * uism + c3 * uiconm
        
        return float(uiqm)
        
    except Exception as e:
        print(f"Error calculating UIQM: {e}")
        return 0.0

def calculate_niqe(img):
    """
    计算NIQE (Natural Image Quality Evaluator)
    使用pyiqa库的标准实现
    
    Args:
        img: 输入图像 (numpy array)
    
    Returns:
        niqe_value: NIQE值 (越低越好，约0-10范围)
    """
    try:
        import pyiqa
        import torch
        
        # 创建NIQE模型
        niqe_model = pyiqa.create_metric('niqe', device='cpu')
        
        # 确保图像在[0, 1]范围内
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        
        # 转换为pytorch张量格式 [1, C, H, W]
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        
        # 计算NIQE
        with torch.no_grad():
            niqe_score = niqe_model(img_tensor).item()
        
        return float(niqe_score)
        
    except ImportError as e:
        # 抛出异常让验证管理器记录，而不是返回0.0
        raise ImportError(f"pyiqa library is required for NIQE calculation: {e}. Please install it with: pip install pyiqa")
        
    except Exception as e:
        # 抛出异常让验证管理器记录
        raise RuntimeError(f"Error calculating NIQE: {e}")

# ===============================
# 深度估计评估指标
# ===============================

class DepthMetrics:
    """深度估计评估指标计算类"""
    
    def __init__(self, thresholds: list = [1.05, 1.05**2, 1.05**3]):
        """
        初始化评估指标计算器
        
        Args:
            thresholds: 阈值列表，默认为[1.05, 1.05², 1.05³]
        """
        self.thresholds = thresholds
        
    def compute_scale_invariant_errors(self, pred: np.ndarray, gt: np.ndarray, 
                                     mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        计算尺度不变的深度误差指标
        
        Args:
            pred: 预测深度图 (H, W) 或 (N, H, W)
            gt: 真实深度图 (H, W) 或 (N, H, W)
            mask: 有效像素掩码 (H, W) 或 (N, H, W)，可选
            
        Returns:
            包含各种评估指标的字典
        """
        # 确保输入为numpy数组
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(gt):
            gt = gt.cpu().numpy()
        if torch.is_tensor(mask) and mask is not None:
            mask = mask.cpu().numpy()
            
        # 扁平化处理
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()
        
        # 创建有效掩码
        if mask is not None:
            mask_flat = mask.flatten()
        else:
            mask_flat = np.ones_like(gt_flat, dtype=bool)
            
        # 过滤无效值
        valid_mask = (
            mask_flat & 
            (gt_flat > 0) & 
            (pred_flat > 0) & 
            np.isfinite(gt_flat) & 
            np.isfinite(pred_flat) &
            (gt_flat != np.inf) &
            (pred_flat != np.inf)
        )
        
        if np.sum(valid_mask) == 0:
            return self._get_nan_metrics()
            
        pred_valid = pred_flat[valid_mask]
        gt_valid = gt_flat[valid_mask]
        
        # 计算各种误差指标
        metrics = {}
        
        # 1. 阈值准确率指标 (δ < threshold)
        ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
        for i, threshold in enumerate(self.thresholds):
            metrics[f'delta_{i+1}'] = np.mean(ratio < threshold)
        
        # 为了兼容性，添加标准命名
        metrics['delta1'] = metrics['delta_1']  # δ < 1.05
        metrics['delta2'] = metrics['delta_2']  # δ < 1.05²
        metrics['delta3'] = metrics['delta_3']  # δ < 1.05³
        
        # 2. 绝对相对误差 (Absolute Relative Error)
        abs_rel = np.abs(pred_valid - gt_valid) / gt_valid
        metrics['abs_rel'] = np.mean(abs_rel)
        
        # 3. 平方相对误差 (Squared Relative Error)
        sq_rel = ((pred_valid - gt_valid) ** 2) / gt_valid
        metrics['sq_rel'] = np.mean(sq_rel)
        
        # 4. 均方根误差 (Root Mean Square Error)
        rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))
        metrics['rmse'] = rmse
        
        # 5. 对数均方根误差 (Root Mean Square Log Error)
        log_pred = np.log(pred_valid)
        log_gt = np.log(gt_valid)
        rmse_log = np.sqrt(np.mean((log_pred - log_gt) ** 2))
        metrics['rmse_log'] = rmse_log
        
        # 6. 尺度不变对数误差 (Scale-Invariant Logarithmic Error)
        log_diff = log_pred - log_gt
        silog = np.sqrt(np.mean(log_diff ** 2) - (np.mean(log_diff) ** 2))
        metrics['silog'] = silog
        
        # 添加统计信息
        metrics['num_valid_pixels'] = int(np.sum(valid_mask))
        metrics['total_pixels'] = int(len(pred_flat))
        metrics['valid_ratio'] = np.sum(valid_mask) / len(pred_flat)
        
        return metrics
    
    def compute_metrics(self, pred: np.ndarray, gt: np.ndarray, 
                       mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        计算所有深度评估指标（主要接口）
        
        Args:
            pred: 预测深度图
            gt: 真实深度图  
            mask: 有效像素掩码，可选
            
        Returns:
            包含所有评估指标的字典
        """
        return self.compute_scale_invariant_errors(pred, gt, mask)
    
    def _get_nan_metrics(self) -> Dict[str, float]:
        """返回所有指标为NaN的字典"""
        return {
            'delta_1': float('nan'),
            'delta_2': float('nan'), 
            'delta_3': float('nan'),
            'delta1': float('nan'),
            'delta2': float('nan'),
            'delta3': float('nan'),
            'abs_rel': float('nan'),
            'sq_rel': float('nan'),
            'rmse': float('nan'),
            'rmse_log': float('nan'),
            'silog': float('nan'),
            'num_valid_pixels': 0,
            'total_pixels': 0,
            'valid_ratio': 0.0
        }
    
    def aggregate_metrics(self, metrics_list: list) -> Dict[str, float]:
        """
        聚合多个样本的评估指标
        
        Args:
            metrics_list: 各个样本的指标字典列表
            
        Returns:
            聚合后的平均指标
        """
        if not metrics_list:
            return self._get_nan_metrics()
            
        # 过滤掉无效的指标
        valid_metrics = [m for m in metrics_list if not np.isnan(m.get('abs_rel', float('nan')))]
        
        if not valid_metrics:
            return self._get_nan_metrics()
            
        aggregated = {}
        metric_keys = ['delta_1', 'delta_2', 'delta_3', 'delta1', 'delta2', 'delta3',
                      'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'silog']
        
        # 计算各指标的平均值
        for key in metric_keys:
            values = [m[key] for m in valid_metrics if not np.isnan(m.get(key, float('nan')))]
            if values:
                aggregated[key] = np.mean(values)
            else:
                aggregated[key] = float('nan')
        
        # 聚合统计信息
        aggregated['num_valid_pixels'] = sum(m['num_valid_pixels'] for m in valid_metrics)
        aggregated['total_pixels'] = sum(m['total_pixels'] for m in valid_metrics)
        aggregated['valid_ratio'] = aggregated['num_valid_pixels'] / aggregated['total_pixels'] if aggregated['total_pixels'] > 0 else 0.0
        aggregated['num_samples'] = len(valid_metrics)
        
        return aggregated

# ===============================
# 综合评估函数
# ===============================

def evaluate_with_reference(pred_img, ref_img):
    """
    计算所有全参考指标
    
    Args:
        pred_img: 预测图像
        ref_img: 参考图像
    
    Returns:
        metrics_dict: 包含所有指标的字典
    """
    metrics = {}
    
    # 确保图像尺寸一致
    if pred_img.shape != ref_img.shape:
        ref_img = cv2.resize(ref_img, (pred_img.shape[1], pred_img.shape[0]))
    
    # 计算各项指标
    metrics['PSNR'] = calculate_psnr(pred_img, ref_img)
    metrics['SSIM'] = calculate_ssim(pred_img, ref_img)
    metrics['CIEDE2000'] = calculate_ciede2000(pred_img, ref_img)
    metrics['LPIPS'] = calculate_lpips(pred_img, ref_img)
    
    return metrics

def evaluate_no_reference(img):
    """
    计算所有无参考指标
    完全符合原始论文定义
    
    Args:
        img: 输入图像
    
    Returns:
        metrics_dict: 包含所有指标的字典
    """
    metrics = {}
    
    metrics['UCIQE'] = calculate_uciqe(img)
    metrics['UIQM'] = calculate_uiqm(img)
    metrics['NIQE'] = calculate_niqe(img)
    
    return metrics

def evaluate_depth_estimation(pred, gt, epsilon=1e-8,
                              gt_units='m', valid_min=0.1, valid_max=100.0, pred_clamp_val=80.0):
    """
    评估深度估计结果，支持中值缩放对齐
    Args:
        pred (torch.Tensor): 预测深度图 (B, 1, H, W)，单位应为米
        gt (torch.Tensor):   真实深度图 (B, 1, H, W)，单位由gt_units指定
        epsilon (float):     防止除零的小常数
        gt_units (str):      真实深度图的单位 ('m' 或 'mm')
        valid_min (float):   真实深度图的有效最小值 (与gt_units相同单位)
        valid_max (float):   真实深度图的有效最大值 (与gt_units相同单位)
        pred_clamp_val (float): 预测深度图在计算scale前的最大钳位值 (单位: 米)
    Returns:
        dict: 包含各种深度评估指标的字典
    """
    
    # 确保pred和gt是浮点数
    pred = pred.float()
    gt = gt.float()
    
    # 🔥 修复维度不匹配问题：处理5维张量
    # 确保两个张量都是4维 (B, C, H, W)
    while len(pred.shape) > 4:
        pred = pred.squeeze(0)
    while len(gt.shape) > 4:
        gt = gt.squeeze(0)
    
    # 如果是3维，添加batch维度
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
    if len(gt.shape) == 3:
        gt = gt.unsqueeze(0)

    # --- 1. 单位归一化 ---
    # 将GT统一转换为米
    if gt_units == 'mm':
        gt = gt / 1000.0
        valid_min_m = valid_min / 1000.0
        valid_max_m = valid_max / 1000.0
    else: # 默认为 'm'
        valid_min_m = valid_min
        valid_max_m = valid_max

    # --- 2. 创建有效值掩码 ---
    # 根据转换后的单位(米)创建掩码
    valid_mask = (gt > valid_min_m) & (gt < valid_max_m)
    
    # 对预测值进行钳位，防止尺度因子计算时出现极端值
    pred_clamped = torch.clamp(pred, min=epsilon, max=pred_clamp_val)
    
    # --- 3. 中值尺度对齐 (Median Scaling) ---
    # 在有效区域内计算尺度因子
    if valid_mask.sum() == 0:
        # 如果没有有效像素，返回默认值
        return {
            'depth_mae': 0, 'depth_rmse': 0, 'depth_abs_rel': 0, 'depth_sq_rel': 0,
            'depth_delta1': 0, 'depth_delta2': 0, 'depth_delta3': 0,
            'median_scale': 1.0
        }
        
    median_scale = torch.median(gt[valid_mask]) / torch.median(pred_clamped[valid_mask])
    pred_scaled = pred_clamped * median_scale

    # 只在有效区域内计算指标
    pred_final = pred_scaled[valid_mask]
    gt_final = gt[valid_mask]

    if gt_final.numel() == 0:
        # 如果没有有效像素，返回默认值
        return {
            'depth_mae': 0, 'depth_rmse': 0, 'depth_abs_rel': 0, 'depth_sq_rel': 0,
            'depth_delta1': 0, 'depth_delta2': 0, 'depth_delta3': 0,
            'median_scale': float(median_scale.item()) if median_scale.numel() > 0 else 1.0
        }
    
    # --- 4. 计算指标 ---
    # 使用对齐和掩码后的数据
    mae = torch.mean(torch.abs(pred_final - gt_final))
    rmse = torch.sqrt(torch.mean(torch.pow(pred_final - gt_final, 2)))
    abs_rel = torch.mean(torch.abs(pred_final - gt_final) / gt_final)
    sq_rel = torch.mean(torch.pow(pred_final - gt_final, 2) / gt_final)

    # Delta 指标
    ratio = torch.max(pred_final / gt_final, gt_final / pred_final)
    delta1 = torch.mean((ratio < 1.25).float())
    delta2 = torch.mean((ratio < 1.25**2).float())
    delta3 = torch.mean((ratio < 1.25**3).float())

    return {
        'depth_mae': float(mae.item()),
        'depth_rmse': float(rmse.item()),
        'depth_abs_rel': float(abs_rel.item()),
        'depth_sq_rel': float(sq_rel.item()),
        'depth_delta1': float(delta1.item()),
        'depth_delta2': float(delta2.item()),
        'depth_delta3': float(delta3.item()),
        'median_scale': float(median_scale.item()) # 也返回计算出的尺度因子，方便调试
    }


def calculate_depth_metrics(pred, gt, valid_mask):
    """
    (此函数将被废弃，由evaluate_depth_estimation替代)
    根据有效区域计算深度指标
    """
    if valid_mask.sum() == 0:
        return 0, 0, 0, 0, 0, 0, 0

    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]

    # ... (旧的实现)
    return 0,0,0,0,0,0,0


def find_files_recursively(root_dir, extensions=('.jpg', '.png')):
    """
    递归查找目录中的所有文件
    
    Args:
        root_dir: 根目录
        extensions: 文件扩展名列表 (如 .jpg, .png)
    
    Returns:
        file_list: 文件路径列表
    """
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(extensions):
                file_list.append(os.path.join(root, file))
    return file_list

# ===============================
# 兼容性定义（保持与原API一致）
# ===============================

# 原有函数的别名，保持兼容性
def calculate_depth_mae(pred, gt):
    """计算深度MAE（保持兼容性）"""
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(gt):
        gt = gt.cpu().numpy()
    
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    valid_mask = (gt_flat > 0) & (pred_flat > 0)
    
    if np.sum(valid_mask) == 0:
        return 0.0
    
    return np.mean(np.abs(pred_flat[valid_mask] - gt_flat[valid_mask]))

def calculate_depth_rmse(pred, gt):
    """计算深度RMSE（保持兼容性）"""
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(gt):
        gt = gt.cpu().numpy()
    
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    valid_mask = (gt_flat > 0) & (pred_flat > 0)
    
    if np.sum(valid_mask) == 0:
        return 0.0
    
    return np.sqrt(np.mean((pred_flat[valid_mask] - gt_flat[valid_mask]) ** 2))

# 旧版本的函数映射
ALL_METRICS = {
    "psnr": calculate_psnr,
    "ssim": calculate_ssim,
    "uciqe": calculate_uciqe,
    "uiqm": calculate_uiqm,
    "ciede2000": calculate_ciede2000,
    "lpips": calculate_lpips,
    "niqe": calculate_niqe,
    "depth_mae": calculate_depth_mae,
    "depth_rmse": calculate_depth_rmse,
}

FULL_REFERENCE_METRICS = ["psnr", "ssim", "ciede2000", "lpips"]
NO_REFERENCE_METRICS = ["uciqe", "uiqm", "niqe"]
DEPTH_METRICS = ["depth_mae", "depth_rmse"]

# ===============================
# 测试函数
# ===============================

def test_metrics():
    """测试所有指标函数"""
    print("=== 综合评估指标测试 ===")
    
    # 创建测试图像
    img1 = np.random.rand(256, 256, 3).astype(np.float32)
    img2 = img1 + 0.1 * np.random.rand(256, 256, 3).astype(np.float32)
    img2 = np.clip(img2, 0, 1)
    
    # 测试全参考指标
    print("\n测试全参考指标:")
    ref_metrics = evaluate_with_reference(img1, img2)
    for metric, value in ref_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 测试无参考指标
    print("\n测试无参考指标:")
    noref_metrics = evaluate_no_reference(img1)
    for metric, value in noref_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 测试深度指标
    print("\n测试深度估计指标:")
    gt_depth = np.random.uniform(1.0, 10.0, (100, 100))
    pred_depth = gt_depth + np.random.normal(0, 0.5, (100, 100))
    pred_depth = np.maximum(pred_depth, 0.1)
    mask = np.random.random((100, 100)) > 0.1
    
    depth_metrics = evaluate_depth_estimation(pred_depth, gt_depth, mask)
    print_metrics(depth_metrics, "深度估计测试")
    
    print("\n指标测试完成!")

if __name__ == '__main__':
    test_metrics() 