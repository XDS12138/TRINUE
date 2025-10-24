#!/usr/bin/env python3
"""
将Folder格式的多退化数据集转换为LMDB格式

优势:
- 加速随机访问
- 减少文件系统I/O
- 适合大规模数据集

LMDB结构:
  每个样本存储为一个键值对
  键: sample_{index:06d}
  值: {
    'raw_imgs': [15, 3, H, W] 的numpy数组 (15种退化)
    'gt': [3, H, W] 的numpy数组
    'depth': [1, H, W] 的numpy数组
    'basename': 文件名
  }
"""

import os
import sys
import lmdb
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

def load_image_as_array(path):
    """加载图像为numpy数组"""
    try:
        img = Image.open(path).convert('RGB')
        return np.array(img)
    except Exception as e:
        print(f"加载图像失败: {path}, {e}")
        return None

def load_depth_as_array(path):
    """加载深度图为numpy数组（16位）"""
    try:
        depth = Image.open(path)
        depth_array = np.array(depth, dtype=np.uint16)
        return depth_array
    except Exception as e:
        print(f"加载深度图失败: {path}, {e}")
        return None


def load_sample_data(basename, data_root, degradation_folders, gt_folder, depth_folder, target_size):
    """
    加载单个样本的所有数据（用于多进程）
    
    Returns:
        (basename, raw_imgs, gt, depth) 或 (basename, None, None, None) 如果出错
    """
    try:
        # 1. 加载15种退化图
        raw_imgs_pil = []
        for deg_folder in degradation_folders:
            deg_path = os.path.join(data_root, deg_folder, f"{basename}.png")
            if not os.path.exists(deg_path):
                return basename, None, None, None
            img = Image.open(deg_path).convert('RGB')
            raw_imgs_pil.append(img)
        
        # 统一尺寸
        if target_size:
            unified_size = (target_size[1], target_size[0])
        else:
            sizes = [img.size for img in raw_imgs_pil]
            min_width = min(s[0] for s in sizes)
            min_height = min(s[1] for s in sizes)
            unified_size = (min_width, min_height)
        
        # Resize并转numpy
        raw_imgs_list = []
        for img in raw_imgs_pil:
            if img.size != unified_size:
                img = img.resize(unified_size, Image.LANCZOS)
            raw_imgs_list.append(np.array(img))
        
        raw_imgs = np.stack(raw_imgs_list, axis=0)
        raw_imgs = np.transpose(raw_imgs, (0, 3, 1, 2))  # [15, 3, H, W]
        
        # 2. 加载GT
        gt_path = os.path.join(data_root, gt_folder, f"{basename}.png")
        gt_img_pil = Image.open(gt_path).convert('RGB')
        if gt_img_pil.size != unified_size:
            gt_img_pil = gt_img_pil.resize(unified_size, Image.LANCZOS)
        gt = np.transpose(np.array(gt_img_pil), (2, 0, 1))  # [3, H, W]
        
        # 3. 加载深度图
        depth_path = os.path.join(data_root, depth_folder, f"{basename}.png")
        depth_img_pil = Image.open(depth_path)
        if depth_img_pil.size != unified_size:
            depth_img_pil = depth_img_pil.resize(unified_size, Image.NEAREST)
        depth_img = np.array(depth_img_pil, dtype=np.uint16)
        depth = np.expand_dims(depth_img, axis=0) if depth_img.ndim == 2 else depth_img
        
        return basename, raw_imgs.astype(np.uint8), gt.astype(np.uint8), depth.astype(np.uint16)
        
    except Exception as e:
        return basename, None, None, None


def create_multi_degradation_lmdb(data_root, degradation_folders, gt_folder, depth_folder,
                                  output_lmdb_path, target_size=None, map_size_gb=1024, num_workers=16):
    """
    创建多退化LMDB数据集
    
    Args:
        data_root: 数据根目录 (如 F:/DATASATES/UBB_train)
        degradation_folders: 退化文件夹列表
        gt_folder: GT文件夹名
        depth_folder: 深度文件夹名
        output_lmdb_path: 输出LMDB路径
        target_size: 统一尺寸(H, W)，如果None则使用原始尺寸
        map_size_gb: LMDB最大容量(GB)
    """
    
    print("="*60)
    print("创建多退化LMDB数据集（多进程加速）")
    print("="*60)
    print(f"数据根目录: {data_root}")
    print(f"退化类型数: {len(degradation_folders)}")
    print(f"输出LMDB: {output_lmdb_path}")
    print(f"目标尺寸: {target_size if target_size else '原始尺寸'}")
    print(f"最大容量: {map_size_gb}GB")
    print(f"工作进程: {num_workers}")
    
    # 检查GT文件夹
    gt_path = os.path.join(data_root, gt_folder)
    if not os.path.exists(gt_path):
        print(f"错误: GT文件夹不存在 - {gt_path}")
        return
    
    # 获取所有样本的basename
    gt_files = sorted([f for f in os.listdir(gt_path) if f.endswith('.png')])
    basenames = [os.path.splitext(f)[0] for f in gt_files]
    
    print(f"\n找到 {len(basenames)} 个样本")
    
    # 🔥 如果target_size为None，保持每个样本的原始尺寸（不统一）
    if target_size is None:
        print(f"保持原始尺寸（不resize）")
    else:
        print(f"统一尺寸: {target_size}")
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_lmdb_path), exist_ok=True)
    
    # 创建LMDB环境
    map_size = map_size_gb * 1024 * 1024 * 1024  # 转换为字节
    env = lmdb.open(output_lmdb_path, map_size=map_size)
    
    # 🔥 批量处理避免内存溢出
    success_count = 0
    error_count = 0
    commit_interval = 500
    batch_size = 100  # 每次提交100个任务到进程池
    
    txn = env.begin(write=True)
    
    # 创建部分函数
    load_func = partial(load_sample_data, data_root=data_root, 
                       degradation_folders=degradation_folders,
                       gt_folder=gt_folder, depth_folder=depth_folder,
                       target_size=target_size)
    
    # 分批处理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        pbar = tqdm(total=len(basenames), desc="写入LMDB")
        
        for batch_start in range(0, len(basenames), batch_size):
            batch_end = min(batch_start + batch_size, len(basenames))
            batch_basenames = basenames[batch_start:batch_end]
            
            # 提交当前批次
            futures = [executor.submit(load_func, bn) for bn in batch_basenames]
            
            # 处理当前批次的结果
            for i, future in enumerate(futures):
                idx = batch_start + i
                basename, raw_imgs, gt, depth = future.result()
                
                if raw_imgs is None:
                    error_count += 1
                    if error_count <= 10:
                        print(f"\n错误 #{error_count}: {basename} - 加载失败")
                    pbar.update(1)
                    continue
                
                try:
                    sample = {
                        'raw_imgs': raw_imgs,
                        'gt': gt,
                        'depth': depth,
                        'basename': basename,
                        'num_degradations': len(degradation_folders),
                    }
                    
                    key = f"sample_{idx:06d}".encode('ascii')
                    value = pickle.dumps(sample)
                    txn.put(key, value)
                    
                    success_count += 1
                    
                    # 定期提交
                    if (idx + 1) % commit_interval == 0:
                        txn.commit()
                        txn = env.begin(write=True)
                    
                except Exception as e:
                    error_count += 1
                    if error_count <= 10:
                        print(f"\n错误 #{error_count}: {basename} - {e}")
                
                pbar.update(1)
        
        pbar.close()
    
    # 最后一次提交
    if success_count % commit_interval != 0:
        txn.commit()
    
    # 写入元数据（新事务）
    with env.begin(write=True) as txn:
        meta = {
            'num_samples': success_count,
            'num_degradations': len(degradation_folders),
            'degradation_folders': degradation_folders,
            'resolution': "mixed" if not target_size else f"{target_size[0]}x{target_size[1]}",
        }
        txn.put(b'__meta__', pickle.dumps(meta))
    
    env.close()
    
    # 总结
    print(f"\n{'='*60}")
    print("转换完成")
    print(f"{'='*60}")
    print(f"成功: {success_count} 个样本")
    print(f"失败: {error_count} 个样本")
    print(f"LMDB文件: {output_lmdb_path}")
    
    # 显示文件大小
    if os.path.exists(output_lmdb_path):
        lmdb_size = sum(os.path.getsize(os.path.join(output_lmdb_path, f)) 
                       for f in os.listdir(output_lmdb_path) 
                       if os.path.isfile(os.path.join(output_lmdb_path, f)))
        print(f"大小: {lmdb_size / (1024**3):.2f} GB")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='创建多退化LMDB数据集')
    parser.add_argument('--data-root', default='F:/DATASATES/UBB_train',
                       help='数据根目录')
    parser.add_argument('--output', default='F:/DATASATES/UBB_train.lmdb',
                       help='输出LMDB路径')
    parser.add_argument('--target-size', type=int, nargs=2, default=None,
                       help='统一图像尺寸(H W)，如: --target-size 256 256')
    parser.add_argument('--map-size-gb', type=int, default=1024,
                       help='LMDB最大容量(GB)')
    parser.add_argument('--num-workers', type=int, default=16,
                       help='并行加载进程数（默认16）')
    
    args = parser.parse_args()
    
    # 转换target_size为tuple
    target_size = tuple(args.target_size) if args.target_size else None
    
    # 退化文件夹列表
    degradation_folders = [
        'color_B_1', 'color_B_2', 'color_B_3',
        'color_BG_1', 'color_BG_2', 'color_BG_3',
        'color_G_1', 'color_G_2', 'color_G_3',
        'color_Y_1', 'color_Y_2', 'color_Y_3',
        'color_YG_1', 'color_YG_2', 'color_YG_3',
    ]
    
    create_multi_degradation_lmdb(
        data_root=args.data_root,
        degradation_folders=degradation_folders,
        gt_folder='gt',
        depth_folder='depth',
        output_lmdb_path=args.output,
        target_size=target_size,
        map_size_gb=args.map_size_gb,
        num_workers=args.num_workers
    )
    
    return 0


if __name__ == '__main__':
    try:
        import lmdb
    except ImportError:
        print("错误: 需要安装lmdb库")
        print("安装命令: pip install lmdb")
        sys.exit(1)
    
    sys.exit(main())

