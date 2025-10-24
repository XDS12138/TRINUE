#!/usr/bin/env python3
"""
将单输入格式训练集转换为LMDB

输入结构:
  D:/UBB_train_single_input/
    ├── input/ (166,470个)
    └── gt/    (166,470个)

输出LMDB结构:
  每个样本存储:
  {
    'input': [3, H, W] numpy数组 (RGB)
    'gt': [3, H, W] numpy数组 (RGB)
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
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def load_single_sample(filename, input_dir, gt_dir, target_size):
    """加载单个样本（用于多进程）"""
    try:
        # 加载input
        input_path = os.path.join(input_dir, filename)
        input_img = Image.open(input_path).convert('RGB')
        
        # 加载GT
        gt_path = os.path.join(gt_dir, filename)
        gt_img = Image.open(gt_path).convert('RGB')
        
        # 获取统一尺寸
        if target_size:
            unified_size = (target_size[1], target_size[0])
        else:
            # 使用input的尺寸
            unified_size = input_img.size
        
        # Resize
        if input_img.size != unified_size:
            input_img = input_img.resize(unified_size, Image.LANCZOS)
        if gt_img.size != unified_size:
            gt_img = gt_img.resize(unified_size, Image.LANCZOS)
        
        # 转numpy
        input_array = np.transpose(np.array(input_img), (2, 0, 1))  # [3,H,W]
        gt_array = np.transpose(np.array(gt_img), (2, 0, 1))
        
        return filename, input_array.astype(np.uint8), gt_array.astype(np.uint8)
    except Exception as e:
        return filename, None, None


def create_single_input_lmdb(input_dir, gt_dir, output_lmdb_path, 
                             target_size=None, map_size_gb=1024, num_workers=24):
    """
    创建单输入LMDB数据集
    
    Args:
        input_dir: input文件夹路径
        gt_dir: GT文件夹路径
        output_lmdb_path: 输出LMDB路径
        target_size: 统一尺寸(H, W)，None则保持原始
        map_size_gb: LMDB最大容量(GB)
    """
    
    print("="*60)
    print("创建单输入LMDB数据集（多进程加速）")
    print("="*60)
    print(f"Input目录: {input_dir}")
    print(f"GT目录: {gt_dir}")
    print(f"输出LMDB: {output_lmdb_path}")
    print(f"目标尺寸: {target_size if target_size else '保持原始'}")
    print(f"最大容量: {map_size_gb}GB")
    print(f"工作进程: {num_workers}")
    
    # 检查目录
    if not os.path.exists(input_dir):
        print(f"错误: Input目录不存在 - {input_dir}")
        return
    
    if not os.path.exists(gt_dir):
        print(f"错误: GT目录不存在 - {gt_dir}")
        return
    
    # 获取所有input文件
    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    print(f"\n找到 {len(input_files)} 个样本")
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_lmdb_path), exist_ok=True)
    
    # 创建LMDB环境
    map_size = map_size_gb * 1024 * 1024 * 1024
    env = lmdb.open(output_lmdb_path, map_size=map_size)
    
    success_count = 0
    error_count = 0
    
    # 🔥 批量处理避免内存溢出
    commit_interval = 1000
    batch_size = 200  # 每批200个任务
    
    txn = env.begin(write=True)
    
    load_func = partial(load_single_sample, input_dir=input_dir, 
                       gt_dir=gt_dir, target_size=target_size)
    
    # 分批处理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        pbar = tqdm(total=len(input_files), desc="写入LMDB")
        
        for batch_start in range(0, len(input_files), batch_size):
            batch_end = min(batch_start + batch_size, len(input_files))
            batch_files = input_files[batch_start:batch_end]
            
            # 提交当前批次
            futures = [executor.submit(load_func, fn) for fn in batch_files]
            
            # 处理结果
            for i, future in enumerate(futures):
                idx = batch_start + i
                filename, input_array, gt_array = future.result()
                
                if input_array is None:
                    error_count += 1
                    if error_count <= 10:
                        print(f"\n错误 #{error_count}: {filename} - 加载失败")
                    pbar.update(1)
                    continue
                
                try:
                    basename = os.path.splitext(filename)[0]
                    
                    sample = {
                        'input': input_array,
                        'gt': gt_array,
                        'basename': basename,
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
                        print(f"\n错误 #{error_count}: {filename} - {e}")
                
                pbar.update(1)
        
        pbar.close()
    
    # 最后一次提交
    if success_count % commit_interval != 0:
        txn.commit()
    
    # 写入元数据（新事务）
    with env.begin(write=True) as txn:
        meta = {
            'num_samples': success_count,
            'resolution': "mixed" if not target_size else f"{target_size[0]}x{target_size[1]}",
            'format': 'single_input',
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
    
    parser = argparse.ArgumentParser(description='创建单输入LMDB数据集')
    parser.add_argument('--input-dir', default='D:/UBB_train_single_input/input',
                       help='Input文件夹路径')
    parser.add_argument('--gt-dir', default='D:/UBB_train_single_input/gt',
                       help='GT文件夹路径')
    parser.add_argument('--output', default='E:/DATASATES/UBB_train_single_input.lmdb',
                       help='输出LMDB路径（默认E盘）')
    parser.add_argument('--target-size', type=int, nargs=2, default=None,
                       help='统一图像尺寸(H W)，如: --target-size 540 960')
    parser.add_argument('--map-size-gb', type=int, default=1024,
                       help='LMDB最大容量(GB)，默认1TB')
    parser.add_argument('--num-workers', type=int, default=24,
                       help='并行加载进程数（默认24）')
    
    args = parser.parse_args()
    
    # 转换target_size为tuple
    target_size = tuple(args.target_size) if args.target_size else None
    
    # 检查依赖
    try:
        import lmdb
    except ImportError:
        print("错误: 需要安装lmdb库")
        print("安装命令: pip install lmdb")
        return 1
    
    # 执行转换
    create_single_input_lmdb(
        input_dir=args.input_dir,
        gt_dir=args.gt_dir,
        output_lmdb_path=args.output,
        target_size=target_size,
        map_size_gb=args.map_size_gb,
        num_workers=args.num_workers
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

