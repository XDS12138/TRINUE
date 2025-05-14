#!/usr/bin/env python3
"""
从train/gt_backup随机选择10%的图像移动到val/gt_backup目录
"""

import os
import glob
import random
import shutil

# 路径定义
TRAIN_GT_DIR = "DATA/train/gt_backup"
VAL_GT_DIR = "DATA/val/gt_backup"

def move_files_to_val():
    """随机选择10%的文件移动到验证集"""
    # 确保目标目录存在
    os.makedirs(VAL_GT_DIR, exist_ok=True)
    
    # 清空val目录中已有的文件
    existing_val_files = glob.glob(os.path.join(VAL_GT_DIR, "*.png"))
    for f in existing_val_files:
        os.remove(f)
    print(f"已清空验证集目录，删除了 {len(existing_val_files)} 个文件")
    
    # 获取所有训练集文件
    train_files = glob.glob(os.path.join(TRAIN_GT_DIR, "*.png"))
    print(f"训练集目录中有 {len(train_files)} 个文件")
    
    # 计算需要移动的文件数量（10%）
    num_to_move = int(len(train_files) * 0.1)
    print(f"将随机选择 {num_to_move} 个文件移动到验证集")
    
    # 随机选择文件
    files_to_move = random.sample(train_files, num_to_move)
    
    # 移动文件
    moved_count = 0
    for file_path in files_to_move:
        filename = os.path.basename(file_path)
        target_path = os.path.join(VAL_GT_DIR, filename)
        
        shutil.move(file_path, target_path)
        moved_count += 1
        
        if moved_count % 50 == 0:
            print(f"已移动 {moved_count}/{num_to_move} 个文件...")
    
    print(f"\n移动完成！已将 {moved_count} 个文件从训练集移动到验证集")
    print(f"现在训练集有 {len(glob.glob(os.path.join(TRAIN_GT_DIR, '*.png')))} 个文件")
    print(f"现在验证集有 {len(glob.glob(os.path.join(VAL_GT_DIR, '*.png')))} 个文件")

if __name__ == "__main__":
    # 设置随机种子，确保结果可重现
    random.seed(42)
    move_files_to_val() 