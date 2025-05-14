#!/usr/bin/env python3
"""
处理数据集：
1. 将depth_backup和gt_backup中的文件按照raw目录中的命名格式重命名
2. 将10%的数据同步移动到验证集
"""

import os
import glob
import shutil
import random
from pathlib import Path
import re

# 路径定义
TRAIN_RAW_DIR = "DATA/train/raw"
TRAIN_DEPTH_DIR = "DATA/train/depth_backup"
TRAIN_GT_DIR = "DATA/train/gt_backup"

VAL_RAW_DIR = "DATA/val/raw"
VAL_DEPTH_DIR = "DATA/val/depth_backup"
VAL_GT_DIR = "DATA/val/gt_backup"

def get_file_mapping(directory):
    """获取目录中文件名与序列号的映射"""
    files = glob.glob(os.path.join(directory, "*.png"))
    mapping = {}
    
    for file_path in files:
        filename = os.path.basename(file_path)
        # 提取序列号
        match = re.search(r'-(\d+)\.png$', filename)
        if match:
            seq_num = match.group(1)
            mapping[seq_num] = filename
    
    return mapping

def rename_files():
    """重命名depth和gt目录中的文件，使其与raw目录中的命名格式一致"""
    # 获取raw目录中的文件
    raw_files = glob.glob(os.path.join(TRAIN_RAW_DIR, "*.png"))
    print(f"Raw目录中有 {len(raw_files)} 个文件")
    
    # 获取depth和gt目录中文件的映射
    depth_mapping = get_file_mapping(TRAIN_DEPTH_DIR)
    gt_mapping = get_file_mapping(TRAIN_GT_DIR)
    
    renamed_count = 0
    
    # 遍历raw文件，并相应地重命名depth和gt文件
    for raw_file in raw_files:
        raw_filename = os.path.basename(raw_file)
        # 获取序列号
        match = re.search(r'-(\d+)\.png$', raw_filename)
        if match:
            seq_num = match.group(1)
            
            # 处理depth文件
            if seq_num in depth_mapping:
                old_depth_name = depth_mapping[seq_num]
                new_depth_name = raw_filename
                
                old_depth_path = os.path.join(TRAIN_DEPTH_DIR, old_depth_name)
                new_depth_path = os.path.join(TRAIN_DEPTH_DIR, new_depth_name)
                
                if old_depth_name != new_depth_name:
                    shutil.move(old_depth_path, new_depth_path)
            
            # 处理gt文件
            if seq_num in gt_mapping:
                old_gt_name = gt_mapping[seq_num]
                new_gt_name = raw_filename
                
                old_gt_path = os.path.join(TRAIN_GT_DIR, old_gt_name)
                new_gt_path = os.path.join(TRAIN_GT_DIR, new_gt_name)
                
                if old_gt_name != new_gt_name:
                    shutil.move(old_gt_path, new_gt_path)
            
            renamed_count += 1
            if renamed_count % 100 == 0:
                print(f"已处理 {renamed_count} 个文件...")
    
    print(f"\n重命名完成！处理了 {renamed_count} 个文件")

def move_to_validation():
    """将10%的数据同步移动到验证集"""
    # 确保验证集目录存在
    os.makedirs(VAL_RAW_DIR, exist_ok=True)
    os.makedirs(VAL_DEPTH_DIR, exist_ok=True)
    os.makedirs(VAL_GT_DIR, exist_ok=True)
    
    # 清空验证集目录
    for directory in [VAL_RAW_DIR, VAL_DEPTH_DIR, VAL_GT_DIR]:
        for file in glob.glob(os.path.join(directory, "*.png")):
            os.remove(file)
    
    print("已清空验证集目录")
    
    # 获取训练集中的raw文件
    raw_files = glob.glob(os.path.join(TRAIN_RAW_DIR, "*.png"))
    print(f"训练集中有 {len(raw_files)} 个原始图像")
    
    # 计算需要移动的文件数量
    num_to_move = int(len(raw_files) * 0.1)
    print(f"将移动 {num_to_move} 个样本到验证集")
    
    # 随机选择文件
    files_to_move = random.sample(raw_files, num_to_move)
    
    # 移动文件
    moved_count = 0
    for raw_file in files_to_move:
        filename = os.path.basename(raw_file)
        
        # 定义相应的depth和gt文件路径
        train_raw_path = raw_file
        train_depth_path = os.path.join(TRAIN_DEPTH_DIR, filename)
        train_gt_path = os.path.join(TRAIN_GT_DIR, filename)
        
        val_raw_path = os.path.join(VAL_RAW_DIR, filename)
        val_depth_path = os.path.join(VAL_DEPTH_DIR, filename)
        val_gt_path = os.path.join(VAL_GT_DIR, filename)
        
        # 检查所有文件是否存在
        if os.path.exists(train_raw_path) and os.path.exists(train_depth_path) and os.path.exists(train_gt_path):
            # 移动文件
            shutil.move(train_raw_path, val_raw_path)
            shutil.move(train_depth_path, val_depth_path)
            shutil.move(train_gt_path, val_gt_path)
            
            moved_count += 1
            if moved_count % 50 == 0:
                print(f"已移动 {moved_count}/{num_to_move} 个样本...")
    
    print(f"\n移动完成！成功移动了 {moved_count} 个样本到验证集")
    print(f"训练集现在有：")
    print(f"  Raw: {len(glob.glob(os.path.join(TRAIN_RAW_DIR, '*.png')))} 个文件")
    print(f"  Depth: {len(glob.glob(os.path.join(TRAIN_DEPTH_DIR, '*.png')))} 个文件")
    print(f"  GT: {len(glob.glob(os.path.join(TRAIN_GT_DIR, '*.png')))} 个文件")
    print(f"验证集现在有：")
    print(f"  Raw: {len(glob.glob(os.path.join(VAL_RAW_DIR, '*.png')))} 个文件")
    print(f"  Depth: {len(glob.glob(os.path.join(VAL_DEPTH_DIR, '*.png')))} 个文件")
    print(f"  GT: {len(glob.glob(os.path.join(VAL_GT_DIR, '*.png')))} 个文件")

def copy_to_working_dirs():
    """将backup目录中的文件复制到工作目录中"""
    # 复制训练集文件
    for src, dst in [
        (TRAIN_DEPTH_DIR, "DATA/train/depth"),
        (TRAIN_GT_DIR, "DATA/train/gt"),
        (VAL_DEPTH_DIR, "DATA/val/depth"),
        (VAL_GT_DIR, "DATA/val/gt")
    ]:
        os.makedirs(dst, exist_ok=True)
        files = glob.glob(os.path.join(src, "*.png"))
        copied = 0
        
        for file in files:
            filename = os.path.basename(file)
            dst_path = os.path.join(dst, filename)
            shutil.copy2(file, dst_path)
            copied += 1
            
        print(f"已将 {copied} 个文件从 {src} 复制到 {dst}")

if __name__ == "__main__":
    random.seed(42)  # 设置随机种子，确保结果可重现
    
    # 步骤1：重命名文件
    print("=== 步骤1：重命名文件 ===")
    rename_files()
    
    # 步骤2：移动到验证集
    print("\n=== 步骤2：移动到验证集 ===")
    move_to_validation()
    
    # 步骤3：复制到工作目录
    print("\n=== 步骤3：复制到工作目录 ===")
    copy_to_working_dirs()
    
    print("\n数据集处理完成!") 