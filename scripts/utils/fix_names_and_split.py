#!/usr/bin/env python3
"""
1. 修复文件名中的不一致（删除文件名中的A/B/D字母）
2. 从训练集中取出序列靠后的400张图片移到验证集
"""

import os
import glob
import shutil
import re

# 路径定义
TRAIN_RAW_DIR = "DATA/train/raw"
TRAIN_DEPTH_DIR = "DATA/train/depth"
TRAIN_GT_DIR = "DATA/train/gt"

VAL_RAW_DIR = "DATA/val/raw"
VAL_DEPTH_DIR = "DATA/val/depth"
VAL_GT_DIR = "DATA/val/gt"

def fix_filenames():
    """删除文件名中的A/B/D标识"""
    print("正在修复文件名...")
    
    # 处理raw目录
    renamed_raw = 0
    for file in glob.glob(os.path.join(TRAIN_RAW_DIR, "*.png")):
        filename = os.path.basename(file)
        # 替换 _A, _B 或 _D 为空
        new_name = re.sub(r'_[ABD]-', '-', filename)
        if new_name != filename:
            new_path = os.path.join(TRAIN_RAW_DIR, new_name)
            shutil.move(file, new_path)
            renamed_raw += 1
    
    print(f"修复了 {renamed_raw} 个raw文件名")
    
    # 处理depth目录
    renamed_depth = 0
    for file in glob.glob(os.path.join(TRAIN_DEPTH_DIR, "*.png")):
        filename = os.path.basename(file)
        # 替换 _A, _B 或 _D 为空
        new_name = re.sub(r'_[ABD]-', '-', filename)
        if new_name != filename:
            new_path = os.path.join(TRAIN_DEPTH_DIR, new_name)
            shutil.move(file, new_path)
            renamed_depth += 1
    
    print(f"修复了 {renamed_depth} 个depth文件名")
    
    # 处理gt目录
    renamed_gt = 0
    for file in glob.glob(os.path.join(TRAIN_GT_DIR, "*.png")):
        filename = os.path.basename(file)
        # 替换 _A, _B 或 _D 为空
        new_name = re.sub(r'_[ABD]-', '-', filename)
        if new_name != filename:
            new_path = os.path.join(TRAIN_GT_DIR, new_name)
            shutil.move(file, new_path)
            renamed_gt += 1
    
    print(f"修复了 {renamed_gt} 个gt文件名")

def split_dataset():
    """从训练集中抽取序列靠后的400张图片到验证集"""
    print("\n正在拆分数据集...")
    
    # 确保验证集目录存在
    os.makedirs(VAL_RAW_DIR, exist_ok=True)
    os.makedirs(VAL_DEPTH_DIR, exist_ok=True)
    os.makedirs(VAL_GT_DIR, exist_ok=True)
    
    # 清空验证集目录
    for directory in [VAL_RAW_DIR, VAL_DEPTH_DIR, VAL_GT_DIR]:
        for file in glob.glob(os.path.join(directory, "*.png")):
            os.remove(file)
    
    # 获取raw目录中的所有文件并按序列号排序
    raw_files = glob.glob(os.path.join(TRAIN_RAW_DIR, "*.png"))
    
    # 提取序列号并排序
    file_with_seq = []
    for file in raw_files:
        filename = os.path.basename(file)
        match = re.search(r'-(\d+)\.png$', filename)
        if match:
            seq_num = int(match.group(1))
            file_with_seq.append((file, seq_num))
    
    # 按序列号排序
    file_with_seq.sort(key=lambda x: x[1], reverse=True)  # 倒序排列，序列靠后的排在前面
    
    # 选取前400个文件
    files_to_move = file_with_seq[:400]
    print(f"选择了 {len(files_to_move)} 个序列靠后的文件")
    
    # 移动文件到验证集
    moved_count = 0
    for file_info in files_to_move:
        raw_file, _ = file_info
        filename = os.path.basename(raw_file)
        
        # 源文件路径
        train_raw_path = raw_file
        train_depth_path = os.path.join(TRAIN_DEPTH_DIR, filename)
        train_gt_path = os.path.join(TRAIN_GT_DIR, filename)
        
        # 目标文件路径
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
                print(f"已移动 {moved_count}/400 个文件")
    
    print(f"\n成功移动 {moved_count} 个文件到验证集")

def print_summary():
    """打印数据集统计信息"""
    train_raw = len(glob.glob(os.path.join(TRAIN_RAW_DIR, "*.png")))
    train_depth = len(glob.glob(os.path.join(TRAIN_DEPTH_DIR, "*.png")))
    train_gt = len(glob.glob(os.path.join(TRAIN_GT_DIR, "*.png")))
    
    val_raw = len(glob.glob(os.path.join(VAL_RAW_DIR, "*.png")))
    val_depth = len(glob.glob(os.path.join(VAL_DEPTH_DIR, "*.png")))
    val_gt = len(glob.glob(os.path.join(VAL_GT_DIR, "*.png")))
    
    print("\n=== 数据集汇总 ===")
    print(f"训练集: Raw={train_raw}, Depth={train_depth}, GT={train_gt}")
    print(f"验证集: Raw={val_raw}, Depth={val_depth}, GT={val_gt}")
    print(f"总计: Raw={train_raw+val_raw}, Depth={train_depth+val_depth}, GT={train_gt+val_gt}")

if __name__ == "__main__":
    print("=== 开始处理数据集 ===")
    
    # 步骤1: 修复文件名
    fix_filenames()
    
    # 步骤2: 拆分数据集
    split_dataset()
    
    # 打印汇总信息
    print_summary()
    
    print("\n数据集处理完成!") 