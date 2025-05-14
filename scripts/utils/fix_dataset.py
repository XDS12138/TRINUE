#!/usr/bin/env python3
"""
修复数据集问题：
1. 确保raw、depth、gt文件数量一致
2. 检查文件名格式并统一
3. 重新分配训练集和验证集
"""

import os
import glob
import shutil
import random
import re

# 路径定义
TRAIN_RAW_DIR = "DATA/train/raw"
TRAIN_DEPTH_DIR = "DATA/train/depth_backup"
TRAIN_GT_DIR = "DATA/train/gt_backup"
TRAIN_DEPTH_WORK_DIR = "DATA/train/depth"
TRAIN_GT_WORK_DIR = "DATA/train/gt"

VAL_RAW_DIR = "DATA/val/raw"
VAL_DEPTH_DIR = "DATA/val/depth_backup"
VAL_GT_DIR = "DATA/val/gt_backup"
VAL_DEPTH_WORK_DIR = "DATA/val/depth"
VAL_GT_WORK_DIR = "DATA/val/gt"

# 临时恢复目录
BACKUP_RAW_DIR = "DATA/train/raw_backup"

def backup_files():
    """备份当前所有文件"""
    print("正在备份文件...")
    
    # 创建raw备份目录
    os.makedirs(BACKUP_RAW_DIR, exist_ok=True)
    
    # 备份raw文件
    for file in glob.glob(os.path.join(TRAIN_RAW_DIR, "*.png")):
        filename = os.path.basename(file)
        dest = os.path.join(BACKUP_RAW_DIR, filename)
        if not os.path.exists(dest):
            shutil.copy2(file, dest)
    
    # 备份val/raw文件
    for file in glob.glob(os.path.join(VAL_RAW_DIR, "*.png")):
        filename = os.path.basename(file)
        dest = os.path.join(BACKUP_RAW_DIR, filename)
        if not os.path.exists(dest):
            shutil.copy2(file, dest)
    
    raw_count = len(glob.glob(os.path.join(BACKUP_RAW_DIR, "*.png")))
    print(f"备份了 {raw_count} 个raw文件")

def restore_raw_files():
    """恢复raw文件"""
    print("正在恢复raw文件...")
    
    # 清空现有目录
    for file in glob.glob(os.path.join(TRAIN_RAW_DIR, "*.png")):
        os.remove(file)
    for file in glob.glob(os.path.join(VAL_RAW_DIR, "*.png")):
        os.remove(file)
    
    # 从备份恢复所有文件到训练集目录
    copied = 0
    for file in glob.glob(os.path.join(BACKUP_RAW_DIR, "*.png")):
        filename = os.path.basename(file)
        dest = os.path.join(TRAIN_RAW_DIR, filename)
        shutil.copy2(file, dest)
        copied += 1
    
    print(f"已恢复 {copied} 个raw文件到训练集目录")

def fix_file_names():
    """修复文件名称问题"""
    print("正在修复文件名称...")
    
    # 获取所有raw文件的序列号
    raw_files = glob.glob(os.path.join(TRAIN_RAW_DIR, "*.png"))
    raw_seqs = {}
    for file in raw_files:
        filename = os.path.basename(file)
        match = re.search(r'-(\d+)\.png$', filename)
        if match:
            seq_num = match.group(1)
            prefix = filename.split(f"-{seq_num}.png")[0]
            raw_seqs[seq_num] = (filename, prefix)
    
    print(f"在raw目录中找到 {len(raw_seqs)} 个文件")
    
    # 修复depth文件名
    renamed_depth = 0
    for file in glob.glob(os.path.join(TRAIN_DEPTH_DIR, "*.png")):
        filename = os.path.basename(file)
        match = re.search(r'-(\d+)\.png$', filename)
        if match:
            seq_num = match.group(1)
            if seq_num in raw_seqs:
                new_name = raw_seqs[seq_num][0]
                if filename != new_name:
                    new_path = os.path.join(TRAIN_DEPTH_DIR, new_name)
                    shutil.move(file, new_path)
                    renamed_depth += 1
    
    print(f"重命名了 {renamed_depth} 个depth文件")
    
    # 修复gt文件名
    renamed_gt = 0
    for file in glob.glob(os.path.join(TRAIN_GT_DIR, "*.png")):
        filename = os.path.basename(file)
        match = re.search(r'-(\d+)\.png$', filename)
        if match:
            seq_num = match.group(1)
            if seq_num in raw_seqs:
                new_name = raw_seqs[seq_num][0]
                if filename != new_name:
                    new_path = os.path.join(TRAIN_GT_DIR, new_name)
                    shutil.move(file, new_path)
                    renamed_gt += 1
    
    print(f"重命名了 {renamed_gt} 个gt文件")

def verify_files():
    """验证文件一致性"""
    print("正在验证文件一致性...")
    
    # 获取各目录文件数量
    raw_files = set(os.path.basename(f) for f in glob.glob(os.path.join(TRAIN_RAW_DIR, "*.png")))
    depth_files = set(os.path.basename(f) for f in glob.glob(os.path.join(TRAIN_DEPTH_DIR, "*.png")))
    gt_files = set(os.path.basename(f) for f in glob.glob(os.path.join(TRAIN_GT_DIR, "*.png")))
    
    print(f"Raw文件: {len(raw_files)}个")
    print(f"Depth文件: {len(depth_files)}个")
    print(f"GT文件: {len(gt_files)}个")
    
    # 找出三个目录都存在的文件
    common_files = raw_files.intersection(depth_files).intersection(gt_files)
    print(f"三个目录共有: {len(common_files)}个文件")
    
    # 删除不一致的文件
    for directory, file_set in [
        (TRAIN_RAW_DIR, raw_files - common_files),
        (TRAIN_DEPTH_DIR, depth_files - common_files),
        (TRAIN_GT_DIR, gt_files - common_files)
    ]:
        for filename in file_set:
            file_path = os.path.join(directory, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
    
    print("已删除不一致的文件")
    
    # 再次检查文件数量
    raw_count = len(glob.glob(os.path.join(TRAIN_RAW_DIR, "*.png")))
    depth_count = len(glob.glob(os.path.join(TRAIN_DEPTH_DIR, "*.png")))
    gt_count = len(glob.glob(os.path.join(TRAIN_GT_DIR, "*.png")))
    
    print(f"处理后文件数量 - Raw: {raw_count}, Depth: {depth_count}, GT: {gt_count}")

def split_dataset():
    """重新分割数据集"""
    print("正在重新分割数据集...")
    
    # 确保验证集目录存在
    os.makedirs(VAL_RAW_DIR, exist_ok=True)
    os.makedirs(VAL_DEPTH_DIR, exist_ok=True)
    os.makedirs(VAL_GT_DIR, exist_ok=True)
    
    # 清空验证集目录
    for directory in [VAL_RAW_DIR, VAL_DEPTH_DIR, VAL_GT_DIR]:
        for file in glob.glob(os.path.join(directory, "*.png")):
            os.remove(file)
    
    # 获取训练集中的文件
    raw_files = glob.glob(os.path.join(TRAIN_RAW_DIR, "*.png"))
    total_files = len(raw_files)
    print(f"训练集中有 {total_files} 个文件")
    
    # 计算需要移动的文件数量
    num_to_move = int(total_files * 0.1)
    print(f"将移动 {num_to_move} 个文件到验证集")
    
    # 随机选择文件
    files_to_move = random.sample(raw_files, num_to_move)
    
    # 移动文件
    moved_count = 0
    for raw_file in files_to_move:
        filename = os.path.basename(raw_file)
        
        # 定义相应的文件路径
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
                print(f"已移动 {moved_count}/{num_to_move} 个文件")
    
    print(f"成功移动 {moved_count} 个文件到验证集")

def copy_to_working_dirs():
    """复制文件到工作目录"""
    print("正在复制文件到工作目录...")
    
    # 定义复制来源和目标
    copy_map = [
        (TRAIN_DEPTH_DIR, TRAIN_DEPTH_WORK_DIR),
        (TRAIN_GT_DIR, TRAIN_GT_WORK_DIR),
        (VAL_DEPTH_DIR, VAL_DEPTH_WORK_DIR),
        (VAL_GT_DIR, VAL_GT_WORK_DIR)
    ]
    
    # 清空目标目录并复制文件
    for src, dst in copy_map:
        # 创建目标目录
        os.makedirs(dst, exist_ok=True)
        
        # 清空目标目录
        for file in glob.glob(os.path.join(dst, "*.png")):
            os.remove(file)
        
        # 复制文件
        copied = 0
        for file in glob.glob(os.path.join(src, "*.png")):
            filename = os.path.basename(file)
            dst_path = os.path.join(dst, filename)
            shutil.copy2(file, dst_path)
            copied += 1
        
        print(f"已复制 {copied} 个文件从 {src} 到 {dst}")

def print_summary():
    """打印数据集汇总信息"""
    train_raw = len(glob.glob(os.path.join(TRAIN_RAW_DIR, "*.png")))
    train_depth = len(glob.glob(os.path.join(TRAIN_DEPTH_WORK_DIR, "*.png")))
    train_gt = len(glob.glob(os.path.join(TRAIN_GT_WORK_DIR, "*.png")))
    
    val_raw = len(glob.glob(os.path.join(VAL_RAW_DIR, "*.png")))
    val_depth = len(glob.glob(os.path.join(VAL_DEPTH_WORK_DIR, "*.png")))
    val_gt = len(glob.glob(os.path.join(VAL_GT_WORK_DIR, "*.png")))
    
    print("\n=== 数据集汇总 ===")
    print(f"训练集: Raw={train_raw}, Depth={train_depth}, GT={train_gt}")
    print(f"验证集: Raw={val_raw}, Depth={val_depth}, GT={val_gt}")
    print(f"总计: Raw={train_raw+val_raw}, Depth={train_depth+val_depth}, GT={train_gt+val_gt}")

if __name__ == "__main__":
    random.seed(42)  # 设置随机种子，确保结果可重现
    
    print("=== 开始修复数据集 ===")
    
    # 步骤1: 备份文件
    backup_files()
    
    # 步骤2: 恢复raw文件
    restore_raw_files()
    
    # 步骤3: 修复文件名
    fix_file_names()
    
    # 步骤4: 验证文件一致性
    verify_files()
    
    # 步骤5: 分割数据集
    split_dataset()
    
    # 步骤6: 复制到工作目录
    copy_to_working_dirs()
    
    # 打印汇总信息
    print_summary()
    
    print("\n数据集修复完成!") 