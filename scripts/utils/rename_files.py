#!/usr/bin/env python3
"""
重命名水下图像数据集文件脚本
将不同文件夹中的文件名统一为相同的格式，保留原始序号
"""

import os
import glob
import shutil
import re

# 数据目录
DATA_ROOT = "DATA"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")

# 文件夹路径
def get_folders(base_dir):
    return {
        "raw": os.path.join(base_dir, "raw"),
        "gt": os.path.join(base_dir, "gt"),
        "depth": os.path.join(base_dir, "depth")
    }

def create_backup(folder):
    """为文件夹创建备份"""
    backup_folder = folder + "_backup"
    if not os.path.exists(backup_folder):
        print(f"创建备份: {backup_folder}")
        shutil.copytree(folder, backup_folder)
    else:
        print(f"备份已存在: {backup_folder}")

def restore_from_backup(folder):
    """从备份恢复文件"""
    backup_folder = folder + "_backup"
    if os.path.exists(backup_folder):
        print(f"从备份恢复: {backup_folder} -> {folder}")
        # 清空当前文件夹内容
        for f in glob.glob(os.path.join(folder, "*")):
            if os.path.isfile(f):
                os.remove(f)
        # 从备份复制文件
        for f in glob.glob(os.path.join(backup_folder, "*")):
            if os.path.isfile(f):
                shutil.copy(f, folder)
        return True
    else:
        print(f"备份不存在: {backup_folder}")
        return False

def rename_files(dataset_dir):
    """重命名指定数据集目录下的文件"""
    folders = get_folders(dataset_dir)
    
    # 确保所有文件夹存在备份
    for folder in folders.values():
        create_backup(folder)
        # 先从备份恢复，以防之前的重命名操作有问题
        restore_from_backup(folder)
    
    # 重命名raw文件
    raw_folder = folders["raw"]
    for raw_file in glob.glob(os.path.join(raw_folder, "*.png")):
        basename = os.path.basename(raw_file)
        # 提取前缀和序号，去掉A字母
        match = re.match(r"(seq\d+_veh\d+_cam\w+)_A-(\d+)\.png", basename)
        if match:
            prefix, seq_num = match.groups()
            new_name = f"{prefix}-{seq_num}.png"
            new_path = os.path.join(raw_folder, new_name)
            print(f"重命名: {basename} -> {new_name}")
            os.rename(raw_file, new_path)
    
    # 重命名gt文件
    gt_folder = folders["gt"]
    for gt_file in glob.glob(os.path.join(gt_folder, "*.png")):
        basename = os.path.basename(gt_file)
        # 提取前缀和序号，去掉B字母
        match = re.match(r"(seq\d+_veh\d+_cam\w+)_B-(\d+)\.png", basename)
        if match:
            prefix, seq_num = match.groups()
            new_name = f"{prefix}-{seq_num}.png"
            new_path = os.path.join(gt_folder, new_name)
            print(f"重命名: {basename} -> {new_name}")
            os.rename(gt_file, new_path)
    
    # 重命名depth文件
    depth_folder = folders["depth"]
    for depth_file in glob.glob(os.path.join(depth_folder, "*.png")):
        basename = os.path.basename(depth_file)
        # 提取前缀和序号，去掉C字母
        match = re.match(r"(seq\d+_veh\d+_cam\w+)_C-(\d+)\.png", basename)
        if match:
            prefix, seq_num = match.groups()
            new_name = f"{prefix}-{seq_num}.png"
            new_path = os.path.join(depth_folder, new_name)
            print(f"重命名: {basename} -> {new_name}")
            os.rename(depth_file, new_path)

if __name__ == "__main__":
    # 创建备份并重命名训练集文件
    print("处理训练集...")
    rename_files(TRAIN_DIR)
    
    # 创建备份并重命名验证集文件
    print("\n处理验证集...")
    rename_files(VAL_DIR)
    
    print("\n重命名完成!")
    print("如果需要恢复原始文件名，请使用备份文件夹中的文件") 