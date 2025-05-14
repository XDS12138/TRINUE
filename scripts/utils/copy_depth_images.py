#!/usr/bin/env python3
"""
从原始路径复制深度图像到目标目录
如果文件已存在则跳过
"""

import os
import shutil
import glob

# 路径定义
SOURCE_DIR = "/media/xxx/233-3/data/TURINE/2021-08-17_SEQ1/vehicle0/cam0/D"
TARGET_DIR = "DATA/train/depth_backup"

def copy_depth_files():
    """复制深度图文件并返回复制的数量"""
    # 确保目标目录存在
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # 获取所有源文件
    source_files = glob.glob(os.path.join(SOURCE_DIR, "*.png"))
    print(f"源目录中找到 {len(source_files)} 个文件")
    
    # 获取目标目录中已有的文件名集合
    existing_files = set(os.path.basename(f) for f in glob.glob(os.path.join(TARGET_DIR, "*.png")))
    print(f"目标目录中已有 {len(existing_files)} 个文件")
    
    # 计数器
    copied_count = 0
    skipped_count = 0
    
    # 复制文件
    for source_file in source_files:
        filename = os.path.basename(source_file)
        target_file = os.path.join(TARGET_DIR, filename)
        
        if filename in existing_files:
            skipped_count += 1
            if skipped_count % 100 == 0:
                print(f"已跳过 {skipped_count} 个文件...")
        else:
            shutil.copy2(source_file, target_file)
            copied_count += 1
            if copied_count % 10 == 0:
                print(f"已复制 {copied_count} 个文件...")
    
    print(f"\n复制完成！复制了 {copied_count} 个文件，跳过了 {skipped_count} 个已存在的文件")
    return copied_count

if __name__ == "__main__":
    copied = copy_depth_files()
    print(f"现在目标目录应该有 {len(glob.glob(os.path.join(TARGET_DIR, '*.png')))} 个文件") 