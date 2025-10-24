#!/usr/bin/env python3
"""
替换修复后的损坏文件到所有数据集

损坏文件:
1. s3_cam_dx+24.00_dy-18.00_yaw315.png (color_BG_3)
2. s3_cam_dx-24.00_dy-36.00_yaw225.png (color_G_2)
"""

import os
import shutil

# 修复文件源目录
source_dir = r"C:\Users\xds11\Downloads"

# 损坏文件信息
corrupted_files = [
    {
        'filename': 's3_cam_dx+24.00_dy-18.00_yaw315.png',
        'basename': 's3_cam_dx+24.00_dy-18.00_yaw315',
        'scene': 's3',
        'core': 'cam_dx+24.00_dy-18.00_yaw315',
        'degradation': 'color_BG_3',
        'degradation_prefix': 'GB3',
    },
    {
        'filename': 's3_cam_dx-24.00_dy-36.00_yaw225.png',
        'basename': 's3_cam_dx-24.00_dy-36.00_yaw225',
        'scene': 's3',
        'core': 'cam_dx-24.00_dy-36.00_yaw225',
        'degradation': 'color_G_2',
        'degradation_prefix': 'G2',
    }
]

all_degradations = ['B1', 'B2', 'B3', 'GB1', 'GB2', 'GB3', 
                   'G1', 'G2', 'G3', 'Y1', 'Y2', 'Y3', 'YG1', 'YG2', 'YG3']

print("="*60)
print("替换修复后的损坏文件")
print("="*60)

total_replaced = 0

for file_info in corrupted_files:
    source_file = os.path.join(source_dir, file_info['filename'])
    
    if not os.path.exists(source_file):
        print(f"\n⚠️  源文件不存在: {source_file}")
        continue
    
    print(f"\n处理: {file_info['filename']}")
    
    # 1. 替换D:/UBB_train中对应退化文件夹的文件
    target1 = os.path.join('D:/UBB_train', file_info['degradation'], file_info['filename'])
    if os.path.exists(os.path.dirname(target1)):
        shutil.copy2(source_file, target1)
        print(f"  ✅ 替换: D:/UBB_train/{file_info['degradation']}/")
        total_replaced += 1
    
    # 2. 替换D:/UBB_train_single_input中所有退化版本
    # 单输入格式: s3__GB3__cam_dx+24.00_dy-18.00_yaw315.png
    for deg in all_degradations:
        single_filename = f"{file_info['scene']}__{deg}__{file_info['core']}.png"
        
        # 替换input
        target_input = os.path.join('D:/UBB_train_single_input/input', single_filename)
        if os.path.exists(os.path.dirname(target_input)):
            # 只有当前退化匹配时才替换input
            if deg == file_info['degradation_prefix']:
                shutil.copy2(source_file, target_input)
                print(f"  ✅ 替换: D:/UBB_train_single_input/input/{single_filename}")
                total_replaced += 1
        
        # 替换gt（所有退化都使用同一个图像）
        # 注意：单输入格式的gt应该是干净的GT图，不是退化图
        # 这里不需要替换GT，因为损坏的是退化图
    
    # 3. 替换F:/DATASATES/UBBraw中scene_3对应文件夹的文件
    # UBBraw格式: scene_3/color_BG_3/cam_dx+24.00_dy-18.00_yaw315.png (无场景前缀)
    ubraw_folder = f"F:/DATASATES/UBBraw/scene_3/{file_info['degradation']}"
    ubraw_filename = f"{file_info['core']}.png"  # 去除场景前缀
    target3 = os.path.join(ubraw_folder, ubraw_filename)
    
    if os.path.exists(os.path.dirname(target3)):
        shutil.copy2(source_file, target3)
        print(f"  ✅ 替换: F:/DATASATES/UBBraw/scene_3/{file_info['degradation']}/")
        total_replaced += 1

print(f"\n{'='*60}")
print(f"替换完成: 共 {total_replaced} 个文件")
print(f"{'='*60}")
print("\n下一步:")
print("  1. 重新检查: python scripts/check_all_images_parallel.py --threads 64")
print("  2. 确认无损坏后执行LMDB转换")




