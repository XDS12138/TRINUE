#!/usr/bin/env python3
"""
准备UBB验证集（最终正确版本）

关键点:
1. 保留退化前缀 (B1, B2, ..., YG3) - 作为唯一标识
2. 场景映射: 24mm→s1, 2→s2, 3→s3, 85mm→s4
3. GT也需要退化前缀，虽然图像相同
4. 原地重命名，不复制到新位置

文件名格式:
  Input: s1__B1__cam_dx+0.00_dy+0.50_yaw000.png
  GT:    s1__B1__cam_dx+0.00_dy+0.50_yaw000.png (去除__GT后缀)
"""

import os
import sys
import re

SCENE_MAPPING = {
    '24mm': 's1',
    '2': 's2',
    '3': 's3',
    '85mm': 's4',
}

def normalize_filename(filename: str) -> str:
    """规范化文件名（保留退化前缀）"""
    basename = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1]
    
    # 去除__GT后缀
    basename = re.sub(r'__GT$', '', basename)
    
    # 解析: {scene}__{degradation}__{core}
    parts = basename.split('__')
    
    if len(parts) >= 3:
        scene = parts[0]
        degradation = parts[1]
        core = '__'.join(parts[2:])
        
        # 修复空格
        core = re.sub(r'cam_dx\s+(\d)', r'cam_dx+\1', core)
        core = re.sub(r'_dy\s+(\d)', r'_dy+\1', core)
        core = re.sub(r'cam_dx\s+\-', r'cam_dx-', core)
        core = re.sub(r'_dy\s+\-', r'_dy-', core)
        
        # 场景映射
        if scene in SCENE_MAPPING:
            scene = SCENE_MAPPING[scene]
        
        return f"{scene}__{degradation}__{core}{ext}"
    
    return filename

def rename_folder_inplace(folder_path: str, dry_run: bool = False):
    """原地重命名"""
    if not os.path.exists(folder_path):
        return 0, 0
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    renamed = 0
    skipped = 0
    
    print(f"\n{folder_path}: {len(files)} 文件")
    
    for i, old_name in enumerate(files):
        new_name = normalize_filename(old_name)
        
        if old_name == new_name:
            skipped += 1
            continue
        
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)
        
        if os.path.exists(new_path):
            skipped += 1
            continue
        
        if not dry_run:
            os.rename(old_path, new_path)
        renamed += 1
        
        if renamed <= 3:
            print(f"  {old_name} -> {new_name}")
        
        if (i+1) % 2000 == 0:
            print(f"  进度: {i+1}/{len(files)}")
    
    print(f"  重命名: {renamed}, 跳过: {skipped}")
    return renamed, skipped

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref-dir', default='DATA/validation/UBB-M_reference')
    parser.add_argument('--noref-dir', default='DATA/validation/UBB-M_noreference')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    print("="*60)
    print("验证集文件名规范化（保留退化前缀）")
    print("="*60)
    print(f"模式: {'模拟' if args.dry_run else '执行'}")
    
    total_renamed = 0
    
    # 有参考
    if os.path.exists(args.ref_dir):
        print(f"\n有参考验证集:")
        r, s = rename_folder_inplace(os.path.join(args.ref_dir, 'input'), args.dry_run)
        total_renamed += r
        r, s = rename_folder_inplace(os.path.join(args.ref_dir, 'gt'), args.dry_run)
        total_renamed += r
    
    # 无参考
    if os.path.exists(args.noref_dir):
        print(f"\n无参考验证集:")
        r, s = rename_folder_inplace(os.path.join(args.noref_dir, 'input'), args.dry_run)
        total_renamed += r
    
    print(f"\n总计重命名: {total_renamed}")
    return 0

if __name__ == '__main__':
    sys.exit(main())




