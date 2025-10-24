#!/usr/bin/env python3
"""
原地重命名验证集文件

功能：直接在原验证集目录中重命名文件，不复制

原文件名：24mm__B1__cam_dx+0.00_dy+0.50_yaw000.png
新文件名：s1_cam_dx+0.00_dy+0.50_yaw000.png

GT文件名：24mm__B1__cam_dx+0.00_dy+0.50_yaw000__GT.png
新文件名：s1_cam_dx+0.00_dy+0.50_yaw000.png (去除__GT后缀)
"""

import os
import sys
import re
from typing import Tuple

# 场景映射
SCENE_MAPPING = {
    '24mm': '1',
    '2': '2',
    '3': '3',
    '85mm': '4',
}


def parse_and_normalize(filename: str) -> str:
    """
    解析并规范化验证集文件名
    
    24mm__B1__cam_dx+0.00_dy+0.50_yaw000.png -> s1_cam_dx+0.00_dy+0.50_yaw000.png
    24mm__B1__cam_dx 0.00_dy 2.50_yaw135__GT.png -> s1_cam_dx+0.00_dy+2.50_yaw135.png
    """
    basename = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1]
    
    # 去除__GT后缀
    basename = re.sub(r'__GT$', '', basename)
    
    # 解析：{scene}__{degradation}__{core}
    parts = basename.split('__')
    
    if len(parts) >= 3:
        scene_prefix = parts[0]
        degradation = parts[1]
        core_basename = '__'.join(parts[2:])
        
        # 规范化core_basename（修复空格）
        core_basename = re.sub(r'cam_dx\s+(\d)', r'cam_dx+\1', core_basename)
        core_basename = re.sub(r'_dy\s+(\d)', r'_dy+\1', core_basename)
        core_basename = re.sub(r'cam_dx\s+\-', r'cam_dx-', core_basename)
        core_basename = re.sub(r'_dy\s+\-', r'_dy-', core_basename)
        
        # 添加场景前缀
        if scene_prefix in SCENE_MAPPING:
            scene_id = SCENE_MAPPING[scene_prefix]
            return f"s{scene_id}_{core_basename}{ext}"
    
    # 无法解析，返回原文件名
    return filename


def rename_folder_inplace(folder_path: str, dry_run: bool = False):
    """原地重命名文件夹中的所有PNG文件"""
    
    if not os.path.exists(folder_path):
        print(f"⚠️  目录不存在: {folder_path}")
        return 0, 0
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    print(f"\n处理: {folder_path}")
    print(f"文件数: {len(files)}")
    
    renamed = 0
    skipped = 0
    
    for i, old_name in enumerate(files):
        new_name = parse_and_normalize(old_name)
        
        if old_name == new_name:
            skipped += 1
            continue
        
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)
        
        # 检查目标文件是否存在
        if os.path.exists(new_path):
            print(f"  ⚠️  目标文件已存在，跳过: {new_name}")
            skipped += 1
            continue
        
        if not dry_run:
            os.rename(old_path, new_path)
        
        renamed += 1
        
        # 显示前3个示例
        if renamed <= 3:
            print(f"  {old_name}")
            print(f"  -> {new_name}")
        
        # 进度
        if (i + 1) % 2000 == 0:
            print(f"  进度: {i+1}/{len(files)}")
    
    print(f"✅ 重命名: {renamed}, 跳过: {skipped}")
    
    return renamed, skipped


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='原地重命名验证集文件')
    parser.add_argument('--val-ref-dir', default='DATA/validation/UBB-M_reference',
                       help='有参考验证集目录')
    parser.add_argument('--val-noref-dir', default='DATA/validation/UBB-M_noreference',
                       help='无参考验证集目录')
    parser.add_argument('--dry-run', action='store_true', help='只模拟不实际重命名')
    
    args = parser.parse_args()
    
    print("="*60)
    print("验证集原地重命名工具")
    print("="*60)
    print(f"模式: {'模拟运行' if args.dry_run else '执行重命名'}")
    
    total_renamed = 0
    total_skipped = 0
    
    # 处理有参考验证集
    if os.path.exists(args.val_ref_dir):
        print(f"\n{'='*60}")
        print("有参考验证集")
        print(f"{'='*60}")
        
        # input文件夹
        input_dir = os.path.join(args.val_ref_dir, 'input')
        renamed, skipped = rename_folder_inplace(input_dir, args.dry_run)
        total_renamed += renamed
        total_skipped += skipped
        
        # gt文件夹
        gt_dir = os.path.join(args.val_ref_dir, 'gt')
        renamed, skipped = rename_folder_inplace(gt_dir, args.dry_run)
        total_renamed += renamed
        total_skipped += skipped
    
    # 处理无参考验证集
    if os.path.exists(args.val_noref_dir):
        print(f"\n{'='*60}")
        print("无参考验证集")
        print(f"{'='*60}")
        
        # input文件夹
        input_dir = os.path.join(args.val_noref_dir, 'input')
        renamed, skipped = rename_folder_inplace(input_dir, args.dry_run)
        total_renamed += renamed
        total_skipped += skipped
    
    # 总结
    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")
    print(f"重命名: {total_renamed} 个文件")
    print(f"跳过: {total_skipped} 个文件")
    
    if not args.dry_run:
        print(f"\n✅ 完成！验证集文件已原地重命名")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())




