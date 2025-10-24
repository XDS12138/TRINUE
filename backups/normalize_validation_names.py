#!/usr/bin/env python3
"""
原地规范化验证集文件名

规范化规则:
1. 场景映射: 24mm -> s1, 2 -> s2, 3 -> s3, 85mm -> s4
2. 保留退化前缀: B1, B2, B3, ... (重要！作为唯一标识)
3. 修复命名错误: cam_dx 0.00 -> cam_dx+0.00 (空格问题)
4. 去除__GT后缀

示例:
  原文件名: 24mm__B1__cam_dx+0.00_dy+0.50_yaw000.png
  新文件名: s1__B1__cam_dx+0.00_dy+0.50_yaw000.png

  原GT文件: 24mm__B1__cam_dx 0.00_dy 2.50_yaw135__GT.png
  新GT文件: s1__B1__cam_dx+0.00_dy+2.50_yaw135.png
"""

import os
import sys
import re

# 场景映射
SCENE_MAPPING = {
    '24mm': 's1',
    '2': 's2',
    '3': 's3',
    '85mm': 's4',
}


def normalize_validation_filename(filename: str) -> str:
    """
    规范化验证集文件名（保留退化前缀）
    
    Args:
        filename: 原文件名
    
    Returns:
        规范化后的文件名
    """
    basename = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1]
    
    # 去除__GT后缀
    basename = re.sub(r'__GT$', '', basename)
    
    # 解析格式: {scene}__{degradation}__{core}
    parts = basename.split('__')
    
    if len(parts) >= 3:
        scene_prefix = parts[0]
        degradation = parts[1]  # 保留！
        core_basename = '__'.join(parts[2:])
        
        # 规范化core_basename（修复空格问题）
        core_basename = re.sub(r'cam_dx\s+(\d)', r'cam_dx+\1', core_basename)
        core_basename = re.sub(r'_dy\s+(\d)', r'_dy+\1', core_basename)
        core_basename = re.sub(r'cam_dx\s+\-', r'cam_dx-', core_basename)
        core_basename = re.sub(r'_dy\s+\-', r'_dy-', core_basename)
        
        # 映射场景前缀
        if scene_prefix in SCENE_MAPPING:
            new_scene = SCENE_MAPPING[scene_prefix]
            return f"{new_scene}__{degradation}__{core_basename}{ext}"
    
    # 无法解析，返回原文件名
    return filename


def rename_files_in_folder(folder_path: str, dry_run: bool = False):
    """原地重命名文件夹中的所有PNG文件"""
    
    if not os.path.exists(folder_path):
        print(f"⚠️  目录不存在: {folder_path}")
        return 0, 0, 0
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    print(f"\n处理: {folder_path}")
    print(f"文件数: {len(files)}")
    
    renamed = 0
    skipped_same = 0
    skipped_exists = 0
    
    for i, old_name in enumerate(files):
        new_name = normalize_validation_filename(old_name)
        
        if old_name == new_name:
            skipped_same += 1
            continue
        
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)
        
        # 检查目标文件是否存在
        if os.path.exists(new_path):
            skipped_exists += 1
            if skipped_exists <= 3:
                print(f"  ⚠️  目标已存在: {new_name}")
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
    
    print(f"✅ 重命名: {renamed}, 跳过(相同): {skipped_same}, 跳过(已存在): {skipped_exists}")
    
    return renamed, skipped_same, skipped_exists


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='原地规范化验证集文件名（保留退化前缀）')
    parser.add_argument('--val-ref-dir', default='DATA/validation/UBB-M_reference',
                       help='有参考验证集目录')
    parser.add_argument('--val-noref-dir', default='DATA/validation/UBB-M_noreference',
                       help='无参考验证集目录')
    parser.add_argument('--dry-run', action='store_true', help='只模拟不实际重命名')
    
    args = parser.parse_args()
    
    print("="*60)
    print("验证集原地规范化工具（保留退化前缀）")
    print("="*60)
    print(f"模式: {'模拟运行' if args.dry_run else '执行重命名'}")
    print("\n规范化规则:")
    print("  - 场景映射: 24mm→s1, 2→s2, 3→s3, 85mm→s4")
    print("  - 保留退化前缀: B1, B2, B3, GB1, ... (作为唯一标识)")
    print("  - 修复空格: cam_dx 0.00 → cam_dx+0.00")
    print("  - 去除__GT后缀")
    
    total_renamed = 0
    total_skipped_same = 0
    total_skipped_exists = 0
    
    # 处理有参考验证集
    if os.path.exists(args.val_ref_dir):
        print(f"\n{'='*60}")
        print("有参考验证集")
        print(f"{'='*60}")
        
        # input文件夹
        input_dir = os.path.join(args.val_ref_dir, 'input')
        r, s_same, s_exists = rename_files_in_folder(input_dir, args.dry_run)
        total_renamed += r
        total_skipped_same += s_same
        total_skipped_exists += s_exists
        
        # gt文件夹
        gt_dir = os.path.join(args.val_ref_dir, 'gt')
        r, s_same, s_exists = rename_files_in_folder(gt_dir, args.dry_run)
        total_renamed += r
        total_skipped_same += s_same
        total_skipped_exists += s_exists
    
    # 处理无参考验证集
    if os.path.exists(args.val_noref_dir):
        print(f"\n{'='*60}")
        print("无参考验证集")
        print(f"{'='*60}")
        
        # input文件夹
        input_dir = os.path.join(args.val_noref_dir, 'input')
        r, s_same, s_exists = rename_files_in_folder(input_dir, args.dry_run)
        total_renamed += r
        total_skipped_same += s_same
        total_skipped_exists += s_exists
    
    # 总结
    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")
    print(f"重命名: {total_renamed} 个文件")
    print(f"跳过(文件名相同): {total_skipped_same} 个文件")
    print(f"跳过(目标已存在): {total_skipped_exists} 个文件")
    
    if not args.dry_run:
        print(f"\n✅ 完成！验证集文件已原地规范化")
        print(f"\n示例文件名:")
        print(f"  s1__B1__cam_dx+0.00_dy+0.50_yaw000.png")
        print(f"  s2__GB2__cam_dx+1.25_dy+3.00_yaw180.png")
        print(f"  s4__YG3__cam_dx-2.50_dy+4.75_yaw090.png")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())




