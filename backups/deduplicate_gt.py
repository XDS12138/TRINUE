#!/usr/bin/env python3
"""
去重GT文件

验证集GT中有重复文件（不同退化对应同一个GT）
保留第一个，删除其余
"""

import os
import sys
from collections import defaultdict

def deduplicate_gt(gt_dir: str, dry_run: bool = False):
    """去重GT文件夹"""
    
    print(f"GT目录: {gt_dir}")
    
    if not os.path.exists(gt_dir):
        print("⚠️  目录不存在")
        return
    
    files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.png')])
    print(f"原始文件数: {len(files)}")
    
    # 按basename分组
    basename_to_files = defaultdict(list)
    for f in files:
        basename = os.path.splitext(f)[0]
        basename_to_files[basename].append(f)
    
    print(f"唯一basename: {len(basename_to_files)}")
    
    # 找出重复的
    duplicates = {k: v for k, v in basename_to_files.items() if len(v) > 1}
    
    if duplicates:
        print(f"⚠️  发现 {len(duplicates)} 个basename有重复文件")
        print(f"  示例: {list(duplicates.keys())[:3]}")
    
    # 删除重复文件（保留第一个）
    deleted = 0
    for basename, file_list in basename_to_files.items():
        if len(file_list) > 1:
            # 保留第一个，删除其余
            for f in file_list[1:]:
                file_path = os.path.join(gt_dir, f)
                if not dry_run:
                    os.remove(file_path)
                deleted += 1
                if deleted <= 5:
                    print(f"  删除: {f}")
    
    print(f"\n{'模拟' if dry_run else '实际'}删除: {deleted} 个重复文件")
    print(f"剩余: {len(files) - deleted} 个唯一GT")
    
    return len(files) - deleted

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='去重GT文件')
    parser.add_argument('--gt-dir', default='F:/DATASATES/UBB_validation_reference/gt',
                       help='GT目录')
    parser.add_argument('--dry-run', action='store_true', help='只模拟不删除')
    
    args = parser.parse_args()
    
    print("="*60)
    print("GT去重工具")
    print("="*60)
    
    result = deduplicate_gt(args.gt_dir, args.dry_run)
    
    if not args.dry_run and result:
        print(f"\n✅ 完成！GT已去重")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())




