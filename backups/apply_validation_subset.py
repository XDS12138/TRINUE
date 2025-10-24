#!/usr/bin/env python3
"""
应用验证集精简

根据subsample_by_degradation.py生成的JSON文件，
移动未选中的验证文件到备份目录
"""

import os
import sys
import json
import shutil

def apply_subset(input_dir: str, gt_dir: str, subset_json: str, 
                backup_dir: str, dry_run: bool = False):
    """应用验证集精简"""
    
    print("="*60)
    print("应用验证集精简")
    print("="*60)
    
    # 加载选中文件列表
    with open(subset_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    selected_input = set(data['input_files'])
    selected_gt = set(data['gt_files'])
    
    print(f"选中input: {len(selected_input)}")
    print(f"选中GT: {len(selected_gt)}")
    
    stats = {
        'input_kept': 0,
        'input_moved': 0,
        'gt_kept': 0,
        'gt_moved': 0,
    }
    
    # 处理input文件夹
    if os.path.exists(input_dir):
        print(f"\n处理 input/")
        input_backup = os.path.join(backup_dir, 'input')
        if not dry_run:
            os.makedirs(input_backup, exist_ok=True)
        
        all_input = [f for f in os.listdir(input_dir) if f.endswith('.png')]
        
        for f in all_input:
            if f in selected_input:
                stats['input_kept'] += 1
            else:
                # 移动到备份
                if not dry_run:
                    src = os.path.join(input_dir, f)
                    dst = os.path.join(input_backup, f)
                    shutil.move(src, dst)
                stats['input_moved'] += 1
        
        print(f"  保留: {stats['input_kept']}")
        print(f"  备份: {stats['input_moved']}")
    
    # 处理GT文件夹
    if gt_dir and os.path.exists(gt_dir):
        print(f"\n处理 gt/")
        gt_backup = os.path.join(backup_dir, 'gt')
        if not dry_run:
            os.makedirs(gt_backup, exist_ok=True)
        
        all_gt = [f for f in os.listdir(gt_dir) if f.endswith('.png')]
        
        for f in all_gt:
            if f in selected_gt:
                stats['gt_kept'] += 1
            else:
                # 移动到备份
                if not dry_run:
                    src = os.path.join(gt_dir, f)
                    dst = os.path.join(gt_backup, f)
                    shutil.move(src, dst)
                stats['gt_moved'] += 1
        
        print(f"  保留: {stats['gt_kept']}")
        print(f"  备份: {stats['gt_moved']}")
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='DATA/validation/UBB-M_reference/input')
    parser.add_argument('--gt-dir', default='DATA/validation/UBB-M_reference/gt', nargs='?')
    parser.add_argument('--subset-json', default='F:/DATASATES/validation_subset.json')
    parser.add_argument('--backup-dir', default='F:/DATASATES/UBB_validation_backup')
    parser.add_argument('--dry-run', action='store_true')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.subset_json):
        print(f"[ERROR] 精简列表不存在: {args.subset_json}")
        print("\n请先运行:")
        print("  python scripts/subsample_by_degradation.py --samples-per-deg 150")
        return 1
    
    stats = apply_subset(args.input_dir, args.gt_dir, args.subset_json, 
                        args.backup_dir, args.dry_run)
    
    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")
    print(f"验证集保留: {stats['input_kept']} input, {stats['gt_kept']} GT")
    print(f"备份文件: {stats['input_moved']} input, {stats['gt_moved']} GT")
    
    if not args.dry_run:
        print(f"\n[OK] 完成！未选中文件已备份到:")
        print(f"  {args.backup_dir}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

