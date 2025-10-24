#!/usr/bin/env python3
"""
将多输入训练集转换为单输入格式

用途: 用于其他单输入任务的训练

转换逻辑:
  多输入格式（当前）:
    color_B_1/ - s1_cam_dx+0.00_dy+0.50_yaw000.png
    color_B_2/ - s1_cam_dx+0.00_dy+0.50_yaw000.png
    gt/        - s1_cam_dx+0.00_dy+0.50_yaw000.png (1个)
  
  单输入格式（目标）:
    input/ - s1__B1__cam_dx+0.00_dy+0.50_yaw000.png (从color_B_1)
           - s1__B2__cam_dx+0.00_dy+0.50_yaw000.png (从color_B_2)
    gt/    - s1__B1__cam_dx+0.00_dy+0.50_yaw000.png (GT复制，添加B1前缀)
           - s1__B2__cam_dx+0.00_dy+0.50_yaw000.png (同一GT，添加B2前缀)
"""

import os
import sys
import shutil
from tqdm import tqdm

# 退化文件夹到前缀的映射
DEGRADATION_MAPPING = {
    'color_B_1': 'B1',
    'color_B_2': 'B2',
    'color_B_3': 'B3',
    'color_BG_1': 'GB1',
    'color_BG_2': 'GB2',
    'color_BG_3': 'GB3',
    'color_G_1': 'G1',
    'color_G_2': 'G2',
    'color_G_3': 'G3',
    'color_Y_1': 'Y1',
    'color_Y_2': 'Y2',
    'color_Y_3': 'Y3',
    'color_YG_1': 'YG1',
    'color_YG_2': 'YG2',
    'color_YG_3': 'YG3',
}


def convert_to_single_input(source_root, target_root, dry_run=False):
    """
    转换多输入训练集为单输入格式
    
    Args:
        source_root: 源目录（多输入格式）
        target_root: 目标目录（单输入格式）
        dry_run: 是否只模拟
    """
    
    print("="*60)
    print("多输入 → 单输入格式转换")
    print("="*60)
    print(f"源目录: {source_root}")
    print(f"目标目录: {target_root}")
    print(f"模式: {'模拟' if dry_run else '执行'}")
    
    # 检查源目录
    gt_source = os.path.join(source_root, 'gt')
    if not os.path.exists(gt_source):
        print(f"错误: GT文件夹不存在 - {gt_source}")
        return
    
    # 获取GT文件列表（作为样本列表）
    gt_files = sorted([f for f in os.listdir(gt_source) if f.endswith('.png')])
    basenames = [os.path.splitext(f)[0] for f in gt_files]
    
    print(f"\n找到 {len(basenames)} 个样本")
    print(f"将转换为 {len(basenames) * 15} 对（input-gt）")
    
    # 创建目标目录
    input_target = os.path.join(target_root, 'input')
    gt_target = os.path.join(target_root, 'gt')
    
    if not dry_run:
        os.makedirs(input_target, exist_ok=True)
        os.makedirs(gt_target, exist_ok=True)
    
    stats = {
        'input_copied': 0,
        'gt_copied': 0,
        'errors': 0,
    }
    
    # 处理每个样本
    for basename in tqdm(basenames, desc="转换样本"):
        # GT文件路径
        gt_src = os.path.join(gt_source, f"{basename}.png")
        
        if not os.path.exists(gt_src):
            stats['errors'] += 1
            continue
        
        # 对每种退化
        for deg_folder, deg_prefix in DEGRADATION_MAPPING.items():
            try:
                # 1. 复制input（退化图）
                input_src = os.path.join(source_root, deg_folder, f"{basename}.png")
                
                if not os.path.exists(input_src):
                    # 某些退化可能缺失，跳过
                    continue
                
                # 添加退化前缀到文件名
                # s1_cam_dx+0.00_dy+0.50_yaw000.png → s1__B1__cam_dx+0.00_dy+0.50_yaw000.png
                parts = basename.split('_', 1)  # 分割场景前缀
                if len(parts) == 2:
                    scene = parts[0]  # s1, s2, s3, s4
                    core = parts[1]   # cam_dx+0.00_dy+0.50_yaw000
                    new_filename = f"{scene}__{deg_prefix}__{core}.png"
                else:
                    # 没有场景前缀的情况
                    new_filename = f"{deg_prefix}__{basename}.png"
                
                input_dst = os.path.join(input_target, new_filename)
                
                if not dry_run:
                    shutil.copy2(input_src, input_dst)
                stats['input_copied'] += 1
                
                # 2. 复制GT（同一个GT复制15份，添加不同退化前缀）
                gt_dst = os.path.join(gt_target, new_filename)
                
                if not dry_run:
                    shutil.copy2(gt_src, gt_dst)
                stats['gt_copied'] += 1
                
            except Exception as e:
                stats['errors'] += 1
                if stats['errors'] <= 5:
                    print(f"\n错误: {basename} - {deg_prefix} - {e}")
    
    # 总结
    print(f"\n{'='*60}")
    print("转换完成")
    print(f"{'='*60}")
    print(f"Input复制: {stats['input_copied']}")
    print(f"GT复制: {stats['gt_copied']}")
    print(f"错误: {stats['errors']}")
    
    if not dry_run:
        print(f"\n输出目录:")
        print(f"  {target_root}/")
        print(f"    ├── input/ ({stats['input_copied']} 文件)")
        print(f"    └── gt/    ({stats['gt_copied']} 文件)")
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='转换为单输入格式训练集')
    parser.add_argument('--source', default='F:/DATASATES/UBB_train',
                       help='源目录（多输入格式）')
    parser.add_argument('--target', default='D:/UBB_train_single_input',
                       help='目标目录（单输入格式）')
    parser.add_argument('--dry-run', action='store_true', help='只模拟不实际复制')
    
    args = parser.parse_args()
    
    # 检查源目录
    if not os.path.exists(args.source):
        print(f"错误: 源目录不存在 - {args.source}")
        return 1
    
    # 执行转换
    convert_to_single_input(args.source, args.target, args.dry_run)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())




