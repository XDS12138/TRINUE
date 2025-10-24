#!/usr/bin/env python3
"""
准备深度验证集

功能:
1. 从重命名后的验证集复制input文件（RGB图像）
2. 从训练集的depth文件夹中提取对应的depth GT
3. 生成深度预测验证集

结构:
  F:/DATASATES/UBB_depth_validation/
    ├── rgb/     # 从验证集input复制
    └── depth/   # 从训练集depth提取

要求:
- 先运行 prepare_validation_sets.py 重命名验证集
- 训练集 UBB_train 已经整理完成
"""

import os
import sys
import shutil
import re
from typing import Set, Dict

def extract_basename(filename: str) -> str:
    """从文件名中提取basename（去除扩展名和场景前缀）"""
    basename = os.path.splitext(filename)[0]
    # 去除场景前缀 s1_, s2_, s3_, s4_
    basename = re.sub(r'^s[1-4]_', '', basename)
    return basename


def prepare_depth_validation(val_input_dir: str, train_depth_dir: str, 
                             target_dir: str, dry_run: bool = False) -> Dict:
    """
    准备深度验证集
    
    Args:
        val_input_dir: 已重命名的验证集input目录
        train_depth_dir: 训练集的depth目录
        target_dir: 目标深度验证集目录
        dry_run: 是否只模拟
    
    Returns:
        统计信息
    """
    print(f"\n{'='*60}")
    print("准备深度验证集")
    print(f"{'='*60}")
    print(f"验证集input: {val_input_dir}")
    print(f"训练集depth: {train_depth_dir}")
    print(f"目标目录: {target_dir}")
    print(f"模式: {'模拟运行' if dry_run else '执行复制'}")
    
    stats = {
        'input_files_copied': 0,
        'depth_files_copied': 0,
        'depth_files_not_found': 0,
        'total_samples': 0,
    }
    
    if not os.path.exists(val_input_dir):
        print(f"⚠️  验证集input目录不存在: {val_input_dir}")
        return stats
    
    if not os.path.exists(train_depth_dir):
        print(f"⚠️  训练集depth目录不存在: {train_depth_dir}")
        return stats
    
    # 创建目标目录
    rgb_target = os.path.join(target_dir, 'rgb')
    depth_target = os.path.join(target_dir, 'depth')
    
    if not dry_run:
        os.makedirs(rgb_target, exist_ok=True)
        os.makedirs(depth_target, exist_ok=True)
    
    # 获取所有input文件
    input_files = sorted([f for f in os.listdir(val_input_dir) if f.endswith('.png')])
    print(f"\n找到 {len(input_files)} 个input文件")
    
    # 获取训练集depth文件的映射 (basename -> filename)
    train_depth_files = {extract_basename(f): f for f in os.listdir(train_depth_dir) if f.endswith('.png')}
    print(f"训练集depth文件: {len(train_depth_files)}")
    
    not_found_list = []
    
    for i, input_filename in enumerate(input_files):
        # 1. 复制input到rgb/
        input_src = os.path.join(val_input_dir, input_filename)
        rgb_dst = os.path.join(rgb_target, input_filename)
        
        if not dry_run:
            shutil.copy2(input_src, rgb_dst)
        stats['input_files_copied'] += 1
        
        # 2. 查找对应的depth文件
        input_basename = extract_basename(input_filename)
        
        # 在训练集中查找匹配的depth文件
        # 训练集文件名格式: s1_cam_dx+0.00_dy+0.50_yaw000.png
        # 验证集文件名格式: s1_cam_dx+0.00_dy+0.50_yaw000.png (相同)
        # 但我们需要匹配去除场景前缀后的basename
        
        depth_filename = None
        # 尝试直接匹配
        if input_filename in train_depth_files.values():
            depth_filename = input_filename
        # 尝试匹配无前缀的basename
        elif input_basename in train_depth_files:
            depth_filename = train_depth_files[input_basename]
        else:
            # 尝试匹配带场景前缀的
            for scene_id in ['1', '2', '3', '4']:
                candidate = f"s{scene_id}_{input_basename}"
                if candidate in [os.path.splitext(f)[0] for f in train_depth_files.values()]:
                    depth_filename = candidate + '.png'
                    break
        
        if depth_filename and os.path.exists(os.path.join(train_depth_dir, depth_filename)):
            # 复制depth文件
            depth_src = os.path.join(train_depth_dir, depth_filename)
            depth_dst = os.path.join(depth_target, input_filename)  # 使用相同的文件名
            
            if not dry_run:
                shutil.copy2(depth_src, depth_dst)
            stats['depth_files_copied'] += 1
        else:
            stats['depth_files_not_found'] += 1
            not_found_list.append(input_filename)
            if len(not_found_list) <= 5:
                print(f"  ⚠️  未找到depth: {input_filename}")
        
        stats['total_samples'] += 1
        
        # 进度显示
        if (i + 1) % 1000 == 0:
            print(f"  进度: {i+1}/{len(input_files)}")
    
    # 总结
    print(f"\n{'='*60}")
    print("处理总结")
    print(f"{'='*60}")
    print(f"Input文件复制: {stats['input_files_copied']}")
    print(f"Depth文件复制: {stats['depth_files_copied']}")
    print(f"Depth未找到: {stats['depth_files_not_found']}")
    print(f"配对成功率: {stats['depth_files_copied']/stats['input_files_copied']*100:.2f}%")
    
    if not_found_list:
        print(f"\n未找到depth的文件示例 (共{len(not_found_list)}个):")
        for fn in not_found_list[:10]:
            print(f"  - {fn}")
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='准备深度验证集')
    parser.add_argument('--val-input', default='F:/DATASATES/UBB_validation_reference/input',
                       help='已重命名的验证集input目录')
    parser.add_argument('--train-depth', default='F:/DATASATES/UBB_train/depth',
                       help='训练集depth目录')
    parser.add_argument('--target', default='F:/DATASATES/UBB_depth_validation',
                       help='深度验证集目标目录')
    parser.add_argument('--dry-run', action='store_true', help='只模拟不实际复制')
    parser.add_argument('--use-noref-input', action='store_true', 
                       help='使用无参考验证集的input（默认使用有参考）')
    
    args = parser.parse_args()
    
    # 如果使用无参考验证集的input
    if args.use_noref_input:
        args.val_input = 'F:/DATASATES/UBB_validation_noreference/input'
    
    print("="*60)
    print("深度验证集准备工具")
    print("="*60)
    
    # 检查目录是否存在
    if not os.path.exists(args.val_input):
        print(f"⚠️  验证集input目录不存在: {args.val_input}")
        print("\n请先运行:")
        print("  python scripts/prepare_validation_sets.py")
        return 1
    
    if not os.path.exists(args.train_depth):
        print(f"⚠️  训练集depth目录不存在: {args.train_depth}")
        print("\n请等待训练集整理完成:")
        print("  python scripts/prepare_ubb_dataset.py --source DATA/UBB --target F:/DATASATES/UBB_train")
        return 1
    
    # 准备深度验证集
    stats = prepare_depth_validation(
        val_input_dir=args.val_input,
        train_depth_dir=args.train_depth,
        target_dir=args.target,
        dry_run=args.dry_run
    )
    
    if not args.dry_run and stats['depth_files_copied'] > 0:
        print(f"\n✅ 深度验证集准备完成！")
        print(f"\n目录结构:")
        print(f"  {args.target}/")
        print(f"    ├── rgb/    ({stats['input_files_copied']} 文件)")
        print(f"    └── depth/  ({stats['depth_files_copied']} 文件)")
        
        print(f"\n可以在configs/train.yaml中配置:")
        print(f"  validation_sets:")
        print(f"    ubb_depth:")
        print(f"      name: 'UBB_Depth'")
        print(f"      type: 'depth_prediction'")
        print(f"      data_root: '{args.target}'")
        print(f"      folder_structure:")
        print(f"        rgb: 'rgb'")
        print(f"        depth: 'depth'")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())




