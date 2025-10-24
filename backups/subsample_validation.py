#!/usr/bin/env python3
"""
验证集精简工具

从现有验证集中随机抽取一定比例，保持15种退化的平衡

策略：
- 随机选择N个唯一位置（如200个位置）
- 每个位置保留全部15种退化
- 结果：200 × 15 = 3,000个验证样本
"""

import os
import sys
import re
import random
import json
import shutil
from collections import defaultdict
from typing import Set, List, Dict

def parse_validation_filename(filename: str):
    """
    解析验证集文件名
    
    s1__B1__cam_dx+0.00_dy+0.50_yaw000.png
    -> (scene='s1', degradation='B1', core='cam_dx+0.00_dy+0.50_yaw000')
    """
    basename = os.path.splitext(filename)[0]
    parts = basename.split('__')
    
    if len(parts) >= 3:
        scene = parts[0]
        degradation = parts[1]
        core = '__'.join(parts[2:])
        
        # 去除渲染后缀
        core = re.sub(r'_(rgb|mist|depth|normal)(_vis)?(_\d+)?$', '', core)
        
        return scene, degradation, core
    
    return None, None, None


def analyze_validation_set(input_dir: str):
    """分析验证集的位置分布"""
    
    if not os.path.exists(input_dir):
        return {}
    
    files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    
    # 按位置分组 (scene + core)
    location_to_files = defaultdict(list)
    degradation_count = defaultdict(int)
    
    for f in files:
        scene, degradation, core = parse_validation_filename(f)
        if scene and core:
            location_key = f"{scene}_{core}"
            location_to_files[location_key].append(f)
            degradation_count[degradation] += 1
    
    return {
        'total_files': len(files),
        'unique_locations': len(location_to_files),
        'location_to_files': location_to_files,
        'degradation_count': degradation_count,
    }


def subsample_validation(input_dir: str, gt_dir: str, num_locations: int, 
                        output_json: str, seed: int = 42):
    """
    从验证集中随机抽取指定数量的位置
    
    Args:
        input_dir: 验证集input目录
        gt_dir: 验证集GT目录
        num_locations: 要保留的位置数量
        output_json: 输出选中文件列表的JSON
        seed: 随机种子
    """
    
    print("="*60)
    print("验证集精简工具")
    print("="*60)
    
    # 分析验证集
    analysis = analyze_validation_set(input_dir)
    
    total_files = analysis['total_files']
    unique_locations = analysis['unique_locations']
    location_to_files = analysis['location_to_files']
    deg_count = analysis['degradation_count']
    
    print(f"\n当前验证集:")
    print(f"  总文件数: {total_files}")
    print(f"  唯一位置: {unique_locations}")
    print(f"  每位置文件数: {total_files // unique_locations if unique_locations > 0 else 0}")
    
    print(f"\n退化类型分布:")
    for deg, count in sorted(deg_count.items()):
        print(f"  {deg}: {count}")
    
    # 随机抽取位置
    random.seed(seed)
    all_locations = sorted(list(location_to_files.keys()))
    
    if num_locations >= len(all_locations):
        print(f"\n⚠️  请求位置数({num_locations})超过可用位置数({len(all_locations)})")
        selected_locations = all_locations
    else:
        selected_locations = random.sample(all_locations, num_locations)
    
    # 收集选中的文件
    selected_input_files = []
    selected_gt_files = []
    
    for loc in selected_locations:
        selected_input_files.extend(location_to_files[loc])
    
    # GT文件（如果存在）
    if gt_dir and os.path.exists(gt_dir):
        gt_files_all = [f for f in os.listdir(gt_dir) if f.endswith('.png')]
        gt_basename_to_file = {os.path.splitext(f)[0]: f for f in gt_files_all}
        
        for input_file in selected_input_files:
            input_basename = os.path.splitext(input_file)[0]
            if input_basename in gt_basename_to_file:
                selected_gt_files.append(gt_basename_to_file[input_basename])
    
    # 统计
    print(f"\n精简结果:")
    print(f"  选中位置: {len(selected_locations)}/{unique_locations} ({len(selected_locations)/unique_locations*100:.1f}%)")
    print(f"  Input文件: {len(selected_input_files)}/{total_files} ({len(selected_input_files)/total_files*100:.1f}%)")
    print(f"  GT文件: {len(selected_gt_files)}")
    
    # 保存文件列表
    output_data = {
        'num_locations': len(selected_locations),
        'num_input_files': len(selected_input_files),
        'num_gt_files': len(selected_gt_files),
        'selected_locations': sorted(selected_locations),
        'input_files': sorted(selected_input_files),
        'gt_files': sorted(selected_gt_files),
        'seed': seed,
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n[OK] 选中文件列表已保存: {output_json}")
    
    return output_data


def estimate_training_set_reduction(train_root: str, validation_locations: List[str]):
    """
    估算从训练集排除验证集后的剩余样本数
    
    Args:
        train_root: 训练集根目录
        validation_locations: 验证集选中的位置列表 (如 ['s1_cam_dx+...', 's2_cam_dx+...'])
    """
    
    print(f"\n{'='*60}")
    print("训练集影响评估")
    print(f"{'='*60}")
    
    if not os.path.exists(train_root):
        print(f"⚠️  训练集目录不存在")
        return
    
    # 构建验证集basename集合（去除场景前缀）
    val_basenames = set()
    for loc in validation_locations:
        # loc格式: s1_cam_dx+0.00_dy+0.50_yaw000
        # 去除场景前缀
        parts = loc.split('_', 1)
        if len(parts) == 2:
            val_basenames.add(parts[1])  # cam_dx+0.00_dy+0.50_yaw000
    
    print(f"验证集唯一位置: {len(validation_locations)}")
    print(f"验证集basename数: {len(val_basenames)}")
    
    # 检查训练集GT文件夹的匹配情况
    gt_dir = os.path.join(train_root, 'gt')
    if os.path.exists(gt_dir):
        train_files = [f for f in os.listdir(gt_dir) if f.endswith('.png')]
        
        # 训练集文件格式: s1_cam_dx+0.00_dy+0.50_yaw000.png
        matched = 0
        for train_file in train_files:
            train_basename = os.path.splitext(train_file)[0]
            # 去除场景前缀 s1_, s2_, etc.
            train_core = re.sub(r'^s[1-4]_', '', train_basename)
            
            if train_core in val_basenames:
                matched += 1
        
        remaining = len(train_files) - matched
        
        print(f"\n训练集GT文件夹:")
        print(f"  原始文件: {len(train_files)}")
        print(f"  匹配验证集: {matched}")
        print(f"  删除后剩余: {remaining}")
        print(f"  保留比例: {remaining/len(train_files)*100:.1f}%")
        
        return remaining
    
    return 0


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='验证集精简工具')
    parser.add_argument('--input-dir', default='DATA/validation/UBB-M_reference/input',
                       help='验证集input目录')
    parser.add_argument('--gt-dir', default='DATA/validation/UBB-M_reference/gt',
                       help='验证集GT目录')
    parser.add_argument('--num-locations', type=int, default=200,
                       help='要保留的位置数量（默认200）')
    parser.add_argument('--percentage', type=float, default=0,
                       help='要保留的百分比（如果设置，会覆盖num-locations）')
    parser.add_argument('--output-json', default='F:/DATASATES/validation_subset_selected.json',
                       help='输出选中文件列表的JSON')
    parser.add_argument('--train-root', default='F:/DATASATES/UBB_train',
                       help='训练集根目录（用于评估影响）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 如果指定了百分比，计算实际位置数
    if args.percentage > 0:
        analysis = analyze_validation_set(args.input_dir)
        unique_locations = analysis['unique_locations']
        args.num_locations = int(unique_locations * args.percentage)
        print(f"\n按{args.percentage*100}%计算，抽取 {args.num_locations} 个位置")
    
    # 精简验证集
    result = subsample_validation(
        input_dir=args.input_dir,
        gt_dir=args.gt_dir,
        num_locations=args.num_locations,
        output_json=args.output_json,
        seed=args.seed,
    )
    
    # 评估对训练集的影响
    remaining = estimate_training_set_reduction(
        train_root=args.train_root,
        validation_locations=result['selected_locations']
    )
    
    # 总结建议
    print(f"\n{'='*60}")
    print("建议")
    print(f"{'='*60}")
    
    if remaining < 10000:
        print(f"⚠️  训练集剩余{remaining}样本较少，建议：")
        print(f"  - 增加验证集抽取比例（当前{args.num_locations}个位置）")
        print(f"  - 或考虑方案2（按场景划分）")
    else:
        print(f"✓ 训练集剩余{remaining}样本充足")
        print(f"\n下一步执行:")
        print(f"  1. python scripts/apply_validation_subset.py")
        print(f"       （移动未选中的文件到备份目录）")
        print(f"  2. python scripts/exclude_validation_from_train.py")
        print(f"       （从训练集删除验证样本）")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())




