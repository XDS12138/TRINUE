#!/usr/bin/env python3
"""
按退化类型均衡抽取验证集

策略: 从每种退化中随机抽取N个样本，保持15种退化的平衡

示例: 每种退化抽取150个
  → 验证集: 150 × 15 = 2,250个样本
  → 训练集剩余: ~11,000样本 (83%)
"""

import os
import sys
import re
import random
import json
from collections import defaultdict

def parse_validation_filename(filename: str):
    """解析验证集文件名"""
    basename = os.path.splitext(filename)[0]
    parts = basename.split('__')
    
    if len(parts) >= 3:
        scene = parts[0]
        degradation = parts[1]
        core = '__'.join(parts[2:])
        core = re.sub(r'_(rgb|mist|depth|normal)(_vis)?(_\d+)?$', '', core)
        return scene, degradation, core
    return None, None, None


def subsample_by_degradation(input_dir: str, gt_dir: str, samples_per_deg: int, 
                             output_json: str, seed: int = 42):
    """按退化类型均衡抽取"""
    
    print("="*60)
    print("按退化类型均衡抽取验证集")
    print("="*60)
    
    if not os.path.exists(input_dir):
        print("输入目录不存在")
        return
    
    # 按退化类型分组
    deg_to_files = defaultdict(list)
    
    files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    print(f"\n当前验证集: {len(files)} 个文件")
    
    for f in files:
        scene, deg, core = parse_validation_filename(f)
        if deg:
            deg_to_files[deg].append(f)
    
    print(f"\n退化类型分布:")
    for deg in sorted(deg_to_files.keys()):
        print(f"  {deg}: {len(deg_to_files[deg])} 个")
    
    # 从每种退化中随机抽取
    random.seed(seed)
    selected_input = []
    selected_gt = []
    
    for deg, file_list in sorted(deg_to_files.items()):
        if samples_per_deg >= len(file_list):
            selected = file_list
        else:
            selected = random.sample(file_list, samples_per_deg)
        
        selected_input.extend(selected)
    
    # GT文件
    if gt_dir and os.path.exists(gt_dir):
        gt_files_all = {os.path.splitext(f)[0]: f for f in os.listdir(gt_dir) if f.endswith('.png')}
        
        for input_file in selected_input:
            input_bn = os.path.splitext(input_file)[0]
            if input_bn in gt_files_all:
                selected_gt.append(gt_files_all[input_bn])
    
    print(f"\n精简结果:")
    print(f"  每种退化: {samples_per_deg} 个")
    print(f"  Input文件: {len(selected_input)}/{len(files)} ({len(selected_input)/len(files)*100:.1f}%)")
    print(f"  GT文件: {len(selected_gt)}")
    
    # 保存
    output_data = {
        'samples_per_degradation': samples_per_deg,
        'num_input_files': len(selected_input),
        'num_gt_files': len(selected_gt),
        'input_files': sorted(selected_input),
        'gt_files': sorted(selected_gt),
        'seed': seed,
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n[OK] 文件列表已保存: {output_json}")
    
    return output_data


def estimate_training_impact(train_root: str, selected_files: list):
    """评估对训练集的影响"""
    
    print(f"\n{'='*60}")
    print("训练集影响评估")
    print(f"{'='*60}")
    
    # 从文件名提取basename
    val_basenames = set()
    for f in selected_files:
        scene, deg, core = parse_validation_filename(f)
        if core:
            val_basenames.add(core)
    
    print(f"验证集唯一位置: {len(val_basenames)}")
    
    # 检查训练集匹配
    gt_dir = os.path.join(train_root, 'gt')
    if os.path.exists(gt_dir):
        train_files = [f for f in os.listdir(gt_dir) if f.endswith('.png')]
        
        matched = 0
        for train_file in train_files:
            train_core = re.sub(r'^s[1-4]_', '', os.path.splitext(train_file)[0])
            if train_core in val_basenames:
                matched += 1
        
        remaining = len(train_files) - matched
        
        print(f"\n训练集GT:")
        print(f"  原始: {len(train_files)} 样本")
        print(f"  删除: {matched} 样本")
        print(f"  剩余: {remaining} 样本")
        print(f"  保留率: {remaining/len(train_files)*100:.1f}%")
        
        return remaining
    
    return 0


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='DATA/validation/UBB-M_reference/input')
    parser.add_argument('--gt-dir', default='DATA/validation/UBB-M_reference/gt')
    parser.add_argument('--samples-per-deg', type=int, default=150,
                       help='每种退化保留的样本数（默认150）')
    parser.add_argument('--output-json', default='F:/DATASATES/validation_subset.json')
    parser.add_argument('--train-root', default='F:/DATASATES/UBB_train')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    result = subsample_by_degradation(
        args.input_dir, args.gt_dir, args.samples_per_deg, 
        args.output_json, args.seed
    )
    
    remaining = estimate_training_impact(args.train_root, result['input_files'])
    
    print(f"\n{'='*60}")
    if remaining >= 10000:
        print(f"[OK] 训练集剩余{remaining}样本充足，可以继续")
    elif remaining >= 5000:
        print(f"[WARNING] 训练集剩余{remaining}样本偏少，建议减少验证集")
    else:
        print(f"[ERROR] 训练集剩余{remaining}样本太少！")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())




