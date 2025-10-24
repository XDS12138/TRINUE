#!/usr/bin/env python3
"""
创建深度验证集

结构:
  DATA/validation/UBB-M_depth/
    ├── input/  (从UBB-M_reference/input复制，20,040个)
    └── depth/  (从训练集depth提取并复制15份添加退化前缀，20,040个)

匹配逻辑:
  Input: s1__B1__cam_dx+0.00_dy+0.50_yaw000.png
         └─ 场景s1 + 退化B1 + 位置cam_dx+0.00_dy+0.50_yaw000
  
  训练集depth: s1_cam_dx+0.00_dy+0.50_yaw000.png
               └─ 场景s1 + 位置（无退化前缀）
  
  输出depth: s1__B1__cam_dx+0.00_dy+0.50_yaw000.png
             └─ 添加退化前缀B1，与input匹配
"""

import os
import sys
import shutil
import re
from collections import defaultdict

# 所有退化类型
DEGRADATIONS = ['B1', 'B2', 'B3', 'GB1', 'GB2', 'GB3', 'G1', 'G2', 'G3',
                'Y1', 'Y2', 'Y3', 'YG1', 'YG2', 'YG3']

def parse_validation_filename(filename: str):
    """
    解析验证集文件名并去除渲染后缀
    
    s1__B1__cam_dx+0.00_dy+0.50_yaw000.png
    -> (scene='s1', degradation='B1', core='cam_dx+0.00_dy+0.50_yaw000')
    
    s2__B1__cam_dx+0.00_dy+0.00_yaw090_rgb_0001.png
    -> (scene='s2', degradation='B1', core='cam_dx+0.00_dy+0.00_yaw090')
    """
    basename = os.path.splitext(filename)[0]
    parts = basename.split('__')
    
    if len(parts) >= 3:
        scene = parts[0]  # s1, s2, s3, s4
        degradation = parts[1]  # B1, B2, ..., YG3
        core = '__'.join(parts[2:])  # cam_dx+0.00_dy+0.50_yaw000
        
        # 🔥 去除渲染后缀（与训练集对齐）
        core = re.sub(r'_(rgb|mist|depth|normal)(_vis)?(_\d+)?$', '', core)
        
        return scene, degradation, core
    
    return None, None, None

def find_train_depth_file(train_depth_dir: str, scene: str, core_basename: str):
    """
    在训练集depth目录中查找匹配的文件
    
    查找: s1_cam_dx+0.00_dy+0.50_yaw000.png
    """
    # 训练集格式: s{scene}_{core}.png
    train_filename = f"{scene}_{core_basename}.png"
    train_path = os.path.join(train_depth_dir, train_filename)
    
    if os.path.exists(train_path):
        return train_path
    
    return None

def create_depth_validation(val_input_dir: str, train_depth_dir: str, 
                           target_dir: str, dry_run: bool = False):
    """创建深度验证集"""
    
    print("="*60)
    print("创建深度验证集")
    print("="*60)
    print(f"验证集input: {val_input_dir}")
    print(f"训练集depth: {train_depth_dir}")
    print(f"目标目录: {target_dir}")
    print(f"模式: {'模拟' if dry_run else '执行'}")
    
    if not os.path.exists(val_input_dir):
        print(f"⚠️  验证集input不存在")
        return
    
    if not os.path.exists(train_depth_dir):
        print(f"⚠️  训练集depth不存在")
        return
    
    # 创建目标目录
    input_target = os.path.join(target_dir, 'input')
    depth_target = os.path.join(target_dir, 'depth')
    
    if not dry_run:
        os.makedirs(input_target, exist_ok=True)
        os.makedirs(depth_target, exist_ok=True)
    
    # 获取所有input文件
    input_files = sorted([f for f in os.listdir(val_input_dir) if f.endswith('.png')])
    print(f"\n验证集input文件: {len(input_files)}")
    
    stats = {
        'input_copied': 0,
        'depth_copied': 0,
        'depth_not_found': 0,
        'unique_samples': set(),
    }
    
    not_found_list = []
    
    for i, input_filename in enumerate(input_files):
        # 1. 复制input文件
        input_src = os.path.join(val_input_dir, input_filename)
        input_dst = os.path.join(input_target, input_filename)
        
        if not dry_run:
            shutil.copy2(input_src, input_dst)
        stats['input_copied'] += 1
        
        # 2. 解析文件名
        scene, degradation, core_basename = parse_validation_filename(input_filename)
        
        if not scene or not core_basename:
            print(f"  ⚠️  无法解析: {input_filename}")
            continue
        
        stats['unique_samples'].add(f"{scene}_{core_basename}")
        
        # 3. 查找训练集中的depth文件
        train_depth_path = find_train_depth_file(train_depth_dir, scene, core_basename)
        
        if train_depth_path:
            # 复制depth文件（使用与input相同的文件名，包含退化前缀）
            depth_dst = os.path.join(depth_target, input_filename)
            
            if not dry_run:
                shutil.copy2(train_depth_path, depth_dst)
            stats['depth_copied'] += 1
        else:
            stats['depth_not_found'] += 1
            not_found_list.append(input_filename)
            if len(not_found_list) <= 3:
                print(f"  ⚠️  未找到depth: {input_filename}")
                print(f"      查找: {scene}_{core_basename}.png")
        
        # 进度
        if (i + 1) % 2000 == 0:
            print(f"  进度: {i+1}/{len(input_files)}")
    
    # 总结
    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")
    print(f"Input复制: {stats['input_copied']}")
    print(f"Depth复制: {stats['depth_copied']}")
    print(f"Depth未找到: {stats['depth_not_found']}")
    print(f"唯一样本数: {len(stats['unique_samples'])}")
    print(f"配对成功率: {stats['depth_copied']/stats['input_copied']*100:.2f}%")
    
    if not_found_list:
        print(f"\n⚠️  共{len(not_found_list)}个文件未找到depth")
    
    return stats

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='创建深度验证集')
    parser.add_argument('--val-input', default='DATA/validation/UBB-M_reference/input',
                       help='已规范化的验证集input目录')
    parser.add_argument('--train-depth', default='F:/DATASATES/UBB_train/depth',
                       help='训练集depth目录')
    parser.add_argument('--target', default='DATA/validation/UBB-M_depth',
                       help='深度验证集目标目录')
    parser.add_argument('--dry-run', action='store_true', help='只模拟')
    
    args = parser.parse_args()
    
    create_depth_validation(args.val_input, args.train_depth, args.target, args.dry_run)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

