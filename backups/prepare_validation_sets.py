#!/usr/bin/env python3
"""
UBB验证集准备脚本

功能:
1. 规范化验证集文件名 (去除场景和退化前缀，保持与训练集一致)
2. 将有参考和无参考验证集分别处理
3. 记录规范化后的basename列表，用于从训练集中排除

验证集命名格式:
  Input: 24mm__B1__cam_dx+0.00_dy+0.50_yaw000.png
  GT:    24mm__B1__cam_dx+0.00_dy+0.50_yaw000__GT.png
  
规范化后:
  Input: s1_cam_dx+0.00_dy+0.50_yaw000.png (保留场景前缀用于追溯)
  GT:    s1_cam_dx+0.00_dy+0.50_yaw000.png

场景映射:
  24mm -> Scene 1
  2    -> Scene 2
  3    -> Scene 3
  85mm -> Scene 4
"""

import os
import sys
import re
import shutil
import json
from collections import defaultdict
from typing import Dict, Set, List, Tuple

# 场景映射
SCENE_MAPPING = {
    '24mm': '1',
    '2': '2',
    '3': '3',
    '85mm': '4',
}

# 退化类型映射 (如果需要)
DEGRADATION_MAPPING = {
    'B1': 'color_B_1', 'B2': 'color_B_2', 'B3': 'color_B_3',
    'GB1': 'color_BG_1', 'GB2': 'color_BG_2', 'GB3': 'color_BG_3',
    'G1': 'color_G_1', 'G2': 'color_G_2', 'G3': 'color_G_3',
    'Y1': 'color_Y_1', 'Y2': 'color_Y_2', 'Y3': 'color_Y_3',
    'YG1': 'color_YG_1', 'YG2': 'color_YG_2', 'YG3': 'color_YG_3',
}


def parse_validation_filename(filename: str) -> Tuple[str, str, str]:
    """
    解析验证集文件名并规范化
    
    Args:
        filename: 如 24mm__B1__cam_dx+0.00_dy+0.50_yaw000.png
    
    Returns:
        (scene_prefix, degradation_type, core_basename)
        例如: ('24mm', 'B1', 'cam_dx+0.00_dy+0.50_yaw000')
    """
    basename = os.path.splitext(filename)[0]
    
    # 去除 __GT 后缀（如果存在）
    basename = re.sub(r'__GT$', '', basename)
    
    # 解析格式: {scene}__{degradation}__{core}
    parts = basename.split('__')
    
    if len(parts) >= 3:
        scene_prefix = parts[0]
        degradation_type = parts[1]
        core_basename = '__'.join(parts[2:])  # 剩余部分
        
        # 规范化core_basename（修复空格问题）
        core_basename = re.sub(r'cam_dx\s+(\d)', r'cam_dx+\1', core_basename)
        core_basename = re.sub(r'_dy\s+(\d)', r'_dy+\1', core_basename)
        core_basename = re.sub(r'cam_dx\s+\-', r'cam_dx-', core_basename)
        core_basename = re.sub(r'_dy\s+\-', r'_dy-', core_basename)
        
        return scene_prefix, degradation_type, core_basename
    
    # 如果格式不匹配，返回空
    return '', '', basename


def normalize_validation_filename(filename: str, add_scene_prefix: bool = True) -> str:
    """
    规范化验证集文件名
    
    Args:
        filename: 原始文件名
        add_scene_prefix: 是否添加场景前缀 (s1_, s2_, 等)
    
    Returns:
        规范化后的文件名
    """
    ext = os.path.splitext(filename)[1]
    scene_prefix, degradation, core_basename = parse_validation_filename(filename)
    
    if not core_basename:
        # 无法解析，返回原文件名
        return filename
    
    # 添加场景前缀（如果需要）
    if add_scene_prefix and scene_prefix in SCENE_MAPPING:
        scene_id = SCENE_MAPPING[scene_prefix]
        return f"s{scene_id}_{core_basename}{ext}"
    else:
        return f"{core_basename}{ext}"


def normalize_validation_set(source_dir: str, target_dir: str, 
                             has_gt: bool = True, dry_run: bool = False) -> Dict:
    """
    规范化验证集
    
    Args:
        source_dir: 源验证集目录 (包含input/, gt/子文件夹)
        target_dir: 目标目录
        has_gt: 是否有GT文件夹（有参考 vs 无参考）
        dry_run: 是否只模拟
    
    Returns:
        统计信息和basename列表
    """
    print(f"\n{'='*60}")
    print(f"处理验证集: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"类型: {'有参考' if has_gt else '无参考'}")
    print(f"{'='*60}")
    
    stats = {
        'source_dir': source_dir,
        'target_dir': target_dir,
        'has_gt': has_gt,
        'input_files_copied': 0,
        'gt_files_copied': 0,
        'unique_basenames': set(),
        'scene_distribution': defaultdict(int),
        'degradation_distribution': defaultdict(int),
    }
    
    if not os.path.exists(source_dir):
        print(f"⚠️  源目录不存在")
        return stats
    
    # 处理input文件夹
    input_source = os.path.join(source_dir, 'input')
    input_target = os.path.join(target_dir, 'input')
    
    if os.path.exists(input_source):
        if not dry_run:
            os.makedirs(input_target, exist_ok=True)
        
        input_files = sorted([f for f in os.listdir(input_source) if f.endswith('.png')])
        print(f"\n处理 input/ ({len(input_files)} 文件)")
        
        for i, old_name in enumerate(input_files):
            new_name = normalize_validation_filename(old_name, add_scene_prefix=True)
            
            # 记录统计
            scene_prefix, degradation, core_basename = parse_validation_filename(old_name)
            stats['unique_basenames'].add(core_basename)
            if scene_prefix in SCENE_MAPPING:
                scene_id = SCENE_MAPPING[scene_prefix]
                stats['scene_distribution'][f'scene_{scene_id}'] += 1
            stats['degradation_distribution'][degradation] += 1
            
            if not dry_run:
                old_path = os.path.join(input_source, old_name)
                new_path = os.path.join(input_target, new_name)
                shutil.copy2(old_path, new_path)
            
            stats['input_files_copied'] += 1
            
            if i < 3:  # 显示前3个示例
                print(f"  {old_name}")
                print(f"  -> {new_name}")
        
        print(f"  ✅ 处理 {stats['input_files_copied']} 个文件")
    
    # 处理GT文件夹（如果有）
    if has_gt:
        gt_source = os.path.join(source_dir, 'gt')
        gt_target = os.path.join(target_dir, 'gt')
        
        if os.path.exists(gt_source):
            if not dry_run:
                os.makedirs(gt_target, exist_ok=True)
            
            gt_files = sorted([f for f in os.listdir(gt_source) if f.endswith('.png')])
            print(f"\n处理 gt/ ({len(gt_files)} 文件)")
            
            # 🔥 GT去重：每个样本只保留一个GT（15个退化共享同一个GT）
            processed_basenames = set()
            gt_copied = 0
            gt_skipped = 0
            
            for i, old_name in enumerate(gt_files):
                new_name = normalize_validation_filename(old_name, add_scene_prefix=True)
                
                # 检查是否已经复制过这个basename的GT
                basename = os.path.splitext(new_name)[0]
                if basename in processed_basenames:
                    gt_skipped += 1
                    continue
                
                processed_basenames.add(basename)
                
                if not dry_run:
                    old_path = os.path.join(gt_source, old_name)
                    new_path = os.path.join(gt_target, new_name)
                    shutil.copy2(old_path, new_path)
                
                gt_copied += 1
                stats['gt_files_copied'] += 1
                
                if gt_copied <= 3:  # 显示前3个示例
                    print(f"  {old_name}")
                    print(f"  -> {new_name}")
            
            print(f"  ✅ 复制 {gt_copied} 个唯一GT (跳过 {gt_skipped} 个重复)")
    
    return stats


def save_validation_basenames(basenames: Set[str], output_file: str):
    """保存验证集的basename列表，用于后续从训练集中排除"""
    basenames_list = sorted(list(basenames))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'count': len(basenames_list),
            'basenames': basenames_list,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 验证集basename列表已保存: {output_file}")
    print(f"   共 {len(basenames_list)} 个唯一样本")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='准备UBB验证集')
    parser.add_argument('--source-ref', default='DATA/validation/UBB-M_reference',
                       help='有参考验证集源目录')
    parser.add_argument('--source-noref', default='DATA/validation/UBB-M_noreference',
                       help='无参考验证集源目录')
    parser.add_argument('--target-ref', default='F:/DATASATES/UBB_validation_reference',
                       help='有参考验证集目标目录')
    parser.add_argument('--target-noref', default='F:/DATASATES/UBB_validation_noreference',
                       help='无参考验证集目标目录')
    parser.add_argument('--dry-run', action='store_true', help='只模拟不实际复制')
    parser.add_argument('--validate-only', action='store_true', help='只分析不处理')
    parser.add_argument('--basename-output', default='F:/DATASATES/UBB_validation_basenames.json',
                       help='输出验证集basename列表的JSON文件')
    
    args = parser.parse_args()
    
    print("="*60)
    print("UBB验证集准备工具")
    print("="*60)
    
    all_basenames = set()
    
    # 分析模式
    if args.validate_only:
        print("\n【验证分析模式】")
        
        # 分析有参考验证集
        if os.path.exists(args.source_ref):
            input_dir = os.path.join(args.source_ref, 'input')
            if os.path.exists(input_dir):
                files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
                print(f"\n有参考验证集 ({args.source_ref}):")
                print(f"  Input文件: {len(files)}")
                
                # 场景分布
                scene_dist = defaultdict(int)
                deg_dist = defaultdict(int)
                for f in files:
                    scene_prefix, degradation, core = parse_validation_filename(f)
                    if scene_prefix in SCENE_MAPPING:
                        scene_dist[SCENE_MAPPING[scene_prefix]] += 1
                    deg_dist[degradation] += 1
                    all_basenames.add(core)
                
                print(f"\n  场景分布:")
                for scene, count in sorted(scene_dist.items()):
                    print(f"    Scene {scene}: {count} 文件")
                
                print(f"\n  退化类型分布:")
                for deg, count in sorted(deg_dist.items()):
                    print(f"    {deg}: {count} 文件")
                
                print(f"\n  唯一basename: {len(all_basenames)}")
        
        # 分析无参考验证集
        if os.path.exists(args.source_noref):
            input_dir = os.path.join(args.source_noref, 'input')
            if os.path.exists(input_dir):
                files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
                print(f"\n无参考验证集 ({args.source_noref}):")
                print(f"  Input文件: {len(files)}")
                
                # 显示示例
                print(f"\n  文件名示例:")
                for f in sorted(files)[:5]:
                    print(f"    {f}")
        
        return 0
    
    # 处理有参考验证集
    stats_ref = normalize_validation_set(
        source_dir=args.source_ref,
        target_dir=args.target_ref,
        has_gt=True,
        dry_run=args.dry_run
    )
    all_basenames.update(stats_ref['unique_basenames'])
    
    # 处理无参考验证集
    stats_noref = normalize_validation_set(
        source_dir=args.source_noref,
        target_dir=args.target_noref,
        has_gt=False,
        dry_run=args.dry_run
    )
    all_basenames.update(stats_noref['unique_basenames'])
    
    # 保存basename列表
    if not args.dry_run and all_basenames:
        save_validation_basenames(all_basenames, args.basename_output)
    
    # 总结
    print(f"\n{'='*60}")
    print("处理总结")
    print(f"{'='*60}")
    print(f"有参考验证集: {stats_ref['input_files_copied']} input, {stats_ref['gt_files_copied']} GT")
    print(f"无参考验证集: {stats_noref['input_files_copied']} input")
    print(f"唯一样本数: {len(all_basenames)}")
    
    print(f"\n场景分布:")
    combined_scene_dist = defaultdict(int)
    for k, v in stats_ref['scene_distribution'].items():
        combined_scene_dist[k] += v
    for k, v in stats_noref['scene_distribution'].items():
        combined_scene_dist[k] += v
    for scene, count in sorted(combined_scene_dist.items()):
        print(f"  {scene}: {count} 文件")
    
    if not args.dry_run:
        print(f"\n下一步:")
        print(f"  运行脚本从训练集中排除这些验证样本:")
        print(f"  python scripts/exclude_validation_from_train.py")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

