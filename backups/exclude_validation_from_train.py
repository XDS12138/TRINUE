#!/usr/bin/env python3
"""
从训练集中排除验证集样本

功能:
1. 读取验证集的basename列表
2. 从训练集的17个文件夹中删除对应的文件
3. 确保训练集和验证集无重叠（避免数据泄露）

流程:
1. 先运行 prepare_validation_sets.py 生成 basename 列表
2. 再运行本脚本从训练集中删除

训练集结构:
  F:/DATASATES/UBB_train/
    ├── gt/ (s1_cam_dx+0.00_dy+0.50_yaw000.png)
    ├── depth/
    └── color_B_1/ ... color_YG_3/

验证集 basename 格式（无前缀）:
  cam_dx+0.00_dy+0.50_yaw000
"""

import os
import sys
import json
import re
from collections import defaultdict
from typing import Set, List, Dict

# 场景映射（与验证集脚本保持一致）
SCENE_MAPPING = {
    '24mm': '1',
    '2': '2',
    '3': '3',
    '85mm': '4',
}


def load_validation_basenames(json_file: str) -> Set[str]:
    """加载验证集basename列表"""
    if not os.path.exists(json_file):
        print(f"[WARNING] 验证集basename文件不存在: {json_file}")
        return set()
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    basenames = set(data.get('basenames', []))
    print(f"[OK] 加载验证集basename: {len(basenames)} 个唯一样本")
    
    return basenames


def parse_validation_filename_simple(filename: str) -> str:
    """
    从验证集文件名中提取core basename（去除场景和退化前缀，去除后缀）
    
    s1__B1__cam_dx+0.00_dy+0.50_yaw000.png
    -> cam_dx+0.00_dy+0.50_yaw000
    
    s2__B1__cam_dx+0.00_dy+0.00_yaw090_rgb_0001.png
    -> cam_dx+0.00_dy+0.00_yaw090 (去除_rgb_0001后缀)
    """
    basename = os.path.splitext(filename)[0]
    
    # 解析: {scene}__{degradation}__{core}
    parts = basename.split('__')
    if len(parts) >= 3:
        core = '__'.join(parts[2:])  # 提取core部分
        
        # 去除渲染后缀（与训练集对齐）
        core = re.sub(r'_(rgb|mist|depth|normal)(_vis)?(_\d+)?$', '', core)
        
        return core
    
    return basename


def scan_validation_set_directly(val_source_dir: str) -> Set[str]:
    """
    直接扫描验证集目录获取basename列表（如果JSON不存在）
    """
    basenames = set()
    
    input_dir = os.path.join(val_source_dir, 'input')
    if not os.path.exists(input_dir):
        return basenames
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            core_basename = parse_validation_filename_simple(filename)
            basenames.add(core_basename)
    
    return basenames


def match_train_file_to_val_basename(train_filename: str, val_basenames: Set[str]) -> bool:
    """
    判断训练集文件是否匹配验证集的basename
    
    训练集文件名格式: s1_cam_dx+0.00_dy+0.50_yaw000.png
    验证集basename: cam_dx+0.00_dy+0.50_yaw000
    """
    # 去除扩展名
    basename = os.path.splitext(train_filename)[0]
    
    # 去除场景前缀 s1_, s2_, s3_, s4_
    basename = re.sub(r'^s[1-4]_', '', basename)
    
    # 检查是否在验证集中
    return basename in val_basenames


def exclude_from_train_folder(folder_path: str, val_basenames: Set[str], 
                              dry_run: bool = False) -> Dict:
    """
    从训练集文件夹中删除验证集样本
    """
    stats = {
        'total_files': 0,
        'matched_files': 0,
        'deleted_files': 0,
        'errors': 0,
    }
    
    if not os.path.exists(folder_path):
        return stats
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    stats['total_files'] = len(files)
    
    for filename in files:
        if match_train_file_to_val_basename(filename, val_basenames):
            stats['matched_files'] += 1
            
            if not dry_run:
                try:
                    file_path = os.path.join(folder_path, filename)
                    os.remove(file_path)
                    stats['deleted_files'] += 1
                except Exception as e:
                    print(f"    ⚠️  删除失败: {filename} - {e}")
                    stats['errors'] += 1
    
    return stats


def exclude_validation_from_train(train_root: str, val_basenames: Set[str], 
                                  dry_run: bool = False) -> Dict:
    """
    从整个训练集中排除验证集样本
    """
    print(f"\n{'='*60}")
    print(f"从训练集中排除验证样本")
    print(f"{'='*60}")
    print(f"训练集目录: {train_root}")
    print(f"验证集样本数: {len(val_basenames)}")
    print(f"模式: {'模拟运行' if dry_run else '执行删除'}")
    
    if not os.path.exists(train_root):
        print(f"⚠️  训练集目录不存在: {train_root}")
        return {}
    
    # 需要处理的文件夹
    folders = [
        'gt', 'depth',
        'color_B_1', 'color_B_2', 'color_B_3',
        'color_BG_1', 'color_BG_2', 'color_BG_3',
        'color_G_1', 'color_G_2', 'color_G_3',
        'color_Y_1', 'color_Y_2', 'color_Y_3',
        'color_YG_1', 'color_YG_2', 'color_YG_3',
    ]
    
    total_stats = {
        'folders_processed': 0,
        'total_files_before': 0,
        'total_matched': 0,
        'total_deleted': 0,
        'total_errors': 0,
        'folder_details': {},
    }
    
    for folder in folders:
        folder_path = os.path.join(train_root, folder)
        
        print(f"\n处理: {folder}/")
        
        stats = exclude_from_train_folder(folder_path, val_basenames, dry_run)
        
        total_stats['folders_processed'] += 1
        total_stats['total_files_before'] += stats['total_files']
        total_stats['total_matched'] += stats['matched_files']
        total_stats['total_deleted'] += stats['deleted_files']
        total_stats['total_errors'] += stats['errors']
        total_stats['folder_details'][folder] = stats
        
        if stats['matched_files'] > 0:
            print(f"  原文件数: {stats['total_files']}")
            print(f"  匹配验证集: {stats['matched_files']}")
            if not dry_run:
                print(f"  已删除: {stats['deleted_files']}")
                print(f"  剩余: {stats['total_files'] - stats['deleted_files']}")
        else:
            print(f"  [OK] 无匹配文件，保持原样 ({stats['total_files']} 文件)")
    
    return total_stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='从训练集中排除验证集样本')
    parser.add_argument('--train-root', default='F:/DATASATES/UBB_train',
                       help='训练集根目录')
    parser.add_argument('--val-basename-json', default='F:/DATASATES/UBB_validation_basenames.json',
                       help='验证集basename列表JSON文件')
    parser.add_argument('--val-source-ref', default='DATA/validation/UBB-M_reference',
                       help='有参考验证集源目录（如果JSON不存在，直接扫描）')
    parser.add_argument('--val-source-noref', default='DATA/validation/UBB-M_noreference',
                       help='无参考验证集源目录（如果JSON不存在，直接扫描）')
    parser.add_argument('--dry-run', action='store_true', help='只模拟不实际删除')
    parser.add_argument('--stats-output', default='F:/DATASATES/exclusion_stats.json',
                       help='输出删除统计JSON文件')
    parser.add_argument('--skip-depth-warning', action='store_true', 
                       help='跳过深度验证集警告（如果已提取）')
    
    args = parser.parse_args()
    
    print("="*60)
    print("训练集验证集排除工具")
    print("="*60)
    
    # 🔥 重要警告：检查深度验证集
    depth_val_dir = 'DATA/validation/UBB-M_depth'
    if not args.skip_depth_warning and not os.path.exists(depth_val_dir):
        print("\n" + "="*60)
        print("【重要警告】深度验证集尚未提取！")
        print("="*60)
        print("\n本脚本会从训练集的depth/文件夹中删除验证样本。")
        print("如果你还需要深度验证集，请先运行：")
        print("\n  python scripts/prepare_depth_validation.py \\")
        print("    --val-input 'F:/DATASATES/UBB_validation_reference/input' \\")
        print("    --train-depth 'F:/DATASATES/UBB_train/depth' \\")
        print("    --target 'F:/DATASATES/UBB_depth_validation'")
        print("\n是否继续？(输入 yes 继续，或 Ctrl+C 取消)")
        
        if not args.dry_run:
            response = input("> ")
            if response.lower() != 'yes':
                print("已取消。")
                return 0
        else:
            print("\n(干运行模式，自动继续)")
    
    print()
    
    # 加载验证集basename
    val_basenames = load_validation_basenames(args.val_basename_json)
    
    # 如果JSON不存在，直接扫描验证集目录
    if not val_basenames:
        print("\nJSON不存在，直接扫描验证集目录...")
        val_basenames = scan_validation_set_directly(args.val_source_ref)
        val_basenames.update(scan_validation_set_directly(args.val_source_noref))
        print(f"[OK] 扫描得到 {len(val_basenames)} 个唯一basename")
    
    if not val_basenames:
        print("⚠️  未找到验证集basename，退出")
        return 1
    
    # 显示几个示例basename
    print(f"\n验证集basename示例:")
    for bn in sorted(list(val_basenames))[:5]:
        print(f"  - {bn}")
    
    # 执行排除
    stats = exclude_validation_from_train(args.train_root, val_basenames, args.dry_run)
    
    # 总结
    print(f"\n{'='*60}")
    print("排除总结")
    print(f"{'='*60}")
    print(f"处理文件夹: {stats['folders_processed']}")
    print(f"训练集原始文件数: {stats['total_files_before']}")
    print(f"匹配验证集: {stats['total_matched']}")
    
    if not args.dry_run:
        print(f"已删除: {stats['total_deleted']}")
        print(f"剩余训练样本: {(stats['total_files_before'] - stats['total_deleted']) // 17}")
        print(f"  (假设17组文件均匀删除)")
        
        # 保存统计
        with open(args.stats_output, 'w', encoding='utf-8') as f:
            # 转换set为list以便JSON序列化
            stats_serializable = {
                k: (list(v) if isinstance(v, set) else v)
                for k, v in stats.items()
            }
            json.dump(stats_serializable, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 统计信息已保存: {args.stats_output}")
    else:
        print(f"预计删除: {stats['total_matched']}")
        print(f"预计剩余训练样本: {(stats['total_files_before'] - stats['total_matched']) // 17}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

