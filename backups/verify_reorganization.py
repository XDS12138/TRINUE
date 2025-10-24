#!/usr/bin/env python3
"""
验证UBB数据集整理结果

用法:
  python scripts/verify_reorganization.py F:/DATASATES/UBB_train
"""

import os
import sys

def verify_reorganization(target_root: str, expected_count: int = 13356):
    """验证整理后的数据集完整性"""
    
    folders = [
        'gt', 'depth',
        'color_B_1', 'color_B_2', 'color_B_3',
        'color_BG_1', 'color_BG_2', 'color_BG_3',
        'color_G_1', 'color_G_2', 'color_G_3',
        'color_Y_1', 'color_Y_2', 'color_Y_3',
        'color_YG_1', 'color_YG_2', 'color_YG_3',
    ]
    
    print(f"验证目标目录: {target_root}")
    print("="*70)
    
    if not os.path.exists(target_root):
        print(f"❌ 目标目录不存在: {target_root}")
        return False
    
    all_ok = True
    total_files = 0
    
    # 验证17个数据文件夹
    for folder in folders:
        path = os.path.join(target_root, folder)
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith('.png')]
            count = len(files)
            total_files += count
            status = "✅" if count == expected_count else "❌"
            print(f"{status} {folder:15s}: {count:6d} / {expected_count}")
            if count != expected_count:
                all_ok = False
                # 显示文件名前缀示例
                if count > 0:
                    examples = sorted(files)[:3]
                    print(f"   示例文件: {examples}")
        else:
            print(f"❌ {folder:15s}: 文件夹不存在")
            all_ok = False
    
    print("="*70)
    print(f"总文件数: {total_files} (期望: {expected_count * 17} = {expected_count}×17)")
    
    # 验证映射CSV文件
    print("\n验证映射CSV文件...")
    print("-"*70)
    
    scene_counts = {
        '1': 4200,
        '2': 2268,
        '3': 2688,
        '4': 4200,
    }
    
    csv_ok = True
    for scene, expected_mappings in scene_counts.items():
        csv_path = os.path.join(target_root, f'mapping_scene_{scene}.csv')
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                lines = len(f.readlines()) - 1  # 排除表头
            # 每个样本有17个文件（17个映射行）
            expected_lines = expected_mappings * 17
            status = "✅" if lines == expected_lines else "⚠️ "
            print(f"{status} mapping_scene_{scene}.csv: {lines:6d} 条映射 (期望: {expected_lines})")
            if lines != expected_lines:
                csv_ok = False
        else:
            print(f"❌ mapping_scene_{scene}.csv: 不存在")
            csv_ok = False
    
    # 检查场景前缀
    print("\n验证场景前缀...")
    print("-"*70)
    
    prefix_ok = True
    gt_path = os.path.join(target_root, 'gt')
    if os.path.exists(gt_path):
        files = [f for f in os.listdir(gt_path) if f.endswith('.png')]
        scene_prefixes = {'s1_': 0, 's2_': 0, 's3_': 0, 's4_': 0}
        
        for f in files:
            for prefix in scene_prefixes.keys():
                if f.startswith(prefix):
                    scene_prefixes[prefix] += 1
                    break
        
        print("场景文件分布:")
        for prefix, count in sorted(scene_prefixes.items()):
            scene_num = prefix[1]  # 's1_' -> '1'
            expected = scene_counts.get(scene_num, 0)
            status = "✅" if count == expected else "❌"
            print(f"  {status} {prefix}: {count:6d} 文件 (期望: {expected})")
            if count != expected:
                prefix_ok = False
    
    # 最终总结
    print("\n" + "="*70)
    if all_ok and csv_ok and prefix_ok:
        print("✅ 所有验证通过！数据集整理成功！")
        print(f"\n可以更新 configs/train.yaml 中的 train_root 为:")
        print(f'  train_root: "{target_root}"')
        return True
    else:
        print("❌ 发现问题，请检查整理过程:")
        if not all_ok:
            print("  - 某些文件夹的文件数量不正确")
        if not csv_ok:
            print("  - 映射CSV文件缺失或记录数不正确")
        if not prefix_ok:
            print("  - 场景前缀分布不正确")
        return False


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else "F:/DATASATES/UBB_train"
    success = verify_reorganization(target)
    sys.exit(0 if success else 1)

