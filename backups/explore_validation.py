#!/usr/bin/env python3
import os

print("="*60)
print("验证集数据结构探查")
print("="*60)

# 有参考验证集
ref_dir = 'DATA/validation/UBB-M_reference'
print(f"\n{'='*60}")
print("有参考验证集")
print(f"{'='*60}")

if os.path.exists(ref_dir):
    # input
    input_dir = os.path.join(ref_dir, 'input')
    if os.path.exists(input_dir):
        files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
        print(f"\ninput/ - {len(files)} 文件")
        print("  示例文件名:")
        for f in files[:10]:
            print(f"    {f}")
    
    # gt
    gt_dir = os.path.join(ref_dir, 'gt')
    if os.path.exists(gt_dir):
        files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.png')])
        print(f"\ngt/ - {len(files)} 文件")
        print("  示例文件名:")
        for f in files[:10]:
            print(f"    {f}")

# 无参考验证集
noref_dir = 'DATA/validation/UBB-M_noreference'
print(f"\n{'='*60}")
print("无参考验证集")
print(f"{'='*60}")

if os.path.exists(noref_dir):
    input_dir = os.path.join(noref_dir, 'input')
    if os.path.exists(input_dir):
        files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
        print(f"\ninput/ - {len(files)} 文件")
        print("  示例文件名:")
        for f in files[:10]:
            print(f"    {f}")

print("\n" + "="*60)




