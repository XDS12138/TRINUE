#!/usr/bin/env python3
"""检查训练集图像尺寸分布"""
import os
from PIL import Image
from collections import Counter

train_gt = 'F:/DATASATES/UBB_train/gt'
files = sorted([f for f in os.listdir(train_gt) if f.endswith('.png')])

print(f"检查 {len(files)} 个GT图像的尺寸...")

sizes = []
for i, f in enumerate(files):
    if i % 1000 == 0:
        print(f"  进度: {i}/{len(files)}")
    
    path = os.path.join(train_gt, f)
    try:
        img = Image.open(path)
        sizes.append(img.size)  # (W, H)
    except:
        pass

# 统计尺寸分布
size_counter = Counter(sizes)

print(f"\n尺寸分布（共{len(size_counter)}种）:")
for size, count in size_counter.most_common(10):
    print(f"  {size[0]:4d} × {size[1]:4d}: {count:5d} 个 ({count/len(files)*100:.1f}%)")

print(f"\n总结:")
print(f"  最常见尺寸: {size_counter.most_common(1)[0][0]}")
print(f"  唯一尺寸数: {len(size_counter)}")




