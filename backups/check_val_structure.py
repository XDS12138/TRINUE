#!/usr/bin/env python3
import os, re
from collections import defaultdict

input_dir = 'DATA/validation/UBB-M_reference/input'
files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

print(f"总文件数: {len(files)}")

# 按位置分组（忽略退化）
loc_to_degs = defaultdict(set)

for f in files:
    # s1__B1__cam_dx+0.00_dy+0.50_yaw000.png
    # 提取: s1 + cam_dx+0.00_dy+0.50_yaw000
    parts = f.split('__')
    if len(parts) >= 3:
        scene = parts[0]
        degradation = parts[1]
        core = '__'.join(parts[2:]).replace('.png', '')
        
        # 去除后缀
        core = re.sub(r'_(rgb|mist|depth|normal)(_vis)?(_\d+)?$', '', core)
        
        location_key = f"{scene}_{core}"
        loc_to_degs[location_key].add(degradation)

print(f"唯一位置: {len(loc_to_degs)}")

# 统计每个位置有多少种退化
deg_counts = defaultdict(int)
for loc, degs in loc_to_degs.items():
    deg_counts[len(degs)] += 1

print("\n每位置的退化数量分布:")
for count, num_locs in sorted(deg_counts.items()):
    print(f"  {count}种退化: {num_locs}个位置")

# 找有15种退化的位置
full_locs = [loc for loc, degs in loc_to_degs.items() if len(degs) == 15]
print(f"\n有全部15种退化的位置: {len(full_locs)}")

# 示例
if full_locs:
    sample_loc = full_locs[0]
    print(f"\n示例位置: {sample_loc}")
    print(f"  包含退化: {sorted(loc_to_degs[sample_loc])}")




