#!/usr/bin/env python3
"""快速检查指定数据集的损坏图像"""
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import json

def check_image(path):
    """检查单个图像，返回(is_valid, error)"""
    try:
        img = Image.open(path)
        # 尝试加载完整数据
        img_array = np.array(img)
        
        # 检查有效性
        if img_array.size == 0:
            return False, "数据为空"
        if np.any(np.isnan(img_array)):
            return False, "包含NaN"
        if np.any(np.isinf(img_array)):
            return False, "包含Inf"
        
        img.close()
        return True, None
    except Exception as e:
        return False, str(e)

def check_folder_quick(folder_path, name):
    """快速检查文件夹"""
    if not os.path.exists(folder_path):
        print(f"{name}: 不存在")
        return []
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    corrupted = []
    
    for f in tqdm(files, desc=name):
        is_valid, error = check_image(os.path.join(folder_path, f))
        if not is_valid:
            corrupted.append({'file': f, 'error': error})
    
    if corrupted:
        print(f"  ❌ {len(corrupted)} 个损坏")
        for item in corrupted[:3]:
            print(f"     {item['file']}: {item['error']}")
    else:
        print(f"  ✅ 全部正常")
    
    return corrupted

print("="*60)
print("快速检查损坏图像")
print("="*60)

all_corrupted = {}

# 检查D:/UBB_train
print("\n多输入训练集（D:/UBB_train）:")
for folder in ['gt', 'color_B_1', 'color_BG_1', 'color_G_1', 'color_Y_1', 'color_YG_1']:
    corrupted = check_folder_quick(f'D:/UBB_train/{folder}', f'  {folder}')
    if corrupted:
        all_corrupted[f'train_{folder}'] = corrupted

# 检查验证集
print("\n有参考验证集（DATA/validation/UBB-M_reference）:")
for folder in ['input', 'gt']:
    corrupted = check_folder_quick(f'DATA/validation/UBB-M_reference/{folder}', f'  {folder}')
    if corrupted:
        all_corrupted[f'val_ref_{folder}'] = corrupted

print("\n" + "="*60)
if all_corrupted:
    print(f"❌ 发现损坏文件，总数: {sum(len(v) for v in all_corrupted.values())}")
    
    # 保存报告
    with open('F:/DATASATES/corrupted_report.json', 'w', encoding='utf-8') as f:
        json.dump(all_corrupted, f, indent=2)
    print(f"详细报告: F:/DATASATES/corrupted_report.json")
else:
    print("✅ 所有检查均通过！")




