#!/usr/bin/env python3
import os

ref_input = 'F:/DATASATES/UBB_validation_reference/input'
ref_gt = 'F:/DATASATES/UBB_validation_reference/gt'
noref_input = 'F:/DATASATES/UBB_validation_noreference/input'

print("="*60)
print("验证集状态检查")
print("="*60)

# 有参考验证集
if os.path.exists(ref_input):
    input_files = [f for f in os.listdir(ref_input) if f.endswith('.png')]
    print(f"\n有参考 input: {len(input_files)}/20040")
    
    # 唯一basename
    input_basenames = set([os.path.splitext(f)[0] for f in input_files])
    print(f"  唯一basename: {len(input_basenames)}")
    
    # 文件名示例
    print(f"  示例: {sorted(input_files)[:3]}")
else:
    print(f"\n有参考 input: 不存在")

if os.path.exists(ref_gt):
    gt_files = [f for f in os.listdir(ref_gt) if f.endswith('.png')]
    print(f"\n有参考 GT: {len(gt_files)}")
    
    # 唯一basename
    gt_basenames = set([os.path.splitext(f)[0] for f in gt_files])
    print(f"  唯一basename: {len(gt_basenames)}")
    
    # 文件名示例
    print(f"  示例: {sorted(gt_files)[:3]}")
    
    # 检查input和GT的匹配
    if os.path.exists(ref_input):
        matched = input_basenames & gt_basenames
        print(f"  与input匹配: {len(matched)}/{len(input_basenames)}")
else:
    print(f"\n有参考 GT: 不存在")

# 无参考验证集
if os.path.exists(noref_input):
    noref_files = [f for f in os.listdir(noref_input) if f.endswith('.png')]
    print(f"\n无参考 input: {len(noref_files)}/20040")
    
    noref_basenames = set([os.path.splitext(f)[0] for f in noref_files])
    print(f"  唯一basename: {len(noref_basenames)}")
    print(f"  示例: {sorted(noref_files)[:3]}")
else:
    print(f"\n无参考 input: 不存在")

print("\n" + "="*60)

# 检查Python进程
import subprocess
try:
    result = subprocess.run(['powershell', '-Command', 'Get-Process python -ErrorAction SilentlyContinue'], 
                          capture_output=True, text=True)
    if result.stdout.strip():
        print("Python进程: 运行中")
    else:
        print("Python进程: 无")
except:
    pass




