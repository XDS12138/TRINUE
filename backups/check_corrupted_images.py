#!/usr/bin/env python3
"""
检查数据集中的损坏图像

检查内容:
1. 能否打开图像文件
2. 能否加载完整的图像数据
3. 尺寸是否合理
4. 深度图是否为16位
"""

import os
import sys
from PIL import Image
import numpy as np
from tqdm import tqdm


def check_image_file(image_path, check_data=True, expected_mode='RGB'):
    """
    检查单个图像文件
    
    Args:
        image_path: 图像路径
        check_data: 是否完整加载数据验证
        expected_mode: 期望的图像模式（RGB, L等）
    
    Returns:
        (is_valid, error_message)
    """
    try:
        # 1. 尝试打开图像
        img = Image.open(image_path)
        
        # 2. 检查尺寸
        width, height = img.size
        if width <= 0 or height <= 0:
            return False, f"尺寸无效: {width}x{height}"
        
        if width > 10000 or height > 10000:
            return False, f"尺寸异常大: {width}x{height}"
        
        # 3. 检查模式（如果指定）
        if expected_mode and img.mode != expected_mode:
            # 深度图可能是I或I;16
            if 'depth' in image_path and img.mode in ['I', 'I;16', 'L']:
                pass  # 深度图模式可接受
            # RGB图像允许RGBA（训练时会自动convert）
            elif expected_mode == 'RGB' and img.mode == 'RGBA':
                pass  # RGBA可接受，会自动转换
            else:
                return False, f"模式错误: {img.mode}（期望{expected_mode}）"
        
        # 4. 尝试加载完整数据（检测truncated）
        if check_data:
            img_array = np.array(img)
            
            # 检查数据有效性
            if img_array.size == 0:
                return False, "图像数据为空"
            
            # 检查是否有异常值
            if np.any(np.isnan(img_array)):
                return False, "包含NaN值"
            
            if np.any(np.isinf(img_array)):
                return False, "包含Inf值"
        
        img.close()
        return True, None
        
    except OSError as e:
        return False, f"OSError: {str(e)}"
    except Exception as e:
        return False, f"未知错误: {str(e)}"


def check_folder(folder_path, description="", check_data=True, sample_rate=1.0):
    """
    检查文件夹中的所有图像
    
    Args:
        folder_path: 文件夹路径
        description: 描述
        check_data: 是否完整加载数据
        sample_rate: 抽样比例（1.0=全部检查，0.1=抽查10%）
    
    Returns:
        (total, corrupted, corrupted_files)
    """
    if not os.path.exists(folder_path):
        print(f"  ⚠️  文件夹不存在: {folder_path}")
        return 0, 0, []
    
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    
    # 抽样
    if sample_rate < 1.0:
        import random
        num_check = max(1, int(len(files) * sample_rate))
        files = random.sample(files, num_check)
    
    print(f"\n检查: {description}")
    print(f"  文件数: {len(files)}")
    
    corrupted_files = []
    
    for filename in tqdm(files, desc=f"  验证{description}", leave=False):
        file_path = os.path.join(folder_path, filename)
        
        # 判断是否是深度图
        expected_mode = 'L' if 'depth' in folder_path.lower() else 'RGB'
        
        is_valid, error = check_image_file(file_path, check_data, expected_mode)
        
        if not is_valid:
            corrupted_files.append({
                'file': filename,
                'path': file_path,
                'error': error
            })
    
    if corrupted_files:
        print(f"  ❌ 发现 {len(corrupted_files)} 个损坏文件")
        for item in corrupted_files[:5]:
            print(f"     - {item['file']}: {item['error']}")
        if len(corrupted_files) > 5:
            print(f"     ... 还有 {len(corrupted_files)-5} 个")
    else:
        print(f"  ✅ 全部通过")
    
    return len(files), len(corrupted_files), corrupted_files


def check_dataset(dataset_root, dataset_name, folders_to_check, sample_rate=1.0):
    """检查整个数据集"""
    print(f"\n{'='*60}")
    print(f"{dataset_name}")
    print(f"{'='*60}")
    print(f"根目录: {dataset_root}")
    
    if not os.path.exists(dataset_root):
        print(f"⚠️  数据集不存在")
        return {}
    
    all_corrupted = {}
    
    for folder in folders_to_check:
        folder_path = os.path.join(dataset_root, folder)
        total, corrupted, corrupted_list = check_folder(
            folder_path, folder, check_data=True, sample_rate=sample_rate
        )
        
        if corrupted > 0:
            all_corrupted[folder] = corrupted_list
    
    return all_corrupted


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='检查数据集中的损坏图像')
    parser.add_argument('--sample-rate', type=float, default=0.1,
                       help='抽样比例（0.1=检查10%，1.0=全部检查）')
    parser.add_argument('--full-check', action='store_true',
                       help='完整检查（等同于--sample-rate 1.0）')
    parser.add_argument('--output-json', default='F:/DATASATES/corrupted_images_report.json',
                       help='输出损坏文件列表的JSON')
    
    args = parser.parse_args()
    
    if args.full_check:
        args.sample_rate = 1.0
    
    print("="*60)
    print("UBB数据集完整性检查")
    print("="*60)
    print(f"抽样比例: {args.sample_rate*100:.0f}%")
    
    all_corrupted = {}
    
    # 1. 检查多输入训练集
    multi_folders = ['gt', 'depth', 'color_B_1', 'color_B_2', 'color_B_3',
                     'color_BG_1', 'color_BG_2', 'color_BG_3',
                     'color_G_1', 'color_G_2', 'color_G_3',
                     'color_Y_1', 'color_Y_2', 'color_Y_3',
                     'color_YG_1', 'color_YG_2', 'color_YG_3']
    
    corrupted = check_dataset('D:/UBB_train', '多输入训练集', multi_folders, args.sample_rate)
    if corrupted:
        all_corrupted['multi_input_train'] = corrupted
    
    # 2. 检查单输入训练集
    single_folders = ['input', 'gt']
    corrupted = check_dataset('D:/UBB_train_single_input', '单输入训练集', single_folders, args.sample_rate)
    if corrupted:
        all_corrupted['single_input_train'] = corrupted
    
    # 3. 检查有参考验证集
    ref_folders = ['input', 'gt']
    corrupted = check_dataset('DATA/validation/UBB-M_reference', '有参考验证集', ref_folders, args.sample_rate)
    if corrupted:
        all_corrupted['validation_reference'] = corrupted
    
    # 4. 检查无参考验证集
    noref_folders = ['input']
    corrupted = check_dataset('DATA/validation/UBB-M_noreference', '无参考验证集', noref_folders, args.sample_rate)
    if corrupted:
        all_corrupted['validation_noreference'] = corrupted
    
    # 5. 检查深度验证集
    depth_folders = ['input', 'depth']
    corrupted = check_dataset('DATA/validation/UBB-M_depth', '深度验证集', depth_folders, args.sample_rate)
    if corrupted:
        all_corrupted['validation_depth'] = corrupted
    
    # 总结
    print(f"\n{'='*60}")
    print("检查总结")
    print(f"{'='*60}")
    
    if all_corrupted:
        print(f"❌ 发现损坏文件！")
        for dataset_name, folders in all_corrupted.items():
            print(f"\n{dataset_name}:")
            for folder, files in folders.items():
                print(f"  {folder}: {len(files)} 个损坏")
        
        # 保存详细报告
        import json
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(all_corrupted, f, ensure_ascii=False, indent=2)
        print(f"\n详细报告已保存: {args.output_json}")
    else:
        print(f"✅ 所有数据集均无损坏文件！")
    
    return 0 if not all_corrupted else 1


if __name__ == '__main__':
    sys.exit(main())

