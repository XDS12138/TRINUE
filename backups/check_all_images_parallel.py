#!/usr/bin/env python3
"""
多线程全量检查所有图像

利用NVMe SSD的高性能并发检查
"""

import os
import sys
import json
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def check_single_image(file_path):
    """
    检查单个图像文件
    
    Returns:
        (file_path, is_valid, error_message)
    """
    try:
        # 打开图像
        with Image.open(file_path) as img:
            # 检查尺寸
            width, height = img.size
            if width <= 0 or height <= 0:
                return file_path, False, f"尺寸无效: {width}x{height}"
            
            if width > 10000 or height > 10000:
                return file_path, False, f"尺寸异常: {width}x{height}"
            
            # 尝试加载完整数据（检测truncated/corrupted）
            try:
                img.load()  # 强制加载
            except Exception as e:
                return file_path, False, f"加载失败: {str(e)}"
            
            # 转换为numpy检查数据完整性
            img_array = np.array(img)
            
            if img_array.size == 0:
                return file_path, False, "数据为空"
            
            # 检查异常值
            if np.any(np.isnan(img_array)):
                return file_path, False, "包含NaN"
            
            if np.any(np.isinf(img_array)):
                return file_path, False, "包含Inf"
        
        return file_path, True, None
        
    except OSError as e:
        # 常见的损坏错误
        if 'truncated' in str(e).lower():
            return file_path, False, "文件被截断"
        elif 'cannot identify' in str(e).lower():
            return file_path, False, "无法识别图像格式"
        else:
            return file_path, False, f"OSError: {str(e)}"
    
    except Exception as e:
        return file_path, False, f"未知错误: {str(e)}"


def check_folder_parallel(folder_path, folder_name, num_threads=32):
    """
    多线程检查文件夹
    
    Args:
        folder_path: 文件夹路径
        folder_name: 文件夹名称（用于显示）
        num_threads: 线程数
    """
    if not os.path.exists(folder_path):
        return 0, 0, []
    
    # 获取所有PNG文件
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
    
    if len(files) == 0:
        return 0, 0, []
    
    print(f"\n{folder_name}: {len(files)} 文件")
    
    corrupted_files = []
    checked = 0
    
    # 使用线程池并行检查
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有任务
        futures = {executor.submit(check_single_image, f): f for f in files}
        
        # 使用tqdm显示进度
        with tqdm(total=len(files), desc=f"  检查{folder_name}") as pbar:
            for future in as_completed(futures):
                file_path, is_valid, error = future.result()
                checked += 1
                
                if not is_valid:
                    corrupted_files.append({
                        'file': os.path.basename(file_path),
                        'path': file_path,
                        'error': error
                    })
                
                pbar.update(1)
    
    # 显示结果
    if corrupted_files:
        print(f"  ❌ 损坏: {len(corrupted_files)}")
        for item in corrupted_files[:5]:
            print(f"     {item['file']}: {item['error']}")
        if len(corrupted_files) > 5:
            print(f"     ... 还有 {len(corrupted_files)-5} 个")
    else:
        print(f"  ✅ 全部正常")
    
    return len(files), len(corrupted_files), corrupted_files


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='多线程全量检查图像')
    parser.add_argument('--threads', type=int, default=64,
                       help='线程数（默认64，NVMe可以更高）')
    parser.add_argument('--output', default='F:/DATASATES/corrupted_images_full_report.json',
                       help='输出报告JSON路径')
    
    args = parser.parse_args()
    
    print("="*60)
    print("多线程全量图像检查")
    print("="*60)
    print(f"线程数: {args.threads}")
    
    all_corrupted = {}
    total_checked = 0
    total_corrupted = 0
    
    # 1. 检查D:/UBB_train（多输入训练集）
    print(f"\n{'='*60}")
    print("D:/UBB_train（多输入训练集）")
    print(f"{'='*60}")
    
    train_folders = ['gt', 'depth',
                    'color_B_1', 'color_B_2', 'color_B_3',
                    'color_BG_1', 'color_BG_2', 'color_BG_3',
                    'color_G_1', 'color_G_2', 'color_G_3',
                    'color_Y_1', 'color_Y_2', 'color_Y_3',
                    'color_YG_1', 'color_YG_2', 'color_YG_3']
    
    for folder in train_folders:
        folder_path = os.path.join('D:/UBB_train', folder)
        total, corrupted, corrupted_list = check_folder_parallel(
            folder_path, folder, args.threads
        )
        total_checked += total
        total_corrupted += corrupted
        
        if corrupted > 0:
            all_corrupted[f'train_{folder}'] = corrupted_list
    
    # 2. 检查DATA/validation/UBB-M_reference（有参考验证集）
    print(f"\n{'='*60}")
    print("DATA/validation/UBB-M_reference（有参考验证集）")
    print(f"{'='*60}")
    
    for folder in ['input', 'gt']:
        folder_path = os.path.join('DATA/validation/UBB-M_reference', folder)
        total, corrupted, corrupted_list = check_folder_parallel(
            folder_path, folder, args.threads
        )
        total_checked += total
        total_corrupted += corrupted
        
        if corrupted > 0:
            all_corrupted[f'val_ref_{folder}'] = corrupted_list
    
    # 总结
    print(f"\n{'='*60}")
    print("检查总结")
    print(f"{'='*60}")
    print(f"总检查: {total_checked} 个文件")
    print(f"损坏: {total_corrupted} 个文件")
    print(f"正常率: {(total_checked-total_corrupted)/total_checked*100:.2f}%")
    
    if all_corrupted:
        print(f"\n❌ 发现损坏文件！")
        
        # 保存详细报告
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_corrupted, f, ensure_ascii=False, indent=2)
        
        print(f"\n详细报告已保存: {args.output}")
        
        # 生成删除脚本
        delete_script = args.output.replace('.json', '_delete.txt')
        with open(delete_script, 'w', encoding='utf-8') as f:
            for dataset, files in all_corrupted.items():
                f.write(f"# {dataset}\n")
                for item in files:
                    f.write(f"{item['path']}\n")
        
        print(f"损坏文件列表: {delete_script}")
        
        return 1
    else:
        print("✅ 所有图像均完好！")
        return 0


if __name__ == '__main__':
    sys.exit(main())




