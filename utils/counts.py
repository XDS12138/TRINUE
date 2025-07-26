import os
from PIL import Image

# ✅ 指定图像目录（可改为你想分析的目录）
TARGET_DIR = "/media/xxx/233-3/seatru"  # Windows 示例
# TARGET_DIR = "/home/yourname/images"  # Linux 示例

# 支持的图像格式
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)

def calculate_average_resolution(root_dir):
    total_width, total_height, count = 0, 0, 0

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if is_image_file(filename):
                filepath = os.path.join(dirpath, filename)
                try:
                    with Image.open(filepath) as img:
                        width, height = img.size
                        total_width += width
                        total_height += height
                        count += 1
                except Exception as e:
                    print(f"无法读取图像 {filepath}：{e}")

    if count == 0:
        print("没有找到图像文件。")
        return

    avg_width = total_width / count
    avg_height = total_height / count
    print(f"共统计 {count} 张图像")
    print(f"平均分辨率为：{avg_width:.2f} x {avg_height:.2f} 像素")

if __name__ == "__main__":
    calculate_average_resolution(TARGET_DIR)
