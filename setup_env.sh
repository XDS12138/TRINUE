#!/bin/bash

# 设置代理环境变量
export https_proxy=http://127.0.0.1:7897 
export http_proxy=http://127.0.0.1:7897 
export all_proxy=socks5://127.0.0.1:7897

echo "已设置代理环境变量:"
echo "https_proxy=$https_proxy"
echo "http_proxy=$http_proxy"
echo "all_proxy=$all_proxy"

# 检查环境是否已存在，如果存在则移除
CONDA_ENV_NAME="trinue"
if conda env list | grep -q "$CONDA_ENV_NAME"; then
    echo "发现已存在的 $CONDA_ENV_NAME 环境，正在移除..."
    conda env remove -n "$CONDA_ENV_NAME"
fi

# 创建新的环境
echo "正在创建 $CONDA_ENV_NAME 环境..."
conda env create -f environment.yml

# 激活环境
echo "激活 $CONDA_ENV_NAME 环境..."
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

# 验证环境激活
if [[ "$CONDA_DEFAULT_ENV" == "$CONDA_ENV_NAME" ]]; then
    echo "环境 $CONDA_ENV_NAME 已成功激活！"
    python -c "import torch; print('PyTorch 版本:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available())"
else
    echo "环境激活失败，请手动执行: conda activate $CONDA_ENV_NAME"
fi

# 检查训练配置
echo "检查训练配置..."
if [ -f "configs/train.yaml" ]; then
    echo "训练配置文件存在，现在检查数据路径是否正确..."
    python -c "
import yaml
import os

# 读取配置文件
with open('configs/train.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 检查数据目录是否存在
train_path = config['data']['train_root']
val_path = config['data']['val_root']
print(f'训练数据路径: {train_path}')
print(f'验证数据路径: {val_path}')

# 检查文件夹是否存在
train_exists = os.path.exists(train_path)
val_exists = os.path.exists(val_path)
print(f'训练数据目录存在: {train_exists}')
print(f'验证数据目录存在: {val_exists}')

# 检查文件夹结构
if train_exists and val_exists:
    gt_folder = config['data']['folder_structure']['gt']
    depth_folder = config['data']['folder_structure']['depth']
    degradation_folders = config['data']['degradation_folders']
    
    # 检查训练集文件夹结构
    train_gt_path = os.path.join(train_path, gt_folder)
    train_depth_path = os.path.join(train_path, depth_folder)
    train_deg_paths = [os.path.join(train_path, deg) for deg in degradation_folders]
    
    print(f'训练目标图像目录存在: {os.path.exists(train_gt_path)}')
    print(f'训练深度图目录存在: {os.path.exists(train_depth_path)}')
    for i, path in enumerate(train_deg_paths):
        print(f'训练退化图像目录 {degradation_folders[i]} 存在: {os.path.exists(path)}')
    
    # 检查验证集文件夹结构
    val_gt_path = os.path.join(val_path, gt_folder)
    val_depth_path = os.path.join(val_path, depth_folder)
    val_deg_paths = [os.path.join(val_path, deg) for deg in degradation_folders]
    
    print(f'验证目标图像目录存在: {os.path.exists(val_gt_path)}')
    print(f'验证深度图目录存在: {os.path.exists(val_depth_path)}')
    for i, path in enumerate(val_deg_paths):
        print(f'验证退化图像目录 {degradation_folders[i]} 存在: {os.path.exists(path)}')
    
    # 检查模型配置
    print('\\n模型配置:')
    print(f'基础通道数: {config[\"model\"][\"base_channels\"]}')
    print(f'层级数: {config[\"model\"][\"levels\"]}')
    print(f'注意力头数: {config[\"model\"][\"heads\"]}')
    print(f'Bottleneck块数: {config[\"model\"][\"bottleneck_blocks\"]}')
    
    # 检查训练配置
    print('\\n训练配置:')
    print(f'总轮次: {config[\"train\"][\"epochs\"]}')
    print(f'批次大小: {config[\"data\"][\"batch_size\"]}')
    print(f'学习率: {config[\"optimizer\"][\"lr\"]}')
    print(f'使用GPU: {config[\"gpu\"][\"use_gpu\"]}')
    print(f'分布式训练: {config[\"gpu\"][\"distributed\"]}')
    
else:
    print('警告: 数据目录不存在，请检查配置或创建相应目录')
"
else
    echo "警告: 训练配置文件不存在，请检查 configs/train.yaml"
fi

echo "设置完成！您可以使用以下命令重新激活环境:"
echo "conda activate $CONDA_ENV_NAME" 