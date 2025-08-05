# 多验证集系统使用指南

## 📖 概述

我们的训练系统现在支持同时在多个验证集上进行评估，包括：

- **全参考图像增强验证集**：有清晰GT的数据集（PSNR, SSIM, CIEDE2000, LPIPS）
- **无参考图像增强验证集**：真实场景数据集（UCIQE, UIQM, NIQE）  
- **深度预测验证集**：有深度GT的数据集（MAE, RMSE, δ准确率等）

## 🚀 快速开始

### 1. 配置验证集

在`configs/train.yaml`中添加验证集配置：

```yaml
validation_sets:
  # 📸 全参考验证集
  uieb_test:
    name: "UIEB_Test"
    type: "enhancement_with_reference"
    data_root: "DATA/color_val/UIEB_test"  # 🔧 实际路径
    enabled: true
    description: "UIEB测试集"
    metrics:
      - "psnr"
      - "ssim"
      - "ciede2000"
      - "lpips"
    save_images: 5
    folder_structure:  # 🔧 指定实际的文件夹名称
      input: "input"   # 输入图像文件夹名
      gt: "gt"         # GT图像文件夹名
    
  # 🏔️ 深度预测验证集  
  seathru_depth:
    name: "SeaThru_Depth"
    type: "depth_prediction"
    data_root: "DATA/train/depth_val/SeaThru_D3_D5_TestSet/test"
    enabled: true
    description: "SeaThru深度预测数据集"
    metrics:
      - "depth_mae"
      - "depth_rmse"
      - "depth_abs_rel"
      - "depth_delta1"
    save_images: 5
    folder_structure:
      rgb: "rgb"
      depth: "depth"
    depth_format: "npy"  # 🔧 深度文件格式
```

### 2. 数据目录结构

确保您的验证集目录结构正确（基于实际DATA结构）：

#### 📸 全参考验证集
```
DATA/color_val/UIEB_test/
├── input/                # 输入退化图像
└── gt/                   # GT图像
```

#### 🌊 无参考验证集  
```
DATA/color_val/SQUID/
└── input/                # 输入图像（无需gt文件夹）
```

#### 🏔️ 深度预测验证集
```
DATA/train/depth_val/SeaThru_D3_D5_TestSet/test/
├── rgb/                  # RGB图像
└── depth/                # 深度文件（.npy格式）
```

#### 📁 完整的DATA目录结构
```
DATA/
├── color_val/            # 图像增强验证集
│   ├── UIEB_test/
│   │   ├── input/        # 输入图像  
│   │   └── gt/           # GT图像
│   ├── LSUI_test/
│   │   ├── input/
│   │   └── gt/
│   ├── UIEB_challenging/
│   │   ├── input/
│   │   └── gt/
│   └── SQUID/            # 可作为无参考验证集
├── depth_val/            # （空目录）
└── train/
    └── depth_val/        # 深度预测验证集
        ├── SeaThru_D3_D5_TestSet/test/
        │   ├── rgb/      # D3_T_Sxxxxx.png, D5_LFT_xxxx.png
        │   └── depth/    # 对应的.npy文件
        └── SQUID_Monocular_Left_Camera_TestSet/test/
            ├── rgb/
            └── depth/
```

### 3. 运行训练

```bash
# 正常训练（自动进行多验证集验证）
python scripts/train_refactored.py --config configs/train.yaml

# 仅验证模式
python scripts/train_refactored.py --config configs/train.yaml --eval_only
```

## 📊 结果保存

### 目录结构
训练完成后，结果将保存在以下目录结构中：

```
experiments/train/underwater_enhance_run_YYYYMMDD_HHMMSS/
├── validation_results/
│   ├── epoch_001/
│   │   ├── UIEB_Test/
│   │   │   ├── comparison_000.png      # 对比图（输入-增强-GT）
│   │   │   ├── comparison_001.png
│   │   │   ├── ...
│   │   │   ├── input_000.png           # 单独的输入图像
│   │   │   ├── enhanced_000.png        # 单独的增强图像
│   │   │   ├── gt_000.png              # 单独的GT图像
│   │   │   └── metrics.csv             # 该验证集的详细指标
│   │   ├── Real_Underwater/
│   │   │   ├── comparison_000.png
│   │   │   ├── ...
│   │   │   └── metrics.csv
│   │   └── ...
│   ├── epoch_002/
│   └── ...
├── comprehensive_validation_results.csv    # 所有验证集的综合结果
├── metrics.csv                             # 传统训练指标
├── tensorboard/                            # TensorBoard日志
└── checkpoints/                            # 模型检查点
```

### CSV结果文件

#### 1. 综合结果文件 (`comprehensive_validation_results.csv`)
包含所有验证集在每个epoch的指标：

```csv
epoch,UIEB_Test_psnr,UIEB_Test_ssim,Real_Underwater_uciqe,Real_Underwater_uiqm,...
1,24.56,0.856,0.524,1.234,...
2,25.12,0.867,0.534,1.245,...
...
```

#### 2. 单验证集结果文件 (`validation_results/epoch_XXX/ValidationSetName/metrics.csv`)
每个验证集的详细指标：

```csv
epoch,psnr,ssim,ciede2000,lpips
1,24.56,0.856,12.34,0.123
2,25.12,0.867,11.89,0.115
...
```

## 📈 TensorBoard可视化

启动TensorBoard查看训练过程：

```bash
tensorboard --logdir experiments/train/underwater_enhance_run_YYYYMMDD_HHMMSS/tensorboard
```

可视化内容包括：
- 各验证集的指标曲线
- 对比图像（输入-增强-GT）
- 训练损失和学习率
- 模型参数分布

## 🔧 高级配置

### 自定义指标

支持的指标类型：

#### 全参考指标
- `psnr`: 峰值信噪比
- `ssim`: 结构相似性指数
- `ciede2000`: CIE Delta E 2000色彩差异
- `lpips`: 学习感知图像补丁相似性

#### 无参考指标
- `uciqe`: 水下图像色彩增强指标
- `uiqm`: 水下图像质量度量
- `niqe`: 自然图像质量评价器

#### 深度估计指标
- `depth_mae`: 平均绝对误差
- `depth_rmse`: 均方根误差
- `depth_abs_rel`: 相对绝对误差
- `depth_sq_rel`: 相对平方误差
- `depth_delta1`: δ1准确率（阈值1.25）
- `depth_delta2`: δ2准确率（阈值1.25²）
- `depth_delta3`: δ3准确率（阈值1.25³）

### 可视化配置

```yaml
visualization:
  multi_validation:
    enabled: true                    # 启用多验证集验证
    save_detailed_results: true      # 保存详细CSV结果
    tensorboard_logging: true        # TensorBoard记录
    comparison_layout: "horizontal"   # 对比图布局：horizontal/vertical
    save_individual_metrics: true    # 为每个验证集保存单独指标文件
```

## 📝 添加新验证集

### 步骤1：准备数据
按照上述目录结构准备您的验证集数据。

### 步骤2：配置验证集
在`configs/train.yaml`中添加新的验证集配置：

```yaml
validation_sets:
  your_new_dataset:
    name: "Your_Dataset_Name"
    type: "enhancement_with_reference"  # 或其他类型
    data_root: "path/to/your/dataset"
    enabled: true
    description: "您的数据集描述"
    metrics:
      - "psnr"
      - "ssim"
      # 根据验证集类型添加合适的指标
    save_images: 5
```

### 步骤3：验证配置
运行验证确保配置正确：

```bash
python scripts/train_refactored.py --config configs/train.yaml --eval_only
```

## 🔄 向后兼容性

系统完全向后兼容：
- 如果没有配置`validation_sets`，将使用传统的`val_root`配置
- 传统验证和多验证集验证可以同时进行
- 现有的训练脚本无需修改即可使用

## 📋 依赖安装

为了使用所有指标，建议安装以下依赖：

```bash
# 基础依赖
pip install opencv-python scikit-image

# LPIPS支持
pip install lpips

# NIQE支持（可选）
pip install pyiqa

# CIEDE2000支持（可选）
pip install colorspacious
```

## 🐛 故障排除

### 常见问题

1. **验证集路径不存在**
   - 检查`data_root`路径是否正确
   - 确保目录结构符合要求

2. **指标计算失败**
   - 检查相关依赖是否安装
   - 查看日志中的详细错误信息

3. **图像尺寸不匹配**
   - 确保输入图像、GT图像和深度图尺寸一致
   - 系统会自动调整尺寸，但可能影响精度

4. **内存不足**
   - 减少`batch_size`
   - 减少`save_images`数量
   - 关闭不必要的验证集

### 日志检查
查看详细日志以排查问题：

```bash
# 查看验证相关日志
tail -f experiments/train/underwater_enhance_run_*/logs/validation.log

# 查看错误日志
tail -f experiments/train/underwater_enhance_run_*/logs/error.log
```

## 💡 最佳实践

1. **验证集选择**：选择有代表性的验证集，覆盖不同的图像条件
2. **指标选择**：根据应用场景选择合适的指标组合
3. **保存数量**：`save_images`设置为5-10张即可，避免存储空间浪费
4. **定期检查**：定期检查验证结果，确保模型没有过拟合
5. **结果分析**：结合不同验证集的结果进行综合分析

## 📚 参考资料

- [PSNR和SSIM详解](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)
- [LPIPS论文](https://richzhang.github.io/PerceptualSimilarity/)
- [UCIQE和UIQM详解](https://ieeexplore.ieee.org/document/7300447)
- [深度估计指标说明](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) 