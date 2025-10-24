# UBB数据集使用指南

## 📁 数据集清单

### ✅ 完成的数据集（6个）

| 数据集 | 位置 | 样本数 | 大小 | 用途 |
|--------|------|--------|------|------|
| **多输入训练集** | `F:/DATASATES/UBB_train` | 11,098 | 180GB | 多退化训练（Folder） |
| **LMDB训练集** | `F:/DATASATES/UBB_train.lmdb` | 11,098 | ~350GB | 多退化训练（加速）🔄 |
| **单输入训练集** | `D:/UBB_train_single_input` | 166,470对 | ~300GB | 单输入任务 🔄 |
| **原始归档** | `F:/DATASATES/UBBraw` | 15,356 | 240GB | 备份调试 |
| **有参考验证** | `DATA/validation/UBB-M_reference` | 1,500 | 2GB | RGB评估 |
| **无参考验证** | `DATA/validation/UBB-M_noreference` | 1,500 | 2GB | RGB评估 |
| **深度验证** | `DATA/validation/UBB-M_depth` | 1,500对 | 2GB | 深度评估 |

---

## 🚀 使用方法

### 方法1: Folder格式（当前可用）
```yaml
# configs/train.yaml
data:
  format: "folder"
  train_root: "DATA/UBB_train"  # 需要创建符号链接或复制
  num_workers: 16
  prefetch_factor: 4
```

### 方法2: LMDB格式（转换完成后，推荐）
```yaml
data:
  format: "lmdb"
  lmdb_paths:
    train: "F:/DATASATES/UBB_train.lmdb"
  num_workers: 32       # LMDB可以更多
  prefetch_factor: 8
```

### 方法3: 单输入格式（其他任务）
```yaml
data:
  format: "folder"
  train_root: "D:/UBB_train_single_input"
  # 使用标准单输入Dataset类
```

---

## 📋 核心脚本清单

### 数据整理脚本
- `scripts/prepare_ubb_dataset.py` - 训练集整理（合并场景）
- `scripts/organize_ubb_raw.py` - 原始数据整理（分场景）
- `scripts/prepare_validation_final.py` - 验证集规范化
- `scripts/create_depth_validation.py` - 深度验证集创建
- `scripts/subsample_by_degradation.py` - 验证集精简
- `scripts/apply_validation_subset.py` - 应用精简
- `scripts/exclude_validation_from_train.py` - 训练集排除验证集

### 格式转换脚本
- `scripts/create_lmdb.py` - 创建LMDB数据库
- `scripts/convert_to_single_input.py` - 转换单输入格式
- `scripts/merge_scenes_to_train.py` - 快速合并场景

### 验证和检查脚本
- `scripts/check_train_progress.py` - 检查训练集进度
- `scripts/check_lmdb_progress.py` - 检查LMDB进度
- `scripts/check_validation_status.py` - 检查验证集状态
- `scripts/verify_reorganization.py` - 验证整理结果

---

## 🎯 训练/验证集划分

### 数据量
- **训练集**: 11,098样本（83.1%）
- **验证集**: 1,500样本（每种退化100个）
- **无重叠**: 严格排除，避免数据泄露 ✅

### 验证集类型
1. **有参考**: Input + GT配对（PSNR, SSIM等）
2. **无参考**: Input only（UCIQE, UIQM等）
3. **深度**: RGB-Depth配对（MAE, RMSE等）

---

## ⚙️ 配置要点

### 深度验证统一配置
```yaml
# 所有深度验证集使用
depth_gt_units: "pixel"      # 16位PNG像素值
depth_valid_min: 100.0       # 避免噪声
depth_valid_max: 65535.0     # 16位最大值
depth_format: "png"          # PNG格式
```

### num_workers优化
| 存储 | 格式 | 推荐workers |
|------|------|------------|
| HDD | Folder | 4-8 |
| HDD | LMDB | 8-16 |
| SSD | Folder | 16-32 |
| SSD | LMDB | 32-64 |

---

## 📝 文件命名规范

### 训练集（多输入）
- 格式: `s{scene}_{core}.png`
- 示例: `s1_cam_dx+0.00_dy+0.50_yaw000.png`
- 场景: s1(24mm), s2, s3, s4(85mm)

### 验证集（单输入）
- 格式: `s{scene}__{degradation}__{core}.png`
- 示例: `s1__B1__cam_dx+0.00_dy+0.50_yaw000.png`
- 退化: B1, B2, B3, GB1, ..., YG3

### 为什么不同？
- **训练**: 多退化输入，无需区分退化类型
- **推理**: 单退化输入，需要退化前缀作为唯一标识

---

## 🎯 下一步

### 转换完成后
1. ✅ 检查LMDB完整性
2. ✅ 检查单输入训练集完整性
3. ✅ 更新train.yaml选择使用LMDB或Folder
4. 🚀 开始训练

### 训练命令
```bash
# 使用Folder格式
python scripts/train.py --config configs/train.yaml

# 使用LMDB格式（修改配置后）
python scripts/train.py --config configs/train.yaml
```

---

**所有数据集准备完成！详细文档见各脚本的md文件。** 🎉




