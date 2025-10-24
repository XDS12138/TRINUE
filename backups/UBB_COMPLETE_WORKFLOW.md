# UBB数据集完整处理流程

## 📋 已完成的脚本

### 1️⃣ 训练数据整理脚本（2个版本）

#### A. `prepare_ubb_dataset.py` - 合并场景版（用于训练）
```bash
# 状态: 正在后台运行 ✅
python -u scripts/prepare_ubb_dataset.py \
  --source "DATA/UBB" \
  --target "F:/DATASATES/UBB_train"
```

**输出结构**:
```
F:/DATASATES/UBB_train/
├── gt/ (15,356个) - s1_cam_dx+0.00_dy+0.00_yaw000.png
├── depth/
├── color_B_1/ ... color_YG_3/ (共17个文件夹)
└── mapping_scene_{1-4}.csv
```

#### B. `organize_ubb_raw.py` - 按场景分离版（原始数据归档）
```bash
# 状态: 已完成 ✅
python scripts/organize_ubb_raw.py \
  --source "DATA/UBB" \
  --target "F:/DATASATES/UBBraw"
```

**输出结构**:
```
F:/DATASATES/UBBraw/
├── scene_1/ (17个文件夹 + metadata.json)
├── scene_2/ (18个文件夹 + metadata.json)
├── scene_3/ (18个文件夹 + metadata.json)
└── scene_4/ (18个文件夹 + metadata.json)
```

**已验证配对**:
- Scene 1: 4,200 ✅
- Scene 2: 2,268 ✅
- Scene 3: 2,688 ✅
- Scene 4: 4,200 ✅

---

### 2️⃣ 验证集准备脚本

#### `prepare_validation_sets.py` - 重命名验证集
```bash
# 仅分析（不执行）
python scripts/prepare_validation_sets.py --validate-only

# 执行重命名
python scripts/prepare_validation_sets.py \
  --source-ref "DATA/validation/UBB-M_reference" \
  --source-noref "DATA/validation/UBB-M_noreference" \
  --target-ref "F:/DATASATES/UBB_validation_reference" \
  --target-noref "F:/DATASATES/UBB_validation_noreference"
```

**功能**:
- 规范化文件名: `24mm__B1__cam_dx+0.00_dy+0.50_yaw000.png` → `s1_cam_dx+0.00_dy+0.50_yaw000.png`
- 场景映射: 24mm→s1, 2→s2, 3→s3, 85mm→s4
- 生成basename列表JSON（用于排除）

**统计**:
- 唯一验证样本: **1,336个**
- 总input文件: 20,040 (1,336 × 15种退化)
- 有参考GT: 13,060个
- 无参考input: 9,033个

---

### 3️⃣ 深度验证集准备脚本

#### `prepare_depth_validation.py` - 提取深度验证集
```bash
# 测试
python scripts/prepare_depth_validation.py --dry-run

# 执行
python scripts/prepare_depth_validation.py \
  --val-input "F:/DATASATES/UBB_validation_reference/input" \
  --train-depth "F:/DATASATES/UBB_train/depth" \
  --target "F:/DATASATES/UBB_depth_validation"
```

**功能**:
- RGB input: 从已重命名的验证集复制
- Depth GT: 从训练集depth文件夹提取匹配的文件
- 输出配对的RGB-Depth验证集

**输出结构**:
```
F:/DATASATES/UBB_depth_validation/
├── rgb/    (验证集input文件)
└── depth/  (对应的depth GT)
```

---

### 4️⃣ 训练集排除脚本

#### `exclude_validation_from_train.py` - 从训练集删除验证样本
```bash
# 测试（不删除）
python scripts/exclude_validation_from_train.py \
  --train-root "F:/DATASATES/UBB_train" \
  --dry-run

# 执行删除
python scripts/exclude_validation_from_train.py \
  --train-root "F:/DATASATES/UBB_train"
```

**预计效果**:
- 训练集原始: 15,356样本
- 验证集占用: ~1,336样本
- 删除后剩余: ~14,020样本 (或更少，取决于重叠)

---

## 🚀 完整执行顺序

### 第1步: 等待训练集整理完成
```bash
# 检查进度
python -c "import os; print(f'GT: {len([f for f in os.listdir(\"F:/DATASATES/UBB_train/gt\") if f.endswith(\".png\")])} / 15356')"
```

**预期**: 每个文件夹应有 **15,356** 个文件

### 第2步: 重命名验证集
```bash
python scripts/prepare_validation_sets.py \
  --source-ref "DATA/validation/UBB-M_reference" \
  --source-noref "DATA/validation/UBB-M_noreference" \
  --target-ref "F:/DATASATES/UBB_validation_reference" \
  --target-noref "F:/DATASATES/UBB_validation_noreference" \
  --basename-output "F:/DATASATES/UBB_validation_basenames.json"
```

**输出**:
- `F:/DATASATES/UBB_validation_reference/` (有参考)
- `F:/DATASATES/UBB_validation_noreference/` (无参考)
- `F:/DATASATES/UBB_validation_basenames.json` (basename列表)

### 第3步: 从训练集中排除验证集
```bash
python scripts/exclude_validation_from_train.py \
  --train-root "F:/DATASATES/UBB_train" \
  --val-basename-json "F:/DATASATES/UBB_validation_basenames.json"
```

**效果**: 删除训练集中与验证集重叠的样本

### 第4步: 验证最终结果
```bash
# 验证训练集
python scripts/prepare_ubb_dataset.py \
  --source "F:/DATASATES/UBB_train" \
  --target "F:/DATASATES/UBB_train_verified" \
  --validate-only

# 验证验证集配对
python scripts/organize_ubb_raw.py \
  --target "F:/DATASATES/UBB_validation_reference" \
  --validate-only
```

### 第5步: 更新训练配置
编辑 `configs/train.yaml`:
```yaml
data:
  train_root: "F:/DATASATES/UBB_train"  # 训练集（已排除验证集）

validation_sets:
  ubb_test:
    data_root: "F:/DATASATES/UBB_validation_reference"
  
  ubb_test_noref:
    data_root: "F:/DATASATES/UBB_validation_noreference"
    type: "enhancement_no_reference"
```

---

## 📊 最终数据集划分

### 训练集 (`F:/DATASATES/UBB_train`)
- **样本数**: ~14,020个 (排除验证集后)
- **文件数**: ~238,340个 (14,020 × 17组)
- **用途**: 模型训练

### 有参考验证集 (`F:/DATASATES/UBB_validation_reference`)
- **样本数**: 1,336个
- **input文件**: 20,040个 (1,336 × 15种退化)
- **GT文件**: 13,060个
- **用途**: 全参考指标评估 (PSNR, SSIM, LPIPS, CIEDE2000)

### 无参考验证集 (`F:/DATASATES/UBB_validation_noreference`)
- **样本数**: 1,336个
- **input文件**: 9,033个
- **用途**: 无参考指标评估 (UCIQE, UIQM, NIQE)

### 原始数据归档 (`F:/DATASATES/UBBraw`)
- **样本数**: 15,356个
- **结构**: 按场景分离，保留metadata.json
- **用途**: 原始数据备份和调试

---

## ✅ 脚本清单

| 脚本 | 状态 | 功能 |
|-----|------|------|
| `prepare_ubb_dataset.py` | ✅ 编写完成 | 训练集整理（合并场景） |
| `organize_ubb_raw.py` | ✅ 已执行 | 原始数据整理（分场景） |
| `prepare_validation_sets.py` | ✅ 编写完成 | 验证集重命名 |
| `exclude_validation_from_train.py` | ✅ 编写完成 | 训练集排除验证集 |
| `verify_reorganization.py` | ✅ 编写完成 | 验证整理结果 |

---

## 🎯 当前状态

- ✅ `F:/DATASATES/UBBraw` - 完成
- 🔄 `F:/DATASATES/UBB_train` - 整理中（后台运行）
- ⏸️ 验证集重命名 - 等待训练集完成
- ⏸️ 训练集排除 - 等待验证集重命名

---

**下一步**: 等待 `prepare_ubb_dataset.py` 完成后，按顺序执行第2-5步。

