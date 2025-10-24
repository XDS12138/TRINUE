# UBB数据集处理执行清单

## ✅ 已完成

- [x] 编写所有6个核心脚本
- [x] 验证配对逻辑（15,356样本全部配对）
- [x] 整理原始数据到UBBraw（按场景分离）
- [x] 启动训练集整理（后台运行中）

## ⏸️ 等待中

- [ ] **训练集整理完成** (`prepare_ubb_dataset.py` 后台运行)
  - 预计耗时: 15-30分钟
  - 检查命令: 
    ```bash
    python -c "import os; gt_count=len([f for f in os.listdir('F:/DATASATES/UBB_train/gt') if f.endswith('.png')]) if os.path.exists('F:/DATASATES/UBB_train/gt') else 0; print(f'进度: {gt_count}/15356 ({gt_count/15356*100:.1f}%)')"
    ```

## 📝 待执行步骤（按顺序）

### 步骤1: 重命名验证集 ⏭️
```bash
python scripts/prepare_validation_sets.py \
  --source-ref "DATA/validation/UBB-M_reference" \
  --source-noref "DATA/validation/UBB-M_noreference" \
  --target-ref "F:/DATASATES/UBB_validation_reference" \
  --target-noref "F:/DATASATES/UBB_validation_noreference" \
  --basename-output "F:/DATASATES/UBB_validation_basenames.json"
```

**作用**:
- 规范化验证集文件名
- 生成basename列表用于训练集排除

**预期输出**:
- 有参考: 20,040个input + 13,060个GT
- 无参考: 9,033个input

---

### 步骤2: 提取深度验证集 ⏭️
```bash
python scripts/prepare_depth_validation.py \
  --val-input "F:/DATASATES/UBB_validation_reference/input" \
  --train-depth "F:/DATASATES/UBB_train/depth" \
  --target "F:/DATASATES/UBB_depth_validation"
```

**作用**:
- RGB: 从验证集input复制
- Depth GT: 从训练集depth提取对应文件

**预期输出**:
- rgb/: 20,040个文件
- depth/: ~20,040个文件（匹配率应>99%）

**⚠️ 重要**: 此步骤必须在步骤3之前执行！否则训练集depth被删除后无法提取。

---

### 步骤3: 从训练集排除验证集 ⏭️
```bash
python scripts/exclude_validation_from_train.py \
  --train-root "F:/DATASATES/UBB_train" \
  --val-basename-json "F:/DATASATES/UBB_validation_basenames.json" \
  --stats-output "F:/DATASATES/exclusion_stats.json"
```

**作用**:
- 删除训练集中与验证集重叠的样本
- 避免数据泄露

**预期效果**:
- 删除: ~1,336个样本 × 17组 = ~22,712个文件
- 剩余训练样本: ~14,020个

---

### 步骤4: 验证最终结果 ⏭️
```bash
# 验证训练集
python scripts/verify_reorganization.py "F:/DATASATES/UBB_train"

# 检查配对（应该约14,020样本）
python -c "import os; folders=['gt','depth','color_B_1']; [(print(f'{f}: {len([x for x in os.listdir(f\"F:/DATASATES/UBB_train/{f}\") if x.endswith(\".png\")])}')) for f in folders]"
```

---

### 步骤5: 更新训练配置 ⏭️

编辑 `configs/train.yaml`:

```yaml
data:
  train_root: "F:/DATASATES/UBB_train"  # ← 更新训练集路径

validation_sets:
  # 🔥 UBB有参考验证集
  ubb_reference:
    name: "UBB_Reference"
    type: "enhancement_with_reference"
    data_root: "F:/DATASATES/UBB_validation_reference"
    enabled: true
    description: "UBB有参考验证集 - 全参考指标评估"
    metrics:
      - "psnr"
      - "ssim"
      - "ciede2000"
      - "lpips"
    save_images: 10
    folder_structure:
      input: "input"
      gt: "gt"
  
  # 🔥 UBB无参考验证集
  ubb_noreference:
    name: "UBB_NoReference"
    type: "enhancement_no_reference"
    data_root: "F:/DATASATES/UBB_validation_noreference"
    enabled: true
    description: "UBB无参考验证集 - 无参考质量评估"
    metrics:
      - "uciqe"
      - "uiqm"
      - "niqe"
    save_images: 10
    folder_structure:
      input: "input"
  
  # 🔥 UBB深度验证集
  ubb_depth:
    name: "UBB_Depth"
    type: "depth_prediction"
    data_root: "F:/DATASATES/UBB_depth_validation"
    enabled: true
    description: "UBB深度预测验证集"
    depth_gt_units: "relative"
    depth_valid_min: 0.1
    depth_valid_max: 100.0
    pred_clamp_val: 80.0
    metrics:
      - "depth_mae"
      - "depth_rmse"
      - "depth_abs_rel"
      - "depth_delta1"
    save_images: 10
    folder_structure:
      rgb: "rgb"
      depth: "depth"
    depth_format: "png"
```

---

## 🎯 完整流程总结

```
原始数据 (DATA/UBB) 
    ↓
┌───────────────────────────────────────┐
│ prepare_ubb_dataset.py                │
│ ├─ 规范化命名                         │
│ ├─ 添加场景前缀 (s1_, s2_, s3_, s4_) │
│ └─ 合并到17个文件夹                   │
└───────────────────────────────────────┘
    ↓
F:/DATASATES/UBB_train (15,356样本)
    ↓
┌───────────────────────────────────────┐
│ prepare_validation_sets.py            │
│ ├─ 从UBB-M提取验证集                  │
│ └─ 规范化命名                         │
└───────────────────────────────────────┘
    ↓
验证集 (1,336样本)
├─ UBB_validation_reference (有参考)
└─ UBB_validation_noreference (无参考)
    ↓
┌───────────────────────────────────────┐
│ prepare_depth_validation.py           │
│ ├─ rgb: 从验证集input复制             │
│ └─ depth: 从训练集depth提取           │
└───────────────────────────────────────┘
    ↓
UBB_depth_validation (深度验证集)
    ↓
┌───────────────────────────────────────┐
│ exclude_validation_from_train.py      │
│ └─ 从训练集删除验证样本               │
└───────────────────────────────────────┘
    ↓
最终训练集 (~14,020样本，无数据泄露)
```

---

## 📂 最终目录结构

```
F:/DATASATES/
├── UBBraw/                        # 原始数据归档（按场景）
│   ├── scene_1/ (4,200样本 × 17组 + JSON)
│   ├── scene_2/ (2,268样本 × 18组 + JSON)
│   ├── scene_3/ (2,688样本 × 18组 + JSON)
│   └── scene_4/ (4,200样本 × 18组 + JSON)
│
├── UBB_train/                     # 训练集（合并场景）
│   ├── gt/ (~14,020个，带场景前缀)
│   ├── depth/
│   └── color_B_1/ ... color_YG_3/ (17个文件夹)
│
├── UBB_validation_reference/      # 有参考验证集
│   ├── input/ (20,040个 = 1,336×15)
│   └── gt/ (13,060个)
│
├── UBB_validation_noreference/    # 无参考验证集
│   └── input/ (9,033个)
│
└── UBB_depth_validation/          # 深度验证集
    ├── rgb/ (20,040个)
    └── depth/ (~20,040个)
```

---

## ⚡ 快速检查命令

```bash
# 检查训练集整理进度
python -c "import os; path='F:/DATASATES/UBB_train/gt'; count=len([f for f in os.listdir(path) if f.endswith('.png')]) if os.path.exists(path) else 0; print(f'{count}/15356 ({count/15356*100:.1f}%)')"

# 检查原始归档
python scripts/organize_ubb_raw.py --target "F:/DATASATES/UBBraw" --validate-only

# 检查验证集重命名
python scripts/prepare_validation_sets.py --validate-only

# 检查排除效果（干运行）
python scripts/exclude_validation_from_train.py --train-root "F:/DATASATES/UBB_train" --dry-run
```

---

**所有脚本已准备就绪！** 🎉

等待 `prepare_ubb_dataset.py` 完成后，按顺序执行步骤1-6即可。

