# ⚡ UBB数据集处理 - 最终执行顺序

## 🔴 重要：执行顺序不能错！

特别注意 **步骤3和步骤4的顺序**：
- ✅ **先**提取深度验证集（从训练集depth复制）
- ✅ **后**从训练集排除验证集（删除depth文件）

如果顺序反了，深度验证集就无法获取depth GT！

---

## 📝 完整执行清单

### ✅ 步骤0: 原始数据整理（已完成）
```bash
python scripts/organize_ubb_raw.py \
  --source "DATA/UBB" \
  --target "F:/DATASATES/UBBraw"
```
**状态**: ✅ 完成  
**输出**: `F:/DATASATES/UBBraw/` (按场景分离，15,356样本)

---

### 🔄 步骤1: 训练集整理（后台运行中）
```bash
python -u scripts/prepare_ubb_dataset.py \
  --source "DATA/UBB" \
  --target "F:/DATASATES/UBB_train"
```
**状态**: 🔄 后台运行  
**输出**: `F:/DATASATES/UBB_train/` (合并场景，15,356样本)

**检查进度**:
```bash
python -c "import os; path='F:/DATASATES/UBB_train/gt'; count=len([f for f in os.listdir(path) if f.endswith('.png')]) if os.path.exists(path) else 0; print(f'进度: {count}/15356 ({count/15356*100:.1f}%)')"
```

---

### ⏸️ 步骤2: 重命名验证集（等待步骤1完成）
```bash
python scripts/prepare_validation_sets.py \
  --source-ref "DATA/validation/UBB-M_reference" \
  --source-noref "DATA/validation/UBB-M_noreference" \
  --target-ref "F:/DATASATES/UBB_validation_reference" \
  --target-noref "F:/DATASATES/UBB_validation_noreference" \
  --basename-output "F:/DATASATES/UBB_validation_basenames.json"
```

**输出**:
- `F:/DATASATES/UBB_validation_reference/` (input + gt)
- `F:/DATASATES/UBB_validation_noreference/` (input only)
- `F:/DATASATES/UBB_validation_basenames.json` (排除列表)

**验证**:
```bash
python scripts/prepare_validation_sets.py --validate-only
```

---

### ⏸️ 步骤3: 提取深度验证集（❗必须在步骤4之前）
```bash
python scripts/prepare_depth_validation.py \
  --val-input "F:/DATASATES/UBB_validation_reference/input" \
  --train-depth "F:/DATASATES/UBB_train/depth" \
  --target "F:/DATASATES/UBB_depth_validation"
```

**作用**: 从训练集depth **复制** depth GT到深度验证集

**输出**:
- `F:/DATASATES/UBB_depth_validation/rgb/` (~20,040个)
- `F:/DATASATES/UBB_depth_validation/depth/` (~20,040个)

**⚠️ 重要**: 此步骤必须在步骤4之前执行！

---

### ⏸️ 步骤4: 从训练集排除验证集（❗在步骤3之后）
```bash
# 先测试
python scripts/exclude_validation_from_train.py \
  --train-root "F:/DATASATES/UBB_train" \
  --dry-run

# 再执行
python scripts/exclude_validation_from_train.py \
  --train-root "F:/DATASATES/UBB_train"
```

**作用**: 从训练集的17个文件夹（**包括depth/**）中删除验证样本

**效果**:
- 删除: ~1,336样本 × 17组 = ~22,712个文件
- 剩余: ~14,020样本

**⚠️ 警告**: 脚本会检查深度验证集是否已提取，如未提取会提示确认

---

### ⏸️ 步骤5: 验证最终结果
```bash
# 验证训练集
python scripts/verify_reorganization.py "F:/DATASATES/UBB_train"

# 验证深度验证集配对
python -c "import os; rgb=len([f for f in os.listdir('F:/DATASATES/UBB_depth_validation/rgb') if f.endswith('.png')]) if os.path.exists('F:/DATASATES/UBB_depth_validation/rgb') else 0; depth=len([f for f in os.listdir('F:/DATASATES/UBB_depth_validation/depth') if f.endswith('.png')]) if os.path.exists('F:/DATASATES/UBB_depth_validation/depth') else 0; print(f'RGB: {rgb}, Depth: {depth}, Paired: {min(rgb, depth)}')"
```

---

### ⏸️ 步骤6: 更新训练配置

编辑 `configs/train.yaml`:

```yaml
data:
  train_root: "F:/DATASATES/UBB_train"  # ← 更新

validation_sets:
  # RGB有参考验证
  ubb_reference:
    name: "UBB_Reference"
    type: "enhancement_with_reference"
    data_root: "F:/DATASATES/UBB_validation_reference"
    folder_structure:
      input: "input"
      gt: "gt"
  
  # RGB无参考验证
  ubb_noreference:
    name: "UBB_NoReference"
    type: "enhancement_no_reference"
    data_root: "F:/DATASATES/UBB_validation_noreference"
    folder_structure:
      input: "input"
  
  # 🔥 深度验证
  ubb_depth:
    name: "UBB_Depth"
    type: "depth_prediction"
    data_root: "F:/DATASATES/UBB_depth_validation"
    folder_structure:
      rgb: "rgb"
      depth: "depth"
    depth_format: "png"
```

---

## 🎯 关键点总结

### ✅ 正确顺序
```
步骤2 (重命名验证集)
  ↓
步骤3 (提取深度验证集) ← 从训练集depth复制
  ↓
步骤4 (排除验证集) ← 从训练集depth删除
```

### ❌ 错误顺序（会导致深度验证集无depth GT）
```
步骤2 (重命名验证集)
  ↓
步骤4 (排除验证集) ← 训练集depth被删除！
  ↓
步骤3 (提取深度验证集) ← 找不到depth文件！
```

---

## 📊 最终数据分布

### 验证集样本（1,336个唯一位置）
- RGB有参考: 20,040个input + 13,060个GT (1,336 × 15种退化)
- RGB无参考: 9,033个input
- **深度**: 20,040个RGB + 20,040个depth GT

### 训练集（排除后 ~14,020样本）
- 每个样本: 15退化 + 1GT + 1depth = 17个文件
- 总文件数: ~14,020 × 17 = ~238,340个

---

**答案**: ✅ 是的，`exclude_validation_from_train.py` 会排除depth，但必须先提取深度验证集！




