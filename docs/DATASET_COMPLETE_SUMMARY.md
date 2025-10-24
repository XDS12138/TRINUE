# UBB数据集完整处理总结

## ✅ 已完成的数据集

### 1. 训练集（多输入格式）- `F:/DATASATES/UBB_train`
- **样本数**: 11,098个
- **文件数**: 188,666个（11,098 × 17组）
- **结构**: 17个文件夹（15退化 + gt + depth）
- **文件名**: `s1_cam_dx+0.00_dy+0.50_yaw000.png`（场景前缀）
- **尺寸**: 960×540（70%）和1920×1080（30%）混合
- **用途**: 多退化输入训练
- **状态**: ✅ 完成

### 2. 原始数据归档 - `F:/DATASATES/UBBraw`
- **样本数**: 15,356个
- **结构**: 按场景分离（scene_1/2/3/4）
- **文件名**: `cam_dx+0.00_dy+0.50_yaw000.png`（规范化）
- **包含**: metadata.json
- **用途**: 原始数据备份
- **状态**: ✅ 完成

### 3. 有参考验证集 - `DATA/validation/UBB-M_reference`
- **样本数**: 1,500个（100个/退化 × 15种）
- **Input**: 1,500个
- **GT**: 1,500个
- **文件名**: `s1__B1__cam_dx+0.00_dy+0.50_yaw000.png`（保留退化前缀）
- **用途**: PSNR/SSIM/LPIPS/CIEDE2000评估
- **状态**: ✅ 完成

### 4. 无参考验证集 - `DATA/validation/UBB-M_noreference`
- **样本数**: 1,500个
- **Input**: 1,500个
- **文件名**: `s1__B1__cam_dx+0.00_dy+0.50_yaw000.png`
- **用途**: UCIQE/UIQM/NIQE评估
- **状态**: ✅ 完成

### 5. 深度验证集 - `DATA/validation/UBB-M_depth`
- **样本数**: 1,500对
- **Input**: 1,500个（RGB）
- **Depth**: 1,500个（深度GT）
- **文件名**: `s1__B1__cam_dx+0.00_dy+0.50_yaw000.png`
- **用途**: 深度预测评估
- **状态**: ✅ 完成

---

## 🔄 进行中的数据集

### 6. LMDB训练集 - `F:/DATASATES/UBB_train.lmdb`
- **源**: UBB_train（多输入格式）
- **样本数**: 11,098个
- **格式**: LMDB数据库（内存映射）
- **预计大小**: 300-400GB
- **容量限制**: 1TB
- **用途**: 加速训练（2-5倍）
- **状态**: 🔄 转换中（~10%，预计1.5小时完成）

### 7. 单输入训练集 - `D:/UBB_train_single_input`
- **源**: UBB_train（多输入格式）
- **样本数**: 166,470对（11,098 × 15种退化）
- **Input**: 166,470个（15个退化文件夹合并）
- **GT**: 166,470个（GT复制15份）
- **预计大小**: 300GB
- **文件名**: `s1__B1__cam_dx+0.00_dy+0.50_yaw000.png`
- **用途**: 其他单输入任务训练
- **状态**: 🔄 转换中（~7.4%，预计1小时完成）

---

## 📊 数据集对比

| 数据集 | 格式 | 样本数 | 大小 | 位置 | 用途 |
|--------|------|--------|------|------|------|
| UBB_train | Folder多输入 | 11,098 | 180GB | F盘 | 多退化训练 |
| UBB_train.lmdb | LMDB多输入 | 11,098 | ~350GB | F盘 | 多退化训练（加速）|
| UBB_train_single | Folder单输入 | 166,470 | ~300GB | D盘 | 单输入任务训练 |
| UBBraw | Folder归档 | 15,356 | 240GB | F盘 | 备份 |
| 验证集 × 3 | Folder | 1,500 | ~3GB | DATA | 评估 |

---

## 🎯 配置文件使用

### 多输入训练（Folder格式）
```yaml
data:
  format: "folder"
  train_root: "DATA/UBB_train"  # 或 "F:/DATASATES/UBB_train"
```

### 多输入训练（LMDB格式，推荐）
```yaml
data:
  format: "lmdb"
  lmdb_paths:
    train: "F:/DATASATES/UBB_train.lmdb"
  num_workers: 32       # LMDB可以增加
  prefetch_factor: 4
```

### 单输入训练（其他任务）
```yaml
data:
  format: "folder"
  train_root: "D:/UBB_train_single_input"
  # 使用标准的单输入数据集类
```

---

## 💾 总磁盘占用

- **F盘**: 180GB(train) + 350GB(lmdb) + 240GB(raw) = **770GB**
- **D盘**: 300GB(single_input) = **300GB**
- **总计**: **1.07TB**

---

## 🚀 当前进行中

- LMDB转换: ~10%，预计1.5小时
- 单输入转换: ~7.4%，预计1小时

**两者并行运行，互不影响！** ✅




