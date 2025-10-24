# 损坏图像修复指南

## 📋 检查结果

**全量检查**: 191,667个文件
**损坏文件**: 2个
**正常率**: 99.998% ✅

---

## 🔴 需要修复的文件

### 文件1
- **路径**: `D:/UBB_train/color_BG_3/s3_cam_dx+24.00_dy-18.00_yaw315.png`
- **场景**: Scene 3
- **退化**: color_BG_3（蓝绿色-重度）
- **位置**: cam_dx+24.00_dy-18.00_yaw315
- **错误**: 无法识别图像格式

### 文件2
- **路径**: `D:/UBB_train/color_G_2/s3_cam_dx-24.00_dy-36.00_yaw225.png`
- **场景**: Scene 3
- **退化**: color_G_2（绿色-中度）
- **位置**: cam_dx-24.00_dy-36.00_yaw225
- **错误**: 无法识别图像格式

---

## 🔧 修复方案

### 方案1: 从原始数据恢复（推荐）

**从UBBraw恢复**:
```bash
# 检查UBBraw是否有这两个文件
ls F:/DATASATES/UBBraw/scene_3/GB3/*cam_dx+24.00_dy-18.00_yaw315*
ls F:/DATASATES/UBBraw/scene_3/G2/*cam_dx-24.00_dy-36.00_yaw225*

# 如果存在，复制并添加场景前缀
# （需要确认UBBraw的basename是否匹配）
```

### 方案2: 从原始UBB数据恢复

**从DATA/UBB恢复**:
```bash
# 查找Scene 3的原始文件
# DATA/UBB/3/GB3/cam_dx+24.00_dy-18.00_yaw315_rgb_0001.png
# DATA/UBB/3/G2/cam_dx-24.00_dy-36.00_yaw225_rgb_0001.png
```

### 方案3: 删除损坏文件（如果无法恢复）

```bash
# 删除这两个损坏文件
Remove-Item "D:/UBB_train/color_BG_3/s3_cam_dx+24.00_dy-18.00_yaw315.png"
Remove-Item "D:/UBB_train/color_G_2/s3_cam_dx-24.00_dy-36.00_yaw225.png"

# 相应地，也要删除其他15个退化文件夹中的同basename文件
# 以及单输入训练集中对应的文件
```

---

## ⚠️ 影响范围

### 多输入训练集
- 这2个样本的其他15个退化图应该也要删除（保持配对）
- 剩余: 11,098 - 2 = 11,096样本

### 单输入训练集
- 需要删除这2个basename对应的所有15种退化
- 剩余: 166,470 - 30 = 166,440对

### 验证集
- ✅ 验证集全部正常，无需修改

---

## 🎯 下一步

### 修复后重新检查
```bash
python scripts/check_all_images_parallel.py --threads 64
```

### 确认无损坏后执行LMDB转换
```bash
# 多输入LMDB
python -u scripts/create_lmdb.py \
  --data-root "D:/UBB_train" \
  --output "F:/DATASATES/UBB_train.lmdb" \
  --map-size-gb 1024

# 单输入LMDB
python -u scripts/create_lmdb_single_input.py \
  --input-dir "D:/UBB_train_single_input/input" \
  --gt-dir "D:/UBB_train_single_input/gt" \
  --output "E:/DATASATES/UBB_train_single_input.lmdb" \
  --map-size-gb 1024
```

---

**损坏文件位置已记录！** 修复后再继续。




