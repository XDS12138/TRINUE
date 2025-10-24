# 修复损坏文件后的下一步操作

## ✅ 当前完成的工作

1. ✅ 多输入训练集 - `D:/UBB_train`（11,098样本）
2. ✅ 单输入训练集 - `D:/UBB_train_single_input`（166,470对）
3. ✅ 全量检查完成 - 发现2个损坏文件
4. ✅ 已停止LMDB转换
5. ✅ 已清理未完成的LMDB

---

## 🔴 需要修复的文件（2个）

### 文件1
```
位置: D:/UBB_train/color_BG_3/s3_cam_dx+24.00_dy-18.00_yaw315.png
场景: Scene 3
退化: color_BG_3（蓝绿色-重度）
```

### 文件2
```
位置: D:/UBB_train/color_G_2/s3_cam_dx-24.00_dy-36.00_yaw225.png
场景: Scene 3
退化: color_G_2（绿色-中度）
```

---

## 🔧 修复步骤

### 步骤1: 从原始数据查找

检查`F:/DATASATES/UBBraw/scene_3/`对应文件夹：
```bash
# 文件1的原始位置（可能）
F:/DATASATES/UBBraw/scene_3/color_BG_3/cam_dx+24.00_dy-18.00_yaw315.png

# 文件2的原始位置（可能）
F:/DATASATES/UBBraw/scene_3/color_G_2/cam_dx-24.00_dy-36.00_yaw225.png
```

**如果找到**：复制并重命名为 `s3_...`

### 步骤2: 或从DATA/UBB恢复

```bash
# 原始渲染输出
DATA/UBB/3/GB3/cam_dx+24.00_dy-18.00_yaw315_rgb_0001.png
DATA/UBB/3/G2/cam_dx-24.00_dy-36.00_yaw225_rgb_0001.png
```

### 步骤3: 或删除这2个样本

如果无法恢复，删除这2个样本的**所有17组文件**：
```bash
# 需要删除的basename
basename1: s3_cam_dx+24.00_dy-18.00_yaw315
basename2: s3_cam_dx-24.00_dy-36.00_yaw225

# 从17个文件夹中删除
D:/UBB_train/gt/s3_cam_dx+24.00_dy-18.00_yaw315.png
D:/UBB_train/depth/s3_cam_dx+24.00_dy-18.00_yaw315.png
D:/UBB_train/color_B_1/s3_cam_dx+24.00_dy-18.00_yaw315.png
... (共17×2=34个文件)

# 同时删除单输入训练集中的15×2=30个文件
```

---

## ✅ 修复完成后

### 1. 重新检查
```bash
python scripts/check_all_images_parallel.py --threads 64
```

### 2. 确认无损坏后，执行LMDB转换

**终端1 - 多输入LMDB**:
```bash
python -u scripts/create_lmdb.py \
  --data-root "D:/UBB_train" \
  --output "F:/DATASATES/UBB_train.lmdb" \
  --map-size-gb 1024
```

**终端2 - 单输入LMDB**:
```bash
python -u scripts/create_lmdb_single_input.py \
  --input-dir "D:/UBB_train_single_input/input" \
  --gt-dir "D:/UBB_train_single_input/gt" \
  --output "E:/DATASATES/UBB_train_single_input.lmdb" \
  --map-size-gb 1024
```

---

**等待修复损坏文件后继续！** 📝


