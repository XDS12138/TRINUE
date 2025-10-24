# UBB数据集完整处理总结报告

## ✅ 最终数据集配置

### 📁 训练集 - `F:/DATASATES/UBB_train`
- **样本数**: **11,098个**
- **文件数**: 11,098 × 17组 = 188,666个
- **结构**: 17个文件夹（15退化 + GT + depth）
- **文件名格式**: `s1_cam_dx+0.00_dy+0.50_yaw000.png` (带场景前缀)
- **用途**: 模型训练

### 📁 有参考验证集 - `DATA/validation/UBB-M_reference`
- **样本数**: **1,500个**（100个/退化 × 15种）
- **Input**: 1,500个
- **GT**: 1,500个
- **文件名格式**: `s1__B1__cam_dx+0.00_dy+0.50_yaw000.png` (保留退化前缀)
- **用途**: PSNR/SSIM/LPIPS/CIEDE2000评估

### 📁 无参考验证集 - `DATA/validation/UBB-M_noreference`
- **样本数**: **1,500个**
- **Input**: 1,500个
- **文件名格式**: `s1__B1__cam_dx+0.00_dy+0.50_yaw000.png`
- **用途**: UCIQE/UIQM/NIQE评估

### 📁 深度验证集 - `DATA/validation/UBB-M_depth`
- **样本数**: **1,500个**
- **Input**: 1,500个（RGB图像）
- **Depth**: 1,500个（深度GT）
- **文件名格式**: `s1__B1__cam_dx+0.00_dy+0.50_yaw000.png`
- **用途**: 深度预测评估

### 📁 原始数据归档 - `F:/DATASATES/UBBraw`
- **样本数**: 15,356个
- **结构**: 按场景分离（scene_1, scene_2, scene_3, scene_4）
- **用途**: 原始数据备份和调试

---

## 📊 数据集统计

### 训练/验证划分
| 数据集 | 样本数 | 占比 | 状态 |
|--------|--------|------|------|
| 训练集 | 11,098 | 83.1% | ✅ |
| 验证集（唯一位置） | 1,384 | 10.4% | ✅ |
| 其他/重叠 | 874 | 6.5% | - |

### 场景分布（训练集）
| 场景 | 原始 | 删除 | 剩余 |
|------|------|------|------|
| Scene 1 | 4,200 | ~874 | ~3,326 |
| Scene 2 | 2,268 | ~372 | ~1,896 |
| Scene 3 | 2,688 | ~442 | ~2,246 |
| Scene 4 | 4,200 | ~570 | ~3,630 |

---

## 🔧 配置文件更新

### `configs/train.yaml` 需要更新

```yaml
data:
  train_root: "F:/DATASATES/UBB_train"  # ← 更新训练集路径
  
  folder_structure:
    gt: "gt"
    depth: "depth"
  
  degradation_folders: [
    'color_B_1', 'color_B_2', 'color_B_3',
    'color_BG_1', 'color_BG_2', 'color_BG_3',
    'color_G_1', 'color_G_2', 'color_G_3',
    'color_Y_1', 'color_Y_2', 'color_Y_3',
    'color_YG_1', 'color_YG_2', 'color_YG_3'
  ]

validation_sets:
  # RGB有参考验证集
  ubb_reference:
    name: "UBB_Reference"
    type: "enhancement_with_reference"
    data_root: "DATA/validation/UBB-M_reference"  # ← 已规范化
    enabled: true
    metrics:
      - "psnr"
      - "ssim"
      - "ciede2000"
      - "lpips"
    save_images: 10
    folder_structure:
      input: "input"
      gt: "gt"
  
  # RGB无参考验证集
  ubb_noreference:
    name: "UBB_NoReference"
    type: "enhancement_no_reference"
    data_root: "DATA/validation/UBB-M_noreference"  # ← 已规范化
    enabled: true
    metrics:
      - "uciqe"
      - "uiqm"
      - "niqe"
    save_images: 10
    folder_structure:
      input: "input"
  
  # 深度验证集
  ubb_depth:
    name: "UBB_Depth"
    type: "depth_prediction"
    data_root: "DATA/validation/UBB-M_depth"  # ← 新创建
    enabled: true
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
      rgb: "input"  # ← 注意这里是input不是rgb
      depth: "depth"
    depth_format: "png"
```

---

## 🎯 关键改进

### 1. 文件命名规范化 ✅
- 场景前缀: `24mm`→`s1`, `85mm`→`s4`
- 修复空格: `cam_dx 0.00`→`cam_dx+0.00`
- GT后缀: 去除`__GT`

### 2. 退化前缀保留 ✅
- 训练集: `s1_cam_dx+...` (无退化前缀)
- 验证集: `s1__B1__cam_dx+...` (有退化前缀B1)
- **原因**: 推理时单退化输入需要唯一标识

### 3. 验证集精简 ✅
- 从20,040个→1,500个（7.5%）
- 训练集保留率: 83.1%
- 每种退化均衡抽取100个

### 4. 无数据泄露 ✅
- 训练集已删除验证样本
- 严格配对验证

---

## 📂 最终目录结构

```
F:/DATASATES/
├── UBB_train/                    # 训练集（11,098样本）
│   ├── gt/
│   ├── depth/
│   └── color_*/ (15个)
│
├── UBBraw/                       # 原始归档（15,356样本）
│   ├── scene_1/
│   ├── scene_2/
│   ├── scene_3/
│   └── scene_4/
│
└── 备份文件夹/
    ├── UBB_validation_backup_reference/     (18,540个)
    ├── UBB_validation_backup_noreference/   (18,540个)
    └── UBB_depth_backup/                    (18,540对)

DATA/validation/
├── UBB-M_reference/              # 有参考验证（1,500对）
│   ├── input/
│   └── gt/
│
├── UBB-M_noreference/            # 无参考验证（1,500个）
│   └── input/
│
└── UBB-M_depth/                  # 深度验证（1,500对）
    ├── input/
    └── depth/
```

---

## ✅ 所有任务完成清单

- [x] 整理原始数据到UBBraw（按场景分离）
- [x] 整理训练集到UBB_train（合并场景）
- [x] 规范化验证集文件名（保留退化前缀）
- [x] 创建深度验证集
- [x] 精简验证集（20,040→1,500）
- [x] 从训练集排除验证集样本
- [x] 所有文件备份完成

---

## 🚀 下一步：开始训练

1. 更新 `configs/train.yaml`（见上方配置）
2. 启动训练:
   ```bash
   python scripts/train.py --config configs/train.yaml
   ```

**数据集已完全准备就绪！** 🎉




