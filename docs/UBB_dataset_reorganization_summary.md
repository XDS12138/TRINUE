# UBB数据集整理总结报告

## 📊 原始数据集结构

```
DATA/UBB/
├── 1/  (Scene 1) - 无法线图
│   ├── color_B_1/, color_B_2/, color_B_3/
│   ├── color_BG_1/, color_BG_2/, color_BG_3/
│   ├── color_G_1/, color_G_2/, color_G_3/
│   ├── color_Y_1/, color_Y_2/, color_Y_3/
│   ├── color_YG_1/, color_YG_2/, color_YG_3/
│   ├── gt/
│   └── depth/
├── 2/  (Scene 2) - 有法线图
│   ├── B1/, B2/, B3/ (紧凑命名)
│   ├── GB1/, GB2/, GB3/
│   ├── G1/, G2/, G3/
│   ├── Y1/, Y2/, Y3/
│   ├── YG1/, YG2/, YG3/
│   ├── gt/
│   ├── depth0.01-30/  (不同深度范围命名)
│   └── normal_vis/
├── 3/  (Scene 3) - 有法线图，**命名不一致问题**
│   ├── B1/, B2/, B3/
│   ├── GB1/, GB2/, GB3/
│   ├── G1/, G2/, G3/
│   ├── Y1/, Y2/, Y3/
│   ├── YG1/, YG2/, YG3/
│   ├── gt/
│   ├── depth0.01-150/  (**depth文件缺少'd'前缀**: cam_x vs cam_dx)
│   └── normal_vis/
└── 4/  (Scene 4) - 有法线图
    ├── B1/, B2/, B3/
    ├── GB1/, GB2/, GB3/
    ├── G1/, G2/, G3/
    ├── Y1/, Y2/, Y3/
    ├── YG1/, YG2/, YG3/
    ├── gt/
    ├── depth_vis/
    └── normal_vis/
```

## 🔍 发现的主要问题

### 1. **文件夹命名不统一**
- Scene 1: `color_B_1` (完整命名)
- Scene 2/3/4: `B1` (紧凑命名)
- **解决方案**: 统一映射到规范命名 `color_B_1`

### 2. **渲染输出后缀不一致**
- RGB图像：
  - Scene 1: `cam_dx+0.00_dy+0.00_yaw000.png` (无后缀)
  - Scene 2/3/4: `cam_dx+0.00_dy+0.00_yaw000_rgb_0001.png` (有`_rgb_0001`后缀)
- Depth图像: `cam_dx+0.00_dy+0.00_yaw000_mist_vis_0001.png`
- Normal图像: `cam_dx+0.00_dy+0.00_yaw000_normal_vis_0001.png`
- **解决方案**: 智能strip所有渲染后缀 (`_rgb*`, `_mist*`, `_normal*`)

### 3. **Scene 3深度文件命名错误**
- GT/退化图: `cam_dx+0.00_dy+0.00_yaw000_rgb_0001.png` ✅
- Depth: `cam_x+0.00_y+0.00_yaw000_mist_0001.png` ❌ (缺少'd'前缀)
- **解决方案**: 自动规范化 `cam_x` → `cam_dx`, `_y` → `_dy`

### 4. **Scene 3深度文件多1个**
- 其他组: 2688个文件
- Depth: 2689个文件 (多1个未配对文件)
- **解决方案**: 仅复制严格配对的2688个样本

### 5. **Scene 4 GT文件符号缺失**
- GT: `cam_dx 0.00_dy 0.25_yaw090.png` ❌ (空格，缺少`+`号)
- 其他: `cam_dx+0.00_dy+0.25_yaw090.png` ✅ (正确格式)
- **解决方案**: 自动规范化空格为`+`号 (`cam_dx 0.00` → `cam_dx+0.00`)

### 6. **跨场景文件名重复**
- 不同场景间有12-4113个重复的basename
- 例如: `cam_dx+12.00_dy+0.00_yaw090` 同时出现在Scene 2和Scene 3
- **解决方案**: 文件名添加场景前缀 `s{scene}_`

## ✅ 配对验证结果

| 场景 | 配对样本数 | 法线图 | 17组完整性 |
|------|-----------|--------|-----------|
| Scene 1 | 4,200 | ❌ | ✅ |
| Scene 2 | 2,268 | ✅ | ✅ |
| Scene 3 | 2,688 | ✅ | ✅ |
| Scene 4 | 4,200 | ✅ | ✅ |
| **总计** | **15,356** | - | - |

**严格配对标准**:
- 每个样本必须同时包含: 15个退化图 + 1个GT + 1个depth
- 法线图作为可选项（仅Scene 1没有）

## 📁 整理后的规范结构

```
F:/DATASATES/UBB_train/
├── gt/                  (15,356 files)
│   ├── s1_cam_dx+0.00_dy+0.00_yaw000.png
│   ├── s1_cam_dx+0.00_dy+0.00_yaw045.png
│   ├── s2_cam_dx+0.00_dy+0.00_yaw000.png
│   ├── s3_cam_dx+0.00_dy+0.00_yaw000.png
│   ├── s4_cam_dx+0.00_dy+0.00_yaw000.png
│   └── ...
├── depth/               (13,141 files)
├── color_B_1/           (13,141 files)
├── color_B_2/           (13,141 files)
├── color_B_3/           (13,141 files)
├── color_BG_1/          (13,141 files)
├── color_BG_2/          (13,141 files)
├── color_BG_3/          (13,141 files)
├── color_G_1/           (13,141 files)
├── color_G_2/           (13,141 files)
├── color_G_3/           (13,141 files)
├── color_Y_1/           (13,141 files)
├── color_Y_2/           (13,141 files)
├── color_Y_3/           (13,141 files)
├── color_YG_1/          (13,141 files)
├── color_YG_2/          (13,141 files)
├── color_YG_3/          (13,141 files)
├── mapping_scene_1.csv  (文件映射记录)
├── mapping_scene_2.csv
├── mapping_scene_3.csv
└── mapping_scene_4.csv
```

**文件命名规则**: `s{scene_id}_{original_basename}.{ext}`
- 避免跨场景文件名冲突
- 保留原始渲染信息（相机位置、角度等）
- 可追溯到原始场景

## 🔧 训练配置更新

### 需要修改 `configs/train.yaml`:

```yaml
data:
  format: "folder"
  dataset_type: multi_degradation
  train_root: "F:/DATASATES/UBB_train"  # ← 更新数据路径
  
  folder_structure:
    gt: "gt"
    depth: "depth"
  
  # 保持原有的15个退化类型
  degradation_folders: [
    'color_B_1', 'color_B_2', 'color_B_3',
    'color_BG_1', 'color_BG_2', 'color_BG_3',
    'color_G_1', 'color_G_2', 'color_G_3',
    'color_Y_1', 'color_Y_2', 'color_Y_3',
    'color_YG_1', 'color_YG_2', 'color_YG_3'
  ]
  
  batch_size: 1
  resolution: 256
  # ... 其他配置保持不变
```

## 📈 数据集统计

### 总量
- **总样本数**: 15,356个严格配对样本
- **总文件数**: 15,356 × 17组 = **261,052个文件**
- **估计磁盘占用**: ~15-60GB (取决于PNG压缩率)

### 按场景分布
- Scene 1 (无法线): 27.35% (4,200样本)
- Scene 2: 14.77% (2,268样本)
- Scene 3: 17.50% (2,688样本)
- Scene 4: 27.35% (4,200样本) ← **已修复命名问题**

### 相机采样特征
- **位置采样**: dx∈[-48,+48], dy∈[-78,+78] (单位可能是米或分米)
- **角度采样**: yaw∈{0°, 30°, 45°, 60°, 90°, 120°, 135°, 180°, 225°, 270°, 315°}
- **多视角覆盖**: 每个场景的水下环境从多个相机位置和角度渲染

## 🚀 使用建议

### 训练策略
1. **数据增强**: 由于已有多视角，可以适当降低增强概率（当前0.1）
2. **Batch Size**: 13,141样本 ÷ batch_size=2 (双GPU) ≈ 6,570 steps/epoch
3. **Epoch数**: 建议100-200 epochs充分学习15种退化模式

### 验证集划分
- 当前整理后的数据集作为**训练集**
- 建议从每个场景随机抽取5-10%作为验证集
- 或使用现有的独立验证集（UIEB_test, LSUI_test, UBB-test等）

### 深度图处理
- Scene 2: depth范围 0.01-30m
- Scene 3: depth范围 0.01-150m
- 需要在`loss.depth_processing`中配置合适的`min_depth`和`max_depth`

## 🔄 可追溯性

每个场景都生成了CSV映射文件 (`mapping_scene_X.csv`)，包含：
- **原始文件路径** (source)
- **目标文件路径** (destination)

示例内容:
```csv
src,dst
DATA/UBB/1/color_B_1/cam_dx+0.00_dy+0.00_yaw000.png,F:/DATASATES/UBB_train/color_B_1/s1_cam_dx+0.00_dy+0.00_yaw000.png
DATA/UBB/1/gt/cam_dx+0.00_dy+0.00_yaw000.png,F:/DATASATES/UBB_train/gt/s1_cam_dx+0.00_dy+0.00_yaw000.png
...
```

可用于：
- 问题调试时定位原始文件
- 验证文件完整性
- 后续数据增量更新

## ✅ 整理完成检查清单

- [x] 验证所有场景的文件配对完整性
- [x] 识别并修复命名不一致问题
- [x] 统一文件夹命名到规范格式
- [x] 智能匹配带不同后缀的渲染输出
- [x] 添加场景前缀避免文件名冲突
- [ ] 复制文件到目标目录 (进行中)
- [ ] 生成映射CSV文件
- [ ] 更新训练配置文件
- [ ] 验证整理后的数据集可正常加载

---

**整理脚本**: `scripts/prepare_ubb_dataset.py`
**验证报告**: `F:/DATASATES/UBB_validation_report_final.json`
**生成时间**: 2025-10-21

