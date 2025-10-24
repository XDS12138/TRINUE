# UBB数据集整理指南

## 📋 整理脚本功能

`scripts/prepare_ubb_dataset.py` 提供以下功能：

1. ✅ **严格配对验证**：检查15个退化图 + GT + depth是否完整配对
2. ✅ **智能文件名匹配**：自动处理不同的渲染输出后缀（`_rgb_0001`, `_mist_vis_0001`等）
3. ✅ **命名规范化**：修复Scene 3的命名错误（`cam_x` → `cam_dx`）
4. ✅ **场景前缀**：为每个文件添加场景前缀避免冲突（`s1_`, `s2_`, `s3_`, `s4_`）
5. ✅ **可追溯映射**：生成CSV文件记录源文件→目标文件的映射关系

## 🎯 验证结果摘要

| 场景 | 配对样本数 | 法线图 | 文件夹数 |
|------|-----------|--------|---------|
| Scene 1 | **4,200** | ❌ | 17组 |
| Scene 2 | **2,268** | ✅ | 18组 |
| Scene 3 | **2,688** | ✅ | 18组 |
| Scene 4 | **4,200** | ✅ | 18组 |
| **总计** | **15,356** | - | - |

**预计处理文件数**：15,356 × 17组 = **261,052个文件**

## 🚀 使用方法

### 步骤1：仅验证（不复制文件）

```bash
# Windows PowerShell
python -u scripts/prepare_ubb_dataset.py `
  --source "DATA/UBB" `
  --target "F:/DATASATES/UBB_train" `
  --validate-only `
  --report-json "F:/DATASATES/UBB_validation_report.json" `
  --report-csv "F:/DATASATES/UBB_validation_summary.csv"
```

**输出**：
- `F:/DATASATES/UBB_validation_report.json` - 详细验证报告
- `F:/DATASATES/UBB_validation_summary.csv` - 汇总CSV

### 步骤2：执行数据集整理（复制模式，推荐）

```bash
# 完整复制（保留原始数据）
python -u scripts/prepare_ubb_dataset.py `
  --source "DATA/UBB" `
  --target "F:/DATASATES/UBB_train"

# 预计耗时：5-30分钟（取决于磁盘速度）
```

**文件操作**：
- ✅ **复制**所有配对文件到目标目录
- ✅ 保留原始文件不变
- ✅ 生成映射CSV (`mapping_scene_1.csv` 等)

### 步骤3（可选）：移动模式（清空原目录）

```bash
# ⚠️ 警告：会删除源文件！
python -u scripts/prepare_ubb_dataset.py `
  --source "DATA/UBB" `
  --target "F:/DATASATES/UBB_train" `
  --move
```

### 步骤4（可选）：干运行测试

```bash
# 模拟运行，不实际复制文件
python -u scripts/prepare_ubb_dataset.py `
  --source "DATA/UBB" `
  --target "F:/DATASATES/UBB_train" `
  --dry-run

# 测试少量样本（每个场景10个）
python -u scripts/prepare_ubb_dataset.py `
  --source "DATA/UBB" `
  --target "F:/DATASATES/UBB_train_test" `
  --limit 10
```

## 📁 整理后的目标结构

```
F:/DATASATES/UBB_train/
├── gt/                      # 15,356个GT图像
│   ├── s1_cam_dx+0.00_dy+0.00_yaw000.png
│   ├── s1_cam_dx+0.00_dy+0.00_yaw045.png
│   ├── s2_cam_dx+0.00_dy+0.00_yaw000.png  # 场景2
│   ├── s3_cam_dx+0.00_dy+0.00_yaw000.png  # 场景3
│   ├── s4_cam_dx+0.00_dy+0.00_yaw000.png  # 场景4
│   └── ...
├── depth/                   # 15,356个深度图
│   ├── s1_cam_dx+0.00_dy+0.00_yaw000.png
│   └── ...
├── color_B_1/              # 蓝色退化-级别1 (15,356个)
├── color_B_2/              # 蓝色退化-级别2
├── color_B_3/              # 蓝色退化-级别3
├── color_BG_1/             # 蓝绿色退化-级别1
├── color_BG_2/             # 蓝绿色退化-级别2
├── color_BG_3/             # 蓝绿色退化-级别3
├── color_G_1/              # 绿色退化-级别1
├── color_G_2/              # 绿色退化-级别2
├── color_G_3/              # 绿色退化-级别3
├── color_Y_1/              # 黄色退化-级别1
├── color_Y_2/              # 黄色退化-级别2
├── color_Y_3/              # 黄色退化-级别3
├── color_YG_1/             # 黄绿色退化-级别1
├── color_YG_2/             # 黄绿色退化-级别2
├── color_YG_3/             # 黄绿色退化-级别3
├── mapping_scene_1.csv     # 场景1文件映射记录
├── mapping_scene_2.csv     # 场景2文件映射记录
├── mapping_scene_3.csv     # 场景3文件映射记录
└── mapping_scene_4.csv     # 场景4文件映射记录
```

## 🔍 文件命名规则详解

### 场景前缀
每个文件都带有场景ID前缀：`s{scene}_`

**示例**：
- Scene 1: `s1_cam_dx+0.00_dy+0.00_yaw000.png`
- Scene 2: `s2_cam_dx+0.00_dy+0.00_yaw000.png`
- Scene 3: `s3_cam_dx+0.00_dy+0.00_yaw000.png`
- Scene 4: `s4_cam_dx+0.00_dy+0.00_yaw000.png`

### 原始basename保留
保留相机位置和角度信息：
- `dx+0.00` / `dy+0.00`: 相机位置偏移
- `yaw000`: 相机偏航角（0°到315°）

### 为什么需要场景前缀？
- **避免文件名冲突**：不同场景有12-4113个重复的basename
- **可追溯性**：从文件名就能知道来自哪个场景
- **调试方便**：问题定位时可以快速筛选特定场景的数据

## 📊 映射CSV文件格式

每个场景生成一个CSV文件 (`mapping_scene_X.csv`)：

```csv
src,dst
DATA/UBB/1/color_B_1/cam_dx+0.00_dy+0.00_yaw000.png,F:/DATASATES/UBB_train/color_B_1/s1_cam_dx+0.00_dy+0.00_yaw000.png
DATA/UBB/1/gt/cam_dx+0.00_dy+0.00_yaw000.png,F:/DATASATES/UBB_train/gt/s1_cam_dx+0.00_dy+0.00_yaw000.png
DATA/UBB/1/depth/cam_dx+0.00_dy+0.00_yaw000_mist_vis_0001.png,F:/DATASATES/UBB_train/depth/s1_cam_dx+0.00_dy+0.00_yaw000.png
...
```

**用途**：
- ✅ 验证文件完整性
- ✅ 定位原始文件（调试时）
- ✅ 后续增量更新

## ⚙️ 配置训练脚本

整理完成后，更新 `configs/train.yaml`：

```yaml
data:
  format: "folder"
  dataset_type: multi_degradation
  train_root: "F:/DATASATES/UBB_train"  # ← 更新路径
  
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
  
  batch_size: 1
  num_workers: 48
  resolution: 256
  # ... 保持其他配置不变
```

## ✅ 整理完成后的验证检查

运行以下脚本验证整理结果：

```python
# scripts/verify_reorganization.py
import os

target = "F:/DATASATES/UBB_train"
expected_count = 13141

folders = [
    'gt', 'depth',
    'color_B_1', 'color_B_2', 'color_B_3',
    'color_BG_1', 'color_BG_2', 'color_BG_3',
    'color_G_1', 'color_G_2', 'color_G_3',
    'color_Y_1', 'color_Y_2', 'color_Y_3',
    'color_YG_1', 'color_YG_2', 'color_YG_3',
]

print("验证整理后的数据集...")
print("="*60)

all_ok = True
for folder in folders:
    path = os.path.join(target, folder)
    if os.path.exists(path):
        count = len([f for f in os.listdir(path) if f.endswith('.png')])
        status = "✅" if count == expected_count else "❌"
        print(f"{status} {folder:15s}: {count:6d} / {expected_count}")
        if count != expected_count:
            all_ok = False
    else:
        print(f"❌ {folder:15s}: 文件夹不存在")
        all_ok = False

print("="*60)
if all_ok:
    print("✅ 所有文件夹验证通过！")
else:
    print("❌ 发现问题，请检查整理过程")

# 验证映射CSV
print("\n验证映射文件...")
for scene in ['1', '2', '3', '4']:
    csv_path = os.path.join(target, f'mapping_scene_{scene}.csv')
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            lines = len(f.readlines()) - 1  # 排除表头
        print(f"✅ mapping_scene_{scene}.csv: {lines} 条映射")
    else:
        print(f"❌ mapping_scene_{scene}.csv: 不存在")
```

## 🔧 常见问题

### Q1: 整理需要多长时间？
**A**: 取决于磁盘速度：
- SSD → SSD: 5-10分钟
- HDD → HDD: 15-30分钟
- 网络盘: 可能更长

### Q2: 为什么不包含法线图（normal）？
**A**: Scene 1没有法线图，为了保持数据集一致性，当前配置不包含法线。如需包含，使用 `--require-normal` 会排除Scene 1的数据。

### Q3: 如何恢复原始数据？
**A**: 
- 如果使用**复制模式**（默认）：原始数据在 `DATA/UBB` 保持不变
- 如果使用**移动模式**（`--move`）：需要从备份恢复

### Q4: 磁盘空间不足怎么办？
**A**: 
1. 使用 `--move` 模式（但需先备份！）
2. 或使用 `--limit N` 先整理部分数据测试

### Q5: 如何只整理特定场景？
**A**: 手动修改脚本的 `validate_scenes()` 函数，或临时移动其他场景文件夹

## 📈 预期训练效果

使用整理后的数据集：
- **总样本数**: 15,356个
- **每epoch步数**: ~7,678 steps (batch_size=2, 双GPU)
- **训练覆盖**: 15种退化类型的完整学习
- **场景多样性**: 4个不同的水下环境场景

## 🎯 下一步

1. ✅ **执行数据集整理**（本指南步骤2）
2. ✅ **验证整理结果**（运行验证脚本）
3. ✅ **更新训练配置** (`configs/train.yaml`)
4. ✅ **开始训练** (`python scripts/train.py --config configs/train.yaml`)

---

**脚本路径**: `scripts/prepare_ubb_dataset.py`  
**验证报告**: `F:/DATASATES/UBB_validation_report_final.json`  
**问题反馈**: 检查日志输出或映射CSV文件

