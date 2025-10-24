# ✅ UBB数据集整理 - 准备就绪

## 📊 最终验证结果

| 场景 | 配对样本数 | 状态 |
|------|-----------|------|
| Scene 1 | 4,200 | ✅ 完美配对 |
| Scene 2 | 2,268 | ✅ 完美配对 |
| Scene 3 | 2,688 | ✅ 完美配对 |
| Scene 4 | 4,200 | ✅ 完美配对（已修复命名） |
| **总计** | **15,356** | ✅ 所有问题已解决 |

**总文件数**: 15,356 × 17组 = **261,052个文件**

## 🔧 已解决的所有问题

1. ✅ **文件夹命名不统一** - 统一映射到 `color_B_1` 等规范格式
2. ✅ **渲染输出后缀不一致** - 智能strip `_rgb_0001`, `_mist_vis_0001` 等
3. ✅ **Scene 3深度文件命名错误** - 自动规范化 `cam_x` → `cam_dx`
4. ✅ **Scene 3深度文件多1个** - 仅复制严格配对的样本
5. ✅ **Scene 4 GT文件符号缺失** - 自动规范化空格为 `+` 号
6. ✅ **跨场景文件名重复** - 添加场景前缀 `s1_`, `s2_`, `s3_`, `s4_`

## 🚀 执行命令（当你准备好时）

### 第1步：最终验证（确认配对）
```powershell
python -u scripts/prepare_ubb_dataset.py `
  --source "DATA/UBB" `
  --target "F:/DATASATES/UBB_train" `
  --validate-only `
  --report-json "F:/DATASATES/UBB_validation_FINAL.json"
```

**预期输出**:
```
Scene 1: paired=4200, has_normal=False, present_groups=17
Scene 2: paired=2268, has_normal=True, present_groups=18
Scene 3: paired=2688, has_normal=True, present_groups=18
Scene 4: paired=4200, has_normal=True, present_groups=18
```

### 第2步：执行数据集整理（复制文件）
```powershell
# 完整复制 - 约需15-30分钟
python -u scripts/prepare_ubb_dataset.py `
  --source "DATA/UBB" `
  --target "F:/DATASATES/UBB_train"
```

**文件操作**：
- 复制 261,052 个文件
- 生成 4 个映射CSV文件
- 保留原始 `DATA/UBB` 文件不变

### 第3步：验证整理结果
```powershell
python scripts/verify_reorganization.py F:/DATASATES/UBB_train
```

**预期输出**:
```
✅ gt             : 15356 / 15356
✅ depth          : 15356 / 15356
✅ color_B_1      : 15356 / 15356
...（所有17个文件夹）
✅ 所有验证通过！数据集整理成功！
```

### 第4步：更新训练配置
编辑 `configs/train.yaml`:
```yaml
data:
  train_root: "F:/DATASATES/UBB_train"  # ← 更新这一行
```

### 第5步：开始训练
```powershell
python scripts/train.py --config configs/train.yaml
```

## 📋 文件清单

### 整理脚本
- ✅ `scripts/prepare_ubb_dataset.py` - 主整理脚本
- ✅ `scripts/verify_reorganization.py` - 验证脚本
- ✅ `scripts/check_scene4.py` - 调试脚本

### 文档
- ✅ `scripts/UBB_REORGANIZATION_GUIDE.md` - 详细使用指南
- ✅ `docs/UBB_dataset_reorganization_summary.md` - 总结报告

### 验证报告
- ✅ `F:/DATASATES/UBB_validation_FIXED.json` - 最终验证报告

## 🎯 整理后的文件结构

```
F:/DATASATES/UBB_train/
├── gt/ (15,356个) - 清晰目标图像
├── depth/ (15,356个) - 深度图
├── color_B_1/ (15,356个) - 蓝色退化-轻度
├── color_B_2/ (15,356个) - 蓝色退化-中度
├── color_B_3/ (15,356个) - 蓝色退化-重度
├── color_BG_1/ (15,356个) - 蓝绿色退化-轻度
├── color_BG_2/ (15,356个) - 蓝绿色退化-中度
├── color_BG_3/ (15,356个) - 蓝绿色退化-重度
├── color_G_1/ (15,356个) - 绿色退化-轻度
├── color_G_2/ (15,356个) - 绿色退化-中度
├── color_G_3/ (15,356个) - 绿色退化-重度
├── color_Y_1/ (15,356个) - 黄色退化-轻度
├── color_Y_2/ (15,356个) - 黄色退化-中度
├── color_Y_3/ (15,356个) - 黄色退化-重度
├── color_YG_1/ (15,356个) - 黄绿色退化-轻度
├── color_YG_2/ (15,356个) - 黄绿色退化-中度
├── color_YG_3/ (15,356个) - 黄绿色退化-重度
├── mapping_scene_1.csv - Scene 1映射 (71,400行)
├── mapping_scene_2.csv - Scene 2映射 (38,556行)
├── mapping_scene_3.csv - Scene 3映射 (45,696行)
└── mapping_scene_4.csv - Scene 4映射 (71,400行)
```

## 📝 文件命名示例

所有文件都带场景前缀，避免冲突：

```
s1_cam_dx+0.00_dy+0.00_yaw000.png  # Scene 1, 位置(0,0), 角度0°
s1_cam_dx+3.00_dy+1.50_yaw045.png  # Scene 1, 位置(3,1.5), 角度45°
s2_cam_dx+0.00_dy+0.00_yaw000.png  # Scene 2, 位置(0,0), 角度0°
s3_cam_dx-6.00_dy-6.00_yaw090.png  # Scene 3, 位置(-6,-6), 角度90°
s4_cam_dx+12.00_dy+0.00_yaw180.png # Scene 4, 位置(12,0), 角度180°
```

## 💾 磁盘空间要求

- **F:/DATASATES** 需要 **至少60GB空闲空间**
- 实际占用：约15-50GB（取决于PNG压缩）
- 建议保留100GB缓冲空间

## ⏱️ 预计耗时

| 步骤 | 耗时 |
|-----|------|
| 验证 | ~30秒 |
| 复制文件 (HDD→HDD) | 15-30分钟 |
| 复制文件 (SSD→SSD) | 5-10分钟 |
| 验证整理结果 | ~10秒 |

## ⚠️ 重要提醒

1. **确保F盘有足够空间** (>60GB)
2. **默认是复制模式**，原始文件会保留在 `DATA/UBB`
3. 如需节省空间使用 `--move`，但建议**先备份**！
4. 整理过程可随时中断，不影响原始数据
5. 每个映射CSV记录了所有文件的源和目标路径

## ✅ 检查清单

执行前检查：
- [ ] F盘空闲空间 >60GB
- [ ] `DATA/UBB` 目录完整（4个场景子文件夹）
- [ ] 已阅读 `scripts/UBB_REORGANIZATION_GUIDE.md`

执行后检查：
- [ ] 运行验证脚本确认文件数量正确
- [ ] 检查4个映射CSV文件是否生成
- [ ] 更新 `configs/train.yaml` 中的 `train_root`

---

**准备就绪！** 当你想执行时，直接运行上述命令即可。

**问题排查**: 查看 `scripts/UBB_REORGANIZATION_GUIDE.md` 的"常见问题"章节



