# 验证集精简方案

## 📊 当前状态

### 验证集
- 总文件: 20,040个
- 每种退化: 1,336个
- **问题**: 占用训练集90%的样本

### 训练集
- 总样本: 13,356个
- 如删除全部验证集 → 剩余1,329个 ❌ 太少！

---

## ✅ 推荐方案：按退化均衡抽取

### 配置参数测试

| 每退化样本数 | 验证集总数 | 训练集剩余 | 保留率 | 推荐 |
|-------------|-----------|-----------|--------|------|
| 100 | 1,500 | 11,179 | 83.7% | ⭐⭐⭐ 推荐 |
| 150 | 2,250 | 10,096 | 75.6% | ⭐⭐ 可行 |
| 200 | 3,000 | 8,893 | 66.6% | ⭐ 偏少 |
| 50 | 750 | 12,262 | 91.8% | 验证集太小 |

### 推荐配置：每退化100-150个

**方案A**: 每退化100个（保守）
```bash
python scripts/subsample_by_degradation.py --samples-per-deg 100
```
- 验证集: 1,500个（7.5%）
- 训练集: 11,179个（83.7%）✅ 推荐

**方案B**: 每退化150个（平衡）
```bash
python scripts/subsample_by_degradation.py --samples-per-deg 150
```
- 验证集: 2,250个（11.2%）
- 训练集: 10,096个（75.6%）✅ 可行

---

## 📝 执行步骤（先不执行，只校验）

### 步骤1: 生成精简列表
```bash
# 测试150个/退化的效果（已执行）
python scripts/subsample_by_degradation.py --samples-per-deg 150

# 如果想调整，可以重新运行
python scripts/subsample_by_degradation.py --samples-per-deg 100
```

**输出**: `F:/DATASATES/validation_subset.json`（选中的文件列表）

---

### 步骤2: 应用精简（移动未选中文件到备份）
```bash
# 先测试
python scripts/apply_validation_subset.py --dry-run

# 再执行
python scripts/apply_validation_subset.py
```

**效果**:
- 有参考验证集缩减到2,250个（150×15）
- 未选中的17,790个文件备份到 `F:/DATASATES/UBB_validation_backup`

---

### 步骤3: 同样精简无参考验证集
```bash
python scripts/apply_validation_subset.py \
  --input-dir "DATA/validation/UBB-M_noreference/input" \
  --gt-dir "" \
  --subset-json "F:/DATASATES/validation_subset.json" \
  --backup-dir "F:/DATASATES/UBB_validation_noref_backup"
```

---

### 步骤4: 精简深度验证集
```bash
python scripts/apply_validation_subset.py \
  --input-dir "DATA/validation/UBB-M_depth/input" \
  --gt-dir "DATA/validation/UBB-M_depth/depth" \
  --subset-json "F:/DATASATES/validation_subset.json" \
  --backup-dir "F:/DATASATES/UBB_depth_backup"
```

---

### 步骤5: 从训练集排除验证集
```bash
python scripts/exclude_validation_from_train.py \
  --train-root "F:/DATASATES/UBB_train"
```

**最终结果**:
- 训练集: ~10,096样本
- 验证集: 2,250样本

---

## 🎯 当前推荐

**执行前确认**:
- 当前已生成: 150个/退化的精简列表
- 预计训练集剩余: 10,096样本（75.6%）

**是否满意？** 
- 如果觉得训练集还太少，可以改为100个/退化（保留83.7%）
- 如果可以接受，继续执行步骤2-5

---

**等待你的确认，然后我执行！** 🚀




