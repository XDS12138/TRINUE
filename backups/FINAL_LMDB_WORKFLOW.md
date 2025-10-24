# 最终LMDB转换工作流

## 🎯 目标

从D盘（数据源）并行创建两个LMDB数据库到不同目标盘

---

## 📋 当前进行中的任务

### 任务1: 移动UBB_train到D盘 🔄
```bash
robocopy "F:\DATASATES\UBB_train" "D:\UBB_train" /E /MOVE /MT:16
```
- 源: F:/DATASATES/UBB_train（180GB）
- 目标: D:/UBB_train
- 操作: 移动（F盘原文件会删除）
- 预计: 10-30分钟

### 任务2: 转换单输入格式（D盘内） 🔄
```bash
python scripts/convert_to_single_input.py \
  --source "F:/DATASATES/UBB_train" \
  --target "D:/UBB_train_single_input"
```
- 输入: F:/DATASATES/UBB_train
- 输出: D:/UBB_train_single_input（300GB）
- 进度: ~12%
- 预计: 50分钟

---

## 🚀 等待完成后执行

### 步骤1: 验证D盘数据完整性
```bash
# 检查多输入训练集
python -c "import os; count=len([f for f in os.listdir('D:/UBB_train/gt') if f.endswith('.png')]); print(f'多输入: {count}/11098')"

# 检查单输入训练集
python -c "import os; input_count=len([f for f in os.listdir('D:/UBB_train_single_input/input') if f.endswith('.png')]); gt_count=len([f for f in os.listdir('D:/UBB_train_single_input/gt') if f.endswith('.png')]); print(f'单输入: input={input_count}/166470, gt={gt_count}/166470')"
```

### 步骤2: 并行创建两个LMDB

**终端1 - 多输入LMDB（D盘 → F盘）**:
```bash
python -u scripts/create_lmdb.py \
  --data-root "D:/UBB_train" \
  --output "F:/DATASATES/UBB_train.lmdb" \
  --map-size-gb 1024
```

**终端2 - 单输入LMDB（D盘 → E盘）**:
```bash
python -u scripts/create_lmdb_single_input.py \
  --input-dir "D:/UBB_train_single_input/input" \
  --gt-dir "D:/UBB_train_single_input/gt" \
  --output "E:/DATASATES/UBB_train_single_input.lmdb" \
  --map-size-gb 1024
```

**注意**: 在两个不同的PowerShell窗口中运行，真正并行！

---

## 📊 预期转换速度对比

### 多输入LMDB（D→F）
- 源: D盘（SSD?）
- 目标: F盘（HDD）
- 每样本: 17个文件
- 样本数: 11,098
- **预测速度**: 2-3 it/s（读取快，写入慢）
- **预测耗时**: 1-1.5小时

### 单输入LMDB（D→E）
- 源: D盘（SSD?）
- 目标: E盘（?）
- 每样本: 2个文件
- 样本数: 166,470
- **预测速度**: 10-15 it/s（文件少，处理快）
- **预测耗时**: 3-5小时

---

## 💡 速度对比分析

**如果单输入确实更快**：
- 原因1: 每样本只需2个PNG解码 vs 17个
- 原因2: D盘读取可能比F盘快（SSD vs HDD）
- 原因3: 跨盘写入避免同盘竞争

**测试价值**：
- 验证文件数量对转换速度的影响
- 为未来数据集规划提供参考

---

## 🎯 最终数据布局

### D盘（源数据）
```
D:/
├── UBB_train/              (180GB，多输入格式)
└── UBB_train_single_input/ (300GB，单输入格式)
```

### F盘（多输入LMDB）
```
F:/DATASATES/
└── UBB_train.lmdb/         (~350GB)
```

### E盘（单输入LMDB）
```
E:/DATASATES/
└── UBB_train_single_input.lmdb/  (~500GB)
```

---

**准备就绪！** 等D盘数据准备完成后，在两个终端窗口并行执行LMDB转换。




