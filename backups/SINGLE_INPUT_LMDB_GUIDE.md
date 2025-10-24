# 单输入格式LMDB转换指南

## 📋 转换流程

### 前置条件
- ✅ 单输入Folder格式已转换完成（`D:/UBB_train_single_input`）
- ✅ lmdb库已安装

### 执行转换（输出到E盘）
```bash
python scripts/create_lmdb_single_input.py \
  --input-dir "D:/UBB_train_single_input/input" \
  --gt-dir "D:/UBB_train_single_input/gt" \
  --output "E:/DATASATES/UBB_train_single_input.lmdb" \
  --map-size-gb 1024
```

**注意**: 从D盘读取，写入到E盘（测试不同盘的转换速度）

---

## 📊 预期结果

### 输入
- Input文件夹: 166,470个PNG
- GT文件夹: 166,470个PNG
- 总大小: ~300GB（Folder格式）

### 输出
- LMDB文件: 1个数据库
- 预计大小: **450-550GB**（未压缩）
- 每个样本: input[3,H,W] + gt[3,H,W]

### 转换耗时
- 速度: 约1-2 it/s
- 总时长: **约2-3小时**（46,000-83,000秒）

---

## 🎯 LMDB格式优势（单输入）

| 指标 | Folder | LMDB | 提升 |
|------|--------|------|------|
| 文件数 | 332,940个 | 1个 | - |
| 随机I/O | 每样本2次 | 每样本1次 | 2x |
| 训练速度 | 基准 | 2-3倍 | ⚡⚡ |

---

## ⚙️ 可选配置

### 保持原始尺寸（默认，推荐）
```bash
python scripts/create_lmdb_single_input.py \
  --map-size-gb 500
```
- 保留训练时的random_crop多样性
- LMDB较大（~500GB）

### 统一到960×540（节省空间）
```bash
python scripts/create_lmdb_single_input.py \
  --target-size 540 960 \
  --map-size-gb 300
```
- 预先resize到主流尺寸
- LMDB较小（~300GB）
- 损失部分random_crop多样性

---

## 💾 磁盘空间需求

### D盘空间
- Folder格式: 300GB
- LMDB格式: 500GB（与Folder同时保留）
- **总需求**: 800GB

### 建议
如果D盘空间紧张，可以：
1. 转换完成后删除Folder格式（节省300GB）
2. 或使用 `--target-size 540 960` 减小LMDB（节省200GB）

---

## 🚀 使用单输入LMDB训练

转换完成后，配置如下：

```yaml
data:
  format: "lmdb"
  lmdb_paths:
    train: "D:/UBB_train_single_input.lmdb"
  batch_size: 2       # 单输入可以增大batch
  num_workers: 32
  prefetch_factor: 8
```

---

**脚本已准备好，等单输入Folder格式转换完成后执行！** 📝

