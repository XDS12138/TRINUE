# 多文件日志系统使用指南

## 概述

新的多文件日志系统将不同类型的日志分别保存到不同的文件中，使日志更有组织性和可读性。

## 日志文件结构

在每个实验目录下，日志文件保存在 `logs/` 子目录中：

```
experiments/train/your_experiment/
├── logs/
│   ├── train/
│   │   ├── epoch_1.log
│   │   └── epoch_2.log
│   ├── loss/
│   │   ├── epoch_1.log
│   │   └── epoch_2.log
│   ├── metrics/
│   │   └── epoch_1.log
│   ├── model/
│   │   └── epoch_1.log
│   ├── data/
│   │   └── epoch_1.log
│   ├── optimizer/
│   │   └── epoch_1.log
│   ├── checkpoint/
│   │   └── epoch_1.log
│   ├── error/
│   │   └── epoch_1.log
│   ├── debug/
│   │   └── epoch_1.log
│   └── general/
│       └── epoch_1.log
├── tensorboard/        # TensorBoard日志
└── checkpoints/        # 模型检查点
```

## 日志文件说明

### 1. train/epoch_N.log
- **内容**：训练进度、epoch开始/结束、主要里程碑
- **级别**：INFO及以上
- **示例内容**：
  ```
  2024-12-23 10:00:00 [INFO] [TRINUE.train] ================================================================================
  2024-12-23 10:00:00 [INFO] [TRINUE.train] TRAINING STARTED
  2024-12-23 10:00:00 [INFO] [TRINUE.train] Experiment: underwater_enhance_run
  2024-12-23 10:00:00 [INFO] [TRINUE.train] Epochs: 100
  2024-12-23 10:00:00 [INFO] [TRINUE.train] Batch size: 4
  ```

### 2. loss/epoch_N.log
- **内容**：详细的损失值记录，包括总损失和各个组件
- **级别**：INFO用于总损失，DEBUG用于组件
- **示例内容**：
  ```
  2024-12-23 10:00:05 [INFO] [TRINUE.loss] [train] Step 0: Total Loss = 0.823456
  2024-12-23 10:00:05 [DEBUG] [TRINUE.loss] [train] Step 0: Components - loss_l1=0.234567, loss_ssim=0.345678, loss_depth=0.123456
  ```

### 3. metrics/epoch_N.log
- **内容**：验证指标（PSNR、SSIM等）
- **级别**：INFO
- **示例内容**：
  ```
  2024-12-23 10:05:00 [INFO] [TRINUE.metrics] [val] Step 0: psnr=28.3456, ssim=0.8912, mae=0.0234
  2024-12-23 10:05:00 [INFO] [TRINUE.metrics] Epoch 1 Validation Summary: PSNR=28.35, SSIM=0.8912
  ```

### 4. model/epoch_N.log
- **内容**：模型架构信息、参数更新、梯度统计
- **级别**：INFO和DEBUG
- **示例内容**：
  ```
  2024-12-23 10:00:00 [INFO] [TRINUE.model] === 模型架构信息 ===
  2024-12-23 10:00:00 [INFO] [TRINUE.model] 基础通道数: 48
  2024-12-23 10:00:00 [INFO] [TRINUE.model] 编码器层级数: 4
  ```

### 5. data/epoch_N.log
- **内容**：数据加载信息、批次统计、增强操作
- **级别**：INFO和DEBUG
- **示例内容**：
  ```
  2024-12-23 10:00:00 [INFO] [TRINUE.data] 数据集准备完成
  2024-12-23 10:00:00 [INFO] [TRINUE.data] 训练集路径: DATA/train
  2024-12-23 10:00:00 [INFO] [TRINUE.data] 批次大小: 4
  ```

### 6. optimizer/epoch_N.log
- **内容**：优化器设置、学习率、参数分组
- **级别**：INFO
- **示例内容**：
  ```
  2024-12-23 10:00:00 [INFO] [TRINUE.optimizer] 使用差异化学习率: 主干 0.0002, 注意力模块 0.00002
  2024-12-23 10:00:00 [INFO] [TRINUE.optimizer] 总参数数量: 34,512,987
  ```

### 7. checkpoint/epoch_N.log
- **内容**：检查点保存与加载
- **级别**：INFO
- **示例内容**：
  ```
  2024-12-23 10:12:00 [INFO] [TRINUE.checkpoint] Checkpoint saved at epoch 5 (best=True)
  ```

### 8. error/epoch_N.log
- **内容**：错误信息、警告、异常堆栈
- **级别**：WARNING和ERROR
- **示例内容**：
  ```
  2024-12-23 10:02:30 [WARNING] [TRINUE.error] Depth prediction collapsed at epoch 14, std=0.000123
  2024-12-23 10:02:31 [ERROR] [TRINUE.error] DECL loss computation failed: RuntimeError: ...
  ```

### 9. debug/epoch_N.log
- **内容**：详细的调试信息、张量形状、中间值
- **级别**：DEBUG
- **示例内容**：
  ```
  2024-12-23 10:00:05 [DEBUG] [TRINUE.debug] DepthDecoder bottleneck shape: torch.Size([4, 192, 32, 32])
  2024-12-23 10:00:05 [DEBUG] [TRINUE.debug] GPU Memory: Allocated 2.34GB, Reserved 3.12GB
  ```

## 使用方法

### 1. 在训练脚本中使用

```python
# 训练脚本已自动集成多文件日志系统
# 无需额外配置，只需正常运行训练
python scripts/train.py --config configs/train.yaml
```

### 2. 在自定义脚本中使用

```python
from utils.multi_logger import create_multi_logger

# 创建多文件日志管理器
multi_logger = create_multi_logger(config, exp_dir)

# 获取特定类型的logger
train_logger = multi_logger.get_logger('train')
loss_logger = multi_logger.get_logger('loss')
optimizer_logger = multi_logger.get_logger('optimizer')
checkpoint_logger = multi_logger.get_logger('checkpoint')

# 使用便捷方法记录
multi_logger.log_training_start(config)
multi_logger.log_epoch_start(epoch, total_epochs)
multi_logger.log_loss(losses_dict, step, 'train')
multi_logger.log_metrics(metrics_dict, step, 'val')
multi_logger.log_epoch_end(epoch, summary_dict)
optimizer_logger.info("Optimizer configured")
checkpoint_logger.info("Checkpoint saved")

# 记录错误和警告
multi_logger.log_error("Something went wrong", exc_info=True)
multi_logger.log_warning("This might be a problem")

# 记录调试信息
multi_logger.log_debug("Detailed debug information")

# 关闭日志系统
multi_logger.close()
```

### 3. 配置日志级别

在 `configs/train.yaml` 中配置：

```yaml
logging:
  console_level: "INFO"    # 控制台输出级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
  file_level: "DEBUG"      # 文件记录级别：通常设为DEBUG以捕获所有信息
```

## 日志查看技巧

### 1. 实时查看训练进度
```bash
# 查看主要训练进度
tail -f experiments/train/your_experiment/logs/train/epoch_1.log

# 查看损失变化
tail -f experiments/train/your_experiment/logs/loss/epoch_1.log | grep "Total Loss"

# 查看验证指标
tail -f experiments/train/your_experiment/logs/metrics/epoch_1.log
```

### 2. 查看错误和警告
```bash
# 查看所有错误
grep ERROR experiments/train/your_experiment/logs/error/epoch_1.log

# 查看最近的警告
tail -n 50 experiments/train/your_experiment/logs/error/epoch_1.log | grep WARNING
```

### 3. 搜索特定信息
```bash
# 搜索特定epoch的信息
grep "Epoch 14" experiments/train/your_experiment/logs/*/epoch_*.log

# 搜索深度预测相关的问题
grep -i "depth.*collapse" experiments/train/your_experiment/logs/*/epoch_*.log
```

### 4. 合并查看多个日志
```bash
# 按时间顺序查看所有日志
sort -k1,2 experiments/train/your_experiment/logs/*/epoch_*.log | less
```

## 日志文件大小管理

每个日志文件使用 `TimedRotatingFileHandler` 进行管理：
- 每天午夜自动创建新的日志文件
- 旧的日志文件将重命名为 `filename.log.YYYY-MM-DD` 格式
- 系统会保留所有历史日志文件，不会自动删除
- 这种设计确保您可以保留完整的训练历史记录

如果您希望节省磁盘空间，可以手动归档或压缩不再需要的旧日志：
```bash
# 手动压缩旧日志文件
find experiments/train/your_experiment/logs -name "*.log.*" -mtime +30 -exec gzip {} \;

# 将旧日志归档到备份目录
mkdir -p logs_archive
find experiments/train/your_experiment/logs -name "*.log.*" -mtime +60 -exec mv {} logs_archive/ \;
```

## 故障排查

### 问题：日志文件没有创建
- 检查实验目录权限
- 确认磁盘空间充足
- 查看控制台是否有错误输出

### 问题：某些日志类型为空
- 检查对应功能是否启用（如验证）
- 确认日志级别设置正确
- 查看 `general/epoch_N.log` 是否有相关信息

### 问题：日志输出过多/过少
- 调整 `console_level` 和 `file_level`
- 对于特定模块，可以单独设置日志级别

## 最佳实践

1. **保持日志简洁**：INFO级别只记录关键信息
2. **使用正确的日志类型**：将日志写入对应的文件
3. **定期清理**：完成的实验可以压缩或删除旧日志
4. **结合TensorBoard**：日志文件适合文本分析，TensorBoard适合可视化

## 示例脚本

查看 `examples/multi_logger_example.py` 了解完整的使用示例。 