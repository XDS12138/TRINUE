# TensorBoard 参数详细说明文档

## 📊 概述

本文档详细说明了TRINUE水下图像增强模型训练过程中所有TensorBoard记录的参数指标，帮助理解模型训练状态和性能变化。

---

## 🎯 参数分组概览

```
TensorBoard指标分组:
├── train/              # 训练指标
├── val/               # 验证指标  
├── physics/           # 物理参数指标
├── params/            # 模型参数分布
├── grads/             # 梯度分布
├── optimizer/         # 优化器指标
├── gpu/              # GPU和梯度指标
├── uncertainty/       # 不确定性权重
├── loss_*/           # 详细损失分解
└── attention/        # 注意力图可视化
```

---

## 📈 训练指标 (`train/`)

### 基础损失指标

| 指标名称 | 说明 | 期望范围 | 趋势 |
|---------|------|----------|------|
| `train/loss` | 总体训练损失 | 0.1-10.0 | 下降 |
| `train/img_total_loss` | 图像质量总损失 | 0.05-5.0 | 下降 |
| `train/depth_total_loss` | 深度预测总损失 | 0.01-2.0 | 下降 |

### 图像质量损失分解

| 指标名称 | 说明 | 计算方式 | 权重影响 |
|---------|------|----------|----------|
| `train/l1_loss` | L1像素差异损失 | `‖I_pred - I_gt‖₁` | 基础重建质量 |
| `train/ssim_loss` | 结构相似性损失 | `1 - SSIM(I_pred, I_gt)` | 结构保持 |
| `train/perc_loss` | 感知损失(VGG) | VGG特征空间距离 | 视觉质量 |
| `train/fft_loss` | 频域损失 | FFT域的差异 | 频率特征 |
| `train/grad_loss` | 梯度损失 | 梯度域的差异 | 边缘细节 |

### 深度相关损失

| 指标名称 | 说明 | 用途 |
|---------|------|------|
| `train/depth_pred_loss` | 深度预测损失 | 连续深度回归精度 |
| `train/depth_smooth_loss` | 深度平滑损失 | 深度图平滑性约束 |
| `train/depth_decoder_loss` | 深度解码器损失 | 多尺度深度特征损失 |
| `train/depth_rec_loss` | 深度重建损失 | 深度特征重建质量 |

### 物理模型损失

| 指标名称 | 说明 | 物理意义 |
|---------|------|----------|
| `train/phy_A_L1` | Beer-Lambert模型L1损失 | 水下光传播建模精度 |
| `train/phy_A_SSIM` | Beer-Lambert模型SSIM损失 | 光衰减建模结构一致性 |
| `train/L_phy_A` | 总Beer-Lambert损失 | 水下物理退化建模总体质量 |
| `train/phy_D_L1` | 深度模糊模型L1损失 | 深度相关模糊建模精度 |
| `train/phy_D_SSIM` | 深度模糊模型SSIM损失 | 模糊建模结构一致性 |
| `train/L_phy_D` | 总深度模糊损失 | 深度模糊建模总体质量 |

### 注意力损失

| 指标名称 | 说明 | 作用 |
|---------|------|------|
| `train/attn_cons_loss` | 注意力一致性损失 | RGB-深度交叉注意力约束 |

---

## 🔍 验证指标 (`val/`)

### 图像质量评估

| 指标名称 | 说明 | 最佳值 | 单位 |
|---------|------|--------|------|
| `val/psnr` | 峰值信噪比 | >30 | dB |
| `val/ssim` | 结构相似性指数 | >0.9 | 0-1 |
| `val/lpips` | 感知图像距离 | <0.1 | 0-1 |

### 深度评估指标

| 指标名称 | 说明 | 期望值 |
|---------|------|--------|
| `val/depth_mae` | 深度平均绝对误差 | <5000 |
| `val/depth_rmse` | 深度均方根误差 | <8000 |
| `val/depth_rel` | 深度相对误差 | <0.2 |

---

## ⚗️ 物理参数指标 (`physics/`)

### 🔧 固定物理参数

| 指标名称 | 物理含义 | 初始值 | 单位 |
|---------|----------|--------|------|
| `physics/fixed_beta_c_mean` | Beer-Lambert衰减系数平均值 | ~0.974 | 1/m |
| `physics/fixed_beta_c_ch0/1/2` | RGB各通道衰减系数 | ~0.974 | 1/m |
| `physics/fixed_B_c_mean` | 全局背景光平均值 | ~0.000 | 0-1 |
| `physics/fixed_B_c_ch0/1/2` | RGB各通道背景光 | ~0.000 | 0-1 |
| `physics/fixed_blur_scale` | 深度模糊比例 | ~0.049 | - |

**物理意义说明：**
- **β_c**: 水中光衰减速度，值越大表示水质越浑浊
- **B_c**: 环境背景光强度，影响整体亮度
- **blur_scale**: 深度相关模糊强度系数

### 🔥 动态预测物理参数

#### β_c动态预测统计
| 指标名称 | 说明 | 训练演进 |
|---------|------|----------|
| `physics/dynamic_beta_c_batch_mean` | 批次β_c平均值 | 初期~0.974 → 后期自适应 |
| `physics/dynamic_beta_c_batch_std` | 批次β_c标准差 | 初期~0 → 后期>0.01 |
| `physics/dynamic_beta_c_batch_min/max` | 批次β_c范围 | 初期一致 → 后期分化 |
| `physics/dynamic_beta_c_ch0/1/2_mean` | 各通道β_c均值 | 通道特异性学习 |
| `physics/dynamic_beta_c_ch0/1/2_std` | 各通道β_c方差 | 通道差异化程度 |

#### B_c动态预测统计
| 指标名称 | 说明 | 期望变化 |
|---------|------|----------|
| `physics/dynamic_B_c_batch_mean` | 批次背景光均值 | 环境光照自适应 |
| `physics/dynamic_B_c_batch_std` | 批次背景光方差 | 光照条件差异性 |
| `physics/dynamic_B_c_ch0/1/2_mean` | 各通道背景光 | 色温自适应 |

#### 模糊比例动态预测
| 指标名称 | 说明 | 自适应含义 |
|---------|------|-----------|
| `physics/dynamic_blur_scale_batch_mean` | 批次模糊比例均值 | 深度模糊强度自适应 |
| `physics/dynamic_blur_scale_batch_std` | 批次模糊比例方差 | 场景深度复杂度 |

### 📊 预测差异性指标

| 指标名称 | 说明 | 阈值判断 |
|---------|------|----------|
| `physics/beta_c_prediction_diversity` | β_c预测多样性 | >0.001表示开始分化 |
| `physics/B_c_prediction_diversity` | B_c预测多样性 | >0.001表示光照自适应 |
| `physics/blur_scale_prediction_diversity` | 模糊预测多样性 | >0.0001表示深度自适应 |

**多样性指标含义：**
- **0.000**: 所有图像预测相同参数（训练初期）
- **>0.001**: 开始为不同图像预测不同参数
- **>0.01**: 强自适应性，参数完全图像化

### ⚖️ 固定vs动态参数比较

| 指标名称 | 说明 | 演进趋势 |
|---------|------|----------|
| `physics/beta_c_dynamic_vs_fixed_diff` | β_c动态与固定差异 | 0 → 逐渐增大 |
| `physics/B_c_dynamic_vs_fixed_diff` | B_c动态与固定差异 | 0 → 根据场景变化 |
| `physics/blur_scale_dynamic_vs_fixed_diff` | 模糊比例差异 | 0 → 深度自适应 |

### 🧠 物理参数预测头统计

| 指标名称 | 说明 | 参考值 |
|---------|------|--------|
| `physics/predictor_total_params` | 预测头总参数量 | 21,127 |
| `physics/predictor_avg_grad_norm` | 预测头平均梯度范数 | 0.1-1.0 |
| `physics/predictor_max_grad_norm` | 预测头最大梯度范数 | 0.5-5.0 |

---

## 🧮 模型参数分布 (`params/`)

### 参数直方图

`params/`分组记录了模型所有可训练参数的分布情况，按网络层分组显示。

| 参数分组 | 说明 | 监控重点 |
|---------|------|----------|
| `params/sfe.*` | 浅层特征提取器参数 | 初期特征学习状态 |
| `params/encoder.*` | RGB-深度编码器参数 | 多模态融合学习 |
| `params/bottleneck.*` | 瓶颈层参数 | 高级特征表示 |
| `params/decoder.*` | 多任务解码器参数 | 重建质量关键 |
| `params/depth_*.*` | 深度相关模块参数 | 深度理解能力 |
| `params/physics_predictor.*` | 物理参数预测头 | 自适应物理建模 |
| `params/raw_beta_c` | 固定β_c参数 | 基础衰减系数 |
| `params/raw_B_c` | 固定B_c参数 | 基础背景光 |
| `params/raw_blur_scale` | 固定模糊比例参数 | 基础模糊强度 |

### 关键参数层解析

#### 🔍 浅层特征提取器 (`params/sfe.*`)
```
params/sfe.conv.weight          # 3×3卷积权重分布
params/sfe.conv.bias           # 卷积偏置分布
```
**监控要点：**
- 权重分布应该逐渐从随机初始化形状变为有意义的特征检测器
- 偏置值不应过大，避免饱和

#### 🔀 交叉注意力模块 (`params/encoder.*attn*`)
```
params/encoder.depth2rgb_attn_blocks.*.to_q.weight    # 查询变换权重
params/encoder.depth2rgb_attn_blocks.*.to_k.weight    # 键变换权重  
params/encoder.depth2rgb_attn_blocks.*.to_v.weight    # 值变换权重
params/encoder.rgb2depth_attn_blocks.*.to_q.weight    # RGB→深度注意力权重
```
**监控要点：**
- 注意力权重应该显示有意义的分布模式
- 过于尖锐的分布可能表示过度集中注意力
- 过于平坦的分布可能表示注意力机制未充分学习

#### 🧠 物理参数预测头 (`params/physics_predictor.*`)
```
params/physics_predictor.channel_attention.fc1.weight  # 通道注意力第一层
params/physics_predictor.channel_attention.fc2.weight  # 通道注意力第二层
params/physics_predictor.beta_conv.weight             # β_c预测卷积
params/physics_predictor.B_conv.weight                # B_c预测卷积
params/physics_predictor.blur_conv.weight             # 模糊预测卷积
```
**监控要点：**
- 预测头参数应该逐渐学会区分不同图像特征
- 通道注意力权重显示特征重要性模式
- 预测卷积权重体现物理参数与视觉特征的关联

#### 📐 固定物理参数 (`params/raw_*`)
```
params/raw_beta_c      # 原始β_c参数 [1,3,1,1]
params/raw_B_c         # 原始B_c参数 [1,3,1,1]  
params/raw_blur_scale  # 原始模糊比例参数 [1]
```
**监控要点：**
- 这些是全局固定参数，通常变化较慢
- 在使用动态预测时，这些参数可能保持相对稳定
- 可以作为动态预测的参考基准

### 参数分布健康指标

| 分布特征 | 健康状态 | 异常状态 | 处理建议 |
|---------|----------|----------|----------|
| **方差** | 0.01-1.0 | <0.001 或 >10 | 调整初始化/学习率 |
| **均值** | 接近0 | 偏离过远 | 检查数据归一化 |
| **形状** | 近似正态分布 | 过度尖锐/平坦 | 监控梯度流 |
| **更新速度** | 渐进变化 | 剧烈跳跃 | 降低学习率 |

---

## 📈 梯度分布 (`grads/`)

### 梯度直方图

`grads/`分组记录了模型所有参数的梯度分布，与`params/`对应。

| 梯度分组 | 说明 | 健康范围 |
|---------|------|----------|
| `grads/sfe.*` | 浅层特征提取器梯度 | 1e-4 到 1e-1 |
| `grads/encoder.*` | 编码器梯度 | 1e-5 到 1e-2 |
| `grads/bottleneck.*` | 瓶颈层梯度 | 1e-6 到 1e-2 |
| `grads/decoder.*` | 解码器梯度 | 1e-5 到 1e-1 |
| `grads/physics_predictor.*` | 物理预测头梯度 | 1e-6 到 1e-2 |

### 梯度流分析

#### 🌊 健康梯度流特征
1. **逐层衰减**：从输出层到输入层梯度幅度逐渐减小
2. **非零梯度**：所有层都应该有显著的非零梯度
3. **稳定分布**：梯度分布形状相对稳定，不剧烈变化
4. **合理范围**：梯度值在健康范围内，不过大或过小

#### ⚠️ 梯度异常诊断
| 异常类型 | 症状 | 可能原因 | 解决方案 |
|---------|------|----------|----------|
| **梯度消失** | 梯度值<1e-8 | 网络过深/激活函数 | 残差连接/调整激活 |
| **梯度爆炸** | 梯度值>10 | 学习率过大 | 梯度裁剪/降低学习率 |
| **梯度死区** | 某层梯度为0 | ReLU饱和/权重初始化 | 改用LeakyReLU/重新初始化 |
| **不平衡** | 不同层梯度差异过大 | 网络设计问题 | 批归一化/权重衰减 |

### 物理参数预测头梯度重点监控

由于物理参数预测是新增功能，需要特别关注：

```
grads/physics_predictor.channel_attention.*  # 通道注意力梯度
grads/physics_predictor.beta_conv.*          # β_c预测梯度
grads/physics_predictor.B_conv.*             # B_c预测梯度  
grads/physics_predictor.blur_conv.*          # 模糊预测梯度
```

**期望表现：**
- 训练初期：梯度较小但非零（~1e-5）
- 训练中期：梯度增大，显示活跃学习（~1e-4）
- 训练后期：梯度稳定，精细调整（~1e-5到1e-4）

---

## 🎛️ 优化器指标 (`optimizer/`)

| 指标名称 | 说明 | 期望范围 |
|---------|------|----------|
| `optimizer/lr` | 学习率 | 1e-6 到 1e-3 |

---

## 🖥️ GPU和梯度指标 (`gpu/`)

| 指标名称 | 说明 | 健康范围 |
|---------|------|----------|
| `gpu/grad_norm` | 梯度范数 | 0.1-10.0 |

**梯度范数判断：**
- **<0.01**: 梯度消失，学习停滞
- **0.1-10**: 正常范围
- **>100**: 梯度爆炸，需要调整学习率

---

## 🎲 不确定性权重 (`uncertainty/`)

| 指标名称 | 说明 | 自适应机制 |
|---------|------|-----------|
| `uncertainty/uncertainty_img` | 图像损失不确定性权重 | 自动平衡图像vs深度损失 |
| `uncertainty/uncertainty_depth` | 深度损失不确定性权重 | 任务难度自适应 |
| `uncertainty/log_var_img` | 图像损失对数方差 | 损失方差估计 |
| `uncertainty/log_var_depth` | 深度损失对数方差 | 任务不确定性量化 |

---

## 🔍 注意力可视化 (`attention/`)

| 可视化类型 | 说明 | 查看位置 |
|-----------|------|----------|
| `train/depth2rgb_attention` | 深度→RGB注意力图 | IMAGES标签页 |
| `train/rgb2depth_attention` | RGB→深度注意力图 | IMAGES标签页 |

---

## 📊 训练阶段解读

### 🌅 训练初期 (Steps 0-1000)

**特征：**
- 所有动态物理参数接近初始值
- `prediction_diversity` ≈ 0
- `dynamic_vs_fixed_diff` ≈ 0
- 总损失快速下降

**解释：** 模型在学习基础的图像增强能力，物理参数预测头还未开始分化。

### 🌤️ 训练中期 (Steps 1000-5000)

**特征：**
- `batch_std` 开始增大 (>0.001)
- `prediction_diversity` 出现正值
- 各类损失稳步下降
- 物理参数开始图像化

**解释：** 模型开始为不同类型的图像预测不同的物理参数。

### ☀️ 训练后期 (Steps 5000+)

**特征：**
- 物理参数完全自适应
- 验证指标持续改善
- 注意力图显示有意义的关联
- 损失收敛到较低值

**解释：** 模型已学会根据图像内容自适应调整物理建模参数。

---

## 🚨 异常情况诊断

### ⚠️ 训练停滞

**症状：**
- 损失不再下降
- 物理参数不再变化
- 梯度范数过小

**可能原因：**
- 学习率过小
- 数据过拟合
- 模型容量不足

### ⚠️ 训练不稳定

**症状：**
- 损失剧烈波动
- 梯度范数过大
- 物理参数振荡

**可能原因：**
- 学习率过大
- 批次大小不当
- 数据不平衡

### ⚠️ 物理参数异常

**症状：**
- 参数超出物理合理范围
- 预测差异性过大/过小
- 固定vs动态差异异常

**排查方法：**
- 检查参数约束是否正确
- 验证损失权重设置
- 观察注意力图是否合理

---

## 💡 最佳实践建议

### 📈 监控重点

1. **损失趋势**: 关注总损失的下降速度和收敛性
2. **物理参数进化**: 观察从一致性到差异化的过程
3. **验证指标**: PSNR、SSIM等客观指标的提升
4. **梯度健康**: 确保梯度范数在合理范围
5. **参数分布**: 监控模型参数的学习过程和分布变化
6. **梯度流动**: 确保各层梯度正常传播，无消失或爆炸

### 🎯 调优建议

1. **学习率调整**: 根据梯度范数和损失变化调整
2. **权重平衡**: 通过不确定性权重观察任务平衡性
3. **物理约束**: 确保预测的物理参数在合理范围内
4. **注意力分析**: 利用注意力图验证模型理解的合理性

---

## 📚 参考资料

- [TensorBoard官方文档](https://www.tensorflow.org/tensorboard)
- [PyTorch TensorBoard集成](https://pytorch.org/docs/stable/tensorboard.html)
- [水下图像增强物理模型原理](./physics_model.md)
- [模型架构说明](./architecture.md)

---

## ⚙️ 记录频率配置

TensorBoard指标记录频率现在完全可配置，通过`configs/train.yaml`中的`visualization.tensorboard`部分控制：

### 📋 默认配置

```yaml
visualization:
  tensorboard:
    # 基础训练指标 (train/*, optimizer/*, gpu/*)
    train_metrics_freq: 1  # 每N步记录一次训练指标 (1=每步记录)
    
    # 物理参数指标 (physics/*)
    physics_metrics:
      enable: true  # 是否启用物理参数记录
      freq: 1  # 每N步记录一次物理参数 (1=每步记录)
    
    # 模型参数分布 (params/*)
    model_params:
      enable: false  # 是否启用参数直方图记录 (耗时，建议调试时启用)
      freq: 500  # 每N步记录一次参数分布
    
    # 梯度分布 (grads/*)
    gradients:
      enable: false  # 是否启用梯度直方图记录 (耗时，建议调试时启用)
      freq: 500  # 每N步记录一次梯度分布
    
    # 验证指标 (val/*)
    val_metrics_freq: 1  # 验证时每N个batch记录一次 (1=每个batch记录)
    
    # 不确定性权重 (uncertainty/*)
    uncertainty_weights:
      enable: true  # 是否记录不确定性权重
      freq: 10  # 每N步记录一次不确定性权重
    
    # 注意力图可视化 (attention/*)
    attention_maps:
      enable: true  # 是否保存注意力图
      freq: 100  # 每N步保存一次注意力图
```

### 🎯 性能优化建议

| 指标类型 | 推荐设置 | 性能影响 | 使用场景 |
|---------|----------|----------|----------|
| **训练指标** | `freq: 1` | 低 | 始终启用 |
| **物理参数** | `freq: 1` | 低 | 始终启用 |
| **参数分布** | `enable: false` | **高** | 仅调试时启用 |
| **梯度分布** | `enable: false` | **高** | 仅调试时启用 |
| **不确定性权重** | `freq: 10` | 低 | 根据需要调整 |
| **注意力图** | `freq: 100` | 中 | 根据需要调整 |

### ⚡ 训练速度优化

**快速训练模式**（推荐生产环境）：
```yaml
visualization:
  tensorboard:
    train_metrics_freq: 1
    physics_metrics: {enable: true, freq: 1}
    model_params: {enable: false}  # 关闭参数直方图
    gradients: {enable: false}     # 关闭梯度直方图
    uncertainty_weights: {enable: true, freq: 50}
    attention_maps: {enable: true, freq: 200}
```

**详细调试模式**（推荐开发调试）：
```yaml
visualization:
  tensorboard:
    train_metrics_freq: 1
    physics_metrics: {enable: true, freq: 1}
    model_params: {enable: true, freq: 100}   # 启用但降低频率
    gradients: {enable: true, freq: 100}      # 启用但降低频率
    uncertainty_weights: {enable: true, freq: 10}
    attention_maps: {enable: true, freq: 50}
```

**注意：** 参数和梯度直方图记录会显著影响训练速度，建议仅在需要深入分析模型行为时启用。

---

*最后更新: 2025-06-10*
*版本: 2.0 - 完全可配置的TensorBoard记录系统* 