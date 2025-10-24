# 深度图格式统一说明

## 🎯 统一配置

### 训练和推理
- **格式**: PNG（16位灰度图）
- **像素值范围**: 0-65535
- **输出**: 模型直接输出PNG格式深度图

### 验证集深度图
- **格式**: 统一使用PNG（16位）
- **单位**: `depth_gt_units: "pixel"`
- **范围**: `depth_valid_min: 0.0`, `depth_valid_max: 65535.0`
- **自适应缩放**: `use_adaptive_scale: true`

---

## 📋 已更新的验证集配置

### 1. UBB深度验证集 ✅
```yaml
ubb_depth:
  depth_gt_units: "pixel"       # 像素值
  depth_valid_min: 0.0
  depth_valid_max: 65535.0
  use_adaptive_scale: true      # 自适应缩放
  depth_format: "png"           # PNG格式
```

### 2. SeaThru深度 ✅
```yaml
seathru_depth:
  depth_gt_units: "pixel"       # 从"m"改为"pixel"
  use_adaptive_scale: true      # 新增
  depth_format: "png"           # 从"npy"改为"png"
```

### 3. SQUID深度 ✅
```yaml
squid_depth:
  depth_gt_units: "pixel"       # 从"mm"改为"pixel"
  use_adaptive_scale: true      # 新增
  depth_format: "png"           # 从"npy"改为"png"
```

---

## 🔧 自适应缩放说明

**`use_adaptive_scale: true`** 的作用：
- 自动检测深度图的实际像素值范围
- 归一化到模型输出范围
- 计算指标时使用相对误差

**适用场景**:
- 不同数据集的深度范围不同
- PNG格式的16位深度图（0-65535）
- 避免硬编码最小/最大深度值

---

## ✅ 验证指标计算

所有深度验证集使用统一的指标：
- `depth_mae` - 平均绝对误差
- `depth_rmse` - 均方根误差
- `depth_abs_rel` - 相对绝对误差
- `depth_sq_rel` - 相对平方误差
- `depth_log10` - 对数误差
- `depth_delta1/2/3` - 阈值准确率

---

## 📊 数据格式总结

| 验证集 | 格式 | 单位 | 自适应缩放 | 状态 |
|--------|------|------|-----------|------|
| UBB_depth | PNG | pixel | ✅ | 启用 |
| SeaThru | PNG | pixel | ✅ | 禁用（可启用）|
| SQUID | PNG | pixel | ✅ | 禁用（可启用）|

**所有深度验证集已统一为PNG格式+自适应缩放！** ✅




