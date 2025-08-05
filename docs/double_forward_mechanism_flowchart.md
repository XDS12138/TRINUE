# 双次前向机制流程图 (Double Forward Mechanism Flowchart)

本文档详细描述了UnderwaterEnhanceNet的核心创新——双次前向机制的流程图。

## 流程图概览

```ascii
┌─────────────────────────────────────────────────────────────────────────────┐
│                      双次前向机制 (Double Forward Mechanism)                  │
│                                                                            │
│  Input: Multi-Degradation [B,N,3,H,W]                                     │
│                              │                                             │
│                              ▼                                             │
│                     ┌─────────────────┐                                    │
│                     │ Multi-Input     │                                    │
│                     │ Processing      │                                    │
│                     │ raw_flat =      │                                    │
│                     │ raw.reshape(    │                                    │
│                     │   B*N,3,H,W)    │                                    │
│                     └─────────┬───────┘                                    │
│                               │                                             │
│                               ▼                                             │
│                     ┌─────────────────┐                                    │
│                     │ SFE (Shallow    │                                    │
│                     │ Feature Extract)│                                    │
│                     │ [B*N,3,H,W] →   │                                    │
│                     │ [B*N,C,H,W]     │                                    │
│                     └─────────┬───────┘                                    │
│                               │                                             │
│    ┌─────────────────────────▼─────────────────────────┐                  │
│    │              🔥 共享RGB编码器                       │                  │
│    │        (Shared RGB Encoder with Attention)        │                  │
│    │                                                   │                  │
│    │  ┌───────────────┐              ┌──────────────┐  │                  │
│    │  │   Pass-1      │              │   Pass-2     │  │                  │
│    │  │   (R2D)       │              │   (D2R)      │  │                  │
│    │  │               │              │              │  │                  │
│    │  │ RGB Features  │              │RGB Features  │  │                  │
│    │  │      ↓        │    循环反馈    │     ↓        │  │                  │
│    │  │ 双向交叉注意力   │◄──────────────┤双向交叉注意力 │  │                  │
│    │  │(Bi-directional│              │(Bi-direction)│  │                  │
│    │  │Cross-Attention)│              │Cross-Attention│  │                  │
│    │  │   RGB↔Depth   │              │  RGB↔Depth   │  │                  │
│    │  │               │              │              │  │                  │
│    │  │ ↓ 输出特征     │              │ ↓ 输出特征    │  │                  │
│    │  │ student_feats │              │student_feats │  │                  │
│    │  │ bottleneck    │              │bottleneck    │  │                  │
│    │  └───────┬───────┘              └──────┬───────┘  │                  │
│    └──────────┼─────────────────────────────┼──────────┘                  │
│               │                             │                             │
│               ▼                             ▼                             │
│    ┌─────────────────┐              ┌──────────────┐                      │
│    │ Depth Decoder   │              │ Physics      │                      │
│    │(U-Net,非残差网络)│              │ Param Head   │                      │
│    │                 │              │              │                      │
│    │Skip Connections │              │ Bottleneck → │                      │
│    │torch.cat(...)   │              │ beta_c, B_c, │                      │
│    │                 │              │ blur_scale   │                      │
│    │ ↓ 输出          │              └──────┬───────┘                      │
│    │depth_pred       │                     │                             │
│    │[B*N,1,H,W]      │                     │                             │
│    │depth_feats      │                     │                             │
│    └─────────┬───────┘                     │                             │
│              │                             │                             │
│              ▼                             │                             │
│    ┌─────────────────┐                     │                             │
│    │Dynamic Channel  │                     │                             │
│    │Projection       │                     │                             │
│    │适配深度特征通道  │                     │                             │
│    └─────────┬───────┘                     │                             │
│              │                             │                             │
│              └──────────┬──────────────────┘                             │
│                         │                                                 │
│                         ▼                                                 │
│                ┌─────────────────────────┐                               │
│                │   MultiTask Decoder     │                               │
│                │   (双分支解码器)         │                               │
│                │                         │                               │
│                │  ┌─────────────────────┐│                               │
│                │  │   Deblur Branch     ││                               │
│                │  │ ┌─────────────────┐ ││                               │
│                │  │ │Level 0: Insert  │ ││                               │
│                │  │ │PSF-PML + Blocks │ ││                               │
│                │  │ ├─────────────────┤ ││                               │
│                │  │ │Level 1: Insert  │ ││                               │
│                │  │ │PSF-PML + Blocks │ ││                               │
│                │  │ ├─────────────────┤ ││                               │
│                │  │ │Level 2: Insert  │ ││                               │
│                │  │ │PSF-PML + Blocks │ ││                               │
│                │  │ └─────────────────┘ ││                               │
│                │  │ → res_d [B*N,3,H,W] ││                               │
│                │  └─────────────────────┘│                               │
│                │                         │                               │
│                │  ┌─────────────────────┐│                               │
│                │  │   Color Branch      ││                               │
│                │  │ ┌─────────────────┐ ││                               │
│                │  │ │Level 0: Insert  │ ││                               │
│                │  │ │BL-PML + Blocks  │ ││                               │
│                │  │ ├─────────────────┤ ││                               │
│                │  │ │Level 1: Insert  │ ││                               │
│                │  │ │BL-PML + Blocks  │ ││                               │
│                │  │ ├─────────────────┤ ││                               │
│                │  │ │Level 2: Insert  │ ││                               │
│                │  │ │BL-PML + Blocks  │ ││                               │
│                │  │ └─────────────────┘ ││                               │
│                │  │ → res_c [B*N,3,H,W] ││                               │
│                │  └─────────────────────┘│                               │
│                └─────────┬───────────────┘                               │
│                          │                                               │
│                          ▼                                               │
│                ┌─────────────────────────┐                               │
│                │   MultiTask Head        │                               │
│                │(ReconHead: 残差融合)     │                               │
│                │                         │                               │
│                │enhanced = raw_flat +    │                               │
│                │         res_d + res_c   │                               │
│                │depth_refine = depth_    │                               │
│                │              head(feat) │                               │
│                │                         │                               │
│                │输出: enhanced_flat,     │                               │
│                │     depth_refine_flat   │                               │
│                └─────────┬───────────────┘                               │
│                          │                                               │
│                          ▼                                               │
│                ┌─────────────────────────┐                               │
│                │   Output Reshape &      │                               │
│                │   Assembly              │                               │
│                │                         │                               │
│                │enhanced = enhanced_flat.│                               │
│                │  reshape(B,N,3,H,W)     │                               │
│                │depth_pred = depth_      │                               │
│                │  refine_flat.reshape    │                               │
│                │  (B,N,1,H,W)            │                               │
│                │res_d = res_d_flat.      │                               │
│                │  reshape(B,N,3,H,W)     │                               │
│                │res_c = res_c_flat.      │                               │
│                │  reshape(B,N,3,H,W)     │                               │
│                └─────────┬───────────────┘                               │
│                          │                                               │
│                          ▼                                               │
│                ┌─────────────────────────┐                               │
│                │ Ensemble & Final Output │                               │
│                │                         │                               │
│                │primary_enhanced =       │                               │
│                │  torch.mean(enhanced,   │                               │
│                │            dim=1)       │                               │
│                │primary_depth_pred =     │                               │
│                │  torch.mean(depth_pred, │                               │
│                │            dim=1)       │                               │
│                │                         │                               │
│                │Final Outputs:           │                               │
│                │- enhanced [B,3,H,W]     │                               │
│                │- depth_pred [B,1,H,W]   │                               │
│                │- multi_enhanced         │                               │
│                │  [B,N,3,H,W]            │                               │
│                │- multi_depth [B,N,1,H,W]│                               │
│                │- multi_res_d [B,N,3,H,W]│                               │
│                │- multi_res_c [B,N,3,H,W]│                               │
│                └─────────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 关键组件详解

### 1. Multi-Input Processing
- **功能**: 处理多退化输入，展平为批次维度
- **输入**: Multi-degradation images `[B,N,3,H,W]`
- **输出**: Flattened images `raw_flat [B*N,3,H,W]`
- **说明**: `raw_flat`来源于`raw.reshape(B*N, 3, H, W)`的展平操作

### 2. 共享RGB编码器 (Shared RGB Encoder)
- **核心创新**: 同一编码器在两次前向中被重复使用
- **双向交叉注意力**: RGB↔Depth特征的双向信息交换
- **循环机制**: Pass-1的输出指导Pass-2的输入，形成特征学习循环

### 3. Depth Decoder (U-Net架构，非残差网络)
- **功能**: 从 RGB 特征预测深度图和多尺度深度特征
- **架构**: U-Net with Skip Connections (跳跃连接，非残差连接)
  ```python
  x = torch.cat([upsampled, projected_skip], dim=1)  # 拼接而非相加
  x = fusion_proj(x)  # 通道融合
  x = RestormerBlock(x)  # 特征精炼
  ```
- **输入**: Bottleneck features + Student features (Pass-1)
- **输出**: 
  - `depth_pred`: 连续深度预测 [B*N,1,H,W] - **单通道深度**
  - `depth_feats`: 多尺度深度特征列表

### 4. MultiTask Decoder (双分支解码器)
- **功能**: 双分支解码，分别处理去模糊和颜色校正
- **输入**: Bottleneck features + Student features + Depth features + Physics parameters
- **架构**: 
   - **Deblur Branch**: 使用PSF-PML进行点扩散函数调制
     - 每个解码层都 **Insert** PSF-PML模块
     - 输入：融合特征、原图、归一化深度、模糊尺度
     - 输出：去模糊残差 `res_d`
   
   - **Color Branch**: 使用Beer-Lambert PML进行颜色校正
     - 每个解码层都 **Insert** Beer-Lambert PML模块
     - 输入：融合特征、原图、深度、物理参数（β_c, B_c）
     - 输出：颜色残差 `res_c`

   **注**: 流程图中的 "Insert" 表示将物理调制层(PML)插入到常规解码流程中，实现深度引导的物理建模。

### 5. 维度匹配机制
深度和RGB的维度不匹配问题通过以下方式解决：

#### **在物理调制层中**：
```python
depth_resized: [B*N,1,H_i,W_i]               # 单通道深度
depth_3ch = depth_resized.repeat(1,3,1,1)     # [B*N,1,H_i,W_i] → [B*N,3,H_i,W_i]
                                              # 🔥 扩展为3通道匹配RGB
t = exp(-beta_c * depth_3ch)                  # 透射率 (transmission) [B*N,3,H_i,W_i]
b = 1 - t                                     # 后向散射 (backscatter) [B*N,3,H_i,W_i]
I_tilde = 2 * I_raw_resized - 1               # 归一化原图到[-1,1] [B*N,3,H_i,W_i]
```

#### **在最终输出中**：
```python
enhanced = raw_flat + res_d + res_c  # 都是 [B*N,3,H,W] - RGB 3通道
depth_pred = depth_head(feat)        # 输出 [B*N,1,H,W] - 深度 1通道
# 深度和RGB分别处理，不需要维度匹配
```

## 分层深度引导颜色恢复机制 (Hierarchical Depth-Guided Color Restoration)

### 语义层次 (Semantic Level)
- **Dynamic Channel Projection**: 适配深度特征通道维度
- **Bi-directional Cross-Attention**: RGB↔Depth特征交换

### 物理层次 (Physical Level)  
- **Beer-Lambert PML**: 基于Beer-Lambert定律的物理颜色校正
- **PSF PML**: 基于点扩散函数的去模糊调制

### 像素层次 (Pixel Level)
- **Residual Fusion**: 残差学习与图像重建
- **Enhanced Image**: `enhanced = raw + res_d + res_c`

## 术语说明

### 残差 (Residuals) vs. 增强图像 (Enhanced Images)
- **res_d**: 去模糊残差 (Deblur Residuals) - 学习的高频细节补偿
- **res_c**: 颜色残差 (Color Residuals) - 学习的颜色校正补偿  
- **enhanced**: 最终增强图像 (Enhanced Images) - 原图+残差的融合结果

### 关键设计思想
- **深度信息**: 保持单通道 [B*N,1,H,W]，专注于几何信息
- **RGB信息**: 保持三通道 [B*N,3,H,W]，处理颜色信息  
- **物理建模**: 通过repeat操作临时扩展深度到3通道进行物理计算
- **最终输出**: 深度和RGB分别输出，各司其职

这种设计既保持了深度信息的几何纯净性，又实现了与RGB信息的有效融合。 