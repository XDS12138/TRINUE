#提前构建lmdb用于数据加载训练
# UnderwaterEnhance 项目概览

```
UnderwaterEnhance/
├── configs/                     # 配置文件
│   ├── model.yaml               # 模型结构与模块开关定义
│   ├── train.yaml               # 训练超参、数据路径、损失权重等
│   └── infer.yaml               # 推理设置、输出路径、深度预测开关
│
├── data/                        # 数据预处理脚本
│   └── preprocess.py            # 同步几何 & 光度增强，多尺度输出
│
├── modules/                     # 核心模块实现
│   ├── sfe.py                   # Shallow Feature Extractor
│   ├── depth_feature_extractor.py # 训练期深度特征提取
│   ├── depth_head.py            # Mono-Depth Head (蒸馏 & 推理)
│   ├── cross_attention.py       # RGB←Depth 跨模态注意力
│   ├── teacher_student.py       # Teacher-Student 对齐损失
│   ├── encoder.py               # Raw/GT 编码器与 Cross‑Attention 融合
│   ├── depth_gate.py            # Depth-Gate 跳跃连接辅助
│   ├── decoder.py               # 双分支解码器: 高频去模糊(密集跳跃+多尺度深度门控) & 低频色彩校正
│   ├── recon_head.py            # 重建输出头
│   └── loss_fn.py               # 各类损失函数集 (图像/对齐/深度)
│
├── scripts/                     # 运行脚本
│   ├── train.py                 # 训练入口，支持蒸馏 & 剪枝
│   ├── validate.py              # 验证入口(PSNR/SSIM等)
│   ├── infer.py                 # 推理入口，单输入双输出
│   └── export_onnx.py           # 导出 ONNX / TensorRT
│
├── utils/                       # 工具函数与辅助脚本
│   ├── logger.py                # 日志 & TensorBoard 可视化
│   ├── checkpoint.py            # 模型保存/加载
│   ├── lr_scheduler.py          # 学习率调度器
│   ├── metrics.py               # PSNR/SSIM/NIQE 等评测
│   ├── visualization.py         # 中间特征 & Gate 热图
│
├── tests/                       # 单元 & 集成测试
│   ├── test_preprocess.py
│   ├── test_model_forward.py
│   └── test_loss.py
│
├── experiments/                 # 试验记录目录
│   └── run001/                  # 每次实验独立子文件夹
│       ├── train.log            # 训练日志
│       ├── checkpoint.pth       # 模型权重
│       └── config.yaml          # 该次实验配置
│
├── docs/                        # 文档与报告
│   ├── design.md                # 详细设计说明
│   ├── api.md                   # 模块/接口使用说明
│   ├── architecture.md          # 网络结构图与流程
│   └── benchmark.md             # 性能对比与实验结果
│
├── .gitignore                   # Git 忽略规则
├── environment.yml              # Conda 环境描述
├── Dockerfile                   # 容器化部署定义
├── Makefile                     # 常用命令汇总
├── LICENSE                      # 开源许可协议
├── CONTRIBUTING.md              # 贡献指南
├── README.md                    # 项目概览 (当前文件)
└── requirements.txt             # Python 依赖列表
```

## 目录说明

**configs/** 目录下统一管理模型、训练、推理所有超参与数据路径，便于一键复现。

**data/preprocess.py** 实现对齐读取、几何与光度增强、Tensor 转换与多尺度下采。

**modules/** 下各文件对应核心模块：

* *sfe.py*：浅层卷积特征提取
* *depth\_feature\_extractor.py*：训练期真深度多尺度编码
* *depth\_head.py*：蒸馏 & 推理深度门控头
* *cross\_attention.py*：RGB and Depth 跨模态注意力
* *teacher\_student.py*：对齐损失计算
* *encoder.py*：集成 Raw/GT 编码与融合逻辑
* *depth\_gate.py*：统一跳跃连接过滤函数
* *decoder.py*：高频去模糊(密集跳跃连接与多尺度深度门控) & 低频色彩两分支
* *recon\_head.py*：最终图像重建
* *loss\_fn.py*：聚合所有损失公式与权重

**scripts/** 提供训练、验证、推理与导出脚本，逻辑清晰、参数可配置。

**utils/** 封装日志、检查点、学习率、评测指标、可视化以及蒸馏/对齐辅助函数。

**tests/** 用于自动化测试，保证数据与模型前向、损失计算等模块正确性。

**experiments/** 建议按每次试验新建子文件夹，保存日志、权重与配置，便于版本管理。

**docs/** 存放详细设计、API 文档、架构流程图与实验结果报告，方便撰写论文或内部评审。

根目录下的配置与辅助文件（`.gitignore`, `Dockerfile`, `Makefile`, `LICENSE`, `CONTRIBUTING.md` 等）保证项目可持续集成与高效协作。
