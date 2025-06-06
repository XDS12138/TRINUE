# TRINUE 项目结构概览

```text
TRINUE/
├── configs/                  # 训练与模型配置
│   └── train.yaml
├── data/                     # 数据缓存或预处理产物
├── docs/                     # 项目文档（本文件等）
├── examples/                 # 使用示例
│   └── multi_logger_example.py
├── modules/                  # 核心网络与数据集实现
│   ├── blocks.py             # 网络基础模块
│   ├── cross_attention.py    # 深度与RGB交叉注意力
│   ├── datasets.py           # 数据加载与增强
│   ├── decoder.py            # 增强解码器
│   ├── depth.py              # 深度特征与工具
│   ├── depth_decoder.py      # 连续深度解码
│   ├── encoder.py            # 多尺度编码器
│   ├── loss_fn.py            # 损失函数集合
│   ├── model.py              # 主模型组装
│   ├── recon_head.py         # 重建输出头
│   └── sfe.py                # Shallow Feature Extractor
├── scripts/                  # 运行脚本
│   ├── train.py              # 训练入口
│   ├── launch_tensorboard.py # 启动 TensorBoard
│   ├── model_visualizer.py   # 模型可视化
│   ├── view_logs.sh          # 查看日志
│   └── train.py.bak          # 旧版备份
├── utils/                    # 辅助工具
│   ├── checkpoint.py         # 权重保存/加载
│   ├── logger.py             # 日志系统
│   ├── lr_scheduler.py       # 学习率调度
│   ├── metrics.py            # 评测指标
│   ├── metrics_fixed.py      # 修正版指标
│   └── multi_logger.py       # 多文件日志管理
├── model_viz/                # 模型结构与特征可视化
│   ├── model_structure.html
│   ├── model_structure.txt
│   ├── model_structure_detailed.txt
│   └── feature_maps/
├── environment.yml           # Conda 环境描述
├── requirements.txt          # Python 依赖列表
├── fix_indent.py             # 辅助脚本
├── .gitignore
└── Output/                   # 运行输出示例
```

## 目录说明

- **configs/**：集中管理训练与模型参数。
- **examples/**：提供日志系统等使用示例。
- **modules/**：网络主体与数据集逻辑所在，包含编码器、解码器、损失函数等核心代码。
- **scripts/**：训练、可视化及日志相关脚本入口。
- **utils/**：常用工具函数与多文件日志系统。
- **model_viz/**：保存模型结构文本与可视化结果，便于调试与展示。
- **docs/**：技术文档与研究笔记。
- 根目录下的 `environment.yml` 与 `requirements.txt` 记录依赖环境，`fix_indent.py` 为开发辅助脚本。
