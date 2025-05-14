#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model_visualizer.py - 水下图像增强模型可视化工具
-------------------------------------------------
用途：可视化UnderwaterEnhanceNet模型结构和数据流向，无需ONNX

使用示例：
python scripts/model_visualizer.py --checkpoint path/to/model.pth --output_dir ./model_viz
python scripts/model_visualizer.py --checkpoint path/to/model.pth --input_shape 1,3,512,512
"""

import os
import sys
import argparse
import torch
from collections import OrderedDict
import time
from datetime import datetime

# 添加项目根目录到 PATH
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)

# 导入模型定义
from modules.model import UnderwaterEnhanceNet

def parse_args():
    parser = argparse.ArgumentParser(description='可视化UnderwaterEnhanceNet模型结构')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型权重文件路径 (.pth/.pth.tar)')
    parser.add_argument('--output_dir', type=str, default='./model_viz',
                        help='可视化输出目录 (默认: ./model_viz)')
    parser.add_argument('--base_channels', type=int, default=48,
                        help='模型基础通道数 (默认: 48)')
    parser.add_argument('--levels', type=int, default=4,
                        help='编码器层级数 (默认: 4)')
    parser.add_argument('--heads', type=int, default=8,
                        help='Transformer 注意力头数 (默认: 8)')
    parser.add_argument('--bottleneck_blocks', type=int, default=4,
                        help='瓶颈层 Restormer 块数 (默认: 4)')
    parser.add_argument('--input_shape', type=str, default='1,3,256,256',
                        help='输入形状 (N,C,H,W 默认: 1,3,256,256)')
    parser.add_argument('--use_tensorboard', action='store_true',
                        help='使用TensorBoard可视化模型图')
    return parser.parse_args()

def load_model(checkpoint_path, args):
    """加载UnderwaterEnhanceNet模型和权重"""
    print(f"创建UnderwaterEnhanceNet模型 (base_channels={args.base_channels}, levels={args.levels})")
    model = UnderwaterEnhanceNet(
        base_channels=args.base_channels,
        levels=args.levels,
        heads=args.heads,
        bottleneck_blocks=args.bottleneck_blocks
    )
    
    print(f"从 {checkpoint_path} 加载权重")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    # 允许部分加载，有些层可能无法匹配
    model.load_state_dict(new_state_dict, strict=False)
    print("模型权重已加载，注意：有可能某些新添加的层使用了默认初始化。")
    model.eval()
    return model

def print_model_summary(model, input_shape):
    """打印模型结构摘要"""
    try:
        from torchinfo import summary
        
        # 解析输入形状
        dims = [int(x) for x in input_shape.split(',')]
        assert len(dims) == 4, "输入形状必须为 NCHW 格式"
        
        # 创建张量输入，而不是None
        dummy_input = torch.randn(*dims)
        dummy_depth = torch.randn(dims[0], 1, dims[2], dims[3])  # 创建一个与输入尺寸匹配的深度图
        dummy_gt = torch.randn(*dims)  # 创建一个与输入尺寸匹配的GT图像
        
        # 打印模型摘要（使用eval模式，不会激活新投影层）
        model.eval()
        print("\n=== 模型结构摘要 ===")
        
        # 尝试使用完整输入参数
        try:
            print(summary(model, 
                          input_data=[dummy_input, dummy_depth, dummy_gt],
                          depth=4, 
                          device="cpu", 
                          col_names=["input_size", "output_size", "num_params", "trainable"]))
        except Exception as e:
            print(f"完整模型摘要失败，尝试简化模式")
            # 如果完整模式失败，尝试简化模式，只检查子模块
            print("\n=== 模型主要组件 ===")
            
            # 打印编码器
            print("\n▶ 编码器:")
            try:
                print(summary(model.encoder, 
                              input_data=dummy_input,
                              depth=2, 
                              device="cpu", 
                              verbose=0,
                              col_names=["input_size", "output_size", "num_params"]))
            except Exception as e:
                print(f"无法显示编码器摘要")
            
            # 尝试获取一些模型信息用于手动打印
            print("\n=== 模型层级结构 ===")
            print_module_tree(model)
        
    except ImportError:
        print("\n请安装torchinfo以显示详细的模型结构: pip install torchinfo")
        # 简单打印模型结构
        print("\n=== 简略模型结构 ===")
        print(model)
        
        # 打印模型组件结构
        for name, module in model.named_children():
            print(f"\n-- {name} --")
            print(module)

def print_module_tree(module, prefix='', is_last=True, depth=0, max_depth=3):
    """递归打印模块树结构"""
    if depth > max_depth:
        return
    
    # 计算要打印的前缀
    branch = '└── ' if is_last else '├── '
    
    # 获取模块的类名
    class_name = module.__class__.__name__
    
    # 打印当前模块
    print(f"{prefix}{branch}{class_name}")
    
    # 准备子模块的前缀
    extension = '    ' if is_last else '│   '
    
    # 获取子模块
    children = list(module.named_children())
    
    # 递归打印子模块
    for i, (name, child) in enumerate(children):
        is_last_child = i == len(children) - 1
        print_module_tree(
            child, 
            prefix=prefix + extension, 
            is_last=is_last_child,
            depth=depth+1,
            max_depth=max_depth
        )

def visualize_to_tensorboard(model, args):
    """生成TensorBoard模型图可视化"""
    try:
        from torch.utils.tensorboard import SummaryWriter
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        log_dir = os.path.join(args.output_dir, f"graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # 解析输入形状
        input_shape = [int(x) for x in args.input_shape.split(',')]
        assert len(input_shape) == 4, "输入形状必须为 NCHW 格式"
        
        # 创建虚拟输入
        dummy_input = torch.randn(*input_shape)
        dummy_depth = torch.randn(input_shape[0], 1, input_shape[2], input_shape[3])
        dummy_gt = torch.randn(*input_shape)
        
        print(f"\n生成TensorBoard模型图到 {log_dir}")
        writer = SummaryWriter(log_dir)
        
        # 添加模型图，适当处理异常
        try:
            writer.add_graph(model, (dummy_input, dummy_depth, dummy_gt))
        except Exception as e:
            print(f"无法添加完整模型图: {str(e)}")
            print("尝试使用简化模型...")
            
            # 创建一个简化模型
            class SimplifiedModel(torch.nn.Module):
                def __init__(self, orig_model):
                    super().__init__()
                    self.orig_model = orig_model
                
                def forward(self, x):
                    with torch.no_grad():
                        return self.orig_model(x, None, None)[0]
            
            simple_model = SimplifiedModel(model)
            try:
                writer.add_graph(simple_model, dummy_input)
                print("已添加简化模型图")
            except Exception as e:
                print(f"添加简化模型图也失败: {str(e)}")
                
                # 尝试单独添加组件
                try:
                    print("尝试单独添加编码器组件...")
                    class EncoderWrapper(torch.nn.Module):
                        def __init__(self, encoder):
                            super().__init__()
                            self.encoder = encoder
                        
                        def forward(self, x):
                            return self.encoder(x, None, None)[0]
                    
                    writer.add_graph(EncoderWrapper(model.encoder), dummy_input)
                    print("编码器组件已添加")
                except Exception as e:
                    print(f"添加编码器组件失败: {str(e)}")
                
        writer.close()
        
        print(f"\n查看模型图：")
        print(f"1. 执行命令: tensorboard --logdir={log_dir}")
        print(f"2. 打开浏览器访问: http://localhost:6006")
    
    except ImportError:
        print("\n请安装tensorboard以可视化模型图: pip install tensorboard")

def generate_python_model_graph(model, args):
    """生成Python代码表示的模型结构"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 解析输入形状
    input_shape = [int(x) for x in args.input_shape.split(',')]
    assert len(input_shape) == 4, "输入形状必须为 NCHW 格式"
    
    # 创建虚拟输入
    dummy_input = torch.randn(*input_shape)
    dummy_depth = torch.randn(input_shape[0], 1, input_shape[2], input_shape[3])
    dummy_gt = torch.randn(*input_shape)
    
    # 首先使用torchviz尝试生成计算图
    try:
        from torchviz import make_dot
        # 执行一次前向传播
        with torch.no_grad():
            try:
                outputs = model(dummy_input, None, None)
                # 生成计算图
                dot = make_dot(outputs[0], params=dict(model.named_parameters()))
                dot.format = 'png'
                dot.render(os.path.join(args.output_dir, "model_graph"), cleanup=True)
                print(f"\n模型图已生成: {os.path.join(args.output_dir, 'model_graph.png')}")
            except Exception as e:
                print(f"生成模型图失败: {str(e)}")
                # 保存模型结构到文本文件
                save_model_structure_to_file(model, args.output_dir)
    except ImportError:
        print("\n请安装graphviz和torchviz以生成模型计算图: pip install torchviz graphviz")
        # 保存模型结构到文本文件
        save_model_structure_to_file(model, args.output_dir)

def save_model_structure_to_file(model, output_dir):
    """将模型结构保存到文本文件"""
    try:
        # 尝试获取模型的字符串表示
        model_str = str(model)
        # 保存到文件
        with open(os.path.join(output_dir, "model_structure.txt"), 'w') as f:
            f.write(model_str)
        print(f"\n模型结构已保存至: {os.path.join(output_dir, 'model_structure.txt')}")
        
        # 保存更详细的结构（包括参数计数）
        with open(os.path.join(output_dir, "model_structure_detailed.txt"), 'w') as f:
            # 写入模型类名
            f.write(f"模型类型: {model.__class__.__name__}\n\n")
            
            # 写入主要组件
            f.write("主要组件:\n")
            for name, child in model.named_children():
                num_params = sum(p.numel() for p in child.parameters())
                trainable_params = sum(p.numel() for p in child.parameters() if p.requires_grad)
                f.write(f"- {name}: {child.__class__.__name__} "
                        f"(参数数量: {num_params:,}, 可训练: {trainable_params:,})\n")
            
            # 写入总参数数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            f.write(f"\n总参数数量: {total_params:,}\n")
            f.write(f"可训练参数: {trainable_params:,}\n")
            
            # 写入详细的模块树
            f.write("\n模块层次结构:\n")
            capture_module_tree(model, f)
            
        print(f"详细模型结构已保存至: {os.path.join(output_dir, 'model_structure_detailed.txt')}")
        
        # 创建可视化的HTML报告
        create_html_report(model, output_dir)
    except Exception as e:
        print(f"保存模型结构失败: {str(e)}")

def create_html_report(model, output_dir):
    """创建包含模型结构的HTML可视化报告"""
    try:
        html_file = os.path.join(output_dir, 'model_structure.html')
        
        with open(html_file, 'w') as f:
            f.write('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>模型结构可视化</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                    h1, h2 { color: #333; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    .tree-view { margin-left: 20px; }
                    .tree-item { margin: 5px 0; }
                    .tree-branch { border-left: 1px solid #ddd; padding-left: 20px; }
                    .module-name { font-weight: bold; color: #0066cc; }
                    .param-count { color: #666; font-size: 0.9em; }
                    .trainable { color: green; }
                    .not-trainable { color: #999; }
                    .summary-box { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }
                    .flex-container { display: flex; flex-wrap: wrap; }
                    .flex-item { flex: 1; min-width: 300px; margin: 10px; }
                    .layer-module { margin-bottom: 10px; padding: 5px; border-bottom: 1px solid #eee; }
                    
                    /* 折叠效果 */
                    .collapsible { 
                        cursor: pointer; 
                        background-color: #f9f9f9; 
                        padding: 5px 10px;
                        border-radius: 3px;
                    }
                    .active, .collapsible:hover { background-color: #eee; }
                    .content { 
                        max-height: 0;
                        overflow: hidden;
                        transition: max-height 0.2s ease-out;
                    }
                </style>
                <script>
                    window.onload = function() {
                        var coll = document.getElementsByClassName("collapsible");
                        for (var i = 0; i < coll.length; i++) {
                            coll[i].addEventListener("click", function() {
                                this.classList.toggle("active");
                                var content = this.nextElementSibling;
                                if (content.style.maxHeight) {
                                    content.style.maxHeight = null;
                                } else {
                                    content.style.maxHeight = content.scrollHeight + "px";
                                }
                            });
                        }
                    }
                </script>
            </head>
            <body>
                <div class="container">
                    <h1>UnderwaterEnhanceNet 模型结构报告</h1>
            ''')
            
            # 写入模型摘要
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            f.write(f'''
                <div class="summary-box">
                    <h2>模型摘要</h2>
                    <p><strong>模型类型:</strong> {model.__class__.__name__}</p>
                    <p><strong>总参数数量:</strong> {total_params:,}</p>
                    <p><strong>可训练参数:</strong> {trainable_params:,} ({trainable_params/total_params*100:.2f}%)</p>
                </div>
            ''')
            
            # 写入主要组件
            f.write('''
                <div class="flex-container">
                    <div class="flex-item">
                        <h2>主要组件</h2>
                        <div class="tree-view">
            ''')
            
            for name, child in model.named_children():
                num_params = sum(p.numel() for p in child.parameters())
                trainable_params = sum(p.numel() for p in child.parameters() if p.requires_grad)
                trainable_percent = trainable_params/num_params*100 if num_params > 0 else 0
                
                f.write(f'''
                    <div class="tree-item">
                        <span class="module-name">{name}</span>: {child.__class__.__name__}
                        <span class="param-count">
                            (参数: {num_params:,}, <span class="trainable">可训练: {trainable_params:,}</span> - {trainable_percent:.1f}%)
                        </span>
                    </div>
                ''')
            
            f.write('''
                        </div>
                    </div>
            ''')
            
            # 写入分层模块树视图（可折叠）
            f.write('''
                    <div class="flex-item">
                        <h2>详细层次结构</h2>
                        <p>点击展开/折叠详细结构</p>
            ''')
            
            generate_collapsible_tree_html(model, f)
            
            f.write('''
                    </div>
                </div>
            ''')
            
            # 写入结尾
            f.write('''
                </div>
            </body>
            </html>
            ''')
        
        print(f"HTML可视化报告已生成: {html_file}")
    except Exception as e:
        print(f"创建HTML报告失败: {str(e)}")

def generate_collapsible_tree_html(module, file, prefix='', depth=0):
    """生成可折叠的模块树HTML"""
    # 获取模块的类名
    class_name = module.__class__.__name__
    
    # 获取参数数量
    num_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    # 获取子模块
    children = list(module.named_children())
    
    if depth == 0:
        # 顶层模块
        file.write(f'''
            <button class="collapsible"><span class="module-name">{class_name}</span> (参数: {num_params:,})</button>
            <div class="content tree-branch">
        ''')
    
        # 递归处理子模块
        for i, (name, child) in enumerate(children):
            file.write(f'''
                <div class="layer-module">
                    <button class="collapsible"><span class="module-name">{name}</span>: {child.__class__.__name__} (参数: {sum(p.numel() for p in child.parameters()):,})</button>
                    <div class="content tree-branch">
            ''')
            
            # 递归子模块的子模块
            child_children = list(child.named_children())
            for j, (child_name, grandchild) in enumerate(child_children):
                file.write(f'''
                    <div class="tree-item">
                        <span class="module-name">{child_name}</span>: {grandchild.__class__.__name__} 
                        (参数: {sum(p.numel() for p in grandchild.parameters()):,})
                    </div>
                ''')
            
            file.write('''
                    </div>
                </div>
            ''')
        
        file.write('''
            </div>
        ''')
    
def capture_module_tree(module, file, prefix='', is_last=True, depth=0):
    """将模块树结构写入文件"""
    # 计算要打印的前缀
    branch = '└── ' if is_last else '├── '
    
    # 获取模块的类名
    class_name = module.__class__.__name__
    
    # 获取参数数量
    num_params = sum(p.numel() for p in module.parameters())
    
    # 写入当前模块
    if depth > 0:  # 跳过顶层模块（已经在上面打印过了）
        file.write(f"{prefix}{branch}{class_name} (参数: {num_params:,})\n")
    
    # 准备子模块的前缀
    extension = '    ' if is_last else '│   '
    
    # 获取子模块
    children = list(module.named_children())
    
    # 递归写入子模块
    for i, (name, child) in enumerate(children):
        is_last_child = i == len(children) - 1
        capture_module_tree(
            child, 
            file,
            prefix=prefix + extension, 
            is_last=is_last_child,
            depth=depth+1
        )

def analyze_model_feature_maps(model, input_shape, output_dir):
    """分析模型的特征图尺寸和通道数"""
    # 创建特征图信息目录
    feature_map_dir = os.path.join(output_dir, 'feature_maps')
    os.makedirs(feature_map_dir, exist_ok=True)
    
    # 解析输入形状
    dims = [int(x) for x in input_shape.split(',')]
    assert len(dims) == 4, "输入形状必须为 NCHW 格式"
    
    # 创建张量输入
    dummy_input = torch.randn(*dims)
    
    # 模型的输出特征
    model_features = {}
    
    # 为每个模块注册钩子
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor):
                        model_features[f"{name}_output_{i}"] = out.shape
            elif isinstance(output, list):
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor):
                        model_features[f"{name}_output_{i}"] = out.shape
            elif isinstance(output, torch.Tensor):
                model_features[name] = output.shape
        return hook
    
    # 为主要组件注册钩子
    for name, module in model.named_children():
        hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # 进行前向传播
    try:
        with torch.no_grad():
            outputs = model(dummy_input, None, None)
    except Exception as e:
        print(f"模型特征图分析过程中发生错误: {str(e)}")
    finally:
        # 移除所有钩子
        for hook in hooks:
            hook.remove()
    
    # 保存特征图信息
    with open(os.path.join(feature_map_dir, 'feature_map_sizes.txt'), 'w') as f:
        f.write("模型特征图尺寸信息\n")
        f.write("=================\n\n")
        f.write(f"输入形状: {dims}\n\n")
        
        f.write("主要组件输出形状:\n")
        for name, shape in model_features.items():
            f.write(f"- {name}: {shape}\n")
    
    print(f"特征图尺寸信息已保存至: {os.path.join(feature_map_dir, 'feature_map_sizes.txt')}")
    
    # 创建可视化HTML报告
    create_feature_map_html(model_features, dims, feature_map_dir)

def create_feature_map_html(features, input_shape, output_dir):
    """创建特征图可视化的HTML报告"""
    try:
        html_file = os.path.join(output_dir, 'feature_maps.html')
        
        with open(html_file, 'w') as f:
            f.write('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>模型特征图可视化</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                    h1, h2 { color: #333; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
                    th { background-color: #f2f2f2; }
                    tr:hover { background-color: #f5f5f5; }
                    .summary-box { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }
                    .feature-map { 
                        display: inline-block; 
                        margin: 10px;
                        padding: 5px;
                        border: 1px solid #ddd;
                        background-color: #fff;
                        border-radius: 3px;
                    }
                    .feature-map-container {
                        display: flex;
                        flex-wrap: wrap;
                        justify-content: flex-start;
                    }
                    .size-info {
                        font-size: 0.8em;
                        color: #666;
                    }
                    .feature-map-visual {
                        width: 120px;
                        height: 80px;
                        background-color: #eee;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        position: relative;
                        overflow: hidden;
                    }
                    .feature-box {
                        position: absolute;
                        background-color: rgba(0, 102, 204, 0.5);
                        border: 1px solid #0066cc;
                    }
                    .channel-count {
                        font-weight: bold;
                        color: #0066cc;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>模型特征图可视化</h1>
            ''')
            
            # 写入输入信息
            f.write(f'''
                <div class="summary-box">
                    <h2>输入信息</h2>
                    <p><strong>输入形状:</strong> {input_shape} (批次大小, 通道数, 高度, 宽度)</p>
                </div>
            ''')
            
            # 写入特征图表格
            f.write('''
                <h2>特征图尺寸</h2>
                <table>
                    <tr>
                        <th>组件名称</th>
                        <th>输出形状</th>
                        <th>通道数</th>
                        <th>空间尺寸</th>
                        <th>内存占用 (MB)</th>
                    </tr>
            ''')
            
            # 添加输入行
            batch, channels, height, width = input_shape
            memory = batch * channels * height * width * 4 / (1024 * 1024)  # 4 bytes per float32, convert to MB
            
            f.write(f'''
                <tr>
                    <td><strong>输入</strong></td>
                    <td>{input_shape}</td>
                    <td>{channels}</td>
                    <td>{height} × {width}</td>
                    <td>{memory:.2f}</td>
                </tr>
            ''')
            
            # 添加特征图行
            for name, shape in features.items():
                if len(shape) == 4:  # 确保是4D张量 (B,C,H,W)
                    b, c, h, w = shape
                    mem = b * c * h * w * 4 / (1024 * 1024)  # 4 bytes per float32, convert to MB
                    
                    f.write(f'''
                        <tr>
                            <td>{name}</td>
                            <td>{shape}</td>
                            <td>{c}</td>
                            <td>{h} × {w}</td>
                            <td>{mem:.2f}</td>
                        </tr>
                    ''')
            
            f.write('''
                </table>
            ''')
            
            # 写入可视化特征图
            f.write('''
                <h2>特征图可视化</h2>
                <div class="feature-map-container">
            ''')
            
            # 添加输入可视化
            batch, channels, height, width = input_shape
            scale_factor = min(100 / height, 100 / width)
            visual_height = int(height * scale_factor)
            visual_width = int(width * scale_factor)
            
            f.write(f'''
                <div class="feature-map">
                    <div class="feature-map-visual">
                        <div class="feature-box" style="width:{visual_width}px;height:{visual_height}px;"></div>
                    </div>
                    <div class="size-info">
                        <strong>输入</strong><br>
                        尺寸: {height}×{width}<br>
                        通道: <span class="channel-count">{channels}</span>
                    </div>
                </div>
            ''')
            
            # 添加特征图可视化
            for name, shape in features.items():
                if len(shape) == 4:  # 确保是4D张量 (B,C,H,W)
                    b, c, h, w = shape
                    scale = min(100 / h, 100 / w) if h > 0 and w > 0 else 1
                    v_height = max(1, int(h * scale))
                    v_width = max(1, int(w * scale))
                    
                    f.write(f'''
                        <div class="feature-map">
                            <div class="feature-map-visual">
                                <div class="feature-box" style="width:{v_width}px;height:{v_height}px;"></div>
                            </div>
                            <div class="size-info">
                                <strong>{name}</strong><br>
                                尺寸: {h}×{w}<br>
                                通道: <span class="channel-count">{c}</span>
                            </div>
                        </div>
                    ''')
            
            f.write('''
                </div>
            ''')
            
            # 写入结尾
            f.write('''
                </div>
            </body>
            </html>
            ''')
        
        print(f"特征图HTML可视化报告已生成: {html_file}")
    except Exception as e:
        print(f"创建特征图HTML报告失败: {str(e)}")

def main():
    args = parse_args()
    
    # 检查必要的依赖
    required_packages = []
    try:
        import torchinfo
    except ImportError:
        required_packages.append("torchinfo")
        
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        required_packages.append("tensorboard")
    
    if required_packages:
        print(f"请安装以下依赖以获得最佳体验: pip install {' '.join(required_packages)}")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model = load_model(args.checkpoint, args)
    
    # 打印模型摘要
    print_model_summary(model, args.input_shape)
    
    # 生成Python模型图
    generate_python_model_graph(model, args)
    
    # 分析模型特征图
    try:
        analyze_model_feature_maps(model, args.input_shape, args.output_dir)
    except Exception as e:
        print(f"特征图分析失败: {str(e)}")
    
    print("\n可视化完成！")
    print(f"结果已保存到目录: {os.path.abspath(args.output_dir)}")
    print(f"可以使用Web浏览器打开生成的HTML文件查看更详细的可视化报告。")

if __name__ == '__main__':
    main()
