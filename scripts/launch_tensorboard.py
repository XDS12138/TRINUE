#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
launch_tensorboard.py - AutoDL环境专用TensorBoard启动器
-----------------------------------------------
用途：在AutoDL环境中正确启动TensorBoard服务，适配AutoDL平台要求
按照AutoDL指导：先结束默认TensorBoard，再启动指向实验目录的TensorBoard

使用方法：
python scripts/launch_tensorboard.py [--port PORT] [--dir EXPERIMENT_DIR]
"""

import os
import sys
import glob
import argparse
import time
import subprocess
import yaml
from datetime import datetime

def load_config(config_path="configs/train.yaml"):
    """读取训练配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"⚠️  无法读取配置文件 {config_path}: {e}")
        return None

def get_output_dir_from_config(config_path="configs/train.yaml"):
    """从配置文件获取输出目录"""
    config = load_config(config_path)
    if config and 'experiment' in config and 'output_dir' in config['experiment']:
        output_dir = config['experiment']['output_dir']
        # 展开用户目录路径
        return os.path.expanduser(output_dir)
    return None

def kill_existing_tensorboard():
    """结束现有的TensorBoard进程（AutoDL要求）"""
    try:
        print("🔄 结束默认TensorBoard进程...")
        
        # 🔧 修复：更安全的进程查找，避免杀死脚本自身
        # 查找真正的 tensorboard 进程（排除 grep 和脚本自身）
        cmd = "ps -ef | grep 'tensorboard --' | grep -v grep | grep -v launch_tensorboard.py"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.stdout.strip():
            # 提取进程ID并杀死
            lines = result.stdout.strip().split('\n')
            killed_count = 0
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    pid = parts[1]
                    try:
                        subprocess.run(f"kill -9 {pid}", shell=True, stderr=subprocess.DEVNULL)
                        killed_count += 1
                        print(f"  ✓ 已结束进程 PID: {pid}")
                    except:
                        pass
            
            if killed_count > 0:
                time.sleep(2)  # 等待进程完全结束
                print(f"✅ 已结束 {killed_count} 个TensorBoard进程")
            else:
                print("✅ 没有发现需要结束的TensorBoard进程")
        else:
            print("✅ 没有发现运行中的TensorBoard进程")
            
    except Exception as e:
        print(f"⚠️  结束TensorBoard进程时出现警告: {e}")

def find_latest_experiment(base_dirs=None, config_path="configs/train.yaml"):
    """查找最近的实验目录，支持多个基础目录"""
    if base_dirs is None:
        # 🔥 优先从配置文件读取输出目录
        config_output_dir = get_output_dir_from_config(config_path)
        if config_output_dir:
            print(f"📋 从配置文件读取输出目录: {config_output_dir}")
            base_dirs = [config_output_dir]
        else:
            print("⚠️  无法从配置文件读取输出目录，使用默认路径")
            base_dirs = [
                "~/autodl-fs/experiments/train",  # AutoDL持久化存储
                "experiments/train",               # 本地实验目录
                "/root/autodl-fs/experiments/train"  # 绝对路径
            ]
    
    experiments = []
    
    for base_dir in base_dirs:
        # 展开用户目录
        expanded_base_dir = os.path.expanduser(base_dir)
        if not os.path.exists(expanded_base_dir):
            continue
            
        print(f"🔍 搜索实验目录: {expanded_base_dir}")
        
        for exp_dir in glob.glob(os.path.join(expanded_base_dir, "*")):
            if os.path.isdir(exp_dir):
                try:
                    # 尝试从目录名提取时间戳
                    dirname = os.path.basename(exp_dir)
                    if "_" in dirname and len(dirname.split("_")) >= 3:
                        # 目录名格式：underwater_enhance_run_20250731_225709
                        parts = dirname.split("_")
                        if len(parts) >= 4:
                            timestamp_str = f"{parts[-2]}_{parts[-1]}"
                            try:
                                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                                experiments.append((exp_dir, timestamp))
                                print(f"  📁 发现实验: {dirname} ({timestamp})")
                            except ValueError:
                                pass
                    
                    if not experiments or exp_dir not in [exp[0] for exp in experiments]:
                        # 使用目录修改时间作为备选
                        mtime = os.path.getmtime(exp_dir)
                        timestamp = datetime.fromtimestamp(mtime)
                        experiments.append((exp_dir, timestamp))
                        print(f"  📁 发现实验: {dirname} (按修改时间)")
                        
                except Exception as e:
                    print(f"  ⚠️  处理目录 {exp_dir} 时出错: {e}")
    
    if not experiments:
        return None
    
    # 按时间戳排序，返回最新的
    experiments.sort(key=lambda x: x[1], reverse=True)
    latest_exp = experiments[0][0]
    print(f"🎯 选择最新实验: {latest_exp}")
    return latest_exp

def find_tensorboard_dir(exp_dir):
    """查找实验目录中的TensorBoard日志目录"""
    possible_paths = [
        os.path.join(exp_dir, "tensorboard"),
        os.path.join(exp_dir, "logs"),
        exp_dir  # 如果直接在实验目录下
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # 检查是否包含event文件
            for root, dirs, files in os.walk(path):
                for file in files:
                    if "events.out.tfevents" in file:
                        print(f"📊 找到TensorBoard日志: {root}")
                        return root
    
    # 如果没找到event文件，返回tensorboard目录（如果存在）
    tb_dir = os.path.join(exp_dir, "tensorboard")
    if os.path.exists(tb_dir):
        return tb_dir
    
    return exp_dir

def launch_tensorboard_autodl(log_dir, port=6007):
    """在AutoDL环境中启动TensorBoard"""
    print(f"🚀 在AutoDL环境中启动TensorBoard...")
    print(f"📂 监控目录: {log_dir}")
    print(f"🌐 端口: {port}")
    print(f"🔗 访问地址: http://localhost:{port}")
    print()
    print("📋 AutoDL访问说明:")
    print("1. 在AutoDL控制台找到'AutoPanel'访问入口")
    print("2. 选择TensorBoard")
    print("3. 系统会自动跳转到TensorBoard界面")
    print()
    
    # 启动TensorBoard命令
    cmd = ["tensorboard", "--logdir", log_dir, "--port", str(port), "--host", "0.0.0.0"]
    
    print(f"执行命令: {' '.join(cmd)}")
    print("🔄 正在启动TensorBoard...")
    print("按 Ctrl+C 可停止TensorBoard服务")
    print("-" * 50)
    
    try:
        # 在前台运行TensorBoard
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 TensorBoard服务已停止")
    except Exception as e:
        print(f"❌ 启动TensorBoard时出错: {e}")
        print("💡 请确保已安装TensorBoard: pip install tensorboard")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="AutoDL环境专用TensorBoard启动器")
    parser.add_argument("--port", type=int, default=6007, 
                       help="TensorBoard服务端口 (默认6007，适配AutoDL)")
    parser.add_argument("--dir", type=str, default=None, 
                       help="实验目录路径，如不指定则自动查找最新的")
    parser.add_argument("--config", type=str, default="configs/train.yaml",
                       help="训练配置文件路径 (默认: configs/train.yaml)")
    parser.add_argument("--no-kill", action="store_true", 
                       help="不结束现有TensorBoard进程")
    args = parser.parse_args()
    
    print("🎯 AutoDL TensorBoard 启动器")
    print("=" * 40)
    
    # 结束现有TensorBoard进程（除非指定不结束）
    if not args.no_kill:
        kill_existing_tensorboard()
    
    # 查找实验目录
    if args.dir:
        exp_dir = os.path.expanduser(args.dir)
        print(f"📁 使用指定目录: {exp_dir}")
    else:
        print("🔍 自动查找最新实验目录...")
        exp_dir = find_latest_experiment(config_path=args.config)
    
    if not exp_dir or not os.path.exists(exp_dir):
        print("❌ 找不到有效的实验目录！")
        print("💡 请确保:")
        print("   1. 已开始训练，生成了实验目录")
        print("   2. 或使用 --dir 参数手动指定目录")
        print("   3. 检查 configs/train.yaml 中的 output_dir 设置")
        sys.exit(1)
    
    # 查找TensorBoard日志目录
    tb_dir = find_tensorboard_dir(exp_dir)
    
    if not os.path.exists(tb_dir):
        print(f"❌ TensorBoard日志目录不存在: {tb_dir}")
        print("💡 请确保训练已开始并启用了TensorBoard日志记录")
        sys.exit(1)
    
    # 启动TensorBoard
    success = launch_tensorboard_autodl(tb_dir, args.port)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 