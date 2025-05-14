#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
launch_tensorboard.py - 自动启动TensorBoard服务
-----------------------------------------------
用途：查找最新的实验日志，并自动启动TensorBoard，提供实时监控训练进度

使用方法：
python scripts/launch_tensorboard.py [--port PORT] [--dir EXPERIMENT_DIR]
"""

import os
import sys
import glob
import argparse
import time
import webbrowser
import subprocess
from datetime import datetime

def find_latest_experiment(base_dir="experiments/train"):
    """查找最近的实验目录"""
    experiments = []
    
    for exp_dir in glob.glob(os.path.join(base_dir, "*")):
        if os.path.isdir(exp_dir):
            try:
                # 尝试从目录名提取时间戳
                dirname = os.path.basename(exp_dir)
                if "_" in dirname:
                    # 目录名可能包含时间戳，如 underwater_enhance_run_20250513_114603
                    timestamp_str = dirname.split("_")[-2] + "_" + dirname.split("_")[-1]
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    experiments.append((exp_dir, timestamp))
                else:
                    # 使用目录修改时间作为备选
                    mtime = os.path.getmtime(exp_dir)
                    timestamp = datetime.fromtimestamp(mtime)
                    experiments.append((exp_dir, timestamp))
            except (ValueError, IndexError):
                # 如果解析失败，使用目录修改时间
                mtime = os.path.getmtime(exp_dir)
                timestamp = datetime.fromtimestamp(mtime)
                experiments.append((exp_dir, timestamp))
    
    if not experiments:
        return None
    
    # 按时间戳排序，返回最新的
    experiments.sort(key=lambda x: x[1], reverse=True)
    return experiments[0][0]

def launch_tensorboard(log_dir, port=6006):
    """启动TensorBoard服务器，并自动打开浏览器"""
    # 确保目录存在
    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    if not os.path.exists(tensorboard_dir):
        tensorboard_dir = log_dir  # 如果没有tensorboard子目录，直接使用实验目录
    
    # 打印信息
    print(f"启动TensorBoard，监控目录: {tensorboard_dir}")
    print(f"TensorBoard将在 http://localhost:{port} 上可用")
    
    # 尝试打开浏览器
    try:
        # 延迟2秒，让TensorBoard有时间启动
        time.sleep(2)
        webbrowser.open(f"http://localhost:{port}")
        print("已自动打开浏览器。如果浏览器没有打开，请手动访问上面的链接。")
    except Exception as e:
        print(f"无法自动打开浏览器: {str(e)}")
        print(f"请手动访问 http://localhost:{port}")
    
    # 运行TensorBoard
    cmd = ["tensorboard", "--logdir", tensorboard_dir, "--port", str(port)]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nTensorBoard服务已停止")
    except Exception as e:
        print(f"启动TensorBoard时出错: {str(e)}")
        print("请确保已安装TensorBoard: pip install tensorboard")

def main():
    parser = argparse.ArgumentParser(description="启动TensorBoard监控训练进度")
    parser.add_argument("--port", type=int, default=6006, help="TensorBoard服务端口")
    parser.add_argument("--dir", type=str, default=None, help="实验目录路径，如不指定则自动查找最新的")
    args = parser.parse_args()
    
    # 查找实验目录
    if args.dir:
        exp_dir = args.dir
    else:
        exp_dir = find_latest_experiment()
    
    if not exp_dir or not os.path.exists(exp_dir):
        print("找不到有效的实验目录，请确保已开始训练或手动指定目录")
        sys.exit(1)
    
    print(f"使用实验目录: {exp_dir}")
    
    # 启动TensorBoard
    launch_tensorboard(exp_dir, args.port)

if __name__ == "__main__":
    main() 