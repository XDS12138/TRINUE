#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修复后的解码器是否正常工作
"""

import torch
from modules.decoder import MultiTaskDecoder

def test_decoder(decoder, batch_size=1, channels=48, debug=True):
    """
    测试解码器通道数是否匹配，并返回是否通过测试
    
    Args:
        decoder: MultiTaskDecoder实例
        batch_size: 批次大小
        channels: 基础通道数
        debug: 是否打印调试信息
    
    Returns:
        bool: 测试是否通过
    """
    try:
        device = next(decoder.parameters()).device
        # 创建测试数据
        fused_feat = torch.randn(batch_size, channels, 8, 8).to(device)
        skip_feats = [
            torch.randn(batch_size, channels, 64, 64).to(device),
            torch.randn(batch_size, channels, 32, 32).to(device),
            torch.randn(batch_size, channels, 16, 16).to(device),
            torch.randn(batch_size, channels, 8, 8).to(device)
        ]
        depth_feats = [
            torch.randn(batch_size, channels, 64, 64).to(device),
            torch.randn(batch_size, channels, 32, 32).to(device),
            torch.randn(batch_size, channels, 16, 16).to(device),
            torch.randn(batch_size, channels, 8, 8).to(device)
        ]
        raw = torch.randn(batch_size, 3, 128, 128).to(device)
        
        if debug:
            print(f"输入特征: {fused_feat.shape}")
            for i, feat in enumerate(skip_feats):
                print(f"Skip特征 {i}: {feat.shape}")
            for i, feat in enumerate(depth_feats):
                print(f"深度特征 {i}: {feat.shape}")
        
        # 执行前向传播
        res_d, res_c = decoder(fused_feat, skip_feats, depth_feats, raw)
        
        if debug:
            print(f"解码器测试成功！输出形状: {res_d.shape}")
        
        return True
    except Exception as e:
        if debug:
            print(f"解码器测试失败: {e}")
            import traceback
            traceback.print_exc()
        return False

if __name__ == "__main__":
    print("创建解码器...")
    model = MultiTaskDecoder(base_channels=48, levels=3)
    
    print("开始测试解码器...")
    success = test_decoder(model, debug=True)
    
    print(f"测试结果: {'通过' if success else '失败'}") 