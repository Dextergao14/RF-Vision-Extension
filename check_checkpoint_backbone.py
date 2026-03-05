#!/usr/bin/env python3
"""
检查checkpoint使用的backbone类型
"""

import torch
import sys

def check_checkpoint_backbone(checkpoint_path):
    """检查checkpoint使用的backbone"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print("="*60)
    print(f"检查Checkpoint: {checkpoint_path}")
    print("="*60)
    
    # 检查args
    if 'args' in checkpoint:
        args = checkpoint['args']
        print(f"\n训练参数:")
        print(f"  backbone: {args.get('backbone', 'N/A')}")
        print(f"  img_size: {args.get('img_size', 'N/A')}")
        print(f"  num_classes: {args.get('num_classes', 'N/A')}")
        print(f"  pretrained: {args.get('pretrained', 'N/A')}")
    
    # 检查模型权重键
    model_state = checkpoint.get('model_state_dict', {})
    backbone_keys = [k for k in model_state.keys() if 'backbone' in k]
    
    print(f"\nBackbone相关键数量: {len(backbone_keys)}")
    
    # 检查是否有timm结构
    timm_keys = [k for k in backbone_keys if 'backbone.blocks' in k or 'backbone.backbone' in k]
    custom_keys = [k for k in backbone_keys if 'patch_embed' in k or 'transformer' in k]
    
    if timm_keys:
        print(f"\n✅ 检测到timm ViT结构（{len(timm_keys)}个键）")
        print(f"   示例键:")
        for key in timm_keys[:5]:
            print(f"     - {key}")
        print(f"\n   评估时应该使用timm ViT（确保timm库已安装）")
    elif custom_keys:
        print(f"\n✅ 检测到自定义ViT结构（{len(custom_keys)}个键）")
        print(f"   示例键:")
        for key in custom_keys[:5]:
            print(f"     - {key}")
        print(f"\n   评估时应该使用自定义ViT（timm不可用时）")
    else:
        print(f"\n⚠️  无法确定backbone类型")
        print(f"   前10个backbone键:")
        for key in backbone_keys[:10]:
            print(f"     - {key}")

if __name__ == '__main__':
    checkpoint_path = './checkpoints_material_detection/checkpoint_best.pth'
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    
    check_checkpoint_backbone(checkpoint_path)
