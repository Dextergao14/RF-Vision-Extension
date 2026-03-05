#!/usr/bin/env python3
"""
修复MAML快速适应：使用更保守的参数重新适应
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import argparse

from material_detection_model import MaterialDetectionModel
from evaluate_benchmark import BenchmarkDataset
from material_dataset import collate_fn
from train_material_detection import HungarianMatcher, SetCriterion
from train_maml_domain_adaptation import MAML


def conservative_fast_adapt(model, target_dataset, device, args):
    """
    保守的快速适应：使用更少的步数和更小的学习率
    """
    print("\n" + "="*80)
    print("保守的快速适应（修复版本）")
    print("="*80)
    
    # 分割目标域数据
    target_size = len(target_dataset)
    support_size = min(args.adapt_shots, int(target_size * 0.1))
    indices = torch.randperm(target_size).tolist()
    support_subset = Subset(target_dataset, indices[:support_size])
    
    support_loader = DataLoader(
        support_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=False
    )
    
    print(f"目标域support集: {len(support_subset)} 张图像")
    print(f"使用保守参数:")
    print(f"  适应学习率: {args.adapt_lr} (更小)")
    print(f"  适应步数: {args.adapt_steps} (更少)")
    
    # 损失函数
    matcher = HungarianMatcher(
        cost_class=args.cost_class,
        cost_bbox=args.cost_bbox,
        cost_giou=args.cost_giou
    )
    weight_dict = {
        'loss_ce': args.weight_ce,
        'loss_bbox': args.weight_bbox,
        'loss_giou': args.weight_giou
    }
    criterion = SetCriterion(
        num_classes=4,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef
    ).to(device)
    
    # 创建MAML实例（使用更保守的参数）
    maml = MAML(
        model=model,
        inner_lr=args.adapt_lr,  # 使用更小的学习率
        inner_steps=args.adapt_steps,  # 使用更少的步数
        first_order=args.first_order,
        device=device
    )
    
    # 快速适应
    print(f"\n开始保守快速适应...")
    adapted_params, adapt_loss = maml.fast_adapt(support_loader, criterion)
    
    # 应用适应后的参数
    adapted_param_idx = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = adapted_params[adapted_param_idx].data.clone()
            adapted_param_idx += 1
    
    print(f"✅ 保守快速适应完成，适应损失: {adapt_loss:.4f}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='修复MAML快速适应')
    parser.add_argument('--meta_trained_checkpoint', type=str,
                       default='./checkpoints_maml_domain_adaptation_full/maml_meta_trained.pth',
                       help='元训练后的模型检查点')
    parser.add_argument('--target_benchmark_root', type=str,
                       default='./Eval_benchmark',
                       help='目标域benchmark数据集根目录')
    parser.add_argument('--adapt_shots', type=int, default=30,
                       help='适应样本数（减少）')
    parser.add_argument('--adapt_lr', type=float, default=1e-4,
                       help='适应学习率（更小，默认1e-4）')
    parser.add_argument('--adapt_steps', type=int, default=5,
                       help='适应步数（更少，默认5）')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--first_order', action='store_true')
    parser.add_argument('--output_dir', type=str,
                       default='./checkpoints_maml_domain_adaptation_full')
    
    # 损失函数参数
    parser.add_argument('--cost_class', type=float, default=1.0)
    parser.add_argument('--cost_bbox', type=float, default=5.0)
    parser.add_argument('--cost_giou', type=float, default=2.0)
    parser.add_argument('--weight_ce', type=float, default=1.0)
    parser.add_argument('--weight_bbox', type=float, default=5.0)
    parser.add_argument('--weight_giou', type=float, default=2.0)
    parser.add_argument('--eos_coef', type=float, default=0.1)
    parser.add_argument('--img_size', type=int, default=224)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 加载元训练后的模型
    print(f"\n📂 加载元训练后的模型: {args.meta_trained_checkpoint}")
    checkpoint = torch.load(args.meta_trained_checkpoint, map_location=device, weights_only=False)
    
    if 'args' in checkpoint:
        model_args = checkpoint['args']
        backbone_name = model_args.get('backbone', 'vit_base_patch16_224')
        num_queries = model_args.get('num_queries', 100)
        num_decoder_layers = model_args.get('num_decoder_layers', 6)
    else:
        backbone_name = 'vit_base_patch16_224'
        num_queries = 100
        num_decoder_layers = 6
    
    model = MaterialDetectionModel(
        backbone_name=backbone_name,
        img_size=args.img_size,
        num_classes=4,
        num_queries=num_queries,
        num_decoder_layers=num_decoder_layers,
        use_multi_view=False,
        num_views=1,
        pretrained_backbone=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"✅ 模型加载完成")
    
    # 加载目标域数据
    target_dataset = BenchmarkDataset(
        benchmark_root=args.target_benchmark_root,
        img_size=args.img_size,
        num_views=1,
        use_multi_view=False
    )
    
    print(f"目标域数据: {len(target_dataset)} 张图像")
    
    # 保守的快速适应
    adapted_model = conservative_fast_adapt(model, target_dataset, device, args)
    
    # 保存修复后的模型
    output_dir = Path(args.output_dir)
    save_path = output_dir / 'maml_adapted_fixed.pth'
    torch.save({
        'model_state_dict': adapted_model.state_dict(),
        'adapt_shots': args.adapt_shots,
        'adapt_steps': args.adapt_steps,
        'adapt_lr': args.adapt_lr,
        'args': vars(args)
    }, save_path)
    print(f"\n💾 修复后的适应模型已保存到: {save_path}")
    print(f"\n✅ 修复完成！")


if __name__ == '__main__':
    main()
