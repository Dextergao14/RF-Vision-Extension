#!/usr/bin/env python3
"""
在固定 NMS=0.1 的基础上，只扫描 Glass（类别1）的分数阈值，压低 Glass FP。
其他类别阈值固定：Concrete=0.25, Metal=0.4, Wood=0.4。
用法:
  python sweep_glass_threshold_20pct.py --checkpoint /path/to/checkpoint_best.pth [--glass_values "0.25,0.3,0.35,0.4,0.45,0.5"]
"""

import torch
from torch.utils.data import DataLoader, Subset
import argparse
import json
from collections import defaultdict

from material_detection_model import MaterialDetectionModel
from evaluate_benchmark import BenchmarkDataset
from material_dataset import collate_fn
from evaluate_and_visualize_20pct import (
    apply_nms_per_class,
    evaluate_detections,
    finetune_on_support_set,
)
from sweep_nms_20pct import run_eval_with_nms, CLASS_NAMES

# 固定 NMS（与之前找到的最佳一致）
DEFAULT_NMS = 0.1


def main():
    parser = argparse.ArgumentParser(description='20% 微调后 Glass 分数阈值扫描（NMS 固定 0.1）')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--benchmark_root', type=str,
                        default='./Eval_benchmark')
    parser.add_argument('--support_percentage', type=int, default=20)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--finetune_epochs', type=int, default=3)
    parser.add_argument('--finetune_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_step_size', type=int, default=5)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=0.1)
    parser.add_argument('--cost_class', type=float, default=1.0)
    parser.add_argument('--cost_bbox', type=float, default=5.0)
    parser.add_argument('--cost_giou', type=float, default=2.0)
    parser.add_argument('--weight_ce', type=float, default=1.0)
    parser.add_argument('--weight_bbox', type=float, default=5.0)
    parser.add_argument('--weight_giou', type=float, default=2.0)
    parser.add_argument('--eos_coef', type=float, default=0.1)
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--nms_iou_threshold', type=float, default=DEFAULT_NMS,
                        help=f'NMS 阈值，默认 {DEFAULT_NMS}')
    parser.add_argument('--glass_values', type=str, default='0.25,0.3,0.35,0.4,0.45,0.5',
                        help='Glass 分数阈值扫描列表，逗号分隔')
    parser.add_argument('--concrete_threshold', type=float, default=0.25)
    parser.add_argument('--metal_threshold', type=float, default=0.4)
    parser.add_argument('--wood_threshold', type=float, default=0.4)
    parser.add_argument('--output_json', type=str, default='sweep_glass_threshold_20pct_results.json')
    args = parser.parse_args()

    glass_values = [float(x.strip()) for x in args.glass_values.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 设备: {device}")
    print(f"📋 NMS 固定: {args.nms_iou_threshold}")
    print(f"📋 Glass 阈值扫描: {glass_values}")
    print(f"📋 其他类别: Concrete={args.concrete_threshold}, Metal={args.metal_threshold}, Wood={args.wood_threshold}")

    # 加载模型（与 sweep_nms_20pct 一致）
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'args' in checkpoint:
        ma = checkpoint['args']
        backbone_name = ma.get('backbone', 'vit_base_patch16_224')
        num_queries = ma.get('num_queries', 100)
        num_decoder_layers = ma.get('num_decoder_layers', 6)
    else:
        backbone_name, num_queries, num_decoder_layers = 'vit_base_patch16_224', 100, 6

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
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    benchmark_dataset = BenchmarkDataset(
        benchmark_root=args.benchmark_root,
        img_size=args.img_size,
        num_views=1,
        use_multi_view=False
    )
    target_size = len(benchmark_dataset)
    support_size = int(target_size * args.support_percentage / 100)
    torch.manual_seed(42)
    indices = torch.randperm(target_size).tolist()
    support_subset = Subset(benchmark_dataset, indices[:support_size])
    query_subset = Subset(benchmark_dataset, indices[support_size:])
    query_loader = DataLoader(
        query_subset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False
    )

    print("\n🔧 微调中...")
    finetuned_model = finetune_on_support_set(model, support_subset, device, args)
    print("✅ 微调完成")

    all_results = {}
    for g in glass_values:
        class_thresholds = [
            args.concrete_threshold,  # 0
            g,                         # 1 Glass
            args.metal_threshold,     # 2
            args.wood_threshold,       # 3
        ]
        print(f"\n📊 Glass 阈值 = {g} ...")
        res = run_eval_with_nms(
            finetuned_model,
            query_loader,
            device,
            class_thresholds,
            args.iou_threshold,
            args.nms_iou_threshold,
        )
        all_results[str(g)] = res

    print(f"\n{'='*90}")
    print("📊 Glass 阈值扫描结果（NMS=%.2f）" % args.nms_iou_threshold)
    print(f"{'='*90}")
    print(f"{'Glass_thr':<10} {'Overall_F1':<12} {'Overall_P':<10} {'Overall_R':<10} {'Glass_P':<10} {'Glass_R':<10} {'Glass_F1':<10} {'Glass_FP':<8}")
    print("-" * 90)
    best_f1 = -1.0
    best_glass_thr = None
    for g_str in sorted(all_results.keys(), key=float):
        r = all_results[g_str]
        o = r['overall']
        gl = r['Glass']
        if o['f1'] > best_f1:
            best_f1 = o['f1']
            best_glass_thr = float(g_str)
        print(f"{g_str:<10} {o['f1']:<12.4f} {o['precision']:<10.4f} {o['recall']:<10.4f} "
              f"{gl['precision']:<10.4f} {gl['recall']:<10.4f} {gl['f1']:<10.4f} {gl['fp']:<8}")
    print("-" * 90)
    print(f"✅ 最佳 Glass 阈值 = {best_glass_thr} (overall F1 = {best_f1:.4f})")

    out = {
        'nms_iou_threshold': args.nms_iou_threshold,
        'glass_threshold_values': glass_values,
        'best_glass_threshold': best_glass_thr,
        'best_overall_f1': best_f1,
        'fixed_thresholds': {
            'Concrete': args.concrete_threshold,
            'Metal': args.metal_threshold,
            'Wood': args.wood_threshold,
        },
        'results_by_glass_threshold': all_results,
    }
    with open(args.output_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n💾 结果已保存: {args.output_json}")


if __name__ == '__main__':
    main()
