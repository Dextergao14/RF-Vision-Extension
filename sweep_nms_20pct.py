#!/usr/bin/env python3
"""
20% 微调后，只做 NMS 参数扫描：一次微调，对多个 NMS 阈值分别评估，找出最佳 NMS。
用法:
  python sweep_nms_20pct.py --checkpoint /path/to/checkpoint_best.pth [--nms_values "0.35,0.4,0.45,0.5,0.55,0.6"]
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import argparse
from tqdm import tqdm
import json
from collections import defaultdict

from material_detection_model import MaterialDetectionModel
from evaluate_benchmark import BenchmarkDataset
from material_dataset import collate_fn
from train_material_detection import box_cxcywh_to_xyxy, HungarianMatcher, SetCriterion
from evaluate_and_visualize_20pct import (
    apply_nms_per_class,
    evaluate_detections,
    finetune_on_support_set,
)

CLASS_NAMES = ['Concrete', 'Glass', 'Metal', 'Wood']


def run_eval_with_nms(model, query_loader, device, class_thresholds, iou_threshold, nms_iou_threshold):
    """对当前模型用指定 NMS 做一次评估，返回与 evaluation_20pct_*.json 相同结构的结果。"""
    all_tp = defaultdict(int)
    all_fp = defaultdict(int)
    all_fn = defaultdict(int)
    total_gt_objects = 0

    model.eval()
    with torch.no_grad():
        for images, targets in query_loader:
            images = images.to(device)
            outputs = model(images)
            pred_logits = outputs['pred_logits']
            pred_boxes = outputs['pred_boxes']

            for b in range(len(targets)):
                probs = F.softmax(pred_logits[b], dim=-1)
                scores_all, labels_all = probs[:, :-1].max(dim=-1)
                valid_mask = torch.zeros_like(scores_all, dtype=torch.bool)
                for cls_id in range(4):
                    thr = class_thresholds[cls_id]
                    valid_mask |= (labels_all == cls_id) & (scores_all > thr)
                pred_boxes_b = pred_boxes[b][valid_mask]
                pred_labels_b = labels_all[valid_mask]
                pred_scores_b = scores_all[valid_mask]

                if len(pred_boxes_b) > 0:
                    pred_boxes_b, pred_labels_b, pred_scores_b = apply_nms_per_class(
                        pred_boxes_b, pred_labels_b, pred_scores_b,
                        iou_threshold=nms_iou_threshold
                    )

                gt_boxes = targets[b]['boxes'].to(device)
                gt_labels = targets[b]['labels'].to(device)
                total_gt_objects += len(gt_labels)

                tp, fp, fn, _, _ = evaluate_detections(
                    pred_boxes_b, pred_labels_b, pred_scores_b,
                    gt_boxes, gt_labels, iou_threshold
                )
                for cls_id in range(4):
                    all_tp[cls_id] += tp.get(cls_id, 0)
                    all_fp[cls_id] += fp.get(cls_id, 0)
                    all_fn[cls_id] += fn.get(cls_id, 0)

    total_tp = sum(all_tp.values())
    total_fp = sum(all_fp.values())
    total_fn = sum(all_fn.values())
    overall_precision = total_tp / (total_tp + total_fp + 1e-6)
    overall_recall = total_tp / (total_tp + total_fn + 1e-6)
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall + 1e-6)

    results = {}
    for cls_id, name in enumerate(CLASS_NAMES):
        tp, fp, fn = all_tp[cls_id], all_fp[cls_id], all_fn[cls_id]
        p = tp / (tp + fp + 1e-6)
        r = tp / (tp + fn + 1e-6)
        f1 = 2 * p * r / (p + r + 1e-6)
        results[name] = {'precision': p, 'recall': r, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}
    results['overall'] = {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'total_gt_objects': total_gt_objects,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description='20% 微调后 NMS 参数扫描')
    parser.add_argument('--checkpoint', type=str, required=True, help='预训练模型检查点')
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
    parser.add_argument('--class_thresholds', type=str, default='0.25,0.25,0.4,0.4')
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--nms_values', type=str, default='0.35,0.4,0.45,0.5,0.55,0.6',
                        help='要扫描的 NMS IoU 阈值，逗号分隔')
    parser.add_argument('--output_json', type=str, default='sweep_nms_20pct_results.json')
    args = parser.parse_args()

    class_thresholds = [float(x) for x in args.class_thresholds.split(',')]
    if len(class_thresholds) != 4:
        raise ValueError('class_thresholds 必须包含 4 个值')
    nms_values = [float(x.strip()) for x in args.nms_values.split(',')]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    print(f"📋 将扫描 NMS 阈值: {nms_values}")

    # 加载模型
    print(f"\n📂 加载预训练模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
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
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    print("✅ 预训练模型加载完成")

    # 数据集与 20% 划分
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
    print(f"\n📊 Support: {support_size}, Query: {len(query_subset)}")

    # 一次微调
    print("\n🔧 微调中...")
    finetuned_model = finetune_on_support_set(model, support_subset, device, args)
    print("✅ 微调完成")

    # 对每个 NMS 做评估
    all_results = {}
    for nms in nms_values:
        print(f"\n📊 评估 NMS = {nms} ...")
        res = run_eval_with_nms(
            finetuned_model,
            query_loader,
            device,
            class_thresholds,
            args.iou_threshold,
            nms
        )
        all_results[str(nms)] = res

    # 汇总与最佳
    print(f"\n{'='*80}")
    print("📊 NMS 扫描结果（按 overall F1 选最佳）")
    print(f"{'='*80}")
    print(f"{'NMS':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
    print("-" * 80)

    best_nms = None
    best_f1 = -1.0
    for nms_str in sorted(all_results.keys(), key=float):
        r = all_results[nms_str]['overall']
        f1 = r['f1']
        if f1 > best_f1:
            best_f1 = f1
            best_nms = float(nms_str)
        print(f"{nms_str:<8} {r['precision']:<12.4f} {r['recall']:<12.4f} {f1:<12.4f} "
              f"{r['tp']:<8} {r['fp']:<8} {r['fn']:<8}")

    print("-" * 80)
    print(f"✅ 最佳 NMS = {best_nms} (overall F1 = {best_f1:.4f})")

    out = {
        'nms_values': nms_values,
        'best_nms': best_nms,
        'best_overall_f1': best_f1,
        'results_by_nms': all_results,
    }
    with open(args.output_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n💾 详细结果已保存到: {args.output_json}")


if __name__ == '__main__':
    main()
