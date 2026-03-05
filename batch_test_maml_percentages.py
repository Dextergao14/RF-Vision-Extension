#!/usr/bin/env python3
"""
批量测试不同比例的support set（10%, 20%, 30%, 40%, 50%）
在对应的query set上评估
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from collections import defaultdict
import subprocess
import sys
import os

from material_detection_model import MaterialDetectionModel
from evaluate_benchmark import BenchmarkDataset
from material_dataset import collate_fn
from train_material_detection import box_cxcywh_to_xyxy, HungarianMatcher, SetCriterion
from train_maml_domain_adaptation import MAML


def calculate_iou(boxes1, boxes2):
    """计算IoU"""
    inter_x1 = torch.max(boxes1[:, 0:1], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1:2], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2:3], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3:4], boxes2[:, 3])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    box1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    box2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = box1_area.unsqueeze(1) + box2_area - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    return iou


def evaluate_detections(pred_boxes, pred_labels, pred_scores, 
                       gt_boxes, gt_labels, iou_threshold=0.5):
    """评估检测结果"""
    num_classes = 4
    
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)
    correct_classifications = 0
    total_detections = 0
    
    if len(pred_boxes) == 0:
        for label in gt_labels:
            class_fn[label.item()] += 1
        return class_tp, class_fp, class_fn, correct_classifications, total_detections
    
    if len(gt_boxes) == 0:
        for label in pred_labels:
            if 0 <= label.item() < num_classes:
                class_fp[label.item()] += 1
        return class_tp, class_fp, class_fn, correct_classifications, total_detections
    
    pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes_xyxy = pred_boxes_xyxy[sorted_indices]
    pred_labels = pred_labels[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    
    matched_gt = set()
    
    for i, (pred_box, pred_label, pred_score) in enumerate(zip(pred_boxes_xyxy, pred_labels, pred_scores)):
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if j in matched_gt:
                continue
            
            inter_x1 = max(pred_box[0], gt_box[0])
            inter_y1 = max(pred_box[1], gt_box[1])
            inter_x2 = min(pred_box[2], gt_box[2])
            inter_y2 = min(pred_box[3], gt_box[3])
            
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                union_area = pred_area + gt_area - inter_area
                iou = inter_area / (union_area + 1e-6)
            else:
                iou = 0.0
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold:
            total_detections += 1
            if pred_label == gt_labels[best_gt_idx]:
                class_tp[pred_label.item()] += 1
                correct_classifications += 1
                matched_gt.add(best_gt_idx)
            else:
                if 0 <= pred_label.item() < num_classes:
                    class_fp[pred_label.item()] += 1
        else:
            if 0 <= pred_label.item() < num_classes:
                class_fp[pred_label.item()] += 1
    
    for j, gt_label in enumerate(gt_labels):
        if j not in matched_gt:
            class_fn[gt_label.item()] += 1
    
    return class_tp, class_fp, class_fn, correct_classifications, total_detections


def evaluate_model_on_query_set(model, query_dataset, device, score_threshold=0.05, iou_threshold=0.5):
    """在query set上评估模型"""
    model.eval()
    
    query_loader = DataLoader(
        query_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    all_tp = defaultdict(int)
    all_fp = defaultdict(int)
    all_fn = defaultdict(int)
    total_correct_classifications = 0
    total_detections = 0
    total_gt_objects = 0
    
    with torch.no_grad():
        for images, targets in tqdm(query_loader, desc="评估Query Set"):
            images = images.to(device)
            
            outputs = model(images)
            pred_logits = outputs['pred_logits']
            pred_boxes = outputs['pred_boxes']
            
            for b in range(len(targets)):
                probs = F.softmax(pred_logits[b], dim=-1)
                scores_all, labels_all = probs[:, :-1].max(dim=-1)
                
                valid_mask = scores_all > score_threshold
                pred_boxes_b = pred_boxes[b][valid_mask]
                pred_labels_b = labels_all[valid_mask]
                pred_scores_b = scores_all[valid_mask]
                
                gt_boxes = targets[b]['boxes'].to(device)
                gt_labels = targets[b]['labels'].to(device)
                
                total_gt_objects += len(gt_labels)
                
                tp, fp, fn, correct_cls, total_det = evaluate_detections(
                    pred_boxes_b, pred_labels_b, pred_scores_b,
                    gt_boxes, gt_labels, iou_threshold
                )
                
                total_correct_classifications += correct_cls
                total_detections += total_det
                
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
    
    classification_accuracy = total_correct_classifications / (total_detections + 1e-6)
    detection_accuracy = total_tp / (total_gt_objects + 1e-6)
    
    return {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'classification_accuracy': classification_accuracy,
        'detection_accuracy': detection_accuracy,
        'total_correct_classifications': total_correct_classifications,
        'total_detections': total_detections,
        'total_gt_objects': total_gt_objects,
        'per_class': {
            cls_id: {
                'tp': all_tp[cls_id],
                'fp': all_fp[cls_id],
                'fn': all_fn[cls_id],
                'precision': all_tp[cls_id] / (all_tp[cls_id] + all_fp[cls_id] + 1e-6),
                'recall': all_tp[cls_id] / (all_tp[cls_id] + all_fn[cls_id] + 1e-6)
            }
            for cls_id in range(4)
        }
    }


def run_maml_adaptation(meta_checkpoint, benchmark_root, support_percentage, 
                       device, args):
    """
    运行MAML快速适应
    
    Returns:
        adapted_model: 适应后的模型
        support_size: support set大小
        query_size: query set大小
    """
    print(f"\n{'='*80}")
    print(f"🔬 MAML快速适应 - {support_percentage}% Support Set")
    print(f"{'='*80}")
    
    # 加载meta-trained模型
    print(f"\n📂 加载meta-trained模型: {meta_checkpoint}")
    checkpoint = torch.load(meta_checkpoint, map_location=device, weights_only=False)
    
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
    print(f"✅ Meta-trained模型加载完成")
    
    # 加载benchmark数据集
    benchmark_dataset = BenchmarkDataset(
        benchmark_root=benchmark_root,
        img_size=args.img_size,
        num_views=1,
        use_multi_view=False
    )
    
    # 分割support和query set
    target_size = len(benchmark_dataset)
    support_size = int(target_size * support_percentage / 100)
    
    torch.manual_seed(42)
    indices = torch.randperm(target_size).tolist()
    support_indices = indices[:support_size]
    query_indices = indices[support_size:]
    
    support_subset = Subset(benchmark_dataset, support_indices)
    query_subset = Subset(benchmark_dataset, query_indices)
    
    print(f"\n📊 数据集分割:")
    print(f"  Benchmark总图像数: {target_size}")
    print(f"  Support set: {support_size} 张图像 ({support_percentage}%)")
    print(f"  Query set: {len(query_subset)} 张图像 ({100-support_percentage}%)")
    
    # 准备数据加载器
    support_loader = DataLoader(
        support_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=False
    )
    
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
    
    # MAML快速适应
    maml = MAML(
        model=model,
        inner_lr=args.adapt_lr,
        inner_steps=args.adapt_steps,
        first_order=args.first_order,
        device=device
    )
    
    print(f"\n🚀 开始快速适应...")
    print(f"  适应学习率: {args.adapt_lr}")
    print(f"  适应步数: {args.adapt_steps}")
    
    adapted_params, adapt_loss = maml.fast_adapt(support_loader, criterion)
    
    # 应用适应后的参数
    adapted_param_idx = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = adapted_params[adapted_param_idx].data.clone()
            adapted_param_idx += 1
    
    print(f"✅ 快速适应完成，适应损失: {adapt_loss:.4f}")
    
    return model, support_size, len(query_subset), query_subset


def main():
    parser = argparse.ArgumentParser(description='批量测试不同比例的support set')
    parser.add_argument('--meta_checkpoint', type=str, required=True,
                       help='Meta-trained模型检查点')
    parser.add_argument('--benchmark_root', type=str,
                       default='./Eval_benchmark',
                       help='Benchmark数据集根目录')
    parser.add_argument('--percentages', type=int, nargs='+', 
                       default=[10, 20, 30, 40, 50],
                       help='要测试的support set比例（百分比）')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--adapt_lr', type=float, default=1e-3,
                       help='快速适应学习率')
    parser.add_argument('--adapt_steps', type=int, default=5,
                       help='快速适应步数')
    parser.add_argument('--first_order', action='store_true',
                       help='使用一阶近似（更快但可能不准确）')
    parser.add_argument('--cost_class', type=float, default=1.0)
    parser.add_argument('--cost_bbox', type=float, default=5.0)
    parser.add_argument('--cost_giou', type=float, default=2.0)
    parser.add_argument('--weight_ce', type=float, default=1.0)
    parser.add_argument('--weight_bbox', type=float, default=5.0)
    parser.add_argument('--weight_giou', type=float, default=2.0)
    parser.add_argument('--eos_coef', type=float, default=0.1)
    parser.add_argument('--score_threshold', type=float, default=0.05)
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--output', type=str, default='maml_percentage_results.json')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    all_results = {}
    
    for percentage in args.percentages:
        print(f"\n{'#'*80}")
        print(f"# 测试 {percentage}% Support Set")
        print(f"{'#'*80}")
        
        try:
            # 运行MAML适应
            adapted_model, support_size, query_size, query_subset = run_maml_adaptation(
                args.meta_checkpoint,
                args.benchmark_root,
                percentage,
                device,
                args
            )
            
            # 在query set上评估
            print(f"\n📊 在Query Set上评估...")
            results = evaluate_model_on_query_set(
                adapted_model,
                query_subset,
                device,
                score_threshold=args.score_threshold,
                iou_threshold=args.iou_threshold
            )
            
            results['support_percentage'] = percentage
            results['support_size'] = support_size
            results['query_size'] = query_size
            
            all_results[f'{percentage}pct'] = results
            
            print(f"\n✅ {percentage}% 配置完成:")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall: {results['recall']:.4f}")
            print(f"  F1-Score: {results['f1']:.4f}")
            
        except Exception as e:
            print(f"❌ {percentage}% 配置失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[f'{percentage}pct'] = {'error': str(e)}
    
    # 保存所有结果
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 打印汇总
    print(f"\n{'='*80}")
    print("📊 汇总结果")
    print(f"{'='*80}")
    print(f"{'比例':<10} {'Support':<10} {'Query':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*80)
    
    for percentage in args.percentages:
        key = f'{percentage}pct'
        if key in all_results and 'error' not in all_results[key]:
            r = all_results[key]
            print(f"{percentage}%{'':<6} {r['support_size']:<10} {r['query_size']:<10} "
                  f"{r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1']:<12.4f}")
        else:
            print(f"{percentage}%{'':<6} {'ERROR':<10}")
    
    print(f"\n💾 详细结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
