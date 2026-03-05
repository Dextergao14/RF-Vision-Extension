#!/usr/bin/env python3
"""
评估MAML适应后的模型，只在query set上评估（排除support set）
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

from material_detection_model import MaterialDetectionModel
from evaluate_benchmark import BenchmarkDataset
from material_dataset import collate_fn
from train_material_detection import box_cxcywh_to_xyxy


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
    """
    评估检测结果（只有4个已知类，无unknown类）
    
    Args:
        pred_boxes: [N_pred, 4] (cxcywh格式，归一化)
        pred_labels: [N_pred] (只包含已知类，0-3)
        pred_scores: [N_pred]
        gt_boxes: [N_gt, 4] (xyxy格式，归一化)
        gt_labels: [N_gt] (只包含已知类，0-3)
        iou_threshold: IoU阈值
    
    Returns:
        tp, fp, fn counts per class, correct_classifications (正确分类的数量)
    """
    num_classes = 4  # 已知类数量
    
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)
    correct_classifications = 0  # 正确分类的数量（IoU匹配且类别正确）
    total_detections = 0  # 总检测数量（IoU匹配的）
    
    if len(pred_boxes) == 0:
        # 所有ground truth都是false negative
        for label in gt_labels:
            class_fn[label.item()] += 1
        return class_tp, class_fp, class_fn, correct_classifications, total_detections
    
    if len(gt_boxes) == 0:
        # 所有预测都是false positive
        for label in pred_labels:
            if 0 <= label.item() < num_classes:  # 只统计已知类
                class_fp[label.item()] += 1
        return class_tp, class_fp, class_fn, correct_classifications, total_detections
    
    # 转换pred_boxes到xyxy格式
    pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    
    # 按分数排序
    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes_xyxy = pred_boxes_xyxy[sorted_indices]
    pred_labels = pred_labels[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    
    # 匹配预测和ground truth
    matched_gt = set()
    
    for i, (pred_box, pred_label, pred_score) in enumerate(zip(pred_boxes_xyxy, pred_labels, pred_scores)):
        best_iou = 0.0
        best_gt_idx = -1
        
        # 找到最佳匹配的ground truth
        for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if j in matched_gt:
                continue
            
            # 计算IoU
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
        
        # 判断是TP还是FP
        if best_iou >= iou_threshold:
            total_detections += 1
            if pred_label == gt_labels[best_gt_idx]:
                # 正确匹配
                class_tp[pred_label.item()] += 1
                correct_classifications += 1
                matched_gt.add(best_gt_idx)
            else:
                # IoU匹配但类别错误（预测为错误的已知类）
                if 0 <= pred_label.item() < num_classes:  # 只统计已知类
                    class_fp[pred_label.item()] += 1
                # 对应的ground truth也算作未匹配
        else:
            # IoU不匹配，是FP
            if 0 <= pred_label.item() < num_classes:  # 只统计已知类
                class_fp[pred_label.item()] += 1
    
    # 未匹配的ground truth是false negative
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
        pbar = tqdm(query_loader, desc="评估Query Set")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)
            
            outputs = model(images)
            pred_logits = outputs['pred_logits']  # [B, num_queries, num_classes+1]
            pred_boxes = outputs['pred_boxes']  # [B, num_queries, 4] (cxcywh)
            
            for b in range(len(targets)):
                # 获取预测
                probs = F.softmax(pred_logits[b], dim=-1)  # [num_queries, num_classes+1]
                scores_all, labels_all = probs[:, :-1].max(dim=-1)  # 排除背景类
                
                # 过滤低分预测
                valid_mask = scores_all > score_threshold
                pred_boxes_b = pred_boxes[b][valid_mask]
                pred_labels_b = labels_all[valid_mask]
                pred_scores_b = scores_all[valid_mask]
                
                # 获取ground truth
                gt_boxes = targets[b]['boxes'].to(device)
                gt_labels = targets[b]['labels'].to(device)
                
                total_gt_objects += len(gt_labels)
                
                # 评估
                tp, fp, fn, correct_cls, total_det = evaluate_detections(
                    pred_boxes_b, pred_labels_b, pred_scores_b,
                    gt_boxes, gt_labels, iou_threshold
                )
                
                total_correct_classifications += correct_cls
                total_detections += total_det
                
                # 累计（只有4个已知类）
                for cls_id in range(4):  # 0-3是已知类
                    all_tp[cls_id] += tp.get(cls_id, 0)
                    all_fp[cls_id] += fp.get(cls_id, 0)
                    all_fn[cls_id] += fn.get(cls_id, 0)
    
    # 计算指标
    class_names = ['Concrete', 'Glass', 'Metal', 'Wood']
    results = {}
    
    print("\n" + "="*80)
    print("📊 Query Set 评估结果")
    print("="*80)
    print(f"{'类别':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
    print("-"*80)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for cls_id, class_name in enumerate(class_names):
        tp = all_tp[cls_id]
        fp = all_fp[cls_id]
        fn = all_fn[cls_id]
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        results[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
        
        print(f"{class_name:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {tp:<8} {fp:<8} {fn:<8}")
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # 总体指标
    overall_precision = total_tp / (total_tp + total_fp + 1e-6)
    overall_recall = total_tp / (total_tp + total_fn + 1e-6)
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall + 1e-6)
    
    print("-"*80)
    print(f"{'总体':<15} {overall_precision:<12.4f} {overall_recall:<12.4f} {overall_f1:<12.4f} {total_tp:<8} {total_fp:<8} {total_fn:<8}")
    
    # 分类准确率
    classification_accuracy = total_correct_classifications / (total_detections + 1e-6)
    detection_accuracy = total_tp / (total_gt_objects + 1e-6)
    overall_accuracy = total_tp / (total_gt_objects + 1e-6)
    
    print("\n📈 详细指标:")
    print(f"  分类准确率 (Classification Accuracy): {classification_accuracy:.4f} ({total_correct_classifications}/{total_detections})")
    print(f"  检测准确率 (Detection Accuracy): {detection_accuracy:.4f} ({total_tp}/{total_gt_objects})")
    print(f"  总体准确率 (Overall Accuracy): {overall_accuracy:.4f} ({total_tp}/{total_gt_objects})")
    
    results['overall'] = {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'classification_accuracy': classification_accuracy,
        'detection_accuracy': detection_accuracy,
        'overall_accuracy': overall_accuracy,
        'total_correct_classifications': total_correct_classifications,
        'total_detections': total_detections,
        'total_gt_objects': total_gt_objects
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='在query set上评估MAML适应后的模型')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='MAML适应后的模型检查点')
    parser.add_argument('--benchmark_root', type=str,
                       default='./Eval_benchmark',
                       help='Benchmark数据集根目录')
    parser.add_argument('--support_size', type=int, required=True,
                       help='Support set大小（用于确定query set）')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子（必须与训练时一致）')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--score_threshold', type=float, default=0.05)
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--output', type=str, default='maml_query_set_results.json')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 加载模型
    print(f"\n📂 加载检查点: {args.checkpoint}")
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
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"✅ 模型加载完成")
    
    # 加载benchmark数据集
    benchmark_dataset = BenchmarkDataset(
        benchmark_root=args.benchmark_root,
        img_size=args.img_size,
        num_views=1,
        use_multi_view=False
    )
    
    # 使用相同的随机种子分割support和query set
    torch.manual_seed(args.random_seed)
    target_size = len(benchmark_dataset)
    support_size = args.support_size
    indices = torch.randperm(target_size).tolist()
    
    support_indices = indices[:support_size]
    query_indices = indices[support_size:]
    
    query_dataset = Subset(benchmark_dataset, query_indices)
    
    print(f"\n📊 数据集信息:")
    print(f"  Benchmark总图像数: {target_size}")
    print(f"  Support set: {support_size} 张图像（用于适应，不评估）")
    print(f"  Query set: {len(query_dataset)} 张图像（用于评估）")
    
    # 评估query set
    results = evaluate_model_on_query_set(
        model, query_dataset, device,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold
    )
    
    # 保存结果
    results['query_set_info'] = {
        'support_size': support_size,
        'query_size': len(query_dataset),
        'total_size': target_size,
        'random_seed': args.random_seed
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 评估结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
