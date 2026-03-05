#!/usr/bin/env python3
"""
使用20%配置微调后的模型进行评估和可视化
Score threshold: 0.5
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
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import random

from material_detection_model import MaterialDetectionModel
from evaluate_benchmark import BenchmarkDataset
from material_dataset import collate_fn
from train_material_detection import box_cxcywh_to_xyxy, HungarianMatcher, SetCriterion

try:
    from torchvision.ops import nms as torch_nms
    NMS_AVAILABLE = True
except ImportError:
    NMS_AVAILABLE = False


def apply_nms_per_class(pred_boxes, pred_labels, pred_scores, iou_threshold=0.5):
    """按类别分别应用NMS，减少重复检测导致的FP"""
    if not NMS_AVAILABLE or len(pred_boxes) == 0:
        return pred_boxes, pred_labels, pred_scores

    pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    num_classes = 4

    keep_indices = []
    for cls_id in range(num_classes):
        cls_mask = pred_labels == cls_id
        if cls_mask.sum() == 0:
            continue
        cls_indices = torch.where(cls_mask)[0]
        cls_boxes = pred_boxes_xyxy[cls_indices]
        cls_scores = pred_scores[cls_indices]

        if len(cls_boxes) == 0:
            continue
        nms_keep = torch_nms(cls_boxes.float(), cls_scores.float(), iou_threshold)
        keep_indices.append(cls_indices[nms_keep])

    if len(keep_indices) == 0:
        return pred_boxes[[]], pred_labels[[]], pred_scores[[]]

    keep_indices = torch.cat(keep_indices)
    keep_indices = torch.sort(keep_indices)[0]
    return pred_boxes[keep_indices], pred_labels[keep_indices], pred_scores[keep_indices]


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


def finetune_on_support_set(model, support_dataset, device, args):
    """在support set上进行微调"""
    support_loader = DataLoader(
        support_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
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
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.finetune_lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )
    
    model.train()
    
    for epoch in range(args.finetune_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for images, targets in support_loader:
            images = images.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
    
    return model


def visualize_predictions(model, query_dataset, device, args, num_samples=10):
    """可视化预测结果"""
    model.eval()
    
    class_names = ['Concrete', 'Glass', 'Metal', 'Wood']
    colors = {
        0: (1, 0, 0),      # Concrete - Red
        1: (0, 1, 0),      # Glass - Green
        2: (0, 0, 1),      # Metal - Blue
        3: (1, 1, 0),      # Wood - Yellow
    }
    
    # 随机选择样本
    indices = list(range(len(query_dataset)))
    random.seed(42)
    selected_indices = random.sample(indices, min(num_samples, len(indices)))
    
    # 创建临时dataset用于加载选中的样本
    selected_dataset = Subset(query_dataset, selected_indices)
    
    query_loader = DataLoader(
        selected_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(query_loader, desc="可视化")):
            images = images.to(device)
            
            outputs = model(images)
            pred_logits = outputs['pred_logits'][0]
            pred_boxes = outputs['pred_boxes'][0]
            
            # 处理预测（类别特定threshold）
            probs = F.softmax(pred_logits, dim=-1)
            scores_all, labels_all = probs[:, :-1].max(dim=-1)
            valid_mask = torch.zeros_like(scores_all, dtype=torch.bool)
            for cls_id in range(4):
                thr = args.class_thresholds[cls_id]
                valid_mask |= (labels_all == cls_id) & (scores_all > thr)
            pred_boxes_b = pred_boxes[valid_mask]
            pred_labels_b = labels_all[valid_mask]
            pred_scores_b = scores_all[valid_mask]
            if args.use_nms and len(pred_boxes_b) > 0:
                pred_boxes_b, pred_labels_b, pred_scores_b = apply_nms_per_class(
                    pred_boxes_b, pred_labels_b, pred_scores_b,
                    iou_threshold=args.nms_iou_threshold
                )
            
            # 获取ground truth
            gt_boxes = targets[0]['boxes'].to(device)
            gt_labels = targets[0]['labels'].to(device)
            
            # 加载图像
            # 从dataset获取原始图像路径
            dataset_idx = selected_indices[batch_idx]
            try:
                # BenchmarkDataset应该有image_files属性
                if hasattr(query_dataset, 'indices'):
                    # Subset的情况
                    actual_idx = query_dataset.indices[dataset_idx]
                    image_path = query_dataset.dataset.image_files[actual_idx]
                else:
                    # 直接dataset的情况
                    image_path = query_dataset.image_files[dataset_idx]
                
                if Path(image_path).exists():
                    image = cv2.imread(str(image_path))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    raise FileNotFoundError
            except Exception as e:
                # 如果无法获取路径，从tensor恢复图像
                img_tensor = images[0].cpu()
                # 反归一化（假设使用ImageNet normalization）
                mean = torch.tensor([0.485, 0.456, 0.406])
                std = torch.tensor([0.229, 0.224, 0.225])
                img_tensor = img_tensor * std[:, None, None] + mean[:, None, None]
                image = img_tensor.permute(1, 2, 0).numpy()
                image = np.clip(image, 0, 1)
                image = (image * 255).astype(np.uint8)
            
            h, w = image.shape[:2]
            
            # 创建图像
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            ax.imshow(image)
            ax.axis('off')
            
            # 绘制ground truth
            for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                x1, y1, x2, y2 = gt_box.cpu().numpy()
                x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
                
                rect = Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='green', facecolor='none',
                    linestyle='--'
                )
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, f'GT: {class_names[gt_label.item()]}',
                       color='green', fontsize=10, weight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # 绘制预测
            for i, (pred_box, pred_label, pred_score) in enumerate(zip(pred_boxes_b, pred_labels_b, pred_scores_b)):
                pred_box_xyxy = box_cxcywh_to_xyxy(pred_box.unsqueeze(0))[0]
                x1, y1, x2, y2 = pred_box_xyxy.cpu().numpy()
                x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
                
                color = colors[pred_label.item()]
                rect = Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, f'{class_names[pred_label.item()]}: {pred_score.item():.2f}',
                       color=color, fontsize=10, weight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # 添加标题
            title = f'Sample {batch_idx + 1}\n'
            title += f'GT: {len(gt_boxes)} objects, Pred: {len(pred_boxes_b)} detections'
            ax.set_title(title, fontsize=12, weight='bold')
            
            # 保存
            output_path = output_dir / f'visualization_{batch_idx + 1:03d}.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"\n✅ 可视化结果已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='20%配置评估和可视化')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='预训练模型检查点')
    parser.add_argument('--benchmark_root', type=str,
                       default='./Eval_benchmark',
                       help='Benchmark数据集根目录')
    parser.add_argument('--support_percentage', type=int, default=20,
                       help='Support set比例')
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
    parser.add_argument('--score_threshold', type=float, default=0.4, help='默认阈值(无class_thresholds时使用)')
    parser.add_argument('--class_thresholds', type=str, default='0.25,0.45,0.40,0.40',
                       help='类别特定阈值: Concrete,Glass,Metal,Wood，逗号分隔，如0.25,0.25,0.4,0.4')
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--no_nms', action='store_true', help='禁用NMS后处理')
    parser.add_argument('--nms_iou_threshold', type=float, default=0.6, help='NMS的IoU阈值')
    parser.add_argument('--num_visualizations', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='./visualizations_20pct_0.5')
    parser.add_argument('--output_json', type=str, default='evaluation_20pct_0.5.json')
    
    args = parser.parse_args()
    args.use_nms = not args.no_nms
    # 解析类别特定阈值
    args.class_thresholds = [float(x) for x in args.class_thresholds.split(',')]
    if len(args.class_thresholds) != 4:
        raise ValueError('class_thresholds 必须包含4个值: Concrete,Glass,Metal,Wood')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 加载预训练模型
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
    
    print(f"✅ 预训练模型加载完成")
    
    # 加载benchmark数据集
    benchmark_dataset = BenchmarkDataset(
        benchmark_root=args.benchmark_root,
        img_size=args.img_size,
        num_views=1,
        use_multi_view=False
    )
    
    # 分割support和query set
    target_size = len(benchmark_dataset)
    support_size = int(target_size * args.support_percentage / 100)
    
    torch.manual_seed(42)
    indices = torch.randperm(target_size).tolist()
    support_indices = indices[:support_size]
    query_indices = indices[support_size:]
    
    support_subset = Subset(benchmark_dataset, support_indices)
    query_subset = Subset(benchmark_dataset, query_indices)
    
    print(f"\n📊 数据集分割:")
    print(f"  Benchmark总图像数: {target_size}")
    print(f"  Support set: {support_size} 张图像 ({args.support_percentage}%)")
    print(f"  Query set: {len(query_subset)} 张图像 ({100-args.support_percentage}%)")
    
    # 在support set上微调
    print(f"\n🔧 开始微调...")
    finetuned_model = finetune_on_support_set(
        model,
        support_subset,
        device,
        args
    )
    print(f"✅ 微调完成")
    
    # 评估
    print(f"\n📊 开始评估...")
    query_loader = DataLoader(
        query_subset,
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
    
    finetuned_model.eval()
    with torch.no_grad():
        for images, targets in tqdm(query_loader, desc="评估"):
            images = images.to(device)
            
            outputs = finetuned_model(images)
            pred_logits = outputs['pred_logits']
            pred_boxes = outputs['pred_boxes']
            
            for b in range(len(targets)):
                probs = F.softmax(pred_logits[b], dim=-1)
                scores_all, labels_all = probs[:, :-1].max(dim=-1)
                
                # 类别特定score threshold
                valid_mask = torch.zeros_like(scores_all, dtype=torch.bool)
                for cls_id in range(4):
                    thr = args.class_thresholds[cls_id]
                    valid_mask |= (labels_all == cls_id) & (scores_all > thr)
                pred_boxes_b = pred_boxes[b][valid_mask]
                pred_labels_b = labels_all[valid_mask]
                pred_scores_b = scores_all[valid_mask]

                # NMS后处理
                if args.use_nms and len(pred_boxes_b) > 0:
                    pred_boxes_b, pred_labels_b, pred_scores_b = apply_nms_per_class(
                        pred_boxes_b, pred_labels_b, pred_scores_b,
                        iou_threshold=args.nms_iou_threshold
                    )
                
                gt_boxes = targets[b]['boxes'].to(device)
                gt_labels = targets[b]['labels'].to(device)
                
                total_gt_objects += len(gt_labels)
                
                tp, fp, fn, correct_cls, total_det = evaluate_detections(
                    pred_boxes_b, pred_labels_b, pred_scores_b,
                    gt_boxes, gt_labels, args.iou_threshold
                )
                
                total_correct_classifications += correct_cls
                total_detections += total_det
                
                for cls_id in range(4):
                    all_tp[cls_id] += tp.get(cls_id, 0)
                    all_fp[cls_id] += fp.get(cls_id, 0)
                    all_fn[cls_id] += fn.get(cls_id, 0)
    
    # 计算指标
    class_names = ['Concrete', 'Glass', 'Metal', 'Wood']
    total_tp = sum(all_tp.values())
    total_fp = sum(all_fp.values())
    total_fn = sum(all_fn.values())
    
    overall_precision = total_tp / (total_tp + total_fp + 1e-6)
    overall_recall = total_tp / (total_tp + total_fn + 1e-6)
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall + 1e-6)
    
    print(f"\n{'='*80}")
    print("📊 评估结果")
    print(f"{'='*80}")
    print(f"{'类别':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
    print("-"*80)
    
    results = {}
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
    
    print("-"*80)
    print(f"{'总体':<15} {overall_precision:<12.4f} {overall_recall:<12.4f} {overall_f1:<12.4f} {total_tp:<8} {total_fp:<8} {total_fn:<8}")
    
    results['overall'] = {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'total_gt_objects': total_gt_objects
    }
    
    # 保存结果
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 评估结果已保存到: {args.output_json}")
    
    # 可视化
    print(f"\n🎨 开始可视化...")
    visualize_predictions(finetuned_model, query_subset, device, args, args.num_visualizations)


if __name__ == '__main__':
    main()
