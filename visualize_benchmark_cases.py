#!/usr/bin/env python3
"""
可视化benchmark评估中的case，展示模型预测和GT的差距
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import json
import warnings
warnings.filterwarnings('ignore')

# 延迟导入，避免timm导入问题
import os
import sys

# 设置环境变量，让torchvision跳过CUDA检查（如果可能）
os.environ['TORCHVISION_DISABLE_CUDA_CHECK'] = '1'

try:
    # 先尝试导入material_detection_model，它内部会处理timm导入失败
    from material_detection_model import MaterialDetectionModel
    MaterialDetectionModel = MaterialDetectionModel
except Exception as e:
    print(f"⚠️  导入模型时出错: {e}")
    print("⚠️  尝试使用自定义ViT实现...")
    # 如果导入失败，尝试直接使用evaluate_benchmark中的导入方式
    try:
        import warnings
        warnings.filterwarnings('ignore')
        from material_detection_model import MaterialDetectionModel
    except:
        MaterialDetectionModel = None

from evaluate_benchmark import BenchmarkDataset
from material_dataset import collate_fn
from train_material_detection import box_cxcywh_to_xyxy


def draw_boxes_on_image(image, boxes, labels, scores, class_names, color='red', linewidth=2):
    """在图像上绘制边界框"""
    h, w = image.shape[:2]
    
    for box, label, score in zip(boxes, labels, scores):
        # box是归一化坐标，转换为像素坐标
        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int(box[2] * w)
        y2 = int(box[3] * h)
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), 
                     color=(255, 0, 0) if color == 'red' else (0, 255, 0),
                     thickness=linewidth)
        
        # 添加标签和分数
        label_text = f"{class_names[label]}: {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(image, (x1, y1 - text_height - baseline - 5), 
                     (x1 + text_width, y1), 
                     color=(255, 0, 0) if color == 'red' else (0, 255, 0),
                     thickness=-1)
        cv2.putText(image, label_text, (x1, y1 - baseline - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image


def calculate_iou(box1, box2):
    """计算两个框的IoU"""
    # box格式: [x1, y1, x2, y2] (归一化)
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / (union_area + 1e-6)


def visualize_cases(model, dataloader, device, output_dir, num_cases=10, 
                   score_threshold=0.05, iou_threshold=0.5):
    """可视化评估cases"""
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = ['Concrete', 'Glass', 'Metal', 'Wood']
    
    case_count = 0
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if case_count >= num_cases:
                break
            
            images = images.to(device)
            outputs = model(images)
            
            pred_logits = outputs['pred_logits']  # [B, num_queries, num_classes+1]
            pred_boxes = outputs['pred_boxes']  # [B, num_queries, 4] (cxcywh)
            
            for b in range(len(targets)):
                if case_count >= num_cases:
                    break
                
                # 获取预测
                probs = F.softmax(pred_logits[b], dim=-1)  # [num_queries, num_classes+1]
                scores_all, labels_all = probs[:, :-1].max(dim=-1)  # 排除背景类
                
                # 过滤低分预测
                valid_mask = scores_all > score_threshold
                pred_boxes_cxcywh = pred_boxes[b][valid_mask]
                pred_labels = labels_all[valid_mask]
                pred_scores = scores_all[valid_mask]
                
                # 转换为xyxy格式
                pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes_cxcywh)
                
                # 获取ground truth
                gt_boxes = targets[b]['boxes']  # [N_gt, 4] (xyxy格式，归一化)
                gt_labels = targets[b]['labels']  # [N_gt]
                
                # 加载原始图像
                dataset = dataloader.dataset
                image_idx = batch_idx * dataloader.batch_size + b
                if image_idx < len(dataset):
                    img_path = dataset.image_paths[image_idx]
                    original_img = cv2.imread(str(img_path))
                    if original_img is None:
                        print(f"⚠️  无法加载图像: {img_path}")
                        continue
                    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                    h, w = original_img.shape[:2]
                else:
                    continue
                
                # 如果图像被resize了，需要恢复到原始尺寸
                # 但这里我们直接使用原始图像，因为GT boxes是归一化的
                
                # 创建可视化图像
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # 1. 原始图像 + GT
                img_with_gt = original_img.copy()
                if len(gt_boxes) > 0:
                    gt_boxes_np = gt_boxes.cpu().numpy()
                    gt_labels_np = gt_labels.cpu().numpy()
                    for gt_box, gt_label in zip(gt_boxes_np, gt_labels_np):
                        x1 = int(gt_box[0] * w)
                        y1 = int(gt_box[1] * h)
                        x2 = int(gt_box[2] * w)
                        y2 = int(gt_box[3] * h)
                        cv2.rectangle(img_with_gt, (x1, y1), (x2, y2), 
                                     color=(0, 255, 0), thickness=2)
                        label_text = f"GT: {class_names[gt_label]}"
                        cv2.putText(img_with_gt, label_text, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                axes[0].imshow(img_with_gt)
                axes[0].set_title(f'Ground Truth ({len(gt_boxes)} objects)', fontsize=12)
                axes[0].axis('off')
                
                # 2. 原始图像 + 预测
                img_with_pred = original_img.copy()
                if len(pred_boxes_xyxy) > 0:
                    pred_boxes_np = pred_boxes_xyxy.cpu().numpy()
                    pred_labels_np = pred_labels.cpu().numpy()
                    pred_scores_np = pred_scores.cpu().numpy()
                    for pred_box, pred_label, pred_score in zip(pred_boxes_np, pred_labels_np, pred_scores_np):
                        x1 = int(pred_box[0] * w)
                        y1 = int(pred_box[1] * h)
                        x2 = int(pred_box[2] * w)
                        y2 = int(pred_box[3] * h)
                        cv2.rectangle(img_with_pred, (x1, y1), (x2, y2), 
                                     color=(255, 0, 0), thickness=2)
                        label_text = f"{class_names[pred_label]}: {pred_score:.2f}"
                        cv2.putText(img_with_pred, label_text, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                axes[1].imshow(img_with_pred)
                axes[1].set_title(f'Predictions ({len(pred_boxes_xyxy)} detections)', fontsize=12)
                axes[1].axis('off')
                
                # 3. 叠加显示（GT绿色，预测红色）
                img_overlay = original_img.copy()
                
                # 绘制GT（绿色）
                if len(gt_boxes) > 0:
                    gt_boxes_np = gt_boxes.cpu().numpy()
                    gt_labels_np = gt_labels.cpu().numpy()
                    for gt_box, gt_label in zip(gt_boxes_np, gt_labels_np):
                        x1 = int(gt_box[0] * w)
                        y1 = int(gt_box[1] * h)
                        x2 = int(gt_box[2] * w)
                        y2 = int(gt_box[3] * h)
                        cv2.rectangle(img_overlay, (x1, y1), (x2, y2), 
                                     color=(0, 255, 0), thickness=2)
                        label_text = f"GT: {class_names[gt_label]}"
                        cv2.putText(img_overlay, label_text, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 绘制预测（红色）
                if len(pred_boxes_xyxy) > 0:
                    pred_boxes_np = pred_boxes_xyxy.cpu().numpy()
                    pred_labels_np = pred_labels.cpu().numpy()
                    pred_scores_np = pred_scores.cpu().numpy()
                    for pred_box, pred_label, pred_score in zip(pred_boxes_np, pred_labels_np, pred_scores_np):
                        x1 = int(pred_box[0] * w)
                        y1 = int(pred_box[1] * h)
                        x2 = int(pred_box[2] * w)
                        y2 = int(pred_box[3] * h)
                        cv2.rectangle(img_overlay, (x1, y1), (x2, y2), 
                                     color=(255, 0, 0), thickness=2)
                        label_text = f"{class_names[pred_label]}: {pred_score:.2f}"
                        cv2.putText(img_overlay, label_text, (x1, y1 + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                axes[2].imshow(img_overlay)
                axes[2].set_title('Overlay (Green=GT, Red=Pred)', fontsize=12)
                axes[2].axis('off')
                
                # 计算匹配信息
                matched_gt = set()
                match_info = []
                
                for i, (pred_box, pred_label, pred_score) in enumerate(zip(
                    pred_boxes_xyxy.cpu().numpy(), 
                    pred_labels.cpu().numpy(), 
                    pred_scores.cpu().numpy()
                )):
                    best_iou = 0.0
                    best_gt_idx = -1
                    
                    for j, (gt_box, gt_label) in enumerate(zip(
                        gt_boxes.cpu().numpy(), 
                        gt_labels.cpu().numpy()
                    )):
                        if j in matched_gt:
                            continue
                        iou = calculate_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = j
                    
                    if best_iou >= iou_threshold:
                        gt_label = gt_labels[best_gt_idx].item()
                        is_correct = (pred_label == gt_label)
                        match_info.append({
                            'pred_idx': i,
                            'pred_class': class_names[pred_label],
                            'pred_score': float(pred_score),
                            'gt_idx': best_gt_idx,
                            'gt_class': class_names[gt_label],
                            'iou': float(best_iou),
                            'matched': True,
                            'correct': is_correct
                        })
                        matched_gt.add(best_gt_idx)
                    else:
                        match_info.append({
                            'pred_idx': i,
                            'pred_class': class_names[pred_label],
                            'pred_score': float(pred_score),
                            'gt_idx': None,
                            'gt_class': None,
                            'iou': float(best_iou),
                            'matched': False,
                            'correct': False
                        })
                
                # 添加未匹配的GT
                for j, gt_label in enumerate(gt_labels.cpu().numpy()):
                    if j not in matched_gt:
                        match_info.append({
                            'pred_idx': None,
                            'pred_class': None,
                            'pred_score': None,
                            'gt_idx': j,
                            'gt_class': class_names[gt_label],
                            'iou': None,
                            'matched': False,
                            'correct': False
                        })
                
                # 添加统计信息到标题
                tp_count = sum(1 for m in match_info if m.get('matched') and m.get('correct'))
                fp_count = sum(1 for m in match_info if m.get('pred_idx') is not None and not m.get('matched'))
                fn_count = sum(1 for m in match_info if m.get('gt_idx') is not None and not m.get('matched'))
                
                fig.suptitle(
                    f'Case {case_count + 1}: TP={tp_count}, FP={fp_count}, FN={fn_count} | '
                    f'GT={len(gt_boxes)}, Pred={len(pred_boxes_xyxy)}',
                    fontsize=14, fontweight='bold'
                )
                
                # 保存图像
                output_path = output_dir / f'case_{case_count + 1:02d}.png'
                plt.tight_layout()
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # 保存匹配信息到JSON
                info_path = output_dir / f'case_{case_count + 1:02d}_info.json'
                with open(info_path, 'w') as f:
                    json.dump({
                        'image_path': str(img_path),
                        'gt_count': len(gt_boxes),
                        'pred_count': len(pred_boxes_xyxy),
                        'tp': tp_count,
                        'fp': fp_count,
                        'fn': fn_count,
                        'matches': match_info
                    }, f, indent=2)
                
                print(f"✅ Case {case_count + 1}: GT={len(gt_boxes)}, Pred={len(pred_boxes_xyxy)}, "
                      f"TP={tp_count}, FP={fp_count}, FN={fn_count}")
                
                case_count += 1
    
    print(f"\n✅ 已可视化 {case_count} 个cases，保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='可视化benchmark评估cases')
    parser.add_argument('--checkpoint', type=str, 
                       default='./checkpoints_material_detection_new_data/checkpoint_best.pth',
                       help='模型检查点路径')
    parser.add_argument('--benchmark_root', type=str,
                       default='./Eval_benchmark',
                       help='Benchmark数据集根目录')
    parser.add_argument('--img_size', type=int, default=224, help='图像尺寸')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--score_threshold', type=float, default=0.05, help='分数阈值')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU阈值')
    parser.add_argument('--num_cases', type=int, default=10, help='可视化的case数量')
    parser.add_argument('--output_dir', type=str, default='benchmark_visualization',
                       help='输出目录')
    
    args = parser.parse_args()
    
    if MaterialDetectionModel is None:
        print("❌ 无法导入MaterialDetectionModel，请检查环境")
        return
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 加载checkpoint
    print(f"📂 加载检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # 获取模型配置
    if 'args' in checkpoint:
        model_args = checkpoint['args']
        backbone_name = model_args.get('backbone', 'vit_base_patch16_224')
        num_queries = model_args.get('num_queries', 100)
        num_decoder_layers = model_args.get('num_decoder_layers', 6)
        use_multi_view = model_args.get('use_multi_view', False)
        num_views = model_args.get('num_views', 3)
    else:
        backbone_name = 'vit_base_patch16_224'
        num_queries = 100
        num_decoder_layers = 6
        use_multi_view = False
        num_views = 3
    
    # 检查checkpoint版本
    checkpoint_keys = set(checkpoint['model_state_dict'].keys())
    class_embed_key = None
    for key in checkpoint['model_state_dict'].keys():
        if 'class_embed.weight' in key:
            class_embed_key = key
            break
    
    checkpoint_class_embed_dim = None
    if class_embed_key:
        checkpoint_class_embed_dim = checkpoint['model_state_dict'][class_embed_key].shape[0]
    
    # 创建模型
    pretrained_backbone = False
    if 'args' in checkpoint and checkpoint['args'].get('pretrained', False):
        pretrained_backbone = True
    
    model = MaterialDetectionModel(
        backbone_name=backbone_name,
        img_size=args.img_size,
        num_classes=4,
        num_queries=num_queries,
        num_decoder_layers=num_decoder_layers,
        use_multi_view=use_multi_view,
        num_views=num_views,
        pretrained_backbone=pretrained_backbone
    ).to(device)
    
    # 加载权重
    checkpoint_state_dict = checkpoint['model_state_dict']
    model_state_dict = model.state_dict()
    
    if checkpoint_class_embed_dim == 6:
        # 适配6维checkpoint到5维模型
        adapted_state_dict = {}
        for key, value in checkpoint_state_dict.items():
            if 'class_embed.weight' in key:
                adapted_state_dict[key] = value[[0, 1, 2, 3, 5]]
            elif 'class_embed.bias' in key:
                adapted_state_dict[key] = value[[0, 1, 2, 3, 5]]
            else:
                if key in model_state_dict and model_state_dict[key].shape == value.shape:
                    adapted_state_dict[key] = value
        model.load_state_dict(adapted_state_dict, strict=False)
        print("✅ 模型加载完成（已移除unknown类）")
    elif checkpoint_class_embed_dim == 5:
        model.load_state_dict(checkpoint_state_dict, strict=False)
        print("✅ 模型加载完成")
    else:
        model.load_state_dict(checkpoint_state_dict, strict=False)
        print("✅ 模型加载完成（使用strict=False）")
    
    # Benchmark数据集
    benchmark_dataset = BenchmarkDataset(
        benchmark_root=args.benchmark_root,
        img_size=args.img_size,
        num_views=1,
        use_multi_view=False
    )
    
    benchmark_loader = DataLoader(
        benchmark_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # 使用0避免多进程问题，确保能访问dataset属性
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    print(f"📊 Benchmark数据集大小: {len(benchmark_dataset)}")
    print(f"📊 将可视化 {args.num_cases} 个cases")
    print(f"📁 输出目录: {args.output_dir}")
    
    print(f"📊 Benchmark数据集大小: {len(benchmark_dataset)}")
    print(f"📊 将可视化 {args.num_cases} 个cases")
    
    # 可视化
    visualize_cases(
        model, benchmark_loader, device,
        output_dir=args.output_dir,
        num_cases=args.num_cases,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold
    )


if __name__ == '__main__':
    main()
