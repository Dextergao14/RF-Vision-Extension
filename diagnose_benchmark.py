#!/usr/bin/env python3
"""
诊断benchmark评估问题：检查IoU分布和预测质量
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

from material_detection_model import MaterialDetectionModel
from evaluate_benchmark import BenchmarkDataset
from material_dataset import collate_fn
from train_material_detection import box_cxcywh_to_xyxy


def diagnose_benchmark_outputs(model, dataloader, device, score_threshold=0.05, iou_threshold=0.5):
    """诊断benchmark上的模型输出"""
    model.eval()
    
    all_pred_scores = []
    all_pred_labels = []
    all_ious = []
    all_max_ious = []  # 每个预测与所有GT的最大IoU
    num_images_with_detections = 0
    num_images_without_detections = 0
    total_gt_objects = 0
    total_pred_objects = 0
    
    # 统计每个GT的最大IoU（与所有预测的）
    gt_max_ious = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="诊断中")):
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
                pred_boxes_cxcywh = pred_boxes[b][valid_mask]
                pred_labels = labels_all[valid_mask]
                pred_scores = scores_all[valid_mask]
                
                # 转换为xyxy格式
                pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes_cxcywh)
                
                # 获取ground truth
                gt_boxes = targets[b]['boxes']  # [N_gt, 4] (xyxy格式，归一化)
                gt_labels = targets[b]['labels']  # [N_gt]
                
                total_gt_objects += len(gt_boxes)
                total_pred_objects += len(pred_boxes_xyxy)
                
                if len(pred_boxes_xyxy) > 0:
                    num_images_with_detections += 1
                    all_pred_scores.extend(pred_scores.cpu().numpy().tolist())
                    all_pred_labels.extend(pred_labels.cpu().numpy().tolist())
                    
                    # 计算每个预测与所有GT的最大IoU
                    if len(gt_boxes) > 0:
                        for pred_box in pred_boxes_xyxy:
                            max_iou = 0.0
                            for gt_box in gt_boxes:
                                # 计算IoU
                                inter_x1 = max(pred_box[0].item(), gt_box[0].item())
                                inter_y1 = max(pred_box[1].item(), gt_box[1].item())
                                inter_x2 = min(pred_box[2].item(), gt_box[2].item())
                                inter_y2 = min(pred_box[3].item(), gt_box[3].item())
                                
                                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                                    pred_area = (pred_box[2] - pred_box[0]).item() * (pred_box[3] - pred_box[1]).item()
                                    gt_area = (gt_box[2] - gt_box[0]).item() * (gt_box[3] - gt_box[1]).item()
                                    union_area = pred_area + gt_area - inter_area
                                    iou = inter_area / (union_area + 1e-6)
                                    max_iou = max(max_iou, iou)
                            
                            all_max_ious.append(max_iou)
                        
                        # 计算每个GT与所有预测的最大IoU
                        for gt_box in gt_boxes:
                            max_iou = 0.0
                            for pred_box in pred_boxes_xyxy:
                                inter_x1 = max(pred_box[0].item(), gt_box[0].item())
                                inter_y1 = max(pred_box[1].item(), gt_box[1].item())
                                inter_x2 = min(pred_box[2].item(), gt_box[2].item())
                                inter_y2 = min(pred_box[3].item(), gt_box[3].item())
                                
                                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                                    pred_area = (pred_box[2] - pred_box[0]).item() * (pred_box[3] - pred_box[1]).item()
                                    gt_area = (gt_box[2] - gt_box[0]).item() * (gt_box[3] - gt_box[1]).item()
                                    union_area = pred_area + gt_area - inter_area
                                    iou = inter_area / (union_area + 1e-6)
                                    max_iou = max(max_iou, iou)
                            
                            gt_max_ious.append(max_iou)
                else:
                    num_images_without_detections += 1
    
    # 统计信息
    print("\n" + "="*80)
    print("Benchmark模型输出诊断结果")
    print("="*80)
    print(f"总图像数: {num_images_with_detections + num_images_without_detections}")
    print(f"有检测的图像数: {num_images_with_detections}")
    print(f"无检测的图像数: {num_images_without_detections}")
    print(f"总GT目标数: {total_gt_objects}")
    print(f"总预测目标数: {total_pred_objects}")
    print(f"平均每张图像GT目标数: {total_gt_objects / (num_images_with_detections + num_images_without_detections):.2f}")
    print(f"平均每张图像预测目标数: {total_pred_objects / (num_images_with_detections + num_images_without_detections):.2f}")
    
    if len(all_pred_scores) > 0:
        print(f"\n预测分数统计 (score_threshold={score_threshold}):")
        print(f"  最小值: {np.min(all_pred_scores):.4f}")
        print(f"  最大值: {np.max(all_pred_scores):.4f}")
        print(f"  平均值: {np.mean(all_pred_scores):.4f}")
        print(f"  中位数: {np.median(all_pred_scores):.4f}")
        print(f"  标准差: {np.std(all_pred_scores):.4f}")
        
        print(f"\n预测类别分布:")
        unique_labels, counts = np.unique(all_pred_labels, return_counts=True)
        class_names = ['Concrete', 'Glass', 'Metal', 'Wood']
        for label, count in zip(unique_labels, counts):
            print(f"  {class_names[label]}: {count} ({count/len(all_pred_labels)*100:.1f}%)")
        
        if len(all_max_ious) > 0:
            print(f"\n预测框IoU统计 (每个预测与所有GT的最大IoU):")
            print(f"  最小值: {np.min(all_max_ious):.4f}")
            print(f"  最大值: {np.max(all_max_ious):.4f}")
            print(f"  平均值: {np.mean(all_max_ious):.4f}")
            print(f"  中位数: {np.median(all_max_ious):.4f}")
            print(f"  标准差: {np.std(all_max_ious):.4f}")
            print(f"  IoU >= {iou_threshold}: {np.sum(np.array(all_max_ious) >= iou_threshold)} ({np.sum(np.array(all_max_ious) >= iou_threshold)/len(all_max_ious)*100:.2f}%)")
            print(f"  IoU >= 0.3: {np.sum(np.array(all_max_ious) >= 0.3)} ({np.sum(np.array(all_max_ious) >= 0.3)/len(all_max_ious)*100:.2f}%)")
            print(f"  IoU >= 0.1: {np.sum(np.array(all_max_ious) >= 0.1)} ({np.sum(np.array(all_max_ious) >= 0.1)/len(all_max_ious)*100:.2f}%)")
        
        if len(gt_max_ious) > 0:
            print(f"\nGT框IoU统计 (每个GT与所有预测的最大IoU):")
            print(f"  最小值: {np.min(gt_max_ious):.4f}")
            print(f"  最大值: {np.max(gt_max_ious):.4f}")
            print(f"  平均值: {np.mean(gt_max_ious):.4f}")
            print(f"  中位数: {np.median(gt_max_ious):.4f}")
            print(f"  标准差: {np.std(gt_max_ious):.4f}")
            print(f"  IoU >= {iou_threshold}: {np.sum(np.array(gt_max_ious) >= iou_threshold)} ({np.sum(np.array(gt_max_ious) >= iou_threshold)/len(gt_max_ious)*100:.2f}%)")
            print(f"  IoU >= 0.3: {np.sum(np.array(gt_max_ious) >= 0.3)} ({np.sum(np.array(gt_max_ious) >= 0.3)/len(gt_max_ious)*100:.2f}%)")
            print(f"  IoU >= 0.1: {np.sum(np.array(gt_max_ious) >= 0.1)} ({np.sum(np.array(gt_max_ious) >= 0.1)/len(gt_max_ious)*100:.2f}%)")
    else:
        print("\n⚠️  没有任何预测！模型可能没有输出任何检测结果。")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='诊断benchmark评估问题')
    parser.add_argument('--checkpoint', type=str, 
                       default='./checkpoints_material_detection_new_data/checkpoint_best.pth',
                       help='模型检查点路径')
    parser.add_argument('--benchmark_root', type=str,
                       default='./Eval_benchmark',
                       help='Benchmark数据集根目录')
    parser.add_argument('--img_size', type=int, default=224, help='图像尺寸')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--score_threshold', type=float, default=0.05, help='分数阈值')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU阈值')
    
    args = parser.parse_args()
    
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
    
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state_dict, strict=False)
    if missing_keys:
        print(f"⚠️  缺少的键: {len(missing_keys)} 个")
    if unexpected_keys:
        print(f"⚠️  意外的键: {len(unexpected_keys)} 个")
    
    print(f"✅ 模型加载完成 (epoch {checkpoint.get('epoch', 'unknown')})")
    
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
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    print(f"📊 Benchmark数据集大小: {len(benchmark_dataset)}")
    
    # 诊断
    diagnose_benchmark_outputs(
        model, benchmark_loader, device,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold
    )


if __name__ == '__main__':
    main()
