#!/usr/bin/env python3
"""
评估材料检测模型在Eval Benchmark上的性能
专门用于最终模型性能评估
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from collections import defaultdict
from datetime import datetime
import random

from material_detection_model import MaterialDetectionModel
from material_dataset import MaterialDetectionDataset, collate_fn
from train_material_detection import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou


class BenchmarkDataset(MaterialDetectionDataset):
    """适配benchmark数据结构的数据集类（benchmark目录结构不同）"""
    def __init__(self, benchmark_root, img_size=224, num_views=1, use_multi_view=False, transform=None):
        # 调用父类初始化，但传入benchmark_root作为data_root，split设为空字符串
        # 然后重写路径
        super().__init__(
            data_root=benchmark_root,
            split='',  # 空字符串，因为images和labels直接在根目录
            img_size=img_size,
            num_views=num_views,
            use_multi_view=use_multi_view,
            transform=transform
        )
        # 重写路径：benchmark的结构是 benchmark_root/images 和 benchmark_root/labels
        self.image_dir = Path(benchmark_root) / 'images'
        self.label_dir = Path(benchmark_root) / 'labels'
        # 重新收集图像路径
        self.image_paths = sorted(list(self.image_dir.glob('*.jpg')))
        # 重新分组（如果需要）
        if use_multi_view:
            self.scene_groups = self._group_by_scene()
        print(f"✅ 加载Benchmark数据集: {len(self.image_paths)} 张图像")
        if use_multi_view and self.scene_groups:
            print(f"   场景组数: {len(self.scene_groups)}")


def calculate_iou(boxes1, boxes2):
    """计算IoU"""
    # boxes: [N, 4] (x1, y1, x2, y2)
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


def evaluate_benchmark(model, dataloader, device, score_threshold=0.05, iou_threshold=0.5):
    """评估模型在benchmark上的性能"""
    model.eval()
    
    all_tp = defaultdict(int)
    all_fp = defaultdict(int)
    all_fn = defaultdict(int)
    total_correct_classifications = 0
    total_detections = 0
    total_gt_objects = 0
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="评估Benchmark"):
            # 移动到设备
            if images.dim() == 5:
                images = images.to(device)
            else:
                images = images.to(device)
            
            # 前向传播
            outputs = model(images)
            
            pred_logits = outputs['pred_logits']  # [B, num_queries, num_classes+1] (4类+background)
            pred_boxes = outputs['pred_boxes']  # [B, num_queries, 4] (cxcywh)
            
            # 处理每个样本
            for b in range(len(targets)):
                # 获取预测
                probs = F.softmax(pred_logits[b], dim=-1)  # [num_queries, num_classes+1] (4类+background)
                
                # 选择概率最高的类别（排除background）
                # 类别索引: 0-3是已知类，4是background
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
    print("📊 Eval Benchmark 评估结果")
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
        
        # 累计总体指标（所有4个已知类）
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
    overall_accuracy = total_correct_classifications / (total_gt_objects + 1e-6)
    
    print("="*80)
    print("\n📈 详细指标:")
    print(f"  分类准确率 (Classification Accuracy): {classification_accuracy:.4f} ({total_correct_classifications}/{total_detections})")
    print(f"  检测准确率 (Detection Accuracy): {detection_accuracy:.4f} ({total_tp}/{total_gt_objects})")
    print(f"  总体准确率 (Overall Accuracy): {overall_accuracy:.4f} ({total_correct_classifications}/{total_gt_objects})")
    print("="*80)
    
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
        'total_correct_classifications': int(total_correct_classifications),
        'total_detections': int(total_detections),
        'total_gt_objects': int(total_gt_objects)
    }
    
    # 添加benchmark统计信息
    results['benchmark_info'] = {
        'total_images': len(dataloader.dataset),
        'total_gt_objects': int(total_gt_objects),
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'score_threshold': score_threshold,
        'iou_threshold': iou_threshold
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='评估材料检测模型在Eval Benchmark上的性能')
    parser.add_argument('--checkpoint', type=str, 
                       default='./checkpoints_material_detection/checkpoint_best.pth',
                       help='模型检查点路径')
    parser.add_argument('--benchmark_root', type=str,
                       default='./Eval_benchmark',
                       help='Benchmark数据集根目录')
    parser.add_argument('--img_size', type=int, default=224, help='图像尺寸')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--score_threshold', type=float, default=0.05, help='分数阈值（默认0.05，与test set评估一致）')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU阈值')
    parser.add_argument('--use_multi_view', action='store_true', help='使用多视图')
    parser.add_argument('--num_views', type=int, default=3, help='视图数量')
    parser.add_argument('--output', type=str, default='benchmark_evaluation_results.json', 
                       help='结果保存路径')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 加载检查点
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
        use_multi_view = args.use_multi_view
        num_views = args.num_views
    
    # 检查checkpoint使用的backbone类型和模型版本
    checkpoint_keys = set(checkpoint['model_state_dict'].keys())
    has_timm_in_checkpoint = any('backbone.backbone' in k for k in checkpoint_keys)
    
    # 检查模型版本（通过class_embed的维度）
    class_embed_key = None
    for key in checkpoint['model_state_dict'].keys():
        if 'class_embed.weight' in key:
            class_embed_key = key
            break
    
    checkpoint_num_classes = None
    if class_embed_key:
        checkpoint_class_embed_dim = checkpoint['model_state_dict'][class_embed_key].shape[0]
        # 维度-1是background类，所以已知类数量 = 维度 - 1
        checkpoint_num_classes = checkpoint_class_embed_dim - 1
        print(f"📋 Checkpoint模型版本检测:")
        print(f"   分类器输出维度: {checkpoint_class_embed_dim}")
        if checkpoint_class_embed_dim == 5:
            print(f"   ✅ 旧版本模型（4类+背景，不支持unknown）")
        elif checkpoint_class_embed_dim == 6:
            print(f"   ✅ 新版本模型（4类+unknown+背景）")
        else:
            print(f"   ⚠️  未知版本（{checkpoint_class_embed_dim}类输出）")
    
    # 创建模型
    # 确保使用与训练时相同的backbone配置
    pretrained_backbone = False  # 评估时不需要预训练权重，使用checkpoint中的权重
    if 'args' in checkpoint and checkpoint['args'].get('pretrained', False):
        # 如果训练时使用了pretrained，评估时也应该设置（虽然会用checkpoint权重覆盖）
        pretrained_backbone = True
    
    # 如果checkpoint使用timm ViT，确保评估时也使用timm
    if has_timm_in_checkpoint:
        print("📋 Checkpoint使用timm ViT，确保评估时也使用timm")
        # 检查timm是否可用
        try:
            from timm import create_model
            print("   ✅ timm库可用")
        except ImportError:
            print("   ⚠️  timm库不可用，将尝试使用自定义ViT（可能导致权重不匹配）")
    
    # 根据checkpoint的版本创建模型
    # 注意：当前代码总是创建新版本模型（支持unknown），但checkpoint可能是旧版本
    # 我们需要创建一个兼容旧版本的评估函数，或者修改模型以兼容旧checkpoint
    model = MaterialDetectionModel(
        backbone_name=backbone_name,
        img_size=args.img_size,
        num_classes=4,  # 已知类数量（不包括unknown）
        num_queries=num_queries,
        num_decoder_layers=num_decoder_layers,
        use_multi_view=use_multi_view,
        num_views=num_views,
        pretrained_backbone=pretrained_backbone
    ).to(device)
    
    # 检查backbone类型是否匹配
    model_state_keys = set(model.state_dict().keys())
    backbone_model_keys = {k for k in model_state_keys if 'backbone' in k}
    backbone_checkpoint_keys = {k for k in checkpoint_keys if 'backbone' in k}
    
    has_timm_in_model = any('backbone.backbone' in k for k in backbone_model_keys)
    
    if has_timm_in_checkpoint and not has_timm_in_model:
        print("⚠️  警告：Checkpoint使用timm ViT，但模型创建了自定义ViT")
        print("   这会导致backbone权重无法加载！")
        print("   请确保timm库已安装且可用：pip install timm")
        print("   或者检查TIMM_AVAILABLE是否为True")
    elif not has_timm_in_checkpoint and has_timm_in_model:
        print("⚠️  警告：Checkpoint使用自定义ViT，但模型创建了timm ViT")
        print("   这会导致backbone权重无法加载！")
    elif has_timm_in_checkpoint and has_timm_in_model:
        print("✅ Backbone类型匹配：都使用timm ViT")
    else:
        print("✅ Backbone类型匹配：都使用自定义ViT")
    
    # 加载权重（处理版本不匹配问题）
    checkpoint_state_dict = checkpoint['model_state_dict']
    model_state_dict = model.state_dict()
    
    # 如果checkpoint是旧版本（5维），模型也是5维（已移除unknown），直接加载
    # 如果checkpoint是新版本（6维），但模型是5维（已移除unknown），需要适配
    if checkpoint_class_embed_dim == 6:
        print("🔧 检测到checkpoint支持unknown类（6维输出），但当前模型已移除unknown类（5维输出）")
        print("   将适配checkpoint权重到当前模型...")
        # 创建适配的state_dict，移除unknown类权重
        adapted_state_dict = {}
        for key, value in checkpoint_state_dict.items():
            if 'class_embed.weight' in key:
                # 移除unknown类权重：从[6, 768] -> [5, 768]
                # 保留前4类+背景类，移除unknown类（第5维）
                old_weight = value  # [6, 768]
                new_weight = old_weight[[0, 1, 2, 3, 5]]  # 保留0-3类和背景类（第5维），跳过unknown（第4维）
                adapted_state_dict[key] = new_weight
                print(f"   ✅ 适配 class_embed.weight: [6, 768] -> [5, 768]（移除unknown类）")
            elif 'class_embed.bias' in key:
                # 移除unknown类bias：从[6] -> [5]
                old_bias = value  # [6]
                new_bias = old_bias[[0, 1, 2, 3, 5]]  # 保留0-3类和背景类，跳过unknown
                adapted_state_dict[key] = new_bias
                print(f"   ✅ 适配 class_embed.bias: [6] -> [5]（移除unknown类）")
            else:
                # 其他权重直接复制
                if key in model_state_dict:
                    if model_state_dict[key].shape == value.shape:
                        adapted_state_dict[key] = value
                    else:
                        print(f"   ⚠️  跳过形状不匹配的键: {key}")
                else:
                    print(f"   ⚠️  跳过不存在的键: {key}")
        
        # 加载适配后的权重
        missing_keys, unexpected_keys = model.load_state_dict(adapted_state_dict, strict=False)
        print(f"✅ 模型加载完成（已移除unknown类，epoch {checkpoint.get('epoch', 'unknown')}）")
        if missing_keys:
            print(f"   缺少的键: {len(missing_keys)} 个")
        if unexpected_keys:
            print(f"   意外的键: {len(unexpected_keys)} 个")
    elif checkpoint_num_classes == 4 and checkpoint_class_embed_dim == 5:
        print("✅ Checkpoint版本匹配：都是5维输出（4类+背景，无unknown）")
        # 直接加载，版本匹配
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state_dict, strict=False)
        print(f"✅ 模型加载完成 (epoch {checkpoint.get('epoch', 'unknown')})")
        if missing_keys:
            print(f"   缺少的键: {len(missing_keys)} 个")
        if unexpected_keys:
            print(f"   意外的键: {len(unexpected_keys)} 个")
    else:
        # 正常加载（版本匹配）
        try:
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint_state_dict, strict=False
            )
            if missing_keys:
                print(f"⚠️  缺少的键（将使用随机初始化）: {len(missing_keys)} 个")
                if len(missing_keys) < 20:  # 只显示前20个
                    for key in missing_keys[:20]:
                        print(f"     - {key}")
                else:
                    for key in missing_keys[:10]:
                        print(f"     - {key}")
                    print(f"     ... 还有 {len(missing_keys) - 10} 个")
            if unexpected_keys:
                print(f"⚠️  意外的键（将被忽略）: {len(unexpected_keys)} 个")
                if len(unexpected_keys) < 20:
                    for key in unexpected_keys[:20]:
                        print(f"     - {key}")
            print(f"✅ 模型加载完成 (epoch {checkpoint.get('epoch', 'unknown')})")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("   尝试使用strict=False加载...")
            model.load_state_dict(checkpoint_state_dict, strict=False)
            print(f"✅ 模型加载完成（部分权重，epoch {checkpoint.get('epoch', 'unknown')}）")
    
    # Benchmark数据集
    benchmark_dataset = BenchmarkDataset(
        benchmark_root=args.benchmark_root,
        img_size=args.img_size,
        num_views=num_views,
        use_multi_view=use_multi_view
    )
    
    benchmark_loader = DataLoader(
        benchmark_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"📊 Benchmark数据集大小: {len(benchmark_dataset)}")
    
    # 评估
    results = evaluate_benchmark(
        model, benchmark_loader, device,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold
    )
    
    # 保存结果
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 评估结果已保存到: {args.output}")
    
    # 打印总结
    print("\n" + "="*80)
    print("📋 Benchmark评估总结")
    print("="*80)
    print(f"总体准确率: {results['overall']['overall_accuracy']:.4f}")
    print(f"F1-Score: {results['overall']['f1']:.4f}")
    print(f"Precision: {results['overall']['precision']:.4f}")
    print(f"Recall: {results['overall']['recall']:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
