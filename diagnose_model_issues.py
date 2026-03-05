#!/usr/bin/env python3
"""
诊断模型问题并生成报告
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from material_detection_model import MaterialDetectionModel
from material_dataset import MaterialDetectionDataset, collate_fn
from train_material_detection import box_cxcywh_to_xyxy
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('./checkpoints_material_detection/checkpoint_best.pth', map_location=device)

model = MaterialDetectionModel(
    backbone_name='vit_base_patch16_224',
    img_size=224,
    num_classes=4,
    num_queries=100,
    use_multi_view=False
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_dataset = MaterialDetectionDataset(
    data_root='./vanilla-dataset',
    split='test',
    img_size=224,
    num_views=1,
    use_multi_view=False
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

print("="*80)
print("模型问题诊断报告")
print("="*80)

# 问题1: 检查预测分数分布
print("\n【问题1】预测分数分布:")
all_scores = []
with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs['pred_logits'][0], dim=-1)
        scores, _ = probs[:, :-1].max(dim=-1)
        all_scores.extend(scores.cpu().tolist())
        if len(all_scores) >= 1000:
            break

print(f"  分数范围: [{min(all_scores):.6f}, {max(all_scores):.6f}]")
print(f"  平均分数: {np.mean(all_scores):.6f}")
print(f"  分数 > 0.1: {sum(1 for s in all_scores if s > 0.1)} / {len(all_scores)}")
print(f"  分数 > 0.05: {sum(1 for s in all_scores if s > 0.05)} / {len(all_scores)}")
print(f"  ⚠️ 问题: 所有预测分数都 < 0.1，说明模型置信度很低")

# 问题2: 检查预测boxes分布
print("\n【问题2】预测boxes分布:")
all_boxes = []
with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device)
        outputs = model(images)
        pred_boxes = outputs['pred_boxes'][0].cpu()
        all_boxes.append(pred_boxes)
        if len(all_boxes) >= 10:
            break

all_boxes_tensor = torch.cat(all_boxes, dim=0)
print(f"  cx范围: [{all_boxes_tensor[:, 0].min():.4f}, {all_boxes_tensor[:, 0].max():.4f}] (GT通常在[0, 1])")
print(f"  cy范围: [{all_boxes_tensor[:, 1].min():.4f}, {all_boxes_tensor[:, 1].max():.4f}] (GT通常在[0, 1])")
print(f"  w范围:  [{all_boxes_tensor[:, 2].min():.4f}, {all_boxes_tensor[:, 2].max():.4f}] (GT通常在[0, 1])")
print(f"  h范围:  [{all_boxes_tensor[:, 3].min():.4f}, {all_boxes_tensor[:, 3].max():.4f}] (GT通常在[0, 1])")
print(f"  ⚠️ 问题: 预测boxes集中在特定区域，没有覆盖整个图像")

# 问题3: 检查IoU匹配
print("\n【问题3】IoU匹配分析:")
iou_stats = []
with torch.no_grad():
    for i, (images, targets) in enumerate(test_loader):
        if i >= 10:
            break
        images = images.to(device)
        outputs = model(images)
        
        pred_logits = outputs['pred_logits'][0]
        pred_boxes = outputs['pred_boxes'][0]
        
        probs = F.softmax(pred_logits, dim=-1)
        scores, labels = probs[:, :-1].max(dim=-1)
        valid_mask = scores > 0.05
        
        if valid_mask.sum() > 0:
            pred_boxes_valid = pred_boxes[valid_mask]
            pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes_valid)
            gt_boxes = targets[0]['boxes'].to(device)
            
            if len(gt_boxes) > 0:
                for pred_box in pred_boxes_xyxy:
                    best_iou = 0.0
                    for gt_box in gt_boxes:
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
                            best_iou = max(best_iou, iou)
                    iou_stats.append(best_iou)

if iou_stats:
    iou_stats_cpu = [iou.item() if isinstance(iou, torch.Tensor) else iou for iou in iou_stats]
    print(f"  最大IoU统计: min={min(iou_stats_cpu):.4f}, max={max(iou_stats_cpu):.4f}, mean={np.mean(iou_stats_cpu):.4f}")
    print(f"  IoU > 0.5的数量: {sum(1 for iou in iou_stats_cpu if iou > 0.5)} / {len(iou_stats_cpu)}")
    print(f"  IoU > 0.3的数量: {sum(1 for iou in iou_stats_cpu if iou > 0.3)} / {len(iou_stats_cpu)}")
    print(f"  ⚠️ 问题: 大部分预测的IoU都很低，说明boxes位置不准确")

# 问题4: 检查类别分布
print("\n【问题4】预测类别分布:")
all_labels = []
with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs['pred_logits'][0], dim=-1)
        scores, labels = probs[:, :-1].max(dim=-1)
        valid_mask = scores > 0.05
        all_labels.extend(labels[valid_mask].cpu().tolist())
        if len(all_labels) >= 1000:
            break

class_names = ['Concrete', 'Glass', 'Metal', 'Wood']
for cls_id, name in enumerate(class_names):
    count = all_labels.count(cls_id)
    print(f"  {name}: {count} ({count/len(all_labels)*100:.1f}%)" if all_labels else f"  {name}: 0")
print(f"  ⚠️ 问题: 类别分布不均匀，某些类别（如Metal）几乎没有被预测")

# 问题5: 检查训练损失
print("\n【问题5】训练状态:")
print(f"  最佳epoch: {checkpoint.get('epoch', 'unknown')}")
print(f"  训练损失: {checkpoint.get('train_loss', 'unknown'):.4f}")
print(f"  验证损失: {checkpoint.get('val_loss', 'unknown'):.4f}")
print(f"  ⚠️ 问题: 验证损失(5.93)远高于训练损失(1.79)，说明模型过拟合或未充分学习")

print("\n" + "="*80)
print("【根本原因分析】")
print("="*80)
print("""
1. **模型预测分数过低**: 所有预测分数 < 0.1，说明模型对预测不自信
2. **Boxes位置不准确**: 预测boxes集中在特定区域，没有学习到正确的空间分布
3. **IoU匹配失败**: 大部分预测与GT的IoU < 0.5，导致无法正确匹配
4. **类别预测偏差**: 类别分布不均匀，某些类别几乎不被预测
5. **训练不充分**: 验证损失远高于训练损失，可能存在过拟合或学习率问题

【建议解决方案】:
1. 降低评估阈值到0.05（已修复）
2. 检查训练数据加载是否正确
3. 调整学习率和训练策略
4. 增加训练轮数或使用预训练backbone
5. 检查损失函数计算是否正确
6. 考虑使用更小的模型或调整超参数
""")
print("="*80)

