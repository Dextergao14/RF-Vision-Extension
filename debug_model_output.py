#!/usr/bin/env python3
"""
详细检查模型输出
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from material_detection_model import MaterialDetectionModel
from material_dataset import MaterialDetectionDataset, collate_fn
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

print("检查模型输出的详细统计...")
print("="*80)

all_scores = []
all_labels = []
all_boxes = []

with torch.no_grad():
    for i, (images, targets) in enumerate(test_loader):
        if i >= 10:  # 检查前10个样本
            break
        
        images = images.to(device)
        outputs = model(images)
        
        pred_logits = outputs['pred_logits'][0]
        pred_boxes = outputs['pred_boxes'][0]
        
        probs = F.softmax(pred_logits, dim=-1)
        scores, labels = probs[:, :-1].max(dim=-1)
        
        all_scores.extend(scores.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_boxes.append(pred_boxes.cpu())

print(f"\n预测分数统计 (前10个样本，共{len(all_scores)}个预测):")
print(f"  最小值: {min(all_scores):.6f}")
print(f"  最大值: {max(all_scores):.6f}")
print(f"  平均值: {np.mean(all_scores):.6f}")
print(f"  中位数: {np.median(all_scores):.6f}")
print(f"  分数 > 0.1的数量: {sum(1 for s in all_scores if s > 0.1)}")
print(f"  分数 > 0.05的数量: {sum(1 for s in all_scores if s > 0.05)}")
print(f"  分数 > 0.01的数量: {sum(1 for s in all_scores if s > 0.01)}")

print(f"\n预测类别分布:")
for cls_id in range(4):
    count = all_labels.count(cls_id)
    print(f"  类别 {cls_id}: {count} 个 ({count/len(all_labels)*100:.1f}%)")

all_boxes_tensor = torch.cat(all_boxes, dim=0)
print(f"\n预测boxes统计 (cxcywh格式，归一化):")
print(f"  cx: min={all_boxes_tensor[:, 0].min():.4f}, max={all_boxes_tensor[:, 0].max():.4f}, mean={all_boxes_tensor[:, 0].mean():.4f}")
print(f"  cy: min={all_boxes_tensor[:, 1].min():.4f}, max={all_boxes_tensor[:, 1].max():.4f}, mean={all_boxes_tensor[:, 1].mean():.4f}")
print(f"  w:  min={all_boxes_tensor[:, 2].min():.4f}, max={all_boxes_tensor[:, 2].max():.4f}, mean={all_boxes_tensor[:, 2].mean():.4f}")
print(f"  h:  min={all_boxes_tensor[:, 3].min():.4f}, max={all_boxes_tensor[:, 3].max():.4f}, mean={all_boxes_tensor[:, 3].mean():.4f}")

# 检查一个样本的详细输出
print("\n" + "="*80)
print("样本1的详细输出:")
with torch.no_grad():
    images, targets = next(iter(test_loader))
    images = images.to(device)
    outputs = model(images)
    
    pred_logits = outputs['pred_logits'][0]
    pred_boxes = outputs['pred_boxes'][0]
    
    probs = F.softmax(pred_logits, dim=-1)
    scores, labels = probs[:, :-1].max(dim=-1)
    
    print(f"GT boxes: {targets[0]['boxes']}")
    print(f"GT labels: {targets[0]['labels']}")
    print(f"\n前10个预测 (按分数排序):")
    sorted_indices = torch.argsort(scores, descending=True)
    for idx in sorted_indices[:10]:
        print(f"  查询{idx.item()}: 类别={labels[idx].item()}, 分数={scores[idx].item():.6f}, "
              f"box=({pred_boxes[idx, 0]:.4f}, {pred_boxes[idx, 1]:.4f}, {pred_boxes[idx, 2]:.4f}, {pred_boxes[idx, 3]:.4f})")
    
    # 检查背景类的概率
    bg_probs = probs[:, -1]
    print(f"\n背景类概率统计:")
    print(f"  最小值: {bg_probs.min():.6f}")
    print(f"  最大值: {bg_probs.max():.6f}")
    print(f"  平均值: {bg_probs.mean():.6f}")
    print(f"  背景概率 > 0.9的数量: {(bg_probs > 0.9).sum().item()}")

print("="*80)




