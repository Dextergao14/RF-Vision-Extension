#!/usr/bin/env python3
"""
检查训练过程，确认模型是否真的在学习
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from material_detection_model import MaterialDetectionModel
from material_dataset import MaterialDetectionDataset, collate_fn
from train_material_detection import HungarianMatcher, SetCriterion, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*80)
print("检查训练过程和数据加载")
print("="*80)

# 1. 检查数据加载
print("\n【1】检查数据加载:")
train_dataset = MaterialDetectionDataset(
    data_root='./vanilla-dataset',
    split='train',
    img_size=224,
    num_views=1,
    use_multi_view=False
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)

# 检查第一个batch
images, targets = next(iter(train_loader))
print(f"  图像形状: {images.shape}")
print(f"  第一个样本GT boxes数量: {len(targets[0]['boxes'])}")
print(f"  第一个样本GT labels: {targets[0]['labels'].tolist()}")
print(f"  第一个样本GT boxes (xyxy): {targets[0]['boxes'][:3] if len(targets[0]['boxes']) > 0 else 'None'}")

# 2. 检查模型初始化
print("\n【2】检查模型初始化:")
model = MaterialDetectionModel(
    backbone_name='vit_base_patch16_224',
    img_size=224,
    num_classes=4,
    num_queries=100,
    use_multi_view=False
).to(device)

# 检查初始输出
model.eval()
with torch.no_grad():
    test_images = images[:1].to(device)
    initial_outputs = model(test_images)
    initial_probs = F.softmax(initial_outputs['pred_logits'][0], dim=-1)
    initial_scores, initial_labels = initial_probs[:, :-1].max(dim=-1)
    print(f"  初始预测分数范围: [{initial_scores.min():.6f}, {initial_scores.max():.6f}]")
    print(f"  初始预测类别分布: {[initial_labels.tolist().count(i) for i in range(4)]}")

# 3. 检查损失函数计算
print("\n【3】检查损失函数计算:")
matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
weight_dict = {'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}
criterion = SetCriterion(4, matcher, weight_dict, eos_coef=0.1).to(device)

model.train()
test_images = images.to(device)
test_targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in t.items()} for t in targets]

with torch.no_grad():
    outputs = model(test_images)
    loss_dict = criterion(outputs, test_targets)
    total_loss = sum(loss_dict.values())
    
    print(f"  分类损失: {loss_dict['loss_ce'].item():.4f}")
    print(f"  边界框损失: {loss_dict['loss_bbox'].item():.4f}")
    print(f"  GIoU损失: {loss_dict['loss_giou'].item():.4f}")
    print(f"  总损失: {total_loss.item():.4f}")

# 4. 检查匹配过程
print("\n【4】检查匈牙利匹配:")
with torch.no_grad():
    outputs = model(test_images)
    indices = matcher(outputs, test_targets)
    
    print(f"  第一个样本匹配结果:")
    if len(indices) > 0:
        src_idx, tgt_idx = indices[0]
        print(f"    匹配的查询数量: {len(src_idx)}")
        print(f"    匹配的GT数量: {len(tgt_idx)}")
        print(f"    查询索引: {src_idx.tolist()[:10]}...")
        print(f"    GT索引: {tgt_idx.tolist()}")
        
        # 检查匹配的质量
        matched_pred_boxes = outputs['pred_boxes'][0][src_idx]
        matched_gt_boxes = test_targets[0]['boxes'][tgt_idx]
        matched_gt_labels = test_targets[0]['labels'][tgt_idx]
        
        # 转换格式
        pred_boxes_xyxy = box_cxcywh_to_xyxy(matched_pred_boxes)
        gt_boxes_cxcywh = box_xyxy_to_cxcywh(matched_gt_boxes)
        
        # 计算L1距离
        l1_dist = torch.abs(matched_pred_boxes - gt_boxes_cxcywh).mean(dim=1)
        print(f"    匹配的boxes L1距离: min={l1_dist.min():.4f}, max={l1_dist.max():.4f}, mean={l1_dist.mean():.4f}")
        
        # 检查类别匹配
        matched_pred_labels = outputs['pred_logits'][0][src_idx].argmax(dim=-1)
        matched_pred_labels = matched_pred_labels[matched_pred_labels < 4]  # 排除背景
        if len(matched_pred_labels) > 0 and len(matched_gt_labels) > 0:
            correct_cls = (matched_pred_labels[:len(matched_gt_labels)] == matched_gt_labels).sum().item()
            print(f"    匹配的类别正确数: {correct_cls} / {len(matched_gt_labels)}")

# 5. 检查训练后的模型
print("\n【5】检查训练后的模型:")
checkpoint = torch.load('./checkpoints_material_detection/checkpoint_best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
with torch.no_grad():
    test_images = images[:1].to(device)
    trained_outputs = model(test_images)
    trained_probs = F.softmax(trained_outputs['pred_logits'][0], dim=-1)
    trained_scores, trained_labels = trained_probs[:, :-1].max(dim=-1)
    
    print(f"  训练后预测分数范围: [{trained_scores.min():.6f}, {trained_scores.max():.6f}]")
    print(f"  训练后预测类别分布: {[trained_labels.tolist().count(i) for i in range(4)]}")
    
    # 计算损失
    model.train()
    loss_dict_trained = criterion(trained_outputs, test_targets[:1])
    total_loss_trained = sum(loss_dict_trained.values())
    print(f"  训练后损失: {total_loss_trained.item():.4f} (初始: {total_loss.item():.4f})")
    
    if total_loss_trained.item() < total_loss.item():
        print(f"  ✅ 损失降低了 {total_loss.item() - total_loss_trained.item():.4f}")
    else:
        print(f"  ⚠️ 损失没有降低，可能存在问题")

# 6. 检查数据标签
print("\n【6】检查数据标签分布:")
all_labels = []
for i, (_, targets) in enumerate(train_loader):
    for t in targets:
        all_labels.extend(t['labels'].tolist())
    if i >= 10:
        break

class_names = ['Concrete', 'Glass', 'Metal', 'Wood']
print(f"  前10个batch的标签分布:")
for cls_id, name in enumerate(class_names):
    count = all_labels.count(cls_id)
    print(f"    {name}: {count} ({count/len(all_labels)*100:.1f}%)" if all_labels else f"    {name}: 0")

print("\n" + "="*80)
print("【结论】")
print("="*80)
print("""
如果损失没有明显降低，或者预测分布与初始状态相似，说明：
1. 模型可能没有学习到有效特征
2. 训练过程可能存在问题（学习率、优化器等）
3. 数据加载或标签可能有问题
4. 损失函数计算可能有问题
""")
print("="*80)




