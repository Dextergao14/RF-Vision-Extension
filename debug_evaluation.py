#!/usr/bin/env python3
"""
调试评估脚本 - 检查模型输出和评估逻辑
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from material_detection_model import MaterialDetectionModel
from material_dataset import MaterialDetectionDataset, collate_fn

# 加载模型
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

# 加载测试集
test_dataset = MaterialDetectionDataset(
    data_root='./vanilla-dataset',
    split='test',
    img_size=224,
    num_views=1,
    use_multi_view=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

print("检查前3个样本的模型输出...")
print("="*80)

with torch.no_grad():
    for i, (images, targets) in enumerate(test_loader):
        if i >= 3:
            break
        
        images = images.to(device)
        outputs = model(images)
        
        pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes+1]
        pred_boxes = outputs['pred_boxes'][0]  # [num_queries, 4]
        
        probs = F.softmax(pred_logits, dim=-1)
        scores, labels = probs[:, :-1].max(dim=-1)
        
        # 过滤低分预测
        valid_mask = scores > 0.1
        valid_preds = valid_mask.sum().item()
        
        print(f"\n样本 {i+1}:")
        print(f"  图像形状: {images.shape}")
        print(f"  GT boxes数量: {len(targets[0]['boxes'])}")
        print(f"  GT labels: {targets[0]['labels'].tolist()}")
        print(f"  GT boxes (前3个): {targets[0]['boxes'][:3] if len(targets[0]['boxes']) > 0 else 'None'}")
        print(f"  预测的查询数量: {len(pred_logits)}")
        print(f"  有效预测数量 (score > 0.1): {valid_preds}")
        
        if valid_preds > 0:
            print(f"  预测的类别分布:")
            pred_labels_list = labels[valid_mask].cpu().tolist()
            for cls_id in range(4):
                count = pred_labels_list.count(cls_id)
                if count > 0:
                    print(f"    类别 {cls_id}: {count} 个")
            
            print(f"  预测分数范围: {scores[valid_mask].min().item():.4f} - {scores[valid_mask].max().item():.4f}")
            print(f"  预测boxes范围 (cxcywh):")
            valid_boxes = pred_boxes[valid_mask]
            print(f"    cx: [{valid_boxes[:, 0].min():.4f}, {valid_boxes[:, 0].max():.4f}]")
            print(f"    cy: [{valid_boxes[:, 1].min():.4f}, {valid_boxes[:, 1].max():.4f}]")
            print(f"    w:  [{valid_boxes[:, 2].min():.4f}, {valid_boxes[:, 2].max():.4f}]")
            print(f"    h:  [{valid_boxes[:, 3].min():.4f}, {valid_boxes[:, 3].max():.4f}]")
            
            # 检查IoU
            if len(targets[0]['boxes']) > 0:
                from train_material_detection import box_cxcywh_to_xyxy
                pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[valid_mask])
                gt_boxes = targets[0]['boxes'].to(device)
                
                # 计算所有pair的IoU
                max_ious = []
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
                    max_ious.append(best_iou)
                
                if max_ious:
                    print(f"  预测与GT的最大IoU: min={min(max_ious):.4f}, max={max(max_ious):.4f}, mean={sum(max_ious)/len(max_ious):.4f}")
                    print(f"  IoU > 0.5的数量: {sum(1 for iou in max_ious if iou > 0.5)}")
        else:
            print(f"  没有有效预测 (所有预测分数都 <= 0.1)")

print("\n" + "="*80)




