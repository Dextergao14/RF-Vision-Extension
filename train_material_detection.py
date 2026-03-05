#!/usr/bin/env python3
"""
训练材料检测模型
使用ViT/Swin backbone + DETR-style detection head
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import os
import argparse
from tqdm import tqdm
import json
from torch.cuda.amp import GradScaler, autocast
from scipy.optimize import linear_sum_assignment

from material_detection_model import MaterialDetectionModel
from material_dataset import MaterialDetectionDataset, collate_fn


class HungarianMatcher(nn.Module):
    """匈牙利匹配器，用于DETR的二分图匹配"""
    
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with 'pred_logits' [B, num_queries, num_classes+1] 
                     and 'pred_boxes' [B, num_queries, 4]
            targets: list of dicts with 'boxes' [N, 4] and 'labels' [N]
        Returns:
            list of indices tuples (pred_idx, target_idx) for each batch
        """
        bs, num_queries = outputs['pred_logits'].shape[:2]
        
        # 计算成本矩阵
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)  # [B*num_queries, num_classes+1]
        out_bbox = outputs['pred_boxes'].flatten(0, 1)  # [B*num_queries, 4]
        
        # 构建成本矩阵
        cost_list = []
        for b in range(bs):
            target_labels = targets[b]['labels']  # [N]
            target_boxes = targets[b]['boxes']  # [N, 4]
            
            # 分类成本（只考虑已知类，不包括unknown和background）
            cost_class = -out_prob[b * num_queries:(b + 1) * num_queries, target_labels]  # [num_queries, N]
            
            # 边界框成本（pred是cxcywh，target是xyxy，需要转换）
            pred_boxes_cxcywh = out_bbox[b * num_queries:(b + 1) * num_queries]
            target_boxes_cxcywh = box_xyxy_to_cxcywh(target_boxes)
            cost_bbox = torch.cdist(pred_boxes_cxcywh, target_boxes_cxcywh, p=1)  # [num_queries, N]
            
            # GIoU成本（pred_boxes是cxcywh格式，target_boxes是xyxy格式）
            pred_boxes_xyxy = box_cxcywh_to_xyxy(out_bbox[b * num_queries:(b + 1) * num_queries])
            cost_giou = -generalized_box_iou(pred_boxes_xyxy, target_boxes)  # [num_queries, N]
            
            # 总成本
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            cost_list.append(C)
        
        # 使用匈牙利算法匹配
        indices = [linear_sum_assignment(c.cpu().numpy()) for c in cost_list]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) 
                for i, j in indices]


def box_cxcywh_to_xyxy(x):
    """将(cx, cy, w, h)格式转换为(x1, y1, x2, y2)格式"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """将(x1, y1, x2, y2)格式转换为(cx, cy, w, h)格式"""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """计算广义IoU"""
    # 计算交集
    inter = box_intersection(boxes1, boxes2)
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter
    
    iou = inter / union
    
    # 计算最小外接矩形
    lti = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rbi = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    
    whi = (rbi - lti).clamp(min=0)
    areai = whi[:, :, 0] * whi[:, :, 1]
    
    giou = iou - (areai - union) / areai
    return giou


def box_intersection(boxes1, boxes2):
    """计算两个边界框集合的交集面积"""
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    return wh[:, :, 0] * wh[:, :, 1]


def box_area(boxes):
    """计算边界框面积"""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


class SetCriterion(nn.Module):
    """DETR的集合预测损失（只有4个已知类，无unknown类）"""
    
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes  # 已知类数量（4）
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        
        # 损失函数权重: [4个已知类, background]
        # 格式: num_classes + 1 = 4 + 1(background) = 5
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef  # 背景类权重
        self.register_buffer('empty_weight', empty_weight)
    
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """分类损失（只有4个已知类，无unknown类）"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # [B, num_queries, num_classes+1] (4类+background)
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        # 目标类别: 匹配的查询使用真实标签，未匹配的查询使用背景类（最后一个）
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,  # background类ID
                                   dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        # 标准交叉熵损失
        loss_ce = nn.functional.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        
        losses = {'loss_ce': loss_ce}
        return losses
    
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """边界框损失"""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]  # [num_matched, 4] (cxcywh格式)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # [num_matched, 4] (xyxy格式)
        
        # 将target转换为cxcywh格式用于L1损失
        target_boxes_cxcywh = box_xyxy_to_cxcywh(target_boxes)
        
        # L1损失
        loss_bbox = nn.functional.l1_loss(src_boxes, target_boxes_cxcywh, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}
        
        # GIoU损失（需要xyxy格式）
        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes_xyxy, target_boxes))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        
        return losses
    
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def forward(self, outputs, targets):
        """计算总损失"""
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # 匹配
        indices = self.matcher(outputs_without_aux, targets)
        
        # 计算匹配的目标数量
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        # 计算损失
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        
        # 加权求和
        losses = {k: losses[k] * self.weight_dict.get(k, 1.0) for k in losses.keys()}
        return losses


def train_epoch(model, dataloader, criterion, optimizer, device, scaler, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    loss_dict_sum = {}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (images, targets) in enumerate(pbar):
        # 移动到设备
        if images.dim() == 5:
            # 多视图: [B, num_views, C, H, W]
            images = images.to(device)
        else:
            # 单视图: [B, C, H, W]
            images = images.to(device)
        
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        with autocast():
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict.values())
        
        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 统计
        total_loss += loss.item()
        for k, v in loss_dict.items():
            if k not in loss_dict_sum:
                loss_dict_sum[k] = 0.0
            loss_dict_sum[k] += v.item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss/(batch_idx+1):.4f}",
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_loss_dict = {k: v / len(dataloader) for k, v in loss_dict_sum.items()}
    
    return avg_loss, avg_loss_dict


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    loss_dict_sum = {}
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validating"):
            # 移动到设备
            if images.dim() == 5:
                images = images.to(device)
            else:
                images = images.to(device)
            
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            with autocast():
                outputs = model(images)
                loss_dict = criterion(outputs, targets)
                loss = sum(loss_dict.values())
            
            total_loss += loss.item()
            for k, v in loss_dict.items():
                if k not in loss_dict_sum:
                    loss_dict_sum[k] = 0.0
                loss_dict_sum[k] += v.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_loss_dict = {k: v / len(dataloader) for k, v in loss_dict_sum.items()}
    
    return avg_loss, avg_loss_dict


def main():
    parser = argparse.ArgumentParser(description='训练材料检测模型')
    parser.add_argument('--data_root', type=str, 
                       default='./vanilla-dataset',
                       help='数据集根目录')
    parser.add_argument('--backbone', type=str, default='vit_base_patch16_224',
                       help='Backbone模型名称')
    parser.add_argument('--img_size', type=int, default=224, help='图像尺寸')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率（推荐1e-5）')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--num_queries', type=int, default=5, help='查询数量')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='解码器层数')
    parser.add_argument('--use_multi_view', action='store_true', help='使用多视图')
    parser.add_argument('--num_views', type=int, default=3, help='视图数量')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_material_detection',
                       help='保存目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点')
    parser.add_argument('--pretrained', action='store_true', default=True, help='使用预训练backbone（默认启用）')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 数据集
    train_dataset = MaterialDetectionDataset(
        data_root=args.data_root,
        split='train',
        img_size=args.img_size,
        num_views=args.num_views,
        use_multi_view=args.use_multi_view
    )
    
    val_dataset = MaterialDetectionDataset(
        data_root=args.data_root,
        split='valid',
        img_size=args.img_size,
        num_views=args.num_views,
        use_multi_view=args.use_multi_view
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # 模型（支持unknown类）
    model = MaterialDetectionModel(
        backbone_name=args.backbone,
        img_size=args.img_size,
        num_classes=4,  # 已知类数量（不包括unknown）
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers,
        use_multi_view=args.use_multi_view,
        num_views=args.num_views,
        pretrained_backbone=args.pretrained
    ).to(device)
    
    print(f"📊 模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 损失函数 - 调整权重和背景类权重（只有4个已知类，无unknown类）
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    weight_dict = {'loss_ce': 2.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}  # 增加分类损失权重
    criterion = SetCriterion(4, matcher, weight_dict, eos_coef=0.25).to(device)  # 增加背景类权重
    
    # 优化器 - 使用更小的学习率
    base_lr = args.lr if args.lr < 1e-4 else 1e-5  # 如果lr >= 1e-4，使用1e-5
    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器 - 添加warmup
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # 线性warmup
        else:
            # Cosine annealing after warmup
            progress = (epoch - warmup_epochs) / (args.num_epochs - warmup_epochs)
            import math
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"✅ 从epoch {start_epoch}恢复训练")
    
    # 训练
    print(f"🚀 开始训练，共 {args.num_epochs} 个epoch...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.num_epochs):
        # 训练
        train_loss, train_loss_dict = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch + 1
        )
        
        # 验证
        val_loss, val_loss_dict = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 打印损失
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print(f"  训练损失: {train_loss:.4f} {train_loss_dict}")
        print(f"  验证损失: {val_loss:.4f} {val_loss_dict}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'args': vars(args)
        }
        
        # 保存最新检查点
        torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint_latest.pth'))
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint_best.pth'))
            print(f"  💾 保存最佳模型 (val_loss: {val_loss:.4f})")
        
        # 定期保存
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print(f"✅ 训练完成！最佳验证损失: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()

