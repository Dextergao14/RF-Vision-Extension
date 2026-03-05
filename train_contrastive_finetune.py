#!/usr/bin/env python3
"""
在已有 checkpoint 上 fine-tune，加入 Supervised Contrastive Loss。
目标：让同类 decoder 特征聚拢、不同类推远，形成紧致的类别簇。
推理时用投影特征到类别原型的距离判定 Unknown。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from scipy.optimize import linear_sum_assignment
import os
import math
import argparse
from tqdm import tqdm
from collections import defaultdict

from material_detection_model import MaterialDetectionModel
from material_dataset import MaterialDetectionDataset, collate_fn
from train_material_detection import (
    HungarianMatcher, SetCriterion,
    box_cxcywh_to_xyxy, box_xyxy_to_cxcywh,
    generalized_box_iou, train_epoch, validate,
)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (SupCon).
    For each matched query, pull same-class queries together and push different-class queries away.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, proj_features, labels, mask):
        """
        Args:
            proj_features: [N, dim] L2-normalized projected features (matched queries only)
            labels: [N] class labels for each feature
            mask: not used, kept for interface compatibility
        Returns:
            scalar loss
        """
        if len(proj_features) < 2:
            return torch.tensor(0.0, device=proj_features.device)

        N = proj_features.shape[0]
        sim_matrix = proj_features @ proj_features.T / self.temperature  # [N, N]

        # Mask: same class = positive, different class = negative
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [N, N]
        # Remove self-similarity from positives
        self_mask = ~torch.eye(N, dtype=torch.bool, device=proj_features.device)
        positives = label_eq & self_mask
        
        # Need at least one positive per anchor
        has_positive = positives.sum(dim=1) > 0
        if has_positive.sum() == 0:
            return torch.tensor(0.0, device=proj_features.device)

        # For numerical stability
        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        # Log-sum-exp over all non-self entries (denominator)
        exp_sim = torch.exp(sim_matrix) * self_mask.float()
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)

        # Mean of log(exp(sim_pos) / sum(exp(sim_all))) over positives
        log_prob = sim_matrix - log_denom
        
        # Only compute for anchors that have at least one positive
        pos_log_prob = (log_prob * positives.float()).sum(dim=1)
        num_positives = positives.sum(dim=1).float().clamp(min=1)
        loss = -(pos_log_prob[has_positive] / num_positives[has_positive]).mean()
        
        return loss


class ContrastiveSetCriterion(SetCriterion):
    """Extended SetCriterion with SupCon loss on projected decoder features."""

    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1,
                 con_weight=1.0, temperature=0.07):
        super().__init__(num_classes, matcher, weight_dict, eos_coef)
        self.supcon = SupConLoss(temperature=temperature)
        self.con_weight = con_weight

    def forward(self, outputs, targets):
        losses = super().forward(outputs, targets)

        # Compute contrastive loss on matched queries
        indices = self.matcher(outputs, targets)
        proj_feats = outputs.get('proj_features')
        if proj_feats is not None:
            matched_feats = []
            matched_labels = []
            for b, (pred_idx, gt_idx) in enumerate(indices):
                if len(pred_idx) == 0:
                    continue
                matched_feats.append(proj_feats[b, pred_idx])
                matched_labels.append(targets[b]['labels'][gt_idx])

            if matched_feats:
                all_feats = torch.cat(matched_feats, dim=0)     # [M, 128]
                all_labels = torch.cat(matched_labels, dim=0)   # [M]
                loss_con = self.supcon(all_feats, all_labels, None)
                losses['loss_con'] = loss_con * self.con_weight

        return losses


def train_epoch_contrastive(model, dataloader, criterion, optimizer, device, scaler, epoch):
    model.train()
    total_loss = 0.0
    loss_dict_sum = {}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict.values())

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_dict_sum[k] = loss_dict_sum.get(k, 0.0) + v.item()

        pbar.set_postfix({'loss': f"{loss.item():.4f}",
                          'con': f"{loss_dict.get('loss_con', torch.tensor(0.0)).item():.4f}"})

    n = len(dataloader)
    return total_loss / n, {k: v / n for k, v in loss_dict_sum.items()}


def main():
    parser = argparse.ArgumentParser(description='Contrastive fine-tune')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str,
                        default='./Lab Scenes.v2-2026_train_0120.yolov8')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--con_weight', type=float, default=1.0,
                        help='Weight for contrastive loss')
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--save_dir', type=str,
                        default='./checkpoints_contrastive')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    print(f"📂 Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ma = ckpt.get('args', {})
    backbone_name = ma.get('backbone', 'vit_base_patch16_224')
    num_queries = ma.get('num_queries', 100)
    num_decoder_layers = ma.get('num_decoder_layers', 6)

    model = MaterialDetectionModel(
        backbone_name=backbone_name, img_size=args.img_size, num_classes=4,
        num_queries=num_queries, num_decoder_layers=num_decoder_layers,
        use_multi_view=False, num_views=1, pretrained_backbone=False,
    ).to(device)

    # Load existing weights (projection_head is new, won't be in checkpoint)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    print("✅ Model loaded (projection_head initialized randomly)")

    # Dataset
    train_dataset = MaterialDetectionDataset(
        data_root=args.data_root, split='train', img_size=args.img_size,
        num_views=1, use_multi_view=False
    )
    val_dataset = MaterialDetectionDataset(
        data_root=args.data_root, split='valid', img_size=args.img_size,
        num_views=1, use_multi_view=False
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_fn, pin_memory=True)
    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Criterion with contrastive loss
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    weight_dict = {'loss_ce': 2.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}
    criterion = ContrastiveSetCriterion(
        4, matcher, weight_dict, eos_coef=0.25,
        con_weight=args.con_weight, temperature=args.temperature
    ).to(device)

    # Optimizer: higher lr for projection head, lower for the rest
    proj_params = list(model.projection_head.parameters())
    other_params = [p for n, p in model.named_parameters()
                    if 'projection_head' not in n]
    optimizer = optim.AdamW([
        {'params': other_params, 'lr': args.lr * 0.1},
        {'params': proj_params, 'lr': args.lr},
    ], weight_decay=args.weight_decay)

    # Cosine scheduler with warmup
    warmup_epochs = 3
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.num_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    print(f"🚀 Contrastive fine-tuning for {args.num_epochs} epochs (con_weight={args.con_weight}, T={args.temperature})")
    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        train_loss, train_dict = train_epoch_contrastive(
            model, train_loader, criterion, optimizer, device, scaler, epoch + 1
        )
        val_loss, val_dict = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print(f"  Train: {train_loss:.4f} {train_dict}")
        print(f"  Val:   {val_loss:.4f} {val_dict}")
        print(f"  LR: backbone={optimizer.param_groups[0]['lr']:.2e}, proj={optimizer.param_groups[1]['lr']:.2e}")

        ckpt_save = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'args': {**ma, 'con_weight': args.con_weight, 'temperature': args.temperature},
        }
        torch.save(ckpt_save, os.path.join(args.save_dir, 'checkpoint_latest.pth'))
        if epoch % 10 == 9:
            torch.save(ckpt_save, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt_save, os.path.join(args.save_dir, 'checkpoint_best.pth'))
            print(f"  ✅ Best model saved (val_loss={val_loss:.4f})")

    print(f"\n✅ Training complete. Best val_loss={best_val_loss:.4f}")
    print(f"💾 Checkpoints saved to {args.save_dir}/")


if __name__ == '__main__':
    main()
