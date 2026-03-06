#!/usr/bin/env python3
"""
方案 C：两阶段 Unknown 检测
1. Fine-tune 原始模型（20% Eval_benchmark）
2. 收集 decoder features：匹配 GT 的 = known(1)，未匹配的 = negative(0)
3. 训练轻量 binary MLP（known vs OOD）
4. 在 brick / hard_samples 上推理 + 可视化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import cv2
import random
from pathlib import Path
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from tqdm import tqdm
from collections import defaultdict
import argparse

from material_detection_model import MaterialDetectionModel
from evaluate_benchmark import BenchmarkDataset
from material_dataset import collate_fn
from train_material_detection import (
    box_cxcywh_to_xyxy, HungarianMatcher, SetCriterion
)

try:
    from torchvision.ops import nms as torch_nms
except ImportError:
    torch_nms = None

CLASS_NAMES = ['Concrete', 'Glass', 'Metal', 'Wood', 'Unknown']
COLORS_BGR = {
    0: (0, 0, 255),
    1: (0, 200, 0),
    2: (255, 100, 0),
    3: (0, 220, 220),
    4: (200, 0, 200),
}
GT_COLOR = (0, 255, 0)
SCORE_THRESHOLDS = [0.25, 0.45, 0.4, 0.4]
NMS_IOU = 0.1
BG_THRESHOLD = 0.5


class OODHead(nn.Module):
    """Binary MLP: decoder feature -> known(1) / unknown(0)"""
    def __init__(self, d_model=768, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def finetune_model(model, dataset, device, support_pct=0.2, epochs=30, seed=42):
    """Fine-tune on support set, return model + support/eval indices."""
    n = len(dataset)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    n_support = int(n * support_pct)
    support_idx = indices[:n_support]
    eval_idx = indices[n_support:]

    support_loader = DataLoader(
        Subset(dataset, support_idx), batch_size=4, shuffle=True,
        num_workers=0, collate_fn=collate_fn
    )

    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    criterion = SetCriterion(num_classes=4, matcher=matcher,
                             weight_dict={'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, targets in support_loader:
            images = images.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()} for t in targets]
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict[k] * criterion.weight_dict.get(k, 1.0)
                       for k in loss_dict if k in criterion.weight_dict)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Fine-tune epoch {epoch+1}: loss={total_loss/len(support_loader):.4f}")

    model.eval()
    return support_idx, eval_idx


def collect_features(model, dataset, indices, device):
    """从检测结果收集 known/negative decoder features。"""
    loader = DataLoader(
        Subset(dataset, indices), batch_size=8, shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )

    known_feats = []
    negative_feats = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Collecting features"):
            images = images.to(device)
            targets_dev = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in t.items()} for t in targets]
            out = model(images)
            B = images.shape[0]

            for b in range(B):
                logits = out['pred_logits'][b]
                boxes = out['pred_boxes'][b]
                feats = out['decoder_features'][b]  # [Q, d_model]
                gt_boxes = targets_dev[b]['boxes']
                gt_labels = targets_dev[b]['labels']

                probs = F.softmax(logits, dim=-1)
                bg = probs[:, -1]
                kp = probs[:, :-1]
                max_scores, max_labels = kp.max(dim=-1)

                det_queries = []
                for q in range(logits.shape[0]):
                    if bg[q] >= BG_THRESHOLD:
                        continue
                    c = max_labels[q].item()
                    if max_scores[q].item() < SCORE_THRESHOLDS[c]:
                        continue
                    det_queries.append(q)

                if not det_queries or len(gt_boxes) == 0:
                    for q in det_queries:
                        negative_feats.append(feats[q].cpu())
                    continue

                pred_xyxy = box_cxcywh_to_xyxy(boxes[det_queries])
                gt_matched = set()

                for di, qi in enumerate(det_queries):
                    pbox = pred_xyxy[di]
                    best_iou = 0.0
                    best_gi = -1
                    for gi in range(len(gt_boxes)):
                        if gi in gt_matched:
                            continue
                        gbox = gt_boxes[gi]
                        ix1 = max(pbox[0].item(), gbox[0].item())
                        iy1 = max(pbox[1].item(), gbox[1].item())
                        ix2 = min(pbox[2].item(), gbox[2].item())
                        iy2 = min(pbox[3].item(), gbox[3].item())
                        if ix2 > ix1 and iy2 > iy1:
                            inter = (ix2 - ix1) * (iy2 - iy1)
                            a1 = (pbox[2] - pbox[0]).item() * (pbox[3] - pbox[1]).item()
                            a2 = (gbox[2] - gbox[0]).item() * (gbox[3] - gbox[1]).item()
                            iou = inter / (a1 + a2 - inter + 1e-6)
                        else:
                            iou = 0.0
                        if iou > best_iou:
                            best_iou = iou
                            best_gi = gi

                    if best_iou >= 0.3:
                        gt_matched.add(best_gi)
                        known_feats.append(feats[qi].cpu())
                    else:
                        negative_feats.append(feats[qi].cpu())

    print(f"  Known features: {len(known_feats)}, Negative features: {len(negative_feats)}")
    return known_feats, negative_feats


def train_ood_head(known_feats, negative_feats, device, epochs=100, d_model=768):
    """训练 binary MLP."""
    X_pos = torch.stack(known_feats)
    X_neg = torch.stack(negative_feats)
    y_pos = torch.ones(len(X_pos))
    y_neg = torch.zeros(len(X_neg))

    X = torch.cat([X_pos, X_neg], dim=0)
    y = torch.cat([y_pos, y_neg], dim=0)
    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]

    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    head = OODHead(d_model=d_model).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    # class weight: balance known/negative
    pos_weight = torch.tensor([len(X_neg) / max(len(X_pos), 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    head.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = head(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            preds = (logits > 0).float()
            correct += (preds == yb).sum().item()
            total += len(yb)
        if (epoch + 1) % 20 == 0:
            acc = correct / total * 100
            print(f"  OOD head epoch {epoch+1}: loss={total_loss/total:.4f}, acc={acc:.1f}%")

    head.eval()
    return head


def load_coco_gt(ann_path):
    with open(ann_path) as f:
        coco = json.load(f)
    cat_map = {c['id']: c['name'] for c in coco['categories'] if c['name'] != 'objects'}
    gt_by_file = {}
    img_id_to_file = {im['id']: im['file_name'] for im in coco['images']}
    for ann in coco['annotations']:
        fname = img_id_to_file[ann['image_id']]
        x, y, w, h = [float(v) for v in ann['bbox']]
        cat_name = cat_map.get(ann['category_id'], 'unknown')
        gt_by_file.setdefault(fname, []).append({
            'bbox_xyxy': [x, y, x + w, y + h],
            'class_name': cat_name,
        })
    return gt_by_file


def nms_per_class(boxes, labels, scores, iou_thr):
    if torch_nms is None or len(boxes) == 0:
        return boxes, labels, scores
    keep = []
    for c in range(5):
        mask = labels == c
        if mask.sum() == 0:
            continue
        idx = torch.where(mask)[0]
        k = torch_nms(boxes[idx].float(), scores[idx].float(), iou_thr)
        keep.append(idx[k])
    if not keep:
        return boxes[:0], labels[:0], scores[:0]
    keep = torch.cat(keep)
    keep = torch.sort(keep)[0]
    return boxes[keep], labels[keep], scores[keep]


def infer_and_visualize(model, ood_head, img_dir, gt_by_file, device, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img_files = sorted(Path(img_dir).glob('*.png'))
    for img_path in img_files:
        fname = img_path.name
        orig = cv2.imread(str(img_path))
        if orig is None:
            continue
        oh, ow = orig.shape[:2]
        vis = orig.copy()

        img_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        inp = tfm(img_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(inp)
        logits = out['pred_logits'][0]
        boxes = out['pred_boxes'][0]
        feats = out['decoder_features'][0]

        probs = F.softmax(logits, dim=-1)
        bg = probs[:, -1]
        kp = probs[:, :-1]
        max_scores, max_labels = kp.max(dim=-1)

        all_boxes, all_labels, all_scores = [], [], []
        for q in range(logits.shape[0]):
            if bg[q] >= BG_THRESHOLD:
                continue
            c = max_labels[q].item()
            sc = max_scores[q].item()
            if sc < min(SCORE_THRESHOLDS):
                continue

            bxy = box_cxcywh_to_xyxy(boxes[q:q+1])[0]

            with torch.no_grad():
                ood_logit = ood_head(feats[q:q+1])
                is_known = (ood_logit > 0).item()

            if is_known and sc >= SCORE_THRESHOLDS[c]:
                all_boxes.append(bxy)
                all_labels.append(c)
                all_scores.append(sc)
            elif not is_known and sc >= min(SCORE_THRESHOLDS):
                all_boxes.append(bxy)
                all_labels.append(4)
                all_scores.append(sc)

        if all_boxes:
            all_boxes = torch.stack(all_boxes)
            all_labels = torch.tensor(all_labels, device=device)
            all_scores = torch.tensor(all_scores, device=device)
            all_boxes, all_labels, all_scores = nms_per_class(
                all_boxes, all_labels, all_scores, NMS_IOU)
        else:
            all_boxes = torch.zeros((0, 4), device=device)
            all_labels = torch.zeros(0, dtype=torch.long, device=device)
            all_scores = torch.zeros(0, device=device)

        gts = gt_by_file.get(fname, [])
        for g in gts:
            x1, y1, x2, y2 = [int(v) for v in g['bbox_xyxy']]
            cv2.rectangle(vis, (x1, y1), (x2, y2), GT_COLOR, 2, cv2.LINE_AA)
            label_gt = f"GT:{g['class_name']}"
            (tw, th), _ = cv2.getTextSize(label_gt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), GT_COLOR, -1)
            cv2.putText(vis, label_gt, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(all_labels)):
            bx = all_boxes[i].cpu().numpy()
            x1, y1 = int(bx[0] * ow), int(bx[1] * oh)
            x2, y2 = int(bx[2] * ow), int(bx[3] * oh)
            cls = all_labels[i].item()
            sc = all_scores[i].item()
            color = COLORS_BGR.get(cls, (255, 255, 255))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
            label_p = f"{CLASS_NAMES[cls]}:{sc:.2f}"
            (tw, th), _ = cv2.getTextSize(label_p, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(vis, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(vis, label_p, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        n_known = (all_labels < 4).sum().item()
        n_unknown = (all_labels == 4).sum().item()
        cv2.imwrite(str(out_dir / f"vis_{fname}"), vis)
        print(f"  {fname}: GT={len(gts)}, Known={n_known}, Unknown={n_unknown}")

    print(f"\n  Saved to: {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='../checkpoints_material_detection_new_data/checkpoint_best.pth')
    parser.add_argument('--benchmark', default='./Eval_benchmark')
    parser.add_argument('--brick', action='store_true')
    parser.add_argument('--unknown', action='store_true', help='marble')
    parser.add_argument('--finetune_epochs', type=int, default=30)
    parser.add_argument('--ood_epochs', type=int, default=100)
    parser.add_argument('--support_pct', type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.brick:
        img_dir = './unknown_material_samples_2/train'
        ann_path = './unknown_material_samples_2/train/_annotations.coco.json'
        out_dir = './unknown_material_visualizations_brick'
    elif args.unknown:
        img_dir = './unknown_material_samples/train'
        ann_path = './unknown_material_samples/train/_annotations.coco.json'
        out_dir = './unknown_material_visualizations'
    else:
        img_dir = './hard_samples/test'
        ann_path = './hard_samples/test/_annotations.coco.json'
        out_dir = './hard_samples_visualizations'

    # Step 1: Load model
    print(f"[1/4] Loading model: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ma = ckpt.get('args', {})
    model = MaterialDetectionModel(
        backbone_name=ma.get('backbone', 'vit_base_patch16_224'),
        img_size=224, num_classes=4,
        num_queries=ma.get('num_queries', 100),
        num_decoder_layers=ma.get('num_decoder_layers', 6),
        use_multi_view=False, num_views=1, pretrained_backbone=False,
    ).to(device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    d_model = ma.get('d_model', 768)

    # Step 2: Fine-tune
    print(f"[2/4] Fine-tuning on {args.support_pct*100:.0f}% Eval_benchmark...")
    dataset = BenchmarkDataset(benchmark_root=args.benchmark, img_size=224)
    support_idx, eval_idx = finetune_model(
        model, dataset, device,
        support_pct=args.support_pct, epochs=args.finetune_epochs
    )
    print(f"  Support: {len(support_idx)}, Eval: {len(eval_idx)}")

    # Step 3: Collect features + train OOD head
    print(f"[3/4] Collecting features from eval set...")
    known_feats, neg_feats = collect_features(model, dataset, eval_idx, device)

    if len(neg_feats) < 10:
        print(f"  Not enough negatives ({len(neg_feats)}), adding random noise features...")
        for _ in range(max(200, len(known_feats))):
            neg_feats.append(torch.randn(d_model))

    print(f"  Training OOD head ({args.ood_epochs} epochs)...")
    ood_head = train_ood_head(known_feats, neg_feats, device,
                              epochs=args.ood_epochs, d_model=d_model)

    # Step 4: Infer + visualize
    print(f"[4/4] Inference on target images...")
    gt_by_file = load_coco_gt(ann_path)
    print(f"  GT: {len(gt_by_file)} images, {sum(len(v) for v in gt_by_file.values())} boxes")
    infer_and_visualize(model, ood_head, img_dir, gt_by_file, device, out_dir)


if __name__ == '__main__':
    main()
