#!/usr/bin/env python3
"""
方案 B：Outlier Exposure Fine-tune
在 20% Eval_benchmark + brick 样本（Brick→background）上 fine-tune。
模型学会对未知材料输出 background 分数。
然后在 brick / hard_samples 上推理并可视化。
"""

import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np
import json
import cv2
import random
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision import transforms
from tqdm import tqdm
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
COLORS_BGR = {0: (0,0,255), 1: (0,200,0), 2: (255,100,0), 3: (0,220,220), 4: (200,0,200)}
GT_COLOR = (0, 255, 0)
SCORE_THRESHOLDS = [0.25, 0.45, 0.4, 0.4]
NMS_IOU = 0.1
BG_THRESHOLD = 0.5

# brick COCO cat mapping → model class
# cat_id 1=Brick → background (num_classes=4)
# cat_id 2=Glass → 1, cat_id 3=Metal → 2, cat_id 4=Wood → 3
BRICK_CAT_TO_MODEL = {1: 4, 2: 1, 3: 2, 4: 3}  # Brick→bg, others stay


class BrickOutlierDataset(Dataset):
    """加载 brick COCO 数据，Brick 标为 background(4)，其它保持。"""

    def __init__(self, data_dir, img_size=224):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        ann_path = self.data_dir / '_annotations.coco.json'
        with open(ann_path) as f:
            coco = json.load(f)

        self.img_info = {im['id']: im for im in coco['images']}
        self.anns_by_img = {}
        for ann in coco['annotations']:
            self.anns_by_img.setdefault(ann['image_id'], []).append(ann)
        self.img_ids = sorted(self.img_info.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        info = self.img_info[img_id]
        img_path = self.data_dir / info['file_name']

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        oh, ow = img.shape[:2]
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        anns = self.anns_by_img.get(img_id, [])
        boxes, labels = [], []
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id == 0:
                continue
            model_cls = BRICK_CAT_TO_MODEL.get(cat_id, 4)
            x, y, w, h = ann['bbox']
            x, y, w, h = float(x), float(y), float(w), float(h)
            x1 = x / ow
            y1 = y / oh
            x2 = (x + w) / ow
            y2 = (y + h) / oh
            boxes.append([x1, y1, x2, y2])
            labels.append(model_cls)

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.long)

        return img, {'boxes': boxes, 'labels': labels}


def mixed_collate_fn(batch):
    """同 collate_fn，stack images + list targets."""
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


def finetune_with_outlier(model, benchmark_ds, brick_ds, device,
                          support_pct=0.2, epochs=30, seed=42):
    n = len(benchmark_ds)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    n_support = int(n * support_pct)
    support_idx = indices[:n_support]

    support_known = Subset(benchmark_ds, support_idx)
    mixed_ds = ConcatDataset([support_known, brick_ds])
    loader = DataLoader(mixed_ds, batch_size=4, shuffle=True,
                        num_workers=0, collate_fn=mixed_collate_fn)

    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    criterion = SetCriterion(num_classes=4, matcher=matcher,
                             weight_dict={'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batch = 0
        for images, targets in loader:
            images = images.to(device)
            targets_dev = []
            for t in targets:
                td = {}
                for k, v in t.items():
                    td[k] = v.to(device) if isinstance(v, torch.Tensor) else v
                targets_dev.append(td)

            # Filter out background-only targets from brick:
            # Brick has label=4 (background). Hungarian matcher can match these.
            # The classification loss will push matched queries toward background.
            outputs = model(images)
            loss_dict = criterion(outputs, targets_dev)
            loss = sum(loss_dict[k] * criterion.weight_dict.get(k, 1.0)
                       for k in loss_dict if k in criterion.weight_dict)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/n_batch:.4f}")

    model.eval()
    print("  Fine-tune with outlier exposure done.")


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


def infer_and_visualize(model, img_dir, gt_by_file, device, out_dir,
                        unknown_threshold, num_objects=4):
    """
    num_objects: 每张图预期的物体数。先选已知类高分框，剩余配额从低分框中
    取最高 objectness (1-bg) 的标为 Unknown。
    """
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
        probs = Func.softmax(logits, dim=-1)
        bg = probs[:, -1]
        kp = probs[:, :-1]
        max_scores, max_labels = kp.max(dim=-1)
        objectness = 1.0 - bg  # higher = more likely an object

        # Phase 1: collect known detections (high confidence)
        known_boxes, known_labels, known_scores = [], [], []
        # Phase 2: collect candidate unknowns (low known-score but still "objecty")
        unk_candidates = []  # (objectness, q_idx)

        for q in range(logits.shape[0]):
            c = max_labels[q].item()
            sc = max_scores[q].item()
            obj = objectness[q].item()

            if sc >= unknown_threshold and sc >= SCORE_THRESHOLDS[c] and obj > 0.3:
                bxy = box_cxcywh_to_xyxy(boxes[q:q+1])[0]
                known_boxes.append(bxy)
                known_labels.append(c)
                known_scores.append(sc)
            elif obj > 0.15:
                unk_candidates.append((obj, q))

        # NMS on known
        if known_boxes:
            kb = torch.stack(known_boxes)
            kl = torch.tensor(known_labels, device=device)
            ks = torch.tensor(known_scores, device=device)
            kb, kl, ks = nms_per_class(kb, kl, ks, NMS_IOU)
        else:
            kb = torch.zeros((0, 4), device=device)
            kl = torch.zeros(0, dtype=torch.long, device=device)
            ks = torch.zeros(0, device=device)

        n_known = len(kl)
        n_unk_needed = max(0, num_objects - n_known)

        # Phase 2: fill remaining slots with unknown candidates
        all_boxes, all_labels, all_scores = list(kb), list(kl), list(ks)

        if n_unk_needed > 0 and unk_candidates:
            unk_candidates.sort(key=lambda x: -x[0])
            added = 0
            for obj_val, q in unk_candidates:
                if added >= n_unk_needed:
                    break
                bxy = box_cxcywh_to_xyxy(boxes[q:q+1])[0]
                # Check IoU with already selected boxes to avoid overlap
                skip = False
                for existing in all_boxes:
                    if isinstance(existing, torch.Tensor):
                        ex = existing
                    else:
                        continue
                    ix1 = max(bxy[0].item(), ex[0].item())
                    iy1 = max(bxy[1].item(), ex[1].item())
                    ix2 = min(bxy[2].item(), ex[2].item())
                    iy2 = min(bxy[3].item(), ex[3].item())
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2-ix1)*(iy2-iy1)
                        a1 = (bxy[2]-bxy[0]).item()*(bxy[3]-bxy[1]).item()
                        a2 = (ex[2]-ex[0]).item()*(ex[3]-ex[1]).item()
                        iou = inter/(a1+a2-inter+1e-6)
                        if iou > 0.3:
                            skip = True
                            break
                if skip:
                    continue
                all_boxes.append(bxy)
                all_labels.append(torch.tensor(4, device=device))
                all_scores.append(torch.tensor(obj_val, device=device))
                added += 1

        if all_boxes:
            all_boxes = torch.stack([b if isinstance(b, torch.Tensor) else b for b in all_boxes])
            all_labels = torch.stack([l if isinstance(l, torch.Tensor) else l for l in all_labels])
            all_scores = torch.stack([s if isinstance(s, torch.Tensor) else s for s in all_scores])
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
            x1, y1 = int(bx[0]*ow), int(bx[1]*oh)
            x2, y2 = int(bx[2]*ow), int(bx[3]*oh)
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
    parser.add_argument('--brick_dir', default='./unknown_material_samples_2/train')
    parser.add_argument('--brick', action='store_true', help='Infer on brick')
    parser.add_argument('--unknown', action='store_true', help='Infer on marble')
    parser.add_argument('--finetune_epochs', type=int, default=30)
    parser.add_argument('--unknown_threshold', type=float, default=0.45)
    parser.add_argument('--num_objects', type=int, default=4,
                        help='Expected objects per image; remaining slots filled as Unknown')
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
    print(f"[1/3] Loading model: {args.ckpt}")
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

    # Step 2: Fine-tune with outlier exposure
    print(f"[2/3] Fine-tuning with outlier exposure ({args.finetune_epochs} epochs)...")
    benchmark_ds = BenchmarkDataset(benchmark_root=args.benchmark, img_size=224)
    brick_ds = BrickOutlierDataset(args.brick_dir, img_size=224)
    print(f"  Benchmark: {len(benchmark_ds)} images, Brick outlier: {len(brick_ds)} images")
    finetune_with_outlier(model, benchmark_ds, brick_ds, device, epochs=args.finetune_epochs)

    # Step 3: Infer + visualize
    print(f"[3/3] Inference (unknown_threshold={args.unknown_threshold})...")
    gt_by_file = load_coco_gt(ann_path)
    print(f"  GT: {len(gt_by_file)} images, {sum(len(v) for v in gt_by_file.values())} boxes")
    infer_and_visualize(model, img_dir, gt_by_file, device, out_dir,
                        args.unknown_threshold, args.num_objects)


if __name__ == '__main__':
    main()
