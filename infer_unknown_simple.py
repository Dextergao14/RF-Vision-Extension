#!/usr/bin/env python3
"""
简单的 Unknown 检测方案（无需原型）：
1. 先用 20% Eval_benchmark fine-tune 原始模型
2. 用 fine-tuned 模型推理：softmax max score < threshold → Unknown
3. 在 brick/marble 上可视化
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
import cv2
from pathlib import Path
from torchvision import transforms

from material_detection_model import MaterialDetectionModel
from evaluate_benchmark import BenchmarkDataset
from material_dataset import collate_fn
from train_material_detection import (
    box_cxcywh_to_xyxy, HungarianMatcher, SetCriterion
)
from evaluate_and_visualize_20pct import (
    apply_nms_per_class, finetune_on_support_set
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--brick', action='store_true')
    parser.add_argument('--unknown', action='store_true', help='marble')
    parser.add_argument('--ckpt', default='../checkpoints_material_detection_new_data/checkpoint_best.pth')
    parser.add_argument('--benchmark', default='./Eval_benchmark')
    parser.add_argument('--support_pct', type=float, default=0.2)
    parser.add_argument('--score_threshold', type=float, default=0.5,
                        help='Max known score below this → Unknown')
    parser.add_argument('--nms_iou', type=float, default=0.1)
    parser.add_argument('--finetune_epochs', type=int, default=30)
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

    # Step 1: Load + fine-tune
    print(f"Loading model: {args.ckpt}")
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

    print(f"Fine-tuning on {args.support_pct*100:.0f}% of Eval_benchmark for {args.finetune_epochs} epochs...")
    dataset = BenchmarkDataset(benchmark_root=args.benchmark, img_size=224)
    from torch.utils.data import DataLoader, Subset
    import random
    n = len(dataset)
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)
    n_support = int(n * args.support_pct)
    support_indices = indices[:n_support]
    support_subset = Subset(dataset, support_indices)
    support_loader = DataLoader(support_subset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)

    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    criterion = SetCriterion(num_classes=4, matcher=matcher, weight_dict={
        'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0
    }).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    model.train()
    for epoch in range(args.finetune_epochs):
        total_loss = 0
        for images, targets in support_loader:
            images = images.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict[k] * criterion.weight_dict.get(k, 1.0) for k in loss_dict if k in criterion.weight_dict)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/len(support_loader):.4f}")
    model.eval()
    print("Fine-tuning done.")

    # Step 2: Infer on target images
    gt_by_file = load_coco_gt(ann_path)
    print(f"GT: {len(gt_by_file)} images, {sum(len(v) for v in gt_by_file.values())} boxes")
    print(f"Unknown threshold: max_known_score < {args.score_threshold}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    SCORE_THRESHOLDS = [0.25, 0.45, 0.4, 0.4]
    BG_THRESHOLD = 0.5

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
        probs = F.softmax(logits, dim=-1)
        bg = probs[:, -1]
        kp = probs[:, :-1]
        max_scores, max_labels = kp.max(dim=-1)

        # Detect objects
        all_boxes, all_labels, all_scores = [], [], []
        for q in range(logits.shape[0]):
            if bg[q] >= BG_THRESHOLD:
                continue
            c = max_labels[q].item()
            sc = max_scores[q].item()
            if sc < min(SCORE_THRESHOLDS):
                continue
            bxy = box_cxcywh_to_xyxy(boxes[q:q+1])[0]
            if sc >= args.score_threshold:
                all_boxes.append(bxy)
                all_labels.append(c)
                all_scores.append(sc)
            else:
                all_boxes.append(bxy)
                all_labels.append(4)  # Unknown
                all_scores.append(sc)

        if all_boxes:
            all_boxes = torch.stack(all_boxes)
            all_labels = torch.tensor(all_labels, device=device)
            all_scores = torch.tensor(all_scores, device=device)
            all_boxes, all_labels, all_scores = nms_per_class(all_boxes, all_labels, all_scores, args.nms_iou)
        else:
            all_boxes = torch.zeros((0, 4), device=device)
            all_labels = torch.zeros(0, dtype=torch.long, device=device)
            all_scores = torch.zeros(0, device=device)

        # Draw GT
        gts = gt_by_file.get(fname, [])
        for g in gts:
            x1, y1, x2, y2 = [int(v) for v in g['bbox_xyxy']]
            cv2.rectangle(vis, (x1, y1), (x2, y2), GT_COLOR, 2, cv2.LINE_AA)
            label_gt = f"GT:{g['class_name']}"
            (tw, th), _ = cv2.getTextSize(label_gt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), GT_COLOR, -1)
            cv2.putText(vis, label_gt, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw predictions
        for i in range(len(all_labels)):
            bx = all_boxes[i].cpu().numpy()
            x1, y1, x2, y2 = int(bx[0]*ow), int(bx[1]*oh), int(bx[2]*ow), int(bx[3]*oh)
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


if __name__ == '__main__':
    main()
