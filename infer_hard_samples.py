#!/usr/bin/env python3
"""
对 hard_samples/test 做推理并可视化：绿色虚线框=GT，彩色实线框=预测。
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import cv2
from pathlib import Path
from material_detection_model import MaterialDetectionModel
from train_material_detection import box_cxcywh_to_xyxy
from torchvision import transforms

try:
    from torchvision.ops import nms as torch_nms
except ImportError:
    torch_nms = None

CLASS_NAMES = ['Concrete', 'Glass', 'Metal', 'Wood', 'Unknown']
COLORS_BGR = {
    0: (0, 0, 255),    # Concrete - Red
    1: (0, 200, 0),    # Glass - Green
    2: (255, 100, 0),  # Metal - Blue
    3: (0, 220, 220),  # Wood - Yellow
    4: (200, 0, 200),  # Unknown - Magenta
}
GT_COLOR = (0, 255, 0)

SCORE_THRESHOLDS = [0.25, 0.45, 0.4, 0.4]  # Concrete, Glass, Metal, Wood
NMS_IOU = 0.1
BG_THRESHOLD = 0.5
# Per-class cos sim threshold to own prototype: if below → Unknown
# Based on Eval_benchmark stats (mean - 2*std as safe margin)
# Concrete: mean=0.952, std=0.038, min=0.663 → threshold ~0.65
# Glass:    mean=0.871, std=0.062, min=0.540 → threshold ~0.50
# Metal:    mean=0.902, std=0.023, min=0.724 → threshold ~0.70
# Wood:     mean=0.866, std=0.076, min=0.680 → threshold ~0.65
PROTO_COS_THRESHOLDS = [0.65, 0.50, 0.70, 0.65]  # per-class thresholds


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'args' in ckpt:
        ma = ckpt['args']
        backbone = ma.get('backbone', 'vit_base_patch16_224')
        nq = ma.get('num_queries', 100)
        ndl = ma.get('num_decoder_layers', 6)
    else:
        backbone, nq, ndl = 'vit_base_patch16_224', 100, 6

    model = MaterialDetectionModel(
        backbone_name=backbone, img_size=224, num_classes=4,
        num_queries=nq, num_decoder_layers=ndl,
        use_multi_view=False, num_views=1, pretrained_backbone=False,
    ).to(device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


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
    for c in range(5):  # 0-3: known classes, 4: unknown
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


def load_prototypes(proto_path, device):
    data = torch.load(proto_path, map_location=device, weights_only=False)
    proto_list = []
    for cls_id in range(4):
        proto_list.append(data['prototypes'][cls_id])
    prototypes = torch.stack(proto_list).to(device)  # [4, d_model]
    return prototypes


def infer_and_visualize(model, img_dir, gt_by_file, device, out_dir, prototypes):
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
        logits = out['pred_logits'][0]          # [Q, 5]
        boxes = out['pred_boxes'][0]            # [Q, 4]
        feats = out['decoder_features'][0]      # [Q, d_model]

        probs = F.softmax(logits, dim=-1)
        bg_scores = probs[:, -1]
        known_probs = probs[:, :-1]
        max_known_scores, max_known_labels = known_probs.max(dim=-1)

        # Cosine similarity to prototypes: [Q, 4]
        feats_norm = F.normalize(feats, dim=-1)
        cos_sim = feats_norm @ prototypes.T     # [Q, 4]

        # Step 1: find object queries (not background)
        is_object = bg_scores < BG_THRESHOLD

        # Step 2: among objects, check if known class score is high enough
        known_mask = torch.zeros_like(max_known_scores, dtype=torch.bool)
        for c in range(4):
            known_mask |= (max_known_labels == c) & (max_known_scores > SCORE_THRESHOLDS[c])

        # Step 3: for each known detection, check cos sim to its PREDICTED class prototype
        # If the feature is far from the predicted class prototype → reclassify as Unknown
        proto_unknown = torch.zeros_like(known_mask)
        for c in range(4):
            cls_mask = known_mask & (max_known_labels == c)
            if cls_mask.sum() == 0:
                continue
            cos_to_cls = cos_sim[:, c]
            proto_unknown |= cls_mask & (cos_to_cls < PROTO_COS_THRESHOLDS[c])

        truly_known = known_mask & ~proto_unknown
        reclassified_unknown = proto_unknown

        # --- Known predictions ---
        boxes_known = boxes[truly_known]
        labels_known = max_known_labels[truly_known]
        scores_known = max_known_scores[truly_known]

        # --- Unknown predictions ---
        unk_mask = reclassified_unknown
        boxes_unknown = boxes[unk_mask]
        labels_unknown = torch.full((unk_mask.sum().item(),), 4, dtype=torch.long, device=device)
        scores_unknown = (1.0 - bg_scores[unk_mask])

        # Merge
        if len(boxes_unknown) > 0:
            boxes_v = torch.cat([boxes_known, boxes_unknown], dim=0)
            labels_v = torch.cat([labels_known, labels_unknown], dim=0)
            scores_v = torch.cat([scores_known, scores_unknown], dim=0)
        else:
            boxes_v, labels_v, scores_v = boxes_known, labels_known, scores_known

        boxes_xyxy = box_cxcywh_to_xyxy(boxes_v)
        boxes_xyxy, labels_v, scores_v = nms_per_class(boxes_xyxy, labels_v, scores_v, NMS_IOU)

        # draw GT
        gts = gt_by_file.get(fname, [])
        for g in gts:
            x1, y1, x2, y2 = [int(v) for v in g['bbox_xyxy']]
            cv2.rectangle(vis, (x1, y1), (x2, y2), GT_COLOR, 2, cv2.LINE_AA)
            label_gt = f"GT:{g['class_name']}"
            (tw, th), _ = cv2.getTextSize(label_gt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), GT_COLOR, -1)
            cv2.putText(vis, label_gt, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # draw predictions
        for i in range(len(labels_v)):
            bx = boxes_xyxy[i].cpu().numpy()
            x1 = int(bx[0] * ow)
            y1 = int(bx[1] * oh)
            x2 = int(bx[2] * ow)
            y2 = int(bx[3] * oh)
            cls = labels_v[i].item()
            sc = scores_v[i].item()
            color = COLORS_BGR.get(cls, (255, 255, 255))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
            label_p = f"{CLASS_NAMES[cls]}:{sc:.2f}"
            (tw, th), _ = cv2.getTextSize(label_p, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(vis, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(vis, label_p, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        n_known = (labels_v < 4).sum().item()
        n_unknown = (labels_v == 4).sum().item()
        out_path = out_dir / f"vis_{fname}"
        cv2.imwrite(str(out_path), vis)
        print(f"  {fname}: GT={len(gts)}, Known={n_known}, Unknown={n_unknown}")

    print(f"\n✅ 可视化结果已保存到: {out_dir}/")


def main():
    import sys
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = './checkpoints_material_detection_new_data/checkpoint_best.pth'

    if '--unknown' in sys.argv:
        img_dir = './unknown_material_samples/train'
        ann_path = './unknown_material_samples/train/_annotations.coco.json'
        out_dir = './unknown_material_visualizations'
    else:
        img_dir = './hard_samples/test'
        ann_path = './hard_samples/test/_annotations.coco.json'
        out_dir = './hard_samples_visualizations'

    proto_path = './class_prototypes.pth'

    print("📂 加载模型...")
    model = load_model(ckpt, device)
    print("📂 加载原型...")
    prototypes = load_prototypes(proto_path, device)
    print("📂 加载 GT 标注...")
    gt_by_file = load_coco_gt(ann_path)
    print(f"📊 共 {len(gt_by_file)} 张图, {sum(len(v) for v in gt_by_file.values())} 个 GT 框")
    print(f"📊 原型余弦阈值(per-class): {PROTO_COS_THRESHOLDS}")
    print("\n🔍 推理 + 可视化...")
    infer_and_visualize(model, img_dir, gt_by_file, device, out_dir, prototypes)


if __name__ == '__main__':
    main()
