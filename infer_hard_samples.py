#!/usr/bin/env python3
"""
推理 + 可视化：绿色框=GT，彩色框=已知类预测，紫色框=Unknown。
Unknown 判定：softmax 仅判断是否为物体，类别由 feature 空间最近原型决定。
若 max cos < threshold → Unknown。
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

SCORE_THRESHOLDS = [0.25, 0.45, 0.4, 0.4]
NMS_IOU = 0.1
BG_THRESHOLD = 0.5


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


def load_prototypes(proto_path, device):
    data = torch.load(proto_path, map_location=device, weights_only=False)
    proto_list = []
    for cls_id in range(4):
        proto_list.append(data['prototypes'][cls_id])
    prototypes = torch.stack(proto_list).to(device)
    cos_stats = data.get('cos_stats', None)
    return prototypes, cos_stats


def infer_and_visualize(model, img_dir, gt_by_file, device, out_dir,
                        prototypes, feat_key, per_class_thresholds):
    """per_class_thresholds: [4] tensor, 每个类的 cos 下限 (min_cos - margin)。"""
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
        feats = out[feat_key][0]

        probs = F.softmax(logits, dim=-1)
        bg_scores = probs[:, -1]
        known_probs = probs[:, :-1]
        max_known_scores, max_known_labels = known_probs.max(dim=-1)

        feats_norm = F.normalize(feats, dim=-1)
        cos_sim = feats_norm @ prototypes.T
        max_cos, nearest_cls = cos_sim.max(dim=-1)

        # Step 1: object detection via softmax
        is_object = bg_scores < BG_THRESHOLD
        passes_score = torch.zeros_like(bg_scores, dtype=torch.bool)
        for c in range(4):
            passes_score |= (max_known_labels == c) & (max_known_scores > SCORE_THRESHOLDS[c])
        detected = is_object & passes_score

        # Step 2: per-class Unknown check — compare max cos to nearest class's threshold
        per_query_threshold = per_class_thresholds[nearest_cls]  # [Q]
        is_known = detected & (max_cos >= per_query_threshold)
        is_unknown = detected & (max_cos < per_query_threshold)

        # Known: class assigned by nearest prototype, score = max cos
        boxes_known = boxes[is_known]
        labels_known = nearest_cls[is_known]
        scores_known = max_cos[is_known]

        # Unknown
        boxes_unknown = boxes[is_unknown]
        labels_unknown = torch.full((is_unknown.sum().item(),), 4, dtype=torch.long, device=device)
        scores_unknown = max_cos[is_unknown]

        if len(boxes_unknown) > 0:
            boxes_v = torch.cat([boxes_known, boxes_unknown], dim=0)
            labels_v = torch.cat([labels_known, labels_unknown], dim=0)
            scores_v = torch.cat([scores_known, scores_unknown], dim=0)
        else:
            boxes_v, labels_v, scores_v = boxes_known, labels_known, scores_known

        boxes_xyxy = box_cxcywh_to_xyxy(boxes_v)
        boxes_xyxy, labels_v, scores_v = nms_per_class(boxes_xyxy, labels_v, scores_v, NMS_IOU)

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

    print(f"\n  可视化已保存到: {out_dir}/")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--contrastive', action='store_true')
    parser.add_argument('--unknown', action='store_true', help='marble samples')
    parser.add_argument('--brick', action='store_true', help='brick samples')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--proto', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override unknown threshold (max cos to any proto)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.contrastive:
        ckpt = args.ckpt or '../checkpoints_contrastive/checkpoint_best.pth'
        proto_path = args.proto or './class_prototypes_contrastive.pth'
        feat_key = 'proj_features'
    else:
        ckpt = args.ckpt or './checkpoints_material_detection_new_data/checkpoint_best.pth'
        proto_path = args.proto or '../class_prototypes.pth'
        feat_key = 'decoder_features'

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

    print(f"Loading model: {ckpt}")
    model = load_model(ckpt, device)
    print(f"Loading prototypes: {proto_path}")
    prototypes, cos_stats = load_prototypes(proto_path, device)

    margin = 0.02
    if cos_stats:
        per_class_thresholds = []
        for c in range(4):
            s = cos_stats[c]
            thr = s['min_cos'] - margin
            per_class_thresholds.append(thr)
            print(f"  {CLASS_NAMES[c]:10s}: mean={s['mean_cos']:.4f}, std={s['std_cos']:.4f}, "
                  f"min={s['min_cos']:.4f} -> threshold={thr:.4f}")
    else:
        per_class_thresholds = [0.65, 0.50, 0.70, 0.65]
        print(f"  Using default thresholds: {per_class_thresholds}")

    if args.threshold is not None:
        per_class_thresholds = [args.threshold] * 4
        print(f"  Override: all thresholds = {args.threshold:.4f}")

    per_class_thresholds = torch.tensor(per_class_thresholds, device=device)

    gt_by_file = load_coco_gt(ann_path)
    print(f"GT: {len(gt_by_file)} images, {sum(len(v) for v in gt_by_file.values())} boxes")

    infer_and_visualize(model, img_dir, gt_by_file, device, out_dir,
                        prototypes, feat_key, per_class_thresholds)


if __name__ == '__main__':
    main()
