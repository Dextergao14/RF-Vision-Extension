#!/usr/bin/env python3
"""
从 Eval_benchmark 的推理检测结果中收集 proj_features 原型。
使用和推理相同的管线（softmax + score threshold），而非 Hungarian matching。
通过 IoU 与 GT 匹配来确定正确检测，按 GT 类别分组计算原型。
"""
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from material_detection_model import MaterialDetectionModel
from evaluate_benchmark import BenchmarkDataset
from material_dataset import collate_fn
from train_material_detection import box_cxcywh_to_xyxy

CLASS_NAMES = ['Concrete', 'Glass', 'Metal', 'Wood']
SCORE_THRESHOLDS = [0.25, 0.45, 0.4, 0.4]
BG_THRESHOLD = 0.5


def compute_iou(box1, box2):
    """box format: [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='/home/user/wentao/checkpoints_contrastive/checkpoint_best.pth')
    p.add_argument('--benchmark', default='./Eval_benchmark')
    p.add_argument('--out', default='./class_prototypes_detection.pth')
    p.add_argument('--feat_key', default='proj_features', choices=['proj_features', 'decoder_features'])
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ma = ckpt.get('args', {})
    model = MaterialDetectionModel(
        backbone_name=ma.get('backbone', 'vit_base_patch16_224'),
        img_size=224, num_classes=4,
        num_queries=ma.get('num_queries', 100),
        num_decoder_layers=ma.get('num_decoder_layers', 6),
        use_multi_view=False, num_views=1, pretrained_backbone=False,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    dataset = BenchmarkDataset(benchmark_root=args.benchmark, img_size=224, num_views=1, use_multi_view=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=False)

    class_features = defaultdict(list)
    total_matched = 0
    total_detected = 0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Collecting detection features"):
            images = images.to(device)
            out = model(images)
            B = images.shape[0]

            for b in range(B):
                logits = out['pred_logits'][b]
                boxes_cxcywh = out['pred_boxes'][b]
                feats = out[args.feat_key][b]

                probs = F.softmax(logits, dim=-1)
                bg = probs[:, -1]
                kp = probs[:, :-1]
                max_scores, max_labels = kp.max(dim=-1)

                gt_boxes = targets[b]['boxes']   # [N, 4] xyxy normalized
                gt_labels = targets[b]['labels']  # [N]

                if len(gt_boxes) == 0:
                    continue

                det_indices = []
                for q in range(logits.shape[0]):
                    if bg[q] >= BG_THRESHOLD:
                        continue
                    c = max_labels[q].item()
                    if max_scores[q].item() < SCORE_THRESHOLDS[c]:
                        continue
                    det_indices.append(q)

                if not det_indices:
                    continue

                total_detected += len(det_indices)
                pred_boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh[det_indices])

                if total_detected <= 10:
                    print(f"  [DEBUG] det={len(det_indices)}, gt={len(gt_boxes_np) if 'gt_boxes_np' in dir() else len(gt_boxes)}")
                    print(f"    pred[0]: {pred_boxes_xyxy[0].cpu().numpy()}")
                    print(f"    gt[0]:   {gt_boxes[0].cpu().numpy()}")

                gt_boxes_np = gt_boxes.cpu().numpy()
                gt_labels_np = gt_labels.cpu().numpy()
                pred_boxes_np = pred_boxes_xyxy.cpu().numpy()

                gt_matched = set()
                for di, qi in enumerate(det_indices):
                    pred_box = pred_boxes_np[di]
                    best_iou = 0
                    best_gi = -1
                    for gi in range(len(gt_boxes_np)):
                        if gi in gt_matched:
                            continue
                        iou = compute_iou(pred_box, gt_boxes_np[gi])
                        if iou > best_iou:
                            best_iou = iou
                            best_gi = gi
                    if best_iou > 0.3 and best_gi >= 0:
                        gt_matched.add(best_gi)
                        cls_id = int(gt_labels_np[best_gi])
                        feat = feats[qi]
                        class_features[cls_id].append(feat.cpu())
                        total_matched += 1
                    elif best_iou <= 0.3 and di == 0:
                        pass  # debug: skip first unmatched silently

    print(f"\nTotal detected: {total_detected}, matched to GT: {total_matched}")

    prototypes = {}
    class_cos_stats = {}
    for cls_id in range(4):
        if not class_features[cls_id]:
            print(f"  {CLASS_NAMES[cls_id]}: no samples!")
            continue
        feats = torch.stack(class_features[cls_id])
        proto = feats.mean(dim=0)
        proto = F.normalize(proto, dim=0)
        prototypes[cls_id] = proto

        feats_norm = F.normalize(feats, dim=1)
        cos_sims = feats_norm @ proto
        class_cos_stats[cls_id] = {
            'mean_cos': cos_sims.mean().item(),
            'std_cos': cos_sims.std().item(),
            'min_cos': cos_sims.min().item(),
        }
        print(f"  {CLASS_NAMES[cls_id]:10s}: {len(class_features[cls_id])} samples, "
              f"cos mean={cos_sims.mean():.4f}, std={cos_sims.std():.4f}, min={cos_sims.min():.4f}")

    torch.save({'prototypes': prototypes, 'cos_stats': class_cos_stats}, args.out)
    print(f"\nPrototypes saved to {args.out}")


if __name__ == '__main__':
    main()
