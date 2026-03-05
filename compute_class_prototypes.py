#!/usr/bin/env python3
"""
从 Eval_benchmark 收集每个已知类的 decoder 特征原型。
对每张图做推理，用 Hungarian 匹配将 query 与 GT 对齐，
收集匹配正确的 query 的 decoder feature，按类别取均值 → 原型。
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
from train_material_detection import box_cxcywh_to_xyxy, HungarianMatcher

CLASS_NAMES = ['Concrete', 'Glass', 'Metal', 'Wood']


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = './checkpoints_material_detection_new_data/checkpoint_best.pth'
    benchmark_root = './Eval_benchmark'
    out_path = './class_prototypes.pth'

    # Load model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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

    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)

    dataset = BenchmarkDataset(benchmark_root=benchmark_root, img_size=224, num_views=1, use_multi_view=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=False)

    class_features = defaultdict(list)  # class_id -> list of [d_model] tensors

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Collecting features"):
            images = images.to(device)
            targets_dev = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            outputs = model(images)

            indices = matcher(outputs, targets_dev)
            feats = outputs['decoder_features']  # [B, Q, d_model]

            for b, (pred_idx, gt_idx) in enumerate(indices):
                gt_labels = targets_dev[b]['labels']
                for pi, gi in zip(pred_idx, gt_idx):
                    cls_id = gt_labels[gi].item()
                    feat = feats[b, pi]  # [d_model]
                    class_features[cls_id].append(feat.cpu())

    # Compute prototypes (L2-normalized mean)
    prototypes = {}
    for cls_id in range(4):
        feats = torch.stack(class_features[cls_id])  # [N, d_model]
        proto = feats.mean(dim=0)
        proto = F.normalize(proto, dim=0)
        prototypes[cls_id] = proto
        print(f"  {CLASS_NAMES[cls_id]}: {len(class_features[cls_id])} samples, prototype norm={proto.norm():.4f}")

    # Also save per-class std for reference
    class_cos_stats = {}
    for cls_id in range(4):
        feats = torch.stack(class_features[cls_id])
        feats_norm = F.normalize(feats, dim=1)
        cos_sims = feats_norm @ prototypes[cls_id]
        class_cos_stats[cls_id] = {
            'mean_cos': cos_sims.mean().item(),
            'std_cos': cos_sims.std().item(),
            'min_cos': cos_sims.min().item(),
        }
        print(f"  {CLASS_NAMES[cls_id]}: cos sim to prototype: mean={cos_sims.mean():.4f}, std={cos_sims.std():.4f}, min={cos_sims.min():.4f}")

    torch.save({
        'prototypes': prototypes,
        'cos_stats': class_cos_stats,
    }, out_path)
    print(f"\n✅ Prototypes saved to {out_path}")


if __name__ == '__main__':
    main()
