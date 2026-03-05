#!/usr/bin/env python3
"""
计算 per-query-index 的类别特征统计（均值+协方差的逆），用于 Mahalanobis distance。
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
from train_material_detection import HungarianMatcher

CLASS_NAMES = ['Concrete', 'Glass', 'Metal', 'Wood']


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = './checkpoints_material_detection_new_data/checkpoint_best.pth'
    benchmark_root = './Eval_benchmark'

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
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # Collect features keyed by (query_index, class_id)
    qc_features = defaultdict(list)  # (query_idx, class_id) -> list of features

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Collecting per-query features"):
            images = images.to(device)
            targets_dev = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            outputs = model(images)
            indices = matcher(outputs, targets_dev)
            feats = outputs['decoder_features']

            for b, (pred_idx, gt_idx) in enumerate(indices):
                gt_labels = targets_dev[b]['labels']
                for pi, gi in zip(pred_idx, gt_idx):
                    qi = pi.item()
                    cls_id = gt_labels[gi].item()
                    qc_features[(qi, cls_id)].append(feats[b, qi].cpu())

    # For each (query_idx, class_id) with enough samples, compute mean and covariance
    stats = {}
    for (qi, cls_id), feat_list in sorted(qc_features.items()):
        if len(feat_list) < 10:
            continue
        feats = torch.stack(feat_list)  # [N, 768]
        mean = feats.mean(dim=0)
        diff = feats - mean
        cov = (diff.T @ diff) / (len(feat_list) - 1)
        # Regularize covariance
        cov += 0.01 * torch.eye(cov.shape[0])
        stats[(qi, cls_id)] = {
            'mean': mean,
            'cov_inv': torch.linalg.inv(cov),
            'n_samples': len(feat_list),
        }
        print(f"  Q{qi:02d}-{CLASS_NAMES[cls_id]}: {len(feat_list)} samples")

    torch.save(stats, './query_class_stats.pth')
    print(f"\n✅ Saved {len(stats)} (query, class) stats")


if __name__ == '__main__':
    main()
