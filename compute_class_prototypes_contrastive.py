#!/usr/bin/env python3
"""
用对比学习 checkpoint 在 Eval_benchmark 上收集 proj_features（128-d）的类别原型。
用于 unknown 检测时与已知类原型比较。
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from material_detection_model import MaterialDetectionModel
from evaluate_benchmark import BenchmarkDataset
from material_dataset import collate_fn
from train_material_detection import HungarianMatcher

CLASS_NAMES = ['Concrete', 'Glass', 'Metal', 'Wood']


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='../checkpoints_contrastive/checkpoint_best.pth', help='Contrastive checkpoint')
    p.add_argument('--benchmark', default='./Eval_benchmark', help='Eval_benchmark root')
    p.add_argument('--out', default='./class_prototypes_contrastive.pth', help='Output prototype file')
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

    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    dataset = BenchmarkDataset(benchmark_root=args.benchmark, img_size=224, num_views=1, use_multi_view=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=False)

    class_features = defaultdict(list)  # class_id -> list of [128]

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Collecting proj_features"):
            images = images.to(device)
            targets_dev = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            outputs = model(images)
            indices = matcher(outputs, targets_dev)
            feats = outputs['proj_features']  # [B, Q, 128] already L2-normalized

            for b, (pred_idx, gt_idx) in enumerate(indices):
                gt_labels = targets_dev[b]['labels']
                for pi, gi in zip(pred_idx, gt_idx):
                    cls_id = gt_labels[gi].item()
                    feat = feats[b, pi]
                    class_features[cls_id].append(feat.cpu())

    prototypes = {}
    class_cos_stats = {}
    for cls_id in range(4):
        feats = torch.stack(class_features[cls_id])
        proto = feats.mean(dim=0)
        proto = F.normalize(proto, dim=0)
        prototypes[cls_id] = proto
        print(f"  {CLASS_NAMES[cls_id]}: {len(class_features[cls_id])} samples, dim={feats.shape[-1]}")

        feats_norm = F.normalize(feats, dim=1)
        cos_sims = feats_norm @ proto
        class_cos_stats[cls_id] = {
            'mean_cos': cos_sims.mean().item(),
            'std_cos': cos_sims.std().item(),
            'min_cos': cos_sims.min().item(),
        }
        print(f"    cos to prototype: mean={cos_sims.mean():.4f}, std={cos_sims.std():.4f}, min={cos_sims.min():.4f}")

    torch.save({'prototypes': prototypes, 'cos_stats': class_cos_stats}, args.out)
    print(f"\n✅ Prototypes (contrastive proj_features) saved to {args.out}")


if __name__ == '__main__':
    main()
