#!/usr/bin/env python3
"""
分析 known vs brick 在 cos sim margin（top1 - top2）和 spread 上的差异。
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from material_detection_model import MaterialDetectionModel
from evaluate_benchmark import BenchmarkDataset
from material_dataset import collate_fn
from train_material_detection import HungarianMatcher
from torchvision import transforms

CLASS_NAMES = ['Concrete', 'Glass', 'Metal', 'Wood']
SCORE_THRESHOLDS = [0.25, 0.45, 0.4, 0.4]
BG_THRESHOLD = 0.5


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = '/home/user/wentao/checkpoints_contrastive/checkpoint_best.pth'
    proto_path = './class_prototypes_contrastive.pth'

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

    proto_data = torch.load(proto_path, map_location=device, weights_only=False)
    prototypes = torch.stack([proto_data['prototypes'][i] for i in range(4)]).to(device)

    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)

    # --- Known classes: from Eval_benchmark ---
    dataset = BenchmarkDataset(benchmark_root='./Eval_benchmark', img_size=224, num_views=1, use_multi_view=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=False)

    known_margins = []
    known_spreads = []
    known_cos_to_pred = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Known classes"):
            images = images.to(device)
            targets_dev = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            outputs = model(images)
            indices = matcher(outputs, targets_dev)
            proj = outputs['proj_features']

            for b, (pred_idx, gt_idx) in enumerate(indices):
                for pi, gi in zip(pred_idx, gt_idx):
                    feat = proj[b, pi]  # [128]
                    cos = (feat @ prototypes.T).cpu().numpy()  # [4]
                    sorted_cos = np.sort(cos)[::-1]
                    margin = sorted_cos[0] - sorted_cos[1]
                    spread = sorted_cos[0] - sorted_cos[-1]
                    known_margins.append(margin)
                    known_spreads.append(spread)
                    known_cos_to_pred.append(sorted_cos[0])

    known_margins = np.array(known_margins)
    known_spreads = np.array(known_spreads)
    known_cos_to_pred = np.array(known_cos_to_pred)

    # --- Brick ---
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    brick_margins = []
    brick_spreads = []
    brick_cos_to_pred = []

    img_files = sorted(Path('./unknown_material_samples_2/train').glob('*.png'))
    with torch.no_grad():
        for img_path in img_files:
            orig = cv2.imread(str(img_path))
            if orig is None:
                continue
            img_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            inp = tfm(img_rgb).unsqueeze(0).to(device)
            out = model(inp)
            logits = out['pred_logits'][0]
            proj = out['proj_features'][0]
            probs = F.softmax(logits, dim=-1)
            bg_scores = probs[:, -1]
            known_probs = probs[:, :-1]
            max_scores, max_labels = known_probs.max(dim=-1)

            for q in range(logits.shape[0]):
                if bg_scores[q] >= BG_THRESHOLD:
                    continue
                cls = max_labels[q].item()
                if max_scores[q].item() < SCORE_THRESHOLDS[cls]:
                    continue
                feat = proj[q]
                cos = (feat @ prototypes.T).cpu().numpy()
                sorted_cos = np.sort(cos)[::-1]
                margin = sorted_cos[0] - sorted_cos[1]
                spread = sorted_cos[0] - sorted_cos[-1]
                brick_margins.append(margin)
                brick_spreads.append(spread)
                brick_cos_to_pred.append(sorted_cos[0])

    brick_margins = np.array(brick_margins)
    brick_spreads = np.array(brick_spreads)
    brick_cos_to_pred = np.array(brick_cos_to_pred)

    print("=" * 70)
    print("MARGIN (top1 cos - top2 cos): 越大说明越确信属于某个已知类")
    print(f"  Known:  mean={known_margins.mean():.4f}, std={known_margins.std():.4f}, "
          f"min={known_margins.min():.4f}, p5={np.percentile(known_margins, 5):.4f}")
    print(f"  Brick:  mean={brick_margins.mean():.4f}, std={brick_margins.std():.4f}, "
          f"max={brick_margins.max():.4f}, p95={np.percentile(brick_margins, 95):.4f}")
    gap_margin = np.percentile(known_margins, 5) - np.percentile(brick_margins, 95)
    print(f"  Gap (known_p5 - brick_p95): {gap_margin:.4f}")
    if gap_margin > 0:
        print(f"  ✅ Margin 可分！ 建议阈值范围: [{np.percentile(brick_margins, 95):.4f}, {np.percentile(known_margins, 5):.4f}]")
    else:
        print(f"  ❌ Margin 有重叠")

    print()
    print("SPREAD（top1 cos - bottom cos）: 越大说明特征偏向某个类")
    print(f"  Known:  mean={known_spreads.mean():.4f}, std={known_spreads.std():.4f}, "
          f"min={known_spreads.min():.4f}, p5={np.percentile(known_spreads, 5):.4f}")
    print(f"  Brick:  mean={brick_spreads.mean():.4f}, std={brick_spreads.std():.4f}, "
          f"max={brick_spreads.max():.4f}, p95={np.percentile(brick_spreads, 95):.4f}")
    gap_spread = np.percentile(known_spreads, 5) - np.percentile(brick_spreads, 95)
    print(f"  Gap (known_p5 - brick_p95): {gap_spread:.4f}")
    if gap_spread > 0:
        print(f"  ✅ Spread 可分！ 建议阈值范围: [{np.percentile(brick_spreads, 95):.4f}, {np.percentile(known_spreads, 5):.4f}]")

    print()
    print("COS TO PREDICTED CLASS（最高 cos）")
    print(f"  Known:  mean={known_cos_to_pred.mean():.4f}, std={known_cos_to_pred.std():.4f}, "
          f"min={known_cos_to_pred.min():.4f}, p5={np.percentile(known_cos_to_pred, 5):.4f}")
    print(f"  Brick:  mean={brick_cos_to_pred.mean():.4f}, std={brick_cos_to_pred.std():.4f}, "
          f"max={brick_cos_to_pred.max():.4f}, p95={np.percentile(brick_cos_to_pred, 95):.4f}")

    print()
    print("=" * 70)
    print("建议策略: margin + cos_to_pred 联合判定")
    print("  如果 margin < M_thresh 且 cos_to_pred < C_thresh → Unknown")


if __name__ == '__main__':
    main()
