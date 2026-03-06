#!/usr/bin/env python3
"""快速检查对比学习模型在 Eval_benchmark 上的 cos to predicted class 分布"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import DataLoader
from collections import defaultdict
from material_detection_model import MaterialDetectionModel
from evaluate_benchmark import BenchmarkDataset
from material_dataset import collate_fn

CLASS_NAMES = ['Concrete', 'Glass', 'Metal', 'Wood']
SCORE_THRESHOLDS = [0.25, 0.45, 0.4, 0.4]
BG_THRESHOLD = 0.5


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for label, ckpt_path, feat_key in [
        ("Original", '/home/user/wentao/checkpoints_material_detection_new_data/checkpoint_best.pth', 'decoder_features'),
        ("Contrastive", '/home/user/wentao/checkpoints_contrastive/checkpoint_best.pth', 'proj_features'),
    ]:
        print(f"\n{'='*60}")
        print(f"Model: {label}")
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

        # Load corresponding prototypes
        if label == "Contrastive":
            proto_data = torch.load('./class_prototypes_contrastive.pth', map_location=device, weights_only=False)
        else:
            proto_data = torch.load('/home/user/wentao/class_prototypes.pth', map_location=device, weights_only=False)
        prototypes = torch.stack([proto_data['prototypes'][i] for i in range(4)]).to(device)

        dataset = BenchmarkDataset(benchmark_root='./Eval_benchmark', img_size=224, num_views=1, use_multi_view=False)
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=False)

        total_known = 0
        total_unknown = 0
        total_detected = 0
        cos_to_pred_list = []
        thresholds = [proto_data['cos_stats'][c]['min_cos'] - 0.02 for c in range(4)]

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                out = model(images)
                B = images.shape[0]
                for b in range(B):
                    logits = out['pred_logits'][b]
                    feats = out[feat_key][b]
                    probs = F.softmax(logits, dim=-1)
                    bg = probs[:, -1]
                    kp = probs[:, :-1]
                    ms, ml = kp.max(dim=-1)
                    feats_n = F.normalize(feats, dim=-1)
                    cos = feats_n @ prototypes.T
                    is_obj = bg < BG_THRESHOLD
                    for q in range(logits.shape[0]):
                        if not is_obj[q]:
                            continue
                        c = ml[q].item()
                        if ms[q].item() < SCORE_THRESHOLDS[c]:
                            continue
                        total_detected += 1
                        cos_val = cos[q, c].item()
                        cos_to_pred_list.append(cos_val)
                        if cos_val < thresholds[c]:
                            total_unknown += 1
                        else:
                            total_known += 1

        arr = np.array(cos_to_pred_list)
        print(f"  Thresholds (min_cos-0.02): {[f'{t:.4f}' for t in thresholds]}")
        print(f"  Total detected: {total_detected}")
        print(f"  Known: {total_known}, Unknown (false): {total_unknown}")
        print(f"  False Unknown rate: {total_unknown/max(total_detected,1)*100:.1f}%")
        print(f"  Cos to predicted: mean={arr.mean():.4f}, std={arr.std():.4f}, "
              f"min={arr.min():.4f}, p5={np.percentile(arr,5):.4f}")


if __name__ == '__main__':
    main()
