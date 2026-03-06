#!/usr/bin/env python3
"""在 Eval_benchmark 上统计 per-class threshold 方案的 false Unknown rate"""
import torch, torch.nn.functional as F, numpy as np
from torch.utils.data import DataLoader
from material_detection_model import MaterialDetectionModel
from evaluate_benchmark import BenchmarkDataset
from material_dataset import collate_fn
from train_material_detection import HungarianMatcher

CLASS_NAMES = ['Concrete', 'Glass', 'Metal', 'Wood']
SCORE_THRESHOLDS = [0.25, 0.45, 0.4, 0.4]
BG_THRESHOLD = 0.5

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load('/home/user/wentao/checkpoints_contrastive/checkpoint_best.pth', map_location=device, weights_only=False)
    ma = ckpt.get('args', {})
    model = MaterialDetectionModel(
        backbone_name=ma.get('backbone','vit_base_patch16_224'), img_size=224, num_classes=4,
        num_queries=ma.get('num_queries',100), num_decoder_layers=ma.get('num_decoder_layers',6),
        use_multi_view=False, num_views=1, pretrained_backbone=False,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    proto_data = torch.load('./class_prototypes_contrastive.pth', map_location=device, weights_only=False)
    prototypes = torch.stack([proto_data['prototypes'][i] for i in range(4)]).to(device)
    per_class_thr = torch.tensor([proto_data['cos_stats'][c]['min_cos'] - 0.02 for c in range(4)], device=device)
    print("Per-class thresholds:", [f"{t:.4f}" for t in per_class_thr.tolist()])

    dataset = BenchmarkDataset(benchmark_root='./Eval_benchmark', img_size=224)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)

    total_det = 0
    total_known = 0
    total_unknown = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            out = model(images)
            B = images.shape[0]
            for b in range(B):
                logits = out['pred_logits'][b]
                feats = out['proj_features'][b]
                probs = F.softmax(logits, dim=-1)
                bg = probs[:, -1]
                kp = probs[:, :-1]
                ms, ml = kp.max(dim=-1)
                feats_n = F.normalize(feats, dim=-1)
                cos = feats_n @ prototypes.T
                max_cos, nearest = cos.max(dim=-1)
                is_obj = bg < BG_THRESHOLD
                for q in range(logits.shape[0]):
                    if not is_obj[q]:
                        continue
                    c = ml[q].item()
                    if ms[q].item() < SCORE_THRESHOLDS[c]:
                        continue
                    total_det += 1
                    nc = nearest[q].item()
                    if max_cos[q].item() >= per_class_thr[nc].item():
                        total_known += 1
                    else:
                        total_unknown += 1

    print(f"\nEval_benchmark: {total_det} detections")
    print(f"  Known: {total_known} ({total_known/total_det*100:.1f}%)")
    print(f"  False Unknown: {total_unknown} ({total_unknown/total_det*100:.1f}%)")
    print(f"  Expected: ~{500*4}=2000 GT boxes, detected {total_det}")

if __name__ == '__main__':
    main()
