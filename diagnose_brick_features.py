#!/usr/bin/env python3
"""
诊断 brick 在对比学习特征空间中的分布：
- 每个检测框被判成什么类、softmax 分数多少
- 与 4 个已知类原型的 cos sim 分别是多少
- 与最近原型的 cos sim vs 已知类 benchmark 的统计做对比
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import cv2
from pathlib import Path
from collections import defaultdict
from material_detection_model import MaterialDetectionModel
from train_material_detection import box_cxcywh_to_xyxy
from torchvision import transforms

CLASS_NAMES = ['Concrete', 'Glass', 'Metal', 'Wood']
SCORE_THRESHOLDS = [0.25, 0.45, 0.4, 0.4]
BG_THRESHOLD = 0.5


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = '/home/user/wentao/checkpoints_contrastive/checkpoint_best.pth'
    proto_path = './class_prototypes_contrastive.pth'
    img_dir = './unknown_material_samples_2/train'

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

    # Load prototypes and stats
    proto_data = torch.load(proto_path, map_location=device, weights_only=False)
    prototypes = torch.stack([proto_data['prototypes'][i] for i in range(4)]).to(device)  # [4, 128]
    cos_stats = proto_data['cos_stats']

    print("=" * 70)
    print("已知类在 Eval_benchmark 上的 cos sim 统计（对比学习 proj_features）：")
    for c in range(4):
        s = cos_stats[c]
        print(f"  {CLASS_NAMES[c]:10s}: mean={s['mean_cos']:.4f}, std={s['std_cos']:.4f}, min={s['min_cos']:.4f}")
    print("=" * 70)

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Collect brick detection stats
    brick_cos_to_predicted = []
    brick_cos_to_all = []  # (max_cos_to_any_proto, predicted_class)
    brick_max_cos_all = []
    brick_predicted_classes = defaultdict(int)

    img_files = sorted(Path(img_dir).glob('*.png'))
    for img_path in img_files:
        orig = cv2.imread(str(img_path))
        if orig is None:
            continue
        img_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        inp = tfm(img_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(inp)
        logits = out['pred_logits'][0]
        boxes = out['pred_boxes'][0]
        proj_feats = out['proj_features'][0]  # [Q, 128]

        probs = F.softmax(logits, dim=-1)
        bg_scores = probs[:, -1]
        known_probs = probs[:, :-1]
        max_scores, max_labels = known_probs.max(dim=-1)

        cos_sim = proj_feats @ prototypes.T  # [Q, 4]

        is_object = bg_scores < BG_THRESHOLD
        for q in range(logits.shape[0]):
            if not is_object[q]:
                continue
            cls = max_labels[q].item()
            sc = max_scores[q].item()
            if sc < SCORE_THRESHOLDS[cls]:
                continue

            cos_to_pred = cos_sim[q, cls].item()
            cos_to_all = cos_sim[q].cpu().numpy()
            max_cos = cos_to_all.max()

            brick_cos_to_predicted.append(cos_to_pred)
            brick_max_cos_all.append(max_cos)
            brick_predicted_classes[CLASS_NAMES[cls]] += 1
            brick_cos_to_all.append(cos_to_all)

    print(f"\nBrick 被检测出的框数: {len(brick_cos_to_predicted)}")
    print(f"被判成的类别分布: {dict(brick_predicted_classes)}")

    if brick_cos_to_predicted:
        arr = np.array(brick_cos_to_predicted)
        print(f"\nBrick → 被预测类原型的 cos sim:")
        print(f"  mean={arr.mean():.4f}, std={arr.std():.4f}, min={arr.min():.4f}, max={arr.max():.4f}")

        arr_max = np.array(brick_max_cos_all)
        print(f"\nBrick → 最近已知类原型的 max cos sim:")
        print(f"  mean={arr_max.mean():.4f}, std={arr_max.std():.4f}, min={arr_max.min():.4f}, max={arr_max.max():.4f}")

        all_cos = np.stack(brick_cos_to_all)  # [N, 4]
        print(f"\nBrick → 每个已知类原型的 cos sim:")
        for c in range(4):
            col = all_cos[:, c]
            print(f"  → {CLASS_NAMES[c]:10s}: mean={col.mean():.4f}, std={col.std():.4f}, min={col.min():.4f}, max={col.max():.4f}")

    # Compare with known class stats
    print("\n" + "=" * 70)
    print("对比：已知类 min cos  vs  Brick max cos")
    known_mins = [cos_stats[c]['min_cos'] for c in range(4)]
    print(f"  已知类 min cos: {[f'{v:.4f}' for v in known_mins]}")
    if brick_max_cos_all:
        print(f"  Brick max cos:  mean={np.mean(brick_max_cos_all):.4f}, max={np.max(brick_max_cos_all):.4f}")
        gap = min(known_mins) - np.max(brick_max_cos_all)
        print(f"  Gap (known_min - brick_max): {gap:.4f}")
        if gap > 0:
            print(f"  ✅ 有正 gap，可以用 max-cos-to-any-proto 策略分离！")
            print(f"  建议阈值范围: [{np.max(brick_max_cos_all):.4f}, {min(known_mins):.4f}]")
        else:
            print(f"  ❌ 没有 gap，brick 与已知类在 cos sim 上有重叠")

    # Check energy score difference
    print("\n" + "=" * 70)
    print("Energy score 分析（-log(sum(exp(known_logits)))）：")
    brick_energies = []
    for img_path in img_files:
        orig = cv2.imread(str(img_path))
        if orig is None:
            continue
        img_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        inp = tfm(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
        logits = out['pred_logits'][0]
        probs = F.softmax(logits, dim=-1)
        bg_scores = probs[:, -1]
        known_probs = probs[:, :-1]
        max_scores, max_labels = known_probs.max(dim=-1)
        known_logits = logits[:, :-1]  # [Q, 4]
        energy = -torch.logsumexp(known_logits, dim=-1)  # [Q]
        is_object = bg_scores < BG_THRESHOLD
        for q in range(logits.shape[0]):
            if not is_object[q]:
                continue
            cls = max_labels[q].item()
            if max_scores[q].item() < SCORE_THRESHOLDS[cls]:
                continue
            brick_energies.append(energy[q].item())

    if brick_energies:
        arr_e = np.array(brick_energies)
        print(f"  Brick energy: mean={arr_e.mean():.4f}, std={arr_e.std():.4f}, min={arr_e.min():.4f}, max={arr_e.max():.4f}")


if __name__ == '__main__':
    main()
