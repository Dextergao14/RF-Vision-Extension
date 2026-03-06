#!/usr/bin/env python3
"""
诊断 brick 样本上各检测框的 softmax 分数分布，
区分 brick 物体和 Glass/Metal/Wood 物体。
使用 outlier exposure fine-tuned 模型。
"""
import torch, torch.nn.functional as F, numpy as np, json, cv2, random
from pathlib import Path
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import transforms
from collections import defaultdict

from material_detection_model import MaterialDetectionModel
from evaluate_benchmark import BenchmarkDataset
from material_dataset import collate_fn
from train_material_detection import box_cxcywh_to_xyxy, HungarianMatcher, SetCriterion
from finetune_outlier_exposure import BrickOutlierDataset, mixed_collate_fn, finetune_with_outlier

CLASS_NAMES = ['Concrete', 'Glass', 'Metal', 'Wood']
BG_THRESHOLD = 0.5


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load('../checkpoints_material_detection_new_data/checkpoint_best.pth',
                       map_location=device, weights_only=False)
    ma = ckpt.get('args', {})
    model = MaterialDetectionModel(
        backbone_name=ma.get('backbone', 'vit_base_patch16_224'), img_size=224, num_classes=4,
        num_queries=ma.get('num_queries', 100), num_decoder_layers=ma.get('num_decoder_layers', 6),
        use_multi_view=False, num_views=1, pretrained_backbone=False).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)

    benchmark_ds = BenchmarkDataset(benchmark_root='./Eval_benchmark', img_size=224)
    brick_ds = BrickOutlierDataset('./unknown_material_samples_2/train', img_size=224)
    finetune_with_outlier(model, benchmark_ds, brick_ds, device, epochs=30)

    # Load brick COCO to know which box is Brick vs known
    with open('./unknown_material_samples_2/train/_annotations.coco.json') as f:
        coco = json.load(f)
    img_id_to_file = {im['id']: im['file_name'] for im in coco['images']}
    anns_by_file = defaultdict(list)
    for ann in coco['annotations']:
        fname = img_id_to_file[ann['image_id']]
        x, y, w, h = float(ann['bbox'][0]), float(ann['bbox'][1]), float(ann['bbox'][2]), float(ann['bbox'][3])
        cat_id = ann['category_id']
        anns_by_file[fname].append({'bbox': [x, y, x+w, y+h], 'cat_id': cat_id})

    tfm = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224, 224)),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    brick_scores = []  # max known score for detections matching Brick GT
    known_scores = defaultdict(list)  # max known score for detections matching Glass/Metal/Wood GT
    brick_pred_classes = defaultdict(int)

    img_files = sorted(Path('./unknown_material_samples_2/train').glob('*.png'))
    model.eval()
    for img_path in img_files:
        fname = img_path.name
        orig = cv2.imread(str(img_path))
        if orig is None: continue
        oh, ow = orig.shape[:2]
        img_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        inp = tfm(img_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(inp)
        logits = out['pred_logits'][0]
        boxes_cxcywh = out['pred_boxes'][0]
        probs = F.softmax(logits, dim=-1)
        bg = probs[:, -1]
        kp = probs[:, :-1]
        max_scores, max_labels = kp.max(dim=-1)

        gts = anns_by_file.get(fname, [])
        det_queries = []
        for q in range(logits.shape[0]):
            if bg[q] >= BG_THRESHOLD: continue
            if max_scores[q].item() < 0.15: continue
            det_queries.append(q)

        if not det_queries or not gts: continue
        pred_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh[det_queries])

        gt_matched = set()
        for di, qi in enumerate(det_queries):
            pbox = pred_xyxy[di].cpu().numpy()
            px = np.array([pbox[0]*ow, pbox[1]*oh, pbox[2]*ow, pbox[3]*oh])
            best_iou, best_gi = 0, -1
            for gi, g in enumerate(gts):
                if gi in gt_matched: continue
                gx = np.array(g['bbox'])
                ix1 = max(px[0], gx[0]); iy1 = max(px[1], gx[1])
                ix2 = min(px[2], gx[2]); iy2 = min(px[3], gx[3])
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2-ix1)*(iy2-iy1)
                    a1 = (px[2]-px[0])*(px[3]-px[1])
                    a2 = (gx[2]-gx[0])*(gx[3]-gx[1])
                    iou = inter/(a1+a2-inter+1e-6)
                else: iou = 0
                if iou > best_iou: best_iou = iou; best_gi = gi

            if best_iou >= 0.3 and best_gi >= 0:
                gt_matched.add(best_gi)
                cat = gts[best_gi]['cat_id']
                sc = max_scores[qi].item()
                cls = max_labels[qi].item()
                if cat == 1:  # Brick
                    brick_scores.append(sc)
                    brick_pred_classes[CLASS_NAMES[cls]] += 1
                else:  # Glass(2), Metal(3), Wood(4)
                    cat_name = {2: 'Glass', 3: 'Metal', 4: 'Wood'}.get(cat, '?')
                    known_scores[cat_name].append(sc)

    print("\n" + "="*60)
    print(f"Brick detections: {len(brick_scores)}")
    if brick_scores:
        arr = np.array(brick_scores)
        print(f"  Max known score: mean={arr.mean():.4f}, std={arr.std():.4f}, "
              f"min={arr.min():.4f}, max={arr.max():.4f}")
        print(f"  Predicted as: {dict(brick_pred_classes)}")
        for thr in [0.3, 0.35, 0.4, 0.45, 0.5]:
            pct = (arr < thr).sum() / len(arr) * 100
            print(f"  score < {thr}: {pct:.0f}% would be Unknown")

    print()
    for cls_name, scores in sorted(known_scores.items()):
        arr = np.array(scores)
        print(f"{cls_name} detections ({len(arr)}):")
        print(f"  Max known score: mean={arr.mean():.4f}, std={arr.std():.4f}, "
              f"min={arr.min():.4f}, max={arr.max():.4f}")
        for thr in [0.3, 0.35, 0.4, 0.45, 0.5]:
            pct = (arr < thr).sum() / len(arr) * 100
            print(f"  score < {thr}: {pct:.0f}% false Unknown")


if __name__ == '__main__':
    main()
