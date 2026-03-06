# RF-Vision-Extension

An extension for **material detection and few-shot evaluation** based on **RF path gain** data. The project uses a **ViT + DETR-style detection head** (optionally Swin) and supports **4 material classes** (**Concrete / Glass / Metal / Wood**), **support-set fine-tuning**, and **unknown-class detection** experiments.

---

## Features

- **RF path gain → material detection** pipeline
- **ViT/Swin backbone** + **DETR-style detection head**
- **4-class detection**: Concrete / Glass / Metal / Wood
- **Few-shot / support-set fine-tuning** (e.g., 20% support)
- **Unknown material detection** via feature prototypes / thresholds
- Benchmark evaluation + visualization scripts
- Optional **contrastive fine-tuning** and **MAML domain adaptation**

---

## Project Structure

```text
RF-Vision-Extension/
├── README.md                   # This file
├── requirements.txt
├── material_detection_model.py  # Model: ViT/Swin backbone + DETR head + optional projection head
├── material_dataset.py          # YOLO-format dataset loader
├── train_material_detection.py  # Main training script
├── train_contrastive_finetune.py    # Contrastive fine-tuning (for unknown detection)
├── evaluate_benchmark.py        # Eval_benchmark evaluation
├── evaluate_and_visualize_20pct.py   # 20% fine-tune + evaluation and visualization
├── batch_finetune_percentages.py     # Support fine-tuning across multiple percentages
├── sweep_nms_20pct.py           # NMS threshold sweep
├── sweep_glass_threshold_20pct.py    # Glass-class score-threshold sweep
├── infer_hard_samples.py        # Inference & visualization (incl. unknown prototype decision)
├── compute_class_prototypes.py  # Compute class feature prototypes
├── compute_query_prototypes.py  # Per-query Mahalanobis statistics
├── train_maml_domain_adaptation.py   # MAML domain adaptation
├── evaluate_maml_query_set.py    # MAML evaluation
├── Eval_benchmark/              # Benchmark images and labels
├── vanilla-dataset/             # Train/val/test (YOLO format)
├── hard_samples/                # Hard samples
├── unknown_material_samples/    # Unknown material samples (e.g., marble)
└── docs/
   ├── README_material_detection.md
   ├── BENCHMARK_EVAL_README.md
   └── UNKNOWN_CLASS_IMPLEMENTATION.md
