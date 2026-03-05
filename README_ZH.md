# RF-Vision-Extension

基于 RF path gain 数据的材料检测与少样本评估扩展：ViT + DETR 风格检测头，支持 4 类材料（Concrete / Glass / Metal / Wood）、support set 微调与 unknown 检测实验。

## 项目结构

```
RF-Vision-Extension/
├── README.md                 # 本文件
├── requirements.txt
├── material_detection_model.py # 模型：ViT/Swin backbone + DETR head + 可选 projection head
├── material_dataset.py       # YOLO 格式数据集加载
├── train_material_detection.py      # 主训练脚本
├── train_contrastive_finetune.py    # 对比学习微调（unknown 检测用）
├── evaluate_benchmark.py      # Eval_benchmark 评估
├── evaluate_and_visualize_20pct.py   # 20% 微调 + 评估与可视化
├── batch_finetune_percentages.py    # 多比例 support 微调
├── sweep_nms_20pct.py        # NMS 阈值扫描
├── sweep_glass_threshold_20pct.py   # Glass 类别分数阈值扫描
├── infer_hard_samples.py     # 推理与可视化（含 unknown 原型判定）
├── compute_class_prototypes.py      # 计算类别特征原型
├── compute_query_prototypes.py      # 按 query 的 Mahalanobis 统计
├── train_maml_domain_adaptation.py  # MAML 域适应
├── evaluate_maml_query_set.py       # MAML 评估
├── Eval_benchmark/           # 评估用图像与标签
├── vanilla-dataset/         # 训练/验证/测试（YOLO 格式）
├── hard_samples/            # 困难样本
├── unknown_material_samples/ # 未知材料样本（如 marble）
└── docs: README_material_detection.md, BENCHMARK_EVAL_README.md, UNKNOWN_CLASS_IMPLEMENTATION.md 等
```

## 环境与依赖

```bash
pip install -r requirements.txt
# 需 PyTorch、torchvision、timm、opencv-python、scipy、tqdm 等
```

## 快速开始

1. **训练**（使用本仓库内 `vanilla-dataset`）  
   ```bash
   python train_material_detection.py --data_root ./vanilla-dataset --save_dir ./checkpoints_material_detection
   ```

2. **在 Eval_benchmark 上评估**  
   ```bash
   python evaluate_benchmark.py --checkpoint_path ./checkpoints_material_detection/checkpoint_best.pth --benchmark_root ./Eval_benchmark
   ```

3. **Support set 微调 + 评估**  
   ```bash
   python evaluate_and_visualize_20pct.py --benchmark_root ./Eval_benchmark
   ```
   脚本内默认从 `./checkpoints_material_detection_new_data/checkpoint_best.pth` 加载；若路径不同，请通过参数传入。

4. **推理与可视化**（含 unknown 原型阈值）  
   ```bash
   python infer_hard_samples.py
   ```
   默认读取 `./hard_samples/test` 或 `./unknown_material_samples/train`，输出到 `./hard_samples_visualizations` 或 `./unknown_material_visualizations`。需先运行 `compute_class_prototypes.py` 生成 `class_prototypes.pth`。

更多说明见 **README_material_detection.md**、**BENCHMARK_EVAL_README.md**、**UNKNOWN_CLASS_IMPLEMENTATION.md**。

## 开源到 GitHub

1. 在 GitHub 新建空仓库（如 `RF-Vision-Extension`）。
2. 在本目录初始化并推送：
   ```bash
   cd /path/to/RF-Vision-Extension
   git init
   git add .
   git commit -m "Initial: RF material detection and evaluation"
   git remote add origin git@github.com:<你的用户名>/RF-Vision-Extension.git
   git branch -M main
   git push -u origin main
   ```
3. 若 `.gitignore` 已忽略 `checkpoints_*/` 与 `*.pth`，权重需自行备份或通过 Release 提供。

## License

请根据你的需求添加 LICENSE 文件（如 MIT、Apache-2.0）。
