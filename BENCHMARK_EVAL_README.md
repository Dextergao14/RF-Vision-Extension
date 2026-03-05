# Eval Benchmark 评估指南

## 概述

`evaluate_benchmark.py` 是专门用于评估模型在Eval Benchmark数据集上性能的脚本。Benchmark数据集是独立于训练数据的评估集，用于最终模型性能评估。

## 使用方法

### 基本用法

```bash
python evaluate_benchmark.py \
    --checkpoint ./checkpoints_material_detection/checkpoint_best.pth \
    --benchmark_root /home/user/wentao/RF-Vision-Extension/Eval_benchmark
```

### 完整参数

```bash
python evaluate_benchmark.py \
    --checkpoint ./checkpoints_material_detection/checkpoint_best.pth \
    --benchmark_root /home/user/wentao/RF-Vision-Extension/Eval_benchmark \
    --img_size 224 \
    --batch_size 8 \
    --score_threshold 0.05 \
    --iou_threshold 0.5 \
    --output benchmark_evaluation_results.json
```

### 参数说明

- `--checkpoint`: 模型检查点路径（必需）
- `--benchmark_root`: Benchmark数据集根目录（默认: `/home/user/wentao/RF-Vision-Extension/Eval_benchmark`）
- `--img_size`: 图像尺寸（默认: 224）
- `--batch_size`: 批次大小（默认: 8）
- `--score_threshold`: 预测分数阈值（默认: 0.05）
- `--iou_threshold`: IoU阈值（默认: 0.5）
- `--use_multi_view`: 使用多视图（可选）
- `--num_views`: 视图数量（默认: 3，仅在use_multi_view时有效）
- `--output`: 结果保存路径（默认: `benchmark_evaluation_results.json`）

## Benchmark数据集结构

Benchmark数据集应该有以下结构：

```
Eval_benchmark/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels/
    ├── image1.txt
    ├── image2.txt
    └── ...
```

注意：标签文件名应该与图像文件名对应（除了扩展名）。

## 输出结果

评估结果会保存为JSON文件，包含以下信息：

1. **各类别指标**（Concrete, Glass, Metal, Wood, Unknown）：
   - Precision（精确率）
   - Recall（召回率）
   - F1-Score
   - TP（True Positive）
   - FP（False Positive）
   - FN（False Negative）

2. **总体指标**（只统计已知类，不包括Unknown）：
   - Precision
   - Recall
   - F1-Score
   - Classification Accuracy（分类准确率）
   - Detection Accuracy（检测准确率）
   - Overall Accuracy（总体准确率）

3. **Benchmark信息**：
   - 总图像数
   - 总GT对象数
   - 评估日期
   - 使用的阈值

## 示例输出

```json
{
  "Concrete": {
    "precision": 0.9565,
    "recall": 0.8800,
    "f1": 0.9167,
    "tp": 88,
    "fp": 4,
    "fn": 12
  },
  "overall": {
    "precision": 0.9495,
    "recall": 0.8925,
    "f1": 0.9201,
    "overall_accuracy": 0.8925,
    ...
  },
  "benchmark_info": {
    "total_images": 500,
    "total_gt_objects": 400,
    "evaluation_date": "2026-02-05 21:30:00",
    ...
  }
}
```

## 注意事项

1. **Benchmark数据集独立性**：
   - Benchmark数据不应该用于训练
   - 不应该根据benchmark结果调整模型
   - Benchmark用于最终性能评估和模型比较

2. **评估阈值**：
   - `score_threshold`: 控制预测的最低置信度
   - `iou_threshold`: 控制检测匹配的IoU阈值
   - 可以根据需要调整这些阈值

3. **Unknown类处理**：
   - 模型支持Unknown类的检测
   - Unknown类的评估指标会单独报告
   - 总体指标只统计已知类（不包括Unknown）

## 与Test Set评估的区别

| 特性 | Test Set | Benchmark |
|------|----------|-----------|
| 数据来源 | 与训练数据同一数据集 | 独立的新数据 |
| 用途 | 开发阶段快速评估 | 最终性能评估 |
| 使用频率 | 可以多次使用（调参） | 只在最终评估时使用 |
| 数据分布 | 与训练数据相同 | 可能不同（更真实） |

## 故障排除

### 问题1: 找不到图像或标签文件

**错误**: `无法加载图像: ...` 或 `FileNotFoundError`

**解决**: 
- 检查 `benchmark_root` 路径是否正确
- 确认 `images/` 和 `labels/` 目录存在
- 确认文件命名一致

### 问题2: 模型加载失败

**错误**: `KeyError` 或 `RuntimeError`

**解决**:
- 检查checkpoint路径是否正确
- 确认模型架构与checkpoint匹配
- 检查是否支持unknown类（需要重新训练的模型）

### 问题3: CUDA内存不足

**解决**:
- 减小 `batch_size`（例如改为4或2）
- 使用CPU: 脚本会自动检测CUDA，如果没有GPU会使用CPU

## 快速开始示例

```bash
# 1. 评估最佳模型
python evaluate_benchmark.py \
    --checkpoint ./checkpoints_material_detection/checkpoint_best.pth

# 2. 评估特定epoch的模型
python evaluate_benchmark.py \
    --checkpoint ./checkpoints_material_detection/checkpoint_epoch_50.pth \
    --output benchmark_epoch50_results.json

# 3. 使用不同的阈值
python evaluate_benchmark.py \
    --checkpoint ./checkpoints_material_detection/checkpoint_best.pth \
    --score_threshold 0.1 \
    --iou_threshold 0.6
```

## 相关文件

- `evaluate_material_detection.py`: 用于评估test set的脚本
- `material_dataset.py`: 数据集加载器
- `material_detection_model.py`: 模型定义
