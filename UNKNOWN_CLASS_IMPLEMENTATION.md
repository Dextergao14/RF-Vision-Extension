# Unknown类实现说明

## 概述

本实现为材料检测模型添加了"Unknown"类别，用于检测数据集中未标注的未知材料。由于训练数据中没有unknown类的标注，我们采用了**开放集识别（Open-Set Recognition）**的方法来实现这一功能。

## 实现方法

### 1. 模型架构修改

- **分类器输出维度**：从 `num_classes + 1`（4类+背景）改为 `num_classes + 2`（4类+unknown+背景）
- **类别索引**：
  - 0-3: 已知类（Concrete, Glass, Metal, Wood）
  - 4: Unknown类
  - 5: Background类（用于DETR的"无对象"查询）

### 2. 训练策略

由于数据集没有unknown类的标注，我们采用了以下策略：

#### 2.1 损失函数
- **标准交叉熵损失**：对于匹配的查询，使用真实标签进行监督学习
- **能量损失（Energy Loss）**：对于未匹配的查询，鼓励低置信度的预测预测为unknown
  - 如果已知类概率 < 0.3，鼓励预测为unknown
  - 如果已知类概率 > 0.5，不鼓励预测为unknown

#### 2.2 权重设置
- Unknown类权重：`unknown_coef = 0.5`（可调整）
- Background类权重：`eos_coef = 0.25`（可调整）
- 能量损失权重：`loss_energy = 0.1`（较小，避免干扰主要损失）

### 3. 推理策略

在推理时，使用以下逻辑判断unknown类：

1. **计算概率分布**：
   - 已知类概率：`known_probs = sum(probs[0:4])`
   - Unknown类概率：`unknown_prob = probs[4]`

2. **判断条件**：
   - 如果 `unknown_prob > score_threshold` 且 `known_probs < 0.3`，则预测为unknown
   - 否则，选择概率最高的已知类

3. **阈值设置**：
   - `score_threshold`：默认0.05（可调整）
   - `known_probs < 0.3`：已知类概率阈值（可调整）

## 使用方法

### 训练模型

```bash
python train_material_detection.py \
    --data_root /path/to/dataset \
    --num_epochs 100 \
    --lr 1e-5 \
    --batch_size 8
```

模型会自动支持unknown类的训练。

### 评估模型

```bash
python evaluate_material_detection.py \
    --checkpoint ./checkpoints_material_detection/checkpoint_best.pth \
    --data_root /path/to/dataset \
    --score_threshold 0.05 \
    --iou_threshold 0.5
```

评估结果会包含unknown类的指标。

### 调整参数

如果需要调整unknown类的检测敏感度，可以修改以下参数：

1. **训练时**（`train_material_detection.py`）：
   - `unknown_coef`：unknown类权重（默认0.5）
   - `loss_energy`权重：能量损失权重（默认0.1）

2. **推理时**（`evaluate_material_detection.py`）：
   - `score_threshold`：分数阈值（默认0.05）
   - `known_probs < 0.3`：已知类概率阈值（在代码中修改）

## 评估指标

评估结果会包含以下指标：

- **已知类**（Concrete, Glass, Metal, Wood）：
  - Precision, Recall, F1-Score
  - TP, FP, FN

- **Unknown类**：
  - Precision, Recall, F1-Score
  - TP, FP, FN
  - 注意：由于没有unknown类的ground truth，FN可能为0

- **总体指标**：
  - 只统计已知类的总体指标（不包括unknown）

## 注意事项

1. **训练数据**：由于没有unknown类的标注，模型主要通过能量损失学习unknown类，效果可能不如有标注的情况。

2. **阈值调整**：
   - 降低`score_threshold`或`known_probs`阈值会增加unknown类的检测率，但可能增加误检
   - 提高阈值会减少unknown类的检测率，但可能漏检真正的unknown类

3. **模型兼容性**：
   - 旧模型（4类+背景）与新模型（4类+unknown+背景）不兼容
   - 需要重新训练模型才能使用unknown类功能

4. **评估限制**：
   - 由于测试集也没有unknown类的标注，无法准确评估unknown类的召回率
   - 只能评估unknown类的精确率（如果预测为unknown的检测确实不是已知类）

## 未来改进方向

1. **使用外部数据集**：收集一些真正的unknown类样本，添加到训练集中
2. **改进损失函数**：使用更先进的开放集识别方法（如OpenMax、OSRCI等）
3. **自适应阈值**：根据验证集的表现自动调整阈值
4. **集成方法**：结合多个模型的预测结果来判断unknown类

## 相关文件

- `material_detection_model.py`：模型定义
- `train_material_detection.py`：训练脚本
- `evaluate_material_detection.py`：评估脚本
- `material_dataset.py`：数据集加载器
