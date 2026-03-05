# Unknown类添加 - 修改总结

## 修改的文件

### 1. `material_detection_model.py`
- **修改位置**: `DETRDetectionHead.__init__` 和 `forward`
- **修改内容**: 
  - 分类器输出维度从 `num_classes + 1` 改为 `num_classes + 2`
  - 现在输出6个类别：4个已知类 + unknown + background

### 2. `material_dataset.py`
- **修改位置**: `MaterialDetectionDataset.__init__`
- **修改内容**:
  - 添加 `'Unknown'` 到类别名称列表
  - 添加 `unknown_class_id = 4` 属性
  - `num_classes` 保持为4（已知类数量）

### 3. `train_material_detection.py`
- **修改位置**: 
  - `HungarianMatcher.forward`: 更新注释说明输出维度
  - `SetCriterion.__init__`: 添加 `unknown_coef` 参数，调整权重向量
  - `SetCriterion.loss_labels`: 添加能量损失（energy loss）
  - `main`: 更新损失函数和模型初始化

- **主要修改**:
  - 损失函数权重向量从 `num_classes + 1` 改为 `num_classes + 2`
  - 添加能量损失，鼓励低置信度的未匹配查询预测为unknown
  - 添加 `loss_energy` 到 `weight_dict`

### 4. `evaluate_material_detection.py`
- **修改位置**:
  - `evaluate_detections`: 添加 `include_unknown` 参数，处理unknown类的评估
  - `evaluate_model`: 更新预测逻辑，支持unknown类检测
  - 类别名称列表添加 `'Unknown'`

- **主要修改**:
  - 预测逻辑：如果 `unknown_prob > threshold` 且 `known_probs < 0.3`，预测为unknown
  - 评估逻辑：正确处理unknown类的TP/FP/FN
  - 总体指标只统计已知类（不包括unknown）

## 新增的文件

### 1. `UNKNOWN_CLASS_IMPLEMENTATION.md`
- 详细的实现说明文档
- 使用方法、参数调整、注意事项等

### 2. `test_unknown_class.py`
- 测试脚本，验证修改是否正确
- 测试模型输出形状、unknown检测逻辑、损失函数

## 关键参数

### 训练参数
- `unknown_coef = 0.5`: unknown类的损失权重
- `loss_energy = 0.1`: 能量损失的权重
- `eos_coef = 0.25`: background类的权重

### 推理参数
- `score_threshold = 0.05`: 预测分数阈值
- `known_probs < 0.3`: 已知类概率阈值（用于判断unknown）

## 使用方法

### 训练
```bash
python train_material_detection.py --data_root /path/to/dataset
```

### 评估
```bash
python evaluate_material_detection.py --checkpoint /path/to/checkpoint.pth
```

### 测试
```bash
python test_unknown_class.py
```

## 注意事项

1. **模型兼容性**: 旧模型（4类+背景）与新模型（4类+unknown+背景）不兼容，需要重新训练

2. **数据集**: 训练数据仍然只需要4个已知类的标注，不需要unknown类的标注

3. **评估限制**: 由于测试集也没有unknown类的标注，无法准确评估unknown类的召回率

4. **阈值调整**: 可以根据实际需求调整 `score_threshold` 和 `known_probs` 阈值

## 测试结果

所有测试通过：
- ✅ 模型输出形状正确
- ✅ Unknown类检测逻辑正确
- ✅ 损失函数计算正确
