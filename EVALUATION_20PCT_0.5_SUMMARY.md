# 20%配置评估结果总结（Score Threshold 0.5）

## 📊 评估结果

### 总体指标

| 指标 | 值 |
|------|-----|
| **Precision** | **60.69%** |
| **Recall** | **46.40%** |
| **F1-Score** | **52.59%** |
| TP | 741 |
| FP | 480 |
| FN | 856 |
| Total GT | 1,597 |

### 各类别表现

| 类别 | Precision | Recall | F1-Score | TP | FP | FN |
|------|-----------|--------|----------|----|----|----|
| **Metal** | **50.00%** | **99.75%** | **66.61%** | 400 | 400 | 1 |
| **Wood** | **81.00%** | **85.25%** | **83.07%** | 341 | 80 | 59 |
| Concrete | 0.00% | 0.00% | 0.00% | 0 | 0 | 398 |
| Glass | 0.00% | 0.00% | 0.00% | 0 | 0 | 398 |

## 🎯 关键发现

### ✅ 优点

1. **Precision大幅提升**: 60.69%（相比score threshold 0.4的20.52%提升了40%）
2. **FP大幅减少**: 480个FP（相比score threshold 0.4的5,696个减少了92%）
3. **Wood类表现优秀**: Precision 81%, Recall 85.25%, F1 83.07%
4. **Metal类Recall极高**: 99.75%的Recall，几乎检测到所有Metal对象

### ⚠️ 问题

1. **Recall下降**: 46.40%（相比score threshold 0.4的92.11%下降了45.71%）
2. **Concrete和Glass类完全失败**: 0 TP，完全没有检测到这两个类别
3. **FN数量较高**: 856个FN，说明很多ground truth没有被检测到

## 📈 与之前配置对比

| 配置 | Score Thr | Precision | Recall | F1-Score | FP |
|------|-----------|-----------|--------|----------|----|
| V2 (0.2) | 0.2 | 20.52% | 92.11% | 33.57% | 5,696 |
| V3 (0.4) | 0.4 | 20.52% | 92.11% | 33.57% | 5,696 |
| **当前 (0.5)** | **0.5** | **60.69%** | **46.40%** | **52.59%** | **480** |

**变化**:
- ✅ Precision: 20.52% → 60.69%（+40.17%）
- ⚠️ Recall: 92.11% → 46.40%（-45.71%）
- ✅ F1-Score: 33.57% → 52.59%（+19.02%）
- ✅ FP: 5,696 → 480（-92%）

## 💡 分析

### Score Threshold 0.5的影响

**优点**:
- 大幅减少了FP（从5,696减少到480）
- Precision大幅提升（从20.52%提升到60.69%）
- F1-Score提升（从33.57%提升到52.59%）

**缺点**:
- Recall大幅下降（从92.11%下降到46.40%）
- Concrete和Glass类完全没有检测到
- FN数量较高（856个）

### 类别表现分析

**Metal类**:
- ✅ Recall极高（99.75%），几乎检测到所有Metal对象
- ⚠️ Precision中等（50%），FP数量较高（400个）
- 说明模型对Metal类的检测很敏感，但容易产生误检

**Wood类**:
- ✅ Precision很高（81%），FP数量较少（80个）
- ✅ Recall较高（85.25%），漏检较少（59个）
- ✅ F1-Score最高（83.07%）
- 说明模型对Wood类的检测最准确

**Concrete和Glass类**:
- ❌ 完全没有检测到（0 TP）
- 可能原因：
  1. Score threshold 0.5太高，过滤掉了这两个类别的所有预测
  2. 模型对这两个类别的置信度较低
  3. Support set中这两个类别的样本可能不足

## 🎨 可视化结果

已生成10张可视化图像，保存在 `visualizations_20pct_0.5/` 目录：
- `visualization_001.png` 到 `visualization_010.png`

每张图像包含：
- 绿色虚线框：Ground Truth
- 彩色实线框：模型预测（不同颜色代表不同类别）
- 标签：类别名称和置信度分数

## 🎯 建议

### 1. **Score Threshold选择**

**Trade-off分析**:
- **Score 0.2-0.4**: 高Recall（92%+），但低Precision（20%），FP多（5,000+）
- **Score 0.5**: 中等Precision（60%），但低Recall（46%），FN多（856）

**建议**: 
- 如果关注**Precision**，使用score threshold 0.5
- 如果关注**Recall**，使用score threshold 0.2-0.4
- 如果关注**F1-Score**，score threshold 0.5更好（52.59% vs 33.57%）

### 2. **类别不平衡问题**

**问题**: Concrete和Glass类完全没有检测到

**可能原因**:
- Score threshold 0.5太高
- Support set中这两个类别的样本不足
- 模型对这两个类别的置信度较低

**建议**:
- 检查support set的类别分布
- 使用分层采样确保类别平衡
- 考虑降低score threshold或使用类别特定的threshold

### 3. **进一步优化**

1. **使用NMS**: 减少重复检测
2. **类别特定threshold**: 不同类别使用不同的score threshold
3. **检查置信度分布**: 分析模型输出的置信度分布，找到最佳threshold
4. **增加support set**: 如果可能，增加support set的大小，特别是Concrete和Glass类的样本

## 📝 总结

**20%配置 + Score Threshold 0.5的结果**:
- ✅ Precision: 60.69%（优秀）
- ⚠️ Recall: 46.40%（中等）
- ✅ F1-Score: 52.59%（良好）
- ✅ FP: 480（大幅减少）
- ❌ Concrete和Glass类完全失败

**最佳表现类别**: Wood类（Precision 81%, Recall 85.25%, F1 83.07%）

**主要问题**: Concrete和Glass类完全没有检测到，需要进一步调查原因。
