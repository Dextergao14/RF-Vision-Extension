# 微调测试 V3 状态（Score Threshold 0.4）

## 📋 参数配置

- **学习率**: 1e-5
- **训练轮数**: 3 epochs
- **Score threshold**: **0.4**（从0.2提高，进一步减少FP）
- **Batch size**: 4
- **IoU threshold**: 0.5

## 🎯 预期改进

1. **更高的score threshold** → 进一步过滤低置信度预测，大幅减少FP
2. **提高Precision** → 通过减少FP来提升Precision
3. **关注50%配置** → 用户注意到250张support set可能存在数据泄露问题

## ⚠️ 关于50%配置的数据泄露问题

**问题**: 50%配置中，support set和query set各占50%（250张），可能存在数据泄露：
- Support set和query set可能来自相同的场景/分布
- 模型在support set上微调后，在query set上评估时可能"见过"类似的数据

**建议**: 
- 50%配置的结果可能过于乐观
- 应该重点关注10%-40%配置的结果
- 如果可能，应该确保support set和query set来自不同的场景/分布

## 📊 数据集大小

- Benchmark总图像数: 500张
- 各比例对应的support set大小:
  - 10%: 50张图像 → Query: 450张（泄露风险低）
  - 20%: 100张图像 → Query: 400张（泄露风险低）
  - 30%: 150张图像 → Query: 350张（泄露风险中等）
  - 40%: 200张图像 → Query: 300张（泄露风险中等）
  - 50%: 250张图像 → Query: 250张（泄露风险高 ⚠️）

## 🚀 运行状态

**开始时间**: 正在运行中...

**日志文件**: `finetune_percentage_batch_v3.log`

**结果文件**: `finetune_percentage_results_v3.json`

## 📈 预期结果

相比V2（score threshold 0.2），预期：
- ✅ FP数量进一步减少
- ✅ Precision提升
- ⚠️ Recall可能略有下降（因为过滤了更多预测）
- ⚠️ 50%配置的结果需要谨慎解读（可能存在数据泄露）

## 🔍 监控命令

```bash
# 查看实时日志
tail -f finetune_percentage_batch_v3.log

# 查看进程状态
ps aux | grep batch_finetune_percentages

# 查看结果（如果已完成）
cat finetune_percentage_results_v3.json | python -m json.tool
```
