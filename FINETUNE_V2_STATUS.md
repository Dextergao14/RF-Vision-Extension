# 微调测试 V2 状态（优化参数）

## 📋 新参数配置

- **学习率**: 1e-5（从1e-4降低，更保守）
- **训练轮数**: 3 epochs（从10减少，避免过拟合）
- **Score threshold**: 0.2（从0.05提高，减少FP）
- **Batch size**: 4
- **IoU threshold**: 0.5

## 🎯 预期改进

1. **更小的学习率** → 更保守的参数更新，减少过拟合风险
2. **更少的epoch** → 避免在小数据集上过度训练
3. **更高的score threshold** → 过滤低置信度预测，大幅减少FP

## 📊 数据集大小

- Benchmark总图像数: 500张
- 各比例对应的support set大小:
  - 10%: 50张图像 → Query: 450张
  - 20%: 100张图像 → Query: 400张
  - 30%: 150张图像 → Query: 350张
  - 40%: 200张图像 → Query: 300张
  - 50%: 250张图像 → Query: 250张

## 🚀 运行状态

**开始时间**: 正在运行中...

**进程ID**: 2285274

**日志文件**: `finetune_percentage_batch_v2.log`

**结果文件**: `finetune_percentage_results_v2.json`

## 📈 当前进度

- ✅ 10%配置: 已完成微调（3 epochs），正在评估query set
- ⏳ 20%配置: 等待中
- ⏳ 30%配置: 等待中
- ⏳ 40%配置: 等待中
- ⏳ 50%配置: 等待中

## 🔍 监控命令

```bash
# 查看实时日志
tail -f finetune_percentage_batch_v2.log

# 查看进程状态
ps aux | grep batch_finetune_percentages

# 查看GPU使用情况
nvidia-smi

# 查看结果（如果已完成）
cat finetune_percentage_results_v2.json | python -m json.tool
```

## 📊 与V1对比

| 配置项 | V1 | V2 | 改进 |
|--------|----|----|------|
| 学习率 | 1e-4 | 1e-5 | ✅ 更保守 |
| Epochs | 10 | 3 | ✅ 减少过拟合 |
| Score threshold | 0.05 | 0.2 | ✅ 减少FP |

## 💡 预期结果

相比V1，预期：
- ✅ FP数量大幅减少（score threshold提高）
- ✅ 更稳定的性能（学习率降低）
- ✅ 避免30%/40%配置的完全失败（epoch减少）
