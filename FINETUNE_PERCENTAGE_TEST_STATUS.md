# 不同比例Support Set微调测试状态

## 测试配置

- **预训练模型**: `./checkpoints_material_detection_new_data/checkpoint_best.pth`
- **Benchmark数据集**: `/home/user/wentao/RF-Vision-Extension/Eval_benchmark`
- **测试比例**: 10%, 20%, 30%, 40%, 50%
- **微调参数**:
  - 轮数: 10 epochs
  - 学习率: 1e-4
  - Batch size: 4
  - 权重衰减: 1e-4
  - 学习率调度: StepLR (step_size=5, gamma=0.5)
- **评估参数**:
  - Score threshold: 0.05
  - IoU threshold: 0.5

## 数据集大小

- Benchmark总图像数: 500张
- 各比例对应的support set大小:
  - 10%: 50张图像 → Query: 450张
  - 20%: 100张图像 → Query: 400张
  - 30%: 150张图像 → Query: 350张
  - 40%: 200张图像 → Query: 300张
  - 50%: 250张图像 → Query: 250张

## 运行状态

**开始时间**: 正在运行中...

**进程ID**: 2257169

**日志文件**: `finetune_percentage_batch.log`

**结果文件**: `finetune_percentage_results.json`

## 当前进度

- ✅ 10%配置: 正在微调中（第2个epoch）
- ⏳ 20%配置: 等待中
- ⏳ 30%配置: 等待中
- ⏳ 40%配置: 等待中
- ⏳ 50%配置: 等待中

## 监控命令

```bash
# 查看实时日志
tail -f finetune_percentage_batch.log

# 查看进程状态
ps aux | grep batch_finetune_percentages

# 查看GPU使用情况
nvidia-smi

# 查看结果（如果已完成）
cat finetune_percentage_results.json | python -m json.tool
```

## 预期输出

脚本会依次测试每个比例，对每个配置：
1. 在support set上微调10个epoch
2. 在query set上评估
3. 输出Precision、Recall、F1-Score等指标

最终会生成一个汇总表格，方便比较不同比例的效果。
