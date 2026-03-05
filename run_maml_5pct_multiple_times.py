#!/usr/bin/env python3
"""
多次运行MAML 5%配置，取平均结果
"""

import subprocess
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

num_runs = 5
results = []

print("="*80)
print(f"运行MAML 5%配置 {num_runs} 次，取平均结果")
print("="*80)

for run_id in range(num_runs):
    print(f"\n{'='*80}")
    print(f"运行 {run_id + 1}/{num_runs}")
    print(f"{'='*80}")
    
    output_dir = f'./checkpoints_maml_5pct_run{run_id+1}'
    result_file = f'maml_5pct_run{run_id+1}_results.json'
    
    # 训练
    print(f"\n1. 训练MAML...")
    train_cmd = [
        'python', 'train_maml_domain_adaptation.py',
        '--source_data_root', './Lab Scenes.v2-2026_train_0120.yolov8',
        '--target_benchmark_root', './Eval_benchmark',
        '--source_checkpoint', './checkpoints_material_detection_new_data/checkpoint_best.pth',
        '--meta_epochs', '2',
        '--num_tasks', '2',
        '--shots_per_task', '5',
        '--query_size', '3',
        '--adapt_shots', '25',
        '--adapt_steps', '5',
        '--inner_lr', '1e-3',
        '--inner_steps', '5',
        '--meta_lr', '1e-4',
        '--batch_size', '4',
        '--first_order',
        '--output_dir', output_dir
    ]
    
    subprocess.run(train_cmd, check=True)
    
    # 评估
    print(f"\n2. 评估模型...")
    eval_cmd = [
        'python', 'evaluate_benchmark.py',
        '--checkpoint', f'{output_dir}/maml_adapted.pth',
        '--benchmark_root', './Eval_benchmark',
        '--batch_size', '8',
        '--score_threshold', '0.05',
        '--iou_threshold', '0.5',
        '--output', result_file
    ]
    
    subprocess.run(eval_cmd, check=True)
    
    # 加载结果
    try:
        with open(result_file, 'r') as f:
            result = json.load(f)
            results.append(result['overall'])
            print(f"✅ Run {run_id+1} 完成: Recall={result['overall']['recall']:.4f}, TP={result['overall']['tp']}")
    except Exception as e:
        print(f"⚠️  Run {run_id+1} 结果加载失败: {e}")

# 计算平均结果
if results:
    print(f"\n{'='*80}")
    print("平均结果统计")
    print(f"{'='*80}")
    
    avg_recall = np.mean([r['recall'] for r in results])
    std_recall = np.std([r['recall'] for r in results])
    avg_tp = np.mean([r['tp'] for r in results])
    std_tp = np.std([r['tp'] for r in results])
    avg_precision = np.mean([r['precision'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])
    avg_cls_acc = np.mean([r['classification_accuracy'] for r in results])
    
    print(f"\n总体指标:")
    print(f"  Recall: {avg_recall:.4f} ± {std_recall:.4f}")
    print(f"  TP: {avg_tp:.1f} ± {std_tp:.1f}")
    print(f"  Precision: {avg_precision:.4f}")
    print(f"  F1-Score: {avg_f1:.4f}")
    print(f"  分类准确率: {avg_cls_acc*100:.2f}%")
    
    print(f"\n各次运行结果:")
    for i, r in enumerate(results):
        print(f"  Run {i+1}: Recall={r['recall']:.4f}, TP={r['tp']}")
    
    # 保存平均结果
    avg_result = {
        'average': {
            'recall': float(avg_recall),
            'recall_std': float(std_recall),
            'tp': float(avg_tp),
            'tp_std': float(std_tp),
            'precision': float(avg_precision),
            'f1': float(avg_f1),
            'classification_accuracy': float(avg_cls_acc)
        },
        'individual_runs': results,
        'num_runs': num_runs
    }
    
    with open('maml_5pct_averaged_results.json', 'w') as f:
        json.dump(avg_result, f, indent=2)
    
    print(f"\n💾 平均结果已保存到: maml_5pct_averaged_results.json")
else:
    print("\n⚠️  没有成功的结果可以平均")
