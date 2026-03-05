#!/usr/bin/env python3
"""
对比不同MAML配置的结果
"""

import json
from pathlib import Path

print("="*80)
print("MAML域适应配置对比")
print("="*80)

# 加载结果
results = {}

# 4% support set (20张图像)
try:
    with open('maml_quick_test_final_results.json', 'r') as f:
        results['4% (20张)'] = json.load(f)['overall']
except:
    print("⚠️  无法加载4%配置结果")

# 5% support set (25张图像)
try:
    with open('maml_5pct_benchmark_results.json', 'r') as f:
        results['5% (25张)'] = json.load(f)['overall']
except:
    print("⚠️  无法加载5%配置结果")

# Baseline
baseline = {
    'recall': 0.0,
    'tp': 0,
    'fp': 24398,
    'fn': 1997
}

print("\n配置对比:")
print("-"*80)
print(f"{'配置':<20} {'Recall':<12} {'TP':<8} {'FP':<12} {'FN':<8} {'分类准确率':<12}")
print("-"*80)

print(f"{'Baseline (未适应)':<20} {baseline['recall']:<12.4f} {baseline['tp']:<8} {baseline['fp']:<12} {baseline['fn']:<8} {'0.00%':<12}")

for config_name, result in results.items():
    recall = result.get('recall', 0)
    tp = result.get('tp', 0)
    fp = result.get('fp', 0)
    fn = result.get('fn', 0)
    cls_acc = result.get('classification_accuracy', 0)
    
    print(f"{config_name:<20} {recall:<12.4f} {tp:<8} {fp:<12} {fn:<8} {cls_acc*100:<11.2f}%")

print("-"*80)

# 分析
if len(results) >= 2:
    print("\n分析:")
    configs = list(results.keys())
    if len(configs) >= 2:
        r1 = results[configs[0]]
        r2 = results[configs[1]]
        
        recall_diff = r2['recall'] - r1['recall']
        tp_diff = r2['tp'] - r1['tp']
        
        print(f"从{configs[0]}到{configs[1]}:")
        print(f"  Recall变化: {recall_diff:+.4f} ({recall_diff*100:+.2f}%)")
        print(f"  TP变化: {tp_diff:+d}")
        
        if recall_diff > 0:
            print(f"  ✅ {configs[1]}表现更好")
        elif recall_diff < 0:
            print(f"  ⚠️  {configs[0]}表现更好")
        else:
            print(f"  ➡️  表现相似")

print("\n" + "="*80)
