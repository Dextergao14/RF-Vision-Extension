#!/usr/bin/env python3
"""
诊断benchmark评估结果全为0的问题
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from evaluate_benchmark import BenchmarkDataset
from material_dataset import collate_fn
from material_detection_model import MaterialDetectionModel

def diagnose_benchmark(checkpoint_path, benchmark_root, score_threshold=0.05):
    """诊断benchmark评估问题"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("诊断Benchmark评估问题")
    print("="*60)
    
    # 加载checkpoint
    print(f"\n1. 加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Val Loss: {checkpoint.get('val_loss', 'N/A')}")
    
    # 检查模型架构
    model_state = checkpoint.get('model_state_dict', {})
    class_embed_key = None
    for key in model_state.keys():
        if 'class_embed' in key and 'weight' in key:
            class_embed_key = key
            break
    
    if class_embed_key:
        output_dim = model_state[class_embed_key].shape[0]
        print(f"   模型输出维度: {output_dim}")
        if output_dim == 5:
            print("   ⚠️  旧版本模型（4类+背景），不支持unknown")
        elif output_dim == 6:
            print("   ✅ 新版本模型（4类+unknown+背景）")
    
    # 创建模型
    print(f"\n2. 创建模型...")
    model = MaterialDetectionModel(
        backbone_name='vit_base_patch16_224',
        img_size=224,
        num_classes=4,
        num_queries=100,
        num_decoder_layers=6
    ).to(device)
    
    try:
        model.load_state_dict(model_state)
        print("   ✅ 模型加载成功")
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        return
    
    # 加载数据集
    print(f"\n3. 加载benchmark数据集...")
    dataset = BenchmarkDataset(
        benchmark_root=benchmark_root,
        img_size=224,
        num_views=1,
        use_multi_view=False
    )
    print(f"   数据集大小: {len(dataset)}")
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    # 检查前几个样本
    print(f"\n4. 检查模型输出（前5个样本）...")
    model.eval()
    
    all_scores = []
    all_max_probs = []
    valid_predictions = 0
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            if i >= 5:
                break
            
            images = images.to(device)
            outputs = model(images)
            
            pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes+2]
            pred_boxes = outputs['pred_boxes'][0]
            
            probs = F.softmax(pred_logits, dim=-1)
            
            # 检查输出维度
            if i == 0:
                print(f"   pred_logits形状: {pred_logits.shape}")
                print(f"   probs形状: {probs.shape}")
            
            # 获取最高概率
            max_probs, max_labels = probs.max(dim=-1)
            all_max_probs.extend(max_probs.cpu().tolist())
            
            # 检查有多少预测通过阈值
            if output_dim == 6:
                # 新版本：排除background
                scores_all, labels_all = probs[:, :-1].max(dim=-1)
            else:
                # 旧版本：排除background
                scores_all, labels_all = probs[:, :-1].max(dim=-1)
            
            all_scores.extend(scores_all.cpu().tolist())
            valid_count = (scores_all > score_threshold).sum().item()
            valid_predictions += valid_count
            
            gt_count = len(targets[0]['labels'])
            print(f"   样本{i+1}: GT数量={gt_count}, 通过阈值预测数={valid_count}/{len(scores_all)}, "
                  f"最高概率={max_probs.max().item():.4f}, 平均概率={max_probs.mean().item():.4f}")
    
    print(f"\n5. 统计信息:")
    print(f"   总预测数: {len(all_scores)}")
    print(f"   通过阈值(>{score_threshold})的预测数: {sum(1 for s in all_scores if s > score_threshold)}")
    print(f"   最高分数: {max(all_scores):.4f}")
    print(f"   最低分数: {min(all_scores):.4f}")
    print(f"   平均分数: {sum(all_scores)/len(all_scores):.4f}")
    print(f"   中位数分数: {sorted(all_scores)[len(all_scores)//2]:.4f}")
    
    # 分数分布
    print(f"\n6. 分数分布:")
    thresholds = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
    for thresh in thresholds:
        count = sum(1 for s in all_scores if s > thresh)
        print(f"   分数 > {thresh}: {count} ({count/len(all_scores)*100:.1f}%)")
    
    print("\n" + "="*60)
    print("诊断建议:")
    print("="*60)
    
    max_score = max(all_scores)
    if max_score < score_threshold:
        print(f"⚠️  问题：所有预测的分数都低于阈值 {score_threshold}")
        print(f"   建议：降低score_threshold到 {max_score * 0.9:.4f} 或更低")
    elif sum(1 for s in all_scores if s > score_threshold) == 0:
        print(f"⚠️  问题：没有预测通过阈值")
        print(f"   建议：检查模型是否训练好，或降低score_threshold")
    else:
        print(f"✅ 有 {sum(1 for s in all_scores if s > score_threshold)} 个预测通过阈值")
        print(f"   如果结果仍为0，可能是IoU匹配失败（预测位置不准确）")

if __name__ == '__main__':
    import sys
    
    checkpoint_path = './checkpoints_material_detection/checkpoint_best.pth'
    benchmark_root = './Eval_benchmark'
    score_threshold = 0.05
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    if len(sys.argv) > 2:
        benchmark_root = sys.argv[2]
    if len(sys.argv) > 3:
        score_threshold = float(sys.argv[3])
    
    diagnose_benchmark(checkpoint_path, benchmark_root, score_threshold)
