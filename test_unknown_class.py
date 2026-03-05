#!/usr/bin/env python3
"""
测试unknown类功能
"""

import torch
import torch.nn.functional as F
from material_detection_model import MaterialDetectionModel

def test_model_output_shape():
    """测试模型输出形状"""
    print("=" * 60)
    print("测试1: 模型输出形状")
    print("=" * 60)
    
    model = MaterialDetectionModel(
        backbone_name='vit_base_patch16_224',
        img_size=224,
        num_classes=4,  # 已知类数量
        num_queries=100,
        use_multi_view=False
    )
    
    # 单视图测试
    x = torch.randn(2, 3, 224, 224)
    outputs = model(x)
    
    pred_logits = outputs['pred_logits']
    pred_boxes = outputs['pred_boxes']
    
    print(f"输入形状: {x.shape}")
    print(f"pred_logits形状: {pred_logits.shape}")
    print(f"pred_boxes形状: {pred_boxes.shape}")
    
    # 检查输出维度
    # pred_logits应该是 [B, num_queries, num_classes+2] = [2, 100, 6]
    expected_logits_shape = (2, 100, 6)  # 4类+unknown+background
    assert pred_logits.shape == expected_logits_shape, \
        f"pred_logits形状错误: 期望{expected_logits_shape}, 实际{pred_logits.shape}"
    
    print("✅ 模型输出形状正确")
    print()


def test_unknown_detection_logic():
    """测试unknown类检测逻辑"""
    print("=" * 60)
    print("测试2: Unknown类检测逻辑")
    print("=" * 60)
    
    # 模拟模型输出
    num_queries = 10
    pred_logits = torch.randn(num_queries, 6)  # 6类: 4已知+unknown+background
    
    # 计算概率
    probs = F.softmax(pred_logits, dim=-1)
    known_probs = probs[:, :4].sum(dim=-1)  # 已知类的总概率
    unknown_prob = probs[:, 4]  # unknown类的概率
    
    print(f"已知类概率范围: [{known_probs.min():.3f}, {known_probs.max():.3f}]")
    print(f"Unknown类概率范围: [{unknown_prob.min():.3f}, {unknown_prob.max():.3f}]")
    
    # 应用检测逻辑
    score_threshold = 0.05
    scores_all, labels_all = probs[:, :-1].max(dim=-1)  # 排除background
    
    # 如果unknown概率很高且已知类概率很低，使用unknown
    use_unknown_mask = (unknown_prob > score_threshold) & (known_probs < 0.3)
    labels = labels_all.clone()
    labels[use_unknown_mask] = 4  # unknown类ID
    scores = scores_all.clone()
    scores[use_unknown_mask] = unknown_prob[use_unknown_mask]
    
    # 过滤低分预测
    valid_mask = scores > score_threshold
    
    print(f"\n检测结果:")
    print(f"  总查询数: {num_queries}")
    print(f"  有效预测数: {valid_mask.sum().item()}")
    print(f"  Unknown预测数: {(labels[valid_mask] == 4).sum().item()}")
    
    # 显示每个查询的预测
    print(f"\n各查询的预测:")
    for i in range(num_queries):
        if valid_mask[i]:
            pred_class = labels[i].item()
            pred_score = scores[i].item()
            known_p = known_probs[i].item()
            unknown_p = unknown_prob[i].item()
            class_name = ['Concrete', 'Glass', 'Metal', 'Wood', 'Unknown'][pred_class]
            print(f"  查询{i}: {class_name} (分数={pred_score:.3f}, 已知类概率={known_p:.3f}, unknown概率={unknown_p:.3f})")
    
    print("✅ Unknown类检测逻辑测试完成")
    print()


def test_loss_function():
    """测试损失函数"""
    print("=" * 60)
    print("测试3: 损失函数")
    print("=" * 60)
    
    from train_material_detection import SetCriterion, HungarianMatcher
    
    # 创建损失函数
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    weight_dict = {'loss_ce': 2.0, 'loss_bbox': 5.0, 'loss_giou': 2.0, 'loss_energy': 0.1}
    criterion = SetCriterion(4, matcher, weight_dict, eos_coef=0.25, unknown_coef=0.5)
    
    # 模拟输出和目标
    batch_size = 2
    num_queries = 10
    outputs = {
        'pred_logits': torch.randn(batch_size, num_queries, 6),  # 6类
        'pred_boxes': torch.rand(2, num_queries, 4)  # cxcywh格式
    }
    
    targets = [
        {
            'boxes': torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]]),  # xyxy格式
            'labels': torch.tensor([0, 2])  # Concrete和Metal
        },
        {
            'boxes': torch.tensor([[0.2, 0.2, 0.4, 0.4]]),
            'labels': torch.tensor([1])  # Glass
        }
    ]
    
    # 计算损失
    losses = criterion(outputs, targets)
    
    print(f"损失值:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    
    # 检查损失是否包含energy_loss
    assert 'loss_energy' in losses, "损失函数应该包含loss_energy"
    print("✅ 损失函数测试完成")
    print()


if __name__ == '__main__':
    print("开始测试unknown类功能...\n")
    
    try:
        test_model_output_shape()
        test_unknown_detection_logic()
        test_loss_function()
        
        print("=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
