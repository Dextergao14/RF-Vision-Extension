#!/usr/bin/env python3
"""
快速测试训练流程
"""

import torch
from torch.utils.data import DataLoader
from material_detection_model import MaterialDetectionModel
from material_dataset import MaterialDetectionDataset, collate_fn
from train_material_detection import HungarianMatcher, SetCriterion

def test_training():
    """测试训练流程"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据集
    dataset = MaterialDetectionDataset(
        data_root='./vanilla-dataset',
        split='train',
        img_size=224,
        num_views=1,
        use_multi_view=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # 模型
    model = MaterialDetectionModel(
        backbone_name='vit_base_patch16_224',
        img_size=224,
        num_classes=4,
        num_queries=100,
        use_multi_view=False
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 损失函数
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    weight_dict = {'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}
    criterion = SetCriterion(4, matcher, weight_dict, eos_coef=0.1).to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 测试一个batch
    print("\n测试训练流程...")
    model.train()
    for batch_idx, (images, targets) in enumerate(dataloader):
        if batch_idx >= 2:  # 只测试2个batch
            break
        
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        loss_dict = criterion(outputs, targets)
        loss = sum(loss_dict.values())
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        print(f"Batch {batch_idx + 1}:")
        print(f"  损失: {loss.item():.4f}")
        print(f"  详细损失: {loss_dict}")
        print(f"  输出形状 - 类别: {outputs['pred_logits'].shape}, 边界框: {outputs['pred_boxes'].shape}")
    
    print("\n✅ 训练流程测试通过！")

if __name__ == '__main__':
    test_training()




