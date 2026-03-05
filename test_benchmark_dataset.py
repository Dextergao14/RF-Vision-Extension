#!/usr/bin/env python3
"""
快速测试benchmark数据集是否能正确加载
"""

import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from evaluate_benchmark import BenchmarkDataset
from torch.utils.data import DataLoader
from material_dataset import collate_fn

def test_benchmark_dataset():
    """测试benchmark数据集加载"""
    benchmark_root = './Eval_benchmark'
    
    print("="*60)
    print("测试Benchmark数据集加载")
    print("="*60)
    
    try:
        # 创建数据集
        dataset = BenchmarkDataset(
            benchmark_root=benchmark_root,
            img_size=224,
            num_views=1,
            use_multi_view=False
        )
        
        print(f"\n✅ 数据集创建成功")
        print(f"   数据集大小: {len(dataset)}")
        print(f"   图像目录: {dataset.image_dir}")
        print(f"   标签目录: {dataset.label_dir}")
        
        # 测试加载一个样本
        print(f"\n测试加载第一个样本...")
        img, target = dataset[0]
        
        print(f"✅ 样本加载成功")
        print(f"   图像形状: {img.shape}")
        print(f"   目标boxes数量: {len(target['boxes'])}")
        print(f"   目标labels数量: {len(target['labels'])}")
        
        if len(target['boxes']) > 0:
            print(f"   第一个box: {target['boxes'][0]}")
            print(f"   第一个label: {target['labels'][0].item()} (类别: {dataset.class_names[target['labels'][0].item()]})")
        
        # 测试DataLoader
        print(f"\n测试DataLoader...")
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # 使用0避免多进程问题
            collate_fn=collate_fn
        )
        
        images, targets = next(iter(loader))
        print(f"✅ DataLoader测试成功")
        print(f"   Batch图像形状: {images.shape}")
        print(f"   Batch targets数量: {len(targets)}")
        
        print("\n" + "="*60)
        print("✅ 所有测试通过！Benchmark数据集可以正常使用")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_benchmark_dataset()
    sys.exit(0 if success else 1)
