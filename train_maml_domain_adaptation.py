#!/usr/bin/env python3
"""
MAML (Model-Agnostic Meta-Learning) 域适应训练
基于 Finn et al. ICML 2017 的 MAML 方法
适用于 few-shot domain adaptation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from copy import deepcopy
import math

from material_detection_model import MaterialDetectionModel
from material_dataset import MaterialDetectionDataset, collate_fn
from evaluate_benchmark import BenchmarkDataset
from train_material_detection import HungarianMatcher, SetCriterion


class MAML:
    """
    MAML (Model-Agnostic Meta-Learning) 实现
    用于 few-shot domain adaptation
    """
    
    def __init__(self, model, inner_lr=1e-3, inner_steps=5, 
                 first_order=False, device='cuda'):
        """
        Args:
            model: 要训练的模型
            inner_lr: 内层循环学习率（快速适应时的学习率）
            inner_steps: 内层循环步数（快速适应时的梯度步数）
            first_order: 是否使用一阶近似（False时使用二阶梯度，更准确但更慢）
            device: 设备
        """
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.device = device
        
    def fast_adapt(self, support_data, criterion):
        """
        快速适应：在support data上执行内层循环
        
        Args:
            support_data: 支持集数据（用于快速适应的少量数据）
            criterion: 损失函数
        
        Returns:
            adapted_params: 适应后的参数（按顺序的列表）
            loss: 适应后的损失
        """
        # 保存原始参数（用于后续恢复）
        original_params = {n: p.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        
        # 创建参数的副本用于内层循环
        adapted_params = {n: p.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        
        # 执行内层循环的多个梯度步
        total_loss = 0.0
        for step in range(self.inner_steps):
            # 设置模型参数为当前适应后的参数
            param_idx = 0
            adapted_param_list = list(adapted_params.values())
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = adapted_param_list[param_idx].data.clone()
                    param_idx += 1
            
            # 前向传播
            step_loss = 0.0
            for images, targets in support_data:
                images = images.to(self.device)
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                outputs = self.model(images)
                loss_dict = criterion(outputs, targets)
                loss = sum(loss_dict.values())
                step_loss += loss
            
            if step_loss > 0:
                # 反向传播（计算梯度）
                step_loss.backward()
                
                # 手动更新参数（梯度下降）
                param_idx = 0
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        adapted_params[name] = adapted_params[name] - self.inner_lr * param.grad
                        param_idx += 1
                
                # 清零梯度
                self.model.zero_grad()
            
            total_loss = step_loss.item() if step_loss > 0 else 0.0
        
        # 恢复原始参数（保持模型状态不变）
        for name, param in self.model.named_parameters():
            if name in original_params:
                param.data = original_params[name].data.clone()
        
        # 返回适应后的参数（按顺序）
        adapted_param_list = [adapted_params[n] for n, p in self.model.named_parameters() if p.requires_grad]
        
        return adapted_param_list, total_loss
    
    def meta_update(self, meta_batch, criterion, meta_optimizer):
        """
        元更新：在外层循环中更新模型参数
        
        Args:
            meta_batch: 元批次（包含多个任务的数据）
            criterion: 损失函数
            meta_optimizer: 元优化器
        
        Returns:
            meta_loss: 元损失
        """
        meta_optimizer.zero_grad()
        
        # 保存原始参数
        original_params = {n: p.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        
        # 为每个任务计算梯度
        task_losses = []
        
        for task_data in meta_batch:
            # 恢复原始参数（每个任务都从原始参数开始）
            for name, param in self.model.named_parameters():
                if name in original_params:
                    param.data = original_params[name].data.clone()
            
            support_data = task_data['support']
            query_data = task_data['query']
            
            # 内层循环：快速适应
            adapted_params, _ = self.fast_adapt(support_data, criterion)
            
            # 在query data上评估适应后的模型
            query_loss = 0.0
            
            # 设置模型参数为适应后的参数
            adapted_param_idx = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = adapted_params[adapted_param_idx].data.clone()
                    adapted_param_idx += 1
            
            # 计算query loss（需要梯度）
            for images, targets in query_data:
                images = images.to(self.device)
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                outputs = self.model(images)
                loss_dict = criterion(outputs, targets)
                loss = sum(loss_dict.values())
                query_loss += loss
            
            task_losses.append(query_loss)
        
        # 恢复原始参数（用于计算梯度）
        for name, param in self.model.named_parameters():
            if name in original_params:
                param.data = original_params[name].data.clone()
        
        # 聚合所有任务的损失
        meta_loss = sum(task_losses) / len(task_losses) if task_losses else torch.tensor(0.0, device=self.device)
        
        # 反向传播（计算二阶梯度）
        meta_loss.backward()
        
        return meta_loss.item()


def create_meta_tasks(source_dataset, num_tasks=5, shots_per_task=10, 
                     query_size=5, batch_size=4):
    """
    创建元学习任务
    将源域数据分成多个任务，模拟多个域
    
    Args:
        source_dataset: 源域数据集
        num_tasks: 任务数量
        shots_per_task: 每个任务的support样本数
        query_size: 每个任务的query样本数
        batch_size: 批次大小
    
    Returns:
        meta_tasks: 元任务列表
    """
    total_size = len(source_dataset)
    samples_per_task = shots_per_task + query_size
    
    if total_size < num_tasks * samples_per_task:
        # 如果数据不够，减少任务数或重复采样
        num_tasks = total_size // samples_per_task
        if num_tasks == 0:
            num_tasks = 1
            shots_per_task = min(shots_per_task, total_size // 2)
            query_size = total_size - shots_per_task
    
    # 随机打乱数据
    indices = torch.randperm(total_size).tolist()
    
    meta_tasks = []
    for i in range(num_tasks):
        start_idx = i * samples_per_task
        end_idx = start_idx + samples_per_task
        
        if end_idx > len(indices):
            break
        
        task_indices = indices[start_idx:end_idx]
        support_indices = task_indices[:shots_per_task]
        query_indices = task_indices[shots_per_task:]
        
        support_subset = Subset(source_dataset, support_indices)
        query_subset = Subset(source_dataset, query_indices)
        
        support_loader = DataLoader(
            support_subset, batch_size=batch_size, shuffle=True,
            num_workers=0, collate_fn=collate_fn, pin_memory=False
        )
        
        query_loader = DataLoader(
            query_subset, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_fn, pin_memory=False
        )
        
        meta_tasks.append({
            'support': support_loader,
            'query': query_loader
        })
    
    return meta_tasks


def train_maml_meta_learning(model, source_dataset, device, args):
    """
    MAML元训练阶段：在源域数据上训练模型使其能快速适应
    
    Args:
        model: 模型
        source_dataset: 源域数据集
        device: 设备
        args: 参数
    
    Returns:
        trained_model: 训练后的模型
    """
    print("\n" + "="*80)
    print("MAML 元训练阶段")
    print("="*80)
    
    # 创建MAML实例
    maml = MAML(
        model=model,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        first_order=args.first_order,
        device=device
    )
    
    # 损失函数
    matcher = HungarianMatcher(
        cost_class=args.cost_class,
        cost_bbox=args.cost_bbox,
        cost_giou=args.cost_giou
    )
    weight_dict = {
        'loss_ce': args.weight_ce,
        'loss_bbox': args.weight_bbox,
        'loss_giou': args.weight_giou
    }
    criterion = SetCriterion(
        num_classes=4,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef
    ).to(device)
    
    # 元优化器
    meta_optimizer = optim.AdamW(
        model.parameters(),
        lr=args.meta_lr,
        weight_decay=args.weight_decay
    )
    
    # 训练循环
    print(f"\n开始MAML元训练...")
    print(f"  内层学习率: {args.inner_lr}")
    print(f"  内层步数: {args.inner_steps}")
    print(f"  元学习率: {args.meta_lr}")
    print(f"  任务数: {args.num_tasks}")
    print(f"  每个任务support样本数: {args.shots_per_task}")
    
    for epoch in range(args.meta_epochs):
        print(f"\nEpoch {epoch+1}/{args.meta_epochs}")
        print("-" * 80)
        
        # 创建元任务批次
        meta_tasks = create_meta_tasks(
            source_dataset,
            num_tasks=args.num_tasks,
            shots_per_task=args.shots_per_task,
            query_size=args.query_size,
            batch_size=args.batch_size
        )
        
        if len(meta_tasks) == 0:
            print("⚠️  无法创建足够的元任务，跳过此epoch")
            continue
        
        # 元更新
        meta_loss = maml.meta_update(meta_tasks, criterion, meta_optimizer)
        meta_optimizer.step()
        
        print(f"元损失: {meta_loss:.4f}")
        
        # 每几个epoch保存一次
        if (epoch + 1) % args.save_interval == 0:
            save_path = Path(args.output_dir) / f'maml_meta_epoch_{epoch+1}.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'meta_loss': meta_loss,
                'args': vars(args)
            }, save_path)
            print(f"💾 保存检查点: {save_path}")
    
    return model


def maml_fast_adapt_to_target(model, target_dataset, device, args):
    """
    MAML快速适应：在目标域上快速适应（few-shot）
    
    Args:
        model: MAML训练后的模型
        target_dataset: 目标域数据集
        device: 设备
        args: 参数
    
    Returns:
        adapted_model: 适应后的模型
    """
    print("\n" + "="*80)
    print("MAML 快速适应阶段（Few-shot Domain Adaptation）")
    print("="*80)
    
    # 分割目标域数据：support（用于适应）和 query（用于评估）
    target_size = len(target_dataset)
    support_size = min(args.adapt_shots, int(target_size * 0.1))  # 使用10%或指定数量
    query_size = target_size - support_size
    
    # 使用固定随机种子确保可重复性
    torch.manual_seed(42)
    indices = torch.randperm(target_size).tolist()
    support_indices = indices[:support_size]
    query_indices = indices[support_size:]
    
    support_subset = Subset(target_dataset, support_indices)
    query_subset = Subset(target_dataset, query_indices)
    
    support_loader = DataLoader(
        support_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=False
    )
    
    query_loader = DataLoader(
        query_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=False
    )
    
    print(f"目标域support集: {len(support_subset)} 张图像（用于快速适应）")
    print(f"目标域query集: {len(query_subset)} 张图像（用于评估）")
    
    # 损失函数
    matcher = HungarianMatcher(
        cost_class=args.cost_class,
        cost_bbox=args.cost_bbox,
        cost_giou=args.cost_giou
    )
    weight_dict = {
        'loss_ce': args.weight_ce,
        'loss_bbox': args.weight_bbox,
        'loss_giou': args.weight_giou
    }
    criterion = SetCriterion(
        num_classes=4,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef
    ).to(device)
    
    # 创建MAML实例（用于快速适应）
    maml = MAML(
        model=model,
        inner_lr=args.adapt_lr,
        inner_steps=args.adapt_steps,
        first_order=args.first_order,
        device=device
    )
    
    # 快速适应
    print(f"\n开始快速适应...")
    print(f"  适应学习率: {args.adapt_lr}")
    print(f"  适应步数: {args.adapt_steps}")
    
    adapted_params, adapt_loss = maml.fast_adapt(support_loader, criterion)
    
    # 应用适应后的参数
    adapted_param_idx = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = adapted_params[adapted_param_idx].data.clone()
            adapted_param_idx += 1
    
    print(f"✅ 快速适应完成，适应损失: {adapt_loss:.4f}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='MAML域适应训练')
    
    # 数据路径
    parser.add_argument('--source_data_root', type=str,
                       default='./Lab Scenes.v2-2026_train_0120.yolov8',
                       help='源域数据集根目录')
    parser.add_argument('--target_benchmark_root', type=str,
                       default='./Eval_benchmark',
                       help='目标域benchmark数据集根目录')
    parser.add_argument('--source_checkpoint', type=str,
                       default='./checkpoints_material_detection_new_data/checkpoint_best.pth',
                       help='源域预训练模型检查点（可选，如果提供则从此开始）')
    
    # MAML参数
    parser.add_argument('--inner_lr', type=float, default=1e-3,
                       help='内层循环学习率（快速适应时的学习率）')
    parser.add_argument('--inner_steps', type=int, default=5,
                       help='内层循环步数（快速适应时的梯度步数）')
    parser.add_argument('--meta_lr', type=float, default=1e-4,
                       help='元学习率（外层循环学习率）')
    parser.add_argument('--meta_epochs', type=int, default=10,
                       help='元训练轮数')
    parser.add_argument('--first_order', action='store_true',
                       help='使用一阶近似（更快但可能不够准确）')
    
    # 元任务参数
    parser.add_argument('--num_tasks', type=int, default=5,
                       help='每个epoch的元任务数量')
    parser.add_argument('--shots_per_task', type=int, default=10,
                       help='每个任务的support样本数')
    parser.add_argument('--query_size', type=int, default=5,
                       help='每个任务的query样本数')
    
    # 快速适应参数
    parser.add_argument('--adapt_shots', type=int, default=50,
                       help='目标域适应时的support样本数')
    parser.add_argument('--adapt_lr', type=float, default=1e-3,
                       help='目标域适应时的学习率')
    parser.add_argument('--adapt_steps', type=int, default=10,
                       help='目标域适应时的步数')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--save_interval', type=int, default=5, help='保存间隔')
    
    # 损失函数参数
    parser.add_argument('--cost_class', type=float, default=1.0)
    parser.add_argument('--cost_bbox', type=float, default=5.0)
    parser.add_argument('--cost_giou', type=float, default=2.0)
    parser.add_argument('--weight_ce', type=float, default=1.0)
    parser.add_argument('--weight_bbox', type=float, default=5.0)
    parser.add_argument('--weight_giou', type=float, default=2.0)
    parser.add_argument('--eos_coef', type=float, default=0.1)
    
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--output_dir', type=str, default='./checkpoints_maml_domain_adaptation')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载或创建模型
    print(f"\n📂 加载模型...")
    if args.source_checkpoint and Path(args.source_checkpoint).exists():
        checkpoint = torch.load(args.source_checkpoint, map_location=device, weights_only=False)
        
        if 'args' in checkpoint:
            model_args = checkpoint['args']
            backbone_name = model_args.get('backbone', 'vit_base_patch16_224')
            num_queries = model_args.get('num_queries', 100)
            num_decoder_layers = model_args.get('num_decoder_layers', 6)
            use_multi_view = model_args.get('use_multi_view', False)
            num_views = model_args.get('num_views', 3)
        else:
            backbone_name = 'vit_base_patch16_224'
            num_queries = 100
            num_decoder_layers = 6
            use_multi_view = False
            num_views = 3
        
        model = MaterialDetectionModel(
            backbone_name=backbone_name, img_size=args.img_size, num_classes=4,
            num_queries=num_queries, num_decoder_layers=num_decoder_layers,
            use_multi_view=use_multi_view, num_views=num_views,
            pretrained_backbone=False
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✅ 从检查点加载模型 (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        # 创建新模型
        model = MaterialDetectionModel(
            backbone_name='vit_base_patch16_224', img_size=args.img_size, num_classes=4,
            num_queries=100, num_decoder_layers=6,
            use_multi_view=False, num_views=1,
            pretrained_backbone=True
        ).to(device)
        print("✅ 创建新模型")
    
    # 加载数据集
    print("\n📊 加载数据集...")
    source_dataset = MaterialDetectionDataset(
        data_root=args.source_data_root, split='train',
        img_size=args.img_size, num_views=1, use_multi_view=False
    )
    target_dataset = BenchmarkDataset(
        benchmark_root=args.target_benchmark_root,
        img_size=args.img_size, num_views=1, use_multi_view=False
    )
    
    print(f"源域训练集: {len(source_dataset)} 张图像")
    print(f"目标域数据: {len(target_dataset)} 张图像")
    
    # 阶段1: MAML元训练
    meta_trained_model = train_maml_meta_learning(
        model, source_dataset, device, args
    )
    
    # 保存元训练后的模型
    meta_save_path = output_dir / 'maml_meta_trained.pth'
    torch.save({
        'model_state_dict': meta_trained_model.state_dict(),
        'meta_epochs': args.meta_epochs,
        'args': vars(args)
    }, meta_save_path)
    print(f"\n💾 MAML元训练模型已保存到: {meta_save_path}")
    
    # 阶段2: 快速适应到目标域
    adapted_model = maml_fast_adapt_to_target(
        meta_trained_model, target_dataset, device, args
    )
    
    # 保存适应后的模型
    adapt_save_path = output_dir / 'maml_adapted.pth'
    torch.save({
        'model_state_dict': adapted_model.state_dict(),
        'adapt_shots': args.adapt_shots,
        'adapt_steps': args.adapt_steps,
        'args': vars(args)
    }, adapt_save_path)
    print(f"\n💾 MAML适应模型已保存到: {adapt_save_path}")
    
    print("\n✅ MAML域适应训练完成！")
    print(f"\n   模型文件:")
    print(f"   - 元训练模型: {meta_save_path}")
    print(f"   - 适应后模型: {adapt_save_path}")
    print(f"\n   接下来可以使用 evaluate_benchmark.py 评估适应后的模型性能")


if __name__ == '__main__':
    main()
