#!/usr/bin/env python3
"""
材料检测数据集加载器
支持YOLO格式标签和多视图数据
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os
from collections import defaultdict
import random


class MaterialDetectionDataset(Dataset):
    """材料检测数据集"""
    
    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 img_size: int = 224,
                 num_views: int = 1,
                 use_multi_view: bool = False,
                 transform=None):
        """
        Args:
            data_root: 数据集根目录
            split: 'train', 'valid', 'test'
            img_size: 图像尺寸
            num_views: 每个样本使用的视图数量
            use_multi_view: 是否使用多视图
            transform: 数据增强
        """
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.num_views = num_views
        self.use_multi_view = use_multi_view
        self.transform = transform
        
        # 类别名称（只有4个已知类，移除unknown）
        self.class_names = ['Concrete', 'Glass', 'Metal', 'Wood']
        self.num_classes = len(self.class_names)  # 已知类数量
        
        # 加载数据
        self.image_dir = self.data_root / split / 'images'
        self.label_dir = self.data_root / split / 'labels'
        
        # 收集所有图像
        self.image_paths = sorted(list(self.image_dir.glob('*.jpg')))
        
        # 按场景分组（用于多视图）
        if use_multi_view:
            self.scene_groups = self._group_by_scene()
        else:
            self.scene_groups = None
        
        print(f"✅ 加载{split}集: {len(self.image_paths)} 张图像")
        if use_multi_view:
            print(f"   场景组数: {len(self.scene_groups)}")
    
    def _group_by_scene(self) -> Dict[str, List[Path]]:
        """按场景名称分组图像"""
        scene_groups = defaultdict(list)
        
        for img_path in self.image_paths:
            # 提取场景名称（去掉.rf.xxx后缀）
            scene_name = img_path.stem.split('.rf.')[0]
            scene_groups[scene_name].append(img_path)
        
        # 只保留有多个视图的场景
        scene_groups = {k: v for k, v in scene_groups.items() if len(v) > 1}
        return scene_groups
    
    def _load_yolo_label(self, label_path: Path) -> List[Tuple[int, float, float, float, float]]:
        """加载YOLO格式标签"""
        boxes = []
        
        if not label_path.exists():
            return boxes
        
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                boxes.append((class_id, x_center, y_center, width, height))
        
        return boxes
    
    def _load_image(self, img_path: Path) -> np.ndarray:
        """加载图像"""
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"无法加载图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """调整图像大小"""
        return cv2.resize(img, (self.img_size, self.img_size))
    
    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """归一化图像到[0, 1]"""
        return img.astype(np.float32) / 255.0
    
    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """转换为tensor"""
        # [H, W, C] -> [C, H, W]
        return torch.from_numpy(img).permute(2, 0, 1).float()
    
    def _yolo_to_xyxy(self, boxes: List[Tuple[int, float, float, float, float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """将YOLO格式转换为xyxy格式（归一化坐标）"""
        if not boxes:
            return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long)
        
        labels = []
        bboxes = []
        
        for class_id, x_center, y_center, width, height in boxes:
            # YOLO格式已经是归一化的
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # 确保在[0, 1]范围内
            x1 = max(0, min(1, x1))
            y1 = max(0, min(1, y1))
            x2 = max(0, min(1, x2))
            y2 = max(0, min(1, y2))
            
            labels.append(class_id)
            bboxes.append([x1, y1, x2, y2])
        
        return torch.tensor(bboxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        if self.use_multi_view and self.scene_groups:
            return len(self.scene_groups)
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.use_multi_view and self.scene_groups:
            # 多视图模式：返回同一场景的多个视图
            scene_name = list(self.scene_groups.keys())[idx]
            scene_images = self.scene_groups[scene_name]
            
            # 随机选择num_views个视图
            if len(scene_images) >= self.num_views:
                selected_images = random.sample(scene_images, self.num_views)
            else:
                # 如果视图不够，重复使用
                selected_images = scene_images + [scene_images[-1]] * (self.num_views - len(scene_images))
                selected_images = selected_images[:self.num_views]
            
            # 加载多个视图的图像
            images = []
            all_boxes = []
            all_labels = []
            
            for img_path in selected_images:
                # 加载图像
                img = self._load_image(img_path)
                img = self._resize_image(img)
                img = self._normalize_image(img)
                
                if self.transform:
                    img = self.transform(img)
                
                img = self._to_tensor(img)
                images.append(img)
                
                # 加载标签（使用第一个视图的标签，或者合并所有视图的标签）
                label_path = self.label_dir / (img_path.stem.split('.rf.')[0] + '.txt')
                # 尝试找到对应的标签文件
                label_files = list(self.label_dir.glob(f"{img_path.stem.split('.rf.')[0]}*.txt"))
                if label_files:
                    label_path = label_files[0]
                
                boxes = self._load_yolo_label(label_path)
                bboxes, labels = self._yolo_to_xyxy(boxes)
                all_boxes.append(bboxes)
                all_labels.append(labels)
            
            # 堆叠图像
            images = torch.stack(images, dim=0)  # [num_views, C, H, W]
            
            # 使用第一个视图的标签（或者可以合并）
            target = {
                'boxes': all_boxes[0],  # [N, 4]
                'labels': all_labels[0],  # [N]
                'image_id': torch.tensor(idx),
            }
            
            return images, target
        else:
            # 单视图模式
            img_path = self.image_paths[idx]
            
            # 加载图像
            img = self._load_image(img_path)
            img = self._resize_image(img)
            img = self._normalize_image(img)
            
            if self.transform:
                img = self.transform(img)
            
            img = self._to_tensor(img)
            
            # 加载标签
            label_path = self.label_dir / (img_path.stem.split('.rf.')[0] + '.txt')
            # 尝试找到对应的标签文件
            label_files = list(self.label_dir.glob(f"{img_path.stem.split('.rf.')[0]}*.txt"))
            if label_files:
                label_path = label_files[0]
            
            boxes = self._load_yolo_label(label_path)
            bboxes, labels = self._yolo_to_xyxy(boxes)
            
            target = {
                'boxes': bboxes,  # [N, 4]
                'labels': labels,  # [N]
                'image_id': torch.tensor(idx),
            }
            
            return img, target


def collate_fn(batch):
    """自定义collate函数，处理不同数量的目标"""
    images = []
    targets = []
    
    for item in batch:
        if isinstance(item[0], torch.Tensor) and item[0].dim() == 3:
            # 单视图
            images.append(item[0])
        else:
            # 多视图
            images.append(item[0])
        targets.append(item[1])
    
    # 堆叠图像
    if images[0].dim() == 3:
        images = torch.stack(images, dim=0)
    else:
        images = torch.stack(images, dim=0)  # [B, num_views, C, H, W] 或 [B, C, H, W]
    
    return images, targets


if __name__ == '__main__':
    # 测试数据集
    dataset = MaterialDetectionDataset(
        data_root='./vanilla-dataset',
        split='train',
        img_size=224,
        num_views=3,
        use_multi_view=True
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 测试加载
    img, target = dataset[0]
    print(f"图像形状: {img.shape}")
    print(f"目标boxes: {target['boxes'].shape}")
    print(f"目标labels: {target['labels'].shape}")




