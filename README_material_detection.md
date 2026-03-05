# 材料检测模型

基于 ViT/Swin backbone + DETR-style transformer detection head 的材料检测模型，支持多视图cross-attention。

## 模型架构

- **Backbone**: ViT (Vision Transformer) 或 Swin Transformer
- **Detection Head**: DETR-style transformer decoder
- **Multi-view Support**: 可选的多视图cross-attention机制

## 数据集

数据集位于 `./vanilla-dataset`，包含：
- 训练集：`train/images` 和 `train/labels`
- 验证集：`valid/images` 和 `valid/labels`
- 测试集：`test/images` 和 `test/labels`

类别：
- 0: Concrete (混凝土)
- 1: Glass (玻璃)
- 2: Metal (金属)
- 3: Wood (木材)

标签格式：YOLO格式（归一化坐标）

## 安装依赖

```bash
pip install torch torchvision
pip install timm  # 可选，用于预训练backbone
pip install scipy  # 用于匈牙利算法
pip install opencv-python
pip install tqdm
```

## 使用方法

### 1. 单视图训练

```bash
python train_material_detection.py \
    --data_root ./vanilla-dataset \
    --backbone vit_base_patch16_224 \
    --img_size 224 \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 1e-4 \
    --num_queries 100 \
    --save_dir ./checkpoints_material_detection
```

### 2. 多视图训练（推荐）

```bash
python train_material_detection.py \
    --data_root ./vanilla-dataset \
    --backbone vit_base_patch16_224 \
    --img_size 224 \
    --batch_size 4 \
    --num_epochs 100 \
    --lr 1e-4 \
    --num_queries 100 \
    --use_multi_view \
    --num_views 3 \
    --save_dir ./checkpoints_material_detection
```

### 3. 使用预训练backbone

```bash
python train_material_detection.py \
    --data_root ./vanilla-dataset \
    --backbone vit_base_patch16_224 \
    --pretrained \
    --img_size 224 \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 1e-4 \
    --save_dir ./checkpoints_material_detection
```

### 4. 恢复训练

```bash
python train_material_detection.py \
    --data_root ./vanilla-dataset \
    --resume ./checkpoints_material_detection/checkpoint_latest.pth \
    --batch_size 8 \
    --num_epochs 100
```

## 参数说明

- `--data_root`: 数据集根目录
- `--backbone`: Backbone模型名称（如 `vit_base_patch16_224`, `swin_base_patch4_window7_224`）
- `--img_size`: 输入图像尺寸（默认224）
- `--batch_size`: 批次大小（默认8）
- `--num_epochs`: 训练轮数（默认100）
- `--lr`: 学习率（默认1e-4）
- `--weight_decay`: 权重衰减（默认1e-4）
- `--num_queries`: DETR查询数量（默认100）
- `--num_decoder_layers`: Transformer解码器层数（默认6）
- `--use_multi_view`: 启用多视图cross-attention
- `--num_views`: 每个样本使用的视图数量（默认3）
- `--pretrained`: 使用预训练backbone
- `--save_dir`: 模型保存目录
- `--resume`: 恢复训练的检查点路径

## 模型文件

- `material_detection_model.py`: 模型定义
- `material_dataset.py`: 数据集加载器
- `train_material_detection.py`: 训练脚本

## 输出

训练过程中会保存：
- `checkpoint_latest.pth`: 最新检查点
- `checkpoint_best.pth`: 最佳验证损失模型
- `checkpoint_epoch_N.pth`: 每10个epoch的检查点

## 损失函数

模型使用DETR的集合预测损失：
- **分类损失** (Cross-Entropy): 预测类别
- **边界框损失** (L1): 边界框坐标
- **GIoU损失**: 边界框重叠度

总损失 = 1.0 × 分类损失 + 5.0 × 边界框损失 + 2.0 × GIoU损失

## 多视图机制

当启用 `--use_multi_view` 时：
1. 模型会自动按场景名称分组图像
2. 每个场景的多个视图会通过cross-attention机制融合
3. 主视图（第一个视图）作为key和value，其他视图作为query
4. 融合后的特征输入到DETR检测头

## 注意事项

1. 如果GPU内存不足，可以减小 `batch_size` 或 `img_size`
2. 多视图训练需要更多内存，建议减小batch_size
3. 首次运行会下载预训练模型（如果使用 `--pretrained`）
4. 训练时间取决于数据集大小和硬件配置

## 测试模型

```python
import torch
from material_detection_model import MaterialDetectionModel

# 加载模型
model = MaterialDetectionModel(
    backbone_name='vit_base_patch16_224',
    img_size=224,
    num_classes=4,
    num_queries=100,
    use_multi_view=True,
    num_views=3
)

# 单视图推理
x = torch.randn(1, 3, 224, 224)
outputs = model(x)
print(f"预测类别: {outputs['pred_logits'].shape}")
print(f"预测边界框: {outputs['pred_boxes'].shape}")

# 多视图推理
x_multi = torch.randn(1, 3, 3, 224, 224)  # [B, num_views, C, H, W]
outputs = model(x_multi)
print(f"多视图预测类别: {outputs['pred_logits'].shape}")
print(f"多视图预测边界框: {outputs['pred_boxes'].shape}")
```




