#!/usr/bin/env python3
"""
材料检测模型：ViT/Swin backbone + DETR-style transformer detection head
支持多视图cross-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Tuple
import warnings

try:
    from timm import create_model
    TIMM_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    TIMM_AVAILABLE = False
    warnings.warn(f"timm not available ({e}), will use custom ViT implementation")


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiViewCrossAttention(nn.Module):
    """多视图交叉注意力模块"""
    
    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, N_q, d_model] - 当前视图的查询
            key: [B, N_k, d_model] - 其他视图的键
            value: [B, N_k, d_model] - 其他视图的值
        Returns:
            [B, N_q, d_model] - 增强后的特征
        """
        # 交叉注意力
        attn_out, _ = self.cross_attn(query, key, value)
        
        # 残差连接和层归一化
        out = self.norm(query + self.dropout(attn_out))
        return out


class ViTBackbone(nn.Module):
    """ViT backbone（使用timm或自定义实现）"""
    
    def __init__(self, model_name: str = 'vit_base_patch16_224', 
                 img_size: int = 224,
                 pretrained: bool = False,
                 embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        
        if TIMM_AVAILABLE:
            try:
                self.backbone = create_model(
                    model_name,
                    pretrained=pretrained,
                    img_size=img_size,
                    num_classes=0  # 移除分类头
                )
                # 获取实际的embed_dim
                if hasattr(self.backbone, 'embed_dim'):
                    self.embed_dim = self.backbone.embed_dim
                elif hasattr(self.backbone, 'num_features'):
                    self.embed_dim = self.backbone.num_features
            except Exception as e:
                warnings.warn(f"Failed to load timm model {model_name}: {e}, using custom ViT")
                self._build_custom_vit(embed_dim)
        else:
            self._build_custom_vit(embed_dim)
    
    def _build_custom_vit(self, embed_dim: int):
        """构建自定义ViT"""
        patch_size = 16
        num_patches = (self.img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(
            3, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=12,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, N, embed_dim] - 特征序列（包含CLS token）
        """
        if TIMM_AVAILABLE and hasattr(self.backbone, 'forward_features'):
            # 使用timm模型
            features = self.backbone.forward_features(x)
            if isinstance(features, tuple):
                features = features[0]
            return features
        else:
            # 自定义ViT
            B = x.size(0)
            x = self.patch_embed(x)  # [B, embed_dim, H', W']
            x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
            
            # 添加CLS token
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
            # 添加位置编码
            x = x + self.pos_embed
            
            # Transformer编码
            x = self.transformer(x)
            x = self.norm(x)
            return x


class DETRDetectionHead(nn.Module):
    """DETR风格的检测头"""
    
    def __init__(self, 
                 d_model: int = 768,
                 num_queries: int = 5,
                 num_classes: int = 4,
                 nhead: int = 8,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        # 查询嵌入（可学习的对象查询）
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=1000)
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # 预测头
        # num_classes + 1: 4个已知类 + background（移除unknown类）
        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 for background
        self.bbox_embed = MLP(d_model, d_model, 4, 3)  # 4个坐标值
        
    def forward(self, encoder_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            encoder_features: [B, N, d_model] - 编码器特征
        Returns:
            dict with 'pred_logits' and 'pred_boxes'
        """
        B = encoder_features.size(0)
        
        # 准备查询
        query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, d_model]
        query_embed = self.pos_encoder(query_embed)
        
        # 解码
        decoder_output = self.decoder(query_embed, encoder_features)  # [B, num_queries, d_model]
        
        # 预测
        pred_logits = self.class_embed(decoder_output)  # [B, num_queries, num_classes+2] (4类+unknown+background)
        pred_boxes = self.bbox_embed(decoder_output).sigmoid()  # [B, num_queries, 4] (cxcywh格式，归一化)
        
        return {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes,
            'decoder_features': decoder_output,
        }


class MLP(nn.Module):
    """简单的MLP"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MaterialDetectionModel(nn.Module):
    """材料检测模型：ViT backbone + DETR head + 多视图cross-attention"""
    
    def __init__(self,
                 backbone_name: str = 'vit_base_patch16_224',
                 img_size: int = 224,
                 num_classes: int = 4,
                 num_queries: int = 5,
                 d_model: int = 768,
                 num_decoder_layers: int = 6,
                 use_multi_view: bool = True,
                 num_views: int = 3,
                 pretrained_backbone: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.use_multi_view = use_multi_view
        self.num_views = num_views
        
        # Backbone
        self.backbone = ViTBackbone(
            model_name=backbone_name,
            img_size=img_size,
            pretrained=pretrained_backbone,
            embed_dim=d_model
        )
        self.embed_dim = self.backbone.embed_dim
        
        # 多视图交叉注意力（可选）
        if use_multi_view and num_views > 1:
            self.multi_view_attn = nn.ModuleList([
                MultiViewCrossAttention(self.embed_dim, nhead=8)
                for _ in range(num_views - 1)
            ])
        else:
            self.multi_view_attn = None
        
        # 特征融合（如果有多个视图）
        if use_multi_view and num_views > 1:
            self.view_fusion = nn.Linear(self.embed_dim * num_views, self.embed_dim)
        
        # DETR检测头
        self.detection_head = DETRDetectionHead(
            d_model=self.embed_dim,
            num_queries=num_queries,
            num_classes=num_classes,
            num_decoder_layers=num_decoder_layers
        )
        
        # Projection head for contrastive learning (768 → 128)
        self.projection_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 128),
        )
    
    def forward(self, images: torch.Tensor, view_masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [B, num_views, C, H, W] 或 [B, C, H, W] (单视图)
            view_masks: [B, num_views] - 可选，指示哪些视图有效
        Returns:
            dict with 'pred_logits' and 'pred_boxes'
        """
        if images.dim() == 4:
            # 单视图模式
            encoder_features = self.backbone(images)  # [B, N, d_model]
            out = self.detection_head(encoder_features)
            out['proj_features'] = F.normalize(
                self.projection_head(out['decoder_features']), dim=-1
            )  # [B, Q, 128] L2-normalized
            return out
        
        # 多视图模式
        B, num_views, C, H, W = images.shape
        
        # 提取每个视图的特征
        view_features = []
        for v in range(num_views):
            view_img = images[:, v]  # [B, C, H, W]
            features = self.backbone(view_img)  # [B, N, d_model]
            view_features.append(features)
        
        # 多视图交叉注意力
        if self.use_multi_view and self.multi_view_attn is not None and num_views > 1:
            enhanced_features = [view_features[0]]  # 第一个视图作为主视图
            
            for v in range(1, num_views):
                # 当前视图作为query，其他视图作为key和value
                query = view_features[v]
                key_value = view_features[0]  # 使用主视图
                
                # 交叉注意力
                enhanced = self.multi_view_attn[v - 1](query, key_value, key_value)
                enhanced_features.append(enhanced)
            
            # 融合多视图特征
            if num_views > 1:
                # 使用主视图的特征，或者融合所有视图
                fused_features = torch.cat(enhanced_features, dim=-1)  # [B, N, d_model * num_views]
                encoder_features = self.view_fusion(fused_features)  # [B, N, d_model]
            else:
                encoder_features = enhanced_features[0]
        else:
            # 只使用第一个视图
            encoder_features = view_features[0]
        
        # 检测头
        return self.detection_head(encoder_features)


if __name__ == '__main__':
    # 测试模型
    model = MaterialDetectionModel(
        backbone_name='vit_base_patch16_224',
        img_size=224,
        num_classes=4,
        num_queries=5,
        use_multi_view=True,
        num_views=3
    )
    
    # 单视图测试
    x_single = torch.randn(2, 3, 224, 224)
    out_single = model(x_single)
    print(f"单视图输出形状: {out_single['pred_logits'].shape}, {out_single['pred_boxes'].shape}")
    
    # 多视图测试
    x_multi = torch.randn(2, 3, 3, 224, 224)
    out_multi = model(x_multi)
    print(f"多视图输出形状: {out_multi['pred_logits'].shape}, {out_multi['pred_boxes'].shape}")

