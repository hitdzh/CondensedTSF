"""
Transformer Encoder for mapping time series to high-dimensional feature space.

用于将时间序列数据映射到高维特征空间，以便在语义空间中进行数据选择。
"""

import torch
import torch.nn as nn
import numpy as np


class FeatureEncoder(nn.Module):
    """
    特征编码器：使用 Transformer Encoder 将时序数据映射到高维特征空间。

    架构:
        1. Token Embedding: 将输入映射到 d_model 维度
        2. Positional Encoding: 添加位置信息
        3. Transformer Encoder Layers: 多层自注意力
        4. Pooling: 聚合时序特征为固定维度表示
    """

    def __init__(
        self,
        c_in: int = 21,          # 输入特征维度
        d_model: int = 256,      # 模型维度
        n_heads: int = 8,        # 注意力头数
        e_layers: int = 2,       # 编码器层数
        d_ff: int = 512,         # 前馈网络维度
        dropout: float = 0.1,    # Dropout 率
        activation: str = 'gelu', # 激活函数
        pooling: str = 'mean'    # 池化方式: 'mean', 'max', 'cls'
    ):
        super(FeatureEncoder, self).__init__()

        self.d_model = d_model
        self.pooling = pooling

        # Token Embedding
        self.embedding = nn.Linear(c_in, d_model)

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # Transformer Encoder
        from .Transformer_EncDec import Encoder, EncoderLayer
        from .SelfAttention_Family import FullAttention, AttentionLayer

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(attention_dropout=dropout, output_attention=False),
                        d_model=d_model,
                        n_heads=n_heads
                    ),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        # Pooling projection (optional)
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Parameters
        ----------
        x : torch.Tensor
            输入序列，shape (batch_size, seq_len, c_in)
        x_mark : torch.Tensor, optional
            时间标记，shape (batch_size, seq_len, mark_dim)

        Returns
        -------
        features : torch.Tensor
            高维特征表示，shape (batch_size, d_model) 或 (batch_size, feat_dim)
        """
        batch_size, seq_len, c_in = x.shape

        # Token Embedding
        x = self.embedding(x)  # (B, L, d_model)

        # Add CLS token if using CLS pooling
        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, L+1, d_model)

        # Positional Encoding
        x = self.pos_encoding(x)

        # Transformer Encoding
        x, _ = self.encoder(x)  # (B, L, d_model) or (B, L+1, d_model)

        # Pooling
        if self.pooling == 'mean':
            # Average pooling over sequence length
            features = x.mean(dim=1)  # (B, d_model)
        elif self.pooling == 'max':
            # Max pooling over sequence length
            features = x.max(dim=1)[0]  # (B, d_model)
        elif self.pooling == 'cls':
            # Use CLS token
            features = x[:, 0, :]  # (B, d_model)
        elif self.pooling == 'flatten':
            # Flatten all features
            features = x.reshape(batch_size, -1)  # (B, L * d_model)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return features


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            shape (seq_len, batch_size, d_model)

        Returns
        -------
        torch.Tensor
            shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def create_feature_encoder(
    c_in: int = 21,
    d_model: int = 256,
    n_heads: int = 8,
    e_layers: int = 2,
    d_ff: int = 512,
    pretrained_path: str = None,
    device: str = 'cuda'
) -> FeatureEncoder:
    """
    创建特征编码器

    Parameters
    ----------
    c_in : int
        输入特征维度
    d_model : int
        模型维度
    n_heads : int
        注意力头数
    e_layers : int
        编码器层数
    d_ff : int
        前馈网络维度
    pretrained_path : str, optional
        预训练权重路径
    device : str
        设备

    Returns
    -------
    encoder : FeatureEncoder
        特征编码器
    """
    encoder = FeatureEncoder(
        c_in=c_in,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_ff=d_ff,
        dropout=0.0,  # 不使用 dropout，保持特征稳定
        pooling='mean'
    ).to(device)

    if pretrained_path:
        encoder.load_state_dict(torch.load(pretrained_path))
        print(f"Loaded pretrained encoder from {pretrained_path}")

    encoder.eval()  # 设置为评估模式
    return encoder


if __name__ == '__main__':
    # 测试特征编码器
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    batch_size = 4
    seq_len = 96
    c_in = 21
    d_model = 128

    # 创建随机输入
    x = torch.randn(batch_size, seq_len, c_in).cuda()
    x_mark = torch.randn(batch_size, seq_len, 4).cuda()

    # 创建编码器
    encoder = create_feature_encoder(
        c_in=c_in,
        d_model=d_model,
        n_heads=4,
        e_layers=2,
        d_ff=256,
        device='cuda'
    )

    # 前向传播
    with torch.no_grad():
        features = encoder(x, x_mark)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Feature dimension: {features.shape[-1]}")
