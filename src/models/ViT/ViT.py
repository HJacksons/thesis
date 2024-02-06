import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import ViT_config


class ClassToken(nn.Module):
    def __init__(self, hidden_dim):
        super(ClassToken, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, x):
        batch_size = x.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat([cls_token, x], dim=1)


def mlp(x, hidden_dim, mlp_dim, dropout_rate):
    x = nn.Linear(hidden_dim, mlp_dim)(x)
    x = nn.GELU()(x)
    x = nn.Dropout(dropout_rate)(x)
    x = nn.Linear(mlp_dim, hidden_dim)(x)
    x = nn.Dropout(dropout_rate)(x)
    return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_dim, dropout_rate):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, config=ViT_config.config):
        super(ViT, self).__init__()
        self.patch_size = config['patch_size']
        self.num_patches = config['num_patches']
        self.hidden_dim = config['hidden_dim']
        self.num_heads = config['num_heads']
        self.mlp_dim = config['mlp_dim']
        self.dropout_rate = config['dropout_rate']
        self.num_layers = config['num_layers']
        self.num_classes = config['num_classes']
        self.num_channels = config['num_channels']

        self.patch_embedding = nn.Linear((self.patch_size ** 2) * self.num_channels, self.hidden_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.hidden_dim))
        self.cls_token = ClassToken(self.hidden_dim)

        transformer_layer = TransformerEncoderBlock(self.hidden_dim, self.num_heads, self.mlp_dim, self.dropout_rate)
        self.transformer_encoder = TransformerEncoder(transformer_layer, num_layers=self.num_layers)

        self.head = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x + self.position_embedding
        x = self.cls_token(x)
        x = self.transformer_encoder(x)
        x = x[:, 0]
        x = self.head(x)
        return x
