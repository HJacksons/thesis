import sys
import os
sys.path.append(os.path.abspath("../../"))
import torch
import torch.nn as nn
from src.models_.ViT import ViT_config


class ClassToken(nn.Module):
    def __init__(self, hidden_dim):
        super(ClassToken, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, x):
        batch_size = x.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat([cls_token, x], dim=1)


class ViT(nn.Module):
    def __init__(self, config=ViT_config.config):
        super(ViT, self).__init__()
        self.patch_size = config["patch_size"]
        self.num_patches = config["num_patches"]
        self.hidden_dim = config["hidden_dim"]
        self.num_heads = config["num_heads"]
        self.mlp_dim = config["mlp_dim"]
        self.dropout_rate = config["dropout_rate"]
        self.num_layers = config["num_layers"]
        self.num_classes = config["num_classes"]
        self.num_channels = config["num_channels"]

        self.patch_embedding = nn.Linear(
            (self.patch_size ** 2) * self.num_channels, self.hidden_dim
        )
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, self.hidden_dim)
        )
        self.cls_token = ClassToken(self.hidden_dim)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.mlp_dim,
            dropout=self.dropout_rate,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        self.head = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(
            B,
            C,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, -1, C * self.patch_size ** 2)
        x = self.patch_embedding(x)
        x = self.cls_token(x)
        x = x + self.position_embedding
        x = self.transformer_encoder(x)
        x = x[:, 0]
        x = self.head(x)
        return x

# if __name__ == "__main__":
#     model = ViT()
#     print(model)
#     print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
