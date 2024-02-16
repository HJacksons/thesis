import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from src.data.prepare import DatasetPreparer
from src.data import data_config
import matplotlib.pyplot as plt
from src.models_.features.extract_features_and_combine import main_extractor_combiner
from torch.utils.data import TensorDataset, DataLoader


combined_features, vgg19_features, inception_features, vgg19_labels = (
    main_extractor_combiner()
)
print("Inception Features Shape:", inception_features.shape)
print("VGG19 Features Shape:", vgg19_features.shape)


class FeatureAttention(nn.Module):
    def __init__(self, feature_dim):
        super(FeatureAttention, self).__init__()
        self.attention_weights = nn.Linear(feature_dim, feature_dim)

    def forward(self, features):
        attention_scores = F.softmax(self.attention_weights(features), dim=1)
        weighted_features = features * attention_scores
        return weighted_features


class CombinedAttentionModel(nn.Module):
    def __init__(self, inception_feature_dim, vgg19_feature_dim):
        super(CombinedAttentionModel, self).__init__()
        self.inception_attention = FeatureAttention(inception_feature_dim)
        self.vgg19_attention = FeatureAttention(vgg19_feature_dim)
        self.fusion_layer = nn.Linear(2 * inception_feature_dim, 512)
        # Omitting the feature_combiner since we're focusing on saving the attention mechanism

    def forward(self, inception_features, vgg19_features):
        weighted_inception_features = self.inception_attention(inception_features)
        weighted_vgg19_features = self.vgg19_attention(vgg19_features)
        # Feature fusion: concatenation and linear layer
        combined_f = torch.cat(
            (weighted_inception_features, weighted_vgg19_features), dim=1
        )
        combined_feature = self.fusion_layer(combined_f)
        # Directly return weighted features without combining them for ViT
        return combined_feature


# Prepare dataset and loader
dataset = TensorDataset(vgg19_features, inception_features, vgg19_labels)
feature_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CombinedAttentionModel(inception_features.shape[1], vgg19_features.shape[1]).to(
    device
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for vgg19_feat, inception_feat, labels in feature_loader:
        vgg19_feat, inception_feat = vgg19_feat.to(device), inception_feat.to(device)
        optimizer.zero_grad()
        weighted_inception_features, weighted_vgg19_features = model(
            inception_feat, vgg19_feat
        )
        # Since we are not classifying here, we skip loss calculation and update
        # Perform any necessary operations with weighted features if needed
        # For now, we assume a simple training loop for demonstration
        # loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs} completed.")

# Save the trained model
torch.save(model.state_dict(), "attention_model.pth")
