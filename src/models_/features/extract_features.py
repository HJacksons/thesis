import torch
from src.data import data_config
import torch.nn as nn


class ModifiedInception(nn.Module):
    def __init__(self, original_model):
        super(ModifiedInception, self).__init__()
        # Copy all layers except the final fully connected layer
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.avgpool = original_model.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        return x


class ModifiedVGG(nn.Module):
    def __init__(self, original_model):
        super(ModifiedVGG, self).__init__()
        self.features = original_model.features

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return x


def extract_features(model, loader, device):
    model = model.to(device)
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(data_config.DEVICE)
            features = model(images)
            features_list.append(features.cpu())  # Move features to CPU
            labels_list.append(labels)

    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    return features, labels
