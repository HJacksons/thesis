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

dataset = TensorDataset(vgg19_features, inception_features, vgg19_labels)
feature_loader = DataLoader(dataset, batch_size=32, shuffle=True)


class FeatureAttention(nn.Module):
    def __init__(self, feature_dim):
        super(FeatureAttention, self).__init__()
        self.attention_weights = nn.Linear(feature_dim, feature_dim)

    def forward(self, features):
        # Compute attention scores & apply softmax
        attention_scores = F.softmax(self.attention_weights(features), dim=1)
        # Apply the attention weights
        weighted_features = features * attention_scores
        return weighted_features


class CombinedAttentionModel(nn.Module):
    def __init__(self, inception_feature_dim, vgg19_feature_dim):
        super(CombinedAttentionModel, self).__init__()
        self.inception_attention = FeatureAttention(inception_feature_dim)
        self.vgg19_attention = FeatureAttention(vgg19_feature_dim)
        # Assuming the combined feature vector feeds into ViT, adjust dimensions as necessary
        self.feature_combiner = nn.Linear(
            inception_feature_dim + vgg19_feature_dim, 768
        )

    def forward(self, inception_features, vgg19_features):
        weighted_inception_features = self.inception_attention(inception_features)
        weighted_vgg19_features = self.vgg19_attention(vgg19_features)
        combined_features = torch.cat(
            [weighted_inception_features, weighted_vgg19_features], dim=1
        )
        # Prepare for ViT input
        output = self.feature_combiner(combined_features)
        return output


def calculate_accuracy(output, labels):
    _, preds = torch.max(output, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# Initialize lists to store losses and accuracies for visualization
train_losses = []
train_accuracies = []
model = CombinedAttentionModel(inception_features.shape[1], vgg19_features.shape[1]).to(
    data_config.DEVICE
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
num_epochs = 10

for epoch in range(num_epochs):
    epoch_loss = 0.0
    epoch_acc = 0.0
    for batch in feature_loader:
        inception_features, vgg19_features, labels = batch

        # Move tensors to the correct device
        inception_features = inception_features.to(data_config.DEVICE)
        vgg19_features = vgg19_features.to(data_config.DEVICE)
        labels = labels.to(data_config.DEVICE)

        optimizer.zero_grad()
        outputs = model(inception_features, vgg19_features)
        loss = criterion(outputs, labels)
        acc = calculate_accuracy(outputs, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    epoch_loss /= len(feature_loader)
    epoch_acc /= len(feature_loader)

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# # Plotting the training losses and accuracies
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(train_losses, label='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss Over Epochs')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(train_accuracies, label='Training Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Accuracy Over Epochs')
# plt.legend()
#
# plt.show()
