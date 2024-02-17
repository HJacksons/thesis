import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
from src.models_.ViT.ViT import (
    ViT,
)  # Ensure this ViT class is correctly defined as shown previously
from src.models_.features.extract_features_and_combine import main_extractor_combiner
import wandb

# Initialize Weights & Biases
wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"))
# Ensure environment variables for WANDB_PROJECT and WANDB_ENTITY are correctly set

# Load the combined features
combined_features, _, _, vgg19_labels = main_extractor_combiner()
print("combined features:", combined_features.shape)

# Create dataset and dataloader
dataset = TensorDataset(combined_features, vgg19_labels)
feature_loader = DataLoader(dataset, batch_size=32, shuffle=True)


class CustomViTAdapter(nn.Module):
    def __init__(self, original_feature_size, projected_size=768):
        super(CustomViTAdapter, self).__init__()
        self.project = nn.Linear(original_feature_size, projected_size)
        # Additional reshape layer might be necessary depending on your ViT model's input requirements

    def forward(self, x):
        x = self.project(x)
        # Optionally reshape x here to match your ViT model's input shape
        return x


# Initialize model and custom adapter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT(num_labels=3)  # Adjust num_labels as necessary
vit_model_path = "src/models_/_saved_models/ViTModel_224_100.pth"
model.load_state_dict(torch.load(vit_model_path, map_location=device))
model.to(device)

adapter = CustomViTAdapter(6144)  # 6144 is the size of your combined features
adapter.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(model.parameters()) + list(adapter.parameters()), lr=0.001)

# Evaluation loop
model.eval()
adapter.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for features, labels in feature_loader:
        features, labels = features.to(device), labels.to(device)
        adapted_features = adapter(features)
        outputs = model(adapted_features)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

accuracy = 100 * total_correct / total_samples
print(f"Accuracy: {accuracy}%")
wandb.log({"Accuracy": accuracy})

wandb.finish()
