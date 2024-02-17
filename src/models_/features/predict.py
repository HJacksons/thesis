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
combined_features, _, inception_features, vgg19_labels = main_extractor_combiner()
print("combined features:", inception_features.shape)

# Create dataset and dataloader
dataset = TensorDataset(inception_features, vgg19_labels)
feature_loader = DataLoader(dataset, batch_size=32, shuffle=True)


class CustomViTAdapter(nn.Module):
    def __init__(
        self, original_feature_size, img_channels=3, img_height=224, img_width=224
    ):
        super(CustomViTAdapter, self).__init__()
        self.img_channels = img_channels
        self.img_height = img_height
        self.img_width = img_width
        # Calculate the expected size after reshaping
        expected_size = img_channels * img_height * img_width
        self.project = nn.Linear(original_feature_size, expected_size)

    def forward(self, x):
        x = self.project(x)
        # Reshape to mimic an image format: [batch_size, num_channels, height, width]
        x = x.view(-1, self.img_channels, self.img_height, self.img_width)
        return x


# Initialize model and custom adapter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT(num_labels=3)  # Adjust num_labels as necessary
vit_model_path = "src/models_/_saved_models/ViTModel_224_100.pth"
model.load_state_dict(torch.load(vit_model_path, map_location=device))
model.to(device)

adapter = CustomViTAdapter(2048)  # 6144 is the size of your combined features
adapter.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(model.parameters()) + list(adapter.parameters()), lr=0.001)

# Evaluation loop with class index output
model.eval()
adapter.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for features, labels in feature_loader:
        features, labels = features.to(device), labels.to(device)
        adapted_features = adapter(features)
        outputs = model(adapted_features)
        logits = outputs[0]  # Assuming the first element contains the logits
        _, predicted = torch.max(logits, 1)

        # Output predicted class indices for the first batch (as an example)
        if total_samples == 0:  # Just for the first batch
            print(f"Predicted class indices: {predicted.tolist()}")

        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

accuracy = 100 * total_correct / total_samples
print(f"Overall Accuracy: {accuracy}%")
wandb.log({"Accuracy": accuracy})
