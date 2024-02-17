from src.models_.ViT.ViT import ViT
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import logging
from src.models_.features.extract_features_and_combine import main_extractor_combiner
import wandb
from src.data import data_config

# Load the combined features
combined_features, _, _, vgg19_labels = main_extractor_combiner()
print("combined features:", combined_features.shape)

# Assuming you have the combined features and labels loaded
# Adjust the DataLoader to work with the original combined features shape
dataset = TensorDataset(combined_features, vgg19_labels)
feature_loader = DataLoader(dataset, batch_size=32, shuffle=True)


class CustomViTAdapter(nn.Module):
    def __init__(self, original_feature_size, projected_size=768):
        super(CustomViTAdapter, self).__init__()
        # Project the combined features to the size expected by ViT's embedding layer
        self.project = nn.Linear(original_feature_size, projected_size)

    def forward(self, x):
        x = self.project(x)
        # Reshape to match the ViT's expected input shape, here projected_size must match ViT's embedding dimension
        batch_size = x.size(0)
        x = x.view(
            batch_size, 1, 224, 224
        )  # Example reshape, adjust based on your ViT model configuration
        return x


# Initialize model and custom adapter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model_path = "src/models_/_saved_models/ViTModel_224_100.pth"
model = ViT()
model.load_state_dict(torch.load(vit_model_path, map_location=device))
model.to(device)

# Initialize the custom adapter
# Here, 6144 is the size of your combined features for each sample
adapter = CustomViTAdapter(6144)
adapter.to(device)

# Initialize Weights & Biases
wandb.login(key=os.getenv("WANDB_KEY"))
wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"))

criterion = nn.CrossEntropyLoss()
# Include the adapter parameters in the optimizer
optimizer = optim.Adam(
    list(model.parameters()) + list(adapter.parameters()), lr=data_config.LEARNING_RATE
)

with torch.no_grad():
    model.eval()
    adapter.eval()
    total_correct = 0
    total_samples = 0
    for features, labels in feature_loader:
        features, labels = features.to(device), labels.to(device)
        # Use the adapter to adjust features before passing them to the ViT model
        adapted_features = adapter(features)
        outputs = model(adapted_features)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

    accuracy = 100 * total_correct / total_samples
    logging.info(f"Accuracy: {accuracy}%")
    wandb.log({"Accuracy": accuracy})
    wandb.finish()
    print(f"Accuracy: {accuracy}%")
    print("Done!")
