import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
from src.models_.ViT.ViT import ViT
from src.models_.features.extract_features_and_combine import main_extractor_combiner
import wandb

# Initialize Weights & Biases
# Ensure to initialize wandb before logging in
wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"))
wandb.login(key=os.getenv("WANDB_KEY"))

# Load the combined features
combined_features, _, _, vgg19_labels = main_extractor_combiner()
print("combined features:", combined_features.shape)

# Assuming combined_features is a 2D tensor [batch_size, feature_length]
# And you have the labels loaded properly
dataset = TensorDataset(combined_features, vgg19_labels)
feature_loader = DataLoader(dataset, batch_size=32, shuffle=True)


class CustomViTAdapter(nn.Module):
    def __init__(self, original_feature_size, projected_size=768):
        super(CustomViTAdapter, self).__init__()
        # Project the combined features to the size expected by ViT's embedding layer
        self.project = nn.Linear(original_feature_size, projected_size)

    def forward(self, x):
        # Directly project features without reshaping into an image format
        x = self.project(x)
        return x


# Initialize model and custom adapter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model_path = "src/models_/_saved_models/ViTModel_224_100.pth"
model = ViT()
model.load_state_dict(torch.load(vit_model_path, map_location=device))
model.to(device)

# Initialize the custom adapter
adapter = CustomViTAdapter(
    6144
)  # Here, 6144 is the size of your combined features for each sample
adapter.to(device)

criterion = nn.CrossEntropyLoss()
# Include the adapter parameters in the optimizer
optimizer = optim.Adam(
    list(model.parameters()) + list(adapter.parameters()), lr=0.001
)  # Adjust lr as per your config

# Training or Evaluation Loop (Here, simplified for evaluation)
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
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

    accuracy = 100 * total_correct / total_samples
    print(f"Accuracy: {accuracy}%")
    # Log to wandb
    wandb.log({"Accuracy": accuracy})
    wandb.finish()
