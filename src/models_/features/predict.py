# Now lets classify the combined features using the ViT model

from src.models_.ViT.ViT import ViT
import torch
import torch.nn as nn
import torch.optim as optim
from src.data.prepare import DatasetPreparer
from src.data import data_config
import wandb
import os
import logging
from src.models_.features.extract_features_and_combine import main_extractor_combiner
from torch.utils.data import TensorDataset, DataLoader

# Initialize Weights & Biases
wandb.login(key=os.getenv("WANDB_KEY"))
wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"))

# Load the combined features
combined_features, vgg19_features, inception_features, vgg19_labels = (
    main_extractor_combiner()
)
combined_features_reshaped = combined_features.view(-1, 3, 224, 224)

# Prepare dataset and loader
dataset = TensorDataset(combined_features_reshaped, vgg19_labels)
feature_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT()
vit_model_path = "src/models_/_saved_models/ViTModel_224_100.pth"
model.load_state_dict(torch.load(vit_model_path, map_location=device))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=data_config.LEARNING_RATE)

with torch.no_grad():
    model.eval()
    total_correct = 0
    total_samples = 0
    for images, labels in feature_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

    accuracy = 100 * total_correct / total_samples
    logging.info(f"Accuracy: {accuracy}%")
    wandb.log({"Accuracy": accuracy})
    wandb.finish()
    print(f"Accuracy: {accuracy}%")
    print("Done!")
