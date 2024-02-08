import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.data.prepare import DatasetPreparer
from src.data.prepare import data_config
from src.models_.ViT.ViT import ViT
from transformers import ViTFeatureExtractor
from torchvision.transforms.functional import to_pil_image

import wandb
import logging

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Weights & Biases
wandb.login(key=os.getenv("WANDB_KEY"))
wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"))

# Prepare dataset
dataset = DatasetPreparer()
train_loader, vali_loader, test_loader = dataset.prepare_dataset()

# Initialize the model
model = ViT().to(data_config.DEVICE)
feature_extractor = ViTFeatureExtractor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=data_config.LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

# Training and validation loop
for epoch in range(data_config.EPOCHS):
    model.train()
    train_loss, train_correct, total_train_samples = 0, 0, 0

    for images, labels in train_loader:
        # Process images through the feature extractor
        if torch.is_tensor(images):
            images = [to_pil_image(image) for image in images]
        inputs = feature_extractor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(data_config.DEVICE)
        labels = labels.to(data_config.DEVICE)

        # Forward pass
        outputs = model(pixel_values)
        loss = loss_fn(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
        total_train_samples += labels.size(0)

    train_loss /= len(train_loader)
    train_accuracy = train_correct / total_train_samples

    # Validation
    model.eval()
    val_loss, val_correct, total_val_samples = 0, 0, 0

    with torch.no_grad():
        for images, labels in vali_loader:
            if torch.is_tensor(images):
                images = [to_pil_image(image) for image in images]
            inputs = feature_extractor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(data_config.DEVICE)
            labels = labels.to(data_config.DEVICE)

            outputs = model(pixel_values)
            loss = loss_fn(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()
            total_val_samples += labels.size(0)

    val_loss /= len(vali_loader)
    val_accuracy = val_correct / total_val_samples

    # Log results
    logging.info(
        f"Epoch {epoch+1}/{data_config.EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
    )
    wandb.log(
        {
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_accuracy,
        }
    )

# Save the model
torch.save(model.state_dict(), "vit_model.pth")
