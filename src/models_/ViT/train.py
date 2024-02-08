import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.data.prepare import DatasetPreparer
from src.data.prepare import data_config
from src.models_.ViT.ViT import ViT
from transformers import ViTFeatureExtractor
from torchvision.transforms.functional import to_pil_image

import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
import wandb
import logging

wandb.login(key=os.getenv("WANDB_KEY"))
wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"))

dataset = DatasetPreparer()
train_loader, vali_loader, test_loader = dataset.prepare_dataset()

# Model
model = ViT()
model.to(data_config.DEVICE)
feature_extractor = ViTFeatureExtractor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)
optimizer = optim.Adam(model.parameters(), lr=data_config.LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(data_config.EPOCHS):
    # Train the model
    model.train()
    train_loss = 0
    train_accuracy = 0
    for step, (images, labels) in enumerate(train_loader):
        if torch.is_tensor(images):
            images = [to_pil_image(img) for img in images]
        inputs = feature_extractor(images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(data_config.DEVICE)
        labels = labels.to(data_config.DEVICE)

        # Forward pass
        outputs, loss = model(pixel_values, None)
        if loss is not None:
            loss = loss_fn(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_accuracy += (outputs.argmax(1) == labels).float().mean()

        # Validation
        if step % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                val_accuracy = 0
                for val_images, val_labels in vali_loader:
                    if torch.is_tensor(val_images):
                        val_images = [to_pil_image(img) for img in val_images]
                    inputs = feature_extractor(val_images, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(data_config.DEVICE)
                    val_labels = val_labels.to(data_config.DEVICE)

                    val_outputs, val_loss = model(pixel_values, None)
                    if val_loss is not None:
                        val_loss = loss_fn(val_outputs, val_labels)
                        val_accuracy = (
                            (val_outputs.argmax(1) == val_labels).float().mean()
                        )

                        logging.info(
                            f"Epoch [{epoch}/{data_config.EPOCHS}], Step [{step}/{len(train_loader)}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
                        )

            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                }
            )
