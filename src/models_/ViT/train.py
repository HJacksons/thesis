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
    train_loss, train_accuracy = 0, 0
    for images, labels in train_loader:
        if torch.is_tensor(images):
            images = [to_pil_image(img) for img in images]
        inputs = feature_extractor(images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(data_config.DEVICE)
        labels = labels.to(data_config.DEVICE)

        # Forward pass
        outputs, loss = model(pixel_values, None)
        if loss is None:
            loss = loss_fn(outputs, labels)
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = torch.argmax(outputs, 1)
            train_accuracy += (predictions == labels).sum().item()

        # Validation
        model.eval()
        val_loss, val_accuracy = 0, 0
        with torch.no_grad():
            for images, labels in vali_loader:
                if torch.is_tensor(images):
                    images = [to_pil_image(img) for img in images]
                inputs = feature_extractor(images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(data_config.DEVICE)
                labels = labels.to(data_config.DEVICE)

                # Forward pass
                outputs, loss = model(pixel_values, None)
                if loss is None:
                    loss = loss_fn(outputs, labels)

                    # Calculate loss and accuracy
                    val_loss += loss.item()
                    predictions = torch.argmax(outputs, 1)
                    val_accuracy += (predictions == labels).sum().item()

        # Log training and validation results
        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        val_loss /= len(vali_loader)
        val_accuracy /= len(vali_loader)
        logging.info(
            f"Epoch: {epoch} | Train loss: {train_loss} | Train accuracy: {train_accuracy} | Val loss: {val_loss} | Val accuracy: {val_accuracy}"
        )
        wandb.log(
            {
                "Train loss": train_loss,
                "Train accuracy": train_accuracy,
                "Val loss": val_loss,
                "Val accuracy": val_accuracy,
            }
        )
