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
import numpy as np
import wandb
import logging

# Initialize WandB
wandb.login(key=os.getenv("WANDB_KEY"))
wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"))

dataset = DatasetPreparer()
train_loader, vali_loader, test_loader = dataset.prepare_dataset()

# Initialize model
MODEL = ViT().to(data_config.DEVICE)
feature_extractor = ViTFeatureExtractor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)
optimizer = optim.Adam(MODEL.parameters(), lr=data_config.LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()


class Trainer:
    def __init__(
        self,
        model=MODEL,
        train_dl=train_loader,
        vali_dl=train_loader,
        criteria=loss_fn,
        optima=optimizer,
        epochs=data_config.EPOCHS,
        feature_extract=feature_extractor,
    ):
        self.model = model
        self.train_loader = train_dl
        self.vali_loader = vali_dl
        self.criterion = criteria
        self.optimizer = optima
        self.epochs = epochs
        self.feature_extractor = feature_extract

    def train(self):
        self.model.train()
        log_interval = 50

        for epoch in range(data_config.EPOCHS):
            self.model.train()
            total_train_loss = 0
            total_train_correct = 0
            total_train_samples = 0

            for step, (images, labels) in self.train_loader:
                images = (
                    [to_pil_image(img) for img in images]
                    if torch.is_tensor(images)
                    else images
                )
                inputs = self.feature_extractor(images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(data_config.DEVICE)
                labels = labels.to(data_config.DEVICE)

                self.optimizer.zero_grad()
                outputs, loss = self.model(pixel_values, None)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                if step % log_interval == 0:
                    _, predicted = torch.max(outputs, 1)
                    accuracy = (predicted == labels).sum().item()

                    logging.info(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}, Accuracy: {accuracy}")
                    wandb.log(
                        {"Step Loss": loss.item(), "Step Accuracy": accuracy, "Step": step + epoch * len(train_loader)})

            #     total_train_loss += loss.item() * labels.size(0)
            #     _, predicted = torch.max(outputs, 1)
            #     total_train_correct += (predicted == labels).sum().item()
            #     total_train_samples += labels.size(0)
            #
            # avg_train_loss = total_train_loss / total_train_samples
            # train_accuracy = total_train_correct / total_train_samples

            # Validation
            self.model.eval()
            total_val_loss = 0
            total_val_correct = 0
            total_val_samples = 0
            with torch.no_grad():
                for images, labels in self.vali_loader:
                    images = (
                        [to_pil_image(img) for img in images]
                        if torch.is_tensor(images)
                        else images
                    )
                    inputs = self.feature_extractor(images, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(data_config.DEVICE)
                    labels = labels.to(data_config.DEVICE)

                    outputs, loss = self.model(pixel_values, None)
                    loss = self.criterion(outputs, labels)

                    total_val_loss += loss.item() * labels.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_val_correct += (predicted == labels).sum().item()
                    total_val_samples += labels.size(0)

            avg_val_loss = total_val_loss / total_val_samples
            val_accuracy = total_val_correct / total_val_samples

            logging.info(f"Epoch: {epoch}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}")
            wandb.log({"Validation Loss": avg_val_loss, "Validation Accuracy": val_accuracy, "Epoch": epoch})


trainer = Trainer()
trainer.train()
