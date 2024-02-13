import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.data.prepare import DatasetPreparer
from src.data import data_config
from src.models_.CNNs.vgg19 import (
    VGG19,
)  # Update the import statement as per your project structure
import wandb

# Initialize Weights & Biases
wandb.login(key=os.getenv("WANDB_KEY"))
wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"))

# Load the data
dataset = DatasetPreparer(
    model_type="vgg19"
)  # Ensure you have a corresponding configuration for VGG19 in DatasetPreparer
train_loader, vali_loader, _ = dataset.prepare_dataset()

# Model
model = VGG19()
model.to(data_config.DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=data_config.LEARNING_RATE)


class Trainer:
    def __init__(
        self,
        model_,
        train_dl=train_loader,
        vali_dl=vali_loader,
        criteria=criterion,
        optima=optimizer,
        epochs=data_config.EPOCHS,
    ):
        self.model = model_
        self.train_loader = train_dl
        self.vali_loader = vali_dl
        self.criterion = criteria
        self.optimizer = optima
        self.epochs = epochs

    def train_epoch(self):
        self.model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for images, labels in self.train_loader:
            images, labels = images.to(data_config.DEVICE), labels.to(
                data_config.DEVICE
            )
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / len(self.train_loader)
        avg_accuracy = total_correct / total_samples
        return avg_loss, avg_accuracy

    def validate_epoch(self):
        self.model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0

        with torch.no_grad():
            for images, labels in self.vali_loader:
                images, labels = images.to(data_config.DEVICE), labels.to(
                    data_config.DEVICE
                )
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(self.vali_loader)
        avg_accuracy = total_correct / total_samples
        return avg_loss, avg_accuracy

    def train(self):
        for epoch in range(self.epochs):
            train_loss, train_accuracy = self.train_epoch()
            vali_loss, vali_accuracy = self.validate_epoch()

            print(
                f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {vali_loss:.4f}, Val Accuracy: {vali_accuracy:.4f}"
            )
            wandb.log(
                {
                    "Train Loss": train_loss,
                    "Train Accuracy": train_accuracy,
                    "Validation Loss": vali_loss,
                    "Validation Accuracy": vali_accuracy,
                }
            )

        # Save the model at the end of training
        model_path = os.path.join(
            "src/models_/_saved_models", f"vgg19_{self.epochs}.pth"
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)


# Train the model
trainer = Trainer(model)
trainer.train()
