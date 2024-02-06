import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.data.prepare import DatasetPreparer
from src.data.prepare import data_config
from ViT import ViT
import wandb

wandb.login(key=os.getenv("WANDB_KEY"))
wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"))


dataset = DatasetPreparer()
train_loader, vali_loader, test_loader = dataset.prepare_dataset()

# Model
model = ViT()
model.to(data_config.DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=data_config.LEARNING_RATE)


class Trainer:
    def __init__(self, model_, train_dl=train_loader, vali_dl=vali_loader, criteria=criterion, optima=optimizer,
                 epochs=data_config.EPOCHS):
        self.model = model_
        self.train_loader = train_dl
        self.vali_loader = vali_dl
        self.criterion = criteria
        self.optimizer = optima
        self.epochs = epochs

    # Train the model for a epoch
    def train_epoch(self):
        self.model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for images, labels in self.train_loader:
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        ava_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples
        return ava_loss, accuracy

    # Validate the model for a epoch
    def validate_epoch(self):
        self.model.eval()
        val_loss, val_correct, val_samples = 0, 0, 0

        with torch.no_grad():
            for images, labels in self.vali_loader:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_samples += labels.size(0)

        ava_loss = val_loss / len(self.vali_loader)
        val_accuracy = val_correct / val_samples
        return ava_loss, val_accuracy

    # Train the model
    def fit(self):
        # Initialize a new wandb run
        wandb.init(project="your_project_name", entity="your_wandb_username")

        history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1} of {self.epochs}")
            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.validate_epoch()

            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)

            # Log metrics to wandb
            wandb.log({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
                       "val_loss": val_loss, "val_accuracy": val_accuracy})

        # Finish the wandb run
        wandb.finish()

        return history


trainer = Trainer(model)
history = trainer.fit()

# Save the model
torch.save(model.state_dict(), 'model.pth')
