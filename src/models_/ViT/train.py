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
    for step, (x, y) in enumerate(train_loader):
        if torch.is_tensor(x):
            x = [to_pil_image(img) for img in x]
        inputs = feature_extractor(x, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(data_config.DEVICE)
        labels = y.to(data_config.DEVICE)

        output, loss = model(pixel_values, None)
        if loss is None:
            loss = loss_fn(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if step % 50 == 0:
            test = next(iter(test_loader))
            test_x, test_y = test
            if torch.is_tensor(test_x):
                test_x = [to_pil_image(img) for img in test_x]
            test_inputs = feature_extractor(images=test_x, return_tensors="pt")
            test_pixel_values = test_inputs["pixel_values"].to(data_config.DEVICE)
            test_labels = test_y.to(data_config.DEVICE)

            test_output, loss = model(test_pixel_values, test_labels)
            test_output = test_output.argmax(1)
            accuracy = (
                test_output == test_labels
            ).sum().item() / data_config.BATCH_SIZE
            logging.info(
                f"Epoch: {epoch} | Step: {step} | train loss: {loss} | test accuracy: {accuracy}"
            )
