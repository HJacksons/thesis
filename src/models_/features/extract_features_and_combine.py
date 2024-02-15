from src.data.prepare import DatasetPreparer
from src.models_.CNNs.inceptionV3 import Inception
from src.models_.CNNs.vgg19 import VGG19
from src.models_.ViT.ViT import ViT
from src.data import data_config
from src.models_.features.extract_features import (
    ModifiedInception,
    ModifiedVGG,
    extract_features,
)
import torch
import wandb
import os
import matplotlib.pyplot as plt
import pandas as pd


wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"))


# Assuming ModifiedInception and ModifiedVGG are defined as in your provided code


def load_model(model_path, model, device):
    model.load_state_dict(torch.load(model_path, map_location=data_config.DEVICE))
    model.to(data_config.DEVICE)
    return model


def main_feature_extraction():
    # Initialize device
    # Load models
    inception_model = load_model(
        "src/models_/_saved_models/inceptionv3100.pth",
        ModifiedInception,
        data_config.DEVICE,
    )
    vgg_model = load_model(
        "src/models_/_saved_models/vgg19_all_layers_100.pth",
        ModifiedVGG,
        data_config.DEVICE,
    )

    # Prepare the dataset
    inception_dataset = DatasetPreparer(model_type="inception")
    inception_train_loader, _, _ = inception_dataset.prepare_dataset()

    vgg19_dataset = DatasetPreparer(model_type="vit")
    vgg19_train_loader, _, _ = vgg19_dataset.prepare_dataset()

    # Extract features
    inception_features, inception_labels = extract_features(
        inception_model, inception_train_loader, data_config.DEVICE
    )
    vgg_features, vgg_labels = extract_features(
        vgg_model, vgg19_train_loader, data_config.DEVICE
    )

    # Combine features
    combined_features = torch.cat([inception_features, vgg_features], dim=1)

    # Log or print shapes for verification
    print(f"Inception Features Shape: {inception_features.shape}")
    print(f"VGG Features Shape: {vgg_features.shape}")
    print(f"Combined Features Shape: {combined_features.shape}")
    print(
        f"Labels Shape: {inception_labels.shape}"
    )  # Assuming labels are the same for both

    # Return the extracted and combined features, and labels
    return combined_features, inception_features, vgg_features, inception_labels


if __name__ == "__main__":
    combined_features, inception_features, vgg_features, labels = (
        main_feature_extraction()
    )
