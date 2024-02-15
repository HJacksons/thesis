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


import torch
from src.data.prepare import DatasetPreparer  # Ensure this is the correct import path
from src.models_.CNNs.inceptionV3 import (
    Inception,
)  # Ensure this is the correct import path
from src.models_.CNNs.vgg19 import VGG19  # Ensure this is the correct import path
from src.data import data_config
from torch.utils.data import DataLoader

# Assuming ModifiedInception and ModifiedVGG are defined as in your provided code


def load_model(model_path, model_class, device):
    # Instantiate and load the pretrained model
    model = model_class(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


def main_feature_extraction():
    # Initialize device
    device = torch.device(data_config.DEVICE if torch.cuda.is_available() else "cpu")

    # Load models
    inception_model = load_model(
        "src/models_/_saved_models/inceptionv3100.pth", ModifiedInception, device
    )
    vgg_model = load_model(
        "src/models_/_saved_models/vgg19_all_layers_100.pth", ModifiedVGG, device
    )

    # Prepare data loaders
    dataset_preparer = DatasetPreparer()
    _, _, test_loader = dataset_preparer.prepare_dataset()  # Adjust as necessary

    # Extract features
    inception_features, inception_labels = extract_features(
        inception_model, test_loader, device
    )
    vgg_features, vgg_labels = extract_features(vgg_model, test_loader, device)

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
