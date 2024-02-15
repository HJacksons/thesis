import torch
import torch.nn as nn
from torchvision import models
from src.data.prepare import DatasetPreparer  # Adjust import path as necessary
from src.data import data_config

# Model paths
INCEPTION_MODEL_PATH = "src/models_/_saved_models/inceptionv3100.pth"
VGG19_MODEL_PATH = "src/models_/_saved_models/vgg19_all_layers_100.pth"

# Assuming Inception and VGG19 model classes are defined as shown previously
from src.models_.CNNs.inceptionV3 import Inception
from src.models_.CNNs.vgg19 import VGG19


def load_model(model_path, model_class, device):
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model


def extract_features_directly(
    model,
    loader,
    device,
    layer_path,
):
    model.eval()
    model.to(device)
    features_list = []
    labels_list = []

    # Navigate to the layer
    target_layer = model
    for attr in layer_path.split("."):
        target_layer = getattr(target_layer, attr)

    # Define a hook to capture the output of the specified layer
    def hook(module, input, output):
        features_list.append(output.detach())

    # Attach the hook to the layer
    handle = target_layer.register_forward_hook(hook)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            _ = model(images)  # Forward pass to trigger the hook
            labels_list.append(labels)

    # Remove the hook after extraction
    handle.remove()

    # Process and return the features and labels
    features = torch.cat([f.cpu() for f in features_list], dim=0)
    labels = torch.cat(labels_list, dim=0)
    return features, labels


def main_feature_extraction():
    device = torch.device(data_config.DEVICE if torch.cuda.is_available() else "cpu")

    # Load the models
    inception_model = load_model(INCEPTION_MODEL_PATH, Inception, device)
    vgg19_model = load_model(VGG19_MODEL_PATH, VGG19, device)

    # Prepare the dataset
    inception_dataset = DatasetPreparer(model_type="inception")
    inception_train_loader, _, _ = inception_dataset.prepare_dataset()

    vgg19_dataset = DatasetPreparer(model_type="vgg19")
    vgg19_train_loader, _, _ = vgg19_dataset.prepare_dataset()

    # Extract features
    # Adjust 'Mixed_7c' and 'features' based on your model's layer names for feature extraction
    inception_features, inception_labels = extract_features_directly(
        inception_model, inception_train_loader, device, ["Mixed_7c"]
    )
    vgg19_features, vgg19_labels = extract_features_directly(
        vgg19_model, vgg19_train_loader, device, "features"
    )

    # Combine features
    combined_features = torch.cat([inception_features, vgg19_features], dim=1)

    # Log or print shapes for verification
    print(f"Inception Features Shape: {inception_features.shape}")
    print(f"VGG19 Features Shape: {vgg19_features.shape}")
    print(f"Combined Features Shape: {combined_features.shape}")
    print(
        f"Labels Shape: {inception_labels.shape}"
    )  # Assuming labels are the same for both

    # Return the extracted and combined features, and labels
    return combined_features, inception_features, vgg19_features, inception_labels


if __name__ == "__main__":
    combined_features, inception_features, vgg19_features, labels = (
        main_feature_extraction()
    )
