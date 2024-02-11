import numpy as np
import matplotlib.pyplot as plt
import torch
from src.models_.features.extract_features_and_combine import main_extractor_combiner


def visualize_features(features, labels):
    """
    Visualizes features using a scatter plot.

    Parameters:
    - features: A tensor of features.
    - labels: A list or tensor of labels for each feature point.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(features, torch.Tensor):
        features = features.cpu().detach().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().detach().numpy()

    # Assume features have been reduced to 2D for visualization (e.g., via PCA)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, cmap="viridis")
    plt.colorbar(scatter)
    plt.title("Feature Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


# Example usage
combined_features, ViT_features, inception_features, labels = main_extractor_combiner()
# Assuming you've reduced your features to 2D for visualization (e.g., via PCA or t-SNE)
visualize_features(combined_features, labels)
