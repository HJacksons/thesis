import matplotlib.pyplot as plt
import torch
from src.models_.features.extract_features_and_combine import main_extractor_combiner


def visualize_features(features, labels):
    if isinstance(features, torch.Tensor):
        features = features.cpu().detach().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().detach().numpy()

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, cmap="viridis")
    plt.colorbar(scatter)
    plt.xlim(features[:, 0].min() - 1, features[:, 0].max() + 1)
    plt.ylim(features[:, 1].min() - 1, features[:, 1].max() + 1)
    plt.title("Feature Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig("features_visualization.png")  # Save the plot as a PNG file
    # plt.show()  # Comment this out if your environment does not support displaying graphics


# Example usage
combined_features, ViT_features, inception_features, labels = main_extractor_combiner()
visualize_features(combined_features, labels)
