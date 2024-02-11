import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from src.models_.features.extract_features_and_combine import main_extractor_combiner
import seaborn as sns
import pandas as pd
from dotenv import load_dotenv
import os
import logging
import wandb
from PIL import Image

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)

wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"))


def visualize_features(features, labels, title):
    """
    Visualizes the features using PCA and logs the plot to Weights & Biases.
    """
    # Convert features to CPU and numpy, then replace NaNs and Infs
    features_np = features.cpu().detach().numpy()
    features_np = np.nan_to_num(features_np, nan=0.0, posinf=0.0, neginf=0.0)

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_np)

    # Prepare DataFrame for plotting
    df = pd.DataFrame(
        {
            "PC1": pca_result[:, 0],
            "PC2": pca_result[:, 1],
            "Label": labels.cpu().numpy(),
        }
    )

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(df["PC1"], df["PC2"], c=df["Label"], cmap="viridis", alpha=0.5)
    legend = ax.legend(*scatter.legend_elements(), title="Labels")
    ax.add_artist(legend)
    ax.set_title(title)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    # Log plot to W&B
    wandb.log({title: wandb.Image(plt)})
    plt.close()


def main():
    # Extract and combine features
    combined_features, ViT_features, inception_features, ViT_labels = (
        main_extractor_combiner()
    )

    # Visualize and log features
    visualize_features(inception_features, ViT_labels, "Inception Features PCA")
    visualize_features(ViT_features, ViT_labels, "ViT Features PCA")
    visualize_features(combined_features, ViT_labels, "Combined Features PCA")

    # Log feature vectors to W&B as histograms
    wandb.log(
        {
            "Inception Features Histogram": wandb.Histogram(
                inception_features.cpu().detach().numpy()
            ),
            "ViT Features Histogram": wandb.Histogram(
                ViT_features.cpu().detach().numpy()
            ),
            "Combined Features Histogram": wandb.Histogram(
                combined_features.cpu().detach().numpy()
            ),
        }
    )


if __name__ == "__main__":
    main()

inception_features, ViT_features, combined_features, ViT_labels = (
    main_extractor_combiner()
)
