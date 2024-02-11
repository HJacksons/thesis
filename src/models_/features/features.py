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
import io

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)

wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"))


class FeatureVisualizer:
    def __init__(self, inception_features, ViT_features, combined_features, img_labels):
        self.inception_features = inception_features.cpu().numpy()
        self.ViT_features = ViT_features.cpu().numpy()
        self.combined_features = combined_features.cpu().numpy()
        self.labels = img_labels.cpu().numpy()

    def visualize_features(self, features, labels, title):
        # Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
        df["Label"] = self.labels

        # Prepare the plot
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="PC1", y="PC2", hue="Label", ax=ax)
        ax.set_title(title)
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image = Image.open(buf)
        # Log the image to W&B
        wandb.log({title: [wandb.Image(image, caption=title)]})
        buf.close()
        plt.close(fig)

    def apply_and_visualize(self):
        # Apply visualization separately for each feature set
        self.visualize_features(
            self.inception_features,
            self.labels,
            "PCA Visualization of Inception Features",
        )
        self.visualize_features(
            self.ViT_features,
            self.labels,
            "PCA Visualization of ViT Features",
        )
        self.visualize_features(
            self.combined_features,
            self.labels,
            "PCA Visualization of Combined Features",
        )


inception_features, ViT_features, combined_features, ViT_labels = (
    main_extractor_combiner()
)
visualizer = FeatureVisualizer(inception_features, ViT_features, combined_features)
visualizer.apply_and_visualize()
