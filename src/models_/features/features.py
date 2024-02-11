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


class FeatureVisualizer:
    def __init__(self, inception_features, ViT_features, combined_features):
        self.inception_features = inception_features.cpu().numpy()
        self.ViT_features = ViT_features.cpu().numpy()
        self.combined_features = combined_features.cpu().numpy()

    def visualize_features(self, features, labels, title):
        # Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
        df["Label"] = labels

        # Log to W&B
        wandb.log(
            {
                title: wandb.plot.scatter(
                    df,
                    "PC1",
                    "PC2",
                    title=title,
                    labels={"PC1": "PCA Component 1", "PC2": "PCA Component 2"},
                )
            }
        )

    def apply_and_visualize(self):
        # Apply visualization separately for each feature set
        self.visualize_features(
            self.inception_features,
            ["Inception"] * len(self.inception_features),
            "PCA Visualization of Inception Features",
        )
        self.visualize_features(
            self.ViT_features,
            ["ViT"] * len(self.ViT_features),
            "PCA Visualization of ViT Features",
        )
        self.visualize_features(
            self.combined_features,
            ["Combined"] * len(self.combined_features),
            "PCA Visualization of Combined Features",
        )


inception_features, ViT_features, combined_features = main_extractor_combiner()
visualizer = FeatureVisualizer(inception_features, ViT_features, combined_features)
visualizer.apply_and_visualize()
