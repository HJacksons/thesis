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
import io
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
        self.inception_features = inception_features.cpu()
        self.ViT_features = ViT_features.cpu()
        self.combined_features = combined_features.cpu()
        self.all_features = np.concatenate(
            [inception_features, ViT_features, combined_features], axis=0
        )
        self.split_1 = len(self.inception_features)
        self.split_2 = self.split_1 + len(self.ViT_features)
        self.labels = (
            ["Inception"] * len(self.inception_features)
            + ["ViT"] * len(self.ViT_features)
            + ["Combined"] * len(self.combined_features)
        )

    def apply_tsne(self):
        tsne = TSNE(n_components=2, perplexity=30.0, n_iter=1000, random_state=42)
        tsne_results = tsne.fit_transform(self.all_features)
        self.plot_features(tsne_results, title="t-SNE Visualization of Features")

    def apply_pca(self):
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.all_features)
        data = []
        for i, label in enumerate(self.labels):
            data.append([pca_result[i, 0], pca_result[i, 1], label])
        df = pd.DataFrame(data, columns=["PC1", "PC2", "label"])
        wandb.log(
            {
                "PCA Visualization": wandb.plot.scatter(
                    df,
                    "PCA1",
                    "PCA2",
                    title="PCA Visualization of Features",
                    labels={"PCA1": "PCA Component 1", "PCA2": "PCA Component 2"},
                )
            }
        )

    def plot_features(self, results, title):
        plt.figure(figsize=(12, 8))
        plt.scatter(
            results[: self.split_1, 0],
            results[: self.split_1, 1],
            label="Inception Features",
            alpha=0.5,
        )
        plt.scatter(
            results[self.split_1 : self.split_2, 0],
            results[self.split_1 : self.split_2, 1],
            label="ViT Features",
            alpha=0.5,
        )
        plt.scatter(
            results[self.split_2 :, 0],
            results[self.split_2 :, 1],
            label="Combined Features",
            alpha=0.5,
        )
        plt.legend()
        plt.title(title)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image = Image.open(buf)
        wandb.log({title: [wandb.Image(image, caption=title)]})
        buf.close()
        plt.clf()

    def pair_plot(self):
        df = pd.DataFrame(
            self.all_features,
            columns=[f"Feature_{i}" for i in range(self.all_features.shape[1])],
        )
        labels = (
            ["Inception"] * len(self.inception_features)
            + ["ViT"] * len(self.ViT_features)
            + ["Combined"] * len(self.combined_features)
        )
        df["Label"] = labels
        sns.pairplot(df, hue="Label", plot_kws={"alpha": 0.5})
        plt.title("Pair Plot of Features")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image = Image.open(buf)
        wandb.log(
            {
                "Pair Plot of Features": [
                    wandb.Image(image, caption="Pair Plot of Features")
                ]
            }
        )
        buf.close()
        plt.clf()


inception_features, ViT_features, combined_features = main_extractor_combiner()
visualizer = FeatureVisualizer(inception_features, ViT_features, combined_features)
visualizer.apply_tsne()
visualizer.apply_pca()
visualizer.pair_plot()
