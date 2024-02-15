import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import KDTree
import numpy as np
import logging
import os
import wandb
import matplotlib.pyplot as plt
import torch_geometric.utils
import networkx as nx
from src.models_.features.extract_features_and_combine import main_extractor_combiner
from src.data import data_config

wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"))

# Extract combined features
combined_features, vgg19_features, inception_features, vgg19_labels = (
    main_extractor_combiner()
)

# Convert features to numpy for KDTree
features_np = combined_features.cpu().numpy()

# Use KDTree for efficient k-NN search
kdtree = KDTree(features_np)
k = 5  # Number of neighbors to connect to, adjust based on your dataset
distances, indices = kdtree.query(
    features_np, k=k + 1
)  # k+1 because the query includes the point itself

# Prepare edge_index for PyTorch Geometric
source_nodes = np.repeat(np.arange(features_np.shape[0]), k)
target_nodes = indices[
    :, 1:
].flatten()  # Exclude the first column which is the point itself
edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

# Move edge_index to the same device as combined_features
edge_index = edge_index.to(combined_features.device)

# Create a graph data object
data = Data(x=combined_features, edge_index=edge_index)

# Optionally, apply some transformations (e.g., normalization)
data = T.NormalizeFeatures()(data)

logging.basicConfig(level=logging.INFO)
logging.info(f"Graph data object: {data}")

# Convert to networkx graph
G = torch_geometric.utils.to_networkx(data, to_undirected=True)

# Plot the graph and color nodes based on VGG19 labels
plt.figure(figsize=(8, 8))
plt.title("Combined Features Graph")
nx.draw_networkx(
    G,
    pos=nx.spring_layout(G, seed=42),
    with_labels=False,  # Set to True if you want labels, but it might clutter the visualization
    node_size=50,  # Adjust size for better visibility
    node_color=vgg19_labels.numpy(),  # Ensure labels are in correct format
    cmap="hsv",
    vmin=-2,
    vmax=3,
    width=0.5,
    edge_color="grey",
    font_size=12,
)
plt.axis("off")

# Save and log the plot
plot_filename = "combined_features_graph.png"
plt.savefig(plot_filename, format="png")
plt.close()

wandb.log({"combined_features_graph": wandb.Image(plot_filename)})
