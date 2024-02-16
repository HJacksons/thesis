import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
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
from torch.optim import Adam
import torch.nn.functional as F

wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"))

# Extract combined features
combined_features, vgg19_features, inception_features, vgg19_labels = (
    main_extractor_combiner()
)
vgg19_labels_tensor = torch.tensor(vgg19_labels, dtype=torch.long)

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
data = Data(x=combined_features, edge_index=edge_index, y=vgg19_labels_tensor)

# Optionally, apply some transformations (e.g., normalization)
data = T.NormalizeFeatures()(data)
#
# logging.basicConfig(level=logging.INFO)
# logging.info(f"Graph data object: {data}")
#
# # Convert to networkx graph
# G = torch_geometric.utils.to_networkx(data, to_undirected=True)
#
# # Plot the graph and color nodes based on VGG19 labels
# plt.figure(figsize=(15, 15))  # Increased figure size for better clarity
# plt.title("Combined Features Graph")
#
# # Use a Fruchterman-Reingold layout to spread nodes and reduce overlap
# pos = nx.spring_layout(G, k=0.15, iterations=20)
# # Explicit color mapping for three classes
# color_map = {
#     0: "red",
#     1: "cyan",
#     2: "green",
# }  # Example: 0 - Healthy, 1 - Disease 1, 2 - Disease 2
# node_colors = [color_map[label.item()] for label in data.y.cpu().numpy()]
#
# nx.draw_networkx_nodes(
#     G,
#     pos,
#     node_size=50,  # Increased node size for visibility
#     node_color=node_colors,  # Color nodes based on VGG19 labels
#     cmap=plt.cm.hsv,  # Color map to differentiate labels
# )
#
# nx.draw_networkx_edges(
#     G,
#     pos,
#     width=0.05,  # Thinner edges to reduce clutter
#     edge_color="blue",  # Edge color
#     alpha=1,  # Transparency for edges
# )
#
# plt.axis("off")
#
# # Save and log the plot
# plot_filename = "combined_features_graph.png"
# plt.savefig(
#     plot_filename, format="png", dpi=300
# )  # Higher DPI for better quality images
# plt.close()
#
# wandb.log({"combined_features_graph": wandb.Image(plot_filename)})

# Create a training mask
num_nodes = data.num_nodes
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_indices = torch.randperm(num_nodes)[: int(0.8 * num_nodes)]  # 80% for training
train_mask[train_indices] = True
data.train_mask = train_mask


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# Parameters
num_features = data.num_features
hidden_dim = 64  # Example hidden dimension
output_dim = len(torch.unique(data.y))  # Assuming y contains class labels

model = GCN(num_features=num_features, hidden_dim=hidden_dim, output_dim=output_dim)
optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Move model and data to the same device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data = data.to(device)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate():
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = float(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
    acc = correct / data.train_mask.sum().item()
    return acc


# Training loop
epochs = 200
for epoch in range(epochs):
    loss = train()
    if epoch % 10 == 0:
        acc = evaluate()
        logging.info(f"Epoch: {epoch+1}, Loss: {loss:.4f}, Acc: {acc:.4f}")
