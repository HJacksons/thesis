import torch
from torch import nn
from torchvision.models import vgg19, inception_v3
import torch_geometric
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import GCNConv


# Initialize pre-trained models in evaluation mode
vgg19_model = vgg19(pretrained=True).eval()
inception_model = inception_v3(pretrained=True).eval()


# Function to extract avgpool features from Inception-V3
def extract_inception_avgpool_features(model, input_tensor):
    # Manually forward pass up to avgpool layer
    x = model.Conv2d_1a_3x3(input_tensor)
    x = model.Conv2d_2a_3x3(x)
    x = model.Conv2d_2b_3x3(x)
    x = model.maxpool1(x)
    x = model.Conv2d_3b_1x1(x)
    x = model.Conv2d_4a_3x3(x)
    x = model.maxpool2(x)
    x = model.Mixed_5b(x)
    x = model.Mixed_5c(x)
    x = model.Mixed_5d(x)
    x = model.Mixed_6a(x)
    x = model.Mixed_6b(x)
    x = model.Mixed_6c(x)
    x = model.Mixed_6d(x)
    x = model.Mixed_6e(x)
    x = model.Mixed_7a(x)
    x = model.Mixed_7b(x)
    x = model.Mixed_7c(x)
    x = model.avgpool(x)
    # Flatten the output
    return torch.flatten(x, 1)


# Example input tensor (batch_size, channels, height, width)
dummy_input_vgg = torch.rand((32, 3, 224, 224))
dummy_input_inception = torch.rand((32, 3, 299, 299))

# Extract features from VGG19
features_vgg = vgg19_model.features(dummy_input_vgg)

# Extract features from Inception-V3 using the custom function
features_inception = extract_inception_avgpool_features(
    inception_model, dummy_input_inception
)
features_vgg_flattened = features_vgg.view(
    features_vgg.size(0), -1
)  # Reshape to [32, 512*7*7]

print("VGG19 Features Shape:", features_vgg.shape)
print("Inception Features Shape:", features_inception.shape)
# VGG19 Features Shape: torch.Size([32, 512, 7, 7])
# Inception Features Shape: torch.Size([32, 2048])
# Flatten VGG19 features
print("VGG19 :", features_vgg_flattened)
print("Inception:", features_inception)


# Concatenate features
features_combined = torch.cat((features_vgg_flattened, features_inception), dim=1)
print("Combined Features Shape:", features_vgg_flattened.shape)
print("Combined Fea:", features_combined)

num_nodes = features_combined.size(0)
print("Number of nodes:", num_nodes)

# create edges
edge_index = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# create a dummy graph
graph = Data(x=features_combined, edge_index=edge_index)
print(graph)

# Plot the graph

# Convert to networkx graph
G = torch_geometric.utils.to_networkx(graph, to_undirected=True)

# Plot the graph
plt.figure(figsize=(8, 8))
plt.title("Combined Features Graph")
nx.draw_networkx(
    G,
    pos=nx.spring_layout(G, seed=0),
    with_labels=True,
    node_size=800,
    node_color=features_combined.argmax(dim=1),
    cmap="hsv",
    vmin=-2,
    vmax=3,
    width=0.8,
    edge_color="grey",
    font_size=14,
)
plt.show()


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


# initialize the model
input_dim = features_combined.size(1)
output_dim = 768
gcn_model = GCN(input_dim, output_dim)

# apply the model to the graph
enhanced_features = gcn_model(graph)
print(enhanced_features)
