import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from sklearn.metrics.pairwise import cosine_similarity
from src.models_.features.extract_features_and_combine import main_extractor_combiner

# Assuming combined_features is a tensor of shape [num_images, num_features]
combined_features, vgg19_features, inception_features, vgg19_labels = (
    main_extractor_combiner()
)

# Option 1: Fully connected graph (not recommended for large datasets)
# edge_index = torch.combinations(torch.arange(features.size(0)), r=2).t()

# Option 2: Connect nodes based on similarity (simplified example)
# Calculate cosine similarity between feature vectors
similarity = cosine_similarity(combined_features.cpu().numpy())
# Convert to torch tensor
similarity = torch.from_numpy(similarity).type(torch.float)

# Define a threshold for connecting nodes
threshold = 0.9  # This is an arbitrary value; adjust based on your dataset

# Create edge indices based on the threshold
edge_index = (similarity > threshold).nonzero(as_tuple=False).t()

# Ensure edge_index is on the same device as features
edge_index = edge_index.to(combined_features.device)

# Create a graph data object
data = Data(x=combined_features, edge_index=edge_index)

# Optionally, apply some transformations (e.g., normalization)
data = T.NormalizeFeatures()(data)

print(data)
