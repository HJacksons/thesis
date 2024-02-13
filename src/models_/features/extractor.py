import torch
from torchvision import models
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, pretrained_model):
        super(FeatureExtractor, self).__init__()
        self.pretrained_model = pretrained_model
        # Assuming we want to extract features from Mixed_7c for Inception model
        self.pretrained_model.Mixed_7c.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def forward(self, x):
        _ = self.pretrained_model(x)  # We don't actually need the model's output
        return self.features


# Load the pretrained Inception model
weights = models.inception.Inception_V3_Weights.DEFAULT
inception_model = models.inception_v3(weights=weights)
inception_model.eval()  # Set the model to evaluation mode

# Wrap the inception model with the feature extractor
feature_extractor = FeatureExtractor(inception_model)

# Now you can use feature_extractor to extract features
# For example, using a dummy input
dummy_input = torch.randn(1, 3, 299, 299)  # Batch size 1, 3 color channels, 299x299
if torch.cuda.is_available():
    dummy_input = dummy_input.to("cuda")
    feature_extractor.to("cuda")

features = feature_extractor(dummy_input)
print(features.shape)  # This will print the shape of the output from Mixed_7c
print(features)
