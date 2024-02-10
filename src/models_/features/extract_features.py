import torch
from src.data import data_config

# Define global variables to store features
global features_inception, features_vit

class ModelFeatureExtractor:
    def __init__(self, model, model_type="inception"):
        self.model = model
        self.model_type = model_type
        self.prepare_feature_extractor()

    def prepare_feature_extractor(self):
        """Prepare the model to extract features."""
        if self.model_type == "inception":
            self.attach_inception_hook()
        elif self.model_type == "vit":
            self.attach_ViT_hook()

    def attach_inception_hook(self):
        """Attach a hook to the inception model."""
        global features_inception
        features_inception = None  # Initialize to None
        # Choose the layer to extract features from e.g Mixed_7c
        layer = self.model.model.Mixed_7c
        layer.register_forward_hook(self.inception_hook)

    def attach_ViT_hook(self):
        """Attach a hook to the ViT model."""
        global features_vit
        features_vit = None  # Initialize to None
        # Extract feature from last layer of encoder
        self.model.vit.encoder.layer[-1].register_forward_hook(self.vit_hook)

    def inception_hook(self, module, inputs, output):
        """Hook to extract features from the inception model."""
        global features_inception
        features_inception = output.detach()

    def vit_hook(self, module, inputs, output):
        """Hook to extract features from the ViT model."""
        global features_vit
        # CLS token representation
        features_vit = output[0][:, 0].detach()

    def extract_features(self, loader):
        """Extract features from the given DataLoader."""
        self.model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for images, label in loader:
                images = images.to(data_config.DEVICE)
                self.model(images)  # Trigger hooks and store features globally
                if self.model_type == "inception":
                    features.append(features_inception)
                elif self.model_type == "vit":
                    features.append(features_vit)
                labels.append(label)

        features_tensor = torch.cat(features, dim=0)
        labels_tensor = torch.cat(labels, dim=0)
        return features_tensor, labels_tensor

# Example usage
# model = ...  # Your model definition
# loader = ...  # Your DataLoader
# feature_extractor = ModelFeatureExtractor(model, model_type="inception" or "vit")
# features, labels = feature_extractor.extract_features(loader)
