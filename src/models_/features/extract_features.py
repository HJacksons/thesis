import torch
from src.data import data_config


class ModelFeatureExtractor:
    def __init__(self, model, model_type="inception"):
        self.model = model
        self.model_type = model_type
        self.prepare_feature_extractor()
        self.features = None

    def prepare_feature_extractor(self):
        """Prepare the model to extract features."""
        if self.model_type == "inception":
            self.attach_inception_hook()
        elif self.model_type == "vit":
            self.attach_ViT_hook()

    def attach_inception_hook(self):
        """Attach a hook to the inception model."""
        # Choose the layer to extract features from e.g Mixed_7c
        layer = self.model.model.Mixed_7c
        layer.register_forward_hook(self.inception_hook)

    def attach_ViT_hook(self):
        """Attach a hook to the ViT model."""
        # extract feature from last layer of encoder
        self.model.vit.encoder.layer[-1].register_forward_hook(self.vit_hook)

    def inception_hook(self, module, inputs, output):
        """Hook to extract features from the inception model."""
        self.features = output.detach()

    def vit_hook(self, module, inputs, output):
        """Hook to extract features from the ViT model."""
        # CLS token representation
        self.features = output[0][:, 0].detach()

    def extract_features(self, loader):
        """Extract features from the given DataLoader."""
        self.model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for images, label in loader:
                images = images.to(data_config.DEVICE)
                if self.model_type == "inception":
                    feature = self.model(images)
                elif self.model_type == "vit":
                    # For ViT, assume images are preprocessed accordingly
                    feature = self.model(pixel_values=images)[0]
                features.append(feature)
                labels.append(label)
        features_tensor = torch.cat(features, dim=0)
        labels_tensor = torch.cat(labels, dim=0)
        return features_tensor, labels_tensor
