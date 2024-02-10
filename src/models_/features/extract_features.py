import torch
from src.data import data_config


class ModelFeatureExtractor:
    def __init__(self, model, model_type="inception"):
        self.model = model
        self.model_type = model_type
        self.prepare_feature_extractor()

    def prepare_feature_extractor(self):
        """Prepare the model to extract features."""
        if self.model_type == "inception":
            self.get_feature_extractor_inception()
        elif self.model_type == "vit":
            self.get_feature_extractor_ViT()

    def get_feature_extractor_inception(self):
        """Prepare Inception model for feature extraction."""
        self.model.fc = torch.nn.Identity()

    def get_feature_extractor_ViT(self):
        """Prepare ViT model for feature extraction."""
        # Replace the classifier with an identity function to extract transformer features
        self.model.classifier = torch.nn.Identity()

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
