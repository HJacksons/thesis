import torch
from src.data import data_config

# Initialize global variables to store features as lists
features_inception = []
features_vgg = []


class ModelFeatureExtractor:
    def __init__(self, model, model_type="inception"):
        global features_inception, features_vgg
        # Clear the lists in case of previous runs
        features_inception.clear()
        features_vgg.clear()
        self.model = model
        self.model_type = model_type
        self.prepare_feature_extractor()

    def prepare_feature_extractor(self):
        if self.model_type == "inception":
            self.attach_inception_hook()
        elif self.model_type == "vgg":
            self.attach_vgg_hook()

    def attach_inception_hook(self):
        layer = self.model.model.Mixed_7c
        layer.register_forward_hook(self.inception_hook)

    def attach_vgg_hook(self):
        layer = self.model.features
        layer.register_forward_hook(self.vgg_hook)

    def inception_hook(self, module, inputs, output):
        # Apply adaptive average pooling and flatten the output
        features = torch.flatten(
            torch.nn.functional.adaptive_avg_pool2d(output, (1, 1)), start_dim=1
        ).detach()
        features_inception.append(features.cpu())

    def vgg_hook(self, module, inputs, output):
        features = torch.flatten(output, start_dim=1).detach()
        features_vgg.append(features.cpu())

    def extract_features(self, loader):
        self.model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for images, label in loader:
                images = images.to(data_config.DEVICE)
                self.model(images)  # This triggers the hook
                labels.append(label)

        # Choose the correct feature list based on the model type
        if self.model_type == "inception":
            features = features_inception
        elif self.model_type == "vgg":
            features = features_vgg

        # Concatenate all collected features and labels
        features_tensor = torch.cat(features, dim=0)
        labels_tensor = torch.cat(labels, dim=0)
        return features_tensor, labels_tensor
