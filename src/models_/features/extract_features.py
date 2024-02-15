import torch
from src.data import data_config


class ModelFeatureExtractor:
    def __init__(self, model, model_type="inception"):
        self.model = model
        self.model_type = model_type
        self.features = None  # Store features as an instance attribute
        self.prepare_feature_extractor()

    def prepare_feature_extractor(self):
        if self.model_type == "inception":
            # Attach the hook to the output of the adaptive average pooling layer
            self.attach_hook(
                # For Inception, attach the hook to the adaptive average pooling layer
                self.attach_hook(self.model.model.avgpool, self.generic_hook)
            )  # Adjusted for Inception
        elif self.model_type == "vgg":
            # Attach the hook to the last layer of the features component
            self.attach_hook(
                self.model.model.features[-1], self.generic_hook
            )  # Adjusted for VGG

    def attach_hook(self, layer, hook_function):
        layer.register_forward_hook(hook_function)

    def generic_hook(self, module, inputs, output):
        # No need for conditional pooling/flattening based on model type here
        # Assuming output is already in the desired format for both models
        self.features = output.detach()

    def extract_features(self, loader):
        self.model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for images, label in loader:
                images = images.to(data_config.DEVICE)
                self.model(images)  # Triggers the hook and updates self.features
                if self.features is not None:
                    features.append(self.features)
                    labels.append(label)

        features_tensor = torch.cat(features, dim=0)
        labels_tensor = torch.cat(labels, dim=0)
        return features_tensor, labels_tensor
