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
            # Ensure this is the correct layer for your Inception model
            self.attach_hook(
                (
                    self.model.Mixed_7c
                    if hasattr(self.model, "Mixed_7c")
                    else self.model.model.Mixed_7c
                ),
                self.generic_hook,
            )
        elif self.model_type == "vgg":
            # Ensure this is the correct layer for your VGG model
            self.attach_hook(self.model.features[-1], self.generic_hook)

    def attach_hook(self, layer, hook_function):
        layer.register_forward_hook(hook_function)

    def generic_hook(self, module, inputs, output):
        if self.model_type == "inception":
            output = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
        # For VGG, directly flatten the output if no pooling is needed
        self.features = torch.flatten(output, start_dim=1).detach()

    def extract_features(self, loader):
        self.model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for images, label in loader:
                images = images.to(data_config.DEVICE)
                self.model(images)  # Triggers the hook and updates self.features
                features.append(self.features)
                labels.append(label)

        features_tensor = torch.cat(features, dim=0)
        labels_tensor = torch.cat(labels, dim=0)
        return features_tensor, labels_tensor
