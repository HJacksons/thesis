import torch
from src.data import data_config

# Global variables to store features
global features_inception, features_vgg


class ModelFeatureExtractor:
    def __init__(self, model, model_type="inception"):
        self.model = model
        self.model_type = model_type
        self.prepare_feature_extractor()

    # Prepare the feature extractor
    def prepare_feature_extractor(self):
        if self.model_type == "inception":
            self.attach_inception_hook()
        elif self.model_type == "vgg19":
            self.attach_vgg_hook()

    # Attach a hook to the Inception model
    def attach_inception_hook(self):
        global features_inception
        features_inception = None
        # Assuming 'Mixed_7c' is the correct layer for feature extraction
        layer = self.model.model.Mixed_7c
        layer.register_forward_hook(self.inception_hook)

    # Attach a hook to the VGG model
    def attach_vgg_hook(self):
        global features_vgg
        features_vgg = None
        # Assuming 'features' is the last conv layer in VGG for feature extraction
        layer = self.model.model.features
        layer.register_forward_hook(self.vgg_hook)

    def inception_hook(self, module, inputs, output):
        global features_inception
        # For 4D output, apply adaptive average pooling to make it [batch_size, channels, 1, 1]
        pooled_output = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
        # Flatten the output to make it [batch_size, channels]
        features_inception = torch.flatten(pooled_output, 1).detach()

    def vgg_hook(self, module, inputs, output):
        global features_vgg
        processed_output = torch.flatten(output, start_dim=1).detach()

        print(
            "Output stats -- Mean:",
            processed_output.mean().item(),
            "Max:",
            processed_output.max().item(),
        )  # Debugging line

        # Assuming output is already in the desired shape [batch_size, features]
        features_vgg = torch.flatten(output, start_dim=1).detach()

    # Extract features from a single image in the test loader
    # Extract features from the test loader

    def extract_features(self, loader):
        self.model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for images, label in loader:
                images = images.to(data_config.DEVICE)
                self.model(images)  # This will call the hook
                if self.model_type == "inception":
                    features.append(features_inception)
                elif self.model_type == "vgg19":
                    features.append(features_vgg)
                labels.append(label)

        features_tensor = torch.cat(features, dim=0)
        labels_tensor = torch.cat(labels, dim=0)
        return features_tensor, labels_tensor
