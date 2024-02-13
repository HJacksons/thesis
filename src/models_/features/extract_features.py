import torch
from src.data import data_config
import random

global features_inception, features_vit


class ModelFeatureExtractor:
    def __init__(self, model, model_type="inception"):
        self.model = model
        self.model_type = model_type
        self.prepare_feature_extractor()

    # Prepare the feature extractor
    def prepare_feature_extractor(self):

        if self.model_type == "inception":
            self.attach_inception_hook()
        elif self.model_type == "vit":
            self.attach_ViT_hook()

    # Attach a hook to the Inception model
    def attach_inception_hook(self):
        global features_inception
        features_inception = None
        layer = self.model.model.Mixed_7c
        layer.register_forward_hook(self.inception_hook)

    # Attach a hook to the ViT model
    def attach_ViT_hook(self):
        global features_vit
        features_vit = None
        self.model.vit.encoder.layer[-1].register_forward_hook(self.vit_hook)

    def inception_hook(self, module, inputs, output):
        global features_inception
        # For 4D output, apply adaptive average pooling to make it [batch_size, channels, 1, 1]
        pooled_output = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
        # Flatten the output to make it [batch_size, channels]
        features_inception = torch.flatten(pooled_output, 1).detach()

    def vit_hook(self, module, inputs, output):
        global features_vit
        # If output is already in the desired shape [batch_size, features], just detach
        features_vit = output[0][:, 0].detach()

    # # Extract features from the test loader
    # def extract_features(self, loader):
    #     self.model.eval()
    #     features = []
    #     labels = []
    #     with torch.no_grad():
    #         for images, label in loader:
    #             images = images.to(data_config.DEVICE)
    #             self.model(images)  # This will call the hook
    #             if self.model_type == "inception":
    #                 features.append(features_inception)
    #             elif self.model_type == "vit":
    #                 features.append(features_vit)
    #             labels.append(label)
    #
    #     features_tensor = torch.cat(features, dim=0)
    #     labels_tensor = torch.cat(labels, dim=0)
    #     return features_tensor, labels_tensor

    # Extract features from the test loader (one image only)
    def extract_features(self, loader):
        self.model.eval()
        features = None
        labels = None
        random_index = random.randint(
            0, len(loader.dataset) - 1
        )  # Use the length of the dataset
        with torch.no_grad():
            images, label = loader.dataset[random_index]
            images = images.to(data_config.DEVICE).unsqueeze(0)  # Add a batch dimension
            self.model(images)  # This will call the hook
            if self.model_type == "inception":
                features = features_inception
            elif self.model_type == "vit":
                features = features_vit
            labels = label  # Assign label to labels outside the conditional block

        features_tensor = features.unsqueeze(0)  # Add a batch dimension to features
        labels_tensor = torch.tensor(labels).unsqueeze(
            0
        )  # Add a batch dimension to labels and make sure it's a tensor
        return features_tensor, labels_tensor
