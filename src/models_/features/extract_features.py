import torch
from src.data import data_config

global features_inception, features_vgg  # Updated to use features_vgg


class ModelFeatureExtractor:
    def __init__(self, model, model_type="inception"):
        self.model = model
        self.model_type = model_type
        self.prepare_feature_extractor()

    # Prepare the feature extractor
    def prepare_feature_extractor(self):
        if self.model_type == "inception":
            self.attach_inception_hook()
        elif self.model_type == "vgg":  # Updated to check for "vgg"
            self.attach_vgg_hook()  # Updated to call attach_vgg_hook

    # Attach a hook to the Inception model
    def attach_inception_hook(self):
        global features_inception
        features_inception = None
        layer = self.model.model.Mixed_7c
        layer.register_forward_hook(self.inception_hook)

    # Attach a hook to the VGG model
    def attach_vgg_hook(self):  # New method for VGG
        global features_vgg
        features_vgg = None
        # Assuming you want to attach the hook to the last conv layer before the classifier
        layer = self.model.features[-1]  # Adjust based on your VGG model structure
        layer.register_forward_hook(self.vgg_hook)

    def inception_hook(self, module, inputs, output):
        global features_inception
        pooled_output = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
        features_inception = torch.flatten(pooled_output, 1).detach()

    def vgg_hook(self, module, inputs, output):  # New hook function for VGG
        global features_vgg
        # Assuming output is a 4D tensor, flatten directly without pooling
        features_vgg = torch.flatten(output, start_dim=1).detach()

    # Extract features from the test loader
    def extract_features(self, loader):
        self.model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for images, label in loader:
                images = images.to(data_config.DEVICE)
                self.model(images)  # This will call the appropriate hook
                if self.model_type == "inception":
                    features.append(features_inception)
                elif self.model_type == "vgg":  # Updated to handle VGG features
                    features.append(features_vgg)
                labels.append(label)

        features_tensor = torch.cat(features, dim=0)
        labels_tensor = torch.cat(labels, dim=0)
        return features_tensor, labels_tensor
