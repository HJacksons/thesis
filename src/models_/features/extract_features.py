import torch
from src.data import data_config
from src.data.prepare import DatasetPreparer
from src.models_.CNNs.inceptionV3 import Inception
from src.models_.ViT.ViT import ViT
from src.models_.CNNs.vgg19 import VGG19
import torch.nn as nn

# Initialize the models
inception_model = Inception()
vgg19_model = VGG19()

# Load the trained weights
inception_model.load_state_dict(
    torch.load("src/models_/_saved_models/inceptionv3100.pth")
)
vgg19_model.load_state_dict(
    torch.load("src/models_/_saved_models/vgg19_all_layers_100.pth")
)

# Set models to evaluation mode
inception_model.eval()
vgg19_model.eval()


class InceptionFeatures(nn.Module):
    def __init__(self, trained_model):
        super(InceptionFeatures, self).__init__()
        # Copy all layers except the final fully connected layer
        self.features = nn.Sequential(*list(trained_model.children())[:-1])

    def forward(self, x):
        # Forward pass through the modified architecture
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return x


inception_features_model = InceptionFeatures(inception_model)


class VGG19Features(nn.Module):
    def __init__(self, trained_model):
        super(VGG19Features, self).__init__()
        # Retain the convolutional base
        self.features = trained_model.model.features

    def forward(self, x):
        # Forward pass through the convolutional base
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return x


vgg19_features_model = VGG19Features(vgg19_model)

# Prepare the dataset
inception_dataset = DatasetPreparer(model_type="inception")
inception_train_loader, _, _ = inception_dataset.prepare_dataset()

vgg19_dataset = DatasetPreparer(model_type="vgg19")
vgg19_train_loader, _, _ = vgg19_dataset.prepare_dataset()


def extract_features(model, dataloader):
    model.eval()  # Ensure the model is in evaluation mode
    features_list = []
    with torch.no_grad():  # No gradients needed
        for (
            images,
            _,
        ) in dataloader:  # Assuming labels are not needed for feature extraction
            features = model(images)  # Extract features
            features_list.append(features.cpu())
    features = torch.cat(features_list, dim=0)
    return features


inception_features = extract_features(
    inception_features_model, inception_train_loader
)  # or train_loader, vali_loader
vgg19_features = extract_features(vgg19_features_model, vgg19_train_loader)  # Same here

print("Inception features shape:", inception_features.shape)
print("VGG19 features shape:", vgg19_features.shape)
