import os
import torch
from src.models_.CNNs.inceptionV3 import Inception
from src.models_.ViT.ViT import ViT
from src.data.prepare import DatasetPreparer
from src.data.prepare import data_config

# Load the Inception model
model_path = 'src/models_/_saved_models/inceptionv3100.pth'
model = Inception()
model.load_state_dict(torch.load(model_path, map_location=data_config.DEVICE))
model.to(data_config.DEVICE)

# Load the data
dataset = DatasetPreparer()
train_loader, vali_loader, test_loader = dataset.prepare_dataset()


# lets define a function to extract features from the test loader
def get_feature_extractor(model):
    # Remove the final fully connected layer to get the features
    model.fc = torch.nn.Identity()
    return model


feature_extractor_inception = get_feature_extractor(model)
feature_extractor_inception.eval()
feature_extractor_inception.to(data_config.DEVICE)


# Extract features from the test loader
def extract_features(loader, feature_extractor):
    features = []
    labels = []
    with torch.no_grad():
        for images, label in loader:
            images = images.to(data_config.DEVICE)
            feature = feature_extractor(images)
            features.append(feature)
            labels.append(label)
            features_tensor = torch.cat(features, dim=0)
            labels_tensor = torch.cat(labels, dim=0)
    return features_tensor, labels_tensor


incept_features, incept_labels = extract_features(test_loader, feature_extractor_inception)
# print the list of features
print(incept_features)





# # Load the VIT model
# model_path = '../_saved_models/ViTModel100.pth'
# model = ViT()
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# model.to(torch.device('cpu'))
# model.eval()
