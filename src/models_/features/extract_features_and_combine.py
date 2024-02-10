from src.data.prepare import DatasetPreparer
from src.models_.CNNs.inceptionV3 import Inception
from src.models_.ViT.ViT import ViT
from src.data import data_config
from src.models_.features.extract_features import ModelFeatureExtractor
import torch
import logging

# Load the Inception model
inception_model_path = "src/models_/_saved_models/inceptionv3100.pth"
inception_model = Inception()
inception_model.load_state_dict(
    torch.load(inception_model_path, map_location=data_config.DEVICE)
)
inception_model.to(data_config.DEVICE)

ViT_model_path = "src/models_/_saved_models/ViTModel100.pth"
ViT_model = ViT()
ViT_model.load_state_dict(torch.load(ViT_model_path, map_location=data_config.DEVICE))
ViT_model.to(data_config.DEVICE)


# Prepare the dataset
inception_dataset = DatasetPreparer(model_type="inception")
_, _, inception_test_loader = inception_dataset.prepare_dataset()

ViT_dataset = DatasetPreparer(model_type="vit")
_, _, vit_test_loader = ViT_dataset.prepare_dataset()

# Extract features for both models
inception_feature_extractor = ModelFeatureExtractor(
    inception_model, model_type="inception"
)
ViT_feature_extractor = ModelFeatureExtractor(ViT_model, model_type="vit")

inception_features, inception_labels = inception_feature_extractor.extract_features(
    inception_test_loader
)
ViT_features, ViT_labels = ViT_feature_extractor.extract_features(vit_test_loader)

# Combined features, labels are the same for both models
combined_features = torch.cat([inception_features, ViT_features], dim=1)

logging.info(f"Combined features shape: {combined_features.shape}")
logging.info(f"Labels shape: {inception_labels.shape}")
