from src.data.prepare import DatasetPreparer
from src.models_.CNNs.inceptionV3 import Inception
from src.models_.ViT.ViT import ViT
from src.data import data_config
from src.models_.features.extract_features import ModelFeatureExtractor
import torch
import logging


def load_model(model_path, model, device):
    model.load_state_dict(torch.load(model_path, map_location=data_config.DEVICE))
    model.to(data_config.DEVICE)
    return model


def main_extractor_combiner():
    # Load the Inception and ViT model
    inception_model_path = "src/models_/_saved_models/inceptionv3100.pth"
    inception_model = Inception()
    inception_model = load_model(
        inception_model_path, inception_model, data_config.DEVICE
    )

    ViT_model_path = "src/models_/_saved_models/ViTModel_224_100.pth"
    ViT_model = ViT()
    ViT_model = load_model(ViT_model_path, ViT_model, data_config.DEVICE)

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

    # Combined features; labels are the same for both models
    ViT_labels = inception_labels
    combined_features = torch.cat([inception_features, ViT_features], dim=1)

    logging.info(f"Combined features shape: {combined_features.shape}")
    logging.info(f"Inception features: {inception_features.shape}")
    logging.info(f"ViT features: {ViT_features.shape}")
    logging.info(f"ViT features: {ViT_features}")
    logging.info(f"Inception features: {inception_features}")
    logging.info(f"Combined features: {combined_features}")

    return combined_features, ViT_features, inception_features, ViT_labels


if __name__ == "__main__":
    main_extractor_combiner()

# TODO
# Analyse the combined features
# View features in tensorboard
