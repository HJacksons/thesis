from src.data.prepare import DatasetPreparer
from src.models_.CNNs.inceptionV3 import Inception
from src.models_.CNNs.vgg19 import VGG19
from src.data import data_config
from src.models_.features.extract_features import ModelFeatureExtractor
import torch
import wandb
import os
import matplotlib.pyplot as plt
import pandas as pd
import logging

wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"))


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

    vgg19_model_path = "src/models_/_saved_models/vgg19_all_layers_100.pth"
    vgg19_model = VGG19()
    vgg19_model = load_model(vgg19_model_path, vgg19_model, data_config.DEVICE)

    # Prepare the dataset
    inception_dataset = DatasetPreparer(model_type="inception")
    _, _, inception_test_loader = inception_dataset.prepare_dataset()

    vgg19_dataset = DatasetPreparer(model_type="vgg19")
    _, _, vgg19_test_loader = vgg19_dataset.prepare_dataset()

    # Extract features for both models
    inception_feature_extractor = ModelFeatureExtractor(
        inception_model, model_type="inception"
    )
    vgg19_feature_extractor = ModelFeatureExtractor(vgg19_model, model_type="vgg19")

    inception_features, inception_labels = inception_feature_extractor.extract_features(
        inception_test_loader
    )
    vgg19_features, vgg19_labels = vgg19_feature_extractor.extract_features(
        vgg19_test_loader
    )

    # Combined features; labels are the same for both models

    combined_features = torch.cat([inception_features, vgg19_features], dim=1)

    logging.info(f"Combined features shape: {combined_features.shape}")
    logging.info(f"ViT features shape: {vgg19_features.shape}")
    logging.info(f"Inception features shape: {inception_features.shape}")
    logging.info(f"ViT labels shape: {vgg19_labels.shape}")
    logging.info(f"Inception labels shape: {inception_labels.shape}")
    logging.info(f"Inception features: {inception_features}")
    logging.info(f"ViT features: {vgg19_features}")
    logging.info(f"Combined features: {combined_features}")
    logging.info(f"ViT labels: {vgg19_labels}")
    logging.info(f"Inception labels: {inception_labels}")
    #
    # wandb.log(
    #     {
    #         "Combined Features": combined_features,
    #         "Inception Features": inception_features,
    #         "ViT Features": ViT_features,
    #         "Inception Labels": inception_labels,
    #         "ViT Labels": ViT_labels,
    #     }
    # )

    # Export features to excel

    # df = pd.DataFrame(combined_features.numpy())
    # df.to_excel("combined_features.xlsx", index=False)
    # df = pd.DataFrame(inception_features.numpy())
    # df.to_excel("inception_features.xlsx", index=False)
    # df = pd.DataFrame(ViT_features.numpy())
    # df.to_excel("ViT_features.xlsx", index=False)
    # df = pd.DataFrame(inception_labels.numpy())
    # df.to_excel("inception_labels.xlsx", index=False)
    # df = pd.DataFrame(ViT_labels.numpy())
    # df.to_excel("ViT_labels.xlsx", index=False)

    return combined_features, vgg19_features, inception_features, vgg19_labels


if __name__ == "__main__":
    main_extractor_combiner()

# TODO
# Analyse the combined features
