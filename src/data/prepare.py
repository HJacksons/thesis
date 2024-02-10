# Prepare dataset by splitting the data into train, validation, and test sets.
# For each class, the images are split randomly into 70% training, 15% validation, and 15% test sets.
import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.data import data_config
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)


class DatasetPreparer:
    def __init__(
            self,
            dataset=data_config.DATA,
            test_size=data_config.TEST_SIZE,
            vali_size=data_config.VALI_SIZE,
            random_state=data_config.RANDOM_STATE,
            model_type="inception",
    ):
        self.dataset_name = dataset
        self.test_size = test_size
        self.vali_size = vali_size
        self.random_state = random_state
        self.data_transforms = self.get_transforms_for_model(model_type)

    @staticmethod
    def get_transforms_for_model(model_type):
        if model_type == 'inception':
            resize_crop_size = 299
        elif model_type == 'vit':
            resize_crop_size = 256
        else:
            raise ValueError("Unsupported model type")

        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(
                resize_crop_size,
                scale=(0.8, 1.0),
                ratio=(0.95, 1.05),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(hue=0.021, saturation=0.8, brightness=0.43),
            transforms.RandomAffine(
                degrees=0, translate=(0.13, 0.13), scale=(0.95, 1.05)
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        return data_transforms

    def prepare_dataset(self):
        # Load dataset
        dataset = datasets.ImageFolder(
            self.dataset_name, transform=self.data_transforms
        )

        # Get targets/labels from the dataset
        targets = np.array([s[1] for s in dataset.samples])

        # Split the dataset into train, validation, and test sets
        train_indices, temp_indices = train_test_split(
            np.arange(len(targets)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=targets,
        )
        vali_indices, test_indices = train_test_split(
            temp_indices,
            test_size=self.vali_size,
            random_state=self.random_state,
            stratify=targets[temp_indices],
        )

        # Create subsets from the indices
        train_dataset = Subset(dataset, train_indices)
        vali_dataset = Subset(dataset, vali_indices)
        test_dataset = Subset(dataset, test_indices)

        # Print the number of samples in each set
        # logging.info(f"Number of samples in the training set: {len(train_dataset)}")
        # logging.info(f"Number of samples in the validation set: {len(vali_dataset)}")
        # logging.info(f"Number of samples in the test set: {len(test_dataset)}")

        # Create data loaders
        train_dl = DataLoader(
            train_dataset,
            batch_size=data_config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        vali_dl = DataLoader(
            vali_dataset,
            batch_size=data_config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        test_dl = DataLoader(
            test_dataset,
            batch_size=data_config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Print the number of batches in each set
        # logging.info(f"Number of batches in the training set: {len(train_dl)}")
        # logging.info(f"Number of batches in the validation set: {len(vali_dl)}")
        # logging.info(f"Number of batches in the test set: {len(test_dl)}")
        logging.info("Dataset preparation complete.")

        return train_dl, vali_dl, test_dl
