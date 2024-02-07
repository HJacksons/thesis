# Prepare dataset by splitting the data into train, validation, and test sets.
# For each class, the images are split randomly into 70% training, 15% validation, and 15% test sets.
import os
import sys
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from src.data import data_config
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)


class CustomDataset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


class DatasetPreparer:
    def __init__(self, dataset_path=data_config.DATA, test_size=data_config.TEST_SIZE, vali_size=data_config.VALI_SIZE,
                 random_state=data_config.RANDOM_SIZE):
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.vali_size = vali_size
        self.random_state = random_state

        # Define transformations
        self.data_transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0), ratio=(0.95, 1.05),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(hue=0.021, saturation=0.8, brightness=0.43),
            transforms.RandomAffine(degrees=0, translate=(0.13, 0.13), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.data_transforms_vali_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def prepare_dataset(self):
        # Load the base dataset without transformations
        base_dataset = datasets.ImageFolder(self.dataset_path)

        # Get targets for stratification
        targets = [s[1] for s in base_dataset.samples]

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
            stratify=[targets[i] for i in temp_indices],
        )

        # Creates datasets with transformations
        train_dataset = CustomDataset(base_dataset, train_indices, transform=self.data_transforms_train)
        vali_dataset = CustomDataset(base_dataset, vali_indices, transform=self.data_transforms_vali_test)
        test_dataset = CustomDataset(base_dataset, test_indices, transform=self.data_transforms_vali_test)

        # Print the number of samples in each set logging.info(f"Train samples: {len(train_dataset)}, Validation
        # samples: {len(vali_dataset)}, Test samples: {len(test_dataset)}")

        # Create data loaders
        train_dl = DataLoader(train_dataset, batch_size=data_config.BATCH_SIZE, shuffle=True, num_workers=0,
                              pin_memory=True)
        vali_dl = DataLoader(vali_dataset, batch_size=data_config.BATCH_SIZE, shuffle=False, num_workers=0,
                             pin_memory=True)
        test_dl = DataLoader(test_dataset, batch_size=data_config.BATCH_SIZE, shuffle=False, num_workers=0,
                             pin_memory=True)

        logging.info("Dataset preparation complete.")
        return train_dl, vali_dl, test_dl
