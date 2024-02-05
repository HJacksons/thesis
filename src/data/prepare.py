# Prepare dataset by splitting the data into train, validation, and test sets.
# For each class, the images are split randomly into 70% training, 15% validation, and 15% test sets.
import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
load_dotenv()

# Load environment variables
DATASET_PATH = os.getenv("DATASET_PATH")
if DATASET_PATH:
    DATA = os.path.join(DATASET_PATH, "new-potato-leaf-diseases-dataset")
TEST_SIZE = 0.3
VALI_SIZE = 0.5
RANDOM_SIZE = 42


class DatasetPreparer:
    def __init__(
        self,
        dataset=DATA,
        test_size=TEST_SIZE,
        vali_size=VALI_SIZE,
        random_size=RANDOM_SIZE,
        transform=None,
    ):
        self.dataset_name = dataset
        self.data_transforms = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.ToTensor()]
        )
        self.test_size = test_size
        self.vali_size = vali_size
        self.random_state = random_size
        self.train_dataset = None
        self.vali_dataset = None
        self.test_dataset = None

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
        logging.info(f"Number of samples in the training set: {len(train_dataset)}")
        logging.info(f"Number of samples in the validation set: {len(vali_dataset)}")
        logging.info(f"Number of samples in the test set: {len(test_dataset)}")

        return train_dataset, vali_dataset, test_dataset


preparer = DatasetPreparer()
train_ds, vali_ds, test_ds = preparer.prepare_dataset()
