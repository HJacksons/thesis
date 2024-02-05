# Prepare dataset by splitting the data into train, validation, and test sets.
# For each class, the images are split randomly into 70% training, 15% validation, and 15% test sets.
import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

# Load the environment variables
load_dotenv()

dataset_path = os.getenv('DATASET_PATH')
if not dataset_path or not os.path.exists(dataset_path):
    logging.error('Dataset path is invalid or not found.')
    exit(1)

dataset_name = os.path.join(dataset_path, 'new-potato-leaf-diseases-dataset')

# Dataset transformations
data_transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

# Load dataset
dataset = datasets.ImageFolder(dataset_name, transform=data_transforms)

# Get targets/labels from the dataset
targets = np.array([s[1] for s in dataset.samples])

# Split the dataset into train, validation, and test sets
train_indices, temp_indices = train_test_split(np.arange(len(targets)), test_size=0.3, random_state=42, stratify=targets)
vali_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42, stratify=targets[temp_indices])

# Create subsets from the indices
train_dataset = Subset(dataset, train_indices)
vali_dataset = Subset(dataset, vali_indices)
test_dataset = Subset(dataset, test_indices)

