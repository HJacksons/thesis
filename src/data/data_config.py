import os
import torch
from dotenv import load_dotenv

load_dotenv()

DATASET_PATH = os.getenv("DATASET_PATH")
DATA = os.path.join(DATASET_PATH, "new-potato-leaf-diseases-dataset")
TEST_SIZE = 0.3
VALI_SIZE = 0.5
RANDOM_SIZE = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
EPOCHS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
