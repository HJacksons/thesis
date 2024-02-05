import os
from dotenv import load_dotenv

load_dotenv()

DATASET_PATH = os.getenv("DATASET_PATH")
DATA = os.path.join(DATASET_PATH, "new-potato-leaf-diseases-dataset")
TEST_SIZE = 0.3
VALI_SIZE = 0.5
RANDOM_SIZE = 42
BATCH_SIZE = 32
