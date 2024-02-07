import os

# Get the absolute path to the project root directory
project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

# Construct the path to the data directory
DATASET_PATH = os.path.join(project_root_dir, 'data')

print(f"Dataset path: {DATASET_PATH}")
