import os
from dotenv import load_dotenv
import logging
import wandb
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    # filename="rename_images.log",
)

load_dotenv()
dataset_path = os.getenv("DATASET_PATH")
dataset_name = f"{dataset_path}/new-potato-leaf-diseases-dataset"

# List lasses in the dataset and the number of images in each class
classes = os.listdir(dataset_name)
logging.info(f" Classes in the dataset: {classes}")
logging.info(f" Number of classes in the dataset: {len(classes)}")

for category in classes:
    class_path = f"{dataset_name}/{category}"
    class_images = os.listdir(class_path)
    logging.info(f" Number of images in {category} class: {len(class_images)}")


# Function to plot class-wise distribution of images
def plot_class_distribution(class_names, class_counts):
    plt.figure(figsize=(10, 5))
    plt.bar(class_names, class_counts, color="skyblue")
    plt.title("Class-wise Distribution of Images")
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


counts = [len(os.listdir(f"{dataset_name}/{category}")) for category in classes]
plot_class_distribution(classes, counts)
