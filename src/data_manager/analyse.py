# List classes in the dataset and the number of images in each class and visualise samples from each class
import os
from dotenv import load_dotenv
import logging
import wandb
import matplotlib.pyplot as plt
import random

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

# Visualise 3 random sample images from each class in the dataset with image size
num_images = 3
fig, axes = plt.subplots(len(classes), num_images, figsize=(10, 10))
for i, category in enumerate(classes):
    class_path = f"{dataset_name}/{category}"
    class_images = os.listdir(class_path)
    for j in range(num_images):
        idx = random.randint(0, len(class_images) - 1)
        img_path = f"{class_path}/{class_images[idx]}"
        img = plt.imread(img_path)
        axes[i, j].imshow(img)
        axes[i, j].set_title(f"{category}\n{img.shape}")
        axes[i, j].axis("off")
plt.tight_layout()
plt.show()

