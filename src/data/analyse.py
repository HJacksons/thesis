import os
import random
import logging
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)

# Load environment variables
load_dotenv()
dataset_path = os.getenv("DATASET_PATH")
if not dataset_path or not os.path.exists(dataset_path):
    logging.error("Dataset path is invalid or not found.")
    exit(1)

DATASET = os.path.join(dataset_path, "new-potato-leaf-diseases-dataset")


class DatasetAnalyser:
    def __init__(self, dataset_name=DATASET):
        self.dataset = dataset_name

    def list_classes_and_counts(self):
        """
        List classes in the dataset and the number of images in each class.
        """
        try:
            classes = os.listdir(self.dataset)
        except FileNotFoundError:
            logging.error(f"Failed to list classes in the dataset: {self.dataset}")
            return None, None

        logging.info(f"Classes in the dataset: {classes}")
        logging.info(f"Number of classes in the dataset: {len(classes)}")

        counts = []
        for category in classes:
            class_path = os.path.join(self.dataset, category)
            try:
                class_images = os.listdir(class_path)
                counts.append(len(class_images))
                logging.info(
                    f"Number of images in {category} class: {len(class_images)}"
                )
            except FileNotFoundError:
                logging.error(f"Failed to list images in class: {category}")

        return classes, counts

    def plot_class_distribution(self, class_names, class_counts):
        """
        Plot class-wise distribution of images.
        """
        plt.figure(figsize=(10, 5))
        plt.bar(class_names, class_counts, color="skyblue")
        plt.title("Class-wise Distribution of Images")
        plt.xlabel("Classes")
        plt.ylabel("Number of Images")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def visualize_sample_images(self, classes_, num_images=3):
        """
        Visualize random sample images from each class in the dataset.
        """
        fig, axes = plt.subplots(
            len(classes_), num_images, figsize=(15, len(classes_) * 3)
        )
        for i, category in enumerate(classes_):
            class_path = os.path.join(self.dataset, category)
            try:
                class_images = os.listdir(class_path)
                for j in range(num_images):
                    idx = random.randint(0, len(class_images) - 1)
                    img_path = os.path.join(class_path, class_images[idx])
                    img = plt.imread(img_path)
                    axes[i, j].imshow(img)
                    axes[i, j].set_title(f"{category}\n{img.shape}")
                    axes[i, j].axis("off")
            except FileNotFoundError:
                logging.error(f"Failed to load images from class: {category}")
                continue

        plt.tight_layout()
        plt.show()

        # Checking for corrupt files

    def find_corrupt_images(self):
        c_images = {}

        for c_name in os.listdir(self.dataset):
            class_path = os.path.join(self.dataset, c_name)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    file_path = os.path.join(class_path, filename)
                    try:
                        with Image.open(file_path) as img:
                            img.verify()
                    except (IOError, SyntaxError):
                        if c_name not in c_images:
                            c_images[c_name] = []
                        c_images[c_name].append(filename)

        return c_images


# TODO:  Add a function to remove corrupt images


def main():
    analyser = DatasetAnalyser()
    classes, counts = analyser.list_classes_and_counts()
    if classes and counts:
        analyser.plot_class_distribution(classes, counts)
        analyser.visualize_sample_images(classes)
        corrupt_images = analyser.find_corrupt_images()
        # Reporting corrupt images
        if corrupt_images:
            logging.info("Found corrupt images in the following classes:")
            for class_name, images in corrupt_images.items():
                logging.info(f"{class_name}:")
                for image in images:
                    logging.info(f" - {image}")
        else:
            logging.info("No corrupt images found.")


# main()
