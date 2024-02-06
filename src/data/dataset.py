from prepare import DatasetPreparer
from analyse import DatasetAnalyser
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)

if __name__ == "__main__":
    # Prepare the dataset
    preparer = DatasetPreparer()
    train_loader, vali_loader, test_loader = preparer.prepare_dataset()

    # Analyse the dataset
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

        # Remove corrupt images
        analyser.remove_corrupt_images(corrupt_images)
