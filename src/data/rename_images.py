# This script renames the images in the dataset to a format consistent with the potato dataset.
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    filename="rename_images.log",
)
load_dotenv()
earlyBlight_path = os.getenv("EARLY_BLIGHT_PATH")
lateBlight_path = os.getenv("LATE_BLIGHT_PATH")
healthy_path = os.getenv("HEALTHY_PATH")


class DatasetRenamer:
    def __init__(self, directory, class_name):
        self.directory = directory
        self.class_name = class_name

    def rename_images(self):
        """
        This function renames the images in the new plant dataset to a format consistent with potato dataset.
        """
        try:
            for count, filename in enumerate(os.listdir(self.directory), start=1):
                dst = f"{self.class_name}_{str(count).zfill(4)}.jpg"
                src = os.path.join(self.directory, filename)
                dst = os.path.join(self.directory, dst)

                os.rename(src, dst)
                logging.info(f"Renamed {src} to {dst}")
        except Exception as e:
            logging.error(
                f"Error occurred while renaming files in {self.directory}: {e}"
            )


# Confirm env path and rename images
if earlyBlight_path and lateBlight_path and healthy_path:
    renamer1 = DatasetRenamer(earlyBlight_path, "EarlyBlight")
    renamer1.rename_images()
    renamer2 = DatasetRenamer(lateBlight_path, "LateBlight")
    renamer2.rename_images()
    renamer3 = DatasetRenamer(healthy_path, "Healthy")
    renamer3.rename_images()
else:
    logging.error("One or more environment variables are not set.")
