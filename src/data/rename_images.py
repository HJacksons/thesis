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


def rename_images(directory: str, class_name: str):
    """
    Rename the images in the new plant dataset to a format consistent with potato dataset.
    """
    try:
        for count, filename in enumerate(os.listdir(directory), start=1):
            dst = f"{class_name}_{str(count).zfill(4)}.jpg"
            src = os.path.join(directory, filename)
            dst = os.path.join(directory, dst)

            os.rename(src, dst)
            logging.info(f"Renamed {src} to {dst}")
    except Exception as e:
        logging.error(f"Error occurred while renaming files in {directory}: {e}")


load_dotenv()
earlyBlight_path = os.getenv("EARLY_BLIGHT_PATH")
lateBlight_path = os.getenv("LATE_BLIGHT_PATH")
healthy_path = os.getenv("HEALTHY_PATH")

# Confirm env path and rename images
if earlyBlight_path and lateBlight_path and healthy_path:
    rename_images(earlyBlight_path, "EarlyBlight")
    rename_images(lateBlight_path, "LateBlight")
    rename_images(healthy_path, "Healthy")
    logging.info("Images renamed successfully.")
else:
    logging.error("One or more environment variables are not set.")
