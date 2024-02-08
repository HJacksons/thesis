from torchvision import models
import torch
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")

# list the models each model on a new line
logging.info("\n".join([model for model in dir(models) if model[0].isupper()]))




