import os
import yaml
from src import data_preprocessing
from torchvision import transforms
from src import models
from src import utils

BASE_PATH = os.path.dirname(__file__)


def main():
    log_file, session_dir = utils.get_logfile(BASE_PATH)
    logger = utils.get_root_logger(log_file=log_file)

    with open("config/config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    logger.info("Config File has been read!")
    transform = transforms.Compose([transforms.ToTensor()])
    loaders = data_preprocessing.mnist(
        root="data/", transform=transform, **config["dataset_param"]
    )
    logger.info("Data Loaders are ready!")
    model = models.Fnet(**config["learning_param"])
    logger.info("Training Model is ready!")
    utils.train_model(
        loaders=loaders,
        model=model,
        session_dir=session_dir,
        **config["training_param"]
    )


if __name__ == "__main__":
    main()
