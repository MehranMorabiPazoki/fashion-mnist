""" Dataset Driver for Well known Datasets 
"""

import torch
import torchvision
from torch.utils.data import random_split
import os
from .visualiztion import plot_category


class MyStruct:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def mnist(
    root: str,
    transform=None,
    target_transform=None,
    download: bool = False,
    batch_size: int = 16,
    num_workers: int = 1,
    trainset_ratio: float = 0.8,
):
    labels = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }

    trainset = torchvision.datasets.FashionMNIST(
        root=os.path.join(root, "raw/FashionMNIST_train"),
        train=True,
        download=download,
    )
    plot_category(trainset, labels, path=os.path.join(root, "visualization"))
    testset = torchvision.datasets.FashionMNIST(
        root=os.path.join(root, "raw/FashionMNIST_test"),
        train=False,
        download=download,
    )
    trainset.transform = transform
    testset.transform = transform

    train_dataset, validaton_dataset = random_split(
        trainset,
        [
            trainset_ratio,
            (1 - trainset_ratio),
        ],
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers
    )

    validation_loader = torch.utils.data.DataLoader(
        validaton_dataset, batch_size=batch_size, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, num_workers=num_workers
    )

    return MyStruct(
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
    )
