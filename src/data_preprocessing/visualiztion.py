import matplotlib.pyplot as plt
import random
import os


def plot_category(dataset, labels, path):
    fig = plt.figure(figsize=(10, 10))
    cls = []
    counter = 1
    for data in dataset:
        if data[1] not in cls:
            cls.append(data[1])
            plt.subplot(2, 5, counter)
            plt.imshow(data[0])
            plt.title(labels[data[1]] + "(" + str(data[1]) + ")")
            counter += 1
        if len(cls) == 10:
            break
    if not os.path.exists(path):
        os.mkdir(path)
    fig.savefig(os.path.join(path, "cateogry.png"))


def plot_category_test(dataset, pred_label, path):

    rand_int = random.sample(range(0, len(dataset)), 10)
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
    fig = plt.figure(figsize=(10, 10))
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4
    )
    cls = []
    for counter, idx in enumerate(rand_int):
        plt.subplot(5, 2, counter + 1)
        plt.imshow(dataset[idx][0].reshape(28, 28))
        plt.title(
            label=f"(true_label = {labels[dataset[idx][1]]}), (pred_label = {labels[pred_label[idx].item()]})"
        )
    fig.savefig(os.path.join(path, "test_category.png"))
