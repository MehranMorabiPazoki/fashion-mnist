import torch


def cls_accuracy(pred_y, true_class):
    return torch.sum(torch.argmax(pred_y, dim=1) == true_class) / len(true_class)
