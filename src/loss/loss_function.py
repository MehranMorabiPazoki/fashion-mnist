import torch


def cross_entropy(pred_y, true_class):
    data_index = torch.tensor(list(range(len(true_class))))
    return torch.mean(-1 * torch.log(pred_y[data_index, true_class]))
