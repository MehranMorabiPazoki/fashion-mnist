import torch


def relu(tensor):
    x = torch.clone(tensor)
    x[x < 0] = 0
    return x


def softmax(tensor):
    return torch.exp(tensor) / torch.sum(torch.exp(tensor), dim=1)[:, None]
