import torch
from .activation import relu, softmax
from src.loss import cross_entropy


class Fnet:
    def __init__(self, learning_rate=1e-4, weight_decay=1e-3):
        self.fc1 = (torch.randn((784, 512)) / (784**0.5)).cuda().requires_grad_()
        self.bias1 = (torch.zeros((512))).cuda().requires_grad_()
        self.fc2 = (torch.randn((512, 256)) / 512**0.5).cuda().requires_grad_()
        self.bias2 = (torch.zeros((256))).cuda().requires_grad_()
        self.fc3 = (torch.randn((256, 10)) / (256**0.5)).cuda().requires_grad_()
        self.bias3 = (torch.zeros((10))).cuda().requires_grad_()

        self.wd = weight_decay
        self.lr = learning_rate

    def forward(self, input):

        x = relu(input @ self.fc1 + self.bias1)
        x = relu(x @ self.fc2 + self.bias2)
        x = softmax(x @ self.fc3 + self.bias3)

        return x

    def update_weight(self):

        with torch.no_grad():
            self.fc1 = (
                self.fc1 * (1 - self.wd) - self.lr * self.fc1.grad
            ).requires_grad_()
            self.bias1 = (
                self.bias1 * (1 - self.wd) - self.lr * self.bias1.grad
            ).requires_grad_()
            self.Fc2 = (
                self.fc2 * (1 - self.wd) - self.lr * self.fc2.grad
            ).requires_grad_()
            self.bias2 = (
                self.bias2 * (1 - self.wd) - self.lr * self.bias2.grad
            ).requires_grad_()
            self.fc3 = (
                self.fc3 * (1 - self.wd) - self.lr * self.fc3.grad
            ).requires_grad_()
            self.bias3 = (
                self.bias3 * (1 - self.wd) - self.lr * self.bias3.grad
            ).requires_grad_()
        torch.cuda.empty_cache()

    def train(self, x, y):
        predicted_porb = self.forward(x)
        loss = cross_entropy(predicted_porb, y)
        loss.backward()
        self.update_weight()

        return predicted_porb, loss

    def inference(self, x, y):
        predicted_porb = self.forward(x)
        Loss_function = cross_entropy(predicted_porb, y)
        return predicted_porb, Loss_function

    def state_dict(self):
        return {
            "fc1": self.fc1,
            "bias1": self.bias1,
            "fc1": self.fc2,
            "bias1": self.bias2,
            "fc1": self.fc3,
            "bias1": self.bias3,
            "wd": self.wd,
            "lr": self.lr,
        }

    def save(self, ckpt_path):
        torch.save(self.state_dict(), ckpt_path)

    def load(self, ckpt_path):
        model_weight = torch.load(ckpt_path)
        for k, v in model_weight.items():
            if k in self.__dict__():
                setattr(self, k, v)
