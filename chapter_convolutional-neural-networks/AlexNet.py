"""
AlexNet: use dropout layer and Relu activate function
"""
from util import *
from torch.utils.tensorboard import SummaryWriter
from d2l import torch as d2l


batch_size = 256
device = try_gpu(1)
writer = SummaryWriter("log/AlexNet")


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10))

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=224)
    net = AlexNet()
    train_net_multi_gpu(net, train_iter, test_iter, 20, lr=0.05, num_gpus=2, writer=writer)
