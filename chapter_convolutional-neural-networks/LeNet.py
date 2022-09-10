"""
d2l CNN
LeNet
"""
from d2l import torch as d2l
from torch.utils.tensorboard import SummaryWriter
from util import *

batch_size = 256
device = try_gpu(0)
# tensorboard writer
writer = SummaryWriter("log/LeNet")


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    net = LeNet()
    train_net(net, train_iter, test_iter, 50, lr=0.1, device=device, writer=writer)
