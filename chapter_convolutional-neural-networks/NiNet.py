"""
Network in Network: use AdaptiveAvgPool2d rather than FC
"""

from util import *
from torch.utils.tensorboard import SummaryWriter
from d2l import torch as d2l

batch_size = 128
device = d2l.try_gpu(1)
writer = SummaryWriter("log/NiNet")


class NiNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class NiNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            NiNBlock(1, 96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            NiNBlock(96, 256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            NiNBlock(256, 384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            NiNBlock(384, 10, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=224)
    net = NiNet()
    print(net)
    train_net(net, train_iter, test_iter, 20, lr=0.1, device=device, writer=writer)
