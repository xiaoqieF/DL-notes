"""
ResNet: use lots of residuals, make network deeper
"""
from util import *
from torch.nn import functional as F
from d2l import torch as d2l
from torch.utils.tensorboard import SummaryWriter

batch_size = 256
device = try_gpu(1)
writer = SummaryWriter("log/ResNet")


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return F.relu(y)


class ResBlock(nn.Module):
    def __init__(self, in_channels, num_channels, num_residuals, first_block=False):
        super().__init__()
        self.blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.blk.append(Residual(in_channels, num_channels, use_1x1conv=True, strides=2))
            else:
                self.blk.append(Residual(num_channels, num_channels))

    def forward(self, x):
        return self.blk(x)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm2d(64), nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2 = ResBlock(64, 64, 2, first_block=True)
        b3 = ResBlock(64, 128, 2)
        b4 = ResBlock(128, 256, 2)
        b5 = ResBlock(256, 512, 2)

        self.net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, 10))

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=224)
    net = ResNet()
    # train_net(net, train_iter, test_iter, 20, lr=0.05, device=device, writer=writer)
    train_net_multi_gpu(net, train_iter, test_iter, 20, lr=0.1, num_gpus=2, writer=writer)
