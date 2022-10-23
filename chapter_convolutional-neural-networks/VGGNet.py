"""
VGGNet: use many "blocks" build network
"""
from d2l import torch as d2l
from util import *
from torch.utils.tensorboard import SummaryWriter

batch_size = 128
device = try_gpu(0)
writer = SummaryWriter("log/VGGNet")


class VGGBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super().__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class VGGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Sequential()
        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
        in_channels = 1
        for num_convs, out_channels in conv_arch:
            self.net1.append(VGGBlock(num_convs, in_channels, out_channels))
            in_channels = out_channels
        self.net2 = nn.Sequential(nn.Flatten(), nn.Linear(in_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
                                  nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 10))

    def forward(self, x):
        return self.net2(self.net1(x))


if __name__ == '__main__':
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=224)
    net = VGGNet()
    print(net)
    # train_net(net, train_iter, test_iter, 20, lr=0.01, device=device, writer=writer)
