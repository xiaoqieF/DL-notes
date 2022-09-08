"""
d2l CNN
LeNet
"""

import torch
from torch import nn
from d2l import torch as d2l
from torch.utils.tensorboard import SummaryWriter

batch_size = 256
device = d2l.try_gpu(0)
# tensorboard writer
writer = SummaryWriter("log/LeNet")


def _accuracy(y_hat, y):
    y_hat = y_hat.argmax(axis=1)
    return (y_hat.type(y.dtype) == y).sum()


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

    def evaluate_accuracy(self, data_iter):
        self.eval()
        dev = next(iter(self.parameters())).device
        true_nums, total_nums = 0, 0
        with torch.no_grad():
            for x, y in data_iter:
                x, y = x.to(dev), y.to(dev)
                true_nums += _accuracy(self(x), y)
                total_nums += y.numel()
        return true_nums / total_nums

    def train_net(self, train_iter, test_iter, num_epochs, lr, device):
        def init_weight(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)

        self.apply(init_weight)
        print('training on', device)
        self.to(device)
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            self.train()
            train_true_nums, train_total_nums, train_loss = 0, 0, 0
            for x, y in train_iter:
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                y_hat = self(x)
                ls = loss(y_hat, y)
                ls.backward()
                optimizer.step()
                train_true_nums += _accuracy(y_hat, y)
                train_total_nums += y.numel()
                train_loss += ls * y.numel()
            test_acc = self.evaluate_accuracy(test_iter)
            writer.add_scalar("loss", train_loss / train_total_nums, global_step=epoch)
            writer.add_scalars("Acc", tag_scalar_dict={
                'train_acc': train_true_nums / train_total_nums,
                'test_acc': test_acc
            }, global_step=epoch)


if __name__ == '__main__':
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    net = LeNet()
    net.train_net(train_iter, test_iter, 20, lr=0.2, device=device)
