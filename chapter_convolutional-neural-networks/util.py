"""
train and evaluate net
"""

import torch
from torch import nn


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def accuracy(y_hat, y):
    """
    return sum of predict-true labels
    """
    y_hat = y_hat.argmax(axis=1)
    return (y_hat.type(y.dtype) == y).sum()


def evaluate_accuracy(net, data_iter):
    """
    evaluate accuracy of net in data_iter
    device depends on net
    """
    net.eval()
    dev = next(iter(net.parameters())).device
    true_nums, total_nums = 0, 0
    with torch.no_grad():
        for x, y in data_iter:
            x, y = x.to(dev), y.to(dev)
            true_nums += accuracy(net(x), y)
            total_nums += y.numel()
    return true_nums / total_nums


def init_weight(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def print_res(epoch, train_acc, test_acc, avg_loss, writer=None):
    if writer is not None:
        writer.add_scalar("loss", avg_loss, global_step=epoch)
        writer.add_scalars("Acc", tag_scalar_dict={
            'train_acc': train_acc,
            'test_acc': test_acc
        }, global_step=epoch)
    print(f'epoch:{epoch}, loss:{avg_loss}, '
          f'train_acc:{train_acc}, test_acc:{test_acc}')


def train_net(net, train_iter, test_iter, num_epochs, lr, device, writer=None):
    """
    train and evaluate net on device
    """
    net.apply(init_weight)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        net.train()
        train_true_nums, train_total_nums, train_loss = 0, 0, 0
        for x, y in train_iter:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_hat = net(x)
            ls = loss(y_hat, y)
            ls.backward()
            optimizer.step()
            train_true_nums += accuracy(y_hat, y)
            train_total_nums += y.numel()
            train_loss += ls * y.numel()
        test_acc = evaluate_accuracy(net, test_iter)
        print_res(epoch, train_true_nums/train_total_nums, test_acc, train_loss / train_total_nums, writer)
    print("finish")


def train_net_multi_gpu(net, train_iter, test_iter, num_epochs, lr, num_gpus, writer=None):
    net.apply(init_weight)
    devices = [try_gpu(i) for i in range(num_gpus)]
    net = nn.DataParallel(net, device_ids=devices)
    net.to(devices[0])
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        net.train()
        train_true_nums, train_total_nums, train_loss = 0, 0, 0
        for x, y in train_iter:
            optimizer.zero_grad()
            x, y = x.to(devices[0]), y.to(devices[0])
            y_hat = net(x)
            ls = loss(y_hat, y)
            ls.backward()
            optimizer.step()
            train_true_nums += accuracy(y_hat, y)
            train_total_nums += y.numel()
            train_loss += ls * y.numel()
        test_acc = evaluate_accuracy(net, test_iter)
        print_res(epoch, train_true_nums/train_total_nums, test_acc, train_loss / train_total_nums, writer)
    print("finish")
