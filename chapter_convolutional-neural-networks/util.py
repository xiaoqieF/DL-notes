import torch
from torch import nn


def accuracy(y_hat, y):
    y_hat = y_hat.argmax(axis=1)
    return (y_hat.type(y.dtype) == y).sum()


def evaluate_accuracy(net, data_iter):
    net.eval()
    dev = next(iter(net.parameters())).device
    true_nums, total_nums = 0, 0
    with torch.no_grad():
        for x, y in data_iter:
            x, y = x.to(dev), y.to(dev)
            true_nums += accuracy(net(x), y)
            total_nums += y.numel()
    return true_nums / total_nums


def train_net(net, train_iter, test_iter, num_epochs, lr, device, writer=None):
    def init_weight(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
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
        if writer is not None:
            writer.add_scalar("loss", train_loss / train_total_nums, global_step=epoch)
            writer.add_scalars("Acc", tag_scalar_dict={
                'train_acc': train_true_nums / train_total_nums,
                'test_acc': test_acc
            }, global_step=epoch)
        print(f'epoch:{epoch}, loss:{train_loss / train_total_nums}, '
              f'train_acc:{train_true_nums / train_total_nums}, test_acc:{test_acc}')
    print("finish")
