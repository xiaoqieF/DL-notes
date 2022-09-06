import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
from torch.utils.tensorboard import SummaryWriter

import numpy as np

train_data = pd.read_csv("./data/kaggle_house_pred_train.csv")
test_data = pd.read_csv("./data/kaggle_house_pred_test.csv")


def process_data(train_data, test_data):
    # 将训练集和测试集拼接(去除ID和最终预测结果列)
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    print(all_features.dtypes)
    # 找到类型为数字的列索引名称
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    # 标准化数据，并将 NA 数据赋 0
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std())
    )
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    # 处理离散值, one-hot 编码
    all_features = pd.get_dummies(all_features, dummy_na=True)
    print(all_features.shape)

    n_train = train_data.shape[0]
    train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
    train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
    return train_features, test_features, train_labels


train_features, test_features, train_labels = process_data(train_data, test_data)
loss = nn.MSELoss()
in_features = train_features.shape[1]
writer = SummaryWriter("log")


def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net


def log_rmse(net, features, labels):
    net.eval()
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels, num_epochs, lr, weight_decay, batch_size):
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        for x, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(x), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        writer.add_scalar(tag="loss/train", scalar_value=train_ls[-1], global_step=epoch)
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
            writer.add_scalar(tag="loss/test", scalar_value=test_ls[-1], global_step=epoch)
    return train_ls, test_ls


if __name__ == '__main__':
    train_sz = train_features.shape[0] // 5 * 4
    train_ls, test_ls = train(get_net(), train_features[0:train_sz], train_labels[0:train_sz],
                              train_features[train_sz:], train_labels[train_sz:], 1000, 0.005, 0.2, 32)
    writer.close()
