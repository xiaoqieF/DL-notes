import numpy as np
import pandas as pd
import torch
from torch import nn

train_data = pd.read_csv("./data/kaggle_house_pred_train.csv")
test_data = pd.read_csv("./data/kaggle_house_pred_test.csv")

if __name__ == '__main__':
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

    # 处理离散值
    all_features = pd.get_dummies(all_features, dummy_na=True)
    print(all_features.shape)
