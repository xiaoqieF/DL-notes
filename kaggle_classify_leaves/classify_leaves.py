"""
kaggle: https://www.kaggle.com/competitions/classify-leaves/overview
you should download data by yourself, ant put it in '../data/'
"""
from PIL import Image
import os
from torch.utils import data
import torchvision
from util import *
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

writer = SummaryWriter("log/ResNet")

train_csv = pd.read_csv("../data/classify-leaves/train.csv")
test_csv = pd.read_csv("../data/classify-leaves/test.csv")

leaves_labels = sorted(list(set(train_csv['label'])))
n_classes = len(leaves_labels)
class_to_num = dict(zip(leaves_labels, range(n_classes)))
num_to_class = {v: k for k, v in class_to_num.items()}

transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(degrees=90),
        torchvision.transforms.ColorJitter(0.5, 0.5),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class LeavesDataset(data.Dataset):
    def __init__(self, train_csv, train=True, ratio=0.2, transform=None):
        super(LeavesDataset, self).__init__()
        self.trans = transform
        self.train_csv = train_csv
        if train:
            self.data = self.train_csv[:int(self.train_csv.shape[0] * (1 - ratio))]
        else:
            self.data = self.train_csv[int(self.train_csv.shape[0] * (1 - ratio)):]
        print(self.data.shape)

    def __getitem__(self, index):
        label = class_to_num[self.data.iloc[index, 1]]
        image = Image.open(os.path.join('../data/classify-leaves', self.data.iloc[index, 0]))

        if self.trans:
            image = self.trans(image)
        return [image, label]

    def __len__(self):
        return self.data.shape[0]


def predict(net, test_csv):
    print(test_csv)
    net.eval()
    with open("../data/classify-leaves/sub.csv", 'w') as f:
        f.write('image,label\n')
        for path in test_csv.loc[:, 'image']:
            img = Image.open(os.path.join('../data/classify-leaves', path))
            img = transform_test(img).unsqueeze(0).to(try_gpu(0))
            with torch.no_grad():
                y = net(img).cpu()
                print(num_to_class[y.argmax(dim=1).item()])
                f.write(path + ',' + num_to_class[y.argmax(dim=1).item()] + '\n')


if __name__ == '__main__':
    train_iter = data.DataLoader(LeavesDataset(train_csv, train=True, transform=transform_train),
                                 batch_size=16, shuffle=True, num_workers=6, drop_last=True)
    test_iter = data.DataLoader(LeavesDataset(train_csv, train=False, transform=transform_test),
                                batch_size=16, shuffle=False, num_workers=6, drop_last=False)
    net = torchvision.models.resnet18(pretrained=True)
    net.fc = nn.Sequential(
        nn.Linear(net.fc.in_features, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 176))
    print(net)
    train_net_multi_gpu(net, train_iter, test_iter, 100, 0.01, 2, writer=writer)
    torch.save(net.state_dict(), 'model/net.params5')
    # net.load_state_dict(torch.load('model/net.params4'))
    # train_net_multi_gpu(net, train_iter, test_iter, 40, 0.05, 2)
    # net.to(try_gpu(0))
    # predict(net, pd.read_csv('../data/classify-leaves/test.csv'))
