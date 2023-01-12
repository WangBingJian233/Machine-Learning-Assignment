import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode, resize


class Net(nn.Module):
    """ 模型架构 """

    def __init__(
        self,
    ):
        super().__init__()
        self.net = models.resnet152()
        self.dense = nn.Linear(1000, 4)

    def forward(self, X):
        return self.dense(self.net(X))


class MyDataset(Dataset):
    classes = [
        'Field Cricket',
        'Jute Stem Weevil',
        'Spilosoma Obliqua',
        'Yellow Mite'
    ]

    label = {
        c: index
        for index, c in enumerate(classes)
    }

    def __init__(
        self,
        root=r'./dataset',
        is_train=True
    ):
        self.path = os.path.join(root, 'train' if is_train else 'validation')
        self.images = []
        self.merge_images()
        self.t = T.Compose(transforms=[
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float32)
        ])

    def merge_images(self):
        for c in self.classes:
            p = os.path.join(self.path, c)
            self.images.extend(
                [(i, c) for i in glob.glob(os.path.join(p, '*.jpg'))]
            )

    def __getitem__(self, index):
        image_path, image_label = self.images[index]
        image_label = self.label[image_label]
        image = Image.open(image_path)
        image = self.t(resize(image, [224, 224], InterpolationMode.BILINEAR))
        return image, torch.LongTensor([image_label])

    def __len__(self):
        return len(self.images)


def dataload(
    is_train=True,
    batch_size=64
):
    dataset = MyDataset(is_train=is_train)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train
    )


def train_batch(
    batch,
    optimizer,
    loss,
    net
):
    optimizer.zero_grad()
    if torch.cuda.is_available():
        batch = [i.cuda() for i in batch]
    X, Y = batch
    Y_hat = net(X)
    Y_hat = F.softmax(Y_hat, -1)
    l = loss(Y_hat, Y.flatten())
    l.sum().backward()
    optimizer.step()
    with torch.no_grad():
        return l.sum().cpu()


def test_batch(
    batch,
    net,
    loss
):
    if torch.cuda.is_available():
        batch = [i.cuda() for i in batch]
    X, Y = batch
    Y_hat = net(X)
    Y_hat = F.softmax(Y_hat, -1)
    l = loss(Y_hat, Y.flatten())
    return l.sum().cpu()


def save_checkpoint(net, path):
    torch.save(net, path)


def train(
    epoch=15,
    lr=1e-5,
    train_batch_size=32,
    test_batch_size=32,
    path='params.cpt'
):
    net = Net()
    if torch.cuda.is_available():
        net.cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=lr)
    dataset = dataload(is_train=True, batch_size=train_batch_size)
    vaild_dataset = dataload(is_train=False, batch_size=test_batch_size)

    min_mean_loss = None

    for i in range(epoch):
        print('epoch: %d/%d' % (i+1, epoch))
        mean_loss = []
        mean_vaild_loss = []
        net.train()
        for batch in dataset:
            l = train_batch(
                batch=batch,
                optimizer=optimizer,
                loss=loss,
                net=net
            )
            mean_loss.append(l)

        # with torch.no_grad():
        #     net.eval()
        #     for batch in vaild_dataset:
        #         l = test_batch(batch=batch, net=net, loss=loss)
        #         mean_vaild_loss.append(l)

        print('train loss: %.5f' % np.mean(mean_loss))
        # print('test loss: %.5f' % np.mean(mean_vaild_loss))

        if min_mean_loss is None or min_mean_loss > np.mean(mean_loss):
            save_checkpoint(net, path)
            min_mean_loss = np.mean(mean_loss)

def load_net_from_hdf(path):
    net = torch.load(path, map_location=torch.device('cpu'))
    net.eval()
    return net


def predict(X, Y, net=None):
    return net(X).argmax(-1).flatten(), Y.flatten()

if __name__ == '__main__':
    # 训练
    train(
        epoch=100,
        train_batch_size=64,
        test_batch_size=64
    )

    # 预测代码
    # classes = [
    #     'Field Cricket',
    #     'Jute Stem Weevil',
    #     'Spilosoma Obliqua',
    #     'Yellow Mite'
    # ]
    # path = 'params.cpt'
    # net = load_net_from_hdf(path)
    # predict_result = []
    # for X, Y in dataload(False, batch_size=16):
    #     Y_hat, Y = predict(X, Y, net)
    #     for i in range(Y.shape[0]):
    #         print(classes[Y_hat[i]], '\t',classes[Y[i]])
