# %%
import numpy as np
from sklearn.utils.sparsefuncs import incr_mean_variance_axis
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
from torchvision import utils
import matplotlib.pyplot as plt
import torchvision
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import os
# %%


def load_data_stl10(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.STL10(
        root='github/data', split='train', transform=trans, download=True
    )
    mnist_test = torchvision.datasets.STL10(
        root='github/data', split='test', transform=trans, download=True

    )
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                                 num_workers=4)
    test_iter = data.DataLoader(
        mnist_test, batch_size, shuffle=False, num_workers=4)
    return train_iter, test_iter


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root='github/data', train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root='github/data', train=False, transform=trans, download=True

    )
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                                 num_workers=4)
    test_iter = data.DataLoader(
        mnist_test, batch_size, shuffle=False, num_workers=4)
    return train_iter, test_iter


# %%
data_dir = '/home/samael/github/image_generation/'
worker = 8
batch_size = 16
image_size = 224
num_epochs = 25
lr = 1e-2

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #         transforms.ColorJitter(brightness=0.1, contrast=0.05, saturation=0.05, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(
    data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(
    image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=worker) for x in ['train', 'val']}
print('Initialized!')

# %%


def norm_image(image):
    min_val = torch.min(image)
    max_val = torch.max(image)
    return (image - min_val) / (max_val - min_val)


# %%
real_batch = next(iter(dataloaders_dict['train']))
val_batch = next(iter(dataloaders_dict['val']))
plt.figure(figsize=(20, 20))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(utils.make_grid(
    real_batch[0][:9], padding=2, normalize=True, nrow=3), (1, 2, 0)))
plt.show()
# %%


class AlexNet(nn.Module):
    """
    AlexNet形式上与LeNet差不多，但是参数量比LeNet要多很多。
    """

    def __init__(self, input_channel, num_label):
        super(AlexNet, self).__init__()
        self.flat = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.conv1 = nn.Conv2d(
            input_channel, 96, kernel_size=11, stride=4, padding=1)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(6400, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_label)

    def forward(self, X):
        output = self.conv1(X)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.pool(output)
        output = self.flat(output)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc3(output)
        return output


# %%
train_iter = dataloaders_dict['train']
test_iter = dataloaders_dict['val']
loss = nn.CrossEntropyLoss()
net = AlexNet(3, 9)
net = net.cuda()
summary(net, input_size=((3, 224, 224)))
# optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
# epochs = 20
# acc_train_list, acc_test_list = [], []

# %%
label2cate = {0: 'black_tights',
              1: 'boots',
              2: 'fishnet',
              3: 'flesh_colored',
              4: 'other_colors',
              5: 'others',
              6: 'pants',
              7: 'pattern',
              8: 'white_tights'}


def predict(net, test_iter, n=25):
    for X, y in test_iter:
        break
    net = net.cpu()
    y_pred = net(X).argmax(dim=1)
    trues = [label2cate[x] for x in y.detach().tolist()]
    preds = [label2cate[x] for x in y_pred.detach().tolist()]
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    nrows, ncols = 5, 5
    figure, axis = plt.subplots(
        nrows, ncols, figsize=(16, 16), gridspec_kw={'wspace': 0.2, 'hspace': 0.5})
    for i in range(nrows):
        for j in range(ncols):
            axis[i, j].imshow(norm_image(
                X[i*5+j]).numpy().transpose((1, 2, 0)))
            axis[i, j].set_title(titles[i*5+j])
    plt.show()

# %%
# VGG 由一系列卷基层组成，结构上有多个相似的块（block)堆叠而成。一个块里面包含基本的convolution layer
# activation function，还有pooling layer。这里定义一个通用的VGG block，后面可以反复调用。


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(vgg_arch, in_channels, num_labels):
    conv_blks = []
    for (num_convs, out_channels) in vgg_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    net_arch = []
    net_arch.extend(conv_blks)
    net_arch.append(nn.Flatten())
    net_arch.extend([nn.Linear(out_channels * 7 * 7, 4096),
                    nn.ReLU(), nn.Dropout(0.5)])
    net_arch.extend([nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5)])
    net_arch.append(nn.Linear(4096, num_labels))
    return nn.Sequential(*net_arch)


# %%
# 一个VGG-11有5个block，其中，前两个有一个卷积层，后三个有2个卷积层，共8个卷积层，加上3个全连接层，共11层。
vgg_arch = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]
net = vgg(vgg_arch, 3, 9)
net = net.cuda()
summary(net, input_size=((3, 224, 224)))
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
epochs = 20
acc_train_list, acc_test_list = [], []
# %%
for epoch in range(epochs):
    total_acc = 0
    steps = 0
    net.train()
    y_pred_lst, y_true_lst = [], []
    for X, y in train_iter:
        X = X.cuda()
        y = y.cuda()
        y_hat = net(X)
        y_pred = torch.argmax(y_hat, dim=1).cpu().tolist()
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        acc = accuracy_score(y_pred=y_pred, y_true=y.cpu().tolist())
        total_acc += acc
        steps += 1
    acc_train_list.append(total_acc/steps)
    total_acc, steps = 0, 0
    net.eval()
    for X, y in test_iter:
        X = X.cuda()
        y = y.cuda()
        y_hat = net(X)
        y_pred = torch.argmax(y_hat, dim=1).cpu().tolist()
        acc = accuracy_score(y_pred=y_pred, y_true=y.cpu().tolist())
        y_pred_lst.extend(y_pred)
        y_true_lst.extend(y.tolist())
        total_acc += acc
        steps += 1
    acc_test_list.append(total_acc/steps)
    print('epoch %d, train acc %.4f, test acc %.4f' %
          (epoch+1, acc_train_list[-1], acc_test_list[-1]))
print(confusion_matrix(y_pred=y_pred_lst, y_true=y_true_lst))


# %%
plt.figure(figsize=(16, 9))
plt.plot(np.arange(1, epochs+1), acc_train_list)
plt.plot(np.arange(1, epochs+1), acc_test_list)
plt.legend(['train', 'test'])
plt.title('Accuracy')
plt.show()
# %%
predict(net, test_iter)

# %%
# LeNet和VGG都是加深扩大卷积层和全连接层，通过卷积层来获取空间结构特征，通过全连接层对表征的特征进行处理。
# Network in network在每个像素位置应用一个全连接层，就可以看作是一个1*1的卷积层。
# 其中NiN的每一个block都是由一个自定义的卷积层及两个1*1的卷积层组成。


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels,
                  kernel_size, strides, padding))
    layers.append(nn.ReLU())
    layers.extend(
        [nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()])
    layers.extend(
        [nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()])
    return nn.Sequential(*layers)


def nin(in_channels, num_labels):
    layers = []
    layers.append(nin_block(in_channels, out_channels=96,
                  kernel_size=11, strides=4, padding=0))
    layers.append(nn.MaxPool2d(3, stride=2))
    layers.append(nin_block(96, 256, 5, 1, 2))
    layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
    layers.append(nin_block(256, 384, 3, 1, 1))
    layers.append(nn.MaxPool2d(3, 2))
    layers.append(nn.Dropout(0.5))
    layers.append(nin_block(384, num_labels, 3, 1, 1))
    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    layers.append(nn.Flatten())
    return nn.Sequential(*layers)


# %%
net = nin(3, 9)
net = net.cuda()
summary(net, input_size=((3, 224, 224)))
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
epochs = 20
acc_train_list, acc_test_list = [], []
# %%
for epoch in range(epochs):
    total_acc = 0
    steps = 0
    net.train()
    y_pred_lst, y_true_lst = [], []
    for X, y in train_iter:
        X = X.cuda()
        y = y.cuda()
        y_hat = net(X)
        y_pred = torch.argmax(y_hat, dim=1).cpu().tolist()
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        acc = accuracy_score(y_pred=y_pred, y_true=y.cpu().tolist())
        total_acc += acc
        steps += 1
    acc_train_list.append(total_acc/steps)
    total_acc, steps = 0, 0
    net.eval()
    for X, y in test_iter:
        X = X.cuda()
        y = y.cuda()
        y_hat = net(X)
        y_pred = torch.argmax(y_hat, dim=1).cpu().tolist()
        acc = accuracy_score(y_pred=y_pred, y_true=y.cpu().tolist())
        y_pred_lst.extend(y_pred)
        y_true_lst.extend(y.tolist())
        total_acc += acc
        steps += 1
    acc_test_list.append(total_acc/steps)
    print('epoch %d, train acc %.4f, test acc %.4f' %
          (epoch+1, acc_train_list[-1], acc_test_list[-1]))
print(confusion_matrix(y_pred=y_pred_lst, y_true=y_true_lst))


# %%
plt.figure(figsize=(16, 9))
plt.plot(np.arange(1, epochs+1), acc_train_list)
plt.plot(np.arange(1, epochs+1), acc_test_list)
plt.legend(['train', 'test'])
plt.title('Accuracy')
plt.show()
# %%
predict(net, test_iter)
# %%
# GoogLeNet吸收了NiN的思想，对LeNet作出了改进，通过不同的卷积核组合，达到优化。


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat([p1, p2, p3, p4], dim=1)


def GoogLeNet(in_channels, num_labels):
    b1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                       nn.ReLU(),
                       nn.Conv2d(64, 192, kernel_size=3, padding=1),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                       Inception(256, 128, (128, 192), (32, 96), 64),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                       Inception(512, 160, (112, 224), (24, 64), 64),
                       Inception(512, 128, (128, 256), (24, 64), 64),
                       Inception(512, 112, (144, 288), (32, 64), 64),
                       Inception(528, 256, (160, 320), (32, 128), 128),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                       Inception(832, 384, (192, 384), (48, 128), 128),
                       nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                       nn.Flatten())
    return nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, num_labels))


# %%
net = GoogLeNet(3, 9)
net = net.cuda()
summary(net, input_size=((3, 224, 224)))
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
epochs = 20
acc_train_list, acc_test_list = [], []
# %%
for epoch in range(epochs):
    total_acc = 0
    steps = 0
    net.train()
    y_pred_lst, y_true_lst = [], []
    for X, y in train_iter:
        X = X.cuda()
        y = y.cuda()
        y_hat = net(X)
        y_pred = torch.argmax(y_hat, dim=1).cpu().tolist()
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        acc = accuracy_score(y_pred=y_pred, y_true=y.cpu().tolist())
        total_acc += acc
        steps += 1
    acc_train_list.append(total_acc/steps)
    total_acc, steps = 0, 0
    net.eval()
    for X, y in test_iter:
        X = X.cuda()
        y = y.cuda()
        y_hat = net(X)
        y_pred = torch.argmax(y_hat, dim=1).cpu().tolist()
        acc = accuracy_score(y_pred=y_pred, y_true=y.cpu().tolist())
        y_pred_lst.extend(y_pred)
        y_true_lst.extend(y.tolist())
        total_acc += acc
        steps += 1
    acc_test_list.append(total_acc/steps)
    print('epoch %d, train acc %.4f, test acc %.4f' %
          (epoch+1, acc_train_list[-1], acc_test_list[-1]))
print(confusion_matrix(y_pred=y_pred_lst, y_true=y_true_lst))


# %%
plt.figure(figsize=(16, 9))
plt.plot(np.arange(1, epochs+1), acc_train_list)
plt.plot(np.arange(1, epochs+1), acc_test_list)
plt.legend(['train', 'test'])
plt.title('Accuracy')
plt.show()
# %%
predict(net, test_iter)
