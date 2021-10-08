# %%
import numpy as np
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
# ResNet的核心思想是，每个附加层都应该更容易地包含原始函数作为其元素之一。


class BasicResidual(nn.Module):
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
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        output = self.relu(self.bn1(self.conv1(X)))
        output = self.bn2(self.conv2(output))
        if self.conv3:
            X = self.conv3(X)
        output += X
        return self.relu(output)


class BottleNeck(nn.Module):
    def __init__(self, input_channels, num_channels, downsample=False, strides=1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, stride=strides, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv3 = nn.Conv2d(num_channels, num_channels*4,
                               kernel_size=1, stride=strides)
        self.bn3 = nn.BatchNorm2d(num_channels*4)
        self.relu = nn.ReLU(inplace=True)
        if downsample:
            self.conv4 = nn.Conv2d(input_channels, num_channels*4,
                                   kernel_size=1, stride=strides)
            self.bn4 = nn.BatchNorm2d(num_channels*4)
        else:
            self.conv4 = None

    def forward(self, X):
        output = self.relu(self.bn1(self.conv1(X)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        output += X
        return self.relu(output)


def resnet_basic_block(input_channels, num_channels,
                       num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(BasicResidual(input_channels, num_channels,
                                     use_1x1conv=True, strides=2))
        else:
            blk.append(BasicResidual(num_channels, num_channels))
    return blk


def resnet_bottleneck_block(input_channels, num_channels,
                            num_residuals):
    blk = []
    for i in range(num_residuals):
        if i == 0:
            blk.append(BottleNeck(input_channels, num_channels,
                                  downsample=True, strides=1))
        else:
            blk.append(BottleNeck(num_channels*4, num_channels))
    return blk


def resnet(input_channels, num_labels, num_residual_blks, fifty_layers=False):
    """
    ResNet前两层跟GoogLeNet一样，差别在ResNet每个卷积层后增加了BN层。
    原论文里面，50层以下的采用的是BasicResidual的块结构，50层及以上用的是bottleneck的结构
    """
    b1 = nn.Sequential(nn.Conv2d(input_channels, 64,
                       kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    blks = []
    if not fifty_layers:
        for i, num in enumerate(num_residual_blks):
            if i == 0:
                blks.append(nn.Sequential(*resnet_basic_block(64, 64,
                                                              num,
                                                              first_block=True)))
            else:
                blks.append(nn.Sequential(
                    *resnet_basic_block(64*2**(i-1), 64*2**i, num)))
    else:
        for i, num in enumerate(num_residual_blks):
            if i == 0:
                blks.append(nn.Sequential(*resnet_bottleneck_block(64, 64,
                                                                   num)))
            else:
                blks.append(nn.Sequential(
                    *resnet_bottleneck_block(64*2**(i+1), 64*2**i, num)))
    res_blks = nn.Sequential(*blks)
    if not fifty_layers:
        net = nn.Sequential(b1, res_blks,
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten(),
                            nn.Linear(512, num_labels))
    else:
        net = nn.Sequential(b1, res_blks,
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten(),
                            nn.Linear(2048, num_labels))
    return net


# %%
net = resnet(3, 9, [3, 4, 6, 3], fifty_layers=True)
# net = net.cuda()
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
