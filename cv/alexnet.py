# %%
import numpy as np
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
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
train_iter, test_iter = load_data_fashion_mnist(256, resize=224)

# %%


class AlexNet(nn.Module):
    """
    AlexNet形式上与LeNet差不多，但是参数量比LeNet要多很多。
    """

    def __init__(self):
        super(AlexNet, self).__init__()
        self.flat = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.conv1 = nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(6400, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

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
loss = nn.CrossEntropyLoss()
net = AlexNet()
net = net.cuda()
summary(net, input_size=((1, 224, 224)))
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
label2cate = {0: 'T-shirt/top',
              1: 'Trouser',
              2: 'Pullover',
              3: 'Dress',
              4: 'Coat',
              5: 'Sandal',
              6: 'Shirt',
              7: 'Sneaker',
              8: 'Bag',
              9: 'Ankle boot'}


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
            axis[i, j].imshow(X[i*5+j].numpy().reshape(224, 224))
            axis[i, j].set_title(titles[i*5+j])
    plt.show()


# %%
predict(net, test_iter)
# %%
