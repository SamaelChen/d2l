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
train_iter, test_iter = load_data_fashion_mnist(256, resize=28)

# %%


class LeNet(nn.Module):
    """dropout是常用避免过拟合的手段，
    有dropout的train和test的accuracy贴得更近，但并不意味着训练的数字更好看。
    dropout仅在训练时候生效。
    """

    def __init__(self):
        super(LeNet, self).__init__()
        self.flat = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, X):
        output = self.conv1(X)
        output = self.sigmoid(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.sigmoid(output)
        output = self.pool(output)
        output = self.flat(output)
        output = self.fc1(output)
        output = self.sigmoid(output)
        output = self.fc2(output)
        output = self.sigmoid(output)
        output = self.fc3(output)
        return output


# %%
loss = nn.CrossEntropyLoss()
net = LeNet()
net = net.cuda()
summary(net, input_size=((1, 28, 28)))
optimizer = torch.optim.SGD(net.parameters(), lr=0.9)
epochs = 100
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
            axis[i, j].imshow(X[i*5+j].numpy().reshape(28, 28))
            axis[i, j].set_title(titles[i*5+j])
    plt.show()


# %%
predict(net, test_iter)
# %%
