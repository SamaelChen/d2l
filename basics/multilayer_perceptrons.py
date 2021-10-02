# %%
import numpy as np
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
        root='../data', train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root='../data', train=False, transform=trans, download=True

    )
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                                 num_workers=4)
    test_iter = data.DataLoader(
        mnist_test, batch_size, shuffle=False, num_workers=4)
    return train_iter, test_iter


# %%
train_iter, test_iter = load_data_fashion_mnist(32, resize=28)

# %%


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, X):
        output = self.flat(X)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        return output


# %%
loss = nn.CrossEntropyLoss()
net = LogisticRegression()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
epochs = 10
acc_train_list, acc_test_list = [], []
# %%
for epoch in range(epochs):
    total_acc = 0
    steps = 0
    net.train()
    for X, y in train_iter:
        y_hat = net(X)
        y_pred = torch.argmax(y_hat, dim=1).tolist()
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        acc = accuracy_score(y_pred=y_pred, y_true=y.tolist())
        total_acc += acc
        steps += 1
    acc_train_list.append(total_acc/steps)
    total_acc, steps = 0, 0
    net.eval()
    for X, y in test_iter:
        y_hat = net(X)
        y_pred = torch.argmax(y_hat, dim=1).tolist()
        acc = accuracy_score(y_pred=y_pred, y_true=y.tolist())
        total_acc += acc
        steps += 1
    acc_test_list.append(total_acc/steps)
    print('epoch %d, train acc %.4f, test acc %.4f' %
          (epoch+1, acc_train_list[-1], acc_test_list[-1]))


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
