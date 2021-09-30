# %%
from numpy import True_
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
        self.lr = nn.Linear(784, 10)

    def forward(self, X):
        output = self.flat(X)
        output = self.lr(output)
        return output


# %%
loss = nn.CrossEntropyLoss()
net = LogisticRegression()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
epochs = 10
# %%
for epoch in range(epochs):
    for X, y in tqdm(train_iter):
        y_hat = net(X)
        y_pred = torch.argmax(y_hat, dim=1).tolist()
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        acc = accuracy_score(y_pred=y_pred, y_true=y.tolist())
