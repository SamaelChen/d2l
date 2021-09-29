# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, autograd
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# %%


class Timer:  # @save
    """记录多次运行时间。"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器。"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间。"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和。"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间。"""
        return np.array(self.times).cumsum().tolist()
# %%


def synthetic_data(w, b, num_examples):
    """generate sample data"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
test_feats, test_labels = synthetic_data(true_w, true_b, 1000)

# %%
plt.figure(figsize=(16, 9))
plt.scatter(features[:, 0].detach().numpy(), labels.detach().numpy())
plt.show()
# %%
batch_size = 10
train_data = TensorDataset(features, labels)
test_data = TensorDataset(test_feats, test_labels)
train_iter = DataLoader(train_data, batch_size=batch_size)
test_iter = DataLoader(test_data, batch_size=batch_size)
# %%


class LinearRegression(nn.Module):
    def __init__(self, num_feats):
        super(LinearRegression, self).__init__()
        self.lr = nn.Linear(num_feats, 1)

    def forward(self, X):
        output = self.lr(X)
        return output


# %%
epochs = 100
lr = 0.01
loss = nn.MSELoss()
net = LinearRegression(2)
optimizer = optim.SGD(net.parameters(), lr=lr)
for epoch in range(epochs):
    for X, y in train_iter:
        l = loss(net(X), y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch+1}, loss {l:f}')

# %%
w = net.lr.weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net.lr.bias.data
print('b的估计误差：', true_b - b)
# %%
