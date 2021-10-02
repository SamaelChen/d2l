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


def synthetic_data(max_degree=20, num_examples=1000):
    """generate sample data"""
    max_degree = 5 if max_degree < 5 else max_degree
    true_w = np.zeros(max_degree)
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])
    features = np.random.normal(size=(num_examples, 1))
    np.random.shuffle(features)
    poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
    for i in range(max_degree):
        poly_features[:, i] /= math.gamma(i+1)
    labels = np.dot(poly_features, true_w)
    labels += np.random.normal(scale=0.1, size=labels.shape)
    true_w, features, poly_features, labels = [torch.tensor(
        x, dtype=torch.float32) for x in [true_w, features, poly_features, labels]]
    return true_w, poly_features, labels


true_w, poly_features, labels = synthetic_data(max_degree=20)
_, test_poly_feats, test_labels = synthetic_data(max_degree=20)

# %%


class LinearRegression(nn.Module):
    def __init__(self, num_feats, bias=False):
        super(LinearRegression, self).__init__()
        self.bias = bias
        if self.bias:
            self.lr = nn.Linear(num_feats, 1)
        else:
            self.lr = nn.Linear(num_feats, 1, bias=False)

    def forward(self, X):
        output = self.lr(X)
        return output


# %%
# 三阶多项式拟合
batch_size = 10
feats_used = 4
train_data = TensorDataset(poly_features[:, :feats_used], labels)
test_data = TensorDataset(test_poly_feats[:, :feats_used], test_labels)
train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_data, batch_size=batch_size)
epochs = 100
lr = 0.01
loss = nn.MSELoss()
net = LinearRegression(feats_used)
optimizer = optim.SGD(net.parameters(), lr=lr)
for epoch in range(epochs):
    net.train()
    for X, y in train_iter:
        l = loss(net(X), y.reshape(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    train_l = loss(net(poly_features[:, :feats_used]), labels.reshape(-1, 1))
    net.eval()
    test_l = loss(
        net(test_poly_feats[:, :feats_used]), test_labels.reshape(-1, 1))
    print(f'epoch {epoch+1}, train loss {train_l:f}, test loss {test_l:f}')

w = net.lr.weight.data
print('w的估计值', w, 'w的估计误差：',
      true_w[:feats_used] - w.reshape(true_w[:feats_used].shape))

# %%
# underfitting
batch_size = 10
feats_used = 2
train_data = TensorDataset(poly_features[:, :feats_used], labels)
test_data = TensorDataset(test_poly_feats[:, :feats_used], test_labels)
train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_data, batch_size=batch_size)
epochs = 100
lr = 0.01
loss = nn.MSELoss()
net = LinearRegression(feats_used)
optimizer = optim.SGD(net.parameters(), lr=lr)
for epoch in range(epochs):
    net.train()
    for X, y in train_iter:
        l = loss(net(X), y.reshape(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    train_l = loss(net(poly_features[:, :feats_used]), labels.reshape(-1, 1))
    net.eval()
    test_l = loss(
        net(test_poly_feats[:, :feats_used]), test_labels.reshape(-1, 1))
    print(f'epoch {epoch+1}, train loss {train_l:f}, test loss {test_l:f}')

w = net.lr.weight.data
print('w的估计值', w, 'w的估计误差：',
      true_w[:feats_used] - w.reshape(true_w[:feats_used].shape))

# %%
# overfitting
batch_size = 10
feats_used = 20
train_data = TensorDataset(poly_features[:, :feats_used], labels)
test_data = TensorDataset(test_poly_feats[:, :feats_used], test_labels)
train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_data, batch_size=batch_size)
epochs = 100
lr = 0.01
loss = nn.MSELoss()
net = LinearRegression(feats_used)
optimizer = optim.SGD(net.parameters(), lr=lr)
for epoch in range(epochs):
    net.train()
    for X, y in train_iter:
        l = loss(net(X), y.reshape(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    train_l = loss(net(poly_features[:, :feats_used]), labels.reshape(-1, 1))
    net.eval()
    test_l = loss(
        net(test_poly_feats[:, :feats_used]), test_labels.reshape(-1, 1))
    print(f'epoch {epoch+1}, train loss {train_l:f}, test loss {test_l:f}')

w = net.lr.weight.data
print('w的估计值', w, 'w的估计误差：',
      true_w[:feats_used] - w.reshape(true_w[:feats_used].shape))

# %%
# L1/L2 penalty(weight decay=0)，一般不对bias作惩罚。
# 权重衰减使得算法偏向在大量特征上均匀分布权重。惩罚系数越大，越趋近于0。
batch_size = 10
feats_used = 4
train_data = TensorDataset(poly_features[:, :feats_used], labels)
test_data = TensorDataset(test_poly_feats[:, :feats_used], test_labels)
train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_data, batch_size=batch_size)
epochs = 400
lr = 0.01
weight_decay = 0
loss = nn.MSELoss()
net = LinearRegression(feats_used)
# optimizer = optim.SGD([{'params': net.lr.weight, 'weight_decay': weight_decay},
#                        {'params': net.lr.bias}], lr=lr)
optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
for epoch in range(epochs):
    net.train()
    for X, y in train_iter:
        l = loss(net(X), y.reshape(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    train_l = loss(net(poly_features[:, :feats_used]), labels.reshape(-1, 1))
    net.eval()
    test_l = loss(
        net(test_poly_feats[:, :feats_used]), test_labels.reshape(-1, 1))
    print(f'epoch {epoch+1}, train loss {train_l:f}, test loss {test_l:f}')

w = net.lr.weight.data
print('w的估计值', w, 'w的估计误差：',
      true_w[:feats_used] - w.reshape(true_w[:feats_used].shape),
      'w的L2范数', net.lr.weight.norm().item())

# %%
# L1/L2 penalty(weight decay=0)，一般不对bias作惩罚
batch_size = 10
feats_used = 5
train_data = TensorDataset(poly_features[:, :feats_used], labels)
test_data = TensorDataset(test_poly_feats[:, :feats_used], test_labels)
train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_data, batch_size=batch_size)
epochs = 400
lr = 0.01
weight_decay = 100
loss = nn.MSELoss()
net = LinearRegression(feats_used)
# optimizer = optim.SGD([{'params': net.lr.weight, 'weight_decay': weight_decay},
#                        {'params': net.lr.bias}], lr=lr)
optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
for epoch in range(epochs):
    net.train()
    for X, y in train_iter:
        l = loss(net(X), y.reshape(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    train_l = loss(net(poly_features[:, :feats_used]), labels.reshape(-1, 1))
    net.eval()
    test_l = loss(
        net(test_poly_feats[:, :feats_used]), test_labels.reshape(-1, 1))
    print(f'epoch {epoch+1}, train loss {train_l:f}, test loss {test_l:f}')

w = net.lr.weight.data
print('w的估计值', w, 'w的估计误差：',
      true_w[:feats_used] - w.reshape(true_w[:feats_used].shape),
      'w的L2范数', net.lr.weight.norm().item())
