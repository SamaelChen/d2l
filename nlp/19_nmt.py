# %%
import collections
import numpy as np
from scipy.sparse.construct import rand
from torch.nn.modules import conv
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import MaxPool2d
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
import re
import random
from d2l import torch as d2l
# %%
