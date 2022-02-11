# %%
import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from d2l.extra.d2l.extra.cosent.model.model import CoSENT
from utils import calc_corr, load_data, calc_cosim, cosentloss, CustomDataset
from transformers import BertTokenizer, get_linear_schedule_with_warmup
# %%
