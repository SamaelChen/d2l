# %%
import json
import scipy.stats
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
# %%


def calc_corr(x, y):
    return scipy.stats.spearmanr(x, y).correlation


def load_data(path, type='csv', sep=None, pred=False):
    """
    type: csv, json
    """
    if not pred:
        queries, titles, labels = [], [], []
    else:
        queries, titles, labels = [], [], None
    if type == 'csv':
        if not sep:
            raise ValueError('Need a separator when file type is csv')
        with open(path, 'r') as f:
            for line in tqdm(f.readlines()):
                if pred:
                    q, t = line.split(sep)
                    queries.append(q)
                    titles.append(t)
                else:
                    q, t, l = line.split(sep)
                    queries.append(q)
                    titles.append(t)
                    labels.append(l)
    elif type == 'json':
        with open(path, 'r') as f:
            for line in tqdm(f.readlines()):
                tmp = json.loads(line)
                if pred:
                    queries.append(tmp['sentence1'])
                    titles.append(tmp['sentence2'])
                else:
                    queries.append(tmp['sentence1'])
                    titles.append(tmp['sentence2'])
                    labels.append(tmp['label'])
    return {'queries': queries, 'titles': titles, 'labels': labels}


def calc_cosim(vec_a, vec_b):
    sim = torch.sum(vec_a * vec_b, dim=1)
    return sim


def cosentloss(pred, label, alpha):
    pass
# %%


class CustomDataset(Dataset):
    def __init__(self, queries, titles, labels=None):
        self.queries = queries
        self.titles = titles
        self.labels = labels

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        if self.labels:
            return self.queries[index], self.titles[index], self.labels[index]
        else:
            return self.queries[index], self.titles[index]
