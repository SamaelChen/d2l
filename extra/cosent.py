# %%
import time
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import logging
import json
from tqdm import tqdm
import scipy.stats
import os
import argparse

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
# %%


class CoSent(nn.Module):
    def __init__(self, bert="bert-base-chinese") -> None:
        super(CoSent, self).__init__()
        self.config = BertConfig.from_pretrained(bert)
        self.bert = BertModel.from_pretrained(bert)
        self.tokenizer = BertTokenizer.from_pretrained(bert)

    def get_embedding(
        self,
        input_ids,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        encoder_type="first-last-avg",
    ):
        """
        encoder_type: first-last-avg, last-avg, clf, pooler(clf+dense)
        """
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True,
        )
        if encoder_type == "first-last-avg":
            first = output.hidden_states[1]
            last = output.last_hidden_state
            seq_len = first.size(1)
            first_avg = torch.avg_pool1d(
                first.transpose(1, 2), kernel_size=seq_len)
            last_avg = torch.avg_pool1d(
                last.transpose(1, 2), kernel_size=seq_len)
            final_encoding = torch.avg_pool1d(
                torch.cat([first_avg, last_avg], dim=2), kernel_size=2
            ).squeeze(-1)
            return final_encoding
        elif encoder_type == "last-avg":
            last = output.last_hidden_state
            seq_len = last.size(1)
            final_encoding = torch.avg_pool1d(
                last.transpose(1, 2), kernel_size=seq_len
            ).squeeze(-1)
            return final_encoding
        elif encoder_type == "clf":
            last = output.last_hidden_state
            clf = last[:, 0]
            return clf
        elif encoder_type == "pooler":
            pooler = output.pooler_output
            return pooler
        else:
            raise ValueError(
                "{} not supported, only first-last-avg, last-avg, clf, pooler".format(
                    encoder_type
                )
            )

    def cosine_sim(self, query, title,
                   query_token_type_ids=None,
                   query_attention_mask=None,
                   title_token_type_ids=None,
                   title_attention_mask=None,
                   encoder_type="first-last-avg",):
        query_vec = self.get_embedding(
            input_ids=query,
            token_type_ids=query_token_type_ids,
            attention_mask=query_attention_mask,
            encoder_type=encoder_type,
        )
        title_vec = self.get_embedding(
            input_ids=title,
            token_type_ids=title_token_type_ids,
            attention_mask=title_attention_mask,
            encoder_type=encoder_type,
        )
        query_vec = F.normalize(query_vec, p=2, dim=-1)
        title_vec = F.normalize(title_vec, p=2, dim=-1)
        cos_sim = torch.sum(query_vec * title_vec, axis=-1)
        return cos_sim

    def forward(
        self,
        query,
        title,
        query_token_type_ids=None,
        query_attention_mask=None,
        title_token_type_ids=None,
        title_attention_mask=None,
        label=None,
        alpha=20,
        encoder_type="first-last-avg",
    ):
        cos_sim_ori = self.cosine_sim(query=query, title=title,
                                      query_token_type_ids=query_token_type_ids,
                                      query_attention_mask=query_attention_mask,
                                      title_token_type_ids=title_token_type_ids,
                                      title_attention_mask=title_attention_mask,
                                      encoder_type=encoder_type) * alpha
        cos_sim = cos_sim_ori[:, None] - cos_sim_ori[None, :]
        label = label[:, None] < label[None, :]
        label = label.long()
        cos_sim = cos_sim - (1 - label) * 1e12
        # 拼接一个0是因为e^0=1，相当于log中加1
        if torch.cuda.is_available():
            cos_sim = torch.cat(
                (torch.zeros(1).cuda(), cos_sim.view(-1)), dim=0)
        else:
            cos_sim = torch.cat((torch.zeros(1), cos_sim.view(-1)), dim=0)
        loss = torch.logsumexp(cos_sim, dim=0)
        return cos_sim_ori, loss

    def save(self, path, epoch):
        if not os.path.exists(path):
            os.makedirs(path)
        model_path = os.path.join(path, 'model_{}'.format(epoch))
        self.bert.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        self.config.save_pretrained(model_path)


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

# %%


def load_data(path):
    queries, titles, labels = [], [], []
    with open(path, 'r') as f:
        for line in tqdm(f.readlines()):
            tmp = json.loads(line)
            queries.append(tmp['sentence1'])
            titles.append(tmp['sentence2'])
            labels.append(tmp['label'])
    return {'queries': queries, 'titles': titles, 'labels': labels}
# %%


def train(epochs, traindata, testdata, lr=1e-5, train_batch_size=32, test_batch_size=2,
          max_len=64, alpha=20, encoder_type='first-last-avg', bert='bert-base-chinese'):
    trainset = CustomDataset(traindata['queries'],
                             traindata['titles'],
                             traindata['labels'])
    testset = CustomDataset(testdata['queries'],
                            testdata['titles'],
                            testdata['labels'])
    trainiter = DataLoader(trainset, batch_size=train_batch_size,
                           shuffle=True, num_workers=4)
    testiter = DataLoader(testset, batch_size=test_batch_size,
                          shuffle=False, num_workers=4)
    tokenizer = BertTokenizer.from_pretrained(bert)
    cosent = CoSent()
    gpu = torch.cuda.is_available()
    if gpu:
        cosent = cosent.cuda()
    optimizer = AdamW(params=cosent.parameters(), lr=lr)
    total_steps = len(trainiter)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.05 * total_steps,
                                                num_training_steps=total_steps)
    start = time.time()
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        cosent.train()
        for step, (q, t, l) in enumerate(trainiter):
            q = tokenizer(q,
                          padding=True,
                          truncation=True,
                          max_length=max_len,
                          return_tensors='pt')
            t = tokenizer(t,
                          padding=True,
                          truncation=True,
                          max_length=max_len,
                          return_tensors='pt')
            l = torch.tensor([int(x) for x in l], dtype=torch.long)
            q_input_ids = q['input_ids']
            q_attention_mask = q['attention_mask']
            t_input_ids = t['input_ids']
            t_attention_mask = t['attention_mask']
            if gpu:
                q_input_ids = q_input_ids.cuda()
                q_attention_mask = q_attention_mask.cuda()
                t_input_ids = t_input_ids.cuda()
                t_attention_mask = t_attention_mask.cuda()
                l = l.cuda()
            sim, loss = cosent(query=q_input_ids,
                               title=t_input_ids,
                               query_attention_mask=q_attention_mask,
                               title_attention_mask=t_attention_mask,
                               label=l,
                               alpha=alpha,
                               encoder_type=encoder_type,
                               )
            end = time.time()
            dur = end - start
            h = dur // 3600
            dur -= h * 3600
            m = dur // 60
            dur -= m * 60
            s = dur
            print("epoch:{}, steps:{}/{}, loss:{:.5f}, corr:{:.5f}, time: {}:{}:{:.3f}".format(epoch,
                                                                                               step,
                                                                                               len(
                                                                                                   trainiter),
                                                                                               loss,
                                                                                               scipy.stats.spearmanr(l.cpu().tolist(),
                                                                                                                     sim.cpu().tolist()).correlation,
                                                                                               int(h), int(m), s),
                  end='\r')
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        with torch.no_grad():
            cosent.eval()
            sim_val, label_val = [], []
            val_loss = 0
            for step, (q, t, l) in tqdm(enumerate(testiter), total=len(testiter)):
                q = tokenizer(q,
                              padding=True,
                              truncation=True,
                              max_length=max_len,
                              return_tensors='pt')
                t = tokenizer(t,
                              padding=True,
                              truncation=True,
                              max_length=max_len,
                              return_tensors='pt')
                l = torch.tensor([int(x) for x in l], dtype=torch.long)
                q_input_ids = q['input_ids']
                q_attention_mask = q['attention_mask']
                t_input_ids = t['input_ids']
                t_attention_mask = t['attention_mask']
                if gpu:
                    q_input_ids = q_input_ids.cuda()
                    q_attention_mask = q_attention_mask.cuda()
                    t_input_ids = t_input_ids.cuda()
                    t_attention_mask = t_attention_mask.cuda()
                    l = l.cuda()
                sim, loss = cosent(query=q_input_ids,
                                   title=t_input_ids,
                                   query_attention_mask=q_attention_mask,
                                   title_attention_mask=t_attention_mask,
                                   label=l,
                                   alpha=alpha,
                                   encoder_type=encoder_type,
                                   )
                val_loss += loss
                sim_val.extend(sim.cpu().tolist())
                label_val.extend(l.cpu().tolist())
            val_loss = val_loss / (step+1)
            print("epoch:{}, val_loss:{:10f}, val_corr:{:10f}".format(epoch,
                                                                      val_loss,
                                                                      scipy.stats.spearmanr(label_val, sim_val).correlation))
        cosent.save(
            '/home/samael/github/cosent/simclue/model_{}'.format(encoder_type), epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('--CoSent similarity')
    parser.add_argument(
        '--train_data', default='/home/samael/github/cosent/train_pair.json', type=str, help='train dataset')
    parser.add_argument(
        '--test_data', default='/home/samael/github/cosent/test_public.json', type=str, help='test dataset')

    parser.add_argument('--bert',
                        default='bert-base-chinese', type=str, help='pretrained bert model')
    parser.add_argument('--num_train_epochs', default=5,
                        type=int, help='epochs')
    parser.add_argument('--train_batch_size', default=32,
                        type=int, help='train batch size')
    parser.add_argument('--test_batch_size', default=8,
                        type=int, help='test batch size')
    parser.add_argument('--learning_rate', default=1e-5,
                        type=float, help='learning rate')
    parser.add_argument('--max_len', default=64,
                        type=int, help='learning rate')
    parser.add_argument('--alpha', default=20,
                        type=int, help='learning rate')
    parser.add_argument('--encoder_type', default='first-last-avg',
                        type=str, help='first-last-avg, last-avg, clf, pooler(clf+dense)')
    args = parser.parse_args()
    traindata = load_data(args.train_data)
    testdata = load_data(args.test_data)
    train(epochs=args.num_train_epochs,
          traindata=traindata,
          testdata=testdata,
          lr=args.learning_rate,
          train_batch_size=args.train_batch_size,
          test_batch_size=args.test_batch_size,
          max_len=args.max_len,
          alpha=args.alpha,
          encoder_type=args.encoder_type,
          bert=args.bert)
# %%
