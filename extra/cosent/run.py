# %%
from cmath import cos
import os
import sys
import random
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from model import CoSENT
from utils import calc_corr, load_data, calc_cosim, cosentloss, CustomDataset
# %%


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, valdata, max_len, tokenizer, enctype='first-last-avg'):
    model.eval()
    all_sims = []
    all_labels = []
    for q, t, l in tqdm(valdata):
        q = tokenizer(q, padding=True, truncation=True,
                      max_length=max_len, return_tensors='pt')
        t = tokenizer(t, padding=True, truncation=True,
                      max_length=max_len, return_tensors='pt')
        l = torch.tensor([int(x) for x in l], dtype=torch.long)
        q_input_ids = q['input_ids']
        q_attention_mask = q['attention_mask']
        t_input_ids = t['input_ids']
        t_attention_mask = t['attention_mask']
        if torch.cuda.is_available():
            q_input_ids = q_input_ids.cuda()
            q_attention_mask = q_attention_mask.cuda()
            t_input_ids = t_input_ids.cuda()
            t_attention_mask = t_attention_mask.cuda()
            l = l.cuda()
        with torch.no_grad():
            q_vec = model(input_ids=q_input_ids,
                          attention_mask=q_attention_mask, type=enctype)
            t_vec = model(input_ids=t_input_ids,
                          attention_mask=t_attention_mask, type=enctype)
        sim = calc_cosim(q_vec, t_vec)
        all_sims.extend(sim.cpu().tolist())
        all_labels.extend(l.cpu().tolist())
    corr = calc_corr(all_labels, all_sims)
    return corr


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('--CoSENT')
    parser.add_argument('--train_data', default='/home/samael/github/cosent/train_pair.json',
                        type=str, help='train dataset')
    parser.add_argument('--test_data', default='/home/samael/github/cosent/test_public.json',
                        type=str, help='test dataset')
    parser.add_argument('--data_type', default='json',
                        type=str, help='data type json or csv')
    parser.add_argument('--sep', default='\t',
                        type=str, help='seperator of csv type date')
    parser.add_argument('--save_path', default='/home/samael/github/cosent/simclue',
                        type=str, help='path to save model')
    parser.add_argument('--bert', default='bert-base-chinese',
                        type=str, help='pretrained bert model')
    parser.add_argument('--num_train_epochs', default=5,
                        type=int, help='epochs')
    parser.add_argument('--train_batch_size', default=32,
                        type=int, help='train batch size')
    parser.add_argument('--test_batch_size', default=8,
                        type=int, help='test batch size')
    parser.add_argument('--seed', default=42, type=int,
                        help='random seed')
    parser.add_argument('--gradient_accumulation_steps', default=1,
                        type=int, help='update a gradient after how many steps')
    parser.add_argument('--learning_rate', default=2e-5,
                        type=float, help='learning rate')
    parser.add_argument('--max_len', default=64,
                        type=int, help='max length of sentence')
    parser.add_argument('--alpha', default=20,
                        type=int, help='parameter of loss function')
    parser.add_argument('--encoder_type', default='first-last-avg',
                        type=str, help='first-last-avg, last-avg, clf, pooler(clf+dense)')
    args = parser.parse_args()
    set_seed(args.seed)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    traindata = load_data(args.train_data, type=args.data_type, sep=args.sep)
    valdata = load_data(args.test_data, type=args.data_type, sep=args.sep)
    trainset = CustomDataset(traindata['queries'],
                             traindata['titles'],
                             traindata['labels'])
    valset = CustomDataset(valdata['queries'],
                           valdata['titles'],
                           valdata['labels'])
    trainiter = DataLoader(trainset, batch_size=args.train_batch_size,
                           shuffle=True)
    valiter = DataLoader(valset, batch_size=args.test_batch_size,
                         shuffle=False)
    total_steps = len(trainiter) * args.num_train_epochs
    cosent = CoSENT(pretrained=args.bert)
    tokenizer = BertTokenizer.from_pretrained(args.bert)
    gpu = torch.cuda.is_available()
    if gpu:
        cosent = cosent.cuda()
    param_optimizer = list(cosent.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                  lr=args.learning_rate)
    schedular = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=0.05*total_steps,
                                                num_training_steps=total_steps)
    for epoch in range(args.num_train_epochs):
        cosent.train()
        train_label, train_sim = [], []
        train_loss = 0
        for step, (q, t, l) in enumerate(trainiter):
            q = tokenizer(q, padding=True, truncation=True,
                          max_length=args.max_len, return_tensors='pt')
            t = tokenizer(t, padding=True, truncation=True,
                          max_length=args.max_len, return_tensors='pt')
            l = torch.tensor([int(x) for x in l], dtype=torch.long)
            q_input_ids = q['input_ids']
            q_attention_mask = q['attention_mask']
            t_input_ids = t['input_ids']
            t_attention_mask = t['attention_mask']
            if torch.cuda.is_available():
                q_input_ids = q_input_ids.cuda()
                q_attention_mask = q_attention_mask.cuda()
                t_input_ids = t_input_ids.cuda()
                t_attention_mask = t_attention_mask.cuda()
                l = l.cuda()
            q_vec = cosent(input_ids=q_input_ids,
                           attention_mask=q_attention_mask, type=args.encoder_type)
            t_vec = cosent(input_ids=t_input_ids,
                           attention_mask=t_attention_mask, type=args.encoder_type)
            sim = calc_cosim(q_vec, t_vec)
            loss = cosentloss(sim, l, args.alpha)
            loss.backward()
            print('epoch {}, step {} / {}, loss: {:.5f}'.format(epoch +
                  1, step+1, len(trainiter), loss), end='\r')
            train_loss += loss
            if (step+1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                schedular.step()
                optimizer.zero_grad()
        print('')
        corr = evaluate(cosent, valiter, args.max_len, tokenizer)
        print('epoch {}, val corr: {:.10f}'.format(epoch+1, corr))
        model_to_save = cosent.module if hasattr(cosent, 'module') else cosent
        output_path = os.path.join(
            args.save_path, 'model_epoch_{}.pt'.format(epoch+1))
        torch.save(model_to_save.state_dict(), output_path)
