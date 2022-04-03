# %%
import collections
import os
from platform import version
import sys
from tqdm import tqdm
import mido
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import pickle
import random
import logging
# %%
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
# %%


class Vocab:
    """Vocabulary for text."""

    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        """Defined in :numref:`sec_text-sequence`"""
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            if isinstance(tokens[0][0], list):
                tokens = [
                    token for paragraph in tokens for line in paragraph for token in line]
            else:
                tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']


# %%
def midformatter(midi):
    note, velocity, time = [], [], []
    mid = mido.MidiFile(midi)
    for msg in mid:
        if not msg.is_meta and msg.type == 'note_on':
            note.append(str(msg.note))
            velocity.append(str(msg.velocity))
            time.append(msg.time+1)
    return note, velocity, time


def format_midi(file_dir, dump_dir, dump=True, return_res=True):
    if not os.path.isdir(file_dir):
        raise ValueError('Wrong path')
    else:
        files = os.listdir(file_dir)
        notes, velocities, times = [], [], []
        for f in tqdm(files):
            note, velocity, time = midformatter(os.path.join(file_dir, f))
            notes.append(note)
            velocities.append(velocity)
            times.append(time)
    seg_mid(notes, velocities, times)
    notes = flat_3d_list(notes)
    velocities = flat_3d_list(velocities)
    times = flat_3d_list(times)
    if dump:
        if not os.path.isdir(dump_dir):
            os.makedirs(dump_dir)
        with open(os.path.join(dump_dir, 'mid.pkl'), 'wb') as f:
            pickle.dump([notes, velocities, times], f)
    if return_res:
        return notes, velocities, times


def seg_mid(notes, velocities, times):
    logging.info('split midi into 32 notes long sentence')
    for idx, (note, velocity, time) in tqdm(enumerate(zip(notes, velocities, times)), total=len(notes)):
        note_split, velocity_split, time_split = [], [], []
        i = 0
        while i < len(note):
            note_split.append(note[i: i+32])
            velocity_split.append(velocity[i: i+32])
            time_split.append(time[i: i+32])
            i += 32
        note_split[0] = ['<bos>'] + note_split[0]
        note_split[-1] = note_split[-1] + ['<eos>']
        velocity_split[0] = ['<bos>'] + velocity_split[0]
        velocity_split[-1] = velocity_split[-1] + ['<eos>']
        time_split[0] = [-1] + time_split[0]
        time_split[-1] = time_split[-1] + [-1]
        notes[idx] = note_split
        velocities[idx] = velocity_split
        times[idx] = time_split


def flat_3d_list(lst):
    lst = [line for paragraph in lst for line in paragraph]
    return lst


def load_midi(pkl_file):
    if not os.path.exists(pkl_file):
        raise ValueError('Wrong path')
    else:
        with open(pkl_file, 'rb') as f:
            notes, velocities, times = pickle.load(f)
    return notes, velocities, times


def get_tempo(mid):
    for msg in mid:
        if msg.type == 'set_tempo':
            return msg.tempo


def replace_lm_tokens(
        note_tokens,
        velocity_tokens,
        time_series,
        candidate_pred_positions,
        num_mlm_preds):
    # 为遮蔽语言模型的输入创建新的词元副本，其中输入可能包含替换的“<mask>”或随机词元
    pred_positions_and_labels = []
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        pred_positions_and_labels.append(
            (mlm_pred_position, note_tokens[mlm_pred_position],
             velocity_tokens[mlm_pred_position],
             time_series[mlm_pred_position]))
    return pred_positions_and_labels


def get_lm_data_from_tokens(note_tokens,
                            velocity_tokens,
                            time_series,
                            note_vocab,
                            velocity_vocab):
    candidate_pred_positions = []
    # tokens是一个字符串列表
    for i, token in enumerate(note_tokens):
        # 在遮蔽语言模型任务中不会预测特殊词元
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 遮蔽语言模型任务中预测15%的随机词元
    num_mlm_preds = max(1, round(len(note_tokens) * 0.15))
    pred_positions_and_labels = replace_lm_tokens(
        note_tokens, velocity_tokens, time_series,
        candidate_pred_positions, num_mlm_preds)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    note_pred_labels = [v[1] for v in pred_positions_and_labels]
    velocity_pred_labels = [v[2] for v in pred_positions_and_labels]
    time_pred_labels = [v[3] for v in pred_positions_and_labels]
    return (pred_positions, note_vocab[note_tokens], note_vocab[note_pred_labels],
            velocity_vocab[velocity_tokens], velocity_vocab[velocity_pred_labels],
            time_series, time_pred_labels)


def pad_inputs(examples, max_len, note_vocab, velocity_vocab):
    max_num_lm_preds = round(max_len * 0.15)
    all_note_token_ids, all_velocity_token_ids, all_times, valid_lens,  = [], [], [], []
    all_pred_positions, all_lm_weights = [], []
    all_note_lm_labels, all_velocity_lm_labels, all_times_lm_labels = [], [], []
    for (pred_positions, note_token_ids, lm_note_pred_label_ids,
         velocity_token_ids, lm_velocity_pred_label_ids,
         times, lm_time_pred_labels) in examples:
        all_note_token_ids.append(torch.tensor(note_token_ids + [note_vocab['<pad>']] * (
            max_len - len(note_token_ids)), dtype=torch.long))
        all_velocity_token_ids.append(torch.tensor(velocity_token_ids + [velocity_vocab['<pad>']] * (
            max_len - len(velocity_token_ids)), dtype=torch.long))
        all_times.append(torch.tensor(times + [0] * (
            max_len - len(times)), dtype=torch.float32))
        # valid_lens不包括'<pad>'的计数
        valid_lens.append(torch.tensor(len(note_token_ids),
                                       dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_lm_preds - len(pred_positions)), dtype=torch.long))
        # 填充词元的预测将通过乘以0权重在损失中过滤掉
        all_lm_weights.append(
            torch.tensor([1.0] * len(lm_note_pred_label_ids) + [0.0] * (
                max_num_lm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_note_lm_labels.append(torch.tensor(lm_note_pred_label_ids + [0] * (
            max_num_lm_preds - len(lm_note_pred_label_ids)), dtype=torch.long))
        all_velocity_lm_labels.append(torch.tensor(lm_velocity_pred_label_ids + [0] * (
            max_num_lm_preds - len(lm_velocity_pred_label_ids)), dtype=torch.long))
        all_times_lm_labels.append(torch.tensor(lm_time_pred_labels + [0] * (
            max_num_lm_preds - len(lm_time_pred_labels)), dtype=torch.float32))
    return (all_note_token_ids, all_velocity_token_ids, all_times,
            valid_lens, all_pred_positions, all_lm_weights,
            all_note_lm_labels, all_velocity_lm_labels, all_times_lm_labels)


# %%
class MidiDataset(Dataset):
    def __init__(self, notes, velocities, times,
                 note_vocab, velocity_vocab, max_len) -> None:
        exmp = []
        for note, velocity, time in zip(notes, velocities, times):
            exmp.append(get_lm_data_from_tokens(note, velocity, time,
                                                note_vocab, velocity_vocab))
        (self.all_note_token_ids, self.all_velocity_token_ids, self.all_times,
         self.valid_lens, self.all_pred_positions, self.all_lm_weights,
         self.all_note_lm_labels, self.all_velocity_lm_labels,
         self.all_times_lm_labels) = pad_inputs(
            exmp, max_len, note_vocab, velocity_vocab)

    def __getitem__(self, idx):
        return(self.all_note_token_ids[idx], self.all_velocity_token_ids[idx],
               self.all_times[idx], self.valid_lens[idx], self.all_pred_positions[idx],
               self.all_lm_weights[idx], self.all_note_lm_labels[idx],
               self.all_velocity_lm_labels[idx], self.all_times_lm_labels[idx])

    def __len__(self):
        return len(self.all_note_token_ids)


def load_data(batch_size, notes, velocities, times,
              note_vocab, velocity_vocab, max_len):
    num_workers = torch.get_num_threads()
    train_set = MidiDataset(notes, velocities, times,
                            note_vocab, velocity_vocab, max_len)
    train_iter = DataLoader(train_set, batch_size, shuffle=True,
                            num_workers=num_workers)
    return train_iter


# %%
