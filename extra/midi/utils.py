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


def replace_mlm_tokens(
        note_tokens,
        velocity_tokens,
        time_series,
        candidate_pred_positions,
        num_mlm_preds,
        note_vocab,
        velocity_vocab):
    # 为遮蔽语言模型的输入创建新的词元副本，其中输入可能包含替换的“<mask>”或随机词元
    mlm_note_tokens = [token for token in note_tokens]
    mlm_velocity_tokens = [token for token in velocity_tokens]
    pred_positions_and_labels = []
    # 打乱后用于在遮蔽语言模型任务中获取15%的随机词元进行预测
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80%的时间：将词替换为“<mask>”词元
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10%的时间：保持词不变
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10%的时间：用随机词替换该词
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


# %%
format_midi(
    '/Users/samael/Downloads/GiantMIDI-PIano/sample',
    '/Users/samael/Downloads/GiantMIDI-PIano/pkl_file',
    return_res=False)
# %%
notes, velocities, times = load_midi(
    '/Users/samael/Downloads/GiantMIDI-PIano/pkl_file/mid.pkl')
# %%
note_vocab = Vocab(notes, reserved_tokens=[
                   '<pad>', '<mask>', '<cls>', '<sep>'])
# %%
