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
# %%


def mid2sentence(midi):
    note, velocity, time = [], [], []
    mid = mido.MidiFile(midi)
    for msg in mid:
        if not msg.is_meta and msg.type == 'note_on':
            note.append(msg.note)
            velocity.append(msg.velocity)
            time.append(msg.time+1)
    return note, velocity, time


def load_midi(file_dir):
    if not os.path.isdir(file_dir):
        raise ValueError('Wrong path')
    else:
        files = os.listdir(file_dir)
        notes, velocities, times = [], [], []
        for f in files:
            note, velocity, time = mid2sentence(f)
            notes.append(note)
            velocities.append(velocity)
            times.append(time)


def get_tempo(mid):
    for msg in mid:
        if msg.type == 'set_tempo':
            return msg.tempo


# %%
tempo = []
dirs = '/Users/samael/Downloads/GiantMIDI-PIano/surname_checked_midis'
for f in tqdm(os.listdir(dirs)):
    mid = mido.MidiFile(os.path.join(dirs, f))
    tempo.append(mid.ticks_per_beat)

print(set(tempo))

# %%


class Vocab:
    """Vocabulary for text."""

    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        """Defined in :numref:`sec_text-sequence`"""
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
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
