# %%
from tqdm import tqdm
from zmq import device
from model import GPTEncoder, LM, GPTModel
from utils import *
import torch
from torch import nn
import mido
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
# %%
format_midi(
    '/Users/samael/Downloads/GiantMIDI-PIano/sample',
    '/Users/samael/Downloads/GiantMIDI-PIano/pkl_file',
    return_res=False)
# %%
notes, velocities, times = load_midi(
    '/Users/samael/Downloads/GiantMIDI-PIano/pkl_file/mid.pkl')

note_vocab = Vocab(notes, reserved_tokens=[
                   '<pad>', '<mask>', '<cls>', '<sep>'])
velocity_vocab = Vocab(velocities, reserved_tokens=[
    '<pad>', '<mask>', '<cls>', '<sep>'])

# %%
batch_size, max_len = 128, 33
train_iter = load_data(batch_size, notes, velocities,
                       times, note_vocab, velocity_vocab,
                       max_len)
# %%
for (note_token_ids, velocity_token_ids,
     times, valid_lens, pred_positions,
     lm_weights, note_lm_labels,
     velocity_lm_labels, times_lm_labels) in train_iter:
    print(note_token_ids.shape)
    break

# %%
model = GPTModel(len(note_vocab), len(velocity_vocab),
                 num_hiddens=128, norm_shape=[128],
                 ffn_num_input=128, ffn_num_hiddens=256,
                 num_heads=2, num_layers=2, dropout=0.2,
                 key_size=128, query_size=128, value_size=128,
                 hid_in_features=128, lm_in_features=128)

# %%
celoss = nn.CrossEntropyLoss()
mseloss = nn.MSELoss()
gpu = torch.cuda.is_available()

# %%
get_batch_loss(model, celoss, mseloss, len(note_vocab),
               len(velocity_vocab), note_token_ids, velocity_token_ids,
               times, pred_positions, valid_lens, lm_weights,
               note_lm_labels, velocity_lm_labels, times_lm_labels)

# %%
