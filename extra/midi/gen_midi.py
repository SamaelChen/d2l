# %%
import mido
from mido import Message, MidiFile, MidiTrack
from utils import *
# %%


def predict_midi(net, note_vocab, velocity_vocab, num_steps):
    """Predict for sequence to sequence.
    Defined in :numref:`sec_utils`"""
    # Set `net` to eval mode for inference
    net.eval()
    note_tokens = ['<bos>', '<pad>']
    velocity_tokens = ['<bos>', '<pad>']
    time_tokens = [-1., 0.]
    note_tokens_id = torch.tensor(
        [note_vocab[x] for x in note_tokens]).view(1, -1)
    velocity_tokens_ids = torch.tensor(
        [velocity_vocab[x] for x in velocity_tokens]).view(1, -1)
    times = torch.tensor(time_tokens).view(1, -1)
    valid_len = torch.tensor([len(note_tokens)-1])
    tokens = (note_tokens_id, velocity_tokens_ids, times)
    pred_position = torch.tensor([[len(note_tokens)-1]])
    for _ in range(num_steps):
        _, next_note, next_velocity, next_time = net(
            tokens, valid_len, pred_position)
        next_note = torch.argmax(next_note.squeeze(0).squeeze(0))
        next_note = note_vocab.idx_to_token[next_note]
        next_velocity = torch.argmax(next_velocity.squeeze(0).squeeze(0))
        next_velocity = velocity_vocab.idx_to_token[next_velocity]
        next_time = round(float(next_time.squeeze(0).squeeze(0)) - 1, 4)
        next_time = 0 if next_time < 0 and next_time > -1 else next_time
        if (next_note == '<eos>' or next_velocity == '<eos>'):
            return note_tokens[1: -1], velocity_tokens[1: -1], time_tokens[1: -1]
        note_tokens = note_tokens[:-1] + [next_note, '<pad>']
        velocity_tokens = velocity_tokens[:-1] + [next_velocity, '<pad>']
        time_tokens = time_tokens[:-1] + [next_time, 0.]
    return note_tokens[1: -1], velocity_tokens[1: -1], time_tokens[1: -1]


# %%
def generate_midi(notes, velocities, times):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(Message('program_change', program=12, time=0))
    track.append(Message('note_on', note=64, velocity=64, time=32))
    track.append(Message('note_off', note=64, velocity=127, time=32))
    for note, velocity, time in zip(notes, velocities, times):
        track.append(Message('note_on', note=int(note),
                     velocity=int(velocity), time=time, channel=0))
        track.append(Message('note_off', note=int(note),
                     velocity=int(velocity), time=time, channel=0))
    mid.save('new_gen.mid')


# %%
