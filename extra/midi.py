# %%
from tqdm import tqdm
import mido
# %%
mid = mido.MidiFile('/Users/samael/Downloads/beethoven_opus10_1_format0.midi')

# %%
for msg in tqdm(mid.tracks[0]):
    if msg.type == 'note_on' and msg.note <= 20:
        print(msg)
# %%
msg.channel
# %%
len(mid.tracks)
# %%
len(mid.tracks[0])
# %%
msg.type
# %%
mid.tracks[0][1000]
# %%
