# %%
import torch
from torch import nn
from transformers import BertConfig, BertModel, BertTokenizer

# %%
class CoSent(nn.Module):
    def __init__(self, encoder_type="first-last-avg") -> None:
        """
        encoder_type: first-last-avg, last-avg, cls, pooler(cls+dense)
        """
        super(CoSent, self).__init__()
        self.config = BertConfig.from_pretrained("bert-base-chinese")
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.encoder_type = encoder_type

    def forward(self, query, title):
        if self.encoder_type == "first-last-avg":
            first = output.hidden_states[1]


# %%
model = BertModel.from_pretrained("bert-base-chinese")
# %%
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# %%
tokenizer("有什么小说软件好用的")


# %%
