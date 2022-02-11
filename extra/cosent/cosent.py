# %%
import tokenizers
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertConfig, BertModel
from transformers.models.bert import BertTokenizer

# %%


class CoSENT(nn.Module):
    def __init__(self, pretrained='bert-base-chinese') -> None:
        super(CoSENT, self).__init__()
        self.config = BertConfig.from_pretrained(pretrained)
        self.bert = BertModel.from_pretrained(pretrained, config=self.config)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)

    def forward(self, input_ids, attention_mask, encoder_type='first-last-avg'):
        """
        encoder_type: first-last-avg, last-avg, cls, pooler
        """
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
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
        else:
            pooler = output.pooler_output
            return pooler


# %%
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# %%
a = tokenizer(['测试一下不是吧', '是不是'], padding=True, truncation=True, text_pair=None,
              add_special_tokens=True, return_token_type_ids=True, return_tensors='pt')
# %%
a['input_ids']
# %%
a['attention_mask']
# %%
model = CoSENT()
# %%
output = model(a['input_ids'], a['attention_mask'])
# %%
