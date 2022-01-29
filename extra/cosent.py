# %%
import torch
from torch import cosine_similarity, nn
from transformers import BertConfig, BertModel, BertTokenizer

# %%


class CoSent(nn.Module):
    def __init__(self) -> None:
        super(CoSent, self).__init__()
        self.config = BertConfig.from_pretrained("bert-base-chinese")
        self.bert = BertModel.from_pretrained("bert-base-chinese")

    def get_embedding(self,
                      input_ids,
                      token_type_ids=None,
                      position_ids=None,
                      attention_mask=None,
                      encoder_type="first-last-avg"):
        """
        encoder_type: first-last-avg, last-avg, clf, pooler(clf+dense)
        """
        self.encoder_type = encoder_type
        output = self.bert(input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           position_ids=position_ids,
                           return_dict=True,
                           output_hidden_states=True)
        if self.encoder_type == 'first-last_avg':
            first = output.hidden_states[1]
            last = output.last_hidden_state
            seq_len = first.size[1]

            first_avg = torch.avg_pool1d(first.transpose(1, 2),
                                         kernel_size=seq_len)
            last_avg = torch.avg_pool1d(last.transpose(1, 2),
                                        kernel_size=seq_len)
            final_encoding = torch.avg_pool1d(torch.cat([first_avg, last_avg],
                                                        dim=2),
                                              kernel_size=2).squeeze(-1)
            return final_encoding
        elif self.encoder_type == 'last-avg':
            last = output.last_hidden_state
            seq_len = last.size(1)
            final_encoding = torch.avg_pool1d(last.transpose(1, 2),
                                              kernel_size=seq_len).squeeze(-1)
            return final_encoding
        elif self.encoder_type == 'clf':
            last = output.last_hidden_state
            clf = last[:, 0]
            return clf
        elif self.encoder_type == 'pooler':
            pooler = output.pooler_output
            return pooler

    def cosine_sim(self, query_vec, title_vec):
        cosine_similarity = torch.sum(query_vec * title_vec, axis=-1) / (torch.sqrt(torch.sum(
            query_vec*query_vec, axis=-1))*torch.sqrt(torch.sum(title_vec*title_vec, axis=-1)))
        return cosine_similarity

    def forward(self, query, title):
        if self.encoder_type == "first-last-avg":
            first = output.hidden_states[1]


# %%
model = BertModel.from_pretrained("bert-base-chinese")
# %%
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# %%
a = tokenizer(["你说说到底有什么小说软件好用的"], padding=True, truncation=True,
              max_length=32, return_tensors='pt')

# %%
len(a['input_ids'][0])
# %%
output = model(input_ids=torch.tensor(a['input_ids']),
               attention_mask=torch.tensor(a['attention_mask']),
               output_hidden_states=True)

# %%
len(output.hidden_states)
# %%
output.last_hidden_state == output.hidden_states[-1]
# %%
