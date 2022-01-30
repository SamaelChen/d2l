# %%
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertConfig, BertModel, BertTokenizer
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
# %%


class CoSent(nn.Module):
    def __init__(self, bert='bert-base-chinese') -> None:
        super(CoSent, self).__init__()
        self.config = BertConfig.from_pretrained(bert)
        self.bert = BertModel.from_pretrained(bert)
        self.tokenizer = BertTokenizer.from_pretrained(bert)

    def get_embedding(self,
                      input_ids,
                      token_type_ids=None,
                      position_ids=None,
                      attention_mask=None,
                      encoder_type="first-last-avg"):
        """
        encoder_type: first-last-avg, last-avg, clf, pooler(clf+dense)
        """
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           position_ids=position_ids,
                           return_dict=True,
                           output_hidden_states=True)
        if encoder_type == 'first-last-avg':
            first = output.hidden_states[1]
            last = output.last_hidden_state
            seq_len = first.size(1)
            first_avg = torch.avg_pool1d(first.transpose(1, 2),
                                         kernel_size=seq_len)
            last_avg = torch.avg_pool1d(last.transpose(1, 2),
                                        kernel_size=seq_len)
            final_encoding = torch.avg_pool1d(torch.cat([first_avg, last_avg], dim=2),
                                              kernel_size=2).squeeze(-1)
            return final_encoding
        elif encoder_type == 'last-avg':
            last = output.last_hidden_state
            seq_len = last.size(1)
            final_encoding = torch.avg_pool1d(last.transpose(1, 2),
                                              kernel_size=seq_len).squeeze(-1)
            return final_encoding
        elif encoder_type == 'clf':
            last = output.last_hidden_state
            clf = last[:, 0]
            return clf
        elif encoder_type == 'pooler':
            pooler = output.pooler_output
            return pooler
        else:
            raise ValueError(
                '{} not supported, only first-last-avg, last-avg, clf, pooler'.format(encoder_type))

    def cosine_sim(self, query_vec, title_vec):
        query_vec = F.normalize(query_vec, p=2, dim=-1)
        title_vec = F.normalize(title_vec, p=2, dim=-1)
        cos_sim = torch.sum(query_vec * title_vec, axis=-1)
        return cos_sim

    def forward(self, query, title,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None,
                title_token_type_ids=None,
                title_position_ids=None,
                title_attention_mask=None,
                label=None,
                alpha=20,
                encoder_type="first-last-avg"):
        query_vec = self.get_embedding(input_ids=query,
                                       token_type_ids=query_token_type_ids,
                                       position_ids=query_position_ids,
                                       attention_mask=query_attention_mask,
                                       encoder_type=encoder_type)
        title_vec = self.get_embedding(input_ids=title,
                                       token_type_ids=title_token_type_ids,
                                       position_ids=title_position_ids,
                                       attention_mask=title_attention_mask,
                                       encoder_type=encoder_type)
        cos_sim = self.cosine_sim(query_vec, title_vec)*alpha
        cos_sim = cos_sim[:, None] - cos_sim[None, :]
        label = label[:, None] < label[None, :]
        label = label.long()
        cos_sim = cos_sim - (1-label) * 1e12
        # 拼接一个0是因为e^0=1，相当于log中加1
        cos_sim = torch.cat((torch.zeros(1), cos_sim.view(-1)), dim=0)
        loss = torch.logsumexp(cos_sim, dim=0)
        return cos_sim, loss

    def save(self, path):
        self.bert.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.config.save_pretrained(path)


# %%
cosent = CoSent()
# %%
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# %%
a = tokenizer(["你说说到底有什么小说软件好用的", "推荐一个基金"],
              padding=True,
              truncation=True,
              max_length=32, return_tensors='pt')
b = tokenizer(["有什么理财软件好用的", "推荐基金"],
              padding=True,
              truncation=True,
              max_length=32, return_tensors='pt')
# %%
cosent(query=a['input_ids'], title=b['input_ids'],
       query_attention_mask=a['attention_mask'],
       title_attention_mask=b['attention_mask'],
       label=torch.tensor([0, 1]))
# %%
