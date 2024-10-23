import copy
import math

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import BertModel

class Res_BERT(nn.Module):
    def __init__(self):
        super(Res_BERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)
        self.vismap2text = nn.Linear(2048, 768)

    def forward(self, input_ids, visual_embeds_att, input_mask, added_attention_mask, hashtag_input_ids,
                hashtag_input_mask, labels=None):
        # b*75*768
        bert_model = self.bert(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask)
        sequence_output = bert_model[0]
        pooled_output = bert_model[1]
        # batchsize*49*2048
        vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)
        # b*49*768
        visual = self.vismap2text(vis_embed_map)
        # b*1*768
        res = torch.cat([sequence_output, visual], dim=1).mean(1)
        pooled_output = self.dropout(res)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return loss
        else:
            return logits
