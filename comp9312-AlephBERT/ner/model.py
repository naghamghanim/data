from torch import nn
from transformers import BertModel, BertConfig
import logging

logger = logging.getLogger(__name__)


class BertTagger(nn.Module):
    def __init__(self, bert_model="aubmindlab/bert-base-arabertv2", num_labels=2, dropout=0.1):
        super().__init__()

        self.bert_model = bert_model
        self.num_labels = num_labels
        self.dropout = dropout

        self.bert = BertModel.from_pretrained(bert_model)
        bert_config = BertConfig.from_pretrained(bert_model)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(bert_config.hidden_size, num_labels)

    def forward(self, x):
        output = self.bert(x)
        y = self.dropout(output.last_hidden_state)
        logits = self.linear(y)
        return logits
