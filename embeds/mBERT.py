import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import *


class axComp(nn.Module):

    def __init__(self, model_type='bert-base-uncased'):
        super(axComp, self).__init__()

        self.bert = BertModel.from_pretrained(model_type)
        self.tokenizer = BertTokenizer.from_pretrained(model_type)

    def translate(self, input_text, token_type_ids=None, attention_mask=None, labels=None):
        self.bert.eval()

        tokenized_text = self.tokenizer.tokenize(input_text)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        embedded_data, pooled_output = self.bert(torch.tensor(input_ids).view(-1, 1), token_type_ids, attention_mask)

        return embedded_data, pooled_output

    def make_axes(self, col1, col2, df):
        return [self.translate(phrase[0])[0] - self.translate(phrase[1])[0] for phrase in df[[col1, col2]].values]

    def df_targets(self, col1, col2, df):
        data = [' '.join(df[[col1, col2]].loc[i].values.tolist()) for i in df.index]
        return [self.translate(phrase)[0].sum(dim=0) for phrase in data]

    def target(self, phrase):
        return self.translate(phrase)[0].sum(dim=0)

    def lexeme(self, word):
        return self.translate(word)[0].squeeze(1)


