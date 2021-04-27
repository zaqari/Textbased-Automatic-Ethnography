import torch.nn as nn
import torch
from transformers import *

class axComp(nn.Module):

    def __init__(self):
        super(axComp, self).__init__()

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        config = GPT2Config.from_pretrained('gpt2', output_hidden_states=True)
        self.mod = GPT2Model.from_pretrained('gpt2', config=config)

    def translate(self, lexeme_text):
        self.mod.eval()

        tokens = torch.tensor(self.tokenizer.encode(lexeme_text, add_special_tokens=False)).unsqueeze(0)
        outputs = self.mod(tokens)

        return outputs[2][8]

    def make_axes(self, col1, col2, df):
        return [self.translate(phrase[0])[0] - self.translate(phrase[1])[0] for phrase in df[[col1, col2]].values]

    def df_targets(self, col1, col2, df):
        data = [' '.join(df[[col1, col2]].loc[i].values.tolist()) for i in df.index]
        return [self.translate(phrase)[0].sum(dim=0) for phrase in data]

    def target(self, phrase):
        return self.translate(phrase)[0].sum(dim=0)

    def lexeme(self, word):
        return self.translate(word)[0].squeeze(1)