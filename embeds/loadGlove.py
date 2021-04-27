import gensim
#import gensim.models.word2vec as w2v
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import zipfile
import torch

#zip=zipfile.ZipFile('w2vMods/files/glove.6B.zip')

class GloveModel__():

    def __init__(self):
        self.wv = {}
        self.embedding_cols = None

    def loadGlove(self, dims=300):
        file = open("w2vMods/files/_vecModels/glove-6B-300d.txt", 'r')
        model = {}
        print('Loading Glove Model.')
        for _, line in enumerate(file.readlines()):
            splitline = line.split()
            #self.wv.append(np.array([[splitline[0]]+[np.float(i) for i in splitline[1:]]]))
            self.wv[splitline[0]] = np.array([np.float(i) for i in splitline[1:]]).reshape(1,-1)
        #self.embedding_cols = [str(i) for i in range(dims)]
        #self.wv = pd.DataFrame(np.array(self.wv).reshape(-1,dims+1), columns=['lex']+self.embedding_cols)
        file.close()
        print('Glove Model loaded: {} lexemes complete.'.format(len(model)))

    def cosTopN(self, lex, N):
        llex = self.wv[lex]
        val_names = []
        vals = []
        for k,v in self.wv.items():
            if k != lex:
                vals.append(v)
                val_names.append(k)
        vals = np.concatenate(vals, axis=0)
        delta = cosine_similarity(lex, vals)
        i = np.argsort(delta)[:1 + N]
        return [(val_names[j], delta[j]) for j in i]

    """def eucTopN(self, lex, N):
        dif = self.wv[self.embedding_cols].loc[self.wv['lex'].isin([lex])].values - self.wv[self.embedding_cols].values
        delta = np.sum(dif * dif, axis=1)
        i = np.argsort(delta)[1:1+N]
        return [(self.wv['lex'].loc[j], delta[j]) for j in i]"""

    def __getitem__(self, lex):
        return self.wv[lex]

class GloVe():

    def __init__(self):
        super(GloVe, self).__init__()
        self.wv = {}
        self.embedding_cols = None
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    def loadGlove(self, dims=300):
        file = open("w2vMods/files/_vecModels/glove-6B-300d.txt", 'r')
        model = {}
        print('Loading Glove Model.')
        for _, line in enumerate(file.readlines()):
            splitline = line.split()
            #self.wv.append(np.array([[splitline[0]]+[np.float(i) for i in splitline[1:]]]))
            self.wv[splitline[0]] = torch.FloatTensor([np.float(i) for i in splitline[1:]]).view(1,-1)
        #self.embedding_cols = [str(i) for i in range(dims)]
        #self.wv = pd.DataFrame(np.array(self.wv).reshape(-1,dims+1), columns=['lex']+self.embedding_cols)
        file.close()
        print('Glove Model loaded: {} lexemes complete.'.format(len(model)))

    def lex_to_cosine_similarity(self, lex, N):
        llex = self.wv[lex]
        val_names = []
        vals = []
        for k,v in self.wv.items():
            if k != lex:
                vals.append(v)
                val_names.append(k)
        vals = np.concatenate(vals, axis=0)
        delta = self.cos(llex, vals).view(-1)
        i = delta.argsort(descending=True).numpy()[:N]
        return [(val_names[j], delta[j]) for j in i]

    def embed_to_cosine_similarity(self, embed, N):

        val_names = []
        vals = []
        for k, v in self.wv.items():
            vals.append(v)
            val_names.append(k)
        vals = np.concatenate(vals, axis=0)
        delta = self.cos(embed, vals).view(-1)
        i = delta.argsort(descending=True).numpy()[:N]
        return [(val_names[j], delta[j]) for j in i]

    """def eucTopN(self, lex, N):
        dif = self.wv[self.embedding_cols].loc[self.wv['lex'].isin([lex])].values - self.wv[self.embedding_cols].values
        delta = np.sum(dif * dif, axis=1)
        i = np.argsort(delta)[1:1+N]
        return [(self.wv['lex'].loc[j], delta[j]) for j in i]"""

    def collect_item(self, lex):
        file = open("w2vMods/files/_vecModels/glove-6B-300d.txt", 'r')
        items = {}
        for _, line in enumerate(file.readlines()):
            if lex in line:
                splitline = line.split()
                # self.wv.append(np.array([[splitline[0]]+[np.float(i) for i in splitline[1:]]]))
                items[splitline[0]] = torch.FloatTensor([np.float(i) for i in splitline[1:]]).view(1, -1)
        self.wv.update(items)
        file.close()
        return items

    def collect_lexicon(self, lexemes):
        lexes = list(lexemes)+[w.lower() for w in lexemes]
        lexes = list(set(lexes))
        print('collecting items: {}'.format(len(lexes)))
        file = open("w2vMods/files/_vecModels/glove-6B-300d.txt", 'r')
        items = {}
        for _, line in enumerate(file.readlines()):
            splitline = line.split()
            if splitline[0].lower() in lexes:
                # self.wv.append(np.array([[splitline[0]]+[np.float(i) for i in splitline[1:]]]))
                items[splitline[0]] = torch.FloatTensor([np.float(i) for i in splitline[1:]]).view(1, -1)
        print('=======]{}/{} added[======='.format(len(items), len(lexes)))
        self.wv.update(items)
        file.close()

    def vocab(self):
        return self.wv.keys()

    def itms(self):
        return self.wv.items()

    def compare(self, w1, w2):
        return self.cos(self.wv[w1], self.wv[w2])

    def __getitem__(self, lex):
        if lex not in self.wv.keys():
            print('finding item in vocabulary . . .')
            self.collect_item(lex)
        return self.wv[lex]
