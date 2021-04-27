import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_data(x,labels,pal=None):
    dfi = pd.DataFrame(x,columns=labels)
    dfi['L'] = labels
    dfi = dfi.set_index('L')
    if pal:
        sns.clustermap(dfi,cmap=pal)
    else:
        sns.clustermap(dfi)
    plt.show()

def sel(terms,IDX):
    return (IDX == np.array(terms).reshape(-1,1)).sum(axis=0).astype(np.bool)