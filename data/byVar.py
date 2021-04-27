import pandas as pd
import numpy as np

import glob, os, sys

df = glob.glob(os.path.join('CommClusters/data/corpora/byVar/', '*.csv'))
df = {i[32:-4]:i for i in df}
d = []
for k,v in df.items():
    dd = pd.read_csv(v)
    dd['var'] = k
    d.append(dd)
df = pd.concat(d, ignore_index=True)