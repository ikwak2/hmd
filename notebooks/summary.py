import numpy as np

import os
import pandas as pd
import pickle

reslist = ['res/'+fnm for fnm in os.listdir('res/')]

data = []
allkeys = []
for i in range(len(reslist)) :
    with open(reslist[i], 'rb') as f:
        data1 = pickle.load(f)
    data.append(data1)
    allkeys += data1.keys()

allkeys = list(set(allkeys))

newlist = {}
for i in range(len(data)) :
    if i == 0 :
        for k1 in allkeys :
            try :
                newlist[k1] = [ data[i][k1] ]
            except :
                newlist[k1] = [-999]
    else :
        for k1 in allkeys :
            try :
                newlist[k1].append(data[i][k1])
            except :
                newlist[k1].append(-999)

df1 = pd.DataFrame(newlist)

df1[df1.use_mel == True][df1.samp_sec==20][['min_cost', 'min_th', 'max_wc','max_th','mm_weighted_accuracy', 'out_cost']]

df1[df1.use_mel == True][df1.samp_sec==30][['min_cost', 'min_th', 'max_wc','max_th','mm_weighted_accuracy', 'out_cost']]

df1[df1.use_mel == True][df1.samp_sec==40][['min_cost', 'min_th', 'max_wc','max_th','mm_weighted_accuracy', 'out_cost']]

012
512 256  .78
9-11
1024 512   .71

12-14
128 256  .75

15-17
256 512  .72


stft
512 256   .71
1024 512   .68
2048 1024  .58
