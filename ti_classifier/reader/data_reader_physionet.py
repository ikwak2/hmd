import numpy as np
import torch
import sys
import os
sys.path.append("..")
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as AT

from scipy.io import wavfile
import random


class myDataset(Dataset):
    def __init__(self, x_npy, y_npy):    # spec or mel_spec
        self.x_npy = x_npy
        self.y_npy = y_npy

        
    def __getitem__(self, idx):
        cur_idx = idx
        while 1:
            try:
                data_feature = self.x_npy['mel1'][cur_idx]
                age = self.x_npy['age'][cur_idx]
                sex = self.x_npy['sex'][cur_idx]
                hw = self.x_npy['hw'][cur_idx]
                if self.x_npy['preg'][cur_idx] :
                    preg = np.array([1])
                else:
                    preg = np.array([0])
                
                loc = self.x_npy['loc'][cur_idx]
                break
            except:
                cur_idx += 1 
        
        data_label = self.y_npy[cur_idx]
        return data_feature,age,sex,hw,preg,loc, data_label

    def __len__(self):
        return self.x_npy['mel1'].shape[0]

def myCollateFn(sample_batch):
    sample_batch = sorted(sample_batch, key=lambda x: x[0].shape[0], reverse=True)
    data_feature = [torch.from_numpy(x[0]) for x in sample_batch]
    age = [torch.from_numpy(x[1]) for x in sample_batch]
    sex = [torch.from_numpy(x[2]) for x in sample_batch]
    hw = [torch.from_numpy(x[3]) for x in sample_batch]
    preg = [torch.from_numpy(x[4]) for x in sample_batch]
    loc = [torch.from_numpy(x[5]) for x in sample_batch]
    
    data_feature = pad_sequence(data_feature, batch_first=False, padding_value=0.0)
    age = pad_sequence(age, batch_first=False, padding_value=0.0)
    sex = pad_sequence(sex, batch_first=False, padding_value=0.0)
    hw = pad_sequence(hw, batch_first=False, padding_value=0.0)
    preg = pad_sequence(preg, batch_first=False, padding_value=0.0)
    loc = pad_sequence(loc, batch_first=False, padding_value=0.0)
    
    data_label = torch.tensor([x[-1] for x in sample_batch]).unsqueeze(-1)
    return data_feature, age, sex, hw, preg, loc, data_label

class myDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(myDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = myCollateFn
