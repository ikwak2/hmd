#from tensorflow import keras
#from tensorflow.keras import layers

# -*- coding: utf-8 -*-
import argparse
import os
import time
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import sys
import random
from tqdm import tqdm
from tools import utils
from torch import Tensor
import torch.nn.functional as F



def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class Dense(nn.Module):
    
    def __init__(self,length):
        super(Dense, self).__init__()
        h1 = nn.Linear(length, 64)
        h2 = nn.Linear(64, 32)
        h3 = nn.Linear(32, 1)
        self.hidden = nn.Sequential(
            h1,
            nn.ReLU(),
            h2,
            nn.ReLU(),
            h3,
        )
        if torch.cuda.is_available():
            self.hidden = self.hidden.cuda()
        
    def forward(self, x):
        o = self.hidden(x)
        return o.view(-1,1)
    
    
    
class cnn_ti(nn.Module):
    def __init__(self, classes_num):
        
        super(cnn_ti, self).__init__() 
        


        self.bn0 = nn.BatchNorm2d(100)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.dense_age = Dense(6)
        self.dense_sex = Dense(2)
        self.dense_hw = Dense(2)
        self.dense_preg = Dense(1)
        self.dense_loc = Dense(5)
        
        #self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024) 
        #self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(512, 507, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
        
        
    def forward(self, input, age, sex, hw, preg, loc):

        """
        Input: (batch_size, data_length)"""

        #x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        #x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        #print(input.shape)
        #x = input.permute(1, 2, 0) 
        input = input.float()
        #x =input.unsqueeze(1)
        x = input.permute(0, 2,1, 3) 
        #print(x.shape)
        #x = x.transpose(1, 3)
        #print(x.shape)
        x = self.bn0(x)
        #print(x.shape)
        x = x.permute(0, 2,1, 3) 
        #x = x.transpose(1, 3)
        #print(x.shape)
        
        x = self.conv_block1(x, pool_size=(1, 2), pool_type='avg')
        #print(x.shape)
        x = F.dropout(x, p=0.2, training=self.training)
        #print(x.shape)
        x = self.conv_block2(x, pool_size=(1, 2), pool_type='avg')
        #print(x.shape)
        x = F.dropout(x, p=0.2, training=self.training)
        #print(x.shape)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        #print(x.shape)
        x = F.dropout(x, p=0.2, training=self.training)
        #print(x.shape)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        #print(x.shape)
        x = F.dropout(x, p=0.2, training=self.training)
        #x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        #x = F.dropout(x, p=0.2, training=self.training)
        #x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        #x = F.dropout(x, p=0.2, training=self.training)
        #print(x.shape)
        #print(x)
        x = torch.mean(x, dim=3)
        #print(x)
        #print(x.shape)
        (x1, _) = torch.max(x, dim=2)
        #print(x.shape)
        x2 = torch.mean(x, dim=2)
        #print(x.shape)
        x = x1 + x2
        #print(x.shape)
        x = F.dropout(x, p=0.5, training=self.training)
        #print(x.shape)
        x = F.relu_(self.fc1(x))
        
        age_ = self.dense_age(age)
        sex_ = self.dense_sex(sex)
        hw_ = self.dense_hw(hw)
        preg_ = self.dense_preg(preg)
        loc_ = self.dense_loc(loc)
        
        #print(x.shape)
        cat = torch.cat([x, age_,sex_,hw_,preg_,loc_],axis = 1)
        #embedding = F.dropout(x, p=0.5, training=self.training)
        out = self.fc_audioset(cat)
        #print(out.shape)
        #clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        #output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return out








def get_toy(mel_input_shape):
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
        
    ## age embeddig
    age1 = layers.Dense(2, activation = 'relu')(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = 'relu')(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = 'relu')(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = 'relu')(loc)

    ## mel embedding
    mel2 = layers.Conv2D(16, (3,3), activation = 'relu')(mel1)
    mel2 = layers.MaxPooling2D()(mel2)
    mel2 = layers.Conv2D(32, (5,5), activation = 'relu')(mel2)
    mel2 = layers.MaxPooling2D()(mel2)
    mel2 = layers.Conv2D(32, (3,3), activation = 'relu')(mel2)
    mel2 = layers.MaxPooling2D()(mel2)
    mel2 = layers.Conv2D(64, (3,3), activation = 'relu')(mel2)
    mel2 = layers.MaxPooling2D()(mel2)
    mel2 = layers.GlobalAveragePooling2D()(mel2)

    concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, mel2, preg])
    concat1 = layers.Dense(10, activation = 'relu')(concat1)
    concat1 = layers.Dense(3, activation = "softmax")(concat1)
    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1] , outputs = concat1 )
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return(model)