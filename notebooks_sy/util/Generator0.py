import tensorflow.keras
from tensorflow.keras import backend as K
from scipy.io import wavfile
import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, maximum, add, SeparableConv2D
from sklearn.metrics import roc_curve
import os
import sys
import pickle
import soundfile
import pandas as pd
import numpy as np
import tensorflow


class Generator0():
    def __init__(self, X_train, y_train, batch_size=32, beta_param=0.2, mixup = True, lowpass = False, highpass = False, ranfilter2 = False, shuffle=True, datagen=None, chaug = False, cout = False):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = beta_param
        self.mixup = mixup
        self.shuffle = shuffle
        self.sample_num = len(y_train)
        self.datagen = datagen

        ## ffm 
        
        self.lowpass = lowpass
        self.highpass = highpass
        self.ranfilter = ranfilter2
        self.chaug = chaug
        self.cutout = cout        


    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        
        
        def get_box(lambda_value, nf, nt):
            cut_rat = np.sqrt(1.0 - lambda_value)

            cut_w = int(nf * cut_rat)  # rw
            cut_h = int(nt * cut_rat)  # rh

            cut_x = int(np.random.uniform(low=0, high=nf))  # rx
            cut_y = int(np.random.uniform(low=0, high=nt))  # ry

            boundaryx1 = np.minimum(np.maximum(cut_x - cut_w // 2, 0), nf) #tf.clip_by_value(cut_x - cut_w // 2, 0, IMG_SIZE_x)
            boundaryy1 = np.minimum(np.maximum(cut_y - cut_h // 2, 0), nt) #tf.clip_by_value(cut_y - cut_h // 2, 0, IMG_SIZE_y)
            bbx2 = np.minimum(np.maximum(cut_x + cut_w // 2, 0), nf) #tf.clip_by_value(cut_x + cut_w // 2, 0, IMG_SIZE_x)
            bby2 = np.minimum(np.maximum(cut_y + cut_h // 2, 0), nt) #tf.clip_by_value(cut_y + cut_h // 2, 0, IMG_SIZE_y)

            target_h = bby2 - boundaryy1
            if target_h == 0:
                target_h += 1

            target_w = bbx2 - boundaryx1
            if target_w == 0:
                target_w += 1

            return boundaryx1, boundaryy1, target_h, target_w           
        
        
        if isinstance(self.X_train, list):
            X = []
            for X_temp in self.X_train:
                if len(X_temp.shape) == 4: 
                    _, h, w, c = X_temp.shape
                    l = np.random.beta(self.alpha, self.alpha, self.batch_size)
                    X_l = l.reshape(self.batch_size, 1, 1, 1)
                    y_l = l.reshape(self.batch_size, 1)
                elif len(X_temp.shape) == 3:
                    _, h, w = X_temp.shape
                    l = np.random.beta(self.alpha, self.alpha, self.batch_size)
                    X_l = l.reshape(self.batch_size, 1, 1)
                    y_l = l.reshape(self.batch_size, 1)
                elif len(X_temp.shape) == 2:
                    _, h = X_temp.shape
                    l = np.random.beta(self.alpha, self.alpha, self.batch_size)
                    X_l = l.reshape(self.batch_size, 1)
                    y_l = l.reshape(self.batch_size, 1)
                elif len(X_temp.shape) == 1:
                    _= X_temp.shape
                    l = np.random.beta(self.alpha, self.alpha, self.batch_size)
                    X_l = l.reshape(self.batch_size,)
                    y_l = l.reshape(self.batch_size, 1)
                
                X1 = X_temp[batch_ids[:self.batch_size]].copy()
                X2 = X_temp[batch_ids[self.batch_size:]].copy()
                
                if self.mixup :
                    Xn = X1 * X_l + X2 * (1 - X_l)
                else :
                    Xn = X1
                if len(X_temp.shape) == 4: 
                    _, h, w, c = X_temp.shape
                    if h != 1 :
                        if self.lowpass :
                            uv, lp = self.lowpass
                            dec1 = np.random.choice(2, size = self.batch_size)
                            for i in range(self.batch_size) :
                                loc1 = np.random.choice(lp, size = 1)[0]
                                Xn[i,:loc1,:,:] = 0
                        if self.highpass :
                            uv, hp = self.highpass
                            dec1 = np.random.choice(2, size = self.batch_size)
                            for i in range(self.batch_size) :
                                loc1 = np.random.choice(hp, size = 1)[0]
                                Xn[i,loc1:,:,:] = 0
                        if self.ranfilter :                
                            raniter, ranf = self.ranfilter
                            dec1 = np.random.choice(raniter, size = self.batch_size)
                            for i in range(self.batch_size) :
                                if dec1[i] > 0 :
                                    for j in range(dec1[i]) :
                                        b1 = np.random.choice(ranf, size = 1)[0]
                                        loc1 = np.random.choice(h - b1, size = 1)[0]
                                        Xn[i, loc1:(loc1 + b1 - 1), :] = 0
                        if self.chaug :
                            for i in range(self.batch_size) :
                                noiselv = np.random.uniform(low= - self.chaug, high= self.chaug)
                                Xn[i,:] += noiselv
                        if self.cutout :
                            lambda1 = np.random.beta(self.cutout, self.cutout, size = self.batch_size)   ## beta_param default : 0.7  STC페이퍼 추천은 0.6~0.8
                            for i in range(self.batch_size) :
                                boundaryx1, boundaryy1, target_h, target_w = get_box(lambda1[i], h, w)
                                Xn[i, boundaryx1:(boundaryx1+target_h), boundaryy1:(boundaryy1+target_w),: ] = 0
                
#                 if len(X_temp.shape) == 3: 
                    
#                     _, h, w = X_temp.shape
                    
#                     if h != 1 :
                        
#                         if self.lowpass :
#                             uv, lp = self.lowpass
#                             dec1 = np.random.choice(2, size = self.batch_size)
#                             for i in range(self.batch_size) :
#                                 loc1 = np.random.choice(lp, size = 1)[0]
#                                 Xn[i,:loc1,:] = 0
#                         if self.highpass :
#                             uv, hp = self.highpass
#                             dec1 = np.random.choice(2, size = self.batch_size)
#                             for i in range(self.batch_size) :
#                                 loc1 = np.random.choice(hp, size = 1)[0]
#                                 Xn[i,loc1:,:] = 0
#                         if self.ranfilter :                
#                             raniter, ranf = self.ranfilter
#                             dec1 = np.random.choice(raniter, size = self.batch_size)
#                             for i in range(self.batch_size) :
#                                 if dec1[i] > 0 :
#                                     for j in range(dec1[i]) :
#                                         b1 = np.random.choice(ranf, size = 1)[0]
#                                         loc1 = np.random.choice(h - b1, size = 1)[0]
#                                         Xn[i, loc1:(loc1 + b1 - 1), :] = 0                    
                X.append(Xn)
        else:
            if len(self.X_train.shape) == 4: 
                _, h, w, c = self.X_train.shape
                l = np.random.beta(self.alpha, self.alpha, self.batch_size)
                X_l = l.reshape(self.batch_size, 1, 1, 1)
                y_l = l.reshape(self.batch_size, 1)
            elif len(self.X_train.shape) == 3:
                _, h, w = self.X_train.shape
                l = np.random.beta(self.alpha, self.alpha, self.batch_size)
                X_l = l.reshape(self.batch_size, 1, 1)
                y_l = l.reshape(self.batch_size, 1)
            elif len(self.X_train.shape) == 2:
                _, h = self.X_train.shape
                l = np.random.beta(self.alpha, self.alpha, self.batch_size)
                X_l = l.reshape(self.batch_size, 1)
                y_l = l.reshape(self.batch_size, 1)
            elif len(self.X_train.shape) == 1:
                _= self.X_train.shape
                l = np.random.beta(self.alpha, self.alpha, self.batch_size)
                X_l = l.reshape(self.batch_size,)
                y_l = l.reshape(self.batch_size, 1)

            X1 = self.X_train[batch_ids[:self.batch_size]].copy()
            X2 = self.X_train[batch_ids[self.batch_size:]].copy()
            if self.mixup :
                Xn = X1 * X_l + X2 * (1 - X_l)
            else :
                Xn = X1

            if len(self.X_train.shape) == 4: 
                _, h, w, c = X_temp.shape
                if self.lowpass :
                    uv, lp = self.lowpass
                    dec1 = np.random.choice(2, size = self.batch_size)
                    for i in range(self.batch_size) :
                        loc1 = np.random.choice(lp, size = 1)[0]
                        Xn[i,:loc1,:,:] = 0
                if self.highpass :
                    uv, hp = self.highpass
                    dec1 = np.random.choice(2, size = self.batch_size)
                    for i in range(self.batch_size) :
                        loc1 = np.random.choice(hp, size = 1)[0]
                        Xn[i,loc1:,:,:] = 0
                if self.ranfilter :                
                    raniter, ranf = self.ranfilter
                    dec1 = np.random.choice(raniter, size = self.batch_size)
                    for i in range(self.batch_size) :
                        if dec1[i] > 0 :
                            for j in range(dec1[i]) :
                                b1 = np.random.choice(ranf, size = 1)[0]
                                loc1 = np.random.choice(h - b1, size = 1)[0]
                                Xn[i, loc1:(loc1 + b1 - 1), :] = 0
                
#                 if len(self.X_train.shape) == 3:
#                     _, h, w = X_temp.shape
                    
#                     if h != 1 :
#                         if self.lowpass :
#                             uv, lp = self.lowpass
#                             dec1 = np.random.choice(2, size = self.batch_size)
#                             for i in range(self.batch_size) :
#                                 loc1 = np.random.choice(lp, size = 1)[0]
#                                 Xn[i,:loc1,:] = 0
#                         if self.highpass :
#                             uv, hp = self.highpass
#                             dec1 = np.random.choice(2, size = self.batch_size)
#                             for i in range(self.batch_size) :
#                                 loc1 = np.random.choice(hp, size = 1)[0]
#                                 Xn[i,loc1:,:] = 0
#                         if self.ranfilter :                
#                             raniter, ranf = self.ranfilter
#                             dec1 = np.random.choice(raniter, size = self.batch_size)
#                             for i in range(self.batch_size) :
#                                 if dec1[i] > 0 :
#                                     for j in range(dec1[i]) :
#                                         b1 = np.random.choice(ranf, size = 1)[0]
#                                         loc1 = np.random.choice(h - b1, size = 1)[0]
#                                         Xn[i, loc1:(loc1 + b1 - 1), :] = 0
                
            X.append(Xn)

                
        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]].copy()
                y2 = y_train_[batch_ids[self.batch_size:]].copy()
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]].copy()
            y2 = self.y_train[batch_ids[self.batch_size:]].copy()
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y
    
    
