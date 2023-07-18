
from scipy.signal import hilbert
import os
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
import librosa
import librosa.display
import math
import sys
sys.path.insert(0,'/Data/hmd/hmd_sy/evaluation-2022')
sys.path.insert(0,'/Data/hmd/hmd_sy/notebooks')
sys.path.insert(0,'utils')
from helper_code import *
from get_feature import *
from models import *
from Generator0 import *

import datetime
from evaluate_model import *
from scipy import special
import scipy.io as sio

import tensorflow as tf
from keras.preprocessing import sequence
from keras_preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler,ModelCheckpoint

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[4], 'GPU')
    except RuntimeError as e:
        print(e)

# data_folder =  '/Data/hmd/physionet.org/files/circor-heart-sound/1.0.3/training_data'
train_folder =  '/Data/hmd/data_split/murmur/train/'
test_folder = '/Data/hmd/data_split/murmur/test/'

import typing
import warnings

import tensorflow as tf



############################
## filtering (s1, s2 detect)
############################
import numpy as np
from scipy.signal import butter, filtfilt 
import matplotlib.pyplot as plt 

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

############################
## feature_extract_bound_melspec
############################

def feature_extract_bound_melspec(data, samp_sec=20, sr = 4000, pre_emphasis = 0, hop_length=256, win_length = 512, n_mels = 100):
    
    if samp_sec:
        if len(data) > sample_rate * samp_sec :
            n_samp = len(data) // int(sample_rate * samp_sec)
            signal = []
            for i in range(n_samp) :
                signal.append(data[ int(sample_rate * samp_sec)*i:(int(sample_rate * samp_sec)*(i+1))])
        else :
            n_samp = 1
            signal = np.zeros(int(sample_rate*samp_sec,))
            for i in range(int(sample_rate * samp_sec) // len(data)) :
                signal[(i)*len(data):(i+1)*len(data)] = data
            num_last = int(sample_rate * samp_sec) - len(data)*(i+1)
            signal[(i+1)*len(data):int(sample_rate * samp_sec)] = data[:num_last]
            signal = [signal]
    else:
        n_samp = 1
        signal = [data]

    Sig = []
    for i in range(n_samp) :
        if pre_emphasis :
            emphasized_signal = np.append(signal[i][0], signal[i][1:] - pre_emphasis * signal[i][:-1])
        else :
            emphasized_signal = signal[i]

        Sig.append(librosa.power_to_db(librosa.feature.melspectrogram(y=emphasized_signal, sr= sr, n_mels=n_mels, n_fft=win_length, hop_length=hop_length, win_length=win_length)))

    return Sig



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


def get_LCNN_o_1_dr_rr(mel_input_shape, cqt_input_shape, stft_input_shape, 
                       mel_s1s2_input_shape, mel_mm_input_shape, use_s1s2 = True,use_mm = True,
                       use_mel = True, use_cqt = True, use_stft = True, 
                       ord1 = True, dp = .5, fc = False, ext = False, ext2 = False):
    # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
    rr1 = keras.Input(shape=(1,), name = 'rr')
    # qrs1 = keras.Input(shape=(1,), name = 'qrs')
    
    mel1_s1s2 = keras.Input(shape=(mel_s1s2_input_shape), name = 'mel_s1s2')
    mel1_mm = keras.Input(shape=(mel_mm_input_shape), name = 'mel_mm')
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)


    ## mel embedding
    if use_mel :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)


        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
#         batch6 = layers.LeakyReLU()(batch6)



        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)

        # u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(max3)
        # u = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(u)
        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)
        
        # u = Conv2D(filters=64, kernel_size=1,  padding='same', activation=None)(batch10)
        # u = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(u)

        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)
#         batch22 = layers.LeakyReLU()(batch22)



        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        
        # u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(batch19)
        # u = MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')(u)

        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2 = layers.GlobalAveragePooling2D()(mha)
        
   # mel1_s1s2
    if use_s1s2 :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_s1s2)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_s1s2)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)


        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)


        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)

        # u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(max3)
        # u = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(u)
        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)
        
        # u = Conv2D(filters=64, kernel_size=1,  padding='same', activation=None)(batch10)
        # u = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(u)

        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)



        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        
        # u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(batch19)
        # u = MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')(u)

        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2_s1s2 = layers.GlobalAveragePooling2D()(mha)
        
       
   # mel1_s1s2
    if use_mm :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_mm)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_mm)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)


        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)


        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)

        # u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(max3)
        # u = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(u)
        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)
        
        # u = Conv2D(filters=64, kernel_size=1,  padding='same', activation=None)(batch10)
        # u = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(u)

        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)



        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        
        # u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(batch19)
        # u = MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')(u)

        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2_mm = layers.GlobalAveragePooling2D()(mha)

    if use_cqt :
        ## cqt embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        
        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            stft2 = Dropout(dp)(stft2)
    
    
    if use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2, stft2])
    if not use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([cqt2, stft2])
    if use_mel and not use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, stft2])
    if use_mel and use_cqt and not use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2])
    if not use_mel and not use_cqt and use_stft :  ## stft 만
        concat2 = stft2
    if use_mel and not use_cqt and not use_stft :  ### mel만
        concat2 = mel2
    if not use_mel and use_cqt and not use_stft :  ### cqt만
        concat2 = cqt2

    if ext :
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg, rr1, mel2_s1s2, mel2_mm])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])

    if ext2 :
        concat1 = layers.Concatenate()([rr1,mel2_s1s2, mel2_mm])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])
        
    if fc :
        concat2 = layers.Dense(10, activation = "relu")(concat2)
        if dp :
            concat2 = Dropout(dp)(concat2)
        
    if ord1 :
        res1 = layers.Dense(2, activation = "softmax")(concat2)
    else :
        res1 = layers.Dense(3, activation = "softmax")(concat2)

        
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1, rr1, mel1_s1s2, mel1_mm] , outputs = res1 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)


def get_LCNN_2_dr_rr(mel_input_shape, cqt_input_shape, stft_input_shape, 
                    mel_s1s2_input_shape, mel_mm_input_shape, use_s1s2 = True,use_mm = True,
                     use_mel = True, use_cqt = True, use_stft = True, 
                    dp = False, fc = False, ext = False, ext2 = False):
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
    rr1 = keras.Input(shape=(1,), name = 'rr')
    # qrs1 = keras.Input(shape=(1,), name = 'qrs')
    
    mel1_s1s2 = keras.Input(shape=(mel_s1s2_input_shape), name = 'mel_s1s2')
    mel1_mm = keras.Input(shape=(mel_mm_input_shape), name = 'mel_mm')
        
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)

   

   ## mel embedding
    if use_mel :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        
        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        
        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        mel2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            mel2 = Dropout(dp)(mel2)
        
    if use_s1s2:
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_s1s2)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_s1s2)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        
        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        
        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        mel2_s1s2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            mel2_s1s2 = Dropout(dp)(mel2_s1s2)
            
    if use_mm:
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_mm)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_mm)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        
        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        
        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        mel2_mm = layers.GlobalAveragePooling2D()(max28)
        if dp :
            mel2_mm = Dropout(dp)(mel2_mm)

    if use_cqt :
        ## cqt embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)
        
        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            stft2 = Dropout(dp)(stft2)
    
#    concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg])
#    d1 = layers.Dense(2, activation = 'relu')(concat1)
    
    if use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2, stft2])
    if not use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([cqt2, stft2])
    if use_mel and not use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, stft2])
    if use_mel and use_cqt and not use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2])
    if not use_mel and not use_cqt and use_stft :  ## stft 만
        concat2 = stft2
    if use_mel and not use_cqt and not use_stft :  ### mel만
        concat2 = mel2
    if not use_mel and use_cqt and not use_stft :  ### cqt만
        concat2 = cqt2

    if ext :
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg, rr1,mel2_s1s2, mel2_mm])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])

    if ext2 :
        concat1 = layers.Concatenate()([rr1,mel2_s1s2, mel2_mm])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])
        
    if fc :
        concat2 = layers.Dense(10, activation = 'relu')(concat2)
        if dp :
            concat2 = Dropout(dp)(concat2)

    res1 = layers.Dense(3, activation = "softmax")(concat2)
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1, rr1, mel1_s1s2, mel1_mm] , outputs = res2 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)


def sigmoidal_decay(e, start=0, end=100, lr_start=1e-3, lr_end=1e-5):
    if e < start:
        return lr_start
    elif e > end:
        return lr_end

    middle = (start + end) / 2
    s = lambda x: 1 / (1 + np.exp(-x))

    return s(13 * (-e + middle) / np.abs(end - start)) * np.abs(lr_start - lr_end) + lr_end

patient_files_trn = find_patient_files(train_folder)
patient_files_test = find_patient_files(test_folder)

# Run model.
def run_model(model_folder, data_folder, output_folder, allow_failures, verbose):
    # Load models.
    if verbose >= 1:
        print('Loading Challenge model...')

    model = load_challenge_model(model_folder, verbose) ### Teams: Implement this function!!!

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Create a folder for the Challenge outputs if it does not already exist.
    os.makedirs(output_folder, exist_ok=True)

    # Run the team's model on the Challenge data.
    if verbose >= 1:
        print('Running model on Challenge data...')

#    @tf.function
    # Iterate over the patient files.
    for i in range(num_patient_files):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patient_files))

        patient_data = load_patient_data(patient_files[i])
        recordings = load_recordings(data_folder, patient_data)

        # Allow or disallow the model to fail on parts of the data; helpful for debugging.
        try:
            classes, labels, probabilities = run_challenge_model(model, patient_data, recordings, verbose) ### Teams: Implement this function!!!
        except:
            if allow_failures:
                if verbose >= 2:
                    print('... failed.')
                classes, labels, probabilities = list(), list(), list()
            else:
                raise

        # Save Challenge outputs.
        head, tail = os.path.split(patient_files[i])
        root, extension = os.path.splitext(tail)
        output_file = os.path.join(output_folder, root + '.csv')
        patient_id = get_patient_id(patient_data)
        save_challenge_outputs(output_file, patient_id, classes, labels, probabilities)

    if verbose >= 1:
        print('Done.')
        
import pickle as pk
def save_challenge_model(model_folder, model1, model2, m_name1, m_name2, param_feature) :
    os.makedirs(model_folder, exist_ok=True)
    info_fnm = os.path.join(model_folder, 'desc.pk')
    filename1 = os.path.join(model_folder, m_name1 + '_model1.hdf5')
    filename2 = os.path.join(model_folder, m_name2 + '_model2.hdf5')
    model1.save(filename1)
    model2.save(filename2)
    param_feature['model1'] = m_name1
    param_feature['model2'] = m_name2
    param_feature['model_fnm1'] = filename1
    param_feature['model_fnm2'] = filename2
    with open(info_fnm, 'wb') as f:
        pk.dump(param_feature, f, pk.HIGHEST_PROTOCOL)
    return 1

def load_challenge_model(model_folder, verbose):
    info_fnm = os.path.join(model_folder, 'desc.pk')
    with open(info_fnm, 'rb') as f:
        info_m = pk.load(f)
#    if info_m['model'] == 'toy' :
#        model = get_toy(info_m['mel_shape'])
#    filename = os.path.join(model_folder, info_m['model'] + '_model.hdf5')
#    model.load_weights(filename)
    return info_m



# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):

    murmur_classes = ['Present', 'Unknown', 'Absent']
    outcome_classes = ['Abnormal', 'Normal']

    
    if model['model1'] == 'lcnn1_dr_rr' :
        model1 = get_LCNN_o_1_dr_rr(model['mel_shape'],model['cqt_shape'],model['stft_shape'], 
                                    model['s1s2_shape'],model['mm_shape'],model['use_s1s2'], model['use_mm'],
                                    use_mel = model['use_mel'],use_cqt = model['use_cqt'], use_stft = model['use_stft'],
                                    ord1 = model['ord1'], 
                                    dp = model['dp'], fc = model['fc'], ext = False, ext2 = True)
    if model['model2'] == 'lcnn2_dr_rr' :
        model2 = get_LCNN_2_dr_rr(model['mel_shape'],model['cqt_shape'],model['stft_shape'], 
                                  model['s1s2_shape'],model['mm_shape'],model['use_s1s2'], model['use_mm'],
                                  use_mel = model['use_mel'],use_cqt = model['use_cqt'], use_stft = model['use_stft'], 
                                  dp = model['dp'], fc = model['fc'], ext = True, ext2 = False)
    model1.load_weights(model['model_fnm1'])
    model2.load_weights(model['model_fnm2'])

#    classes = model['classes']
    # Load features.
    features = get_feature_one(data, verbose = 0)

    samp_sec = model['samp_sec']
    pre_emphasis = model['pre_emphasis']
    hop_length = model['hop_length']
    win_length = model['win_length']
    n_mels = model['n_mels']
    filter_scale = model['filter_scale']
    n_bins = model['n_bins']
    fmin = model['fmin']
    use_mel = model['use_mel']
    use_cqt = model['use_cqt']
    use_stft = model['use_stft']
    use_raw = model['use_raw']
    trim = model['trim']
    use_rr = model['use_rr']
    use_mm = model['use_mm']
    use_s1s2 = model['use_s1s2']
    use_b_detect = True
    

    features['mel1'] = []
    for i in range(len(recordings)) :
        if use_mel :
            mel1 = feature_extract_melspec(recordings[i]/ 32768, samp_sec=samp_sec, pre_emphasis = pre_emphasis, hop_length=hop_length,
                                           win_length = win_length, n_mels = n_mels, trim = trim)[0]
        else :
            mel1 = np.zeros( (1,1) )
        features['mel1'].append(mel1)
    M, N = features['mel1'][0].shape

    if use_mel :
        for i in range(len(features['mel1'])) :
            features['mel1'][i] = features['mel1'][i].reshape(M,N,1)
    features['mel1'] = np.array(features['mel1'])

    features['cqt1'] = []
    for i in range(len(recordings)) :
        if use_cqt :
            mel1 = feature_extract_cqt(recordings[i], samp_sec=samp_sec, pre_emphasis = pre_emphasis, filter_scale = filter_scale,
                                        n_bins = n_bins, fmin = fmin, trim = trim)[0]
        else :
            mel1 = np.zeros( (1,1))
        features['cqt1'].append(mel1)
    M, N = features['cqt1'][0].shape
    if use_cqt :
        for i in range(len(features['cqt1'])) :
            features['cqt1'][i] = features['cqt1'][i].reshape(M,N,1)
    features['cqt1'] = np.array(features['cqt1'])

    features['stft1'] = []
    for i in range(len(recordings)) :
        if use_stft :
            mel1 = feature_extract_stft(recordings[i]/ 32768, samp_sec=samp_sec, pre_emphasis = pre_emphasis, hop_length=hop_length,
                                        win_length = win_length, trim = trim)[0]
        else :
            mel1 = np.zeros( (1,1) )
        features['stft1'].append(mel1)
    M, N = features['stft1'][0].shape
    if use_stft :
        for i in range(len(features['stft1'])) :
            features['stft1'][i] = features['stft1'][i].reshape(M,N,1)
    features['stft1'] = np.array(features['stft1'])

    features['raw1'] = []
    for i in range(len(recordings)) :
        if use_raw:
            recording1 = recordings[i]
            if len(recording1) >= maxlen : 
                recording1 = recording1[:maxlen]
            else :
                recording1 = np.pad(recording1, (0, maxlen - len(recording1) ), constant_values=(0,0) )
        else :
            recording1 = np.zeros((1))
        features['raw1'].append(recording1)
    features['raw1'] = np.array(features['raw1'])
    
    
    features['rr1'] = []
    for i in range(len(recordings)) :
        if use_rr :
            try:
                recording1 = recordings[i]
                ____, info = nk.ecg_process(recording1, sampling_rate=4000)
                current_rr = np.mean(np.diff(info['ECG_R_Peaks'])/4000)
            except:
#                print(filename)
                current_rr= 0.6414
        else :
            current_rr = 0
        features['rr1'].append(current_rr)
    features['rr1'] = np.array(features['rr1'])
    
    
    features['s1s2_detect1'] = []
    features['mm_detect1'] = []
    features['s1s2_mel'] = []
    features['mm_mel'] = []
    for i in range(len(recordings)) :
        if use_b_detect :
            recording1 = recordings[i]
         # 1. Amplitude normalization
            normal_sig = recording1/np.max(np.abs(recording1))

            # 2. Filtering
            T = len(recording1)/4000 #time interval; Sample Period ;
            fs = 4000 #sample rate 
            cutoff = 150 #sample frequency 

            nyq = 0.5 * fs 
            order = 2  # sin wave can be approx represented as quadratic
            n = int(T*fs) #total number of samples 

            low_ft = butter_lowpass_filter(normal_sig, cutoff, fs, order)

            # 3. smoothing of signal envelope
            duration = 1.0
#                 fs = 4000.0
            samples = int(fs*duration)
            t = np.arange(len(low_ft)) / fs

            analytic_signal = hilbert(low_ft)
            amplitude_envelope = np.abs(analytic_signal)

            # threshld selection
            mu = np.sum(amplitude_envelope)/len(amplitude_envelope)
            var = np.sum((amplitude_envelope-mu)**2)/len(amplitude_envelope)
            t_sh = mu + var +0.05

            thres_list = np.argwhere(amplitude_envelope > t_sh)
            save = []
            for i in thres_list:
                j = i[0]
                save.append(j)

            packet = []
            tmp = []
            v = save.pop(0)
            tmp.append(v)

            while(len(save)>0):
                vv = save.pop(0)
                if v+1 == vv:
                    tmp.append(vv)
                    v = vv
                else:
                    packet.append(tmp)
                    tmp = []
                    tmp.append(vv)
                    v = vv

            packet.append(tmp)
#                 # thresh hold 보다 크지만 연속값이 3개 미만인 리스트 찾아서 삭제
#                 packet2 = []
#                 for i in range(len(packet)):
#                     if len(packet[i]) < 3:
#                         j = packet.index(packet[i])
#                         packet2.append(j)

#                 for i in packet2:
#                     del packet[i]

            min_list = []
            max_list = []
            for i in range(len(packet)):
                min_find = min(packet[i])
                max_find = max(packet[i])
                min_list.append(min_find)
                max_list.append(max_find)

            # s1, s2 boundary가 여러개 그려짐. 추가적인 전처리 필요. 60개보다 커야함
            packet_cp = packet.copy()
            new_list = []
            first_list = []
            last_list = []
            for i in range(len(max_list)-1):
                j = i + 1
                # 연속적인 값인 경우 삭제
                if abs(max_list[i]-min_list[j]) < 60:
                    first = max_list.index(max_list[i])
                    last = min_list.index(min_list[j])
                    re_join = packet_cp[first]+packet_cp[last]
                    new_list.append(re_join)
                    first_list.append(first)
                    last_list.append(last)

            # 연속적인 값 인덱스 찾아서 final_list 만들기
            final_list = first_list+last_list
            final_list.sort()
            set(final_list)

            # 위에서 찾은 final_linst에 들어있는 인덱스 위치는 0으로 처리
            drop_list = packet_cp.copy()
            for i in final_list:
                drop_list[i] = 0
#             print(len(drop_list))

            seq_remake = drop_list+new_list
#             print(len(seq_remake))



            # 0으로 전처리한 값 삭제
            remove_set = [0]

            li = [i for i in seq_remake if i not in remove_set]
#             print(li)

            # 추가 전처리 후 다시 min, max 출력
            min_list1 = []
            max_list1 = []
            for i in range(len(li)):
                min_find1 = min(li[i])
                max_find1 = max(li[i])
                min_list1.append(min_find1)
                max_list1.append(max_find1)

            min_list1.sort()
            max_list1.sort()


            # boundary detected s1 and s2
            s_detect = amplitude_envelope.copy()
            for i in range(len(max_list1)-1):
                j = i+1
                s_detect[max_list1[i]:min_list1[j]] = 0
            s1s2_detect = s_detect.reshape(1, s_detect.shape[0])
            s1s2_detect = s1s2_detect.tolist()
            s1s2_detect = pad_sequences(s1s2_detect, maxlen = 80000, dtype ='float64', padding = 'post', truncating ='post', value=0.0)

            # s1s2_detect 변수에 mel 적용
            s1s2_mel = feature_extract_bound_melspec(s_detect)[0]

            # boundary detected systolic and diastolic murmurs present in pcg signal
            mm_detect = amplitude_envelope.copy()
            for i in range(len(max_list1)):
                mm_detect[min_list1[i]:max_list1[i]+1] = 0
            murmur_detect = mm_detect.reshape(1, mm_detect.shape[0])
            murmur_detect = murmur_detect.tolist()
            murmur_detect = pad_sequences(murmur_detect, maxlen = 80000, dtype ='float64', padding = 'post', truncating ='post', value=0.0)

            # murmur_dect 변수에 mel 적용
            mm_mel = feature_extract_bound_melspec(mm_detect)[0]

        else :
            s1s2_detect = np.zeros((1,1))
            murmur_detect = np.zeros((1,1))
            s1s2_mel = np.zeros( (1,1,1) )
            mm_mel = np.zeros( (1,1,1) )

        features['s1s2_detect1'].append(s1s2_detect)
        features['mm_detect1'].append(murmur_detect)
        features['s1s2_mel'].append(s1s2_mel)
        features['mm_mel'].append(mm_mel)

    features['s1s2_detect1'] = np.array(features['s1s2_detect1'])
    features['mm_detect1'] = np.array(features['mm_detect1'])
    
    
    
    M, N = features['s1s2_mel'][0].shape
    if use_s1s2:
        for i in range(len(features['s1s2_mel'])):
            features['s1s2_mel'][i] = features['s1s2_mel'][i].reshape(M,N,1)
    features['s1s2_mel'] = np.array(features['s1s2_mel'])
    
    
    M, N = features['mm_mel'][0].shape
    if use_mm:
        for i in range(len(features['mm_mel'])):
            features['mm_mel'][i] = features['mm_mel'][i].reshape(M,N,1)
    features['mm_mel'] = np.array(features['mm_mel'])


    # Impute missing data.
    res1 = model1.predict([features['age'], features['sex'], features['hw'], features['preg'], features['loc'], 
                           features['mel1'],features['cqt1'], features['stft1'], features['rr1'], 
                           features['s1s2_mel'], features['mm_mel']])
    res2 = model2.predict([features['age'], features['sex'], features['hw'], features['preg'], features['loc'], 
                           features['mel1'], features['cqt1'], features['stft1'], features['rr1'], 
                           features['s1s2_mel'], features['mm_mel']])

    # Get classifier probabilities.
    if model['ord1'] :
        idx1 = res1.argmax(axis=0)[0]
        murmur_p = res1[idx1,]  ## mumur 확률 최대화 되는 애 뽑기
        murmur_probabilities = np.zeros((3,))
        murmur_probabilities[0] = murmur_p[0]
        murmur_probabilities[1] = 0
        murmur_probabilities[2] = murmur_p[1]
        outcome_probabilities = res2.mean(axis = 0) ##  outcome 은 그냥 평균으로 뽑기
    else :
        if model['mm_mean'] :
            murmur_probabilities = res1.mean(axis = 0)
        else :
            idx1 = res1.argmax(axis=0)[0]
            murmur_probabilities = res1[idx1,]  ## mumur 확률 최대화 되는 애 뽑기
        outcome_probabilities = res2.mean(axis = 0) ##  outcome 은 그냥 평균으로 뽑기



    ## 이부분도 생각 필요.. rule 을 cost를 maximize 하는 기준으로 threshold 탐색 필요할지도..
    # Choose label with highest probability.
    murmur_labels = np.zeros(len(murmur_classes), dtype=np.int_)
    if murmur_probabilities[0] > 0.496 :
        idx = 0
    else :
        idx = 2
#    idx = np.argmax(murmur_probabilities)
    murmur_labels[idx] = 1

    outcome_labels = np.zeros(len(outcome_classes), dtype=np.int_)
    if outcome_probabilities[0] > 0.617 :
        idx = 0
    else :
        idx = 1
#    idx = np.argmax(outcome_probabilities)
    outcome_labels[idx] = 1

    # Concatenate classes, labels, and probabilities.
    classes = murmur_classes + outcome_classes
    labels = np.concatenate((murmur_labels, outcome_labels))
    probabilities = np.concatenate((murmur_probabilities, outcome_probabilities))

    return classes, labels, probabilities






###################### lcnn_dr random search

for i in range(5000) :
    model_folder = 'hyper_1_1'
    output_folder = '/Data/hmd/hmd_sy/2021_hmd/tmp/out_hyper_1_1'

    # maxlen = np.random.choice([120000,80000, 50000, 15000])
    winlen = 512
    hoplen = 256
    nmel = 140 #np.random.choice([100, 120, 140])
    nsec = 50
    trim = 0 #np.random.choice([0,2000, 4000])
    use_mel = True
    use_cqt = False #np.random.choice([True,False])
    use_stft = False#np.random.choice([True, False])
    use_rr = True
    # use_rr_seq = False #True
    use_raw = False #True

    use_b_detect = True
    use_s1s2 = True
    use_mm = True

    #################
    # envelope parameter
    #################
    samp_sec = 50
    sample_rate = 4000
    pre_emphasis  = 0
    sr = 4000
    n_mels = 140

    # maxlen = 120000
    win_length = 512
    hop_length = 256

    fs = 4000 #sample rate 
    cutoff = 150 #sample frequency 
    nyq = 0.5 * fs 



    params_feature = {'samp_sec': nsec,
                #### melspec, stft 피쳐 옵션들  
                'pre_emphasis': 0,
                'hop_length': hoplen,
                'win_length':winlen,
                'n_mels': nmel,
                #### cqt 피쳐 옵션들  
                'filter_scale': 1,
                'n_bins': 80,
                'fmin': 10,

                ### 사용할 피쳐 지정
                    'trim' : trim, # 앞뒤 얼마나 자를지? 4000 이면 1초
                    'use_rr' : use_rr,
                    'use_b_detect': use_b_detect,
                    'use_raw' : use_raw,
                    'use_mel' : use_mel,
                    'use_cqt' : use_cqt,
                    'use_stft' : use_stft          
    }


    mm_weight = 3 #np.random.choice([2,3,4,5])
    oo_weight = 3 #np.random.choice([2,3,4,5,6])
    ord1 = True #np.random.choice([True,False])
    mm_mean = False #np.random.choice([True,False])
    dp = 0 #np.random.choice([0, .1, .2, .3])
    fc = False #np.random.choice([True,False])


    ext = True


    chaug = 10 #np.random.choice([0, 10])
    mixup = True #np.random.choice([True,False])
    cout = .8 #np.random.choice([0, 0.8])
    wunknown = 1 #np.random.choice([1, 0.7, .5, .2])
    n1 = 0 #np.random.choice([0,2])
    if n1 == 0 :
        ranfil = False
    else :
        ranfil = [n1, [18,19,20,21,22,23]]
        
    use_mel = params_feature['use_mel']
    use_cqt = params_feature['use_cqt']
    use_stft = params_feature['use_stft']
    nep = 100

    
    
    import pickle
    with open('/Data/hmd/hmd_sy/2021_hmd/features_trn_envel.pkl','rb') as f:
        features_trn = pickle.load(f)

    with open('/Data/hmd/hmd_sy/2021_hmd/mm_lbs_trn_envel.pkl','rb') as f:
        mm_lbs_trn = pickle.load(f)

    with open('/Data/hmd/hmd_sy/2021_hmd/out_lbs_trn_envel.pkl','rb') as f:
        out_lbs_trn = pickle.load(f)



    with open('/Data/hmd/hmd_sy/2021_hmd/features_test_envel.pkl','rb') as f:
        features_test = pickle.load(f)

    with open('/Data/hmd/hmd_sy/2021_hmd/mm_lbs_test_envel.pkl','rb') as f:
        mm_lbs_test = pickle.load(f)

    with open('/Data/hmd/hmd_sy/2021_hmd/out_lbs_test_envel.pkl','rb') as f:
        out_lbs_test = pickle.load(f)
        
    # (2532, 140, 782) 에서 (2532, 140, 782, 1)로 변경
    a, b, c = features_trn['mel1'].shape
    features_trn['mel1']= features_trn['mel1'].reshape(a,b,c,1)

    a, b, c = features_trn['s1s2_mel'].shape
    features_trn['s1s2_mel'] = features_trn['s1s2_mel'].reshape(a,b,c,1)

    a, b, c = features_trn['mm_mel'].shape
    features_trn['mm_mel'] = features_trn['mm_mel'].reshape(a,b,c,1)

    mel_input_shape = features_trn['mel1'][0].shape
    cqt_input_shape = features_trn['cqt1'][0].shape
    stft_input_shape = features_trn['stft1'][0].shape

    mel_s1s2_input_shape = features_trn['s1s2_mel'][0].shape
    mel_mm_input_shape = features_trn['mm_mel'][0].shape


    params_feature['ord1'] = ord1
    params_feature['mm_mean'] = mm_mean
    params_feature['dp'] = dp
    params_feature['fc'] = fc
    params_feature['ext'] = ext
    params_feature['oo_weight'] = oo_weight
    params_feature['mm_weight'] = mm_weight
    params_feature['chaug'] = chaug
    params_feature['cout'] = cout
    params_feature['wunknown'] = wunknown
    params_feature['mixup'] = mixup
    params_feature['n1'] = n1

    params_feature['mel_shape'] = mel_input_shape
    params_feature['cqt_shape'] = cqt_input_shape
    params_feature['stft_shape'] = stft_input_shape

    params_feature['s1s2_shape'] = mel_s1s2_input_shape
    params_feature['mm_shape'] = mel_mm_input_shape

    params_feature['use_mel'] = use_mel
    params_feature['use_cqt'] = use_cqt
    params_feature['use_stft'] = use_stft

    params_feature['use_rr'] = use_rr
    params_feature['use_s1s2'] = use_s1s2
    params_feature['use_mm'] = use_mm




    print(params_feature)
    
    
        



    model1 = get_LCNN_o_1_dr_rr(mel_input_shape, cqt_input_shape, stft_input_shape, 
                                mel_s1s2_input_shape, mel_mm_input_shape, use_s1s2 = use_s1s2, use_mm = use_mm,
                                use_mel = use_mel, use_cqt = use_cqt, use_stft = use_stft, ord1 = ord1, dp = dp, fc = fc, ext = False, ext2 = True)
    model2 = get_LCNN_2_dr_rr(mel_input_shape, cqt_input_shape, stft_input_shape, 
                            mel_s1s2_input_shape, mel_mm_input_shape, use_s1s2 = use_s1s2, use_mm = use_mm,
                            use_mel = use_mel, use_cqt = use_cqt, use_stft = use_stft, dp = dp, fc = fc, ext = True, ext2 = False)

    n_epoch = nep
    lr = LearningRateScheduler(lambda e: sigmoidal_decay(e, end=n_epoch))
    batch_size = 64

    if mixup :
        beta_param = .7
    else :
        beta_param = 0

    params = {'batch_size': batch_size,
            #          'input_shape': (100, 313, 1),
            'shuffle': True,
            'chaug': chaug,
            'beta_param': beta_param,
            'cout': cout
    #              'mixup': mixup,
            #          'lowpass': [.5, [11,12,13,14,15,16,17,18]]
            #          'highpass': [.5, [78,79,80,81,82,83,84,85]]
    #              'ranfilter2' : [3, [18,19,20,21,22,23]]
            #           'dropblock' : [30, 100]
            #'device' : device
    }

    if mixup :
        params['mixup'] = mixup
        params['ranfilter2'] = ranfil
    else :
        params['cutout'] = cout

    params_no_shuffle = {'batch_size': batch_size,
                        #          'input_shape': (100, 313, 1),
                        'shuffle': False,
                        'beta_param': 0.7,
                        'mixup': False
                        #'device': device
    }

    if ord1 :
        class_weight = {0: mm_weight, 1: 1.}
    else :
        class_weight = {0: mm_weight, 1: wunknown, 2:1.}


    if mixup :
            TrainDGen_1 = Generator0([features_trn['age'],features_trn['sex'], features_trn['hw'], 
                                    features_trn['preg'], features_trn['loc'], features_trn['mel1'],
                                    features_trn['cqt1'],features_trn['stft1'],features_trn['rr1'], 
                                    features_trn['s1s2_mel'],features_trn['mm_mel']], 
                            mm_lbs_trn,  ## our Y
                                **params)()
            model1.fit(TrainDGen_1,
                validation_data = ([features_test['age'],features_test['sex'], features_test['hw'], 
                                    features_test['preg'], features_test['loc'], features_test['mel1'], 
                                    features_test['cqt1'], features_test['stft1'],features_test['rr1'],
                                    features_test['s1s2_mel'],features_test['mm_mel']], 
                                    mm_lbs_test), 
                callbacks=[lr],
                steps_per_epoch=np.ceil(len(mm_lbs_trn)/64),
                class_weight=class_weight, 
                epochs = n_epoch)

    else :
        TrainGen = DataGenerator([features_trn['age'],features_trn['sex'], features_trn['hw'], 
                                features_trn['preg'], features_trn['loc'], features_trn['mel1'],
                                features_trn['cqt1'],features_trn['stft1'],features_trn['rr1'], 
                                features_trn['s1s2_mel'],features_trn['mm_mel']], 
                    mm_lbs_trn,  ## our Y
                    **params)
        model1.fit(TrainGen,
            validation_data = ([features_test['age'],features_test['sex'], features_test['hw'], 
                                features_test['preg'], features_test['loc'], features_test['mel1'], 
                                features_test['cqt1'], features_test['stft1'],features_test['rr1'], 
                                features_test['s1s2_mel'],features_test['mm_mel']], 
                                mm_lbs_test), 
            callbacks=[lr],
            #        steps_per_epoch=np.ceil(len(mm_lbs_trn)/64),
            class_weight=class_weight, 
            epochs = n_epoch)
        
    n_epoch = nep
    lr = LearningRateScheduler(lambda e: sigmoidal_decay(e, end=n_epoch))
    batch_size = 64
    params = {'batch_size': batch_size,
            #          'input_shape': (100, 313, 1),
            'shuffle': True,
            'chaug': chaug,
            'beta_param': beta_param,
            'cout': cout,
#              'mixup': True,
            #          'lowpass': [.5, [11,12,13,14,15,16,17,18]]
#            'highpass': [.5, [78,79,80,81,82,83,84,85]],
#              'ranfilter2' : [3, [18,19,20,21,22,23]]
            #           'dropblock' : [30, 100]
            #'device' : device
    }


    if mixup :
        params['mixup'] = mixup
        params['ranfilter2'] = ranfil
    else :
        params['cutout'] = cout


    params_no_shuffle = {'batch_size': batch_size,
                        #          'input_shape': (100, 313, 1),
                        'shuffle': False,
                        'beta_param': 0.7,
                        'mixup': False
                        #'device': device
    }

    class_weight = {0: oo_weight, 1: 1.}


    
    if mixup :
        TrainDGen_1 = Generator0([features_trn['age'],features_trn['sex'], features_trn['hw'], 
                                    features_trn['preg'], features_trn['loc'], features_trn['mel1'],
                                    features_trn['cqt1'],features_trn['stft1'],features_trn['rr1'], 
                                    features_trn['s1s2_mel'],features_trn['mm_mel']], 
                                    out_lbs_trn,  ## our Y
                        **params)()

        model2.fit(TrainDGen_1,
        validation_data = ([features_test['age'],features_test['sex'], features_test['hw'], 
                                    features_test['preg'], features_test['loc'], features_test['mel1'], 
                                    features_test['cqt1'], features_test['stft1'],features_test['rr1'],
                                    features_test['s1s2_mel'],features_test['mm_mel']], 
                                    out_lbs_test), 
            callbacks=[lr],
            steps_per_epoch=np.ceil(len(out_lbs_trn)/64),
            class_weight=class_weight, 
            epochs = n_epoch)
    else :
        TrainGen = DataGenerator([features_trn['age'],features_trn['sex'], features_trn['hw'], 
                                features_trn['preg'], features_trn['loc'], features_trn['mel1'],
                                features_trn['cqt1'],features_trn['stft1'],features_trn['rr1'], 
                                features_trn['s1s2_mel'],features_trn['mm_mel']], 
                                out_lbs_trn,  ## our Y
                    **params)
        model2.fit(TrainGen,
            validation_data = ([features_test['age'],features_test['sex'], features_test['hw'], 
                                features_test['preg'], features_test['loc'], features_test['mel1'], 
                                features_test['cqt1'], features_test['stft1'],features_test['rr1'],
                                features_test['s1s2_mel'],features_test['mm_mel']], 
                                out_lbs_test), 
            callbacks=[lr],
            class_weight=class_weight, 
            epochs = n_epoch)
        


    # params_feature['mel_shape'] = mel_input_shape
    # params_feature['cqt_shape'] = cqt_input_shape
    # params_feature['stft_shape'] = stft_input_shape

    # params_feature['use_mel'] = use_mel
    # params_feature['use_cqt'] = use_cqt
    # params_feature['use_stft'] = use_stft
    
    save_challenge_model(model_folder, model1, model2, m_name1 = 'lcnn1_dr_rr', m_name2 = 'lcnn2_dr_rr', param_feature = params_feature)

    run_model(model_folder, test_folder, output_folder, allow_failures = True, verbose = 1)

    murmur_scores, outcome_scores = evaluate_model(test_folder, output_folder)
    classes, auroc, auprc, auroc_classes, auprc_classes, f_measure, f_measure_classes, accuracy, accuracy_classes, weighted_accuracy, cost = murmur_scores
    murmur_output_string = 'AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(auroc, auprc, f_measure, accuracy, weighted_accuracy, cost)
    murmur_class_output_string = 'Classes,{}\nAUROC,{}\nAUPRC,{}\nF-measure,{}\nAccuracy,{}\n'.format(
    ','.join(classes),
    ','.join('{:.3f}'.format(x) for x in auroc_classes),
    ','.join('{:.3f}'.format(x) for x in auprc_classes),
    ','.join('{:.3f}'.format(x) for x in f_measure_classes),
    ','.join('{:.3f}'.format(x) for x in accuracy_classes))

    params_feature['mm_weighted_accuracy'] = weighted_accuracy

    classes, auroc, auprc, auroc_classes, auprc_classes, f_measure, f_measure_classes, accuracy, accuracy_classes, weighted_accuracy, cost = outcome_scores
    outcome_output_string = 'AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(auroc, auprc, f_measure, accuracy, weighted_accuracy, cost)
    outcome_class_output_string = 'Classes,{}\nAUROC,{}\nAUPRC,{}\nF-measure,{}\nAccuracy,{}\n'.format(
    ','.join(classes),
    ','.join('{:.3f}'.format(x) for x in auroc_classes),
    ','.join('{:.3f}'.format(x) for x in auprc_classes),
    ','.join('{:.3f}'.format(x) for x in f_measure_classes),
    ','.join('{:.3f}'.format(x) for x in accuracy_classes))

    output_string = '#Murmur scores\n' + murmur_output_string + '\n#Outcome scores\n' + outcome_output_string \
    + '\n#Murmur scores (per class)\n' + murmur_class_output_string + '\n#Outcome scores (per class)\n' + outcome_class_output_string

    params_feature['out_cost'] = cost

    label_folder = test_folder
    murmur_classes = ['Present', 'Unknown', 'Absent']
    outcome_classes = ['Abnormal', 'Normal']

    # Load and parse label and model output files.
    label_files, output_files = find_challenge_files(label_folder, output_folder)
    murmur_labels = load_murmurs(label_files, murmur_classes)
    murmur_binary_outputs, murmur_scalar_outputs = load_classifier_outputs(output_files, murmur_classes)
    outcome_labels = load_outcomes(label_files, outcome_classes)
    outcome_binary_outputs, outcome_scalar_outputs = load_classifier_outputs(output_files, outcome_classes)

    max_wc = 0
    max_th = 0
    min_cost = 100000
    min_th = 1000
    for th1 in range(1, 100) : #  [0.01, 0.05, 0.1, 0.15,0.2, 0.25, 0.3, 0.32, 0.35, 0.38, 0.4, 0.42, 0.45, 0.47, 0.5, 0.52, 0.55, 0.57, 0.6, 0.62, 0.65, 0.7, 0.75, 0.8] :
        th1 = th1 / 100 
        murmur_binary_outputs[:,0] = murmur_scalar_outputs[:,0] > th1
        murmur_binary_outputs[:,2] = murmur_scalar_outputs[:,2] > 1 - th1
        outcome_binary_outputs[:,0] = outcome_scalar_outputs[:,0] > th1
        outcome_binary_outputs[:,1] = outcome_scalar_outputs[:,1] > 1 - th1
        # For each patient, set the 'Present' or 'Abnormal' class to positive if no class is positive or if multiple classes are positive.
        murmur_labels = enforce_positives(murmur_labels, murmur_classes, 'Present')
        murmur_binary_outputs = enforce_positives(murmur_binary_outputs, murmur_classes, 'Present')
        outcome_labels = enforce_positives(outcome_labels, outcome_classes, 'Abnormal')
        outcome_binary_outputs = enforce_positives(outcome_binary_outputs, outcome_classes, 'Abnormal')
        # Evaluate the murmur model by comparing the labels and model outputs.
        murmur_auroc, murmur_auprc, murmur_auroc_classes, murmur_auprc_classes = compute_auc(murmur_labels, murmur_scalar_outputs)
        murmur_f_measure, murmur_f_measure_classes = compute_f_measure(murmur_labels, murmur_binary_outputs)
        murmur_accuracy, murmur_accuracy_classes = compute_accuracy(murmur_labels, murmur_binary_outputs)
        murmur_weighted_accuracy = compute_weighted_accuracy(murmur_labels, murmur_binary_outputs, murmur_classes) # This is the murmur scoring metric.

        if murmur_weighted_accuracy > max_wc :
            max_wc = murmur_weighted_accuracy
            max_th = th1

        outcome_auroc, outcome_auprc, outcome_auroc_classes, outcome_auprc_classes = compute_auc(outcome_labels, outcome_scalar_outputs)
        outcome_f_measure, outcome_f_measure_classes = compute_f_measure(outcome_labels, outcome_binary_outputs)
        outcome_accuracy, outcome_accuracy_classes = compute_accuracy(outcome_labels, outcome_binary_outputs)
        outcome_weighted_accuracy = compute_weighted_accuracy(outcome_labels, outcome_binary_outputs, outcome_classes)
        outcome_cost = compute_cost(outcome_labels, outcome_binary_outputs, outcome_classes, outcome_classes) # This is the clinical outcomes scoring metric.

        if outcome_cost < min_cost :
            min_cost = outcome_cost
            min_th = th1

    params_feature['max_wc'] = max_wc
    params_feature['murmur_auroc'] = murmur_auroc
    params_feature['murmur_auprc'] = murmur_auprc
    params_feature['murmur_f_measure']=murmur_f_measure
    params_feature['murmur_accuracy']=murmur_accuracy
    params_feature['min_cost'] = min_cost
    params_feature['max_th'] = max_th
    params_feature['min_th'] = min_th

    tnow = datetime.datetime.now()
    fnm = 'res_1_1/rec'+ str(tnow)+'.pk'

    print(params_feature)

    with open(fnm, 'wb') as f:
        pickle.dump(params_feature, f, pickle.HIGHEST_PROTOCOL)

 

















