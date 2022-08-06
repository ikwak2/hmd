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
    
    
    
import tensorflow.keras
from tensorflow.keras import backend as K
from scipy.io import wavfile
import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.utils import to_categorical
import msgpack
import msgpack_numpy as m
import os
from tensorflow.keras.layers import Conv2D, maximum, add, SeparableConv2D
from sklearn.metrics import roc_curve

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X_train, y_train, batch_size=32, 
                 shuffle=True, 
                 cutmix = False, cutout = False, specaug = False, specmix = False,
                 beta_param = False, lowpass = False, highpass = False, ranfilter = False, ranfilter2 = False, dropblock = False
                ):
        'Initialization'
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.beta_param = beta_param

        self.cutmix = cutmix
        self.cutout = cutout
        self.specaug = specaug
        self.specmix = specmix
        
        self.lowpass = lowpass
        self.highpass = highpass
        self.ranfilter = ranfilter
        self.ranfilter2 = ranfilter2
        self.dropblock = dropblock
#        self.input_shape = input_shape
        
        self.on_epoch_end()

        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.y_train) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
#        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

#    def get_input_shape(self):
#        return (self.M,self.N,1)
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.y_train))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        
        
        
        def get_box(lambda_value):
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
        
        
        X = []
        y = self.y_train[indexes]
        for X_temp in self.X_train:
            X.append(X_temp[indexes])
            
        nX = X.copy()
        ny = y.copy()        

        if self.beta_param :
            lamda1 = np.random.beta(self.beta_param, self.beta_param, size = self.batch_size)   ## beta_param default : 0.7  STC페이퍼 추천은 0.6~0.8
            ny = np.array([ lamda1[i]*y[i,:] + (1-lamda1[i])*y[i+1,:] if i != self.batch_size-1  else  lamda1[i]*y[i,:] + (1-lamda1[i])*y[0,:] for i in range(self.batch_size) ])
            
            for k in range(len(X)) :
                if len(X[k].shape) == 4: 
                    Xc = X[k]
                    nX[k] = np.array([ lamda1[i]*Xc[i,:,:,:] + (1-lamda1[i])*Xc[i+1,:,:,:] if i != self.batch_size-1  else  lamda1[i]*Xc[i,:,:,:] + (1-lamda1[i])*Xc[0,:,:,:] for i in range(self.batch_size) ])
                if len(X[k].shape) == 3: 
                    Xc = X[k]
                    nX[k] = np.array([ lamda1[i]*Xc[i,:,:] + (1-lamda1[i])*Xc[i+1,:,:] if i != self.batch_size-1  else  lamda1[i]*Xc[i,:,:] + (1-lamda1[i])*Xc[0,:,:] for i in range(self.batch_size) ])
                if len(X[k].shape) == 2: 
                    Xc = X[k]
                    nX[k] = np.array([ lamda1[i]*Xc[i,:] + (1-lamda1[i])*Xc[i+1,:] if i != self.batch_size-1  else  lamda1[i]*Xc[i,:] + (1-lamda1[i])*Xc[0,:] for i in range(self.batch_size) ])
                if len(X[k].shape) == 1: 
                    Xc = X[k]
                    nX[k] = np.array([ lamda1[i]*Xc[i] + (1-lamda1[i])*Xc[i+1] if i != self.batch_size-1  else  lamda1[i]*Xc[i] + (1-lamda1[i])*Xc[0] for i in range(self.batch_size) ])

        for k in range(len(X)) :
            if len(X[k].shape) == 4: 
                nf = nX[k].shape[1]
                nt = nX[k].shape[2]

                if self.cutout : 
                    lambda1 = np.random.beta(self.cutout, self.cutout, size = self.batch_size)   ## beta_param default : 0.7  STC페이퍼 추천은 0.6~0.8
                    for i in range(int(self.batch_size/2)) :
                        boundaryx1, boundaryy1, target_h, target_w = get_box(lambda1[i])
                        nX[k][2*i, boundaryx1:(boundaryx1+target_h), boundaryy1:(boundaryy1+target_w),: ] = 0
                        nX[k][2*i+1, boundaryx1:(boundaryx1+target_h), boundaryy1:(boundaryy1+target_w),: ] = 0

                if self.specaug :  ## ex: specaug = [ [2, 10] , [2, 15] ]
                    f_info, t_info = self.specaug
                    n_band_f, f_len = f_info
                    n_band_t, t_len = t_info
                    for i in range(self.batch_size) :
                        for _ in range(n_band_f) :
                            b1 = np.random.choice(f_len)
                            loc1 = np.random.choice(nf - b1, size = 1)[0]
                            nX[k][i, loc1:(loc1 + b1 - 1), :] = 0
                        for _ in range(n_band_t) :
                            b1 = np.random.choice(t_len)
                            loc1 = np.random.choice(nt - b1, size = 1)[0]
                            nX[k][i, :, loc1:(loc1 + b1 - 1)] = 0

                if self.lowpass :
                    uv, lp = self.lowpass
                    dec1 = np.random.choice(2, size = self.batch_size)
                    for i in range(self.batch_size) :
                        if dec1[i] == 1 :
                            loc1 = np.random.choice(lp, size = 1)[0]
                            nX[k][i,:loc1,:] = 0

                if self.highpass :
                    uv, hp = self.highpass
                    dec1 = np.random.choice(2, size = self.batch_size)
                    for i in range(self.batch_size) :
                        if dec1[i] == 1 :
                            loc1 = np.random.choice(hp, size = 1)[0]
                            nX[k][i, loc1:,:] = 0

                if self.ranfilter :   ## ex ranfilter = [10,11,12,13,14,15]
                    dec1 = np.random.choice(2, size = self.batch_size)
                    for i in range(self.batch_size) :
                        if dec1[i] == 1 :
                            b1 = np.random.choice(self.ranfilter, size = 1)[0]
                            loc1 = np.random.choice(nf - b1, size = 1)[0]
                            nX[k][i, loc1:(loc1 + b1 - 1), :] = 0

                if self.dropblock :  ## ex dropblock = [30, 80]
                    b1, b2 = self.dropblock
                    dec1 = np.random.choice(2, size = self.batch_size)
                    for i in range(self.batch_size) :
                        if dec1[i] == 1 :
                            loc1 = np.random.choice(nf- b1, size = 1)[0]
                            loc2 = np.random.choice(nt- b2, size = 1)[0]
                            nX[k][i, loc1:(loc1 + b1 - 1), loc2:(loc2 + b2 - 1)] = 0

                if self.ranfilter2 :   ## ex ranfilter2 = [4,[10,11,12,13,14,15]]
                    raniter, ranf = self.ranfilter2
                    dec1 = np.random.choice(raniter, size = self.batch_size)
                    for i in range(self.batch_size) :
                        if dec1[i] > 0 :
                            for j in range(dec1[i]) :
                                b1 = np.random.choice(ranf, size = 1)[0]
                                loc1 = np.random.choice(nf - b1, size = 1)[0]
                                nX[k][i, loc1:(loc1 + b1 - 1), :] = 0
                    
        return nX, ny
    



def evalScore(gen, model) :
    batchsize = gen.batch_size
    sc = model.predict_generator(gen)
    return sc[:,1]

def evalEER(gen, model) :
    batchsize = gen.batch_size
    
    label_valid = [ gen.labels[fnm] for fnm in gen.list_IDs ]
    label_valid = label_valid[:batchsize* (len(label_valid) // batchsize) ]

    sc = model.predict_generator(gen)
    fpr, tpr, threshold = roc_curve(label_valid, sc[:,1])
    fnr = 1-tpr
    minloc = np.absolute(fnr-fpr).argmin()
    eer = (fpr[minloc] + fnr[minloc]) / 2
    print('EER: {}'.format(eer))

    return eer

## generate filename

def gen_fname(model_name, params, dropout_rate = 'na', human_weight='na', endtxt = '.txt') :
    str1 = 'model_name!' + model_name + '!dropout_rate!' + str(dropout_rate) + '!human_weight!' + str(human_weight) + '!' 
    for (i,j) in params.items() :
        if (i == "lowpass") or (i == "highpass") or (i == "ranfilter2") or (i == "ranfilter") or (i == "dropblock") :
            str1 = str1 + str(i) + '!--!'
        elif (i != 'data_dir') and (i != 'n_classes') and (i != 'batch_size') and (i !='sr') and (i != 'tofile') :
            str1 = str1 + str(i) + '!' + str(j) + '!'
    str1 += endtxt
    return str1


def eval_track2(valid_gen, eval_gen, model) :

    batchsize = valid_gen.batch_size

    label_valid = [ valid_gen.labels[fnm] for fnm in valid_gen.list_IDs ]
    label_valid = label_valid[:batchsize* (len(label_valid) // batchsize) ]

    sc = model.predict_generator(valid_gen)
    fpr, tpr, threshold = roc_curve(label_valid, sc[:,1])
    fnr = 1-tpr
    minloc = np.absolute(fnr-fpr).argmin()
    eer = (fpr[minloc] + fnr[minloc]) / 2
#    print('EER: {}'.format(eer))

    threshold1 = threshold[minloc]
    sc1 = evalScore(eval_gen, model)
    
    acc1= 0
    tot_n = len(sc1)
    for i in range(tot_n) :
        acc1 += (sc1[i]>threshold1) 
        
    print("Error Rate:{}, threshold:{}".format(1- (acc1 / tot_n), threshold1))
    return 1 - acc1 / tot_n


def evalEER_f(gen, fnms) :

    st = False
    nf = len(fnms)
    for f in fnms:
        sc1 = np.load(f)
        if st :
            sc += sc1 / nf
        else :
            sc = sc1 / nf
            st = True

    batchsize = gen.batch_size
    
    label_valid = [ gen.labels[fnm] for fnm in gen.list_IDs ]
    label_valid = label_valid[:batchsize* (len(label_valid) // batchsize) ]

#    sc = model.predict_generator(gen)
    fpr, tpr, threshold = roc_curve(label_valid, sc)
    fnr = 1-tpr
    minloc = np.absolute(fnr-fpr).argmin()
    eer = (fpr[minloc] + fnr[minloc]) / 2
    print('EER: {}'.format(eer))

    return eer

def evalEER_f2(gen, fnms) :

    sc = []
    scstd = []
    nf = len(fnms)
    for f in fnms:
        sc1 = np.load(f)
        sc.append(sc1)
        scstd.append(sc1.std())

    scstd = np.array(scstd) / sum(scstd)
    sc2 = np.zeros( (len(sc1)) )
    for (sc1,std1) in zip(sc,scstd) :
        sc2 += sc1 / std1

    batchsize = gen.batch_size
    
    label_valid = [ gen.labels[fnm] for fnm in gen.list_IDs ]
    label_valid = label_valid[:batchsize* (len(label_valid) // batchsize) ]

#    sc = model.predict_generator(gen)
    fpr, tpr, threshold = roc_curve(label_valid, sc2)
    fnr = 1-tpr
    minloc = np.absolute(fnr-fpr).argmin()
    eer = (fpr[minloc] + fnr[minloc]) / 2
    print('EER: {}'.format(eer))

    return eer