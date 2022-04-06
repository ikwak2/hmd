#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
import pickle as pk

def get_logger(filename):
    # Logging configuration: set the basic configuration of the logging system
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(processName)s, %(process)s] [%(levelname)-5.5s]  %(message)s',
                                      datefmt='%m-%d %H:%M')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # File logger
    file_handler = logging.FileHandler("{}.log".format(filename))
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    # Stderr logger
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(log_formatter)
    std_handler.setLevel(logging.DEBUG)
    logger.addHandler(std_handler)
    return logger


def cal_indicator(pre,label):
    TP = 0.0
    FN = 0.0
    TN = 0.0
    FP = 0.0
    for i, it in enumerate(pre):
        if it == 1.0 and label[i] == 1.0:
            TP += 1.0
        elif it == 1.0 and label[i] == -0.0:
            FP += 1
        elif it == -0.0 and label[i] == 1.0:
            FN += 1
        elif it == -0.0 and label[i] == -0.0:
            TN += 1.0
    return TP, FP, TN, FN


#ANCHOR Checks of the directory exist and if not, creates a new directory
def checkdir(directory):
    try:
        os.makedirs(directory)
    except OSError:
        pass

def get_class_distribution(obj):
    count_dict = {
        "rating_unk": 0,
        "rating_pre": 0,
        "rating_abs": 0,

    }
    
    for i in obj:
        if i == 0: 
            count_dict['rating_unk'] += 1
        elif i == 1: 
            count_dict['rating_pre'] += 1
        elif i == 2: 
            count_dict['rating_abs'] += 1
        else:
            print("Check classes.")
            
    return count_dict

def multi_acc(y_pred, y_test):
    y_pred_softmax = F.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc) * 100
    
    return acc

def like_acc(y_pred, y_test) : 
    like_acc = (1-torch.sum(abs(y_pred-y_test))/int(y_pred.shape[0]))*100
    return like_acc

def load_train_valid(num) :
    root = '/Data2/ECG2021/의대ecg/3ch_model_ti/preprocessing_neurokit2/data/spectrogram/'

    #label
    with open(root+'train_label_hut'+str(num)+'.pk','rb') as f :
        train_y_data = pk.load(f)
    with open(root+'valid_label_hut'+str(num)+'.pk','rb') as f :
        valid_y_data = pk.load(f)

    #spectrogram
    with open(root+'train_spec'+str(num)+'.pk','rb') as f :
        train_spec = pk.load(f)
    with open(root+'valid_spec'+str(num)+'.pk','rb') as f :
        valid_spec = pk.load(f)
        
    return train_spec,valid_spec,train_y_data,valid_y_data

def load_test(num) :
    root = '/Data2/ECG2021/의대ecg/3ch_model_ti/preprocessing_neurokit2/data/spectrogram/'

    #label
    with open(root+'test_label_hut'+str(num)+'.pk','rb') as f :
        test_y_data = pk.load(f)

    #spectrogram
    with open(root+'test_spec'+str(num)+'.pk','rb') as f :
        test_spec = pk.load(f)
        
    return test_spec, test_y_data

def save_sets_spec() :
#    import msgpack
    root = '/Data2/ECG2021/의대ecg/3ch_model_ti/preprocessing_neurokit2/data/spectrogram/'
    root2 = '/Data2/ECG2021/의대ecg/3ch_model_ti/preprocessing_neurokit2/data/'
#    with open(root2+'data_split.pk','rb') as f :
#        data_split = pk.load(f)
    with open(root2+'data_split0.pk','rb') as f :
        data_split0 = pk.load(f)
    with open(root2+'data_split1.pk','rb') as f :
        data_split1 = pk.load(f)
    with open(root2+'data_split2.pk','rb') as f :
        data_split2 = pk.load(f)
    data_s = [data_split0,data_split1,data_split2]
    for i,data in enumerate(data_s) :
        for sets in ['train','test','valid'] :   # train, valid, test
            set_spec = np.empty((0,3,6,313))

            for idx, fnm in enumerate(data[sets]) :   # sets fnm list
                if (os.path.isfile("/Data2/ECG2021/의대ecg/3ch_model_ti/preprocessing_neurokit2/data/spectrogram/%s/%s.pk"%(sets,fnm))) == False :
                    print('pass')
                    continue
                elif os.path.isfile("/Data2/ECG2021/의대ecg/3ch_model_ti/preprocessing_neurokit2/data/spectrogram/%s/%s.pk"%(sets,fnm)) :
                    fnm_em_spec = np.empty((0,6,313))         
                    with open(root + sets + '/' + fnm + '.pk','rb') as f :
                        fnm_spec = pk.load(f)
                    for lead in list(fnm_spec.keys()) :     # ch 넣기
                        fnm_em_spec = np.concatenate((fnm_em_spec,fnm_spec[lead].reshape(-1,6,313)),axis = 0)

                    set_spec = np.concatenate((set_spec,fnm_em_spec.reshape(-1,3,6,313)),axis = 0)

            #array = msgpack.packb(set_spec.tolist(), use_bin_type=True)
            with open(root + sets + '_spec'+ str(i)+'.pk', "wb") as f:
                pk.dump(set_spec, f, protocol = 4)
            break
            
    return None

def save_sets_label() :
#    import msgpack
    root = '/Data2/ECG2021/의대ecg/3ch_model_ti/preprocessing_neurokit2/data/spectrogram/'
    root2 = '/Data2/ECG2021/의대ecg/3ch_model_ti/preprocessing_neurokit2/data/'
#    with open(root2+'data_split.pk','rb') as f :
#        data_split = pk.load(f)
    with open(root2+'data_split0.pk','rb') as f :
        data_split0 = pk.load(f)
    with open(root2+'data_split1.pk','rb') as f :
        data_split1 = pk.load(f)
    with open(root2+'data_split2.pk','rb') as f :
        data_split2 = pk.load(f)
    data_s = [data_split0,data_split1,data_split2]
    with open(root2+'fnm_label_hut.pk','rb') as f :
        fnm_label_hut = pk.load(f)
    for i,data in enumerate(data_s) :
        for sets in ['train','test','valid'] :   # train, valid, test
            set_label = np.empty((0))

            for idx, fnm in enumerate(data[sets]) :   # sets fnm list
                if (os.path.isfile("/Data2/ECG2021/의대ecg/3ch_model_ti/preprocessing_neurokit2/data/spectrogram/%s/%s.pk"%(sets,fnm))) == False :
                    print('pass')
                    continue
                elif os.path.isfile("/Data2/ECG2021/의대ecg/3ch_model_ti/preprocessing_neurokit2/data/spectrogram/%s/%s.pk"%(sets,fnm)) :

                    set_label = np.concatenate((set_label,fnm_label_hut[fnm]),axis = 0)

            #array = msgpack.packb(set_spec.tolist(), use_bin_type=True)
            with open(root + sets + '_label_hut'+ str(i)+'.pk', "wb") as f:
                pk.dump(set_label, f, protocol = 4)
            break
    return None
