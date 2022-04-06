#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################
import pickle as pk
import glob
import os
import time
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import sys
import random
from tqdm import tqdm
from tools.utils import load_train_valid, get_class_distribution,get_logger, multi_acc, load_test
sys.path.append("..")
from models import cnn_ti
from reader.data_reader_physionet import myDataLoader, myDataset
import gc
#from torch.utils.tensorboard import SummaryWriter
from helper_code import *
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib
from sklearn.impute import SimpleImputer
#from sklearn.ensemble import RandomForestClassifier
from models import *
from get_feature import *
import pickle as pk

#
################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    gc.collect()
    torch.cuda.empty_cache()
    GPU_NUM = 0 # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')


    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    classes = ['Present', 'Unknown', 'Absent']
    num_classes = len(classes)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    features, labels = get_features(data_folder, patient_files)
    
    
#    model = get_toy((100, 313, 1))
    log_dir = './log/'
    project = 'toy0'
    logger = get_logger(log_dir + '/' + project)
    label = np.empty(len(labels))
    for i,l in enumerate(labels) :
        label[i] = int(list(l).index(1))

    class_count = [i for i in get_class_distribution(label).values()]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float)
    
    dataset_train = myDataset(features, label)
    dataloader_train = myDataLoader(dataset=dataset_train,
                                    batch_size=64,
                                    shuffle=False,
                                    num_workers=0)
    all_file_train = len(dataloader_train)
    if verbose >= 1:
        print("- {} training samples".format(len(dataset_train)))
        print("- {} training batches".format(len(dataloader_train)))
    
#    model = get_toy((100, 313, 1))

    
    nnet = cnn_ti(num_classes)
    nnet = nnet.cuda()
    optimizer = optim.Adam(nnet.parameters(), lr=4e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.91)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    # Train the model.
    if verbose >= 1:
        print('Training model...')
    imputer = SimpleImputer().fit(features['hw'])
    features['hw'] = imputer.transform(features['hw'])
    
    
    for iter_ in range(3):  # args.end_iter
        start_time = time.time()
        running_loss = 0.0
        train_epoch_acc = 0.0
        nnet.train()
        
        idx = 0
#    epoch[str(iter_)] = {}
        for audio_feature, age, sex, hw, preg, loc, data_label_torch in tqdm(dataloader_train):  
            #print(audio_feature.shape)
        
            idx = idx + 1
            #audio_feature = torch.nan_to_num(audio_feature, nan=0)
            audio_feature[audio_feature != audio_feature] = 0
        
            audio_feature = audio_feature.permute(1,3,0,2)
            #print(audio_feature.shape)
            audio_feature = audio_feature.cuda()
            age = age.permute(1,0).float().cuda()
            #print(age.shape)
            sex = sex.permute(1,0).float().cuda()
            #print(sex.shape)
            hw = hw.permute(1,0).float().cuda()
            #print(hw.shape)
            preg = preg.permute(1,0).float().cuda()
            #print(preg.shape)
            loc = loc.permute(1,0).float().cuda()
            #print(loc.shape)
    #        for idx in range(len(data_label_torch)) :
    #            data_label_torch[idx][0] = mapping[int(data_label_torch[idx][0])]
            data_label_torch = data_label_torch.cuda()
                
            data_label_torch = data_label_torch.long()
            outputs = nnet(input=audio_feature, age = age , sex = sex, hw = hw, preg = preg, loc = loc)
        
            loss = criterion(outputs.float(), data_label_torch.squeeze(1))
            train_acc = multi_acc(outputs, data_label_torch.squeeze(1))
            #print(outputs.shape)
            #print(data_label_torch.squeeze(1).shape)
            #print(train_acc)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_epoch_acc += train_acc.item()
            #if idx % 100 == 0 :
            #    print("lr: ", optimizer.param_groups[0]['lr'])
            #    print('batch : ',idx, ' done')
            
        logger.info("Iteration:{0}, loss = {1:.6f}, acc = {2:.3f} ".format(iter_, running_loss/all_file_train, train_epoch_acc/all_file_train))
        
        
    
    
    
#    model.fit([features1[0]['age'],features1[0]['sex'], features1[0]['hw'], features1[0]['preg'], features1[0]['loc'], 
#           features1[0]['mel1']], features1[1],
#          epochs = 2)    
    
    

    # Save the model.

    save_challenge_model(model_folder, nnet, classes, m_name = 'toy_ti', mel_shape = (100, 313, 1) )
    if verbose >= 1:
        print('Done.')

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.

#def load_challenge_model(model_folder, verbose):
#    filename = os.path.join(model_folder, 'model.sav')
#    return joblib.load(filename)

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
    
#    if model['model'] == 'toy_ti' :
#        model1 = get_toy(model['mel_shape'])
#    model1.load_weights(model['model_fnm'])
    GPU_NUM = 0 # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)


    if model['model'] == 'toy_ti' :
        trained_model =  torch.load(model['model_fnm'])
        nnet = cnn_ti(3)
        nnet.load_state_dict(trained_model)
        nnet = nnet.cuda()
    nnet.eval()
        
        
    classes = model['classes']
    # Load features.
    features = get_feature_one(data, verbose = 0)

    features['mel1'] = []
    for i in range(len(recordings)) :
        mel1 = feature_extract_melspec(recordings[i])[0]
        features['mel1'].append(mel1)

    M, N = features['mel1'][0].shape
    for i in range(len(features['mel1'])) :
        features['mel1'][i] = features['mel1'][i].reshape(M,N,1)   
        
    features['mel1'] = np.array(features['mel1'])
#    print(features)
    # Impute missing data.
#    imputer = SimpleImputer().fit(features['hw'])
#    features['hw'] = imputer.transform(features['hw'])
    
    audio_feature = torch.from_numpy(features['mel1'])
    
    audio_feature[audio_feature != audio_feature] = 0
        
    audio_feature = audio_feature.permute(0,3,1,2)
    #print(audio_feature.shape)
    audio_feature = audio_feature.cuda()
    
    age = torch.from_numpy(features['age']).float().cuda()
    #print(age.shape)
    sex = torch.from_numpy(features['sex']).float().cuda()
    #print(sex.shape)
    hw = torch.from_numpy(features['hw']).float().cuda()
    print(hw.shape)
    preg = np.empty(len(features['preg']))
    
    for idx, p in enumerate((features['preg'])) :
        if p :
            preg[i] = 1
        else :
            preg[i] = 0
    preg = torch.from_numpy(preg).reshape(-1,1).float().cuda()
#    print(preg.shape)
    loc = torch.from_numpy(features['loc']).float().cuda()
    #print(loc.shape)
    #        for idx in range(len(data_label_torch)) :
    #            data_label_torch[idx][0] = mapping[int(data_label_torch[idx][0])]

    outputs = nnet(input=audio_feature, age = age , sex = sex, hw = hw, preg = preg, loc = loc)
    softmax = nn.Softmax(dim = 1)
    outputs = softmax(outputs)
    
    prob1 = outputs.mean(axis = 0)
    
    prob1 = prob1.data.cpu().numpy()
    
    
    
    
    
#    print(prob1.shape)
    
#    res1 = model1.predict([features['age'], features['sex'], features['hw'], features['preg'], features['loc'], features['mel1']])

    # Get classifier probabilities.
#    prob1 = res1.mean(axis = 0) ## simple rule for now
    idx = np.argmax(prob1)
    # Choose label with higher probability.
    labels = np.zeros(len(classes), dtype=np.int_)
    labels[idx] = 1

    return classes, labels, prob1


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
#def save_challenge_model(model_folder, classes, imputer, classifier):
#    d = {'classes': classes, 'imputer': imputer, 'classifier': classifier}
#    filename = os.path.join(model_folder, 'model.sav')
#    joblib.dump(d, filename, protocol=0)
    
def save_challenge_model(model_folder, model, classes, m_name, mel_shape = (100, 313, 1)) :
    os.makedirs(model_folder, exist_ok=True)
    info_fnm = os.path.join(model_folder, 'desc.pk')
    filename = os.path.join(model_folder, m_name + '_model.hdf5')
#    model.save(filename)
    torch.save(model.state_dict(), filename)
    d = {'model': m_name, 'classes': classes, 'mel_shape': mel_shape, 'model_fnm': filename}    
    with open(info_fnm, 'wb') as f:
        pk.dump(d, f, pk.HIGHEST_PROTOCOL)

        
        
