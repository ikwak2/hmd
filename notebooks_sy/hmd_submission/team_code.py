#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib
from sklearn.impute import SimpleImputer
#from sklearn.ensemble import RandomForestClassifier
from models import *
from get_feature import *
from Generator0 import *
import pickle as pk
import sys
sys.path.insert(0,'peakutils')
import peakutils
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences

from scipy import special
import scipy.io as sio

import tensorflow as tf
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# data_folder = '/home/ubuntu/data/hmd/murmur/train'
# model_folder = '/home/ubuntu/physionet_cau_umn/mod1'
# verbose = 1
# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

# data_folder = '/Data2/hmd/data_split/murmur/test'
# output_folder = '/Data2/hmd/hmd_sy/python-classifier-2022/out1'


# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):

    
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

        
    per_sec = 4000
    winlen = 512
    hoplen = 256
    nmel = 120 #np.random.choice([100, 120, 140])
    nsec = 20 #np.random.choice([10, 20, 30, 40, 50])
    trim = 0 #np.random.choice([0,2000, 4000])
    use_mel = True
    use_cqt = False #np.random.choice([True,False])
    use_stft = False#np.random.choice([True, False])

    use_raw=False
    use_interval=True
    use_wav2=True
    maxlen1 = 246000
    min_dist = 500
    max_interval_len = 192

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
              ### RR interval 옵션
                  'min_dist':min_dist,
                  'max_interval_len' : max_interval_len,
                  'per_sec' : per_sec,
                  'use_interval' : use_interval,
            ### wav2vec2 옵션
                  'maxlen1': maxlen1,
                  'use_wav2' : use_wav2,
              ### 사용할 피쳐 지정
                  'trim' : trim, # 앞뒤 얼마나 자를지? 4000 이면 1초
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
    
    # use_mel = params_feature['use_mel']
    # use_cqt = params_feature['use_cqt']
    # use_stft = params_feature['use_stft']
    nep = 100

    
    features_trn, mm_lbs_trn, out_lbs_trn, mel_input_shape, cqt_input_shape, stft_input_shape,interval_input_shape, wav2_input_shape = get_features_3lb_all_ord(data_folder, patient_files, **params_feature)
    # features_test, mm_lbs_test, out_lbs_test, _, _, _,interval_input_shape,wav2_input_shape = get_features_3lb_all_ord(test_folder, patient_files_test, **params_feature)

    
    # Train the model.
    if verbose >= 1:
        print('Training model...')
        
    model1 = get_LCNN_o_4_dr_1(mel_input_shape, cqt_input_shape, stft_input_shape, interval_input_shape,wav2_input_shape,
                           use_mel = use_mel, use_cqt = use_cqt, use_stft = use_stft, ord1 = ord1, dp = dp, fc = fc, ext = ext)
    model2 = get_LCNN_o_4_dr_1(mel_input_shape, cqt_input_shape, stft_input_shape, interval_input_shape,wav2_input_shape,
                           use_mel = use_mel, use_cqt = use_cqt, use_stft = use_stft, ord1 = ord1, dp = dp, fc = fc, ext = ext)
    
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

    TrainDGen_1 = Generator0([features_trn['age'],features_trn['sex'], features_trn['hw'], features_trn['preg'], features_trn['loc'], 
                                features_trn['interval'], features_trn['wav2'],
                          features_trn['mel1'],features_trn['cqt1'],features_trn['stft1']], 
                         mm_lbs_trn,  ## our Y
                             **params)()
    model1.fit(TrainDGen_1,
            #    validation_data = ([features_test['age'],features_test['sex'], features_test['hw'], 
            #                        features_test['preg'], features_test['loc'], 
            #                        features_test['interval'],features_test['wav2'],
            #                        features_test['mel1'], features_test['cqt1'], features_test['stft1']], 
            #                       mm_lbs_test), 
               callbacks=[lr],
               steps_per_epoch=np.ceil(len(mm_lbs_trn)/batch_size),
               class_weight=class_weight, 
               epochs = n_epoch)


    
    
    
    n_epoch = nep
    lr = LearningRateScheduler(lambda e: sigmoidal_decay(e, end=n_epoch))
    batch_size = 64


    params = {'batch_size': batch_size,
          #          'input_shape': (100, 313, 1),
          'shuffle': True,
          'chaug': 0,
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

    TrainDGen_1 = Generator0([features_trn['age'],features_trn['sex'], features_trn['hw'], features_trn['preg'], features_trn['loc'],
                              features_trn['interval'], features_trn['wav2'],
                          features_trn['mel1'],features_trn['cqt1'],features_trn['stft1']], 
                         out_lbs_trn,  ## our Y
                         **params)()

    model2.fit(TrainDGen_1,
            #    validation_data = ([features_test['age'],features_test['sex'], features_test['hw'], 
            #                        features_test['preg'], features_test['loc'], 
            #                        features_test['interval'],features_test['wav2'],
            #                        features_test['mel1'], 
            #                        features_test['cqt1'], features_test['stft1']], 
            #                       out_lbs_test), 
               callbacks=[lr],
               steps_per_epoch=np.ceil(len(out_lbs_trn)/batch_size),
               class_weight=class_weight, 
               epochs = n_epoch)

    
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

    params_feature['use_mel'] = use_mel
    params_feature['use_cqt'] = use_cqt
    params_feature['use_stft'] = use_stft

    params_feature['interval_input_shape'] = interval_input_shape
    # params_feature['raw1_input_shape'] = raw1_input_shape
    params_feature['wav2_input_shape'] = wav2_input_shape
    params_feature['max_interval_len'] = max_interval_len


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
    
    # Save the model.
    save_challenge_model(model_folder, model1, model2, m_name1 = 'lcnn1_dr', m_name2 = 'lcnn2_dr', param_feature = params_feature)
    if verbose >= 1:
        print('Done.')



# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    info_fnm = os.path.join(model_folder, 'desc.pk')
    with open(info_fnm, 'rb') as f:
        info_m = pk.load(f)
    return info_m

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    
    murmur_classes = ['Present', 'Unknown', 'Absent']
    outcome_classes = ['Abnormal', 'Normal']
    
    if model['model1'] == 'lcnn1_dr' :
        model1 = get_LCNN_o_4_dr_1(model['mel_shape'],model['cqt_shape'],model['stft_shape'], 
                                 model['interval_input_shape'],model['wav2_input_shape'],use_wav2=model['use_wav2'],
                                 use_mel = model['use_mel'],use_cqt = model['use_cqt'], use_stft = model['use_stft'], 
                                 ord1 = model['ord1'], dp = model['dp'], fc = model['fc'], ext = model['ext'])
    if model['model2'] == 'lcnn2_dr' :
        model2 = get_LCNN_o_4_dr_1(model['mel_shape'],model['cqt_shape'],model['stft_shape'], 
                               model['interval_input_shape'],model['wav2_input_shape'],use_wav2=model['use_wav2'],
                               use_mel = model['use_mel'], use_cqt = model['use_cqt'], use_stft = model['use_stft'], 
                               dp = model['dp'], fc = model['fc'], ext = model['ext'])
    model1.load_weights(model['model_fnm1'])
    model2.load_weights(model['model_fnm2'])
    

    
    # maxlen1 = params_feature['maxlen1']
    # min_dist = params_feature['min_dist']
    # max_interval_len = params_feature['max_interval_len']
    
    
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
#    use_raw = model['use_raw']
    trim = model['trim']
    
    # wav2vec2
    use_wav2 = model['use_wav2']
    maxlen1 = model['maxlen1']
    features['wav2'] = []
    tmp_wav=[] 
    tmp = []
    
    # RR interval
    use_interval = model['use_interval']
    min_dist = model['min_dist']
    max_interval_len = model['max_interval_len']
    per_sec = model['per_sec']

    features['interval'] = []
    tmp_total_interval = [] 
    
    if use_wav2:

        for i in range(len(recordings)):
            data = recordings[i]
            tmp_wav.append(data)

        padded =pad_sequences(tmp_wav, maxlen=maxlen1, dtype='float64', padding='post', truncating='post', value=0.0)

        padded=np.array(padded, dtype=np.float32)
        for i in range(len(padded)):
            features['wav2'].append(padded[i])

        features['wav2'] = np.array(features['wav2'])
    

       
    if use_interval:


        for i in tqdm.tqdm(range(len(recordings))) :
            
            datos=recordings[i]
            filtros=sio.loadmat('./Filters1')
            tmp_interval = []
            n_samp = len(datos)//per_sec


            try:
                for k in range(n_samp):
                    
                    X = datos[k*per_sec:(k+1)*per_sec]
                    Fs= per_sec
                    Fpa20=filtros['Fpa20'];			        # High pass filter
                    Fpa20=Fpa20[0];					# High pass filter
                    Fpb100=filtros['Fpb100'];		        # Low-pass Filter
                    Fpb100=Fpb100[0];				# Low-pass Filter
                    Xf=FpassBand(X,Fpa20,Fpb100); 	                # Apply a passband filter
                    Xf=vec_nor(Xf);			

                # Derivate of the Signal
                    dX=derivate(Xf);				# Derivate of the signal
                    dX=vec_nor(dX);					# Vector Normalizing
                # Square of the signal
                    dy=np.square(Xf);
                    dy=vec_nor(dy);

                    size=np.shape(Xf)				# Rank or dimension of the array
                    fil=size[0];					# Number of rows

                    positive=np.zeros((1,fil+1));                   # Initializating Positives Values Vector 
                    positive=positive[0];                           # Getting the Vector

                    points=np.zeros((1,fil));                       # Initializating the all Peak Points Vector
                    points=points[0];                               # Getting the point vector

                    peaks=np.zeros((1,fil));                        # Initializating the s1-s1 Peak Vector
                    peaks=peaks[0];                                 # Getting the point vector


                    for i in range(0,fil):
                        if dX[i]>0:
                            positive[i]=1;
                        else:
                            positive[i]=0;

                    for i in range(0,fil):
                        if (positive[i]==1 and positive[i+1]==0):
                            points[i]=Xf[i];
                        else:
                            points[i]=0;

                    indexes=peakutils.indexes(points,thres=0.5/max(points), min_dist=min_dist);
                    lenght=np.shape(indexes)			# Get the length of the index vector		
                    lenght=lenght[0];				# Get the value of the index vector

                    for i in range(0,lenght):
                        p=indexes[i];
                        peaks[p]=points[p];

                    n=np.arange(0,fil);                            # Vector to the X axes (Number of Samples)
                    indexes =indexes+(k*per_sec)    
                    tmp_peaks = np.array(indexes)


                    tmp_interval.extend(tmp_peaks)

                tmp_interval = np.array(tmp_interval)
                tmp_interval = np.diff(tmp_interval)

                tmp_total_interval.append(tmp_interval)


            except:
                print(i)
                tmp_peaks = np.zeros(max_interval_len)
                tmp_total_interval.append(tmp_peaks)

    else :

        tmp_peaks = np.zeros(max_interval_len)
        tmp_total_interval.append(tmp_peaks)       


    if use_interval:
          
        padded =pad_sequences(tmp_total_interval, maxlen=max_interval_len, dtype='float64', padding='post', truncating='post', value=0.0)

        for i in range(len(padded)):
            features['interval'].append(padded[i])
             
        for i in range(len(features['interval'])):
            features['interval'][i]= features['interval'][i].reshape(-1,1)
    
        features['interval']=np.array(features['interval'])
    
        
    else:
        for i in range(len(tmp_interval)):
            features['interval'].append(tmp_total_interval[i])
        
        for i in range(len(features['interval'])):
            features['interval'][i]= features['interval'][i].reshape(-1,1)
        features['interval']=np.array(features['interval'])
        
        
    
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
            mel1 = feature_extract_cqt(recordings[i]/ 32768, samp_sec=samp_sec, pre_emphasis = pre_emphasis, filter_scale = filter_scale, 
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

    #    print(features)
    # Impute missing data.
    res1 = model1.predict([features['age'], features['sex'], features['hw'], features['preg'], features['loc'], 
#                            features['interval'],
                           features['interval'],features['wav2'],
                           features['mel1'], features['cqt1'], features['stft1']])
    
    res2 = model2.predict([features['age'], features['sex'], features['hw'], features['preg'], features['loc'], 
#                            features['interval'],
                           features['interval'],features['wav2'],
                           features['mel1'], features['cqt1'], features['stft1']])

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

    # Get classifier probabilities.
#    idx1 = res1.argmax(axis=0)[0]
#    murmur_probabilities = res1[idx1,]  ## mumur 확률 최대화 되는 애 뽑기
#    outcome_probabilities = res2.mean(axis = 0) ##  outcome 은 그냥 평균으로 뽑기
#    idx = np.argmax(prob1)

    ## 이부분도 생각 필요.. rule 을 cost를 maximize 하는 기준으로 threshold 탐색 필요할지도..
    # Choose label with highest probability.
    murmur_labels = np.zeros(len(murmur_classes), dtype=np.int_)
#    idx = np.argmax(murmur_probabilities)
    if murmur_probabilities[0] > 0.482 :
        idx = 0
    else :
        idx = 2
    murmur_labels[idx] = 1

    outcome_labels = np.zeros(len(outcome_classes), dtype=np.int_)
    if outcome_probabilities[0] > 0.607 :
        idx = 0
    else :
        idx = 1    
        # idx = np.argmax(outcome_probabilities)
    outcome_labels[idx] = 1
    
    # Concatenate classes, labels, and probabilities.
    classes = murmur_classes + outcome_classes
    labels = np.concatenate((murmur_labels, outcome_labels))
    probabilities = np.concatenate((murmur_probabilities, outcome_probabilities))
    
    return classes, labels, probabilities


    

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.

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
