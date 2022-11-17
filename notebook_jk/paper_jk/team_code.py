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
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

#data_folder = '/home/ubuntu/data/hmd/murmur/train'
#model_folder = '/home/ubuntu/physionet_cau_umn/mod1'
#verbose = 1
#import tensorflow as tf
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_visible_devices(gpus[1], 'GPU')

#data_folder = '/home/ubuntu/data/hmd/murmur/test'
#output_folder = '/home/ubuntu/physionet_cau_umn/out1'

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

        
    winlen = 512
    hoplen = 256
    nmel = 120 #np.random.choice([100, 120, 140])
    nsec = 20 #np.random.choice([10, 20, 30, 40, 50])

    trim = 0 #np.random.choice([0,2000, 4000])
    use_cqt = False #np.random.choice([True,False])
    use_stft = False#np.random.choice([True, False])

    params_feature = {'samp_sec': nsec,
              #### melspec, stft   
              'pre_emphasis': 0,
              'hop_length': hoplen,
              'win_length':winlen,
              'n_mels': nmel,
              #### cqt   
              'filter_scale': 1,
              'n_bins': 80,
                  'fmin': 10,
                ### 
                  'trim' : trim,
                  
                      'use_rr' : True,
                  'use_mel' : True,
                  'use_cqt' : use_cqt,
                  'use_stft' : use_stft
    }
    mm_weight = 3 #np.random.choice([2,3,4,5])
    oo_weight = 3 #np.random.choice([2,3,4,5,6])
    ord1 = True #np.random.choice([True,False])
    mm_mean = False #np.random.choice([True,False])
    dp = 0 #np.random.choice([0, .1, .2, .3])
    fc = False #np.random.choice([True,False])
    ext = False
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

    patient_files_trn = find_patient_files(data_folder)



    if ord1 :
        features_trn, mm_lbs_trn, out_lbs_trn, mel_input_shape, cqt_input_shape, stft_input_shape = get_features_3lb_all_ord_rr(data_folder, patient_files_trn, **params_feature)
#        features_test, mm_lbs_test, out_lbs_test, _, _, _ = get_features_3lb_all_ord(test_folder, patient_files_test, **params_feature)
    else :
        features_trn, mm_lbs_trn, out_lbs_trn, mel_input_shape, cqt_input_shape, stft_input_shape = get_features_3lb_all(data_folder, patient_files_trn, **params_feature)
#        features_test, mm_lbs_test, out_lbs_test, _, _, _ = get_features_3lb_all(test_folder, patient_files_test, **params_feature)

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


    # Train the model.
    if verbose >= 1:
        print('Training model...')
        
    model1 = get_LCNN_o_1_dr_rr(mel_input_shape, cqt_input_shape, stft_input_shape, use_mel = use_mel, use_cqt = use_cqt, use_stft = use_stft, ord1 = ord1, dp = dp, fc = fc, ext = False, ext2 = True)
    model2 = get_LCNN_2_dr_rr(mel_input_shape, cqt_input_shape, stft_input_shape, use_mel = use_mel, use_cqt = use_cqt, use_stft = use_stft, dp = dp, fc = fc, ext = True, ext2 = False)
        
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
        TrainDGen_1 = Generator0([features_trn['age'],features_trn['sex'], features_trn['hw'], features_trn['preg'], features_trn['loc'], 
                          features_trn['mel1'],features_trn['cqt1'],features_trn['stft1'],features_trn['rr1']], 
                         mm_lbs_trn,  ## our Y
                             **params)()
        model1.fit(TrainDGen_1,
#               validation_data = ([features_test['age'],features_test['sex'], features_test['hw'], 
#                                   features_test['preg'], features_test['loc'], features_test['mel1'], 
#                                   features_test['cqt1'], features_test['stft1']], 
#                                  mm_lbs_test), 
               callbacks=[lr],
               steps_per_epoch=np.ceil(len(mm_lbs_trn)/64),
               class_weight=class_weight, 
               epochs = n_epoch)

    else :
        TrainGen = DataGenerator([features_trn['age'],features_trn['sex'], features_trn['hw'], features_trn['preg'], features_trn['loc'], 
                      features_trn['mel1'],features_trn['cqt1'],features_trn['stft1'],features_trn['rr1']], 
                     mm_lbs_trn,  ## our Y
                     **params)
        model1.fit(TrainGen,
#               validation_data = ([features_test['age'],features_test['sex'], features_test['hw'], 
#                                   features_test['preg'], features_test['loc'], features_test['mel1'], 
#                                   features_test['cqt1'], features_test['stft1']], 
#                                  mm_lbs_test), 
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

    if mixup :
        TrainDGen_1 = Generator0([features_trn['age'],features_trn['sex'], features_trn['hw'], features_trn['preg'], features_trn['loc'], 
                          features_trn['mel1'],features_trn['cqt1'],features_trn['stft1'],features_trn['rr1']], 
                         out_lbs_trn,  ## our Y
                         **params)()

        model2.fit(TrainDGen_1,
               callbacks=[lr],
               steps_per_epoch=np.ceil(len(out_lbs_trn)/64),
               class_weight=class_weight, 
               epochs = n_epoch)
    else :
        TrainGen = DataGenerator([features_trn['age'],features_trn['sex'], features_trn['hw'], features_trn['preg'], features_trn['loc'], 
                      features_trn['mel1'],features_trn['cqt1'],features_trn['stft1'],features_trn['rr1']], 
                     out_lbs_trn,  ## our Y
                     **params)
        model2.fit(TrainGen,
               callbacks=[lr],
               class_weight=class_weight, 
               epochs = n_epoch)
    

    params_feature['mel_shape'] = mel_input_shape
    params_feature['cqt_shape'] = cqt_input_shape
    params_feature['stft_shape'] = stft_input_shape

    params_feature['use_mel'] = use_mel
    params_feature['use_cqt'] = use_cqt
    params_feature['use_stft'] = use_stft
    
    # Save the model.
    save_challenge_model(model_folder, model1, model2, m_name1 = 'lcnn1_dr_rr', m_name2 = 'lcnn2_dr_rr', param_feature = params_feature)
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

    if model['model1'] == 'toy1' :
        model1 = get_toy5_1(model['mel_shape'],model['cqt_shape'],model['stft_shape'] )
    if model['model2'] == 'toy2' :
        model2 = get_toy5_2(model['mel_shape'],model['cqt_shape'],model['stft_shape'])
    if model['model1'] == 'lcnn1' :
        model1 = get_LCNN_o_1(model['mel_shape'],model['cqt_shape'],model['stft_shape'], use_mel = model['use_mel'], use_cqt = model['use_cqt'], use_stft = model['use_stft'], ord1 = model['ord1'])
    if model['model2'] == 'lcnn2' :
        model2 = get_LCNN_2(model['mel_shape'],model['cqt_shape'],model['stft_shape'], use_mel = model['use_mel'], use_cqt = model['use_cqt'], use_stft = model['use_stft'])
    if model['model1'] == 'resmax1' :
        model1 = get_ResMax_o_1(model['mel_shape'],model['cqt_shape'],model['stft_shape'], use_mel = model['use_mel'], use_cqt = model['use_cqt'], use_stft = model['use_stft'], ord1 = model['ord1'])
    if model['model2'] == 'resmax2' :
        model2 = get_ResMax_2(model['mel_shape'],model['cqt_shape'],model['stft_shape'], use_mel = model['use_mel'], use_cqt = model['use_cqt'], use_stft = model['use_stft'])
    if model['model1'] == 'lcnn1_dr' :
        model1 = get_LCNN_o_1_dr(model['mel_shape'],model['cqt_shape'],model['stft_shape'], use_mel = model['use_mel'], use_cqt = model['use_cqt'], use_stft = model['use_stft'], ord1 = model['ord1'], dp = model['dp'], fc = model['fc'], ext = model['ext'])
    if model['model2'] == 'lcnn2_dr' :
        model2 = get_LCNN_2_dr(model['mel_shape'],model['cqt_shape'],model['stft_shape'], use_mel = model['use_mel'], use_cqt = model['use_cqt'], use_stft = model['use_stft'], dp = model['dp'], fc = model['fc'], ext = model['ext'])
    if model['model1'] == 'lcnn1_dr_rr' :
        model1 = get_LCNN_o_1_dr_rr(model['mel_shape'],model['cqt_shape'],model['stft_shape'], use_mel = model['use_mel'], use_cqt = model['use_cqt'], use_stft = model['use_stft'], ord1 = model['ord1'], dp = model['dp'], fc = model['fc'], ext = False, ext2 = True)
    if model['model2'] == 'lcnn2_dr_rr' :
        model2 = get_LCNN_2_dr_rr(model['mel_shape'],model['cqt_shape'],model['stft_shape'], use_mel = model['use_mel'], use_cqt = model['use_cqt'], use_stft = model['use_stft'], dp = model['dp'], fc = model['fc'], ext = True, ext2 = False)
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
#    use_raw = model['use_raw']
    trim = model['trim']
    use_rr = model['use_rr']

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
    
    #    print(features)
    # Impute missing data.
    res1 = model1.predict([features['age'], features['sex'], features['hw'], features['preg'], features['loc'], features['mel1'], features['cqt1'], features['stft1'], features['rr1']])
    res2 = model2.predict([features['age'], features['sex'], features['hw'], features['preg'], features['loc'], features['mel1'], features['cqt1'], features['stft1'], features['rr1']])

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

