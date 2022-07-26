import os
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
import librosa
import librosa.display
import math
import sys
#sys.path.insert(0,'/home/ikwak2/hmd/notebooks')
#sys.path.insert(0,'/home/ikwak2/hmd/iy_classifier')
sys.path.insert(0,'/Data2/hmd/hmd_sy/evaluation-2022')
sys.path.insert(0,'utils')
from helper_code import *
from get_feature import *
from models import *
from Generator0 import *
import tensorflow as tf
import datetime
from evaluate_model import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    except RuntimeError as e:
        print(e)
# 교수님 서버
# data_folder =  'physionet.org/files/circor-heart-sound/1.0.3/training_data'
# train_folder =  '/home/ubuntu/data/hmd/murmur/train'
# test_folder = '/home/ubuntu/data/hmd/murmur/test'

# docker
data_folder =  'physionet.org/files/circor-heart-sound/1.0.3/training_data'
train_folder =  '/Data2/hmd/data_split/murmur/train'
test_folder = '/Data2/hmd/data_split/murmur/test'
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

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
    if model['model1'] == 'lcnn1_ex' :
        model1 = get_LCNN_o_1_dr(model['mel_shape'],model['cqt_shape'],model['stft_shape'], use_mel = model['use_mel'], use_cqt = model['use_cqt'], use_stft = model['use_stft'], ord1 = model['ord1'], dp = model['dp'], fc = model['fc'], ext = model['ext'])
    if model['model2'] == 'lcnn2_ex' :
        model2 = get_LCNN_2(model['mel_shape'],model['cqt_shape'],model['stft_shape'], use_mel = model['use_mel'], use_cqt = model['use_cqt'], use_stft = model['use_stft'], dp = model['dp'], fc = model['fc'], ext = model['ext'])
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

    features['rr1'] = []
    for i in range(len(recordings)):
        try:
            
#             ____, recording1 = sp.io.wavfile.read(recordings)
            ____, info = nk.ecg_process(recordings[i], sampling_rate=4000)
            rr = np.mean(np.diff(info['ECG_R_Peaks'])/4000)
            current_rr = rr
        except:
            current_rr=np.zeros(1)
        
        features['rr1'].append(current_rr)  
    features['rr1'] = np.array(features['rr1'])

    features['qrs1'] = []
    for i in range(len(recordings)):
        try:
            R_peaks, S_point, Q_point = EKG_QRS_detect1(recordings[i], 4000, True, False)
            qrs = np.mean(S_point-Q_point)
            current_qrs = qrs
        except:
            current_qrs=np.zeros(1)
            
        features['qrs1'].append(current_qrs)  
    features['qrs1'] = np.array(features['qrs1'])
    
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
    res1 = model1.predict([features['age'], features['sex'], features['hw'], features['preg'], features['loc'], features['rr1'], features['qrs1'], features['mel1'], features['cqt1'], features['stft1']])
    res2 = model2.predict([features['age'], features['sex'], features['hw'], features['preg'], features['loc'], features['rr1'], features['qrs1'], features['mel1'], features['cqt1'], features['stft1']])

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
    idx = np.argmax(murmur_probabilities)
    murmur_labels[idx] = 1
    outcome_labels = np.zeros(len(outcome_classes), dtype=np.int_)
    idx = np.argmax(outcome_probabilities)
    outcome_labels[idx] = 1
    
    # Concatenate classes, labels, and probabilities.
    classes = murmur_classes + outcome_classes
    labels = np.concatenate((murmur_labels, outcome_labels))
    probabilities = np.concatenate((murmur_probabilities, outcome_probabilities))
    
    return classes, labels, probabilities




model_folder = 'lcnn_rr_qrs_1'
# 교수님 서버
# output_folder = '/home/ubuntu/hmd/sy_classifier/notebooks_sy/tmp/out_lcnn_rr_qrs_1'

# docker
output_folder = '/Data2/hmd/hmd_sy/notebooks/tmp/out_lcnn_rr_qrs_1'

winlen = 256
hoplen = 128
nmel = 100

trim = 0
use_cqt = True #np.random.choice([True,False])
use_stft = True #np.random.choice([True,False])
use_rr = True
use_qrs = True

params_feature = {'samp_sec': 20,
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
                      'use_mel' : True,
                      'use_cqt' : use_cqt,
                      'use_stft' : use_stft,
                  'use_rr': use_rr,
                  'use_qrs': use_qrs
}
    
    
if ord1 :
    features_trn, mm_lbs_trn, out_lbs_trn, mel_input_shape, cqt_input_shape, stft_input_shape = get_features_3lb_all_ord(train_folder, patient_files_trn, **params_feature)
    features_test, mm_lbs_test, out_lbs_test, _, _, _ = get_features_3lb_all_ord(test_folder, patient_files_test, **params_feature)
else :
    features_trn, mm_lbs_trn, out_lbs_trn, mel_input_shape, cqt_input_shape, stft_input_shape = get_features_3lb_all_jk(train_folder, patient_files_trn, **params_feature)
    features_test, mm_lbs_test, out_lbs_test, _, _, _ = get_features_3lb_all_jk(test_folder, patient_files_test, **params_feature)


#data = [ features_trn, mm_lbs_trn, out_lbs_trn, mel_input_shape, cqt_input_shape, stft_input_shape, features_test, mm_lbs_test, out_lbs_test] 
    
#with open('data256.pickle', 'wb') as f:
#     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

with open('data256.pickle', 'rb') as f:
    data = pickle.load(f)   

features_trn, mm_lbs_trn, out_lbs_trn, mel_input_shape, cqt_input_shape, stft_input_shape, features_test, mm_lbs_test, out_lbs_test = data
    



mm_weight = 3 #np.random.choice([2,3,4,5])
oo_weight = 3 #np.random.choice([2,3,4,5])
ord1 = False #np.random.choice([True,False])
mm_mean = False #np.random.choice([True,False])
dp = False #np.random.choice([.3,.4,.5,.6])
fc = False #np.random.choice([True,False])
ext = False #np.random.choice([True,False])
beta_param = 0.5#np.random.choice([.7, .5, .3])
n1 = 2#np.random.choice([2,3])
ranfil = [n1, [18,19,20,21,22,23]]
hp1 = True #np.random.choice([True, False])
nep = 100 #np.random.choice([50,80,100,120])

use_mel = params_feature['use_mel']
#    use_cqt = params_feature['use_cqt']
#    use_stft = params_feature['use_stft']


params_feature['ord1'] = ord1
params_feature['mm_mean'] = mm_mean
params_feature['dp'] = dp
params_feature['fc'] = fc
params_feature['ext'] = ext
params_feature['oo_weight'] = oo_weight
params_feature['mm_weight'] = mm_weight
params_feature['beta_param'] = beta_param
params_feature['ranfil'] = ranfil
params_feature['hp1'] = hp1
params_feature['nep'] = nep

print(params_feature)
    
#model1 = get_LCNN_o_1_dr(mel_input_shape, cqt_input_shape, stft_input_shape, use_mel = use_mel, use_cqt = use_cqt, use_stft = use_stft, ord1 = ord1, dp = dp, fc = fc, ext = ext)
#model2 = get_LCNN_2_dr(mel_input_shape, cqt_input_shape, stft_input_shape, use_mel = use_mel, use_cqt = use_cqt, use_stft = use_stft, dp = dp, fc = fc, ext = ext)




def get_toy5_1(mel_input_shape, cqt_input_shape, stft_input_shape, use_mel = True, use_cqt = True, use_stft = True, ord1 = True, dp = 0, fc = False, ext = False):
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    rr = keras.Input(shape=(3,), name = 'rr')
    qrs = keras.Input(shape=(1,), name = 'qrs')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
        
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)

    ## rr interval
    rr1 = layers.Dense(2, activation = None)(rr) 

    ## qrs interval embedding    
    qrs1 = layers.Dense(1, activation=None)(qrs)
    
    ## mel embedding
    if use_mel :
        mel2 = layers.Conv2D(16, (3,3), activation = 'relu')(mel1)
        mel2 = layers.MaxPooling2D()(mel2)
        mel2 = layers.Conv2D(32, (5,5), activation = 'relu')(mel2)
        mel2 = layers.MaxPooling2D()(mel2)
        mel2 = layers.Conv2D(32, (3,3), activation = 'relu')(mel2)
        mel2 = layers.MaxPooling2D()(mel2)
        mel2 = layers.Conv2D(64, (3,3), activation = 'relu')(mel2)
        mel2 = layers.MaxPooling2D()(mel2)
        mel2 = layers.Conv2D(64, (3,3), activation = 'relu')(mel2)
        mel2 = layers.Conv2D(64, (1,1), activation = 'relu')(mel2)
        mel2 = layers.MaxPooling2D()(mel2)
        mel2 = layers.Conv2D(64, (1,1), activation = 'relu')(mel2)
        mel2 = layers.GlobalAveragePooling2D()(mel2)
        if dp :
            mel2 = Dropout(dp)(mel2)

    ## cqt embedding
    if use_cqt :
        cqt2 = layers.Conv2D(16, (3,3), activation = 'relu')(cqt1)
        cqt2 = layers.MaxPooling2D()(cqt2)
        cqt2 = layers.Conv2D(32, (5,5), activation = 'relu')(cqt2)
        cqt2 = layers.MaxPooling2D()(cqt2)
        cqt2 = layers.Conv2D(32, (3,3), activation = 'relu')(cqt2)
        cqt2 = layers.MaxPooling2D()(cqt2)
        cqt2 = layers.Conv2D(64, (3,3), activation = 'relu')(cqt2)
        cqt2 = layers.MaxPooling2D()(cqt2)
        cqt2 = layers.GlobalAveragePooling2D()(cqt2)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    ## stft embedding
    if use_stft :
        stft2 = layers.Conv2D(16, (3,3), activation = 'relu')(stft1)
        stft2 = layers.MaxPooling2D()(stft2)
        stft2 = layers.Conv2D(32, (5,5), activation = 'relu')(stft2)
        stft2 = layers.MaxPooling2D()(stft2)
        stft2 = layers.Conv2D(32, (3,3), activation = 'relu')(stft2)
        stft2 = layers.MaxPooling2D()(stft2)
        stft2 = layers.Conv2D(64, (3,3), activation = 'relu')(stft2)
        stft2 = layers.MaxPooling2D()(stft2)
        stft2 = layers.GlobalAveragePooling2D()(stft2)
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
#        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg, rr1])
        concat1 = layers.Concatenate()([ rr1, qrs1])
#        concat1 = layers.Dense(2, activation = 'relu')(concat1)
        concat2 = layers.Concatenate()([concat2, concat1])

    if fc :
        concat2 = layers.Dense(30, activation = "relu")(concat2)
        concat2 = Dropout(dp)(concat2)
        
    if ord1 :
        res1 = layers.Dense(2, activation = "softmax")(concat2)
    else :
        res1 = layers.Dense(3, activation = "softmax")(concat2)
        
#    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,rr,qrs,mel1,cqt1,stft1] , outputs = res1 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)



def get_LCNN_o_1_dr(mel_input_shape, cqt_input_shape, stft_input_shape, use_mel = True, use_cqt = True, use_stft = True, ord1 = True, dp = .5, fc = False, ext = False):
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    rr = keras.Input(shape=(3,), name = 'rr')
    qrs = keras.Input(shape=(1,), name = 'qrs')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)
    
    ## rr interval embedding    
    rr1 = layers.Dense(2, activation=None)(rr)
    
    ## qrs interval embedding    
    qrs1 = layers.Dense(1, activation=None)(qrs)

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
        mel2 = Dropout(dp)(mel2)

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
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg,rr1,qrs1])
        concat1 = layers.Dense(2, activation = 'relu')(concat1)
#        concat1 = rr1
        concat2 = layers.Concatenate()([concat2, concat1])
        
    if fc :
        concat2 = layers.Dense(10, activation = "relu")(concat2)
        concat2 = Dropout(dp)(concat2)
        
    if ord1 :
        res1 = layers.Dense(2, activation = "softmax")(concat2)
    else :
        res1 = layers.Dense(3, activation = "softmax")(concat2)

        
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,rr,qrs,mel1,cqt1,stft1] , outputs = res1 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])

    return(model)





def get_LCNN_o_1_test(mel_input_shape, cqt_input_shape, stft_input_shape, use_mel = True, use_cqt = True, use_stft = True, ord1 = True, dp = .5, fc = False, ext = False):
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    rr = keras.Input(shape=(3,), name = 'rr')
    qrs = keras.Input(shape=(1,), name = 'qrs')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)
    
    ## rr interval embedding    
    rr1 = layers.Dense(2, activation=None)(rr)
    
    ## qrs interval embedding    
    qrs1 = layers.Dense(1, activation=None)(qrs)

    ## mel embedding
    if use_mel :
        
        conv1_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(mel1)
        conv1_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(mel1)
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

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(mfm27)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(mfm27)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        mel2 = layers.GlobalAveragePooling2D()(max28)
        mel2 = Dropout(dp)(mel2)


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
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg, rr1, qrs1])
        concat1 = layers.Dense(2, activation = 'relu')(concat1)
#        concat1 = rr1
        concat2 = layers.Concatenate()([concat2, concat1])
        
    if fc :
        concat2 = layers.Dense(10, activation = "relu")(concat2)
        concat2 = Dropout(dp)(concat2)
        
    if ord1 :
        res1 = layers.Dense(2, activation = "softmax")(concat2)
    else :
        res1 = layers.Dense(3, activation = "softmax")(concat2)

        
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,rr,qrs, mel1,cqt1,stft1] , outputs = res1 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])

    return(model)


use_cqt = False ; use_stft = False
#model1 = get_LCNN_o_1_test(mel_input_shape, cqt_input_shape, stft_input_shape, use_mel = use_mel, use_cqt = use_cqt, use_stft = use_stft, ord1 = ord1, dp = dp, fc = fc, ext = ext)

model1 = get_LCNN_o_1_dr(mel_input_shape, cqt_input_shape, stft_input_shape, use_mel = use_mel, use_cqt = use_cqt, use_stft = use_stft, ord1 = ord1, dp = dp, fc = fc, ext = ext)
model1.summary()
ext = True

beta_param = False 
# cutout 0.8, ext False  0.8574 
# cqt, st, cutout, ext False 0.8526
# stft ext cutout 0.8637

# ext cutout 0.8637 0.8653 0.8605 0.8621
# cutout 0.8 0.8685 
# cutout 0.8 0.8590 

# specaug 0.859
# cutout specaug 0.855
# cqt, stft, cutout, specaug 8637
# stft, cutout, specaug 8621
# stft, cqt, cutout, 8621 8526
# st, cq, cutout, specaug [2, 10], [2, 15] 8621
# st, cq, cutout, specaug [2, 12] [2, 20] 8510
# st, cq, cutout, specaug [2, 12] [2, 20] 8510
# st, cq, cutout, specaug [2, 7] [2, 10] 8526
# st, cutout, specaug 8590


n_epoch = nep
lr = LearningRateScheduler(lambda e: sigmoidal_decay(e, end=n_epoch))
batch_size = 64

params = {'batch_size': batch_size,
              #          'input_shape': (100, 313, 1),
              'shuffle': True,
              'beta_param': beta_param,
#              'mixup': False,
              #          'lowpass': [.5, [11,12,13,14,15,16,17,18]]
              #          'highpass': [.5, [78,79,80,81,82,83,84,85]]
          'cutout': .8,
#          'specaug': [ [2, 7] , [2, 10] ]
#          'ranfilter2' : ranfil
              #           'dropblock' : [30, 100]
              #'device' : device
}

params_no_shuffle = {'batch_size': batch_size,
                         #          'input_shape': (100, 313, 1),
                         'shuffle': False
#                         'beta_param': 0.7,
#                         'mixup': False
                         #'device': device
}


TrainGen = DataGenerator([features_trn['age'],features_trn['sex'], features_trn['hw'], features_trn['preg'], features_trn['loc'], features_trn['rr1'],features_trn['qrs1'],
                          features_trn['mel1'],features_trn['cqt1'],features_trn['stft1']], 
                         mm_lbs_trn,  ## our Y
                         **params)

ValGen = DataGenerator([features_test['age'],features_test['sex'], features_test['hw'], features_test['preg'], features_test['loc'], features_test['rr1'],features_trn['qrs1'],
                          features_test['mel1'],features_test['cqt1'],features_test['stft1']], 
                         mm_lbs_test,  ## our Y
                         **params_no_shuffle)

if ord1 :
    class_weight = {0: mm_weight, 1: 1.}
else :
    class_weight = {0: mm_weight, 1: 1, 2:1.}

model1.fit(TrainGen,
           validation_data = ([features_test['age'],features_test['sex'], features_test['hw'], 
                               features_test['preg'], features_test['loc'], features_test['rr1'],features_test['qrs1'], 
                               features_test['mel1'], 
                               features_test['cqt1'], features_test['stft1']], 
                              mm_lbs_test), 
           callbacks=[lr],
#        steps_per_epoch=np.ceil(len(mm_lbs_trn)/64),
           class_weight=class_weight, 
           epochs = n_epoch)


# test 8558
# test 8542

# test 8605


# test 8700 8669 8526 8526 8447 8510 ## 앞에만 3,3으로 바꾼버전  weight 1:.2 8590 8653 8574 8494 8637 8510

# 512 test 길어진버전  8653 8574 8621 8605

# 256 test 길어진버전  8669 8621 8558 8558

# 원래꺼 8653 8653 8653

# 원래꺼, dp 0.5 8669






use_cqt = False ; use_stft = False; ext = False; nep = 100; beta_param = 0.7


model1 = get_LCNN_o_1_dr(mel_input_shape, cqt_input_shape, stft_input_shape, use_mel = use_mel, use_cqt = use_cqt, use_stft = use_stft, ord1 = ord1, dp = dp, fc = fc, ext = ext)
model1.summary()

n_epoch = nep
lr = LearningRateScheduler(lambda e: sigmoidal_decay(e, end=n_epoch))
batch_size = 64
params = {'batch_size': batch_size,
              #          'input_shape': (100, 313, 1),
              'shuffle': True,
              'beta_param': beta_param,
              'mixup': True,
              #          'lowpass': [.5, [11,12,13,14,15,16,17,18]]
              #          'highpass': [.5, [78,79,80,81,82,83,84,85]]
              'ranfilter2' : ranfil
              #           'dropblock' : [30, 100]
              #'device' : device
}

params_no_shuffle = {'batch_size': batch_size,
                         #          'input_shape': (100, 313, 1),
                         'shuffle': False
#                         'beta_param': 0.7,
#                         'mixup': False
                         #'device': device
}


TrainDGen_1 = Generator0([features_trn['age'],features_trn['sex'], features_trn['hw'], features_trn['preg'], features_trn['loc'], features_trn['rr1'],features_trn['qrs1'],
                          features_trn['mel1'],features_trn['cqt1'],features_trn['stft1']], 
                         mm_lbs_trn,  ## our Y
                         **params)()

if ord1 :
    class_weight = {0: mm_weight, 1: 1.}
else :
    class_weight = {0: mm_weight, 1: 1, 2:1.}
        
model1.fit(TrainDGen_1,
           validation_data = ([features_test['age'],features_test['sex'], features_test['hw'], 
                               features_test['preg'], features_test['loc'], features_test['rr1'], features_test['qrs1'],
                               features_test['mel1'], 
                               features_test['cqt1'], features_test['stft1']], 
                              mm_lbs_test), 
           callbacks=[lr],
        steps_per_epoch=np.ceil(len(mm_lbs_trn)/64),
           class_weight=class_weight, 
           epochs = n_epoch)


## data512,  ext,   0.8590 0.8574

## data512   0.8497


























model1.fit([features_trn['age'],features_trn['sex'], features_trn['hw'], 
                               features_trn['preg'], features_trn['loc'], features_trn['rr1'], features_trn['qrs1'], 
                               features_trn['mel1'], 
                               features_trn['cqt1'], features_trn['stft1']], 
                              mm_lbs_trn,
           validation_data = ([features_test['age'],features_test['sex'], features_test['hw'], 
                               features_test['preg'], features_test['loc'], features_test['rr1'], features_test['qrs1'], 
                               features_test['mel1'], 
                               features_test['cqt1'], features_test['stft1']], 
                              mm_lbs_test), 
           callbacks=[lr],
        steps_per_epoch=np.ceil(len(mm_lbs_trn)/64),
           class_weight=class_weight, 
           epochs = n_epoch)

model1.fit(TrainDGen_1,
           validation_data = ([features_test['age'],features_test['sex'], features_test['hw'], 
                               features_test['preg'], features_test['loc'], features_test['rr1'], features_test['qrs1'], 
                               features_test['mel1'], 
                               features_test['cqt1'], features_test['stft1']], 
                              mm_lbs_test), 
           callbacks=[lr],
        steps_per_epoch=np.ceil(len(mm_lbs_trn)/64),
           class_weight=class_weight, 
           epochs = 10)
















def get_LCNN_2_dr(mel_input_shape, cqt_input_shape, stft_input_shape, use_mel = True, use_cqt = True, use_stft = True, dp = False, fc = False, ext = False):
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    rr = keras.Input(shape=(3,), name = 'rr')
    qrs = keras.Input(shape=(1,), name = 'qrs')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
        
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)

    ## rr interval embedding    
    rr1 = layers.Dense(2, activation=None)(rr)
    
    ## qrs interval embedding    
    qrs1 = layers.Dense(1, activation=None)(qrs)

    if use_mel :
        ## mel embedding
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
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg, rr1,qrs1])
        concat1 = layers.Dense(2, activation = 'relu')(concat1)
        concat2 = layers.Concatenate()([concat2, concat1])
        
    if fc :
        concat2 = layers.Dense(30, activation = 'relu')(concat2)
        if dp :
            concat2 = Dropout(dp)(concat2)

    res1 = layers.Dense(3, activation = "softmax")(concat2)
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc, rr,qrs, mel1,cqt1,stft1] , outputs = res2 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)



def get_LCNN_2_dr(mel_input_shape, cqt_input_shape, stft_input_shape, use_mel = True, use_cqt = True, use_stft = True, dp = .5, fc = False, ext = False):
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    rr = keras.Input(shape=(3,), name = 'rr')
    qrs = keras.Input(shape=(1,), name = 'qrs')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)
    
    ## rr interval embedding    
    rr1 = layers.Dense(2, activation=None)(rr)

    ## qrs interval embedding    
    qrs1 = layers.Dense(1, activation=None)(qrs)
    
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
        mel2 = Dropout(dp)(mel2)

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
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg,rr1,qrs1])
        concat1 = layers.Dense(2, activation = 'relu')(concat1)
#        concat1 = rr1
        concat2 = layers.Concatenate()([concat2, concat1])
        
    if fc :
        concat2 = layers.Dense(10, activation = "relu")(concat2)
        concat2 = Dropout(dp)(concat2)
        
        
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,rr,qrs,mel1,cqt1,stft1] , outputs = res2 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])

    return(model)




def get_LCNN_2(mel_input_shape, cqt_input_shape, stft_input_shape, use_mel = True, use_cqt = True, use_stft = True, dp = False, ext = False, fc = False):
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    rr = keras.Input(shape=(3,), name = 'rr')  
    qrs = keras.Input(shape=(1,), name = 'qrs')  
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
        
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)

    ## rr interval embedding    
    rr1 = layers.Dense(2, activation=None)(rr)

    ## qrs interval embedding    
    qrs1 = layers.Dense(1, activation=None)(qrs)
    
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

        conv23_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        mel2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            mel2 = Dropout(dp)(mel2)

    ## cqt embedding
    if use_cqt :
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

        conv23_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    ## stft embedding
    if use_stft :
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

        conv23_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch25)
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
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg,rr1,qrs1])
        concat1 = layers.Dense(2, activation = 'relu')(concat1)
        concat2 = layers.Concatenate()([concat2, concat1])

        
    if fc :
        concat2 = layers.Dense(10, activation = 'relu')(concat2)
        if dp :
            concat2 = Dropout(dp)(concat2)

    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,rr, qrs,mel1,cqt1,stft1] , outputs = res2 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)


#model2 = get_LCNN_2_dr(mel_input_shape, cqt_input_shape, stft_input_shape, use_mel = use_mel, use_cqt = use_cqt, use_stft = use_stft, dp = dp, fc = fc, ext = ext)

dp = .5
ext = True
hp1 = False

model2 = get_LCNN_2(mel_input_shape, cqt_input_shape, stft_input_shape, use_mel = use_mel, use_cqt = use_cqt, use_stft = use_stft, dp = dp, fc = fc, ext = ext)


n_epoch = nep
lr = LearningRateScheduler(lambda e: sigmoidal_decay(e, end=n_epoch))
batch_size = 64
beta_param = False


params = {'batch_size': batch_size,
              #          'input_shape': (100, 313, 1),
              'shuffle': True,
              'beta_param': beta_param,
#              'mixup': False,
              #          'lowpass': [.5, [11,12,13,14,15,16,17,18]]
              #          'highpass': [.5, [78,79,80,81,82,83,84,85]]
          'cutout': .8,
#          'specaug': [ [2, 7] , [2, 10] ]
#          'ranfilter2' : ranfil
              #           'dropblock' : [30, 100]
              #'device' : device
}

if hp1 :
    params['highpass'] = [.5, [78,79,80,81,82,83,84,85]]

params_no_shuffle = {'batch_size': batch_size,
                         #          'input_shape': (100, 313, 1),
                         'shuffle': False
#                         'beta_param': 0.7,
#                         'mixup': False
                         #'device': device
}


TrainGen = DataGenerator([features_trn['age'],features_trn['sex'], features_trn['hw'], features_trn['preg'], features_trn['loc'], features_trn['rr1'],features_trn['qrs1'],
                          features_trn['mel1'],features_trn['cqt1'],features_trn['stft1']], 
                         out_lbs_trn,  ## our Y
                         **params)

ValGen = DataGenerator([features_test['age'],features_test['sex'], features_test['hw'], features_test['preg'], features_test['loc'], features_test['rr1'],features_test['qrs1'],
                          features_test['mel1'],features_test['cqt1'],features_test['stft1']], 
                         out_lbs_test,  ## our Y
                         **params_no_shuffle)

class_weight = {0: oo_weight, 1: 1.}

model2.fit(TrainGen,
           validation_data = ([features_test['age'],features_test['sex'], features_test['hw'], 
                               features_test['preg'], features_test['loc'], features_test['rr1'], features_test['qrs1'],
                               features_test['mel1'], 
                               features_test['cqt1'], features_test['stft1']], 
                              out_lbs_test), 
           callbacks=[lr],
#        steps_per_epoch=32,
           class_weight=class_weight, 
           epochs = n_epoch)

## 512, ext, hp, cutout .8, dp .5 6133 
## 512, ext, cutout .8, dp .5 6276 6228
## 512, ext, cutout .8, dp .1 6197
## 512, ext, cutout .8, dp 0 6038

## 512, ext, cutout .8, dp 0.6 6228
## 512, ext, cutout .8, dp 0.5
## 512, ext False, cutout .8, dp 0 5975
## 512, ext False, cutout .8, dp 0.5 6197


model_folder

params_feature['ord1'] = ord1
params_feature['mm_mean'] = mm_mean
params_feature['dp'] = dp
params_feature['fc'] = fc
params_feature['ext'] = ext
params_feature['oo_weight'] = oo_weight
params_feature['mm_weight'] = mm_weight
params_feature['beta_param'] = beta_param
params_feature['ranfil'] = ranfil
params_feature['hp1'] = hp1
params_feature['nep'] = nep
params_feature['mel_shape'] = mel_input_shape
params_feature['cqt_shape'] = (1,1,1) #cqt_input_shape
params_feature['stft_shape'] = (1,1,1)#stft_input_shape
params_feature['use_mel'] = use_mel
params_feature['use_cqt'] = use_cqt
params_feature['use_stft'] = use_stft

save_challenge_model(model_folder, model1, model2, m_name1 = 'lcnn1_ex', m_name2 = 'lcnn2_ex', param_feature = params_feature)

output_folder = 'out_lcnn2'

run_model(model_folder, test_folder, output_folder, allow_failures = True, verbose = 2)

data_folder = test_folder ; verbose = 1

model = load_challenge_model(model_folder, verbose) ### Teams: Implement this function!!!

# Find the patient data files.
patient_files = find_patient_files(data_folder)
num_patient_files = len(patient_files)

patient_data = load_patient_data(patient_files[0])
recordings = load_recordings(data_folder, patient_data)
classes, labels, probabilities = run_challenge_model(model, patient_data, recordings, verbose) ### Teams: Implement this function!!!


data = patient_data 

murmur_classes = ['Present', 'Unknown', 'Absent']
outcome_classes = ['Abnormal', 'Normal']

if model['model1'] == 'lcnn1_ex' :
    model1 = get_LCNN_o_1_dr(model['mel_shape'],model['cqt_shape'],model['stft_shape'], use_mel = model['use_mel'], use_cqt = model['use_cqt'], use_stft = model['use_stft'], ord1 = model['ord1'], dp = model['dp'], fc = model['fc'], ext = model['ext'])
if model['model2'] == 'lcnn2_ex' :
    model2 = get_LCNN_2(model['mel_shape'],model['cqt_shape'],model['stft_shape'], use_mel = model['use_mel'], use_cqt = model['use_cqt'], use_stft = model['use_stft'], dp = model['dp'], fc = model['fc'], ext = model['ext'])
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

features['rr1'] = []
for i in range(len(recordings)):
    try:
            
#             ____, recording1 = sp.io.wavfile.read(recordings)
        ____, info = nk.ecg_process(recordings[i], sampling_rate=4000)
        rr = np.mean(np.diff(info['ECG_R_Peaks'])/4000)
        current_rr = rr
    except:
        current_rr=np.zeros(1)
        
    features['rr1'].append(current_rr)  
features['rr1'] = np.array(features['rr1'])


features['qrs1'] = []
for i in range(len(recordings)):
    try:
        R_peaks, S_point, Q_point = EKG_QRS_detect1(recordings[i], 4000, True, False)
        qrs = np.mean(S_point-Q_point)
        current_qrs = qrs
    except:
        current_qrs=np.zeros(1)
        
    features['qrs1'].append(current_qrs)  
features['qrs1'] = np.array(features['qrs1'])

    
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
res1 = model1.predict([features['age'], features['sex'], features['hw'], features['preg'], features['loc'], features['rr1'],  features['qrs1'], features['mel1'], features['cqt1'], features['stft1']])
res2 = model2.predict([features['age'], features['sex'], features['hw'], features['preg'], features['loc'], features['rr1'],  features['qrs1'], features['mel1'], features['cqt1'], features['stft1']])

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
    idx = np.argmax(murmur_probabilities)
    murmur_labels[idx] = 1
    outcome_labels = np.zeros(len(outcome_classes), dtype=np.int_)
    idx = np.argmax(outcome_probabilities)
    outcome_labels[idx] = 1
    
    # Concatenate classes, labels, and probabilities.
    classes = murmur_classes + outcome_classes
    labels = np.concatenate((murmur_labels, outcome_labels))
    probabilities = np.concatenate((murmur_probabilities, outcome_probabilities))
    
    return classes, labels, probabilities







            

















n_epoch = nep
lr = LearningRateScheduler(lambda e: sigmoidal_decay(e, end=n_epoch))
batch_size = 64

params = {'batch_size': batch_size,
          #          'input_shape': (100, 313, 1),
          'shuffle': True,
          'beta_param': beta_param,
          'mixup': True,
          #          'lowpass': [.5, [11,12,13,14,15,16,17,18]]
          #            'highpass': [.5, [78,79,80,81,82,83,84,85]],
          'ranfilter2' : ranfil
          #           'dropblock' : [30, 100]
          #'device' : device
}

params_no_shuffle = {'batch_size': batch_size,
                     #          'input_shape': (100, 313, 1),
                     'shuffle': False,
                     'beta_param': False,
                    'mixup': False
                     #'device': device
}

if hp1 :
    params['highpass'] = [.5, [78,79,80,81,82,83,84,85]]



TrainDGen_1 = Generator0([features_trn['age'],features_trn['sex'], features_trn['hw'], features_trn['preg'], features_trn['loc'], 
                          features_trn['rr1'],features_trn['qrs1'],
                          features_trn['mel1'],features_trn['cqt1'],features_trn['stft1']], 
                         out_lbs_trn,  ## our Y
                         **params)()

class_weight = {0: oo_weight, 1: 1.}
        
model2.fit(TrainDGen_1,
           validation_data = ([features_test['age'],features_test['sex'], features_test['hw'], 
                               features_test['preg'], features_test['loc'], features_test['rr1'], features_test['qrs1'],
                               features_test['mel1'], 
                               features_test['cqt1'], features_test['stft1']], 
                              out_lbs_test), 
           callbacks=[lr],
#        steps_per_epoch=np.ceil(len(out_lbs_trn)/64),
           steps_per_epoch=32,
           class_weight=class_weight, 
           epochs = n_epoch)












n_epoch = 100
lr = LearningRateScheduler(lambda e: sigmoidal_decay(e, end=n_epoch))
beta_param = 0.7

batch_size = 64
params = {'batch_size': batch_size,
#          'input_shape': (100, 313, 1),
          'shuffle': True,
          'beta_param': beta_param,
          'mixup': True,
#          'lowpass': [.5, [11,12,13,14,15,16,17,18]]
          'highpass': [.5, [78,79,80,81,82,83,84,85]],
          'ranfilter2' : [3, [18,19,20,21,22,23]]
#           'dropblock' : [30, 100]
          #'device' : device
}

params_no_shuffle = {'batch_size': batch_size,
#          'input_shape': (100, 313, 1),
          'shuffle': False,
#          'beta_param': 0.7,
          'mixup': False
          #'device': device
}

TrainDGen_1 = Generator0([features_trn['age'],features_trn['sex'], features_trn['hw'], features_trn['preg'], features_trn['loc'],
                          features_trn['rr1'],features_trn['qrs1'], features_trn['mel1'],features_trn['stft1'],features_trn['cqt1']], 
                        out_lbs_trn,  ## our Y
                        **params)()

#ValDGen_1 = Generator0([features_test[0]['age'],features_test[0]['sex'], features_test[0]['hw'], features_test[0]['preg'], features_test[0]['loc'], 
#           features_test[0]['mel1'],features_test[0]['cqt1'],features_test[0]['stft1']], 
#                        features_test[2],  ## our Y
#                        **params_no_shuffle)()

    
model2.fit(TrainDGen_1,
          validation_data = ([features_test['age'],features_test['sex'], features_test['hw'], 
                              features_test['preg'], features_test['loc'],features_test['rr1'],features_test['qrs1'],
                              features_test['mel1'],
                              features_test['stft1'],features_test['cqt1']], 
                             out_lbs_test), 
                             callbacks=[lr],
                              steps_per_epoch=32,
                             epochs = n_epoch)







TrainDGen_1 = Generator0([features_trn['age'],features_trn['sex'], features_trn['hw'], features_trn['preg'], features_trn['loc'], 
                          features_trn['rr1'], features_trn['qrs1'],
                          features_trn['mel1'],features_trn['cqt1'],features_trn['stft1']], 
                         out_lbs_trn,  ## our Y
                         **params)()

class_weight = {0: oo_weight, 1: 1.}
    
model2.fit(TrainDGen_1,
           validation_data = ([features_test['age'],features_test['sex'], features_test['hw'], 
                               features_test['preg'], features_test['loc'], features_test['rr1'],  features_test['qrs1'],
                               features_test['mel1'], features_test['cqt1'], features_test['stft1']], 
                             out_lbs_test), 
                             callbacks=[lr],
                              steps_per_epoch=np.ceil(len(out_lbs_trn)/64),
                             epochs = n_epoch)


## mixup, RF






































###################### lcnn_dr random search

for i in range(50000) :
    model_folder = 'lcnn_dr_qrs_1'
    # output_folder = '/home/ubuntu/hmd/notebooks/tmp/out_lcnn_dr_qrs_1'
    output_folder = '/Data2/hmd/hmd_sy/notebooks/tmp/out_lcnn_rr_qrs_1'

    winlen = 512
    hoplen = 256
    nmel = 100
#    nmel = int(nmel)

    trim = 0
    use_cqt = False #np.random.choice([True,False])
    use_stft = False #np.random.choice([True,False])
    
    params_feature = {'samp_sec': 20,
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
                      'use_mel' : True,
                      'use_cqt' : use_cqt,
                      'use_stft' : use_stft
    }
    mm_weight = 3 #np.random.choice([2,3,4,5])
    oo_weight = 3 #np.random.choice([2,3,4,5])
    ord1 = True #np.random.choice([True,False])
    mm_mean = False #np.random.choice([True,False])
    dp = False #np.random.choice([.3,.4,.5,.6])
    fc = False #np.random.choice([True,False])
    ext = False #np.random.choice([True,False])
    beta_param = np.random.choice([.7, .5, .3])
    n1 = np.random.choice([2,3])
    ranfil = [n1, [18,19,20,21,22,23]]
    hp1 = np.random.choice([True, False])
    nep = np.random.choice([50,80,100,120])
    
    use_mel = params_feature['use_mel']
#    use_cqt = params_feature['use_cqt']
#    use_stft = params_feature['use_stft']
    
    
    if ord1 :
        features_trn, mm_lbs_trn, out_lbs_trn, mel_input_shape, cqt_input_shape, stft_input_shape = get_features_3lb_all_ord(train_folder, patient_files_trn, **params_feature)
        features_test, mm_lbs_test, out_lbs_test, _, _, _ = get_features_3lb_all_ord(test_folder, patient_files_test, **params_feature)
    else :
        features_trn, mm_lbs_trn, out_lbs_trn, mel_input_shape, cqt_input_shape, stft_input_shape = get_features_3lb_all(train_folder, patient_files_trn, **params_feature)
        features_test, mm_lbs_test, out_lbs_test, _, _, _ = get_features_3lb_all(test_folder, patient_files_test, **params_feature)

    params_feature['ord1'] = ord1
    params_feature['mm_mean'] = mm_mean
    params_feature['dp'] = dp
    params_feature['fc'] = fc
    params_feature['ext'] = ext
    params_feature['oo_weight'] = oo_weight
    params_feature['mm_weight'] = mm_weight
    params_feature['beta_param'] = beta_param
    params_feature['ranfil'] = ranfil
    params_feature['hp1'] = hp1
    params_feature['nep'] = nep

    print(params_feature)
    
    model1 = get_LCNN_o_1_dr(mel_input_shape, cqt_input_shape, stft_input_shape, use_mel = use_mel, use_cqt = use_cqt, use_stft = use_stft, ord1 = ord1, dp = dp, fc = fc, ext = ext)
    model2 = get_LCNN_2_dr(mel_input_shape, cqt_input_shape, stft_input_shape, use_mel = use_mel, use_cqt = use_cqt, use_stft = use_stft, dp = dp, fc = fc, ext = ext)

    n_epoch = nep
    lr = LearningRateScheduler(lambda e: sigmoidal_decay(e, end=n_epoch))
    batch_size = 64
    params = {'batch_size': batch_size,
              #          'input_shape': (100, 313, 1),
              'shuffle': True,
              'beta_param': beta_param,
              'mixup': True,
              #          'lowpass': [.5, [11,12,13,14,15,16,17,18]]
              #          'highpass': [.5, [78,79,80,81,82,83,84,85]]
              'ranfilter2' : ranfil
              #           'dropblock' : [30, 100]
              #'device' : device
    }

    params_no_shuffle = {'batch_size': batch_size,
                         #          'input_shape': (100, 313, 1),
                         'shuffle': False,
                         'beta_param': 0.7,
                         'mixup': False
                         #'device': device
    }

    TrainDGen_1 = Generator0([features_trn['age'],features_trn['sex'], features_trn['hw'], features_trn['preg'], features_trn['loc'], 
                              features_trn['mel1'],features_trn['cqt1'],features_trn['stft1']], 
                             mm_lbs_trn,  ## our Y
                             **params)()

    if ord1 :
        class_weight = {0: mm_weight, 1: 1.}
    else :
        class_weight = {0: mm_weight, 1: .7, 2:1.}
    
    model1.fit(TrainDGen_1,
          validation_data = ([features_test['age'],features_test['sex'], features_test['hw'], 
                              features_test['preg'], features_test['loc'], features_test['mel1'], 
                              features_test['cqt1'], features_test['stft1']], 
                             mm_lbs_test), 
                             callbacks=[lr],
                              steps_per_epoch=np.ceil(len(mm_lbs_trn)/64),
                           class_weight=class_weight, 
                             epochs = n_epoch)

    n_epoch = nep
    lr = LearningRateScheduler(lambda e: sigmoidal_decay(e, end=n_epoch))
    batch_size = 64

    params = {'batch_size': batch_size,
              #          'input_shape': (100, 313, 1),
              'shuffle': True,
              'beta_param': beta_param,
              'mixup': True,
              #          'lowpass': [.5, [11,12,13,14,15,16,17,18]]
#            'highpass': [.5, [78,79,80,81,82,83,84,85]],
              'ranfilter2' : ranfil
            #           'dropblock' : [30, 100]
              #'device' : device
    }

    params_no_shuffle = {'batch_size': batch_size,
                         #          'input_shape': (100, 313, 1),
                         'shuffle': False,
                         'beta_param': 0.7,
                         'mixup': False
                         #'device': device
    }

    if hp1 :
        params['highpass'] = [.5, [78,79,80,81,82,83,84,85]]

    TrainDGen_1 = Generator0([features_trn['age'],features_trn['sex'], features_trn['hw'], features_trn['preg'], features_trn['loc'], 
                              features_trn['mel1'],features_trn['cqt1'],features_trn['stft1']], 
                             out_lbs_trn,  ## our Y
                             **params)()

    if ord1 :
        class_weight = {0: oo_weight, 1: 1.}
    else :
        class_weight = {0: oo_weight, 1: .7, 2:1.}
    
    model2.fit(TrainDGen_1,
          validation_data = ([features_test['age'],features_test['sex'], features_test['hw'], 
                              features_test['preg'], features_test['loc'], features_test['mel1'], 
                              features_test['cqt1'], features_test['stft1']], 
                             out_lbs_test), 
                             callbacks=[lr],
                              steps_per_epoch=np.ceil(len(out_lbs_trn)/64),
                             epochs = n_epoch)


    params_feature['mel_shape'] = mel_input_shape
    params_feature['cqt_shape'] = cqt_input_shape
    params_feature['stft_shape'] = stft_input_shape

    params_feature['use_mel'] = use_mel
    params_feature['use_cqt'] = use_cqt
    params_feature['use_stft'] = use_stft
    
    save_challenge_model(model_folder, model1, model2, m_name1 = 'lcnn1_dr', m_name2 = 'lcnn2_dr', param_feature = params_feature)
    
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
    for th1 in [0.01, 0.05, 0.1, 0.15,0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8] :
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
    params_feature['min_cost'] = min_cost
    params_feature['max_th'] = max_th
    params_feature['min_th'] = min_th

    tnow = datetime.datetime.now()
    fnm = 'res6/rec'+ str(tnow)+'.pk'

    print(params_feature)
    
    with open(fnm, 'wb') as f:
        pickle.dump(params_feature, f, pickle.HIGHEST_PROTOCOL)


        
 
















