import soundfile as sf
import numpy as np
import librosa
from helper_code import *
import math
import os, numpy as np, scipy as sp, scipy.io, scipy.io.wavfile
import neurokit2 as nk
import tqdm
import pywt
import sys
from scipy import special
import scipy.io as sio
from keras.preprocessing.sequence import pad_sequences
sys.path.insert(0,'/home/jk21/Documents/hmd/jk_classifier/lucashnegri-peakutils-51a679cd8428')
import peakutils
from wav2vec2 import Wav2Vec2ForCTC, Wav2Vec2Config
import tensorflow as tf

config = Wav2Vec2Config()
model = Wav2Vec2ForCTC(config)
model.trainable=False

def feature_extract_melspec(fnm, samp_sec=20, sr = 4000, pre_emphasis = 0, hop_length=256, win_length = 512, n_mels = 100, trim = 4000):

    if isinstance(fnm, str) :
        data, sample_rate = librosa.load(fnm, sr = sr)
        data = data[trim:-trim] * 1.0
    else :
        data = fnm[trim:-trim] * 1.0
        sample_rate = sr

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







def feature_extract_stft(fnm, samp_sec=20, sr = 4000, pre_emphasis = 0, hop_length=256, win_length = 512, trim = 4000):

    if isinstance(fnm, str) :
        data, sample_rate = librosa.load(fnm, sr = sr)
        data = data[trim:-trim] * 1.0
    else :
        data = fnm[trim:-trim] * 1.0
        sample_rate = sr

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

#        D_highres = librosa.stft(emphasized_signal, hop_length=hop_length, n_fft=win_length)
#        S_db_hr = librosa.amplitude_to_db(np.abs(D_highres), ref=np.max)
            
        Sig.append(librosa.power_to_db(np.abs(librosa.stft(emphasized_signal, n_fft=win_length, hop_length=hop_length, win_length=win_length))))

    return Sig


def feature_extract_cqt(fnm, samp_sec=20, sr = 4000, pre_emphasis = 0, filter_scale = 1, n_bins = 80, fmin = 10, trim = 4000):

    if isinstance(fnm, str) :
        data, sample_rate = sf.read(fnm, dtype = 'int16')
        data = data[trim:-trim] * 1.0
    else :
        data = fnm[trim:-trim] * 1.0
        sample_rate = sr


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

#        D_highres = librosa.stft(emphasized_signal, hop_length=hop_length, n_fft=win_length)
#        S_db_hr = librosa.amplitude_to_db(np.abs(D_highres), ref=np.max)
        Sig.append(np.log(np.abs(librosa.cqt(emphasized_signal, sr=sample_rate,
                                             filter_scale = filter_scale, n_bins=n_bins,
                                             fmin = fmin))+1) )

    return Sig



def time_vector(sampling_frequency,duration): 
	number_samples= sampling_frequency*duration;
	result=np.arange(1,duration/number_samples,duration-duration/number_samples);

	return result

##----------------------------------------------------------------------------
''' Derivate of an input signal as y[n]= x[n+1]- x[n-1] 
'''
def derivate (x):
	lenght=np.shape(x)				# Get the length of the vector		
	lenght=lenght[0];				# Get the value of the length
	y=np.zeros(lenght);				# Initializate derivate vector
	for i in range(lenght-1):
			y[i]=x[i+1]-x[i];		
	return y

##----------------------------------------------------------------------------
'''To normalized any vector\0-dimentional array in [-1,1] range, by divided the 
   vector by the maximun value of itself, substracting the mean value to the vector
   & dividing each value of the vector by the maximun value of itself 
'''
def vec_nor(x):
	lenght=np.shape(x)				# Get the length of the vector		
	lenght=lenght[0];				# Get the value of the length
	xMax=max(x);					# Get the maximun value of the vector
	nVec=np.zeros(lenght);			        # Initializate derivate vector
	for n in range(lenght):
		nVec[n]=x[n]/xMax;			
	nVec=nVec-np.mean(nVec);
	nVec=np.divide(nVec,np.max(nVec));
	return nVec
##----------------------------------------------------------------------------
'''
  FpassBand is the function that develop a pass band filter of the signal 'x' through the
  discrete convolution of this 'x' first with the coeficients of a High Pass Filter 'hp' and then
  with the discrete convolution of this result with a Low Pass Filter 'lp'
'''
def FpassBand(X,hp,lp):
        llp=np.shape(lp)	  	        # Get the length of the lowpass vector		
        llp=llp[0];				# Get the value of the length
        lhp=np.shape(hp)			# Get the length of the highpass vector		
        lhp=lhp[0];				# Get the value of the length	

        x=np.convolve(X,lp);		        # Disrete convolution 
        x=x[round(llp/2):round(-1-(llp/2))];
        x=x-(np.mean(x));
        x=x/np.max(x);
	
        y=np.convolve(x,hp);			# Disrete onvolution
        y=y[round((lhp/2)):round(-1-(lhp/2))];
        y=y-np.mean(y);
        y=y/np.max(y);

        x=np.convolve(y,lp);		        # Disrete convolution 
        x=x[round((llp/2)):round(-1-(llp/2))];
        x=x-(np.mean(x));
        x=x/np.max(x);
	
        y=np.convolve(x,hp);			# Disrete onvolution
        y=y[round((lhp/2)):round(-1-(lhp/2))];
        y=y-np.mean(y);
        y=y/np.max(y);
        
        y=vec_nor(y);				# Vector Normalizing
        
        return y
     

def get_murmur_loc(data):
    murmur_loc = 0
    for l in data.split('\n'):
        if l.startswith('#Murmur locations: '):
            try:
                murmur_loc = l.split(': ')[1]
#                murmur_loc = murmur_loc.split('+')
            except:
                pass
    return murmur_loc


def get_feature_one(patient_data, verbose = 0) :
    num_locations = get_num_locations(patient_data)
    recording_information = patient_data.split('\n')[1:num_locations+1]

    age_classes = ['Neonate', 'Infant', 'Child', 'Adolescent', 'Young Adult']
    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']

    features = dict()
    features['age'] = []
    features['sex'] = []
    features['hw'] = []
    features['preg'] = []
    features['loc'] = []

#    features['mel1'] = []
    for j in range(num_locations) :
        entries = recording_information[j].split(' ')
        recording_file = entries[2]
        filename = os.path.join(patient_data, recording_file)

        # Extract melspec
#        mel1 = feature_extract_melspec(filename)[0]
#        features['mel1'].append(mel1)

        # Extract age_group
        age_group = get_age(patient_data)
        current_age_group = np.zeros(6, dtype=int)
        if age_group in age_classes:
            j = age_classes.index(age_group)
            current_age_group[j] = 1
        else :
            current_age_group[5] = 1
        features['age'].append(current_age_group)

        # Extract sex
        sex = get_sex(patient_data)
        sex_features = np.zeros(2, dtype=int)
        if compare_strings(sex, 'Female'):
            sex_features[0] = 1
        elif compare_strings(sex, 'Male'):
            sex_features[1] = 1
        features['sex'].append(sex_features)

        # Extract height and weight.
        height = get_height(patient_data)
        weight = get_weight(patient_data)
        ## simple impute
        if math.isnan(height) :
            height = 110.846
        if math.isnan(weight) :
            weight = 23.767

        features['hw'].append(np.array([height, weight]))

        # Extract pregnancy
        is_pregnant = get_pregnancy_status(patient_data)
        features['preg'].append(is_pregnant)

            
        # Extract location
        locations = entries[0]
        num_recording_locations = len(recording_locations)
        loc_features = np.zeros(num_recording_locations)
        if locations in recording_locations:
            j = recording_locations.index(locations)
            loc_features[j] = 1
        features['loc'].append(loc_features)
        
        
#    M, N = features['mel1'][0].shape
#    for i in range(len(features['mel1'])) :
#        features['mel1'][i] = features['mel1'][i].reshape(M,N,1)

    for k1 in features.keys() :
        features[k1] = np.array(features[k1])
        
    if verbose :
        label = get_label(patient_data)
        print(label)
    return features



def get_features_3lb_all(data_folder, patient_files_trn, 
                          samp_sec=20, pre_emphasis = 0, hop_length=256, win_length = 512, n_mels = 100,
                          filter_scale = 1, n_bins = 80, fmin = 10, trim = 4000,max_interval_len=115, maxlen1=120000,min_dist=1000,
                         use_mel = True, use_cqt = False, use_stft= False, use_raw = False,use_interval = False,use_wav2=False
                         ) :
    features = dict()
    features['id'] = []
    features['age'] = []
    features['sex'] = []
    features['hw'] = []
    features['preg'] = []
    features['loc'] = []
    features['mel1'] = []
    features['cqt1'] = []
    features['stft1'] = []
    features['raw1'] = []
    features['interval'] = []
    features['wav2']=[]
#    labels = []
    features['mm_labels'] = []
    features['out_labels'] = []
    tmp_interval=[]
    tmp_wav=[]
    interval_len=[]

    age_classes = ['Neonate', 'Infant', 'Child', 'Adolescent', 'Young Adult']
    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']

    num_patient_files = len(patient_files_trn)

    for i in tqdm.tqdm(range(num_patient_files)):
        

        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files_trn[i])
        num_locations = get_num_locations(current_patient_data)
        recording_information = current_patient_data.split('\n')[1:num_locations+1]
        
        for j in range(num_locations) :
            entries = recording_information[j].split(' ')
            recording_file = entries[2]
            filename = os.path.join(data_folder, recording_file)

            # Extract id
            id1 = recording_file.split('_')[0]
            features['id'].append(id1)

            # Extract melspec
            if use_mel :
                mel1 = feature_extract_melspec(filename, samp_sec=samp_sec, pre_emphasis = pre_emphasis, hop_length=hop_length, 
                                               win_length = win_length, n_mels = n_mels, trim = trim)[0]
            else :
                mel1 = np.zeros( (1,1,1) )
            features['mel1'].append(np.array(mel1))

            if use_cqt :
                mel2 = feature_extract_cqt(filename, samp_sec=samp_sec, pre_emphasis = pre_emphasis, filter_scale = filter_scale, 
                                           n_bins = n_bins, fmin = fmin, trim = trim)[0]
            else :
                mel2 = np.zeros( (1,1,1))                
            features['cqt1'].append(np.array(mel2))

            if use_stft :
                mel3 = feature_extract_stft(filename, samp_sec=samp_sec, pre_emphasis = pre_emphasis, hop_length=hop_length, 
                                            win_length = win_length, trim = trim)[0]
            else :
                mel3 = np.zeros( (1,1,1) )
            features['stft1'].append(np.array(mel3))

            if use_raw :
                recording1,frequency1 = librosa.load(filename)
            else :
                recording1 = np.zeros( (1) )
            features['raw1'].append(recording1)
            
            if use_wav2:
                recording1,frequency1 = librosa.load(filename)
            else :
                recording1 = np.zeros( (1) )
            tmp_wav.append(recording1)                
                
            
            if use_interval :
                               
                datos=sp.io.wavfile.read(filename)
                filtros=sio.loadmat('/home/jk21/Documents/hmd/jk_classifier/S1-S1-Phonocardiogram-Peak-Detection-Method-in-Python/Filters1')
                
                try:
                    
                    X=datos[1]
                    X=X[trim*3:-trim*3]
                    Fs=datos[0]
            
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

                    indexes=peakutils.indexes(points,thres=0.5/max(points), min_dist=1000);
                    lenght=np.shape(indexes)			# Get the length of the index vector		
                    lenght=lenght[0];				# Get the value of the index vector

                    for i in range(0,lenght):
                        p=indexes[i];
                        peaks[p]=points[p];
        
                    n=np.arange(0,fil);                            # Vector to the X axes (Number of Samples)
            
                    tmp_peaks = np.diff(indexes)
                    
                    
                    tmp_interval.append(tmp_peaks)
                    
           
                except:
                    print(filename)
                    tmp_peaks = np.zeros(maxlen)
                    tmp_interval.append(tmp_peaks)
                    
            else :
                        
                tmp_peaks = np.zeros(maxlen)
                tmp_interval.append(tmp_peaks)
                
      
            
            # Extract age_group
            age_group = get_age(current_patient_data)
            current_age_group = np.zeros(6, dtype=int)
            if age_group in age_classes:
                j = age_classes.index(age_group)
                current_age_group[j] = 1
            else :
                current_age_group[5] = 1
            features['age'].append(current_age_group)

            # Extract sex
            sex = get_sex(current_patient_data)
            sex_features = np.zeros(2, dtype=int)
            if compare_strings(sex, 'Female'):
                sex_features[0] = 1
            elif compare_strings(sex, 'Male'):
                sex_features[1] = 1
            features['sex'].append(sex_features)

            # Extract height and weight.
            height = get_height(current_patient_data)
            weight = get_weight(current_patient_data)
            ## simple impute
            if math.isnan(height) :
                height = 110.846
            if math.isnan(weight) :
                weight = 23.767
                
            features['hw'].append(np.array([height, weight]))

            # Extract pregnancy
            is_pregnant = get_pregnancy_status(current_patient_data)
            features['preg'].append(is_pregnant)

            # Extract location
            locations = entries[0]
            num_recording_locations = len(recording_locations)
            loc_features = np.zeros(num_recording_locations)
            if locations in recording_locations:
                j = recording_locations.index(locations)
                loc_features[j] = 1
            features['loc'].append(loc_features)

            # Extract labels 
            mm_label = get_murmur(current_patient_data)
            out_label = get_outcome(current_patient_data)
            current_mm_labels = np.zeros(2)
            current_out_labels = np.zeros(2)
            if mm_label == 'Absent' :
                current_mm_labels = np.array([0, 0, 1])
            elif mm_label == 'unknown' :
                current_mm_labels = np.array([0, 1, 0])
            else :
                mm_loc = get_murmur_loc(current_patient_data)
                if mm_loc == 'nan' :
                    current_mm_labels = np.array([0.9, 0.05, 0.05])
                else :
                    mm_loc = mm_loc.split('+')
                    if locations in mm_loc :
                        current_mm_labels = np.array([1, 0, 0])
                    else :
                        current_mm_labels = np.array([0.7, 0.2, 0.1])

            if out_label == 'Normal' :
                current_out_labels = np.array([0, 1])
            else :
                current_out_labels = np.array([1, 0])
#                if mm_label == 'Absent' :
#                    current_out_labels = np.array([0.8, 0.2])
#                elif mm_label == 'unknown' :
#                    current_out_labels = np.array([0.85, 0.15])
#                else :
#                    current_out_labels = np.array([1, 0])
                
            features['mm_labels'].append(current_mm_labels)
            features['out_labels'].append(current_out_labels)

    if use_mel :
        M, N = features['mel1'][0].shape
        for i in range(len(features['mel1'])) :
            
            features['mel1'][i] = features['mel1'][i].reshape(M,N,1)
            
    else :
        M, N, _ = features['mel1'][0].shape
    print("melspec: ", M,N)
    
    mel_input_shape = (M,N,1)
        
    if use_cqt :
        M, N= features['cqt1'][0].shape
        for i in range(len(features['cqt1'])) :
            features['cqt1'][i] = features['cqt1'][i].reshape(M,N,1)
       
    else :
        M, N,__ = features['cqt1'][0].shape
    print("cqt: ", M,N)
    cqt_input_shape = (M,N,1)

    
    if use_stft :
        M, N = features['stft1'][0].shape
        for i in range(len(features['stft1'])) :
            features['stft1'][i] = features['stft1'][i].reshape(M,N,1)
        
    else :
        M, N ,__= features['stft1'][0].shape
    print("stft: ", M,N)
    stft_input_shape = (M,N,1)
    
    if use_interval:
        
        padded =pad_sequences(tmp_interval, maxlen=max_interval_len, dtype='float64', padding='post', truncating='post', value=0.0)
        
        for i in range(len(padded)):
            features['interval'].append(padded[i])
        
    else:
        for i in range(len(tmp_interval)):
            features['interval'].append(tmp_interval[i])

    if use_interval:
        
        for i in range(len(features['interval'])):
            features['interval'][i]= features['interval'][i].reshape(-1,1)
        
    else:
        for i in range(len(features['interval'])):
            features['interval'][i]= features['interval'][i].reshape(-1,1)

    for k1 in features.keys() :
        features[k1] = np.array(features[k1])          

    
    if use_wav2:
        padded =pad_sequences(tmp_wav, maxlen=maxlen1, dtype='float64', padding='pre', truncating='pre', value=0.0)
        tmp_pad = padded[:,np.newaxis,:]
        
        tmp=[]
        for i in range(len(tmp_pad)):
            tmp_feature = model(tmp_pad[i])
            tmp.append(tmp_feature)
            
        tmp=np.array(tmp, dtype=np.float32)
        new_tmp1=tmp.reshape(-1,374,32)
        features['wav2']=new_tmp1
        
    
    interval_input_shape = features['interval'].shape[1:]
    
    M,N = interval_input_shape
    
    print("interval: ", M,N)
    
    wav2_input_shape = features['wav2'].shape[1:]
    
    M,N = wav2_input_shape
    
    print("wav2: ", M,N)
    
    return features, mel_input_shape, cqt_input_shape, stft_input_shape,interval_input_shape,wav2_input_shape


