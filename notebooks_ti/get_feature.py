#import soundfile as sf
import numpy as np
import librosa
from helper_code import *
from tqdm import tqdm
import pywt
from sklearn.preprocessing import MinMaxScaler



def feature_extract_raw(fnm, samp_sec=20, sr = 4000, pre_emphasis = 0):

    if isinstance(fnm, str) :  #fnm이 str타입이면 
        data, sample_rate = librosa.load(fnm, sr = sr)  # raw data와 sr 로딩
        data = data * 1.0    # 아마 float으로 바꾸기 위함
        data = data[sr:-sr]
    else :
        data = fnm * 1.0   # fnm이 raw data 일때.
        data = data[sr:-sr]
        sample_rate = sr    # 4000인 sr을 가져옴

    if samp_sec:     
        if len(data) > sample_rate * samp_sec :    # data가 20초 넘으면
            n_samp = len(data) // int(sample_rate * samp_sec)   # 얼마나 넘는지 보고 (데이터에 20초가 몇번 들어가나)
            signal = []
            for i in range(n_samp) :  # n번만큼 signal list 원소로 20초씩 자른 데이터를 넣음
                signal.append(data[ int(sample_rate * samp_sec)*i:(int(sample_rate * samp_sec)*(i+1))])
        else :
            n_samp = 1
            signal = np.zeros(int(sample_rate*samp_sec,))  # 빈 20초 signal 생성
            for i in range(int(sample_rate * samp_sec) // len(data)) : # 20초 안에 data가 몇번 들어가는지 만큼 루프
                signal[(i)*len(data):(i+1)*len(data)] = data      # data를 signal에 반복해서 넣어줌
            num_last = int(sample_rate * samp_sec) - len(data)*(i+1)  # n번 넣고 남은 나머지 공백을 계산
            signal[(i+1)*len(data):int(sample_rate * samp_sec)] = data[:num_last]  # 나머지 공백을 data를 잘라서 넣어줌
            signal = [signal]
    else:
        n_samp = 1
        signal = [data]

    for i in range(n_samp) :   # 위의 n_samp을 가져와서 루프
        if pre_emphasis :      # pre-emphasis 진행 보통 y(t) = x(t) - a*x(t-1)  근데 0이라서 pre-emphasis 없는듯
            emphasized_signal = np.append(signal[i][0], signal[i][1:] - pre_emphasis * signal[i][:-1])
        else :  # pre-emphasis 없음
            emphasized_signal = signal[i]

    return signal, n_samp

def feature_extract_wavelet_raw(fnm, samp_sec=20, sr = 1000, pre_emphasis = 0):
#    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    if isinstance(fnm, str) :  
        data, _ = librosa.load(fnm, sr = 4000)  
        data = data * 1.0   
        data = data[4000:-4000]
        sample_rate = sr
        
        data,hb = pywt.dwt(data,wavelet='db2')
        data,hb = pywt.dwt(data,wavelet='db2')
        
#        data = np.array(scaler.fit_transform(np.array(data).reshape(1,-1))).reshape(-1,).tolist()
        
    else :
        data = fnm * 1.0  
        data = data[4000:-4000]
        sample_rate = sr  
        
        data,hb = pywt.dwt(data,wavelet='db2')
        data,hb = pywt.dwt(data,wavelet='db2')
        
#        data = np.array(scaler.fit_transform(np.array(data).reshape(1,-1))).reshape(-1,).tolist()

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

    for i in range(n_samp) :   # 위의 n_samp을 가져와서 루프
        if pre_emphasis :      # pre-emphasis 진행 보통 y(t) = x(t) - a*x(t-1)  근데 0이라서 pre-emphasis 없는듯
            emphasized_signal = np.append(signal[i][0], signal[i][1:] - pre_emphasis * signal[i][:-1])
        else :  # pre-emphasis 없음
            emphasized_signal = signal[i]

    return signal, n_samp

def feature_extract_wavelet_melspec(fnm,rand=False, samp_sec=20, sr = 1000, pre_emphasis = 0, hop_length=64, win_length = 128, n_mels = 64):

#    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))

    if isinstance(fnm, str) :  #fnm이 str타입이면 
        data, sample_rate = librosa.load(fnm, sr = 4000)  # raw data와 sr 로딩
        sample_rate = sr
        data = data * 1.0    
        if rand :
            data = rand_amp(data)
        data = data[4000:-4000]
        data,hb = pywt.dwt(data,wavelet='db2')
        data,hb = pywt.dwt(data,wavelet='db2')
        
#        data = np.array(scaler.fit_transform(np.array(data).reshape(1,-1))).reshape(-1,)
        
        
    else :
        data = fnm * 1.0   # fnm이 raw data 일때.
        sample_rate = sr   
        data = data[4000:-4000]
        data,hb = pywt.dwt(data,wavelet='db2')
        data,hb = pywt.dwt(data,wavelet='db2')
 #       data = np.array(scaler.fit_transform(np.array(data).reshape(1,-1))).reshape(-1,)

    if samp_sec:     
        if len(data) > sample_rate * samp_sec :
            n_samp = len(data) // int(sample_rate * samp_sec)   # 얼마나 넘는지 보고 (데이터에 5초가 몇번 들어가나)
            signal = []
            for i in range(n_samp) :  # n번만큼 signal list 원소로 5초씩 자른 데이터를 넣음
                signal.append(data[ int(sample_rate * samp_sec)*i:(int(sample_rate * samp_sec)*(i+1))])
        else :
            n_samp = 1
            signal = np.zeros(int(sample_rate*samp_sec,))  # 빈 20초 signal 생성
            for i in range(int(sample_rate * samp_sec) // len(data)) : # 20초 안에 data가 몇번 들어가는지 만큼 루프
                signal[(i)*len(data):(i+1)*len(data)] = data      # data를 signal에 반복해서 넣어줌
            num_last = int(sample_rate * samp_sec) - len(data)*(i+1)  # n번 넣고 남은 나머지 공백을 계산
            signal[(i+1)*len(data):int(sample_rate * samp_sec)] = data[:num_last]  # 나머지 공백을 data를 잘라서 넣어줌
            signal = [signal]
    else:
        n_samp = 1
        signal = [data]

    Sig = []
    for i in range(n_samp) :   # 위의 n_samp을 가져와서 루프
        if pre_emphasis :      # pre-emphasis 진행 보통 y(t) = x(t) - a*x(t-1)  근데 0이라서 pre-emphasis 없는듯
            emphasized_signal = np.append(signal[i][0], signal[i][1:] - pre_emphasis * signal[i][:-1])
        else :  # pre-emphasis 없음
            emphasized_signal = signal[i]

        Sig.append(np.array(librosa.power_to_db(librosa.feature.melspectrogram(y=emphasized_signal, sr= sr, n_mels=n_mels, n_fft=win_length, hop_length=hop_length, win_length=win_length))))
    
#    print(n_samp,len(Sig))               
    return Sig


def rand_amp(data) :
    import random
    randnum = random.randint(0,1)
    if randnum ==0 :
        data = data * 0.9
    elif randnum == 1 :
        data = data * 1.1
    return data


def feature_extract_raw_melspec(fnm, rand = False, samp_sec = 20, sr = 4000, pre_emphasis = 0, hop_length=256, win_length = 512, n_mels = 100):

    if isinstance(fnm, str) :  #fnm이 str타입이면 
        data, sample_rate = librosa.load(fnm, sr = 4000)  # raw data와 sr 로딩
        sample_rate = sr
        data = data * 1.0    # 아마 float으로 바꾸기 위함
        if rand :
            data = rand_amp(data)
        data = data[sr:-sr]
        
    else :
        data = fnm * 1.0   # fnm이 raw data 일때.
        sample_rate = sr    # 4000인 sr을 가져옴
        data = data[sr:-sr]
        
    if samp_sec:     
        if len(data) > sample_rate * samp_sec :    # data가 5초 넘으면
            n_samp = len(data) // int(sample_rate * samp_sec)   # 얼마나 넘는지 보고 (데이터에 5초가 몇번 들어가나)
            signal = []
            for i in range(n_samp) :  # n번만큼 signal list 원소로 5초씩 자른 데이터를 넣음
                signal.append(data[ int(sample_rate * samp_sec)*i:(int(sample_rate * samp_sec)*(i+1))])
        else :
            n_samp = 1
            signal = np.zeros(int(sample_rate*samp_sec,))  # 빈 20초 signal 생성
            for i in range(int(sample_rate * samp_sec) // len(data)) : # 20초 안에 data가 몇번 들어가는지 만큼 루프
                signal[(i)*len(data):(i+1)*len(data)] = data      # data를 signal에 반복해서 넣어줌
            num_last = int(sample_rate * samp_sec) - len(data)*(i+1)  # n번 넣고 남은 나머지 공백을 계산
            signal[(i+1)*len(data):int(sample_rate * samp_sec)] = data[:num_last]  # 나머지 공백을 data를 잘라서 넣어줌
            signal = [signal]
    else:
        n_samp = 1
        signal = [data]

    Sig = []
    for i in range(n_samp) :   # 위의 n_samp을 가져와서 루프
        if pre_emphasis :      # pre-emphasis 진행 보통 y(t) = x(t) - a*x(t-1)  근데 0이라서 pre-emphasis 없는듯
            emphasized_signal = np.append(signal[i][0], signal[i][1:] - pre_emphasis * signal[i][:-1])
        else :  # pre-emphasis 없음
            emphasized_signal = signal[i]

        Sig.append(np.array(librosa.power_to_db(librosa.feature.melspectrogram(y=emphasized_signal, sr= sr, n_mels=n_mels, n_fft=win_length, hop_length=hop_length, win_length=win_length))))
    
#    print(n_samp,len(Sig))               
    return Sig





def feature_extract_stft(fnm, samp_sec=20, sr = 4000, pre_emphasis = 0, hop_length=256, win_length = 512, n_mels = 100):

    if isinstance(fnm, str) :
        data, sample_rate = librosa.load(fnm, sr = sr)
        data = data * 1.0
    else :
        data = fnm * 1.0
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


def feature_extract_cqt(fnm, samp_sec=20, sr = 4000, pre_emphasis = 0, filter_scale = 1, n_bins = 80, fmin = 10):

    if isinstance(fnm, str) :
#        data, sample_rate = sf.read(fnm, dtype = 'int16')
        data = data * 1.0
    else :
        data = fnm * 1.0
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





def feature_extract_tsv_melspec(fnm,tsv,rand = False, samp_sec = 2, sr = 4000, pre_emphasis = 0, hop_length=128, win_length = 256, n_mels = 80):

    if isinstance(fnm, str) :  #fnm이 str타입이면 
        data, sample_rate = librosa.load(fnm, sr = 4000)  # raw data와 sr 로딩
        sample_rate = sr
        data = data * 1.0    # 아마 float으로 바꾸기 위함
        if rand :
            data = rand_amp(data)
        data = data[int(tsv[0]):int(tsv[3])]
        
    else :
        data = fnm * 1.0   # fnm이 raw data 일때.
        sample_rate = sr    # 4000인 sr을 가져옴
        data = data[sr : sr + samp_sec*sr]
        
    if samp_sec:     
        if len(data) > sample_rate * samp_sec :    # data가 3초 넘으면
            n_samp = len(data) // int(sample_rate * samp_sec)   # 얼마나 넘는지 보고 (데이터에 5초가 몇번 들어가나)
            signal = []
            for i in range(n_samp) :  # n번만큼 signal list 원소로 5초씩 자른 데이터를 넣음
                signal.append(data[ int(sample_rate * samp_sec)*i:(int(sample_rate * samp_sec)*(i+1))])
        else :
            n_samp = 1
            signal = np.zeros(int(sample_rate*samp_sec,))  # 빈 3초 signal 생성
            for i in range(int(sample_rate * samp_sec) // len(data)) : # 3초 안에 data가 몇번 들어가는지 만큼 루프
                signal[(i)*len(data):(i+1)*len(data)] = data      # data를 signal에 반복해서 넣어줌
            num_last = int(sample_rate * samp_sec) - len(data)*(i+1)  # n번 넣고 남은 나머지 공백을 계산
            signal[(i+1)*len(data):int(sample_rate * samp_sec)] = data[:num_last]  # 나머지 공백을 data를 잘라서 넣어줌
            signal = [signal]
    else:
        n_samp = 1
        signal = [data]

    Sig = []
    for i in range(n_samp) :   # 위의 n_samp을 가져와서 루프
        if pre_emphasis :      # pre-emphasis 진행 보통 y(t) = x(t) - a*x(t-1)  근데 0이라서 pre-emphasis 없는듯
            emphasized_signal = np.append(signal[i][0], signal[i][1:] - pre_emphasis * signal[i][:-1])
        else :  # pre-emphasis 없음
            emphasized_signal = signal[i]

        Sig.append(np.array(librosa.power_to_db(librosa.feature.melspectrogram(y=emphasized_signal, sr= sr, n_mels=n_mels, n_fft=win_length, hop_length=hop_length, win_length=win_length))))
    
#    print(n_samp,len(Sig))               
    return Sig

def feature_extract_tsv_melspec2(fnm,tsv,rand = False, samp_sec = 2, sr = 4000, pre_emphasis = 0, hop_length=128, win_length = 256, n_mels = 80):

    if isinstance(fnm, str) :  #fnm이 str타입이면 
        data, sample_rate = librosa.load(fnm, sr = 4000)  # raw data와 sr 로딩
        sample_rate = sr
        data = data * 1.0    # 아마 float으로 바꾸기 위함
        if rand :
            data = rand_amp(data)
            
    signal = []
    for i in range(len(tsv[:-4])) :
        signal.append(data[int(tsv[i]):int(tsv[i])+sr*samp_sec])
                      
    n_samp = len(signal)

    Sig = []
    for i in range(n_samp) :   # 위의 n_samp을 가져와서 루프
        if pre_emphasis :      # pre-emphasis 진행 보통 y(t) = x(t) - a*x(t-1)  근데 0이라서 pre-emphasis 없는듯
            emphasized_signal = np.append(signal[i][0], signal[i][1:] - pre_emphasis * signal[i][:-1])
        else :  # pre-emphasis 없음
            emphasized_signal = signal[i]

        Sig.append(np.array(librosa.power_to_db(librosa.feature.melspectrogram(y=emphasized_signal, sr= sr, n_mels=n_mels, n_fft=win_length, hop_length=hop_length, win_length=win_length))))
    
#    print(n_samp,len(Sig))               
    return Sig

def feature_extract_tsv_melspec_one(fnm,tsv,rand = False, samp_sec = 2, sr = 4000, pre_emphasis = 0, hop_length=128, win_length = 256, n_mels = 80):

    if isinstance(fnm, str) :  #fnm이 str타입이면 
        data, sample_rate = librosa.load(fnm, sr = 4000)  # raw data와 sr 로딩
        sample_rate = sr
        data = data * 1.0    # 아마 float으로 바꾸기 위함
        if rand :
            data = rand_amp(data)
        data = data[int(tsv[0]):int(tsv[3])]
        
    else :
        data = fnm * 1.0   # fnm이 raw data 일때.
        sample_rate = sr    # 4000인 sr을 가져옴
#        data = data[sr : -sr]
    signal = []
    startframe = np.linspace(0, len(data)-samp_sec*sr, num=100)
    for asf in startframe:
        signal.append(data[int(asf):int(asf)+samp_sec*sr])
#    feats = numpy.stack(feats, axis = 0).astype(numpy.float)
        
    n_samp = len(signal)
    
    Sig = []
    for i in range(n_samp) :   # 위의 n_samp을 가져와서 루프
        if pre_emphasis :      # pre-emphasis 진행 보통 y(t) = x(t) - a*x(t-1)  근데 0이라서 pre-emphasis 없는듯
            emphasized_signal = np.append(signal[i][0], signal[i][1:] - pre_emphasis * signal[i][:-1])
        else :  # pre-emphasis 없음
            emphasized_signal = signal[i]

        Sig.append(np.array(librosa.power_to_db(librosa.feature.melspectrogram(y=emphasized_signal, sr= sr, n_mels=n_mels, n_fft=win_length, hop_length=hop_length, win_length=win_length))))
    
#    print(n_samp,len(Sig))               
    return Sig
























def get_features(data_folder, patient_files_trn, raw = False) :
    features = dict()

    features['mel1'] = []
    features['raw'] = []
    murmurs = np.empty((0,3))
    outcomes = np.empty((0,2))
    labels = {}
    
    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)

    
    age_classes = ['Neonate', 'Infant', 'Child', 'Adolescent', 'Young Adult']
    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']

    num_patient_files = len(patient_files_trn)

    for i in tqdm(range(num_patient_files)):

        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files_trn[i])
        num_locations = get_num_locations(current_patient_data)
        recording_information = current_patient_data.split('\n')[1:num_locations+1]
        if get_murmur(current_patient_data) != 'Present' :
            murmur_location_information = 'nan'
        else :
            murmur_location_information = current_patient_data.split('\n')[-15].split()[2].split('+')
        
        for j in range(num_locations) :
            entries = recording_information[j].split(' ')
            recording_file = entries[2]
            filename = os.path.join(data_folder, recording_file)
            filename_location = recording_file.split('_')[1].split('.')[0]
            
            if raw :
                raw1 = feature_extract_raw(filename)[0]
                features['raw'].append(raw1)
            # Extract melspec
            else :
                mel1 = feature_extract_wavelet_melspec(filename)  # np > shape (n_samp,n_mels,time)(n,64,201)
                n_5sec = len(mel1)
                features['mel1'].extend(mel1)


            # Extract labels and use one-hot encoding.
            
            # 클래스 일단 3개지만 학습 데이터에선 loc 아니면 absent로 봄
            
            current_murmur = np.zeros((1,num_murmur_classes), dtype=int)  
            if (filename_location in murmur_location_information) or murmur_location_information == 'nan' :
                murmur = get_murmur(current_patient_data)
            else :
                murmur = 'Absent'
            if murmur in murmur_classes:
                j = murmur_classes.index(murmur)
                current_murmur[0][j] = 1
            current_murmur = np.repeat(current_murmur,n_5sec,0)
            murmurs = np.concatenate((murmurs,current_murmur),axis=0)
            
            current_outcome = np.zeros((1,num_outcome_classes), dtype=int)
            outcome = get_outcome(current_patient_data)
            if outcome in outcome_classes:
                j = outcome_classes.index(outcome)
                current_outcome[0][j] = 1
            current_outcome = np.repeat(current_outcome,n_5sec,0)  
            outcomes = np.concatenate((outcomes,current_outcome),axis=0)
            
#            print(len(features['mel1']))
            
    features['mel1'] = np.array(features['mel1']).reshape(-1,64,40)
    
    labels['murmur'] = murmurs
    labels['outcome'] = outcomes
    
    
    return features, labels


def get_feature_one(patient_data, verbose = 0) :

    age_classes = ['Neonate', 'Infant', 'Child', 'Adolescent', 'Young Adult']
    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
    
    num_locations = get_num_locations(patient_data)
    recording_information = patient_data.split('\n')[1:num_locations+1]
    if get_murmur(patient_data) != 'Present' :
        murmur_location_information = ['nan']
    else :
        murmur_location_information = patient_data.split('\n')[-15].split()[-1].split('+')

    features = dict()
    features['loc'] = murmur_location_information # murmur location 어딘지
    features['record_file'] = []     # 어느 Location wav file이 있는지
                                
    for j in range(num_locations) :
        entries = recording_information[j].split(' ')
        recording_file = entries[2]
        features['record_file'].append(recording_file.split('.')[0].split('_')[1])
#        print(recording_file.split('.')[0].split('_')[1])
        
#    if verbose :
#        label = get_label(patient_data)
#        print(label)
    return features